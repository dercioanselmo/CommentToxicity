import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import os
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
import re
import string
from transformers import pipeline
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, filename='training_errors.log', filemode='a')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define base directory
BASE_DIR = "/home/branch/projects/CommentToxicity"

# Enhanced text preprocessing
def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        logger.warning("Empty or invalid comment skipped in preprocessing")
        return ""
    text = text.lower()
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()[:400]
    return ' '.join(words)

# Back-translation augmentation with batch processing
try:
    translator_en_to_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device=0)
    translator_es_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device=0)
except Exception as e:
    logger.error(f"Failed to initialize translation pipelines: {str(e)}")
    translator_en_to_es = None
    translator_es_to_en = None

def augment_text_batch(texts):
    if not texts or translator_en_to_es is None or translator_es_to_en is None:
        logger.warning("Skipping augmentation due to empty input or pipeline failure")
        return texts
    try:
        es_texts = translator_en_to_es(texts, max_length=512, truncation=True)
        es_texts = [result['translation_text'] for result in es_texts]
        back_translated = translator_es_to_en(es_texts, max_length=512, truncation=True)
        back_translated = [result['translation_text'] for result in back_translated]
        return [clean_text(text) for text in back_translated]
    except Exception as e:
        logger.error(f"Batch translation error for texts: {str([text[:50] for text in texts[:5]])}...: {str(e)}")
        return texts

# Load and preprocess dataset
df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'train.csv'))
df['comment_text'] = df['comment_text'].apply(clean_text)
df = df[df['comment_text'] != ""]  # Remove empty comments
df = df.head(100)  # Test on small subset

# Oversample each toxic label
non_toxic_df = df[df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) == 0]
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
oversampled_dfs = [non_toxic_df]
for category in categories:
    toxic_df = df[df[category] == 1]
    oversampled = toxic_df.sample(len(non_toxic_df), replace=True, random_state=42)
    batch_size = 16
    augmented_texts = []
    print(f"Augmenting {len(oversampled)} comments for category: {category}")
    for i in tqdm(range(0, len(oversampled), batch_size), desc=f"Augmenting {category}"):
        batch = oversampled['comment_text'].iloc[i:i + batch_size].tolist()
        augmented_texts.extend(augment_text_batch(batch))
    oversampled['comment_text'] = augmented_texts
    oversampled_dfs.append(oversampled)
df = pd.concat(oversampled_dfs).sample(frac=1, random_state=42)
print("Augmented dataset sample:")
print(df.head())

# Prepare data
X = df['comment_text'].values
y = df[categories].values

# Compute class weights with adjusted values
class_weight_dict = {}
for i, category in enumerate(categories):
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y[:, i]
    )
    class_weight_dict[i * 2] = weights[0]
    class_weight_dict[i * 2 + 1] = weights[1] * 2.0 if category in ['severe_toxic', 'identity_hate'] else weights[1] * 1.5

# Text vectorization
MAX_FEATURES = 150000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=500, output_mode='int')
print("Vectorizing text...")
vectorizer.adapt(X)

# Custom F1 score metric for multi-label
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.4, tf.float32)  # Binarize predictions
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# Custom focal loss
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = y_true * tf.pow(1 - y_pred, gamma) + (1 - y_true) * tf.pow(y_pred, gamma)
        return tf.reduce_mean(alpha * weight * cross_entropy)
    return focal_loss_fn

# Build model
model = Sequential([
    Embedding(MAX_FEATURES + 1, 384),
    Dropout(0.3),
    LSTM(384, return_sequences=True, dropout=0.4),
    LSTM(192, dropout=0.4),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='sigmoid')
])

# Compile model
model.compile(
    loss=focal_loss(gamma=2.0, alpha=0.25),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1Score()]
)

# Train model with early stopping
early_stopping = EarlyStopping(monitor='val_f1_score', patience=5, restore_best_weights=True, mode='max')
print("Starting training...")
model.fit(
    X_vectorized,
    y,
    batch_size=128,
    epochs=30,
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=class_weight_dict
)

# Save model
model.save(os.path.join(BASE_DIR, 'toxicity_improved_v33.h5'))

# Test sample comments with lower threshold
sample_comments = [
    'You’re an idiot who doesn’t know anything.',
    'This is a great article, thanks for sharing!',
    'I hate you and hope you die.'
]
for comment in sample_comments:
    sample_vectorized = vectorizer([clean_text(comment)])
    prediction = model.predict(sample_vectorized)
    print(f"Comment: {comment}")
    print({category: bool(prediction[0][idx] > 0.4) for idx, category in enumerate(categories)})