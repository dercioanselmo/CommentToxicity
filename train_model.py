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

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define base directory
BASE_DIR = "/home/branch/projects/CommentToxicity"

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Back-translation augmentation for toxic comments
translator_en_to_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
translator_es_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

def augment_text(text):
    try:
        # Translate English to Spanish and back to English
        es_text = translator_en_to_es(text)[0]['translation_text']
        back_translated = translator_es_to_en(es_text)[0]['translation_text']
        return clean_text(back_translated)
    except:
        return text  # Fallback to original text if translation fails

# Load and preprocess dataset
df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'train.csv'))
df['comment_text'] = df['comment_text'].apply(clean_text)

# Oversample each toxic label separately (90% of non-toxic samples)
non_toxic_df = df[df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) == 0]
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
oversampled_dfs = [non_toxic_df]
for category in categories:
    toxic_df = df[df[category] == 1]
    oversampled = toxic_df.sample(int(0.9 * len(non_toxic_df)), replace=True, random_state=42)
    oversampled['comment_text'] = oversampled['comment_text'].apply(augment_text)
    oversampled_dfs.append(oversampled)
df = pd.concat(oversampled_dfs).sample(frac=1, random_state=42)
print(df.head())

# Prepare data
X = df['comment_text'].values
y = df[categories].values

# Compute class weights for each label
class_weight_dict = {}
for i, category in enumerate(categories):
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y[:, i]
    )
    class_weight_dict[i * 2] = weights[0]  # Negative class (0)
    class_weight_dict[i * 2 + 1] = weights[1] * 2.5  # Positive class (1)

# Text vectorization
MAX_FEATURES = 150000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=500, output_mode='int')
vectorizer.adapt(X)

# Custom F1 score metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# Custom focal loss with label-specific weights
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = y_true * tf.pow(1 - y_pred, gamma) + (1 - y_true) * tf.pow(y_pred, gamma)
        return tf.reduce_mean(alpha * weight * cross_entropy)
    return focal_loss_fn

# Build model with reverted LSTM units and added embedding dropout
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

# Vectorize input data
X_vectorized = vectorizer(X)

# Train model with early stopping
early_stopping = EarlyStopping(monitor='val_f1_score', patience=3, restore_best_weights=True, mode='max')
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
model.save(os.path.join(BASE_DIR, 'toxicity_improved_v32.h5'))

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
    
    print({category: bool(prediction[0][idx] > 0.45) for idx, category in enumerate(categories)})