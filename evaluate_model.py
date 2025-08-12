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
import random

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Define base directory
BASE_DIR = "/home/branch/projects/CommentToxicity"

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Simple text augmentation for toxic comments
def augment_text(text):
    words = text.split()
    if len(words) > 3:
        idx = random.randint(0, len(words) - 1)
        synonyms = ['stupid', 'dumb', 'idiot', 'moron', 'fool'] if 'idiot' in words[idx].lower() else ['bad', 'awful', 'terrible', 'horrible']
        if synonyms and random.random() < 0.3:
            words[idx] = random.choice(synonyms)
    return ' '.join(words)

# Load and preprocess dataset
df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'train.csv'))
df['comment_text'] = df['comment_text'].apply(clean_text)

# Oversample toxic comments (90% of non-toxic samples) with augmentation
toxic_df = df[df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) > 0]
non_toxic_df = df[df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) == 0]
toxic_df_oversampled = toxic_df.sample(int(0.9 * len(non_toxic_df)), replace=True, random_state=42)
toxic_df_oversampled['comment_text'] = toxic_df_oversampled['comment_text'].apply(augment_text)
df = pd.concat([non_toxic_df, toxic_df_oversampled]).sample(frac=1, random_state=42)
print(df.head())

# Define categories
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

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
    class_weight_dict[i * 2 + 1] = weights[1] * 3.0  # Positive class (1)

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

# Custom focal loss
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = y_true * tf.pow(1 - y_pred, gamma) + (1 - y_true) * tf.pow(y_pred, gamma)
        return tf.reduce_mean(alpha * weight * cross_entropy)
    return focal_loss_fn

# Build model with increased LSTM units and adjusted dropout
model = Sequential([
    Embedding(MAX_FEATURES + 1, 384),
    LSTM(512, return_sequences=True, dropout=0.4),
    LSTM(256, dropout=0.4),
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
early_stopping = EarlyStopping(monitor='val_recall', patience=5, restore_best_weights=True, mode='max')
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
model.save(os.path.join(BASE_DIR, 'toxicity_improved_v31.h5'))

# Test sample comments
sample_comments = [
    'You’re an idiot who doesn’t know anything.',
    'This is a great article, thanks for sharing!',
    'I hate you and hope you die.'
]
for comment in sample_comments:
    sample_vectorized = vectorizer([clean_text(comment)])
    prediction = model.predict(sample_vectorized)
    print(f"Comment: {comment}")
    print({category: bool(prediction[0][idx] > 0.5) for idx, category in enumerate(categories)})