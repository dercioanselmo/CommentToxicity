import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import os
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define base directory
BASE_DIR = "/home/ubuntu/projects/CommentToxicity"

# Load dataset
df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'train.csv'))
print(df.head())

# Define categories
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Prepare data
X = df['comment_text'].values
y = df[categories].values

# Compute class weights to handle imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y.flatten())
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Text vectorization
MAX_FEATURES = 50000  # Reduced for faster training
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=500, output_mode='int')
vectorizer.adapt(X)

# Build a more complex model
model = Sequential([
    Embedding(MAX_FEATURES + 1, 64),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(categories), activation='sigmoid')
])

# Compile model with additional metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Vectorize input data
X_vectorized = vectorizer(X)

# Train model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(
    X_vectorized,
    y,
    batch_size=16,
    epochs=10,
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=class_weight_dict
)

# Save model
model.save(os.path.join(BASE_DIR, 'toxicity_improved.h5'))

# Test a sample comment
sample_comment = 'You’re an idiot who doesn’t know anything.'
sample_vectorized = vectorizer([sample_comment])
prediction = model.predict(sample_vectorized)
print({category: bool(prediction[0][idx] > 0.5) for idx, category in enumerate(categories)})