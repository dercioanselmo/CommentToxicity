import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define base directory (relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'train.csv'))
print(df.head())

# Define categories
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Prepare data
X = df['comment_text'].values
y = df[categories].values

# Text vectorization
MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')
vectorizer.adapt(X)

# Build model
model = Sequential([
    Embedding(MAX_FEATURES + 1, 32),
    LSTM(32, return_sequences=False),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(len(categories), activation='sigmoid')
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Vectorize input data
X_vectorized = vectorizer(X)

# Train model
model.fit(X_vectorized, y, batch_size=32, epochs=5, validation_split=0.2)

# Save model
model.save(os.path.join(BASE_DIR, 'toxicity.h5'))

# Test a sample comment
sample_comment = 'You’re an idiot who doesn’t know anything.'
sample_vectorized = vectorizer([sample_comment])
prediction = model.predict(sample_vectorized)
print({category: bool(prediction[0][idx] > 0.5) for idx, category in enumerate(categories)})