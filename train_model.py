import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import numpy as np
import os
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
import re
import string
from nltk.corpus import stopwords

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
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Load and preprocess dataset
df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'train.csv'))
stop_words = set(stopwords.words('english'))
df['comment_text'] = df['comment_text'].apply(clean_text)
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
    class_weight_dict[i * 2 + 1] = weights[1] * 1.2  # Positive class (1), small multiplier

# Text vectorization
MAX_FEATURES = 150000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=500, output_mode='int')
vectorizer.adapt(X)

# Build model with L2 regularization
model = Sequential([
    Embedding(MAX_FEATURES + 1, 256),
    LSTM(256, return_sequences=True, kernel_regularizer=l2(0.01)),
    LSTM(128, kernel_regularizer=l2(0.01)),
    Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(len(categories), activation='sigmoid')
])

# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Vectorize input data
X_vectorized = vectorizer(X)

# Train model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
model.fit(
    X_vectorized,
    y,
    batch_size=256,
    epochs=20,
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=class_weight_dict
)

# Save model
model.save(os.path.join(BASE_DIR, 'toxicity_improved_v18.h5'))

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