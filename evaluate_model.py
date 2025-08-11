import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import os
import re
import string

BASE_DIR = "/home/branch/projects/CommentToxicity"

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load test data
test_df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge/test.csv'))
test_labels = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge/test_labels.csv'))
test_df = test_df.merge(test_labels, on='id')
test_df = test_df[test_df['toxic'] != -1]
test_df['comment_text'] = test_df['comment_text'].apply(clean_text)

# Load model and vectorizer
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'toxicity_improved_v25.h5'))
train_df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge/train.csv'))
train_df['comment_text'] = train_df['comment_text'].apply(clean_text)
vectorizer = TextVectorization(max_tokens=150000, output_sequence_length=500, output_mode='int')
vectorizer.adapt(train_df['comment_text'].values)

# Prepare test data
X_test = vectorizer(test_df['comment_text'].values)
y_test = test_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

# Evaluate
loss, accuracy, precision, recall, f1_score = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}, Test Precision: {precision}, Test Recall: {recall}, Test F1 Score: {f1_score}')

# Test specific comments
sample_comments = [
    'You’re an idiot who doesn’t know anything.',
    'This is a great article, thanks for sharing!',
    'I hate you and hope you die.'
]
for comment in sample_comments:
    sample_vectorized = vectorizer([clean_text(comment)])
    prediction = model.predict(sample_vectorized)
    print(f"Comment: {comment}")
    print({category: bool(prediction[0][idx] > 0.45) for idx, category in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])})