import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import numpy as np
import os
import re
import string

# Define base directory
BASE_DIR = "/home/branch/projects/CommentToxicity"

# Text preprocessing function
def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()[:500]
    return ' '.join(words)

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

# Load test data and labels
test_df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'test.csv'))
test_labels_df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'test_labels.csv'))
test_df = test_df.merge(test_labels_df, on='id')
test_df = test_df[test_df['toxic'] != -1]  # Remove unlabeled data
test_df['comment_text'] = test_df['comment_text'].apply(clean_text)
test_df = test_df[test_df['comment_text'] != ""]  # Remove empty comments

# Prepare test data
X_test = test_df['comment_text'].values
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_test = test_df[categories].values

# Load model and vectorizer
model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, 'toxicity_improved_v32.h5'),
    custom_objects={'focal_loss_fn': lambda y_true, y_pred: y_true, 'F1Score': F1Score}
)
train_df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'train.csv'))
train_df['comment_text'] = train_df['comment_text'].apply(clean_text)
train_df = train_df[train_df['comment_text'] != ""]
MAX_FEATURES = 150000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=500, output_mode='int')
vectorizer.adapt(train_df['comment_text'].values)

# Evaluate model
X_test_vectorized = vectorizer(X_test)
model.compile(
    loss=lambda y_true, y_pred: y_true,  # Dummy loss for evaluation
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1Score()]
)
metrics = model.evaluate(X_test_vectorized, y_test, return_dict=True)
print(f"Test Loss: {metrics['loss']:.4f}, Test Accuracy: {metrics['accuracy']:.4f}, "
      f"Test Precision: {metrics['precision']:.4f}, Test Recall: {metrics['recall']:.4f}, "
      f"Test F1 Score: {metrics['f1_score']:.4f}")

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
    print({category: bool(prediction[0][idx] > 0.45) for idx, category in enumerate(categories)})