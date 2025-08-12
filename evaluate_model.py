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

# Load test data
test_df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge/test.csv'))
test_labels = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge/test_labels.csv'))
test_df = test_df.merge(test_labels, on='id')
test_df = test_df[test_df['toxic'] != -1]
test_df['comment_text'] = test_df['comment_text'].apply(clean_text)

# Balance test set (50% toxic)
toxic_test = test_df[test_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) > 0]
non_toxic_test = test_df[test_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) == 0]
min_size = min(len(toxic_test), len(non_toxic_test))
test_df = pd.concat([toxic_test.sample(min_size, random_state=42), non_toxic_test.sample(min_size, random_state=42)]).sample(frac=1, random_state=42)

# Load model and vectorizer
model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, 'toxicity_improved_v32.h5'),
    custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25), 'F1Score': F1Score}
)
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

# Test specific comments with lower threshold
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