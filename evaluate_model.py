import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import os

BASE_DIR = "/home/branch/projects/CommentToxicity"

# Load test data
test_df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge/test.csv'))
test_labels = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge/test_labels.csv'))
test_df = test_df.merge(test_labels, on='id')
test_df = test_df[test_df['toxic'] != -1]

# Load model and vectorizer
def focal_loss(gamma=3.0, alpha=0.5):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * y_true * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss, axis=-1)
    return focal_loss_fn

model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'toxicity_improved_v12.h5'), custom_objects={'focal_loss_fn': focal_loss(gamma=3.0, alpha=0.5)})
vectorizer = TextVectorization(max_tokens=150000, output_sequence_length=500, output_mode='int')
vectorizer.adapt(pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge/train.csv'))['comment_text'].values)

# Prepare test data
X_test = vectorizer(test_df['comment_text'].values)
y_test = test_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

# Evaluate
loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}, Test Precision: {precision}, Test Recall: {recall}')

# Test specific comments
sample_comments = [
    'You’re an idiot who doesn’t know anything.',
    'This is a great article, thanks for sharing!',
    'I hate you and hope you die.'
]
for comment in sample_comments:
    sample_vectorized = vectorizer([comment])
    prediction = model.predict(sample_vectorized)
    print(f"Comment: {comment}")
    print({category: bool(prediction[0][idx] > 0.3) for idx, category in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])})