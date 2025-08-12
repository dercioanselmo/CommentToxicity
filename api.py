from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization
import os
import logging
import re
import string

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define base directory
BASE_DIR = "/home/branch/projects/CommentToxicity"

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Custom focal loss
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = y_true * tf.pow(1 - y_pred, gamma) + (1 - y_true) * tf.pow(y_pred, gamma)
        return tf.reduce_mean(alpha * weight * cross_entropy)
    return focal_loss_fn

# Load the trained model and vectorizer
model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, 'toxicity_improved_v31.h5'),
    custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
)
df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'train.csv'))
df['comment_text'] = df['comment_text'].apply(clean_text)
MAX_FEATURES = 150000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=500, output_mode='int')
vectorizer.adapt(df['comment_text'].values)

# Define the toxicity categories
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

@app.post("/predict")
async def predict_toxicity(comment: str):
    logger.info(f"Received comment: {comment}")
    vectorized_comment = vectorizer([clean_text(comment)])
    results = model.predict(vectorized_comment)
    logger.info(f"Raw predictions: {results[0].tolist()}")
    response = {category: bool(results[0][idx] > 0.5) for idx, category in enumerate(categories)}
    logger.info(f"Response: {response}")
    return response