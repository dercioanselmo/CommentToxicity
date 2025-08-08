from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization
import os
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define base directory
BASE_DIR = "/home/branch/projects/CommentToxicity"

# Load the trained model and vectorizer
def focal_loss(gamma=3.0, alpha=0.5):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * y_true * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss, axis=-1)
    return focal_loss_fn

model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'toxicity_improved_v12.h5'), custom_objects={'focal_loss_fn': focal_loss(gamma=3.0, alpha=0.5)})
df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'train.csv'))
MAX_FEATURES = 150000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=500, output_mode='int')
vectorizer.adapt(df['comment_text'].values)

# Define the toxicity categories
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

@app.post("/predict")
async def predict_toxicity(comment: str):
    logger.info(f"Received comment: {comment}")
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    logger.info(f"Raw predictions: {results[0].tolist()}")
    response = {category: bool(results[0][idx] > 0.3) for idx, category in enumerate(categories)}
    logger.info(f"Response: {response}")
    return response