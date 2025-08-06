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

# Define base directory (relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and vectorizer
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'toxicity_improved.h5'))
df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'train.csv'))
MAX_FEATURES = 50000
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
    response = {category: bool(results[0][idx] > 0.5) for idx, category in enumerate(categories)}
    logger.info(f"Response: {response}")
    return response