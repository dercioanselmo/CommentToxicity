from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization
import os

app = FastAPI()

# Define base directory
BASE_DIR = "/home/ubuntu/projects/CommentToxicity"

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
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    response = {category: bool(results[0][idx] > 0.5) for idx, category in enumerate(categories)}
    return response