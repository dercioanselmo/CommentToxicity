from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization
import os

app = FastAPI()

# Define Pydantic model for input validation
class CommentRequest(BaseModel):
    comment: str

# Get the directory of the current script (api.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and vectorizer
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'toxicity.h5'))
df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'train.csv'))
MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')
vectorizer.adapt(df['comment_text'].values)

# Define the toxicity categories
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

@app.post("/predict")
async def predict_toxicity(request: CommentRequest):
    # Vectorize the input comment
    vectorized_comment = vectorizer([request.comment])
    # Make prediction
    results = model.predict(vectorized_comment)
    # Format the output
    response = {category: bool(results[0][idx] > 0.5) for idx, category in enumerate(categories)}
    return response