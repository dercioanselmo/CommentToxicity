from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pandas as pd
import re
import string
import os

app = FastAPI()

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

# Load model and vectorizer
model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, 'toxicity_improved_v32.h5'),
    custom_objects={'focal_loss_fn': lambda y_true, y_pred: y_true, 'F1Score': lambda: None}
)
train_df = pd.read_csv(os.path.join(BASE_DIR, 'jigsaw-toxic-comment-classification-challenge', 'train.csv'))
train_df['comment_text'] = train_df['comment_text'].apply(clean_text)
train_df = train_df[train_df['comment_text'] != ""]
MAX_FEATURES = 150000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=500, output_mode='int')
vectorizer.adapt(train_df['comment_text'].values)

# Define request body
class Comment(BaseModel):
    comment: str

# Prediction endpoint
@app.post("/predict")
async def predict(comment: Comment):
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    cleaned_comment = clean_text(comment.comment)
    if not cleaned_comment:
        return {category: False for category in categories}
    vectorized_comment = vectorizer([cleaned_comment])
    prediction = model.predict(vectorized_comment)
    return {category: bool(prediction[0][idx] > 0.45) for idx, category in enumerate(categories)}