#%%
# https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment

#%%
# Model training data:
# Language 	Number of reviews
# English 	150k
# Dutch 	80k
# German 	137k
# French 	140k
# Italian 	72k
# Spanish 	50k


# TODO: 
# Truncation strategy
# Running in batch mode
# Fine tuning with the review label
# Build alternative method

#%%

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
from langdetect import detect
from langdetect import DetectorFactory
# https://stats.stackexchange.com/questions/338904/measures-of-ordinal-classification-error-for-ordinal-regression
from imblearn.metrics import macro_averaged_mean_absolute_error
from torch.nn import Softmax

smax = Softmax(dim=-1)

# NOTE:
# ValueError: Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required.
# --> Zero cases of an imbalanced feature in the data

DetectorFactory.seed = 0
# Use a pipeline as a high-level helper
# sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
data=pd.read_csv('data/finegrained_sentiment_analysis.csv')
data.dropna(inplace=True)
print(data.shape)
print(data['score'].value_counts())
#%%
# -> 
# 5    17574
# 4     3936
# 1     2571
# 3     2078
# 2     1463
# Class imbalance
data=data[~data['review'].duplicated(keep=False)]
print(data.shape)
print(data)
data=data[:50]
text_columns = ['review_summary', 'review']
data['language'] = data['review_summary'].apply(lambda x: detect(x) if x!='' else np.nan)
data['language'].unique()

def clean_response(response):
    response=response[0]
    response['label'] = int(response['label'].split(' ')[0])
    response['sentiment_score'] = response.pop('label')
    response['classification_confidence'] = response.pop('score')
    return pd.Series(response, index=['sentiment_score','classification_confidence'])

#%%
d=data.copy()

def f(x):
    review=x['review']
    tokens = tokenizer(review, padding=True, truncation=True, return_tensors="pt")
    if len(tokens.input_ids) > 545:
        review_summary=x['review_summary']
        tokens = tokenizer(review_summary, padding=True, truncation=True, return_tensors="pt")
        print(len(tokens.input_ids))
    with torch.no_grad():
        logits = model(**tokens, return_dict=True).logits
    probs = model(**tokens, return_dict=True).logits.softmax(-1)[0]
    labels = model.config.id2label
    predicted_class_id = probs.argmax().item()
    classification = labels[predicted_class_id]
    classification_score = probs[predicted_class_id].item()
    return pd.Series({
        'sentiment_score': int(classification.split(' ')[0]),
        'classification_confidence': classification_score
    }, index=['sentiment_score','classification_confidence'])
    # outputs = model(**tokens, return_dict=True)
    # probs0 = smax(outputs.logits)
    # probs0 = probs0.flatten().detach().numpy()

    # prob_pos = probs[1]
    # return model(**tokens, return_dict=True)
d[['sentiment_score','classification_confidence']] = d.apply(f, axis=1)
d['error'] = np.abs(d['score'] - d['sentiment_score'])
d
#%%
d=data.copy()
def f(x):
    review=x['review']
    tokens = tokenizer(review)
    if len(tokens.input_ids) > 545:
        review=x['review_summary']
        print(len(tokens.input_ids))
    return clean_response(sentiment_pipeline(review, truncation=True))
d[['sentiment_score','classification_confidence']] = d['review'].apply(f)
d['error'] = np.abs(d['score'] - d['sentiment_score'])

#%%
macro_averaged_mean_absolute_error(d['score'].values , d['sentiment_score'].values)
