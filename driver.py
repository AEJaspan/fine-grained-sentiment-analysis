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
# Running in batch mode
# Fine tuning
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
from langdetect.lang_detect_exception import LangDetectException
# https://stats.stackexchange.com/questions/338904/measures-of-ordinal-classification-error-for-ordinal-regression
from imblearn.metrics import macro_averaged_mean_absolute_error
from torch.nn import Softmax
from lxml import html

def strip_html(s):
    return str(html.fromstring(s).text_content())
smax = Softmax(dim=-1)

#%%

DetectorFactory.seed = 0
# Use a pipeline as a high-level helper
# sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
data=pd.read_csv('data/finegrained_sentiment_analysis.csv')
data.dropna(inplace=True)
print(data.shape)
print(data['score'].value_counts())
#%%
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
data=data[:]
#%%

print(data['score'].value_counts())
#%%
text_columns = ['review_summary', 'review']
def detect_language(x):
    review = strip_html(x['review'])
    if (review == '') | (review == np.nan):
        return np.nan
    try:
        lan = detect(review)
    except LangDetectException as e:
        print(f'Insufficient input text: {x["review"]} - {e}')
        lan=np.nan
    return lan
# data['language'] = data.apply(detect_language, axis=1)
# print(data['language'].value_counts())
# en    25272
# de        6
# es        5
# pt        1
# fr        1
# cy        1
# tl        1
# it        1
# nl        1
# af        1

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
    tokens_review = tokenizer(review, padding=True, truncation=True, return_tensors="pt")
    review_summary=x['review_summary']
    tokens_review_summary = tokenizer(review_summary, padding=True, truncation=True, return_tensors="pt")
    if len(tokens_review.input_ids) > 545:
        print(len(tokens_review.input_ids))
    with torch.no_grad():
        logits_review = model(**tokens_review, return_dict=True).logits
        logits_review_summary = model(**tokens_review_summary, return_dict=True).logits
    probs_review = logits_review.softmax(-1)[0]
    probs_review_summary = logits_review_summary.softmax(-1)[0]
    probs_combination = (logits_review * logits_review_summary).softmax(-1)[0]
    labels = model.config.id2label
    review_class_id = probs_review.argmax().item()
    review_summary_class_id = probs_review_summary.argmax().item()
    review_combination_class_id = probs_combination.argmax().item()
    return pd.Series({
        'sentiment_score': int(labels[review_combination_class_id].split(' ')[0]),
        'classification_confidence': probs_combination[review_combination_class_id].item(),
        'sentiment_score_review': int(labels[review_class_id].split(' ')[0]),
        'classification_confidence_review': probs_review[review_class_id].item(),
        'sentiment_score_review_summary': int(labels[review_summary_class_id].split(' ')[0]),
        'classification_confidence_review_summary': probs_review_summary[review_summary_class_id].item()
    }, index=['sentiment_score','classification_confidence', 'sentiment_score_review', 'classification_confidence_review', 'sentiment_score_review_summary', 'classification_confidence_review_summary'])
    # outputs = model(**tokens, return_dict=True)
    # probs0 = smax(outputs.logits)
    # probs0 = probs0.flatten().detach().numpy()
    # prob_pos = probs[1]
    # return model(**tokens, return_dict=True)
d[['sentiment_score','classification_confidence', 'sentiment_score_review', 'classification_confidence_review', 'sentiment_score_review_summary', 'classification_confidence_review_summary']] = d.apply(f, axis=1)
d['error'] = np.abs(d['score'] - d['sentiment_score'])
d
#%%
# d=data.copy()
# def f(x):
#     review=x['review']
#     tokens = tokenizer(review)
#     if len(tokens.input_ids) > 545:
#         review=x['review_summary']
#         print(len(tokens.input_ids))
#     return clean_response(sentiment_pipeline(review, truncation=True))
# d[['sentiment_score','classification_confidence']] = d['review'].apply(f)
# d['error'] = np.abs(d['score'] - d['sentiment_score'])

#%%
print(d['score'].value_counts(), d['sentiment_score'].value_counts())
print(np.sort(d['score'].unique()), np.sort(d['sentiment_score'].unique()))
if np.array_equal(np.sort(d['score'].unique()), np.sort(d['sentiment_score'].unique())):
    # NOTE:
    # ValueError: Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required.
    # --> Zero cases of an imbalanced feature in the data
    print('Warning, macro averaged will not work.')
macro_averaged_mean_absolute_error(d['score'].values , d['sentiment_score'].values, sample_weight=d['votes'])
