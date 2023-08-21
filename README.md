# Fine Grained Sentiment Analysis

This is a quick analysis of 30k multilingual customer reviews, with the goal of reporting multi-class sentiment scores on a scale of 1-5 'star' ratings.

A BERT LLM that has been trained on multilingual customer review data base is leveraged from the HuggingFace library to achieve this.

# [Model Source](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

# Model training data:

|Language |   Number of reviews|
|---------|---------|
|English |    150k|
|Dutch |      80k|
|German |     137k|
|French |     140k|
|Italian |    72k|
|Spanish |    50k|

# Setup

```
python3.7 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# TODO: 

* Running in batch mode
* Fine tuning
* Run a translation model to translate reviews into most heavily used language in the bert training data
  * Test on other Python versions
* Only tested in python3.7
* Develop alternative method?