import pandas as pd
import re
import string
import torch
import spacy
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from emoji_translate.emoji_translate import Translator
import emoji

# Loads tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# Cleans input hashtag from user (removes punctuations, spaces, emojis and numbers)
def clean_user_input(text):
    clean_user_text = ''
    for char in text:
        if char not in string.punctuation and not emoji.is_emoji(char) and not char.isnumeric():
            clean_user_text += char
    clean_user_text_no_space = clean_user_text.replace(' ', '')
    return clean_user_text_no_space


# Translates new emojis into text
def translate_emojis(tweet):
    emo = Translator(exact_match_only=False, randomize=True)
    tweet_translated = emo.demojify(tweet)
    return tweet_translated


# Cleans text
def clean_text(sentence):
    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', ':-(': 'sad', ':-<': 'sad',
              ':P': 'raspberry', ':O': 'surprised', ':D': 'smile', 'XD': 'laughing',
              ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed', ':#': 'mute', ':X': 'mute',
              ':^)': 'smile', ':-&': 'confused',
              '$_$': 'greedy', '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj',
              ":'-)": 'sad smile', ';)': 'wink', ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip',
              '=^.^=': 'cat'}

    cleaned_sentence = sentence.lower()  # Lowercase text
    cleaned_sentence = re.sub(r"\S*https?:\S*", "", cleaned_sentence)  # No url
    for emoji in emojis.keys():
        cleaned_sentence = cleaned_sentence.replace(emoji, emojis[emoji])  # Translating old emojis
    re_list = ['@[A-Za-z0–9_]+', '#']  # No @mentions or hashtags symbols
    combined_re = re.compile('|'.join(re_list))
    cleaned_sentence = re.sub(combined_re, '', cleaned_sentence)
    cleaned_sentence = re.sub('[0-9]+', '', cleaned_sentence)  # No numbers
    cleaned_sentence = "".join([i for i in cleaned_sentence if i not in string.punctuation])  # No punctuations
    return cleaned_sentence


# Analyzes tweet and gives sentiment score (1-5) using BERT pretrained model
def sentiment_score(tweet):
    tokens = tokenizer.encode(tweet, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


# Converts df to csv
def convert_df(df):
    return df.to_csv().encode('utf-8')
