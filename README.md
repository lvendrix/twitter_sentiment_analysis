# SenseTwitter (Twitter Sentiment Analysis App)
[SenseTwitter](https://share.streamlit.io/lvendrix/twitter_sentiment_analysis/main/app.py) is a Twitter sentiment analysis application built in Python. It scrapes tweets based on the user's input: hashtags to look for (up to two), the number of tweets desired, and the selected language.  After preprocessing the tweets, they are being analyzed using a pre-trained NLP model [Bert Base Multilingual Uncased Model](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) and are given a sentiment score from 1 (Negative) to 5 (Positive) for each tweet. In the end, a general sentiment score as well as a histogram and wordclouds are given. Users are also given the option to download the data (original and processed tweets, and sentiment score) as a csv for further analysis.
Official link: [SenseTwitter](https://share.streamlit.io/lvendrix/twitter_sentiment_analysis/main/app.py)

## Live demo

![](/Visuals/SenseTwitter_gif.gif)

## Installation

Python 3.9

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages to run the app locally.

```bash
pip install twint
pip install plotly
pip install matplotlib
pip install wordcloud
pip install streamlit
pip install spacy
pip install randfacts
pip install torch
pip install transformers
pip install emoji-translate
pip install emoji
```

# Usage

| Filename                             | Usage                                                     |
|--------------------------------------|-----------------------------------------------------------|
| app.py | App file containing the application (Streamlit app)
| functions.py | Functions file contained all the functions used by app.py|
| Visuals | Folder containing visuals|
| requirements.txt | txt file with all required packages|

# NLP model

As stated above, a pretrained model '[Bert Base Multilingual Uncased Model](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)' is being used to analyze the tweets and give a general sentiment score from 1 to 5. The model is built on a large number of reviews (500k+) in multiple languages. Its accuracy is not perfect but already pretty good.

* Accuracy (exact): 67% for English and +/-60% for other languages.
* Accuracy (off-by-1): 95%

# Screenshots
![](/Visuals/SenseTwitter_start.png)
![](/Visuals/SenseTwitter_progress.png)
![](/Visuals/SenseTwitter_dataframe.png)
![](/Visuals/SenseTwitter_histogram.png)
![](/Visuals/SenseTwitter_wordcloud_all.png)
![](/Visuals/SenseTwitter_wordcloud_neg.png)
![](/Visuals/SenseTwitter_csv.png)

# Further improvements 
* Introduce session states in the app, to protect some variables from being reloaded. When downloading the data, the whole page reloads and information is lost.
* Optimize the pipeline of scraping/preprocessing/analyzing the tweets, to gain efficiency and speed.

# Author
[Logan Vendrix](https://www.linkedin.com/in/loganvendrix/) @Becode

# Timeline
11/10/2021 - 14/10/2021