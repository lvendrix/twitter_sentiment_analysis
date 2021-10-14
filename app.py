import streamlit as st
import twint
import plotly.express as px
import randfacts
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from functions import *

# Configuration of streamlit
st.set_page_config(
    page_title="SenseTwitter",
    page_icon="🐦",
    layout='centered',
    initial_sidebar_state="collapsed")

# Main text
st.markdown(f"<h1 style='text-align: center;'>🐦 #Twitter Sentiment Analysis</h1>", unsafe_allow_html=True)

hashtag_1 = st.text_input('Insert a #hashtag you wish to analyse', value="squidgame")

hashtags_to_scrape = hashtag_1
option = st.radio('Want to analyze more than 1 hashtag? (Max 3)', ('No', 'Yes'))
if option == 'Yes':
    hashtag_2 = st.text_input('Hashtag #2')
    if hashtag_2:
        hashtags_to_scrape = hashtags_to_scrape + ', ' + hashtag_2
    hashtag_3 = st.text_input('Hashtag #3')
    if hashtag_3:
        hashtags_to_scrape = hashtags_to_scrape + ', ' + hashtag_3


st.write(f'Current hashtags: __{hashtags_to_scrape}__')

number_tweets = st.slider('How many tweets do you want to analyze? (The more, the longer the processing time)', 0, 1000, 100)

# Sidebar information
st.sidebar.title('Information')
st.sidebar.write("App created by [Logan Vendrix](https://www.linkedin.com/in/loganvendrix/)")
st.sidebar.write("Using Twint, Bert (Hugging Face), Plotly and WordCloud")
st.sidebar.write("Happy scraping! Happy analysis!")
st.sidebar.write("Interested in the project? Check it out on [Github](https://github.com/lvendrix/twitter_sentiment_analysis)")

# Selects a language
selected_language = st.selectbox('Which language are you interested in?', ('English', 'French', 'German', 'Dutch', 'Spanish'))
if selected_language == 'English':
    language = "en"
elif selected_language == 'French':
    language = "fr"
elif selected_language == 'German':
    language = "de"
elif selected_language == 'Dutch':
    language = "nl"
elif selected_language == 'Spanish':
    language = "es"

col1, col2, col3 = st.columns(3)
# Button to scrape and analyze
if col2.button('Scrape and analyze!'):
    with st.spinner('In progress...'):
        st.write(f"*Summary:* {str(number_tweets)} tweets about {hashtags_to_scrape} in {selected_language} are being scraped and analyzed.")
        x = randfacts.get_fact()
        st.write("")
        st.write(f"*Did you know?* {x}")

        # TWINT configuration
        c = twint.Config()
        c.Search = hashtags_to_scrape

        st.write("")
        st.write("*Scraping the tweets ...*")

        c.Lang = language
        c.Limit = round(number_tweets*2)  # Collects more than what the user wants because we filter more after
        c.Pandas = True  # saved as a pd dataframe
        twint.run.Search(c)

        # Convert it into an easier-to-read dataframe variable
        twint_df = twint.storage.panda.Tweets_df

        # Filter english reviews only (problem with TWINT even if selects only 'en')
        tweets_df_eng = twint_df.loc[twint_df['language'] == language]

        # Only keeps the 'tweet' column and the number of tweets selected by the user
        df = tweets_df_eng[['tweet']]
        df = df.iloc[0:number_tweets]

        st.write("*Preprocessing the tweets ...*")

        # Cleans tweets before sentiment score and creats a new column
        df['tweet_cleaned'] = df['tweet'].apply(lambda x: translate_emojis(x))
        df['tweet_cleaned'] = df['tweet_cleaned'].apply(lambda x: clean_text(x))

        st.write("*Calculating sentiment score ...*")

        # Calculates sentiment score and creates a new column
        df['sentiment'] = df['tweet_cleaned'].apply(lambda x: sentiment_score(x))
        df = df.reset_index(drop=True)

    st.success('Done!')

    # Presenting the dataframe and sentiment analysis plot
    st.markdown(f"<h2 style='text-align: center;'>Sentiment Analysis of #{hashtags_to_scrape}</h2>", unsafe_allow_html=True)
    st.write(df)
    csv = convert_df(df)

    # Downloads dataframe as csv if button push
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f'SenseTwitter_Analysis.csv',
        mime='text/csv')

    # Plots sentiment analysis
    fig = px.histogram(df, x=df.sentiment)
    fig.update_layout(title_text="Sentiment Score Histogram", title_x=0.5)
    fig.update_xaxes(title_text='Sentiment score from 1 (negative) to 5 (positive)')
    fig.update_yaxes(title_text='Count')
    fig.update_layout(bargap=.2)
    st.plotly_chart(fig)

    score = df.sentiment.mean()
    # Showing average sentiment score
    st.metric(label="Average Sentiment Score", value=f"{round(score,2)}/5")

    # Showing wordcloud (All, Positive and Negative)
    st.markdown(f"<h4 style='text-align: center;'>WordClouds of {hashtags_to_scrape}</h4>", unsafe_allow_html=True)

    tweet_All = " ".join(review for review in df['tweet_cleaned'])
    tweet_pos = " ".join(review for review in df[df['sentiment'] > 3].tweet_cleaned)
    tweet_neg = " ".join(review for review in df[df['sentiment'] < 3].tweet_cleaned)

    fig, ax = plt.subplots(3, 1, figsize=(30, 30))
    # Create and generate a word cloud image
    wordcloud_ALL = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_All)
    wordcloud_POS = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_pos)
    wordcloud_NEG = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_neg)

    # Display the generated image
    ax[0].imshow(wordcloud_ALL, interpolation='bilinear')
    ax[0].set_title('All Tweets', fontsize=30, pad=20)
    ax[0].axis('off')
    ax[1].imshow(wordcloud_POS, interpolation='bilinear')
    ax[1].set_title('Positive Tweets', fontsize=30, pad=20)
    ax[1].axis('off')
    ax[2].imshow(wordcloud_NEG, interpolation='bilinear')
    ax[2].set_title('Negative Tweets', fontsize=30, pad=20)
    ax[2].axis('off')

    st.pyplot(fig)
