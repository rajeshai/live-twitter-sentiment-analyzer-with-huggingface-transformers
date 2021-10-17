import streamlit as st
import tweepy as tw
import pandas as pd
from transformers import pipeline


consumer_key = 'OCgWzDW6PaBvBeVimmGBqdAg1'
consumer_secret = 'tBKnmyg5Jfsewkpmw74gxHZbbZkGIH6Ee4rsM0lD1vFL7SrEIM'
access_token = '1449663645412065281-LNjZoEO9lxdtxPcmLtM35BRdIKYHpk'
access_token_secret = 'FL3SGsUWSzPVFnG7bNMnyh4vYK8W1SlABBNtdF7Xcbh7a'
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)
classifier = pipeline('sentiment-analysis')

st.title('Live Twitter Sentiment Analysis with Tweepy and HuggingFace Transformers')
st.markdown('This app uses tweepy to get tweets from twitter based on the input name/phrase. It then processes the tweets through HuggingFace transformers pipeline function for sentiment analysis. The resulting sentiments and corresponding tweets are then put in a dataframe for display which is what you see as result.')

def run():

    with st.form(key='Enter name'):
        search_words = st.text_input('Enter the name for which you want to know the sentiment')
        number_of_tweets = st.number_input('Enter the number of latest tweets for which you want to know the sentiment(Maximum 50 tweets)', 0,50,10)
        submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        tweets =tw.Cursor(api.search_tweets,q=search_words,lang="en").items(number_of_tweets)
        tweet_list = [i.text for i in tweets]
        p = [i for i in classifier(tweet_list)]
        q=[p[i]['label'] for i in range(len(p))]
        df = pd.DataFrame(list(zip(tweet_list, q)),columns =['Latest '+str(number_of_tweets)+' Tweets'+' on '+search_words, 'sentiment'])
        st.write(df)

if __name__=='__main__':
    run()
