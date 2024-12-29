import tweepy
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        auth = tweepy.OAuthHandler(
            os.getenv('TWITTER_API_KEY'),
            os.getenv('TWITTER_API_SECRET')
        )
        auth.set_access_token(
            os.getenv('TWITTER_ACCESS_TOKEN'),
            os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        )
        self.api = tweepy.API(auth, wait_on_rate_limit=True)

    def analyze_sentiment(self, text):
        """Analyze sentiment of text using TextBlob."""
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }

    def get_crypto_sentiment(self, crypto_symbol, count=100):
        """Get sentiment analysis from recent tweets about a cryptocurrency."""
        try:
            tweets = self.api.search_tweets(
                q=f"#{crypto_symbol} OR {crypto_symbol} crypto",
                lang="en",
                count=count,
                tweet_mode="extended"
            )

            sentiment_data = []
            for tweet in tweets:
                sentiment = self.analyze_sentiment(tweet.full_text)
                sentiment_data.append({
                    'timestamp': tweet.created_at,
                    'text': tweet.full_text,
                    'polarity': sentiment['polarity'],
                    'subjectivity': sentiment['subjectivity']
                })

            return pd.DataFrame(sentiment_data)
        except Exception as e:
            print(f"Error fetching Twitter sentiment: {e}")
            return None

    def get_aggregated_sentiment(self, crypto_symbol):
        """Get aggregated sentiment metrics for a cryptocurrency."""
        df = self.get_crypto_sentiment(crypto_symbol)
        if df is not None:
            return {
                'average_polarity': df['polarity'].mean(),
                'average_subjectivity': df['subjectivity'].mean(),
                'sentiment_count': len(df),
                'positive_sentiment': len(df[df['polarity'] > 0]),
                'negative_sentiment': len(df[df['polarity'] < 0]),
                'neutral_sentiment': len(df[df['polarity'] == 0])
            }
        return None
