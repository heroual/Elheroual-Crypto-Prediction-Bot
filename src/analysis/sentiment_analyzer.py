"""
Bot de Prédiction de Cryptomonnaies By ELHEROUAL
================================================
Module: Analyseur de Sentiments
Copyright (c) 2024
Version: 1.0.0

Module d'analyse des sentiments qui collecte et analyse les opinions
sur les cryptomonnaies à partir de Twitter, Reddit et d'autres sources
pour évaluer le sentiment général du marché.
"""

import tweepy
from textblob import TextBlob
import praw
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        # Twitter setup
        self.twitter_api = self._setup_twitter()
        # Reddit setup
        self.reddit_api = self._setup_reddit()
        
    def _setup_twitter(self) -> Optional[tweepy.API]:
        """Setup Twitter API connection."""
        try:
            auth = tweepy.OAuthHandler(
                os.getenv('TWITTER_API_KEY'),
                os.getenv('TWITTER_API_SECRET')
            )
            auth.set_access_token(
                os.getenv('TWITTER_ACCESS_TOKEN'),
                os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
            )
            return tweepy.API(auth, wait_on_rate_limit=True)
        except Exception as e:
            logger.error(f"Error setting up Twitter API: {e}")
            return None

    def _setup_reddit(self) -> Optional[praw.Reddit]:
        """Setup Reddit API connection."""
        try:
            return praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'CryptoPredictionBot 1.0')
            )
        except Exception as e:
            logger.error(f"Error setting up Reddit API: {e}")
            return None

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of a piece of text."""
        try:
            analysis = TextBlob(text)
            return {
                'polarity': analysis.sentiment.polarity,
                'subjectivity': analysis.sentiment.subjectivity,
                'sentiment': self._classify_sentiment(analysis.sentiment.polarity)
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutre'}

    def _classify_sentiment(self, polarity: float) -> str:
        """Classify sentiment based on polarity score."""
        if polarity > 0.3:
            return 'très positif'
        elif polarity > 0:
            return 'positif'
        elif polarity < -0.3:
            return 'très négatif'
        elif polarity < 0:
            return 'négatif'
        return 'neutre'

    def get_twitter_sentiment(self, crypto_symbol: str, count: int = 100) -> Dict:
        """Get sentiment analysis from recent tweets about a cryptocurrency."""
        try:
            if not self.twitter_api:
                return {'error': 'Twitter API not configured'}

            query = f"#{crypto_symbol} OR {crypto_symbol} crypto -filter:retweets"
            tweets = self.twitter_api.search_tweets(
                q=query,
                lang="en",
                count=count,
                tweet_mode="extended"
            )

            sentiments = []
            for tweet in tweets:
                sentiment = self.analyze_sentiment(tweet.full_text)
                sentiments.append({
                    'text': tweet.full_text,
                    'created_at': tweet.created_at,
                    **sentiment
                })

            df = pd.DataFrame(sentiments)
            return self._aggregate_sentiment(df)

        except Exception as e:
            logger.error(f"Error getting Twitter sentiment: {e}")
            return {'error': str(e)}

    def get_reddit_sentiment(self, crypto_symbol: str, subreddit: str = 'cryptocurrency', limit: int = 100) -> Dict:
        """Get sentiment analysis from recent Reddit posts about a cryptocurrency."""
        try:
            if not self.reddit_api:
                return {'error': 'Reddit API not configured'}

            subreddit = self.reddit_api.subreddit(subreddit)
            posts = subreddit.search(crypto_symbol, limit=limit, time_filter='week')

            sentiments = []
            for post in posts:
                # Analyze post title and body
                title_sentiment = self.analyze_sentiment(post.title)
                body_sentiment = self.analyze_sentiment(post.selftext)

                # Combine sentiments with weighted average
                combined_sentiment = {
                    'polarity': (title_sentiment['polarity'] * 0.4 + body_sentiment['polarity'] * 0.6),
                    'subjectivity': (title_sentiment['subjectivity'] * 0.4 + body_sentiment['subjectivity'] * 0.6)
                }

                sentiments.append({
                    'title': post.title,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'score': post.score,
                    **combined_sentiment
                })

            df = pd.DataFrame(sentiments)
            return self._aggregate_sentiment(df)

        except Exception as e:
            logger.error(f"Error getting Reddit sentiment: {e}")
            return {'error': str(e)}

    def _aggregate_sentiment(self, df: pd.DataFrame) -> Dict:
        """Aggregate sentiment metrics from a DataFrame of sentiment data."""
        try:
            if df.empty:
                return {
                    'average_polarity': 0,
                    'average_subjectivity': 0,
                    'sentiment_counts': {'neutre': 1},
                    'sentiment_score': 0,
                    'confidence': 0
                }

            # Calculate basic metrics
            metrics = {
                'average_polarity': df['polarity'].mean(),
                'average_subjectivity': df['subjectivity'].mean(),
                'sentiment_counts': df['sentiment'].value_counts().to_dict(),
                'total_posts': len(df)
            }

            # Calculate weighted sentiment score (-100 to 100)
            metrics['sentiment_score'] = (metrics['average_polarity'] * 100)

            # Calculate confidence based on volume and consistency
            volume_factor = min(df.shape[0] / 100, 1)  # Normalize by expected volume
            consistency_factor = 1 - df['polarity'].std()  # Higher consistency = lower std dev
            metrics['confidence'] = ((volume_factor + consistency_factor) / 2) * 100

            return metrics

        except Exception as e:
            logger.error(f"Error aggregating sentiment: {e}")
            return {
                'average_polarity': 0,
                'average_subjectivity': 0,
                'sentiment_counts': {'neutre': 1},
                'sentiment_score': 0,
                'confidence': 0
            }

    def get_combined_sentiment(self, crypto_symbol: str) -> Dict:
        """Get combined sentiment analysis from all sources."""
        try:
            twitter_sentiment = self.get_twitter_sentiment(crypto_symbol)
            reddit_sentiment = self.get_reddit_sentiment(crypto_symbol)

            # Combine sentiment scores with weights
            weights = {
                'twitter': 0.6,
                'reddit': 0.4
            }

            combined_score = (
                twitter_sentiment.get('sentiment_score', 0) * weights['twitter'] +
                reddit_sentiment.get('sentiment_score', 0) * weights['reddit']
            )

            combined_confidence = (
                twitter_sentiment.get('confidence', 0) * weights['twitter'] +
                reddit_sentiment.get('confidence', 0) * weights['reddit']
            )

            return {
                'combined_sentiment_score': combined_score,
                'confidence': combined_confidence,
                'twitter_sentiment': twitter_sentiment,
                'reddit_sentiment': reddit_sentiment,
                'analysis_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting combined sentiment: {e}")
            return {
                'error': str(e),
                'combined_sentiment_score': 0,
                'confidence': 0
            }
