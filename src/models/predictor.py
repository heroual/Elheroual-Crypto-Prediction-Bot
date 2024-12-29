"""
Bot de Prédiction de Cryptomonnaies By ELHEROUAL
================================================
Module: Prédicteur
Copyright (c) 2024
Version: 1.0.0

Module de prédiction qui utilise des algorithmes d'apprentissage automatique
pour prévoir les mouvements de prix des cryptomonnaies en se basant sur
les données historiques et les indicateurs techniques.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

class CryptoPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.features = None

    def prepare_features(self, df):
        """Prepare features for prediction."""
        # Calculate technical indicators
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        
        # Calculate price changes
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # Drop NaN values
        df = df.dropna()
        
        return df

    def train_model(self, market_data, sentiment_data=None):
        """Train the prediction model using market and sentiment data."""
        df = self.prepare_features(market_data)
        
        if sentiment_data is not None:
            df = pd.merge(df, sentiment_data, on='timestamp', how='left')
            df['polarity'] = df['polarity'].fillna(0)
            df['subjectivity'] = df['subjectivity'].fillna(0)
        
        # Prepare features and target
        features = ['sma_20', 'sma_50', 'macd', 'macd_signal', 'rsi', 
                   'price_change', 'volatility']
        if sentiment_data is not None:
            features.extend(['polarity', 'subjectivity'])
            
        X = df[features].values
        y = df['close'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        self.features = features

    def predict(self, market_data, sentiment_data=None):
        """Make price predictions."""
        df = self.prepare_features(market_data)
        
        if sentiment_data is not None:
            df = pd.merge(df, sentiment_data, on='timestamp', how='left')
            df['polarity'] = df['polarity'].fillna(0)
            df['subjectivity'] = df['subjectivity'].fillna(0)
        
        X = df[self.features].values
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        return predictions

    def get_prediction_metrics(self, market_data, sentiment_data=None):
        """Get prediction metrics and confidence scores."""
        predictions = self.predict(market_data, sentiment_data)
        df = self.prepare_features(market_data)
        
        metrics = {
            'predicted_price': predictions[-1],
            'current_price': df['close'].iloc[-1],
            'price_change_predicted': ((predictions[-1] - df['close'].iloc[-1]) / 
                                     df['close'].iloc[-1] * 100),
            'confidence_score': self.model.score(
                self.scaler.transform(df[self.features].values),
                df['close'].values
            )
        }
        
        return metrics
