import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple, List
import joblib
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoPredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        self.target_column = 'price'
        self.prediction_periods = [1, 7, 30]  # Days to predict
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the prediction model."""
        try:
            # Technical indicators (already calculated)
            features = df.copy()
            
            # Price changes
            features['price_change_1d'] = features['price'].pct_change(1)
            features['price_change_7d'] = features['price'].pct_change(7)
            features['price_change_30d'] = features['price'].pct_change(30)
            
            # Volume indicators
            features['volume_change_1d'] = features['volume'].pct_change(1)
            features['volume_change_7d'] = features['volume'].pct_change(7)
            features['volume_ma_ratio'] = features['volume'] / features['volume'].rolling(window=30).mean()
            
            # Volatility
            features['volatility'] = features['price'].pct_change().rolling(window=30).std()
            
            # Market cap indicators
            features['market_cap_change'] = features['market_cap'].pct_change()
            features['market_cap_ma_ratio'] = features['market_cap'] / features['market_cap'].rolling(window=30).mean()
            
            # Remove NaN values
            features = features.dropna()
            
            self.feature_columns = [col for col in features.columns if col != self.target_column]
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()

    def create_sequences(self, data: pd.DataFrame, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        try:
            X, y = [], []
            
            for i in range(len(data) - sequence_length):
                X.append(data[self.feature_columns].iloc[i:(i + sequence_length)].values)
                y.append(data[self.target_column].iloc[i + sequence_length])
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])

    def train(self, df: pd.DataFrame, sequence_length: int = 30) -> None:
        """Train the prediction model."""
        try:
            # Prepare features
            features_df = self.prepare_features(df)
            if features_df.empty:
                raise ValueError("No features available for training")
            
            # Create sequences
            X, y = self.create_sequences(features_df, sequence_length)
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No sequences created for training")
            
            # Scale features
            X_scaled = np.array([self.scaler.fit_transform(x) for x in X])
            
            # Initialize and train model
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            # Reshape X for training
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], -1)
            
            # Train the model
            self.model.fit(X_reshaped, y)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def predict(self, df: pd.DataFrame, sequence_length: int = 30) -> Dict:
        """Make predictions for different time periods."""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            # Prepare features
            features_df = self.prepare_features(df)
            if features_df.empty:
                raise ValueError("No features available for prediction")
            
            # Get the most recent sequence
            latest_sequence = features_df[self.feature_columns].iloc[-sequence_length:].values
            latest_sequence_scaled = self.scaler.transform(latest_sequence)
            
            # Reshape for prediction
            X_pred = latest_sequence_scaled.reshape(1, -1)
            
            # Make prediction
            base_prediction = self.model.predict(X_pred)[0]
            
            # Calculate predictions for different periods
            predictions = {}
            current_price = df['price'].iloc[-1]
            
            for period in self.prediction_periods:
                prediction_value = self._adjust_prediction(base_prediction, period, current_price)
                confidence = self._calculate_confidence(features_df, period)
                
                predictions[f'{period}d'] = {
                    'price': prediction_value,
                    'change_percent': ((prediction_value - current_price) / current_price) * 100,
                    'confidence': confidence
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {}

    def _adjust_prediction(self, base_prediction: float, period: int, current_price: float) -> float:
        """Adjust prediction based on the time period."""
        try:
            # Add some randomness and trend adjustment based on period
            volatility = np.random.normal(0, 0.02 * period)  # Increased uncertainty with longer periods
            trend_factor = 1 + (period / 365)  # Long-term growth assumption
            
            adjusted_prediction = base_prediction * trend_factor * (1 + volatility)
            
            # Ensure prediction is within reasonable bounds
            max_change = 0.5 * (period / 30)  # Maximum 50% change per month
            min_price = current_price * (1 - max_change)
            max_price = current_price * (1 + max_change)
            
            return np.clip(adjusted_prediction, min_price, max_price)
            
        except Exception as e:
            logger.error(f"Error adjusting prediction: {e}")
            return base_prediction

    def _calculate_confidence(self, features_df: pd.DataFrame, period: int) -> float:
        """Calculate confidence score for the prediction."""
        try:
            # Factors affecting confidence
            factors = {
                'data_quality': self._assess_data_quality(features_df),
                'volatility': self._assess_volatility(features_df),
                'prediction_horizon': self._assess_prediction_horizon(period)
            }
            
            # Weight the factors
            weights = {
                'data_quality': 0.4,
                'volatility': 0.3,
                'prediction_horizon': 0.3
            }
            
            confidence = sum(score * weights[factor] for factor, score in factors.items())
            return min(max(confidence * 100, 0), 100)  # Convert to percentage and clip
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 50.0  # Default to medium confidence

    def _assess_data_quality(self, df: pd.DataFrame) -> float:
        """Assess the quality of input data."""
        try:
            # Check for missing values
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            
            # Check data recency
            data_recency = 1.0  # Assume data is recent
            
            # Check data consistency
            value_ranges = df.max() - df.min()
            consistency_score = 1.0 - (value_ranges.std() / value_ranges.mean())
            
            return (1.0 - missing_ratio) * data_recency * consistency_score
            
        except Exception:
            return 0.5

    def _assess_volatility(self, df: pd.DataFrame) -> float:
        """Assess market volatility."""
        try:
            # Calculate recent volatility
            recent_volatility = df['price'].pct_change().tail(30).std()
            
            # Convert to confidence score (higher volatility = lower confidence)
            return 1.0 - min(recent_volatility * 10, 1.0)
            
        except Exception:
            return 0.5

    def _assess_prediction_horizon(self, period: int) -> float:
        """Assess confidence based on prediction horizon."""
        try:
            # Longer periods have lower confidence
            max_period = max(self.prediction_periods)
            return 1.0 - (period / max_period)
            
        except Exception:
            return 0.5

    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        try:
            if self.model is not None:
                joblib.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns
                }, path)
                logger.info(f"Model saved successfully to {path}")
            else:
                raise ValueError("No model to save")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
