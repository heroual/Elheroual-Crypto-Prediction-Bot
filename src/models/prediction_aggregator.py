"""
Bot de Prédiction de Cryptomonnaies By ELHEROUAL
================================================
Module: Agrégateur de Prédictions
Copyright (c) 2024
Version: 1.0.0

Module d'agrégation des prédictions de différents modèles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ModelAggregator:
    def __init__(self):
        self.models = {
            'lstm': None,
            'xgboost': None,
            'random_forest': None
        }
        self.weights = {
            'lstm': 0.4,
            'xgboost': 0.35,
            'random_forest': 0.25
        }
        self.performance_history = {
            'lstm': [],
            'xgboost': [],
            'random_forest': []
        }

    def create_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Crée un modèle LSTM."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def create_xgboost_model(self) -> xgb.XGBRegressor:
        """Crée un modèle XGBoost."""
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1
        )

    def create_random_forest(self) -> RandomForestRegressor:
        """Crée un modèle Random Forest."""
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

    def prepare_data(self, df: pd.DataFrame, 
                    window_size: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prépare les données pour l'entraînement."""
        features = [
            'close', 'volume', 'high', 'low',
            'sma_20', 'sma_50', 'rsi', 'macd'
        ]
        
        # Calculer les indicateurs techniques
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        
        # Normaliser les données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[features])
        
        X, y = [], []
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i])
            y.append(scaled_data[i, 0])  # Prix de clôture
            
        return np.array(X), np.array(y)

    def train_models(self, df: pd.DataFrame):
        """Entraîne tous les modèles."""
        X, y = self.prepare_data(df)
        
        # Division train/test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # LSTM
        self.models['lstm'] = self.create_lstm_model((X.shape[1], X.shape[2]))
        self.models['lstm'].fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        
        # Préparer les données pour XGBoost et Random Forest
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        
        # XGBoost
        self.models['xgboost'] = self.create_xgboost_model()
        self.models['xgboost'].fit(X_train_2d, y_train)
        
        # Random Forest
        self.models['random_forest'] = self.create_random_forest()
        self.models['random_forest'].fit(X_train_2d, y_train)
        
        # Évaluer les performances
        self._update_weights(X_test, X_test_2d, y_test)

    def _update_weights(self, X_test_3d: np.ndarray, 
                       X_test_2d: np.ndarray, 
                       y_test: np.ndarray):
        """Met à jour les poids des modèles basés sur leurs performances."""
        performances = {
            'lstm': np.mean((self.models['lstm'].predict(X_test_3d) - y_test) ** 2),
            'xgboost': np.mean((self.models['xgboost'].predict(X_test_2d) - y_test) ** 2),
            'random_forest': np.mean((self.models['random_forest'].predict(X_test_2d) - y_test) ** 2)
        }
        
        # Mettre à jour l'historique des performances
        for model, perf in performances.items():
            self.performance_history[model].append(perf)
        
        # Calculer les nouveaux poids
        total_error = sum(1/p for p in performances.values())
        self.weights = {
            model: (1/perf)/total_error 
            for model, perf in performances.items()
        }

    def predict(self, df: pd.DataFrame) -> Dict:
        """Génère des prédictions agrégées."""
        X, _ = self.prepare_data(df)
        X_2d = X.reshape(X.shape[0], -1)
        
        predictions = {
            'lstm': self.models['lstm'].predict(X),
            'xgboost': self.models['xgboost'].predict(X_2d),
            'random_forest': self.models['random_forest'].predict(X_2d)
        }
        
        # Calculer la prédiction pondérée
        weighted_pred = sum(
            pred[-1] * self.weights[model] 
            for model, pred in predictions.items()
        )
        
        # Calculer l'intervalle de confiance
        std_dev = np.std([pred[-1] for pred in predictions.values()])
        confidence_interval = {
            '95%': (weighted_pred - 1.96 * std_dev, 
                   weighted_pred + 1.96 * std_dev),
            '80%': (weighted_pred - 1.28 * std_dev, 
                   weighted_pred + 1.28 * std_dev)
        }
        
        return {
            'prediction': weighted_pred,
            'model_predictions': {
                model: float(pred[-1]) 
                for model, pred in predictions.items()
            },
            'weights': self.weights,
            'confidence_intervals': confidence_interval
        }

    def get_prediction_insights(self, prediction_results: Dict) -> List[str]:
        """Génère des insights sur les prédictions."""
        insights = []
        
        # Analyser la convergence des modèles
        predictions = prediction_results['model_predictions']
        max_diff = max(predictions.values()) - min(predictions.values())
        
        if max_diff < 0.02:
            insights.append(
                "Forte convergence entre les modèles - "
                "Prédiction très fiable"
            )
        elif max_diff < 0.05:
            insights.append(
                "Bonne convergence entre les modèles - "
                "Prédiction fiable"
            )
        else:
            insights.append(
                "Divergence notable entre les modèles - "
                "Prédiction à considérer avec prudence"
            )
        
        # Analyser les performances des modèles
        best_model = max(self.weights.items(), key=lambda x: x[1])[0]
        insights.append(
            f"Le modèle {best_model} montre la meilleure "
            f"performance récente avec un poids de "
            f"{self.weights[best_model]:.2f}"
        )
        
        return insights
