"""
Bot de Prédiction de Cryptomonnaies By ELHEROUAL
================================================
Module: Détecteur d'Anomalies
Copyright (c) 2024
Version: 1.0.0

Module de détection d'anomalies dans les mouvements de prix et de volume.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.features = [
            'price_change',
            'volume_change',
            'volatility',
            'rsi',
            'volume_price_ratio'
        ]

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prépare les caractéristiques pour la détection d'anomalies."""
        features_df = pd.DataFrame()
        
        # Variations de prix
        features_df['price_change'] = df['close'].pct_change()
        
        # Variations de volume
        features_df['volume_change'] = df['volume'].pct_change()
        
        # Volatilité (écart-type sur fenêtre mobile)
        features_df['volatility'] = df['close'].rolling(window=14).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df['rsi'] = 100 - (100 / (1 + rs))
        
        # Ratio volume/prix
        features_df['volume_price_ratio'] = df['volume'] / df['close']
        
        return features_df.dropna()

    def fit_detect(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Entraîne le modèle et détecte les anomalies."""
        features_df = self.prepare_features(df)
        
        # Normalisation des caractéristiques
        scaled_features = self.scaler.fit_transform(features_df)
        
        # Détection des anomalies
        anomalies = self.isolation_forest.fit_predict(scaled_features)
        
        return anomalies, features_df

    def analyze_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Analyse les anomalies détectées et génère des explications."""
        anomalies, features_df = self.fit_detect(df)
        
        # Convertir en DataFrame pour faciliter l'analyse
        features_df['anomaly'] = anomalies
        features_df['date'] = df.index[features_df.index]
        
        anomaly_insights = []
        
        # Analyser chaque anomalie
        for idx in features_df[features_df['anomaly'] == -1].index:
            insight = self._generate_anomaly_insight(
                features_df.loc[idx],
                df.loc[features_df['date'][idx]]
            )
            anomaly_insights.append(insight)
        
        return anomaly_insights

    def _generate_anomaly_insight(self, 
                                anomaly_features: pd.Series,
                                price_data: pd.Series) -> Dict:
        """Génère une explication détaillée pour une anomalie."""
        reasons = []
        severity = "faible"
        
        # Analyser les variations de prix
        if abs(anomaly_features['price_change']) > 0.1:
            reasons.append(
                f"Variation de prix importante de "
                f"{anomaly_features['price_change']*100:.1f}%"
            )
            severity = "élevée"
        
        # Analyser les variations de volume
        if abs(anomaly_features['volume_change']) > 0.5:
            reasons.append(
                f"Variation de volume importante de "
                f"{anomaly_features['volume_change']*100:.1f}%"
            )
            severity = "élevée"
        
        # Analyser la volatilité
        if anomaly_features['volatility'] > price_data['close'] * 0.05:
            reasons.append("Volatilité anormalement élevée")
            severity = "élevée"
        
        # Analyser le RSI
        if anomaly_features['rsi'] > 80 or anomaly_features['rsi'] < 20:
            reasons.append(
                f"RSI extrême à {anomaly_features['rsi']:.1f}"
            )
        
        return {
            'date': anomaly_features['date'],
            'severity': severity,
            'reasons': reasons,
            'explanation': self._get_possible_explanation(reasons),
            'recommendation': self._get_recommendation(severity, reasons)
        }

    def _get_possible_explanation(self, reasons: List[str]) -> str:
        """Génère une explication possible pour l'anomalie."""
        if any('volume' in reason.lower() for reason in reasons):
            return ("Possible activité de baleine ou manipulation du marché "
                   "due aux changements importants de volume")
        elif any('prix' in reason.lower() for reason in reasons):
            return ("Réaction possible à des nouvelles importantes ou "
                   "à des événements du marché")
        else:
            return "Comportement inhabituel du marché nécessitant une surveillance"

    def _get_recommendation(self, severity: str, reasons: List[str]) -> str:
        """Génère une recommandation basée sur l'anomalie."""
        if severity == "élevée":
            if any('prix' in reason.lower() for reason in reasons):
                return ("Considérer une prise de profits ou un stop-loss "
                       "pour protéger contre la volatilité")
            else:
                return ("Surveiller de près les mouvements du marché et "
                       "attendre la stabilisation")
        else:
            return "Maintenir la stratégie actuelle avec une surveillance accrue"
