"""
Bot de Prédiction de Cryptomonnaies By ELHEROUAL
================================================
Module: Analyseur de Corrélations
Copyright (c) 2024
Version: 1.0.0

Module d'analyse des corrélations entre les cryptomonnaies et d'autres actifs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import ccxt
import yfinance as yf
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.assets = {
            'indices': ['^GSPC', '^IXIC', '^DJI'],  # S&P 500, NASDAQ, Dow Jones
            'commodities': ['GC=F', 'SI=F'],  # Gold, Silver
            'stocks': ['TSLA', 'NVDA', 'MSTR']  # Tech stocks related to crypto
        }

    def fetch_crypto_data(self, symbol: str, timeframe: str = '1d', 
                         limit: int = 30) -> pd.DataFrame:
        """Récupère les données historiques d'une cryptomonnaie."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, timeframe, limit=limit
            )
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 
                                            'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données crypto: {e}")
            return pd.DataFrame()

    def fetch_traditional_data(self, symbols: List[str], 
                             period: str = '30d') -> pd.DataFrame:
        """Récupère les données des actifs traditionnels."""
        try:
            data = yf.download(symbols, period=period)['Close']
            return data
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données traditionnelles: {e}")
            return pd.DataFrame()

    def calculate_correlations(self, crypto_symbol: str, 
                             timeframe: str = '30d') -> Dict:
        """Calcule les corrélations entre une crypto et d'autres actifs."""
        # Récupérer les données crypto
        crypto_data = self.fetch_crypto_data(
            crypto_symbol, 
            limit=30 if timeframe == '30d' else 7
        )
        if crypto_data.empty:
            return {}

        results = {
            'correlations': {},
            'interpretation': {},
            'strength': {}
        }

        # Calculer les corrélations pour chaque catégorie d'actifs
        for category, symbols in self.assets.items():
            trad_data = self.fetch_traditional_data(symbols, period=timeframe)
            if trad_data.empty:
                continue

            for symbol in symbols:
                if symbol in trad_data.columns:
                    # Aligner les données sur les mêmes dates
                    aligned_data = pd.concat([
                        crypto_data['close'], 
                        trad_data[symbol]
                    ], axis=1).dropna()
                    
                    if not aligned_data.empty:
                        corr = aligned_data.corr().iloc[0, 1]
                        results['correlations'][symbol] = round(corr, 3)
                        
                        # Interpréter la corrélation
                        results['interpretation'][symbol] = self._interpret_correlation(corr)
                        results['strength'][symbol] = self._correlation_strength(corr)

        return results

    def _interpret_correlation(self, corr: float) -> str:
        """Interprète la valeur de corrélation."""
        if corr > 0.7:
            return "Forte corrélation positive"
        elif corr > 0.3:
            return "Corrélation positive modérée"
        elif corr > -0.3:
            return "Faible corrélation"
        elif corr > -0.7:
            return "Corrélation négative modérée"
        else:
            return "Forte corrélation négative"

    def _correlation_strength(self, corr: float) -> str:
        """Détermine la force de la corrélation."""
        abs_corr = abs(corr)
        if abs_corr > 0.8:
            return "Très forte"
        elif abs_corr > 0.6:
            return "Forte"
        elif abs_corr > 0.4:
            return "Modérée"
        elif abs_corr > 0.2:
            return "Faible"
        else:
            return "Très faible"

    def get_correlation_insights(self, crypto_symbol: str, 
                               timeframe: str = '30d') -> List[str]:
        """Génère des insights basés sur les corrélations."""
        correlations = self.calculate_correlations(crypto_symbol, timeframe)
        insights = []

        if not correlations:
            return ["Données insuffisantes pour l'analyse des corrélations."]

        for symbol, corr in correlations['correlations'].items():
            strength = correlations['strength'][symbol]
            interpretation = correlations['interpretation'][symbol]
            
            insight = f"{crypto_symbol} montre une {strength.lower()} corrélation "
            insight += f"({corr:+.2f}) avec {symbol}: {interpretation}."
            insights.append(insight)

        return insights
