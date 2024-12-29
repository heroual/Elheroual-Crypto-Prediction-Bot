"""
Bot de Prédiction de Cryptomonnaies By ELHEROUAL
================================================
Module: Analyseur Technique
Copyright (c) 2024
Version: 1.0.0

Module d'analyse technique qui calcule et interprète les indicateurs
techniques pour les cryptomonnaies, y compris les moyennes mobiles,
le RSI, le MACD et les bandes de Bollinger.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import ta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
        
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators for the given price data."""
        try:
            # Handle empty DataFrame
            if df is None or df.empty:
                logger.warning("Empty DataFrame provided for technical analysis")
                return self._get_default_indicators()

            # Make sure we have the required columns
            required_columns = ['price', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"DataFrame missing required columns: {required_columns}")
                return self._get_default_indicators()

            # Remove any NaN values
            df_ta = df.copy()
            df_ta = df_ta.dropna()

            if len(df_ta) < 200:  # Need at least 200 data points for meaningful analysis
                logger.warning("Insufficient data points for technical analysis")
                return self._get_default_indicators()

            # Rename columns for ta library
            df_ta['close'] = df_ta['price']
            df_ta['high'] = df_ta['price']
            df_ta['low'] = df_ta['price']
            df_ta['open'] = df_ta['price']

            indicators = {}
            
            # Trend Indicators
            try:
                indicators['sma_20'] = ta.trend.sma_indicator(df_ta['close'], window=20)
                indicators['sma_50'] = ta.trend.sma_indicator(df_ta['close'], window=50)
                indicators['sma_200'] = ta.trend.sma_indicator(df_ta['close'], window=200)
                
                macd = ta.trend.MACD(df_ta['close'])
                indicators['macd_line'] = macd.macd()
                indicators['macd_signal'] = macd.macd_signal()
                indicators['macd_histogram'] = macd.macd_diff()
            except Exception as e:
                logger.error(f"Error calculating trend indicators: {e}")
                indicators.update(self._get_default_trend_indicators())
            
            # Momentum Indicators
            try:
                indicators['rsi'] = ta.momentum.RSIIndicator(df_ta['close']).rsi()
                indicators['stoch_rsi'] = ta.momentum.StochRSIIndicator(df_ta['close']).stochrsi()
            except Exception as e:
                logger.error(f"Error calculating momentum indicators: {e}")
                indicators.update(self._get_default_momentum_indicators())
            
            # Volatility Indicators
            try:
                bb = ta.volatility.BollingerBands(df_ta['close'])
                indicators['bb_high'] = bb.bollinger_hband()
                indicators['bb_low'] = bb.bollinger_lband()
                indicators['bb_mid'] = bb.bollinger_mavg()
            except Exception as e:
                logger.error(f"Error calculating volatility indicators: {e}")
                indicators.update(self._get_default_volatility_indicators())
            
            # Volume Indicators
            try:
                vwap = ta.volume.volume_weighted_average_price(
                    high=df_ta['high'],
                    low=df_ta['low'],
                    close=df_ta['close'],
                    volume=df_ta['volume']
                )
                indicators['volume_ema'] = vwap
            except Exception as e:
                logger.error(f"Error calculating volume indicators: {e}")
                indicators.update(self._get_default_volume_indicators())

            # Get the latest values for each indicator
            latest_indicators = {}
            for key, value in indicators.items():
                if isinstance(value, pd.Series):
                    latest_indicators[key] = value.iloc[-1]
                else:
                    latest_indicators[key] = value

            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(latest_indicators)
            momentum_strength = self._calculate_momentum_strength(latest_indicators)

            return {
                'trend': {
                    'direction': 'haussier' if trend_strength > 0 else 'baissier',
                    'strength': abs(trend_strength),
                    'sma_20': latest_indicators.get('sma_20', 0),
                    'sma_50': latest_indicators.get('sma_50', 0),
                    'sma_200': latest_indicators.get('sma_200', 0)
                },
                'momentum': {
                    'strength': momentum_strength,
                    'rsi': latest_indicators.get('rsi', 50),
                    'stoch_rsi': latest_indicators.get('stoch_rsi', 0.5)
                },
                'volatility': {
                    'bb_high': latest_indicators.get('bb_high', 0),
                    'bb_low': latest_indicators.get('bb_low', 0),
                    'bb_mid': latest_indicators.get('bb_mid', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            return self._get_default_indicators()

    def _calculate_trend_strength(self, indicators: Dict) -> float:
        """Calculate overall trend strength (-1 to 1)."""
        try:
            current_price = indicators.get('close', 0)
            if current_price == 0:
                return 0

            # Compare price to moving averages
            sma_signals = [
                1 if current_price > indicators.get('sma_20', 0) else -1,
                1 if current_price > indicators.get('sma_50', 0) else -1,
                1 if current_price > indicators.get('sma_200', 0) else -1
            ]

            # MACD signal
            macd_signal = 1 if indicators.get('macd_histogram', 0) > 0 else -1

            # Combine signals with weights
            weights = [0.4, 0.3, 0.2, 0.1]  # Weights for different timeframes
            strength = (
                sma_signals[0] * weights[0] +  # Short-term (SMA20)
                sma_signals[1] * weights[1] +  # Medium-term (SMA50)
                sma_signals[2] * weights[2] +  # Long-term (SMA200)
                macd_signal * weights[3]       # MACD confirmation
            )

            return np.clip(strength, -1, 1)
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0

    def _calculate_momentum_strength(self, indicators: Dict) -> float:
        """Calculate momentum strength (0 to 1)."""
        try:
            rsi = indicators.get('rsi', 50)
            stoch_rsi = indicators.get('stoch_rsi', 0.5)

            # RSI strength (0 to 1)
            rsi_strength = abs(rsi - 50) / 50

            # Stochastic RSI strength (0 to 1)
            stoch_strength = abs(stoch_rsi - 0.5) / 0.5

            # Combine with weights
            strength = (rsi_strength * 0.6) + (stoch_strength * 0.4)

            return np.clip(strength, 0, 1)
        except Exception as e:
            logger.error(f"Error calculating momentum strength: {e}")
            return 0

    def _get_default_indicators(self) -> Dict:
        """Return default indicators when analysis fails."""
        return {
            'trend': {
                'direction': 'neutre',
                'strength': 0,
                'sma_20': 0,
                'sma_50': 0,
                'sma_200': 0
            },
            'momentum': {
                'strength': 0,
                'rsi': 50,
                'stoch_rsi': 0.5
            },
            'volatility': {
                'bb_high': 0,
                'bb_low': 0,
                'bb_mid': 0
            }
        }

    def _get_default_trend_indicators(self) -> Dict:
        """Return default trend indicators."""
        return {
            'sma_20': 0,
            'sma_50': 0,
            'sma_200': 0,
            'macd_line': 0,
            'macd_signal': 0,
            'macd_histogram': 0
        }

    def _get_default_momentum_indicators(self) -> Dict:
        """Return default momentum indicators."""
        return {
            'rsi': 50,
            'stoch_rsi': 0.5
        }

    def _get_default_volatility_indicators(self) -> Dict:
        """Return default volatility indicators."""
        return {
            'bb_high': 0,
            'bb_low': 0,
            'bb_mid': 0
        }

    def _get_default_volume_indicators(self) -> Dict:
        """Return default volume indicators."""
        return {
            'volume_ema': 0
        }
