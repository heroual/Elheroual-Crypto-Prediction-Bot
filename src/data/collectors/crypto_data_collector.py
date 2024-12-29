"""
Bot de Prédiction de Cryptomonnaies By ELHEROUAL
================================================
Module: Collecteur de Données
Copyright (c) 2024
Version: 1.0.0

Module de collecte de données qui récupère les informations sur les
cryptomonnaies, y compris les prix historiques, les métriques du marché,
et les données fondamentales à partir de diverses sources.
"""

from pycoingecko import CoinGeckoAPI
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataCollector:
    def __init__(self):
        self.cg = CoinGeckoAPI()
        self._global_market_cap = None
        self._last_global_update = None
        
    def get_all_coins(self) -> List[Dict]:
        """Get list of all available coins with basic info."""
        try:
            coins = self.cg.get_coins_markets(
                vs_currency='usd',
                order='market_cap_desc',
                per_page=250,  # Adjust based on needs
                sparkline=True,
                price_change_percentage='1h,24h,7d,30d'
            )
            return coins
        except Exception as e:
            logger.error(f"Error fetching all coins: {e}")
            return []

    def get_coin_details(self, coin_id: str) -> Optional[Dict]:
        """Get detailed information about a specific coin."""
        try:
            return self.cg.get_coin_by_id(
                id=coin_id,
                localization=False,
                tickers=True,
                market_data=True,
                community_data=True,
                developer_data=True
            )
        except Exception as e:
            logger.error(f"Error fetching coin details for {coin_id}: {e}")
            return None

    def get_global_market_cap(self) -> float:
        """Get the total global cryptocurrency market cap."""
        try:
            # Cache the global market cap for 5 minutes
            now = datetime.now()
            if (self._global_market_cap is None or 
                self._last_global_update is None or 
                (now - self._last_global_update).total_seconds() > 300):
                
                global_data = self.cg.get_global()
                self._global_market_cap = global_data.get('total_market_cap', {}).get('usd', 0)
                self._last_global_update = now
            
            return self._global_market_cap
        except Exception as e:
            logger.error(f"Error fetching global market cap: {e}")
            return 0

    def get_historical_data(self, coin_id: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical price data for a specific coin."""
        try:
            data = self.cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency='usd',
                days=days
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {coin_id}: {e}")
            return None

    def get_market_metrics(self, coin_id: str) -> Dict:
        """Calculate important market metrics for a coin."""
        try:
            df = self.get_historical_data(coin_id)
            if df is None:
                return {}

            metrics = {
                'volatility': df['price'].pct_change().std() * np.sqrt(365),  # Annualized volatility
                'max_drawdown': self._calculate_max_drawdown(df['price']),
                'volume_trend': df['volume'].tail(7).mean() / df['volume'].tail(30).mean(),
                'price_momentum': df['price'].pct_change(7).iloc[-1],
                'market_cap_rank': self.get_coin_details(coin_id).get('market_cap_rank', None)
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating market metrics for {coin_id}: {e}")
            return {}

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate the maximum drawdown from peak."""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min()

    def get_coin_fundamentals(self, coin_id: str) -> Dict:
        """Get fundamental analysis metrics for a coin."""
        try:
            details = self.get_coin_details(coin_id)
            if not details:
                return {}

            return {
                'developer_score': details.get('developer_score', 0),
                'community_score': details.get('community_score', 0),
                'liquidity_score': details.get('liquidity_score', 0),
                'public_interest_score': details.get('public_interest_score', 0),
                'market_maturity': self._calculate_market_maturity(details),
                'development_activity': self._analyze_development_activity(details)
            }
        except Exception as e:
            logger.error(f"Error getting fundamentals for {coin_id}: {e}")
            return {}

    def _calculate_market_maturity(self, details: Dict) -> float:
        """Calculate market maturity score based on various metrics."""
        try:
            factors = {
                'age': min(details.get('genesis_date', '2024') != '2024', 1.0),
                'market_cap': min(details.get('market_data', {}).get('market_cap', {}).get('usd', 0) / 1e9, 1.0),
                'exchanges': min(len(details.get('tickers', [])) / 100, 1.0)
            }
            return sum(factors.values()) / len(factors)
        except Exception:
            return 0.0

    def _analyze_development_activity(self, details: Dict) -> Dict:
        """Analyze development activity metrics."""
        try:
            dev_data = details.get('developer_data', {})
            return {
                'commits': dev_data.get('commits', 0),
                'stars': dev_data.get('stars', 0),
                'contributors': dev_data.get('contributors', 0),
                'recent_updates': dev_data.get('code_additions_deletions_4_weeks', {})
            }
        except Exception:
            return {}
