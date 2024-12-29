from pycoingecko import CoinGeckoAPI
from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class MarketDataCollector:
    def __init__(self):
        self.cg = CoinGeckoAPI()
        self.binance_client = Client(
            os.getenv('BINANCE_API_KEY'),
            os.getenv('BINANCE_SECRET_KEY')
        )

    def get_coin_market_data(self, coin_id, vs_currency='usd', days=30):
        """Fetch historical market data from CoinGecko."""
        try:
            data = self.cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency=vs_currency,
                days=days
            )
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None

    def get_binance_klines(self, symbol, interval='1d', limit=500):
        """Fetch kline/candlestick data from Binance."""
        try:
            klines = self.binance_client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close',
                'volume', 'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignored'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching Binance data: {e}")
            return None

    def get_top_coins(self, limit=100):
        """Get top cryptocurrencies by market cap."""
        try:
            coins = self.cg.get_coins_markets(
                vs_currency='usd',
                order='market_cap_desc',
                per_page=limit,
                sparkline=False
            )
            return pd.DataFrame(coins)
        except Exception as e:
            print(f"Error fetching top coins: {e}")
            return None
