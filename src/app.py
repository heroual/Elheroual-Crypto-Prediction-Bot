"""
Bot de Prédiction de Cryptomonnaies By ELHEROUAL
================================================
Copyright (c) 2024
Version: 1.0.0
"""

from flask import Flask, render_template, request, jsonify
from data.collectors.crypto_data_collector import CryptoDataCollector
import logging
import traceback
from datetime import datetime, timedelta
import numpy as np
import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize components
data_collector = CryptoDataCollector()

def generate_prediction_explanation(coin_data, price_changes, lang='fr'):
    """Generate a detailed prediction explanation based on market data."""
    
    # Calculate trend strength
    trend_strength = abs(sum([
        price_changes['1h'] * 0.2,
        price_changes['24h'] * 0.3,
        price_changes['7d'] * 0.3,
        price_changes['30d'] * 0.2
    ]))
    
    # Determine market phase
    translations = {
        'fr': {
            'strong_bullish': "phase haussière forte",
            'moderate_bullish': "phase haussière modérée",
            'strong_bearish': "phase baissière forte",
            'moderate_bearish': "phase baissière modérée",
            'consolidation': "phase de consolidation",
            'very_high': "très élevé",
            'high': "élevé",
            'moderate': "modéré",
            'bullish': "haussière",
            'bearish': "baissière",
            'buy': "achat",
            'sell': "vente",
            'wait': "attente",
            'risk_factors': [
                "Volatilité du marché global des cryptomonnaies",
                "Évolutions réglementaires potentielles",
                "Sentiment général du marché",
                "Événements macroéconomiques"
            ]
        },
        'en': {
            'strong_bullish': "strong bullish phase",
            'moderate_bullish': "moderate bullish phase",
            'strong_bearish': "strong bearish phase",
            'moderate_bearish': "moderate bearish phase",
            'consolidation': "consolidation phase",
            'very_high': "very high",
            'high': "high",
            'moderate': "moderate",
            'bullish': "bullish",
            'bearish': "bearish",
            'buy': "buy",
            'sell': "sell",
            'wait': "wait",
            'risk_factors': [
                "Global cryptocurrency market volatility",
                "Potential regulatory changes",
                "General market sentiment",
                "Macroeconomic events"
            ]
        }
    }
    
    t = translations[lang]
    
    if price_changes['24h'] > 5:
        market_phase = t['strong_bullish']
    elif price_changes['24h'] > 2:
        market_phase = t['moderate_bullish']
    elif price_changes['24h'] < -5:
        market_phase = t['strong_bearish']
    elif price_changes['24h'] < -2:
        market_phase = t['moderate_bearish']
    else:
        market_phase = t['consolidation']

    # Calculate volatility
    volatility = np.std([
        price_changes['1h'],
        price_changes['24h'],
        price_changes['7d'],
        price_changes['30d']
    ])
    
    # Generate trading volume analysis
    volume_to_mcap = coin_data['basic_info']['total_volume'] / coin_data['basic_info']['market_cap']
    if volume_to_mcap > 0.2:
        volume_analysis = t['very_high']
    elif volume_to_mcap > 0.1:
        volume_analysis = t['high']
    else:
        volume_analysis = t['moderate']

    # Generate prediction text
    prediction = {
        'short_term': {
            'horizon': '24 hours' if lang == 'en' else '24 heures',
            'trend': t['bullish'] if price_changes['24h'] > 0 else t['bearish'],
            'confidence': min(abs(price_changes['24h']) * 2, 100),
            'target': coin_data['basic_info']['current_price'] * (1 + (price_changes['24h'] / 100))
        },
        'medium_term': {
            'horizon': '7 days' if lang == 'en' else '7 jours',
            'trend': t['bullish'] if price_changes['7d'] > 0 else t['bearish'],
            'confidence': min(abs(price_changes['7d']), 100),
            'target': coin_data['basic_info']['current_price'] * (1 + (price_changes['7d'] / 100))
        }
    }

    # Generate explanation text
    context_text = {
        'fr': f"{coin_data['basic_info']['name']} est actuellement en {market_phase}. "
              f"Le volume de trading est {volume_analysis}, ce qui indique un intérêt {volume_analysis} des investisseurs.",
        'en': f"{coin_data['basic_info']['name']} is currently in a {market_phase}. "
              f"Trading volume is {volume_analysis}, indicating {volume_analysis} investor interest."
    }
    
    technical_text = {
        'fr': f"L'analyse technique montre une tendance {prediction['short_term']['trend']} à court terme "
              f"avec une volatilité {volatility:.1f}%. Les indicateurs suggèrent une force de tendance de {trend_strength:.1f}%.",
        'en': f"Technical analysis shows a {prediction['short_term']['trend']} trend in the short term "
              f"with {volatility:.1f}% volatility. Indicators suggest a trend strength of {trend_strength:.1f}%."
    }
    
    short_term_text = {
        'fr': f"À court terme (24h), nous prévoyons une tendance {prediction['short_term']['trend']} "
              f"avec un objectif de prix autour de ${prediction['short_term']['target']:.2f} "
              f"(confiance: {prediction['short_term']['confidence']:.1f}%).",
        'en': f"In the short term (24h), we predict a {prediction['short_term']['trend']} trend "
              f"with a price target around ${prediction['short_term']['target']:.2f} "
              f"(confidence: {prediction['short_term']['confidence']:.1f}%)."
    }
    
    medium_term_text = {
        'fr': f"À moyen terme (7j), la tendance devrait rester {prediction['medium_term']['trend']} "
              f"avec un objectif de prix autour de ${prediction['medium_term']['target']:.2f} "
              f"(confiance: {prediction['medium_term']['confidence']:.1f}%).",
        'en': f"In the medium term (7d), the trend should remain {prediction['medium_term']['trend']} "
              f"with a price target around ${prediction['medium_term']['target']:.2f} "
              f"(confidence: {prediction['medium_term']['confidence']:.1f}%)."
    }

    explanation = {
        'market_context': context_text[lang],
        'technical_analysis': technical_text[lang],
        'predictions': {
            'short_term': short_term_text[lang],
            'medium_term': medium_term_text[lang]
        },
        'risk_factors': t['risk_factors'],
        'recommendation': {
            'action': t['buy'] if price_changes['24h'] > 2 else t['sell'] if price_changes['24h'] < -2 else t['wait'],
            'confidence': min(trend_strength, 100)
        }
    }
    
    return explanation

def get_historical_data(coin_id, days=30):
    """Get historical price data for charting."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # Format data for charts
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        
        chart_data = {
            'labels': [],
            'prices': [],
            'volumes': []
        }
        
        for i in range(len(prices)):
            timestamp = datetime.fromtimestamp(prices[i][0]/1000)
            chart_data['labels'].append(timestamp.strftime('%Y-%m-%d'))
            chart_data['prices'].append(prices[i][1])
            chart_data['volumes'].append(volumes[i][1] if i < len(volumes) else 0)
        
        return chart_data
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        return None

def get_translations(lang='fr'):
    """Get translations for all UI elements."""
    translations = {
        'fr': {
            'market_sentiment': {
                'positive': 'Positif',
                'negative': 'Négatif'
            },
            'volatility': {
                'high': 'Élevée',
                'moderate': 'Modérée',
                'low': 'Faible'
            },
            'market_analysis': {
                'sentiment': 'Sentiment du Marché',
                'volatility': 'Volatilité',
                'volume': 'Volume de Trading',
                'dominance': 'Dominance du Marché'
            },
            'price_history': {
                'ath': 'Plus Haut Historique',
                'ath_date': 'Date du Plus Haut',
                'ath_change': 'Variation depuis le Plus Haut',
                'atl': 'Plus Bas Historique',
                'atl_date': 'Date du Plus Bas',
                'atl_change': 'Variation depuis le Plus Bas'
            },
            'social_info': {
                'twitter': 'Abonnés Twitter',
                'reddit': 'Abonnés Reddit',
                'github': 'Statistiques GitHub',
                'forks': 'Forks',
                'stars': 'Stars',
                'subscribers': 'Abonnés',
                'total_issues': 'Issues Totales',
                'closed_issues': 'Issues Fermées',
                'pull_requests': 'Pull Requests Fusionnées',
                'commits': 'Commits (4 semaines)'
            }
        },
        'en': {
            'market_sentiment': {
                'positive': 'Positive',
                'negative': 'Negative'
            },
            'volatility': {
                'high': 'High',
                'moderate': 'Moderate',
                'low': 'Low'
            },
            'market_analysis': {
                'sentiment': 'Market Sentiment',
                'volatility': 'Volatility',
                'volume': 'Trading Volume',
                'dominance': 'Market Dominance'
            },
            'price_history': {
                'ath': 'All-Time High',
                'ath_date': 'ATH Date',
                'ath_change': 'Change from ATH',
                'atl': 'All-Time Low',
                'atl_date': 'ATL Date',
                'atl_change': 'Change from ATL'
            },
            'social_info': {
                'twitter': 'Twitter Followers',
                'reddit': 'Reddit Subscribers',
                'github': 'GitHub Statistics',
                'forks': 'Forks',
                'stars': 'Stars',
                'subscribers': 'Subscribers',
                'total_issues': 'Total Issues',
                'closed_issues': 'Closed Issues',
                'pull_requests': 'Merged Pull Requests',
                'commits': 'Commits (4 weeks)'
            }
        }
    }
    return translations[lang]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'bitcoin')
        lang = data.get('lang', 'fr')
        
        # Get translations
        translations = get_translations(lang)
        
        # Get detailed coin data
        coins = data_collector.get_all_coins()
        coin_data = next((coin for coin in coins if coin['id'] == symbol), None)
        
        if not coin_data:
            return jsonify({
                'status': 'error',
                'message': f"Unable to find data for {symbol}" if lang == 'en' else f"Impossible de trouver les données pour {symbol}"
            })
        
        # Get additional details
        details = data_collector.get_coin_details(symbol)
        
        # Get historical data for charts
        chart_data = get_historical_data(symbol)
        
        # Calculate basic technical indicators
        price_change_24h = coin_data.get('price_change_percentage_24h', 0)
        market_sentiment = translations['market_sentiment']['positive'] if price_change_24h > 0 else translations['market_sentiment']['negative']
        
        # Determine volatility level
        volatility_level = translations['volatility']['high'] if abs(price_change_24h) > 5 else \
                          translations['volatility']['moderate'] if abs(price_change_24h) > 2 else \
                          translations['volatility']['low']
        
        # Prepare price changes for prediction
        price_changes = {
            '1h': coin_data.get('price_change_percentage_1h_in_currency', 0),
            '24h': coin_data.get('price_change_percentage_24h', 0),
            '7d': coin_data.get('price_change_percentage_7d_in_currency', 0),
            '30d': coin_data.get('price_change_percentage_30d_in_currency', 0)
        }
        
        # Prepare detailed response
        analysis_result = {
            'status': 'success',
            'coin_data': {
                'basic_info': {
                    'name': coin_data['name'],
                    'symbol': coin_data['symbol'].upper(),
                    'current_price': coin_data['current_price'],
                    'market_cap': coin_data['market_cap'],
                    'market_cap_rank': coin_data.get('market_cap_rank', 'N/A'),
                    'total_volume': coin_data.get('total_volume', 0),
                    'circulating_supply': coin_data.get('circulating_supply', 0),
                    'max_supply': coin_data.get('max_supply', 0),
                },
                'price_changes': price_changes,
                'market_data': {
                    'ath': coin_data.get('ath', 0),
                    'ath_change_percentage': coin_data.get('ath_change_percentage', 0),
                    'ath_date': coin_data.get('ath_date', ''),
                    'atl': coin_data.get('atl', 0),
                    'atl_change_percentage': coin_data.get('atl_change_percentage', 0),
                    'atl_date': coin_data.get('atl_date', ''),
                },
                'analysis': {
                    'market_sentiment': market_sentiment,
                    'volatility': volatility_level,
                    'trading_volume': coin_data.get('total_volume', 0) / coin_data.get('market_cap', 1),
                    'market_dominance': (coin_data.get('market_cap', 0) / data_collector.get_global_market_cap()) * 100 if data_collector.get_global_market_cap() else 0
                },
                'translations': translations
            }
        }
        
        # Add prediction explanation
        analysis_result['coin_data']['prediction'] = generate_prediction_explanation(
            analysis_result['coin_data'],
            price_changes,
            lang
        )
        
        # Add chart data
        analysis_result['coin_data']['chart_data'] = chart_data
        
        # Add additional details if available
        if details:
            analysis_result['coin_data'].update({
                'additional_info': {
                    'description': details.get('description', {}).get('fr', 'Non disponible'),
                    'homepage': details.get('links', {}).get('homepage', [''])[0],
                    'github': details.get('links', {}).get('repos_url', {}).get('github', []),
                    'reddit': details.get('links', {}).get('subreddit_url', ''),
                    'twitter_followers': details.get('community_data', {}).get('twitter_followers', 0),
                    'reddit_subscribers': details.get('community_data', {}).get('reddit_subscribers', 0),
                    'developer_data': {
                        'forks': details.get('developer_data', {}).get('forks', 0),
                        'stars': details.get('developer_data', {}).get('stars', 0),
                        'subscribers': details.get('developer_data', {}).get('subscribers', 0),
                        'total_issues': details.get('developer_data', {}).get('total_issues', 0),
                        'closed_issues': details.get('developer_data', {}).get('closed_issues', 0),
                        'pull_requests_merged': details.get('developer_data', {}).get('pull_requests_merged', 0),
                        'commit_count_4_weeks': details.get('developer_data', {}).get('commit_count_4_weeks', 0),
                    }
                }
            })
        
        return jsonify(analysis_result)

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': f"Une erreur s'est produite: {str(e)}"
        })

@app.route('/supported_coins', methods=['GET'])
def get_supported_coins():
    try:
        coins = data_collector.get_all_coins()
        coin_list = [{
            'id': coin['id'],
            'symbol': coin['symbol'].upper(),
            'name': coin['name'],
            'market_cap_rank': coin.get('market_cap_rank', 999999)
        } for coin in coins]
        
        # Sort by market cap rank
        coin_list.sort(key=lambda x: x['market_cap_rank'])
        
        return jsonify({
            'status': 'success',
            'coins': coin_list
        })
    except Exception as e:
        logger.error(f"Error getting supported coins: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Erreur lors de la récupération des cryptomonnaies: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=3000)
