"""
Bot de Prédiction de Cryptomonnaies By ELHEROUAL
================================================
Module: Générateur d'Explications
Copyright (c) 2024
Version: 1.0.0

Module qui génère des explications détaillées pour les recommandations d'investissement.
"""

from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ExplanationGenerator:
    def __init__(self):
        self.risk_levels = {
            'très_faible': 'très prudente',
            'faible': 'prudente',
            'modéré': 'équilibrée',
            'élevé': 'agressive',
            'très_élevé': 'très agressive'
        }

    def generate_detailed_explanation(self, 
                                   technical_analysis: Dict,
                                   sentiment_analysis: Dict,
                                   price_predictions: Dict,
                                   correlation_data: Dict,
                                   historical_data: pd.DataFrame) -> Dict:
        """Génère une explication détaillée de la recommandation."""
        
        # Analyser les tendances
        trend_analysis = self._analyze_trends(technical_analysis, historical_data)
        
        # Analyser le sentiment
        sentiment_insight = self._analyze_sentiment(sentiment_analysis)
        
        # Analyser les prédictions
        prediction_insight = self._analyze_predictions(price_predictions)
        
        # Analyser les corrélations
        correlation_insight = self._analyze_correlations(correlation_data)
        
        # Déterminer la recommandation finale
        recommendation = self._determine_recommendation(
            trend_analysis,
            sentiment_insight,
            prediction_insight,
            correlation_insight
        )
        
        # Générer l'explication complète
        explanation = self._format_explanation(
            recommendation,
            trend_analysis,
            sentiment_insight,
            prediction_insight,
            correlation_insight
        )
        
        return explanation

    def _analyze_trends(self, technical_analysis: Dict, 
                       historical_data: pd.DataFrame) -> Dict:
        """Analyse les tendances techniques."""
        recent_price_change = (
            (historical_data['close'].iloc[-1] / 
             historical_data['close'].iloc[-7] - 1) * 100
        )
        
        volatility = historical_data['close'].pct_change().std() * 100
        
        return {
            'direction': technical_analysis.get('trend', {}).get('direction', 'neutre'),
            'force': technical_analysis.get('trend', {}).get('strength', 0),
            'recent_change': recent_price_change,
            'volatility': volatility,
            'rsi': technical_analysis.get('momentum', {}).get('rsi', 50),
            'trend_description': self._get_trend_description(
                technical_analysis.get('trend', {}).get('direction', 'neutre'),
                technical_analysis.get('trend', {}).get('strength', 0)
            )
        }

    def _analyze_sentiment(self, sentiment_analysis: Dict) -> Dict:
        """Analyse le sentiment global."""
        return {
            'score': sentiment_analysis.get('score', 0),
            'confidence': sentiment_analysis.get('confidence', 0),
            'sources': sentiment_analysis.get('sources', {}),
            'sentiment_description': self._get_sentiment_description(
                sentiment_analysis.get('score', 0),
                sentiment_analysis.get('confidence', 0)
            )
        }

    def _analyze_predictions(self, price_predictions: Dict) -> Dict:
        """Analyse les prédictions de prix."""
        return {
            'direction': 'hausse' if price_predictions.get('prediction', 0) > 0 else 'baisse',
            'magnitude': abs(price_predictions.get('prediction', 0)),
            'confidence': price_predictions.get('confidence', 0),
            'prediction_description': self._get_prediction_description(
                price_predictions.get('prediction', 0),
                price_predictions.get('confidence', 0)
            )
        }

    def _analyze_correlations(self, correlation_data: Dict) -> Dict:
        """Analyse les corrélations avec d'autres actifs."""
        return {
            'strongest_correlation': max(
                correlation_data.get('correlations', {}).items(),
                key=lambda x: abs(x[1])
            ) if correlation_data.get('correlations') else ('none', 0),
            'correlation_description': self._get_correlation_description(
                correlation_data.get('correlations', {})
            )
        }

    def _determine_recommendation(self, trend_analysis: Dict,
                                sentiment_insight: Dict,
                                prediction_insight: Dict,
                                correlation_insight: Dict) -> str:
        """Détermine la recommandation finale."""
        signals = {
            'technique': 1 if trend_analysis['direction'] == 'haussier' else -1,
            'sentiment': 1 if sentiment_insight['score'] > 0 else -1,
            'prediction': 1 if prediction_insight['direction'] == 'hausse' else -1
        }
        
        signal_strength = sum(signals.values())
        
        if signal_strength >= 2:
            return 'achat_fort'
        elif signal_strength == 1:
            return 'achat'
        elif signal_strength == -1:
            return 'vente'
        elif signal_strength <= -2:
            return 'vente_forte'
        else:
            return 'conserver'

    def _format_explanation(self, recommendation: str,
                          trend_analysis: Dict,
                          sentiment_insight: Dict,
                          prediction_insight: Dict,
                          correlation_insight: Dict) -> Dict:
        """Formate l'explication complète."""
        
        # Construire le paragraphe principal
        main_explanation = f"""
Basé sur notre analyse approfondie, nous recommandons une position {self._get_recommendation_text(recommendation)} 
pour les raisons suivantes :

1. Analyse Technique :
{trend_analysis['trend_description']}
La volatilité est {self._format_volatility(trend_analysis['volatility'])} et 
le RSI est à {trend_analysis['rsi']:.1f}, ce qui indique {self._interpret_rsi(trend_analysis['rsi'])}.

2. Analyse des Sentiments :
{sentiment_insight['sentiment_description']}
Cette analyse est basée sur plusieurs sources, notamment les réseaux sociaux et les actualités.

3. Prédictions de Prix :
{prediction_insight['prediction_description']}
Cette prédiction est établie en combinant plusieurs modèles d'apprentissage automatique.

4. Corrélations de Marché :
{correlation_insight['correlation_description']}
"""

        # Ajouter des points d'attention spécifiques
        risk_factors = self._identify_risk_factors(
            trend_analysis,
            sentiment_insight,
            prediction_insight
        )

        # Construire la conclusion
        conclusion = self._build_conclusion(
            recommendation,
            trend_analysis['volatility'],
            sentiment_insight['confidence']
        )

        return {
            'recommendation': recommendation,
            'main_explanation': main_explanation.strip(),
            'risk_factors': risk_factors,
            'conclusion': conclusion,
            'confidence_score': self._calculate_confidence_score(
                trend_analysis,
                sentiment_insight,
                prediction_insight
            )
        }

    def _get_trend_description(self, direction: str, strength: float) -> str:
        """Génère une description de la tendance."""
        if direction == 'haussier':
            strength_desc = 'forte' if strength > 0.7 else 'modérée'
            return f"Le marché montre une tendance haussière {strength_desc} avec une force de {strength:.1%}"
        elif direction == 'baissier':
            strength_desc = 'forte' if strength > 0.7 else 'modérée'
            return f"Le marché est en tendance baissière {strength_desc} avec une force de {strength:.1%}"
        else:
            return "Le marché montre une tendance neutre sans direction claire"

    def _get_sentiment_description(self, score: float, confidence: float) -> str:
        """Génère une description du sentiment."""
        if score > 0.5:
            return f"Le sentiment est très positif (score: {score:.2f}) avec une confiance de {confidence:.1%}"
        elif score > 0:
            return f"Le sentiment est légèrement positif (score: {score:.2f}) avec une confiance de {confidence:.1%}"
        elif score < -0.5:
            return f"Le sentiment est très négatif (score: {score:.2f}) avec une confiance de {confidence:.1%}"
        elif score < 0:
            return f"Le sentiment est légèrement négatif (score: {score:.2f}) avec une confiance de {confidence:.1%}"
        else:
            return "Le sentiment est neutre"

    def _get_prediction_description(self, prediction: float, confidence: float) -> str:
        """Génère une description de la prédiction."""
        direction = "hausse" if prediction > 0 else "baisse"
        magnitude = abs(prediction)
        
        if magnitude > 0.1:
            strength = "forte"
        elif magnitude > 0.05:
            strength = "modérée"
        else:
            strength = "légère"
            
        return (f"Les modèles prédisent une {strength} {direction} "
                f"de {magnitude:.1%} avec une confiance de {confidence:.1%}")

    def _get_correlation_description(self, correlations: Dict) -> str:
        """Génère une description des corrélations."""
        if not correlations:
            return "Aucune corrélation significative n'a été identifiée"
            
        strongest = max(correlations.items(), key=lambda x: abs(x[1]))
        return (f"Une corrélation {self._format_correlation(strongest[1])} "
                f"a été identifiée avec {strongest[0]}")

    def _format_correlation(self, correlation: float) -> str:
        """Formate la description d'une corrélation."""
        abs_corr = abs(correlation)
        if abs_corr > 0.8:
            strength = "très forte"
        elif abs_corr > 0.6:
            strength = "forte"
        elif abs_corr > 0.4:
            strength = "modérée"
        else:
            strength = "faible"
            
        direction = "positive" if correlation > 0 else "négative"
        return f"{strength} {direction}"

    def _format_volatility(self, volatility: float) -> str:
        """Formate la description de la volatilité."""
        if volatility > 0.05:
            return "très élevée"
        elif volatility > 0.03:
            return "élevée"
        elif volatility > 0.02:
            return "modérée"
        else:
            return "faible"

    def _interpret_rsi(self, rsi: float) -> str:
        """Interprète la valeur du RSI."""
        if rsi > 70:
            return "une condition de surachat"
        elif rsi < 30:
            return "une condition de survente"
        else:
            return "des conditions de marché normales"

    def _identify_risk_factors(self, trend_analysis: Dict,
                             sentiment_insight: Dict,
                             prediction_insight: Dict) -> List[str]:
        """Identifie les facteurs de risque importants."""
        risks = []
        
        if trend_analysis['volatility'] > 0.04:
            risks.append("Volatilité élevée - Utiliser des ordres stop-loss")
            
        if abs(sentiment_insight['score']) > 0.8:
            risks.append("Sentiment extrême - Risque de retournement")
            
        if trend_analysis['rsi'] > 70 or trend_analysis['rsi'] < 30:
            risks.append("RSI en zone extrême - Surveiller les retournements")
            
        return risks

    def _build_conclusion(self, recommendation: str,
                         volatility: float,
                         sentiment_confidence: float) -> str:
        """Construit la conclusion de l'analyse."""
        risk_profile = self._determine_risk_profile(volatility)
        confidence_level = "élevée" if sentiment_confidence > 0.7 else "modérée"
        
        return (f"Cette recommandation {recommendation} est adaptée à une "
                f"stratégie d'investissement {self.risk_levels[risk_profile]} "
                f"avec une confiance {confidence_level}.")

    def _determine_risk_profile(self, volatility: float) -> str:
        """Détermine le profil de risque basé sur la volatilité."""
        if volatility > 0.05:
            return "très_élevé"
        elif volatility > 0.04:
            return "élevé"
        elif volatility > 0.03:
            return "modéré"
        elif volatility > 0.02:
            return "faible"
        else:
            return "très_faible"

    def _calculate_confidence_score(self, trend_analysis: Dict,
                                  sentiment_insight: Dict,
                                  prediction_insight: Dict) -> float:
        """Calcule un score de confiance global."""
        weights = {
            'trend': 0.4,
            'sentiment': 0.3,
            'prediction': 0.3
        }
        
        scores = {
            'trend': min(1.0, trend_analysis['force']),
            'sentiment': sentiment_insight['confidence'],
            'prediction': prediction_insight['confidence']
        }
        
        return sum(score * weights[key] for key, score in scores.items())
