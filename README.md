# Bot de Prédiction de Cryptomonnaies By ELHEROUAL

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-Copyright%20ELHEROUAL-red.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)

## Description
Ce bot analyse les cryptomonnaies en utilisant une combinaison d'analyse technique, d'analyse des sentiments et d'apprentissage automatique pour fournir des recommandations d'investissement détaillées.

## Fonctionnalités
- 📊 Analyse technique complète avec indicateurs multiples
- 🌐 Analyse des sentiments via Twitter et Reddit
- 🤖 Prédictions de prix basées sur l'apprentissage automatique
- 🎯 Interface utilisateur intuitive en français
- 💡 Recommandations d'investissement détaillées

## Installation
1. Clonez le dépôt :
```bash
git clone [url-du-repo]
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurez les variables d'environnement dans un fichier `.env` :
```
TWITTER_API_KEY=votre_clé
TWITTER_API_SECRET=votre_secret
TWITTER_ACCESS_TOKEN=votre_token
TWITTER_ACCESS_TOKEN_SECRET=votre_token_secret
REDDIT_CLIENT_ID=votre_client_id
REDDIT_CLIENT_SECRET=votre_client_secret
REDDIT_USER_AGENT=votre_user_agent
```

## Utilisation
1. Lancez l'application :
```bash
python src/app.py
```

2. Ouvrez votre navigateur et accédez à :
```
http://localhost:5000
```

## Structure du Projet
```
crypto_prediction_bot/
├── src/
│   ├── app.py                 # Application principale Flask
│   ├── analysis/              # Modules d'analyse
│   │   ├── technical_analyzer.py
│   │   └── sentiment_analyzer.py
│   ├── data/                  # Collecte de données
│   │   └── collectors/
│   │       └── crypto_data_collector.py
│   ├── models/               # Modèles de prédiction
│   │   └── predictor.py
│   └── templates/           # Templates HTML
│       └── index.html
├── requirements.txt         # Dépendances Python
└── README.md               # Documentation
```

## Technologies Utilisées
- 🐍 Python 3.8+
- 🌐 Flask
- 📊 Pandas
- 🔢 NumPy
- 🤖 Scikit-learn
- 🐦 TweetPy
- 📱 PRAW
- 📈 TA-Lib

## Contribution
Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Auteur
**ELHEROUAL**
- 📧 Email: [votre-email]
- 🌐 GitHub: [votre-profil-github]

## Licence
Copyright (c) 2024 ELHEROUAL. Tous droits réservés.

---
*Fait avec ❤️ par ELHEROUAL*
