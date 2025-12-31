
toxicity-multiclass/
├── data/
│   ├── raw/                     # Raw Tox21 data (original files)
│   ├── processed/               # Cleaned / split datasets
│   └── features/                # Precomputed features (Morgan fingerprints)
│
├── notebooks/
│   ├── 01_exploring_cleaning data.ipynb
│   ├── 02_xgboost_baseline.ipynb
│   ├── 05_chemberta_training.ipynb
│   └── 06_evaluation_base_against_transformers.ipynb
│   
├── src/
│   └── toxicity/
│       ├── data/                # Downloading, cleaning, featurization code
│       ├── models/              # XGBoost, MLP, ChemBERTa implementations
│       └── training/            # Training loops 
│
├── results/                     # Metrics, scores, plots
├── models/                      # Saved trained models
├── scripts/                     # Small helper / launch scripts
│
├── app.py                       # Flask API for deployment
└── README.md


## Tox21 Multi-Label Toxicity Prediction

Ce projet implémente un pipeline complet de machine learning visant à prédire la toxicité moléculaire sur 12 essais biologiques à partir de représentations SMILES.
Des modèles classiques de machine learning (XGBoost, MLP) sont comparés à un Transformer spécialisé dans le domaine chimique (ChemBERTa) afin d’évaluer leur efficacité pour la toxicologie assistée par ordinateur dans un contexte réaliste de données limitées.


## Project Motivation

La découverte de médicaments est marquée par des taux d’échec élevés, souvent dus à des problèmes de toxicité détectés tardivement. Les tests expérimentaux in vivo sont coûteux, longs et soulèvent des enjeux éthiques, ce qui rend le criblage computationnel précoce de plus en plus essentiel.

Ce projet vise à :
• Accélérer l’évaluation de la sécurité via la prédiction de toxicité in silico
• Réduire les coûts expérimentaux en filtrant précocement les composés toxiques
• Comparer des modèles Transformer modernes à des baselines classiques robustes sur des données chimiques


## Dataset

Le projet utilise le jeu de données Tox21 (Toxicology in the 21st Century).
• Environ 7 000 molécules après nettoyage
• 12 cibles de toxicité par composé (par exemple Nuclear Receptors tels que NR-AR et voies Stress Response telles que SR-p53)
• Problème multi-label avec labels manquants et fort déséquilibre de classes

Ces caractéristiques reflètent des contraintes réelles en toxicologie et nécessitent des choix méthodologiques rigoureux en matière de modélisation et d’évaluation.


## Models

 XGBoost (Morgan Fingerprints)

• Les molécules sont représentées par des empreintes Morgan de 2048 bits
• Un classifieur est entraîné par essai biologique (binary relevance)
• Offre des performances solides et stables sur des caractéristiques chimiques tabulaires

 ChemBERTa (Transformer)

• Pré-entraîné sur de larges corpus de SMILES
• Fine-tuné pour la prédiction multi-label de toxicité
• Utilise une masked loss afin que les labels manquants n’influencent pas l’optimisation
• Particulièrement performant pour le classement des composés selon leur risque toxique

 MLP (Multi-Layer Perceptron)

• Réseau de neurones entraîné sur les mêmes empreintes moléculaires que XGBoost
• Sert de point de référence intermédiaire entre les modèles à base d’arbres et les Transformers


## Training Objective

Les modèles sont entraînés à l’aide d’une binary cross-entropy loss dans un cadre multi-label.
Pour ChemBERTa, un masque de labels est appliqué afin d’ignorer les essais manquants lors du calcul de la loss, avec une pondération optionnelle de la classe positive pour atténuer le déséquilibre.

Cette formulation est standard pour la prédiction multi-label en toxicologie et est cohérente avec la structure du dataset Tox21.


## Evaluation

L’évaluation se concentre principalement sur le macro F1 score, avec la précision, le rappel et le ROC-AUC reportés comme métriques complémentaires.
• Le macro F1 reflète une qualité de décision équilibrée entre tous les essais et pénalise les prédictions trivialement négatives
• Le ROC-AUC mesure la capacité de classement indépendamment du seuil de décision

Un seuil de décision global unique est utilisé, avec une valeur conservatrice fixée à 0.7 pour le déploiement.
Toutes les statistiques et comparaisons reportées sont calculées dans le notebook 06 (06_evaluation_base_against_transformers.ipynb).


## Performance Summary

XGBoost obtient les meilleures performances au niveau décisionnel (macro F1), tandis que ChemBERTa présente une meilleure capacité de classement (ROC-AUC).
Ces résultats illustrent le compromis classique entre performance de classement et performance de classification dans des contextes multi-label fortement déséquilibrés.


## Installation and Usage

pip install rdkit xgboost transformers torch pandas numpy scikit-learn matplotlib


## Pipeline Execution

1. **Data cleaning**
   Exécuter 01_exploring_cleaning data.ipynb (ou prepare_data.py) afin de nettoyer et standardiser les SMILES bruts.

2. **Featurization**
   Lancer make_fingerprints.py pour générer les empreintes Morgan de 2048 bits.

3. **Model training**
   o XGBoost : 02_xgboost_baseline.ipynb
   o MLP : 03_mlp_baseline.ipynb
   o ChemBERTa : 05_chemberta_training.ipynb

4. **Evaluation and comparison**
   Reproduire et analyser l’ensemble des métriques rapportées à l’aide de
   06_evaluation_base_against_transformers.ipynb.
