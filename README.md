**Projet : Prédiction de salaire (Machine Learning)**

Ce dépôt contient un script d'entraînement et d'évaluation de modèles de régression pour prédire la variable `Salary` à partir d'un fichier CSV (`Salary_Data.csv`). Le code utilise `scikit-learn` pour le prétraitement, l'entraînement, la recherche d'hyperparamètres et l'évaluation.

**1. Objectifs**
- **But**: comparer deux approches de régression (Régression Linéaire et Arbre de Décision) pour prédire les salaires.
- **Objectifs pédagogiques/pratiques**: démontrer une chaîne complète ML — chargement des données, preprocessing, entraînement, validation croisée, recherche d'hyperparamètres, évaluation, visualisation et sauvegarde des modèles.

**2. Comment lancer le projet**
- **Pré-requis**: Python 3.8+ et `pip`.
- **Installer dans un environnement virtuel (PowerShell)**:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- **Lancer le script**:
```
python main.py
```


**3. Sections (`main.py`)**
Le script est organisé en 10 blocs numérotés (0 à 9) — ci‑dessous le rôle de chaque bloc :
- **0. Chargement du fichier**: lecture du CSV (`pd.read_csv`) et gestion basique d'erreur si le fichier est introuvable. 
- **1. Cible & colonnes**: vérification que la colonne cible `Salary` existe ; séparation `X` / `y` ; détection automatique des colonnes numériques et catégorielles.
- **2. Train/Test split**: séparation des données en ensembles d'entraînement et de test.
- **3. Preprocessing pipeline**: construction d'un `ColumnTransformer` qui applique aux colonnes numériques un `SimpleImputer(strategy='median')` puis `StandardScaler`, et aux colonnes catégorielles un `SimpleImputer(strategy='most_frequent')` puis `OneHotEncoder(handle_unknown='ignore')`.
- **4. Pipelines modèles**: deux `Pipeline` complets (préproc + modèle) : `pipe_lr` pour `LinearRegression` et `pipe_dt` pour `DecisionTreeRegressor`.
- **5. Entraînement & CV (pour DT)**: évaluation cross‑validée (`cross_val_score`) pour la régression linéaire ; `GridSearchCV` sur l'arbre pour optimiser `max_depth` et `min_samples_leaf`.
- **6. Évaluation (train + test)**: calcul des métriques (MSE, RMSE, MAE, R2) sur test et train pour détecter overfitting / surapprentissage.
- **7. Graphiques**: `y_test` vs `y_pred` pour les deux modèles avec la droite identité (réel = prédit) pour visualiser la qualité des prédictions.
- **8. Interprétabilité**: extraction des noms de features après préprocessing ; affichage des coefficients pour la LR et des importances pour le DT.
- **9. Sauvegarde des modèles**: `joblib.dump` pour écrire `model_linear.joblib` et `model_dt_best.joblib` sur disque.

**4. Métriques utilisées — explication et importance**
- **MSE (Mean Squared Error)**: moyenne des carrés des erreurs ( (y_true - y_pred)^2 ). Pénalise fortement les grandes erreurs ; unité au carré de la variable cible.
- **RMSE (Root Mean Squared Error)**: racine carrée du MSE. Même sensibilité que MSE mais en unité comparable à la variable cible.
- **MAE (Mean Absolute Error)**: moyenne des valeurs absolues des erreurs. Moins sensible que MSE/RMSE.
- **R2 (Coefficient de détermination)**: proportion de la variance expliquée par le modèle (1 = parfait, 0 = modèle qui prédit la moyenne). Utile pour comparer la capacité explicative d'un modèle, mais sensible aux caractéristiques des données.

Pourquoi ces métriques :
- **Comparaison complète**: MSE/RMSE pour pénaliser grosses erreurs, MAE pour robustesse aux outliers, R2 pour une métrique normalisée (0..1 souvent) qui donne une idée globale de la qualité.
- **Choix selon besoin métier**: si grosses erreurs sont critiques (par ex. erreurs de salaire élevées) privilégier RMSE/MSE ; si on veut une vision plus robuste, regarder MAE.

**5. Graphes proposés (LR et DT) — explication et importance**
- **Scatter plot Réel vs Prédit**: chaque point est (y_true, y_pred). La ligne identité (`y_true = y_pred`) sert de référence :
  - points proches de la ligne indiquent de bonnes prédictions ;
  - écart vertical montre l'erreur ;
  - dispersion systématique (biais) révèle sous/surestimation selon régions du target.
- **Pourquoi utile**: permet de détecter patterns d'erreur (hétéroscédasticité, biais systématique, outliers mal prédits) qu'une seule métrique ne montre pas.