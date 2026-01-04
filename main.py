import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump


# 0. Chargement du fichier (dataset Kaggle lan)

try:
    df = pd.read_csv('Salary_Data.csv')
except FileNotFoundError:
    raise FileNotFoundError("Fichier non trouvé.")

# print("Aperçu des colonnes et types :")
# print(df.dtypes)
# print("\nPremières lignes :")
# print(df.head())



# 1. Cible & colonnes

target = 'Salary'
if target not in df.columns:
    raise ValueError(f"La colonne cible '{target}' n'existe pas dans le dataset. Colonnes disponibles : {df.columns.tolist()}")

df = df.dropna(subset=[target])  
# Séparer le target des features
X = df.drop(columns=[target])
y = df[target]
print(df.shape)
print(df.tail(10))
print(df.isnull().sum())

# Détecter automatiquement les colonnes numériques et catégorielles
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print(f"\nFeatures numériques ({len(numeric_features)}): {numeric_features}")
# print(f"Features catégorielles ({len(categorical_features)}): {categorical_features}")


# 2. Train/Test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. Preprocessing pipeline

# Numeric: imputer (median) + scaler
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: imputer (most_frequent) + one-hot (ignore unknown)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
], remainder='drop')  


# 4. Pipelines modèles

# Pipeline régression linéaire
pipe_lr = Pipeline(steps=[
    ('preproc', preprocessor),
    ('model', LinearRegression())
])

# Pipeline arbre de décision (pr GridSearch)
pipe_dt = Pipeline(steps=[
    ('preproc', preprocessor),
    ('model', DecisionTreeRegressor(random_state=42))
])


# 5. Entraînement & CV (pour DT)

# Linear Regression - cross-validated score (MSE)
cv_scores_lr = cross_val_score(pipe_lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
# print(f"\nCV MSE (LinearRegression) mean: {-np.mean(cv_scores_lr):.3f} (std: {np.std(cv_scores_lr):.3f})")

# GridSearch pour Decision Tree (trouver max_depth et min_samples_leaf)
param_grid = {
    'model__max_depth': [3, 5, 7, 9],
    'model__min_samples_leaf': [5, 10, 20]
}
grid_dt = GridSearchCV(pipe_dt, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_dt.fit(X_train, y_train)
print("\nMeilleurs paramètres DecisionTree trouvés :", grid_dt.best_params_)
print("Best CV MSE (DT):", -grid_dt.best_score_)

# On entraîne la LR sur tout le train set et on récupère les prédictions
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)

# Pour DT on récupère le meilleur modèle
best_dt = grid_dt.best_estimator_
y_pred_dt = best_dt.predict(X_test)


# 6. Évaluation (train + test)

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

print("\n=== Résultats sur le jeu de test ===")
print("Linear Regression:", regression_metrics(y_test, y_pred_lr))
print("Decision Tree (best):", regression_metrics(y_test, y_pred_dt))

# Évaluer aussi sur le train pour détecter l'overfitting
y_train_pred_lr = pipe_lr.predict(X_train)
y_train_pred_dt = best_dt.predict(X_train)
print("\n=== Résultats sur le jeu d'entraînement ===")
print("Linear Regression (train):", regression_metrics(y_train, y_train_pred_lr))
print("Decision Tree (train):", regression_metrics(y_train, y_train_pred_dt))


# 7. Graphiques pour visualiser nos modèles

plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred_lr, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions (LR)")
plt.title("LR : Réel vs Prédit")
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred_dt, alpha=0.7, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions (DT)")
plt.title("DT : Réel vs Prédit")
plt.show()

# Résidus (exemple pour LR)
residuals = y_test - y_pred_lr
plt.figure(figsize=(6,4))
plt.scatter(y_pred_lr, residuals, alpha=0.7, color='purple')
plt.hlines(0, xmin=y_pred_lr.min(), xmax=y_pred_lr.max(), linestyles='dashed')
plt.xlabel("Prédictions (LR)")
plt.ylabel("Résidus (y_true - y_pred)")
plt.title("LR : résidus vs prédictions")
plt.show()

# Résidus (exemple pour DT)
residuals = y_test - y_pred_dt
plt.figure(figsize=(6,4))
plt.scatter(y_pred_dt, residuals, alpha=0.7, color='orange')
plt.hlines(0, xmin=y_pred_dt.min(), xmax=y_pred_dt.max(), linestyles='dashed')
plt.xlabel("Prédictions (DT)")
plt.ylabel("Résidus (y_true - y_pred)")
plt.title("DT : résidus vs prédictions")
plt.show()


# 8. Interprétabilité

try:
    # construire noms de features après préprocessing
    cat_ohe = pipe_lr.named_steps['preproc'].named_transformers_['cat']['ohe']
    cat_names = []
    if hasattr(cat_ohe, 'get_feature_names_out'):
        cat_names = list(cat_ohe.get_feature_names_out(categorical_features))
    feature_names = numeric_features + cat_names
    coefs = pipe_lr.named_steps['model'].coef_
    coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
    coef_df = coef_df.sort_values(key=lambda s: s.abs(), ascending=False, by='coef')
    print("\nCoefficients LR (top 20) :")
    print(coef_df.head(20))
except Exception as e:
    print("Impossible d'extraire les coefficients.", e)

# Feature importances (DT)
try:
    dt_model = best_dt.named_steps['model']
    importances = dt_model.feature_importances_
    feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    print("\nImportances (DT) top 20 :")
    print(feat_df.head(20))
except Exception as e:
    print("Impossible d'extraire importances DT :", e)

# 9. Sauvegarde des modèles

dump(pipe_lr, 'model_linear.joblib') 
dump(best_dt, 'model_dt_best.joblib')