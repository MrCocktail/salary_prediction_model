# Projet de Machine Learning — Prédiction des Salaires

## 1. Contexte et objectif

Ce projet a pour objectif de **prédire le salaire d’un individu** à partir de caractéristiques socio‑professionnelles (âge, expérience, niveau d’éducation, poste occupé, genre, etc.) en utilisant des **modèles de régression supervisée**.

Deux modèles ont été étudiés et comparés :

* une **Régression Linéaire**, utilisée comme modèle de base (baseline),
* un **Arbre de Décision (Decision Tree Regressor)**, optimisé par validation croisée.

L’objectif final est de **sélectionner le modèle offrant le meilleur compromis entre précision, généralisation et interprétabilité**.

---

## 2. Jeu de données

* **Taille** : 6 705 observations
* **Variable cible** : `Salary`
* **Variables explicatives** :

  * Âge 
  * Années d’expérience 
  * Genre 
  * Niveau d’éducation 
  * Intitulé du poste 

Les variables catégorielles ont été transformées à l’aide du **One‑Hot Encoding**.

---

## 3. Prétraitement des données

Les étapes de prétraitement incluent :

1. **Nettoyage**

   * Remplacement des lignes contenant des valeurs manquantes via `SimpleImputer`.

2. **Encodage des variables catégorielles**

   * Utilisation de `pandas.get_dummies()` avec `drop_first=True` afin d’éviter la colinéarité.
   * Les coefficients sont donc interprétés **par rapport à une catégorie de référence implicite**.

3. **Séparation des données**

   * 80 % pour l’entraînement
   * 20 % pour le test

4. **Normalisation**

   * Standardisation des variables numériques avec `StandardScaler`.

---

## 4. Modèles utilisés

### 4.1 Régression Linéaire

La régression linéaire sert de **modèle de référence**. Elle suppose une relation linéaire entre les variables explicatives et le salaire.

Avantages :

* Simplicité
* Forte interprétabilité

Limites :

* Incapacité à modéliser des relations non linéaires complexes

---

### 4.2 Arbre de Décision (Decision Tree Regressor)

L’arbre de décision permet de capturer des **relations non linéaires** en découpant l’espace des données en régions homogènes.

Afin d’éviter le sur‑apprentissage, une **validation croisée avec GridSearchCV** a été utilisée pour optimiser les hyperparamètres.

#### Grille d’hyperparamètres testée :

* `max_depth` ∈ {3, 5, 7, 9}
* `min_samples_leaf` ∈ {5, 10, 20}

Meilleure combinaison trouvée :

* `max_depth = 9`
* `min_samples_leaf = 5`

Cette combinaison offre un **bon équilibre entre complexité et généralisation**.

---

## 5. Métriques d’évaluation

Les modèles ont été évalués à l’aide des métriques suivantes :

* **MSE (Mean Squared Error)** : pénalise fortement les grandes erreurs
* **RMSE (Root Mean Squared Error)** : erreur moyenne exprimée dans l’unité du salaire
* **MAE (Mean Absolute Error)** : erreur moyenne absolue
* **R² (Coefficient de détermination)** : proportion de variance expliquée par le modèle

---

## 6. Résultats

### 6.1 Validation croisée

* **Régression Linéaire** :

  * CV MSE ≈ 338 894 089

* **Decision Tree (optimisé)** :

  * Best CV MSE ≈ 178 947 647

➡️ L’arbre de décision est nettement supérieur dès la phase de validation.

---

### 6.2 Performances sur le jeu de test

| Modèle              | RMSE     | MAE      | R²    |
| ------------------- | -------- | -------- | ----- |
| Régression Linéaire | ≈ 18 911 | ≈ 13 266 | 0.875 |
| Decision Tree       | ≈ 12 887 | ≈ 8 327  | 0.942 |

➡️ Le Decision Tree offre une **meilleure précision sur toutes les métriques**.

---

### 6.3 Comparaison entraînement vs test

Les écarts entre les performances d’entraînement et de test sont faibles pour les deux modèles, indiquant **l’absence de sur‑apprentissage significatif**, en particulier pour l’arbre régularisé.

---

## 7. Interprétabilité des modèles

### Régression Linéaire

* Les coefficients indiquent l’impact moyen de chaque variable sur le salaire.
* Les postes de direction (CTO, CDO, Directeur Data) présentent des effets positifs importants.
* Les postes juniors ou administratifs ont des effets négatifs.

### Arbre de Décision

* La variable la plus importante est **l’expérience professionnelle**, représentant plus de 75 % du pouvoir décisionnel.
* Le poste et le niveau d’éducation jouent un rôle secondaire.

Les deux modèles convergent vers une **hiérarchie salariale cohérente avec la réalité économique**.

---

## 8. Visualisation et diagnostic

* Les graphes *réel vs prédit* montrent un alignement plus serré pour l’arbre que pour la régression.
* Les graphes de résidus révèlent une **hétéroscédasticité** pour la régression linéaire, expliquant ses limites sur les hauts salaires.

---

## 9. Sauvegarde et réutilisation du modèle

Le modèle final est sauvegardé au format `joblib`, permettant :

* une réutilisation sans ré‑entraînement,
* un déploiement dans une application,
* une meilleure reproductibilité.

---

## 10. Conclusion

Ce projet montre que :

* la **régression linéaire** constitue une base solide mais limitée,
* l’**arbre de décision optimisé** offre une amélioration significative en précision,
* la validation croisée et la régularisation sont essentielles pour éviter le sur‑apprentissage.

➡️ **Le Decision Tree optimisé est retenu comme modèle final pour la prédiction des salaires.**

---
