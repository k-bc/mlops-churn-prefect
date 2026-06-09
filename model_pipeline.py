"""
model_pipeline.py
=================
Pipeline ML modularisé pour la prédiction du churn client (Churn_Modelling.csv).

Atelier 2 : Modularisation du Code — MLOps
Fonctions : prepare_data, train_model, evaluate_model, save_model, load_model
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def prepare_data(data_path="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """
    Charge et prétraite les données du churn client.

    Étapes (identiques au notebook) :
      1. Chargement du CSV.
      2. Encodage de la variable 'Gender' avec LabelEncoder.
      3. Suppression des colonnes non pertinentes :
         'Surname', 'Geography', 'RowNumber', 'CustomerId'.
      4. Séparation features (x) / cible (y = 'Exited').
      5. Split train/test (80/20).
      6. Standardisation des features avec StandardScaler
         (fit sur train uniquement, transform sur test).

    Args:
        data_path (str): Chemin vers le fichier CSV.
        test_size (float): Proportion du jeu de test (défaut 0.2).
        random_state (int): Graine aléatoire pour la reproductibilité.

    Returns:
        tuple: (x_train, x_test, y_train, y_test, scaler)
    """
    # 1. Chargement
    df = pd.read_csv(data_path)
    print(
        f"[prepare_data] Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes"
    )

    # Vérifications rapides (bonnes pratiques)
    print(f"[prepare_data] Valeurs manquantes : {df.isna().sum().sum()}")
    print(f"[prepare_data] Doublons : {df.duplicated().any()}")

    # 2. Encodage de Gender
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # 3. Suppression des colonnes inutiles
    df = df.drop(["Surname", "Geography"], axis=1)

    # 4. Features / cible
    x = df.drop(["Exited"], axis=1)
    y = df["Exited"]
    x = x.drop(columns=["RowNumber", "CustomerId"])

    # 5. Split train/test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # 6. Standardisation (fit sur train uniquement — évite la fuite de données)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print(f"[prepare_data] Train : {x_train.shape}, Test : {x_test.shape}")
    return x_train, x_test, y_train, y_test, scaler


def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """
    Entraîne un RandomForestClassifier sur les données d'entraînement.

    Args:
        x_train: Features d'entraînement.
        y_train: Cible d'entraînement.
        n_estimators (int): Nombre d'arbres (défaut 100).
        random_state (int): Graine aléatoire (défaut 42).

    Returns:
        RandomForestClassifier: Modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    print("[train_model] Modèle RandomForest entraîné.")
    return model


def evaluate_model(model, x_test, y_test):
    """
    Évalue les performances du modèle sur le jeu de test.

    Affiche : accuracy, classification report et matrice de confusion.

    Args:
        model: Modèle entraîné.
        x_test: Features de test.
        y_test: Cible de test.

    Returns:
        float: Score d'accuracy.
    """
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"[evaluate_model] Accuracy score : {accuracy * 100:.2f} %")

    print("\n[evaluate_model] Classification report :")
    print(classification_report(y_test, y_pred))

    print("[evaluate_model] Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    return accuracy


def save_model(model, filename="classifier.joblib"):
    """
    Sauvegarde le modèle entraîné avec joblib.

    Args:
        model: Modèle entraîné à sauvegarder.
        filename (str): Nom du fichier de sortie (défaut 'classifier.joblib').
    """
    joblib.dump(model, filename)
    print(f"[save_model] Modèle sauvegardé dans '{filename}'.")


def load_model(filename="classifier.joblib"):
    """
    Charge un modèle sauvegardé avec joblib.

    Args:
        filename (str): Chemin du fichier modèle.

    Returns:
        Modèle chargé.
    """
    model = joblib.load(filename)
    print(f"[load_model] Modèle chargé depuis '{filename}'.")
    return model
