"""
main.py
=======
Fichier principal : exécute les étapes du pipeline ML via des arguments CLI.

Exemples d'utilisation :
    python main.py --prepare                 # Préparation des données uniquement
    python main.py --train                   # Préparation + entraînement + sauvegarde
    python main.py --evaluate                # Charge le modèle sauvegardé et l'évalue
    python main.py --all                     # Pipeline complet
    python main.py --train --data autre.csv  # Spécifier un autre fichier de données
"""

import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ML — Prédiction du churn client (Atelier 2 MLOps)"
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Préparer les données (chargement + prétraitement)",
    )
    parser.add_argument(
        "--train", action="store_true", help="Entraîner le modèle et le sauvegarder"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Évaluer le modèle sauvegardé sur le jeu de test",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Exécuter le pipeline complet (prepare + train + evaluate + save)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="Churn_Modelling.csv",
        help="Chemin du fichier CSV (défaut : Churn_Modelling.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="classifier.joblib",
        help="Chemin du fichier modèle (défaut : classifier.joblib)",
    )

    args = parser.parse_args()

    # Si aucun argument n'est fourni, afficher l'aide
    if not (args.prepare or args.train or args.evaluate or args.all):
        parser.print_help()
        return

    # --- Étape 1 : Préparation des données (nécessaire à toutes les étapes) ---
    x_train, x_test, y_train, y_test, scaler = prepare_data(data_path=args.data)

    if args.prepare and not (args.train or args.evaluate or args.all):
        print("\nPréparation des données terminée.")
        return

    # --- Étape 2 : Entraînement ---
    if args.train or args.all:
        model = train_model(x_train, y_train)
        save_model(model, filename=args.model)

    # --- Étape 3 : Évaluation ---
    if args.evaluate or args.all:
        model = load_model(filename=args.model)
        evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()
