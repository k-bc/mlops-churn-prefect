"""
pipeline_prefect.py
===================
Orchestration Prefect du pipeline de prediction du churn client.
Reutilise les fonctions de model_pipeline.py (Atelier 2).
"""

import argparse
import subprocess
from prefect import flow, task

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"
# Lint / securite ciblent UNIQUEMENT nos fichiers (pas venv/)
TARGET_FILES = "model_pipeline.py main.py pipeline_prefect.py"


# ------------------------------------------------------------------
# TASKS : environnement & qualite du code
# ------------------------------------------------------------------
@task(retries=2, retry_delay_seconds=5, log_prints=True)
def install_dependencies():
    """Installe les dependances du projet."""
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    return "dependances installees"


@task(log_prints=True)
def format_code():
    """Formatage automatique du code (black) -> reports/black.txt."""
    result = subprocess.run(
        ["black", *TARGET_FILES.split()],
        capture_output=True,
        text=True,
        check=False,
    )
    with open("reports/black.txt", "w") as f:
        f.write(result.stdout + result.stderr)
    print(result.stdout + result.stderr)


@task(log_prints=True)
def lint_code():
    """Qualite / style du code (flake8) -> reports/flake8.txt."""
    result = subprocess.run(
        ["flake8", *TARGET_FILES.split(), "--max-line-length=100"],
        capture_output=True,
        text=True,
        check=False,
    )
    report = result.stdout + result.stderr
    with open("reports/flake8.txt", "w") as f:
        f.write(report if report else "Aucun probleme de style detecte.\n")
    print(report)


@task(log_prints=True)
def security_check():
    """Analyse de securite statique (bandit) -> reports/bandit.txt."""
    result = subprocess.run(
        ["bandit", *TARGET_FILES.split()],
        capture_output=True,
        text=True,
        check=False,
    )
    with open("reports/bandit.txt", "w") as f:
        f.write(result.stdout + result.stderr)
    print(result.stdout + result.stderr)


@task(log_prints=True)
def run_unit_tests():
    """Execution des tests unitaires (pytest) -> reports/pytest.txt."""
    result = subprocess.run(
        ["pytest", "-q", "test_model_pipeline.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    with open("reports/pytest.txt", "w") as f:
        f.write(result.stdout + result.stderr)
    print(result.stdout + result.stderr)


# ------------------------------------------------------------------
# TASKS : data & modele (appellent les fonctions de model_pipeline)
# ------------------------------------------------------------------
@task(log_prints=True)
def prepare_data_task(data_path: str = DATA_PATH):
    """Prepare les donnees."""
    return prepare_data(data_path=data_path)


@task(log_prints=True)
def train_model_task(x_train, y_train):
    """Entraine le modele."""
    return train_model(x_train, y_train)


@task(log_prints=True)
def save_model_task(model, filename: str = MODEL_PATH):
    """Sauvegarde le modele."""
    save_model(model, filename=filename)
    return filename


@task(log_prints=True)
def load_model_task(filename: str = MODEL_PATH):
    """Charge le modele."""
    return load_model(filename=filename)


@task(log_prints=True)
def evaluate_model_task(model, x_test, y_test):
    """Evalue le modele."""
    return evaluate_model(model, x_test, y_test)


# ------------------------------------------------------------------
# FLOWS
# ------------------------------------------------------------------
@flow(name="code")
def code_flow():
    """Suivi de la qualite du code : format, lint, securite, tests."""
    format_code()
    lint_code()
    security_check()
    run_unit_tests()


@flow(name="train")
def train_flow(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    """Preparation des donnees + entrainement + sauvegarde."""
    x_train, x_test, y_train, y_test, _ = prepare_data_task(data_path)
    model = train_model_task(x_train, y_train)
    save_model_task(model, model_path)
    return model_path


@flow(name="evaluate")
def evaluate_flow(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    """Chargement du modele sauvegarde + evaluation."""
    _, x_test, _, y_test, _ = prepare_data_task(data_path)
    model = load_model_task(model_path)
    return evaluate_model_task(model, x_test, y_test)


@flow(name="all")
def all_flow(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    """Pipeline complet : dependances + qualite + train + evaluate."""
    install_dependencies()
    code_flow()
    x_train, x_test, y_train, y_test, _ = prepare_data_task(data_path)
    model = train_model_task(x_train, y_train)
    save_model_task(model, model_path)
    loaded = load_model_task(model_path)
    return evaluate_model_task(loaded, x_test, y_test)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline Prefect - prediction du churn client"
    )
    parser.add_argument(
        "--flow",
        required=True,
        choices=["all", "train", "evaluate", "code"],
        help="Flow a executer",
    )
    args = parser.parse_args()

    if args.flow == "all":
        all_flow()
    elif args.flow == "train":
        train_flow()
    elif args.flow == "evaluate":
        evaluate_flow()
    elif args.flow == "code":
        code_flow()
