"""
deploiement_prefect.py
======================
Planifie les flows du pipeline churn via des deployments Prefect.
Lancement : python deploiement_prefect.py
"""

from prefect import serve
from pipeline_prefect import all_flow, train_flow

if __name__ == "__main__":
    # Deployment du pipeline complet : execution quotidienne a 02h00
    all_deployment = all_flow.to_deployment(
        name="ml-pipeline-all",
        cron="0 2 * * *",
        tags=["mlops", "full-pipeline"],
    )

    # Deployment de l'entrainement seul : declenchement manuel
    train_deployment = train_flow.to_deployment(
        name="ml-pipeline-train",
        tags=["mlops", "train"],
    )

    # serve() enregistre les deployments et lance un worker integre
    serve(all_deployment, train_deployment)
