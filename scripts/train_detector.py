#!/usr/bin/env python3
"""
Entraîne le détecteur ML (TF-IDF + régression logistique) sur la bibliothèque
de payloads d'attaque vs le corpus bénin, avec évaluation par validation
croisée (AUC-ROC, seuil optimal de Youden).

Usage :
    python scripts/train_detector.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.embedding_detector import train_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    detector, evaluation = train_and_evaluate()

    print("\n" + "=" * 60)
    print("ÉVALUATION DU DÉTECTEUR ML (validation croisée stratifiée)")
    print("=" * 60)
    print(f"AUC-ROC                : {evaluation.roc_auc}")
    print(f"Seuil optimal (Youden)  : {evaluation.youden_threshold} (J = {evaluation.youden_j})")
    print(f"Exemples malveillants   : {evaluation.n_positive}")
    print(f"Exemples bénins         : {evaluation.n_negative}")
    print("\n✅ Modèle entraîné et sauvegardé dans data/detector_model.joblib")


if __name__ == "__main__":
    main()
