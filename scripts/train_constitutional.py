#!/usr/bin/env python3
"""
Entraîne le classifieur constitutionnel (données synthétiques dérivées de la
constitution) et le compare à l'ancien détecteur (entraîné sur les payloads
écrits à la main), sur un jeu de test AVEUGLE que ni l'un ni l'autre n'a vu.

Usage :
    python scripts/train_constitutional.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.constitutional_classifier import compare_generalization, train_constitutional_classifier
from src.detection.embedding_detector import train_and_evaluate
from src.detection.synthetic_generator import generate_synthetic_attacks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    synthetic_attacks = generate_synthetic_attacks()
    print(f"Génération de {len(synthetic_attacks)} exemples d'attaque synthétiques "
          f"à partir de la constitution (2 règles × 7 framings × ~5 instructions).\n")

    print("=" * 60)
    print("ENTRAÎNEMENT DE L'ANCIEN DÉTECTEUR (data/attack_payloads, ~20 exemples)")
    print("=" * 60)
    original_detector, original_eval = train_and_evaluate()
    print(f"AUC (validation croisée) : {original_eval.roc_auc}")
    print(f"Seuil calibré : {original_eval.youden_threshold}\n")

    print("=" * 60)
    print("ENTRAÎNEMENT DU CLASSIFIEUR CONSTITUTIONNEL (données synthétiques)")
    print("=" * 60)
    constitutional_detector, constitutional_eval = train_constitutional_classifier()
    print(f"AUC (validation croisée) : {constitutional_eval.roc_auc}")
    print(f"Seuil calibré : {constitutional_eval.youden_threshold}\n")

    print("=" * 60)
    print("COMPARAISON SUR JEU DE TEST AVEUGLE (10 attaques + 10 bénins, formulations inédites)")
    print("=" * 60)
    comparison = compare_generalization(original_detector, constitutional_detector)
    print(f"Rappel (attaques détectées)   : ancien={comparison.original_detector_recall:.0%} | "
          f"constitutionnel={comparison.constitutional_detector_recall:.0%}")
    print(f"Taux de faux positifs (bénin) : ancien={comparison.original_detector_false_positive_rate:.0%} | "
          f"constitutionnel={comparison.constitutional_detector_false_positive_rate:.0%}")

    print("\n✅ Modèles entraînés et sauvegardés dans data/detector_model.joblib "
          "et data/constitutional_detector.joblib")


if __name__ == "__main__":
    main()
