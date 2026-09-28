#!/usr/bin/env python3
"""
Lance la campagne complète de red-teaming : exécute tous les cas d'attaque
sans défense, puis avec défense, et génère le rapport comparatif final.

Prérequis : avoir entraîné le détecteur au préalable (scripts/train_detector.py).

Usage :
    python scripts/run_redteam_campaign.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attacks.payload_library import load_all_attack_cases
from src.config import llm_config
from src.detection.embedding_detector import EmbeddingDetector
from src.detection.pipeline_guard import PipelineGuard
from src.evaluation.attack_runner import run_campaign
from src.reporting.report_generator import generate_campaign_report, save_report
from src.target.llm_provider import get_target_provider

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    print(f"Fournisseur cible : {llm_config.provider} (configurable via TARGET_LLM_PROVIDER dans .env)\n")

    cases = load_all_attack_cases()
    print(f"Chargement de {len(cases)} cas d'attaque, répartis en 4 catégories.\n")

    target = get_target_provider()

    print("--- Campagne SANS défense (baseline) ---")
    undefended = run_campaign(target=target, guard=None, cases=cases)
    print(f"ASR global sans défense : {undefended.overall_asr * 100:.1f}%")
    for cat, asr in undefended.asr_by_category().items():
        print(f"  - {cat}: {asr * 100:.1f}%")

    print("\n--- Chargement du détecteur ML (évaluation croisée déjà calculée à l'entraînement) ---")
    try:
        embedding_detector = EmbeddingDetector.load()
    except FileNotFoundError as exc:
        print(f"⚠️  {exc}")
        print("Lancez d'abord : python scripts/train_detector.py")
        return

    guard = PipelineGuard(embedding_detector=embedding_detector)

    print("\n--- Campagne AVEC défense (PipelineGuard actif) ---")
    defended = run_campaign(target=target, guard=guard, cases=cases)
    print(f"ASR global avec défense : {defended.overall_asr * 100:.1f}%")
    for cat, asr in defended.asr_by_category().items():
        print(f"  - {cat}: {asr * 100:.1f}%")

    # On recharge l'évaluation croisée déjà calculée lors de l'entraînement
    # pour l'inclure dans le rapport (évite de ré-entraîner ici).
    from src.detection.embedding_detector import train_and_evaluate
    _, detector_eval = train_and_evaluate()  # ré-entraîne rapidement (dataset petit) pour le rapport

    report = generate_campaign_report(undefended, defended, detector_eval, target_name=target.name)
    output_path = Path(__file__).resolve().parent.parent / "redteam_campaign_report.md"
    save_report(report, str(output_path))

    print(f"\n📄 Rapport complet généré : {output_path}")
    print(f"\n🎯 Réduction de l'ASR grâce à la défense : "
          f"{(undefended.overall_asr - defended.overall_asr) * 100:.1f} points de pourcentage")


if __name__ == "__main__":
    main()
