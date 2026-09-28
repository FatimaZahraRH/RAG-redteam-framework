"""
Garde de sécurité combiné : fusionne le détecteur heuristique (rapide, peu
coûteux) et le détecteur ML (plus fin, entraîné) en une seule décision
bloquer/laisser passer, avec un score pondéré transparent plutôt qu'une
boîte noire.

Positionnement dans le pipeline : ce garde inspecte le contenu AVANT qu'il
n'atteigne le LLM cible (le contexte récupéré par le RAG, ou le message
utilisateur), conformément au principe de défense en profondeur — on ne
compte pas uniquement sur la robustesse intrinsèque du LLM.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src.config import DetectionConfig, detection_config
from src.detection import heuristic_detector
from src.detection.embedding_detector import EmbeddingDetector

logger = logging.getLogger(__name__)


@dataclass
class GuardDecision:
    blocked: bool
    combined_score: float
    heuristic_score: float
    embedding_score: float
    matched_signals: List[str] = field(default_factory=list)


class PipelineGuard:
    def __init__(self, embedding_detector: Optional[EmbeddingDetector] = None, config: Optional[DetectionConfig] = None):
        self._config = config or detection_config
        self._embedding_detector = embedding_detector or EmbeddingDetector.load()

    def evaluate(self, text: str) -> GuardDecision:
        if not text or not text.strip():
            return GuardDecision(blocked=False, combined_score=0.0, heuristic_score=0.0, embedding_score=0.0)

        heuristic_result = heuristic_detector.score_text(text)
        embedding_score = self._embedding_detector.score(text)

        combined = (
            self._config.heuristic_weight * heuristic_result.score
            + self._config.embedding_weight * embedding_score
        )

        # Règle de sécurité additionnelle : si le détecteur ML dépasse SEUL
        # son seuil optimal (Youden), on bloque même si le score combiné
        # pondéré reste sous le seuil global — évite qu'un fort signal ML
        # soit noyé par un heuristique à zéro.
        blocked = (
            combined >= self._config.embedding_threshold
            or embedding_score >= self._embedding_detector.threshold
        )

        return GuardDecision(
            blocked=blocked,
            combined_score=round(combined, 4),
            heuristic_score=heuristic_result.score,
            embedding_score=embedding_score,
            matched_signals=heuristic_result.matched_signals,
        )
