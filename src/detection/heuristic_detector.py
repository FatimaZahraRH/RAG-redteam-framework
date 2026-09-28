"""
Détecteur heuristique (règles) — premier niveau de filtrage, rapide et peu
coûteux en calcul, typique d'un pipeline de détection en production (on ne
fait appel au modèle ML plus coûteux que si ce premier filtre ne suffit pas
à trancher).

Applique la même normalisation "sémantique" que la cible factice
(src.target.llm_provider.naive_semantic_normalize) pour rester efficace
face aux techniques d'obfuscation de base (Base64, espacement, homoglyphes).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from src.target.llm_provider import naive_semantic_normalize

_SUSPICIOUS_PATTERNS = [
    r"\bignore\b.{0,30}\binstructions?\b",
    r"\bignor[ae]\b.{0,30}\binstru",  # variantes FR/IT
    r"\bdisregard\b.{0,30}\binstructions?\b",
    r"\bsystem\s*override\b",
    r"\bdeveloper\s*mode\b",
    r"\byou\s*are\s*now\s*dan\b",
    r"\breveal\b.{0,30}\b(code|secret|password|token)\b",
    r"\brévèle?\b.{0,30}\bcode\b",
    r"\bconfidential_internal_code\b",
    r"\bdelete_all_data\b",
    r"\[?system\s*note\s*:",
    r"\[?assistant\s*instruction\s*:",
    r"\bnote\s*de\s*l.administrateur\s*système\b",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _SUSPICIOUS_PATTERNS]

# Signaux structurels (pas de mots-clés précis, mais des formes suspectes)
_BASE64_LIKE_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_HIDDEN_HTML_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass
class HeuristicResult:
    score: float  # 0 à 1
    matched_signals: List[str] = field(default_factory=list)


def score_text(text: str) -> HeuristicResult:
    normalized = naive_semantic_normalize(text)
    signals: List[str] = []

    for pattern in _COMPILED_PATTERNS:
        if pattern.search(normalized):
            signals.append(pattern.pattern)

    if _BASE64_LIKE_RE.search(text):
        signals.append("base64_like_substring")
    if _HIDDEN_HTML_RE.search(text):
        signals.append("hidden_html_content")
    if _HTML_COMMENT_RE.search(text):
        signals.append("html_comment_block")

    # Score simple : proportion de signaux distincts détectés, plafonné à 1.0.
    # Volontairement transparent (pas de boîte noire) pour rester auditable.
    max_expected_signals = 4
    score = min(1.0, len(signals) / max_expected_signals)

    return HeuristicResult(score=round(score, 3), matched_signals=signals)
