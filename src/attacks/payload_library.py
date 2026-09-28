"""
Bibliothèque de payloads d'attaque.

Charge les 4 catégories de payloads (direct, indirect, obfuscation, multi-turn)
et les expose sous une forme unifiée (AttackCase) que le reste du framework
(harnais de cible, moteur d'évaluation) manipule sans se soucier du format
JSON d'origine de chaque catégorie.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

from src.config import PAYLOADS_DIR


@dataclass
class AttackCase:
    """Représentation unifiée d'un cas de test, quelle que soit sa catégorie
    d'origine (payload texte simple, document empoisonné, ou séquence
    multi-tours)."""

    case_id: str
    category: str  # direct_injection | indirect_injection | obfuscation | multi_turn
    owasp_llm: str
    goal: str  # "secret_disclosure" | "excessive_agency"
    description: str
    # Pour direct_injection / obfuscation : un seul message utilisateur
    user_message: Optional[str] = None
    # Pour indirect_injection : un document empoisonné inséré dans le contexte RAG
    poisoned_document: Optional[str] = None
    # Pour multi_turn : une séquence de messages
    turns: List[str] = field(default_factory=list)

    @property
    def is_multi_turn(self) -> bool:
        return bool(self.turns)

    @property
    def is_indirect(self) -> bool:
        return self.poisoned_document is not None


def _load_json(filename: str) -> list:
    path = PAYLOADS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_attack_cases() -> List[AttackCase]:
    cases: List[AttackCase] = []

    for entry in _load_json("direct_injection.json"):
        cases.append(AttackCase(
            case_id=entry["id"], category=entry["category"], owasp_llm=entry["owasp_llm"],
            goal=entry["goal"], description=entry["description"], user_message=entry["payload"],
        ))

    for entry in _load_json("obfuscation.json"):
        cases.append(AttackCase(
            case_id=entry["id"], category=entry["category"], owasp_llm=entry["owasp_llm"],
            goal=entry["goal"], description=entry["description"], user_message=entry["payload"],
        ))

    for entry in _load_json("indirect_injection.json"):
        cases.append(AttackCase(
            case_id=entry["id"], category=entry["category"], owasp_llm=entry["owasp_llm"],
            goal=entry["goal"], description=entry["description"],
            poisoned_document=entry["poisoned_document"],
            user_message=entry.get("trigger_query", "Peux-tu résumer les informations importantes de ce document ?"),
        ))

    for entry in _load_json("multi_turn.json"):
        cases.append(AttackCase(
            case_id=entry["id"], category=entry["category"], owasp_llm=entry["owasp_llm"],
            goal=entry["goal"], description=entry["description"], turns=entry["turns"],
        ))

    return cases


def load_benign_documents() -> List[dict]:
    from src.config import CORPUS_PATH
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
