"""
Orchestrateur de campagne de red-teaming : exécute l'ensemble des cas
d'attaque contre la cible, deux fois (sans défense / avec défense), et
agrège les résultats en métriques exploitables (ASR global, ASR par
catégorie, ASR par objectif OWASP).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from src.attacks.payload_library import AttackCase, load_all_attack_cases
from src.detection.pipeline_guard import PipelineGuard
from src.target.llm_provider import TargetProvider
from src.target.rag_pipeline import CaseExecutionResult, RAGRedTeamHarness

logger = logging.getLogger(__name__)


@dataclass
class CampaignResult:
    condition: str  # "undefended" | "defended"
    results: List[CaseExecutionResult] = field(default_factory=list)

    @property
    def overall_asr(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(r.attack_succeeded for r in self.results) / len(self.results), 4)

    def asr_by_category(self) -> Dict[str, float]:
        by_cat: Dict[str, List[bool]] = {}
        for r in self.results:
            by_cat.setdefault(r.case.category, []).append(r.attack_succeeded)
        return {cat: round(sum(v) / len(v), 4) for cat, v in by_cat.items()}

    def asr_by_owasp(self) -> Dict[str, float]:
        by_owasp: Dict[str, List[bool]] = {}
        for r in self.results:
            by_owasp.setdefault(r.case.owasp_llm, []).append(r.attack_succeeded)
        return {cat: round(sum(v) / len(v), 4) for cat, v in by_owasp.items()}


def run_campaign(target: TargetProvider, guard: PipelineGuard | None, cases: List[AttackCase] | None = None) -> CampaignResult:
    cases = cases if cases is not None else load_all_attack_cases()
    harness = RAGRedTeamHarness(target=target, guard=guard)

    condition = "defended" if guard is not None else "undefended"
    results = [harness.run_case(case) for case in cases]

    campaign = CampaignResult(condition=condition, results=results)
    logger.info("Campagne '%s' terminée : ASR global = %.1f%% (%d cas testés)",
                condition, campaign.overall_asr * 100, len(cases))
    return campaign
