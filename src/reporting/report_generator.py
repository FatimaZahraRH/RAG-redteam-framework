"""
Génère un rapport Markdown complet de la campagne de red-teaming :
ASR avant/après défense, détail par catégorie et par catégorie OWASP,
et métriques du détecteur ML (AUC, seuil de Youden).
"""
from __future__ import annotations

from datetime import datetime

from src.evaluation.attack_runner import CampaignResult
from src.evaluation.owasp_mapping import get_description
from src.detection.embedding_detector import CrossValEvaluation


def generate_campaign_report(
    undefended: CampaignResult,
    defended: CampaignResult,
    detector_eval: CrossValEvaluation,
    target_name: str,
) -> str:
    lines: list[str] = []
    lines.append("# Rapport de campagne de red-teaming — Système RAG/Agentique")
    lines.append(f"*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} — Cible testée : `{target_name}`*\n")

    lines.append("## Synthèse")
    reduction = undefended.overall_asr - defended.overall_asr
    reduction_pct = round((reduction / undefended.overall_asr) * 100, 1) if undefended.overall_asr > 0 else 0.0
    lines.append(f"- **ASR sans défense : {undefended.overall_asr * 100:.1f} %**")
    lines.append(f"- **ASR avec défense (PipelineGuard) : {defended.overall_asr * 100:.1f} %**")
    lines.append(f"- **Réduction du taux de succès d'attaque : {reduction_pct} %**\n")

    lines.append("## Performance du détecteur ML (validation croisée)")
    lines.append(f"- AUC-ROC : **{detector_eval.roc_auc}**")
    lines.append(f"- Seuil optimal (indice de Youden) : **{detector_eval.youden_threshold}** (J = {detector_eval.youden_j})")
    lines.append(f"- Jeu d'évaluation : {detector_eval.n_positive} exemples malveillants, {detector_eval.n_negative} exemples bénins\n")

    lines.append("## ASR par catégorie d'attaque")
    lines.append("| Catégorie | Sans défense | Avec défense |")
    lines.append("|---|---|---|")
    undefended_by_cat = undefended.asr_by_category()
    defended_by_cat = defended.asr_by_category()
    for category in sorted(undefended_by_cat.keys()):
        lines.append(
            f"| {category} | {undefended_by_cat[category] * 100:.1f} % "
            f"| {defended_by_cat.get(category, 0.0) * 100:.1f} % |"
        )
    lines.append("")

    lines.append("## ASR par catégorie OWASP Top 10 for LLM")
    lines.append("| Catégorie OWASP | Sans défense | Avec défense |")
    lines.append("|---|---|---|")
    undefended_by_owasp = undefended.asr_by_owasp()
    defended_by_owasp = defended.asr_by_owasp()
    for owasp_cat in sorted(undefended_by_owasp.keys()):
        lines.append(
            f"| {owasp_cat} | {undefended_by_owasp[owasp_cat] * 100:.1f} % "
            f"| {defended_by_owasp.get(owasp_cat, 0.0) * 100:.1f} % |"
        )
    lines.append("")

    lines.append("## Description des catégories OWASP observées")
    for owasp_cat in sorted(undefended_by_owasp.keys()):
        lines.append(f"- **{owasp_cat}** : {get_description(owasp_cat)}")
    lines.append("")

    lines.append("## Détail cas par cas")
    lines.append("| ID | Catégorie | Objectif | Sans défense | Avec défense |")
    lines.append("|---|---|---|---|---|")
    defended_by_id = {r.case.case_id: r for r in defended.results}
    for r in undefended.results:
        d = defended_by_id.get(r.case.case_id)
        undefended_status = "🔴 Réussie" if r.attack_succeeded else "🟢 Échouée"
        if d is None:
            defended_status = "N/A"
        elif d.blocked_by_guard:
            defended_status = "🛡️ Bloquée par le garde"
        elif d.attack_succeeded:
            defended_status = "🔴 Réussie"
        else:
            defended_status = "🟢 Échouée"
        lines.append(f"| {r.case.case_id} | {r.case.category} | {r.case.goal} | {undefended_status} | {defended_status} |")

    lines.append("\n---")
    lines.append(
        "*Rapport généré automatiquement par le framework de red-teaming. "
        "Toutes les attaques ont été exécutées dans un environnement sandboxé, "
        "contre une cible factice ou un LLM de test propre, à des fins défensives.*"
    )

    return "\n".join(lines)


def save_report(markdown: str, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
