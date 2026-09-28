"""
Mapping des catégories d'attaque vers le OWASP Top 10 for LLM Applications.

Référence utilisée : OWASP Top 10 for LLM Applications (édition 2023/2024,
la plus largement diffusée au moment de la rédaction). Le classement OWASP
évolue périodiquement — vérifiez la version la plus récente sur
https://owasp.org/www-project-top-10-for-large-language-model-applications/
avant de citer ces identifiants dans un contexte professionnel.
"""
from __future__ import annotations

OWASP_LLM_DESCRIPTIONS = {
    "LLM01:2023 - Prompt Injection": (
        "Un attaquant manipule un LLM via des prompts habilement conçus, l'amenant à exécuter "
        "les intentions de l'attaquant. Peut être direct (jailbreak) ou indirect (via une source "
        "de données externe, comme un document récupéré par un RAG)."
    ),
    "LLM01:2023 - Prompt Injection (Indirect)": (
        "Variante indirecte : l'instruction malveillante n'est pas saisie par l'utilisateur mais "
        "cachée dans une source de données externe (document, page web) que le modèle traite comme "
        "contexte de confiance."
    ),
    "LLM06:2023 - Sensitive Information Disclosure": (
        "Le LLM révèle accidentellement des informations confidentielles (secrets, données "
        "personnelles, détails d'infrastructure) présentes dans son contexte ou son entraînement."
    ),
    "LLM08:2023 - Excessive Agency": (
        "Un LLM connecté à des outils/API dispose de permissions, fonctionnalités ou autonomie "
        "excessives, lui permettant d'exécuter des actions dommageables non prévues."
    ),
}


def get_description(owasp_id: str) -> str:
    return OWASP_LLM_DESCRIPTIONS.get(owasp_id, "Description non disponible pour cette catégorie.")
