"""
Détecteur ML — classifieur entraîné (TF-IDF + régression logistique) pour
distinguer un texte "porteur d'instruction malveillante injectée" d'un texte
bénin.

Ce module reprend la méthodologie déjà appliquée dans mon projet Cassiopée
(courbe ROC, indice de Youden pour le seuil optimal) — appliquée ici à un
problème différent (détection d'injection de prompt plutôt que near-duplicate
d'image), ce qui démontre que la méthode se généralise au-delà d'un cas
d'usage unique.

Le jeu d'entraînement étant volontairement petit (démonstration), l'évaluation
utilise une validation croisée stratifiée (chaque exemple est prédit une fois
hors de son propre pli d'entraînement) plutôt qu'un simple split train/test,
pour obtenir une estimation plus honnête sur un petit échantillon.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from src.attacks.payload_library import load_all_attack_cases, load_benign_documents

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "detector_model.joblib"


def _flatten_attack_texts() -> List[str]:
    """Extrait tous les textes malveillants (bruts, non obfusqués côté
    apprentissage — le détecteur doit apprendre à généraliser au-delà des
    formulations exactes vues à l'entraînement, cf. évaluation avec mutations
    dans evaluation/attack_runner.py)."""
    texts: List[str] = []
    for case in load_all_attack_cases():
        if case.user_message:
            texts.append(case.user_message)
        if case.poisoned_document:
            texts.append(case.poisoned_document)
        texts.extend(case.turns)
    return texts


def build_training_data() -> Tuple[List[str], List[int]]:
    attack_texts = _flatten_attack_texts()
    benign_texts = [d["text"] for d in load_benign_documents()]

    texts = attack_texts + benign_texts
    labels = [1] * len(attack_texts) + [0] * len(benign_texts)
    return texts, labels


@dataclass
class CrossValEvaluation:
    roc_auc: float
    youden_threshold: float
    youden_j: float
    fpr: List[float] = field(default_factory=list)
    tpr: List[float] = field(default_factory=list)
    thresholds: List[float] = field(default_factory=list)
    n_positive: int = 0
    n_negative: int = 0


def train_and_evaluate() -> Tuple["EmbeddingDetector", CrossValEvaluation]:
    """Entraîne le détecteur ML sur l'ensemble du jeu de données disponible,
    tout en produisant une évaluation honnête par validation croisée (chaque
    prédiction utilisée pour l'évaluation provient d'un modèle qui n'a PAS vu
    cet exemple pendant son propre entraînement)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts, labels = build_training_data()
    labels_arr = np.array(labels)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    n_splits = min(5, min(labels_arr.sum(), len(labels_arr) - labels_arr.sum()))
    n_splits = max(2, n_splits)  # au moins 2 plis
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cross_val_probas = cross_val_predict(pipeline, texts, labels_arr, cv=cv, method="predict_proba")[:, 1]

    fpr, tpr, thresholds = roc_curve(labels_arr, cross_val_probas)
    auc = roc_auc_score(labels_arr, cross_val_probas)

    # Indice de Youden : J = TPR - FPR, maximisé -> seuil optimal
    youden_j_values = tpr - fpr
    best_idx = int(youden_j_values.argmax())
    youden_threshold = float(thresholds[best_idx])
    youden_j = float(youden_j_values[best_idx])

    evaluation = CrossValEvaluation(
        roc_auc=round(float(auc), 4),
        youden_threshold=round(youden_threshold, 4),
        youden_j=round(youden_j, 4),
        fpr=[round(float(x), 4) for x in fpr],
        tpr=[round(float(x), 4) for x in tpr],
        thresholds=[round(float(x), 4) for x in thresholds],
        n_positive=int(labels_arr.sum()),
        n_negative=int(len(labels_arr) - labels_arr.sum()),
    )

    # Entraînement final sur 100% des données (le modèle livré en production
    # utilise tout ce qui est disponible ; seule l'ÉVALUATION ci-dessus a
    # utilisé la validation croisée pour rester honnête).
    pipeline.fit(texts, labels_arr)

    detector = EmbeddingDetector(pipeline=pipeline, threshold=youden_threshold)
    detector.save()

    logger.info(
        "Détecteur ML entraîné : AUC=%.3f, seuil de Youden=%.3f (J=%.3f), %d positifs / %d négatifs",
        evaluation.roc_auc, evaluation.youden_threshold, evaluation.youden_j,
        evaluation.n_positive, evaluation.n_negative,
    )
    return detector, evaluation


class EmbeddingDetector:
    """Wrapper autour du pipeline scikit-learn entraîné, avec persistance et
    chargement, pour être utilisé par le PipelineGuard sans réentraîner à
    chaque exécution."""

    def __init__(self, pipeline=None, threshold: float = 0.5):
        self._pipeline = pipeline
        self.threshold = threshold

    def score(self, text: str) -> float:
        if self._pipeline is None:
            raise RuntimeError("Détecteur non entraîné/chargé. Lancez scripts/train_detector.py d'abord.")
        proba = self._pipeline.predict_proba([text])[0][1]
        return round(float(proba), 4)

    def save(self, path: Path = MODEL_PATH) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "threshold": self.threshold}, path)
        logger.info("Détecteur ML sauvegardé dans %s", path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "EmbeddingDetector":
        import joblib

        if not path.exists():
            raise FileNotFoundError(
                f"Aucun modèle entraîné trouvé à {path}. Lancez : python scripts/train_detector.py"
            )
        state = joblib.load(path)
        return cls(pipeline=state["pipeline"], threshold=state["threshold"])
