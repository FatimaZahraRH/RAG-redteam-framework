"""
Classifieur "constitutionnel" : entraîné sur les données synthétiques
générées à partir de la constitution (src/detection/synthetic_generator.py),
plutôt que sur les ~20 exemples d'attaques rédigés à la main.

Même architecture que EmbeddingDetector (TF-IDF + régression logistique,
calibration ROC/Youden) -- ce qui change n'est pas l'algorithme, mais
l'ORIGINE et le VOLUME des données d'entraînement. C'est précisément le
point démontré empiriquement par ce module : la généralisation à des
attaques jamais vues dépend autant de la diversité des données
d'entraînement que du choix de l'algorithme.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.detection.embedding_detector import CrossValEvaluation
from src.detection.synthetic_generator import build_synthetic_dataset

logger = logging.getLogger(__name__)

CONSTITUTIONAL_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "constitutional_detector.joblib"


def train_constitutional_classifier() -> tuple["ConstitutionalClassifier", CrossValEvaluation]:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    texts, labels = build_synthetic_dataset()
    labels_arr = np.array(labels)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_probas = cross_val_predict(pipeline, texts, labels_arr, cv=cv, method="predict_proba")[:, 1]

    fpr, tpr, thresholds = roc_curve(labels_arr, cross_val_probas)
    auc = roc_auc_score(labels_arr, cross_val_probas)
    youden_j_values = tpr - fpr
    best_idx = int(youden_j_values.argmax())
    youden_threshold = float(thresholds[best_idx])

    pipeline.fit(texts, labels_arr)

    evaluation = CrossValEvaluation(
        roc_auc=round(float(auc), 4),
        youden_threshold=round(youden_threshold, 4),
        youden_j=round(float(youden_j_values[best_idx]), 4),
        fpr=[round(float(x), 4) for x in fpr],
        tpr=[round(float(x), 4) for x in tpr],
        thresholds=[round(float(x), 4) for x in thresholds],
        n_positive=int(labels_arr.sum()),
        n_negative=int(len(labels_arr) - labels_arr.sum()),
    )

    classifier = ConstitutionalClassifier(pipeline=pipeline, threshold=youden_threshold)
    classifier.save()

    logger.info(
        "Classifieur constitutionnel entraîné : AUC=%.3f (validation croisée sur données synthétiques), "
        "seuil=%.3f, %d positifs / %d négatifs",
        evaluation.roc_auc, evaluation.youden_threshold, evaluation.n_positive, evaluation.n_negative,
    )
    return classifier, evaluation


class ConstitutionalClassifier:
    def __init__(self, pipeline=None, threshold: float = 0.5):
        self._pipeline = pipeline
        self.threshold = threshold

    def score(self, text: str) -> float:
        if self._pipeline is None:
            raise RuntimeError("Classifieur non entraîné/chargé. Lancez scripts/train_constitutional.py.")
        return round(float(self._pipeline.predict_proba([text])[0][1]), 4)

    def save(self, path: Path = CONSTITUTIONAL_MODEL_PATH) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: Path = CONSTITUTIONAL_MODEL_PATH) -> "ConstitutionalClassifier":
        import joblib

        if not path.exists():
            raise FileNotFoundError(
                f"Aucun classifieur constitutionnel entraîné trouvé à {path}. "
                "Lancez : python scripts/train_constitutional.py"
            )
        state = joblib.load(path)
        return cls(pipeline=state["pipeline"], threshold=state["threshold"])


@dataclass
class GeneralizationComparison:
    """Compare la capacité de généralisation des deux détecteurs sur un jeu
    de test AVEUGLE (src/evaluation/blind_test_set.py) : aucun des deux
    détecteurs n'a vu ces exemples précis à l'entraînement -- contrairement
    à une comparaison sur data/attack_payloads/*.json, qui favoriserait
    injustement l'ancien détecteur (entraîné dessus)."""

    original_detector_recall: float
    constitutional_detector_recall: float
    original_detector_false_positive_rate: float
    constitutional_detector_false_positive_rate: float
    n_attacks: int
    n_benign: int


def compare_generalization(original_detector, constitutional_detector) -> GeneralizationComparison:
    from src.evaluation.blind_test_set import BLIND_TEST_ATTACKS, BLIND_TEST_BENIGN

    attack_texts = [t for t, _ in BLIND_TEST_ATTACKS]

    def evaluate(detector) -> tuple[float, float]:
        attack_scores = [detector.score(t) for t in attack_texts]
        benign_scores = [detector.score(t) for t in BLIND_TEST_BENIGN]
        recall = sum(1 for s in attack_scores if s >= detector.threshold) / len(attack_scores)
        fpr = sum(1 for s in benign_scores if s >= detector.threshold) / len(benign_scores)
        return round(recall, 4), round(fpr, 4)

    original_recall, original_fpr = evaluate(original_detector)
    constitutional_recall, constitutional_fpr = evaluate(constitutional_detector)

    return GeneralizationComparison(
        original_detector_recall=original_recall,
        constitutional_detector_recall=constitutional_recall,
        original_detector_false_positive_rate=original_fpr,
        constitutional_detector_false_positive_rate=constitutional_fpr,
        n_attacks=len(BLIND_TEST_ATTACKS),
        n_benign=len(BLIND_TEST_BENIGN),
    )
