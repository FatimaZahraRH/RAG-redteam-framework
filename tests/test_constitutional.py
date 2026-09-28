"""Tests unitaires de la constitution, du générateur synthétique et du classifieur constitutionnel."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.constitution import CONSTITUTION, all_core_instructions
from src.detection.synthetic_generator import (
    build_synthetic_dataset,
    generate_synthetic_attacks,
    generate_synthetic_benign,
)


class TestConstitution:
    def test_has_at_least_two_rules(self):
        assert len(CONSTITUTION) >= 2

    def test_each_rule_has_core_instructions(self):
        for rule in CONSTITUTION:
            assert len(rule.core_instructions) > 0

    def test_all_core_instructions_returns_pairs(self):
        pairs = all_core_instructions()
        assert len(pairs) > 0
        for instr, category in pairs:
            assert isinstance(instr, str) and len(instr) > 0
            assert category in ("secret_disclosure", "excessive_agency")


class TestSyntheticGenerator:
    def test_generates_more_examples_than_original_dataset(self):
        """Le point central de cette amélioration : le jeu synthétique doit
        être significativement plus grand que les ~20 exemples originaux."""
        attacks = generate_synthetic_attacks()
        assert len(attacks) > 50

    def test_all_generated_attacks_have_valid_category(self):
        attacks = generate_synthetic_attacks()
        for text, category in attacks:
            assert category in ("secret_disclosure", "excessive_agency")
            assert len(text) > 10

    def test_benign_examples_generated(self):
        benign = generate_synthetic_benign()
        assert len(benign) > 0

    def test_build_synthetic_dataset_balanced_enough(self):
        texts, labels = build_synthetic_dataset()
        n_positive = sum(labels)
        n_negative = len(labels) - n_positive
        # Pas besoin d'un équilibre parfait (class_weight='balanced' au
        # moment de l'entraînement gère l'écart), mais pas non plus d'un
        # déséquilibre extrême (ex. 100:1) qui rendrait l'entraînement instable.
        ratio = max(n_positive, n_negative) / max(1, min(n_positive, n_negative))
        assert ratio < 3.0

    def test_no_duplicate_framing_produces_identical_texts_across_instructions(self):
        """Vérifie que les textes générés sont bien différenciés (pas de
        génération dégénérée qui produirait le même texte partout)."""
        attacks = generate_synthetic_attacks()
        texts = [t for t, _ in attacks]
        assert len(set(texts)) > len(texts) * 0.8  # au moins 80% de textes uniques


class TestConstitutionalClassifierTraining:
    def test_train_produces_valid_model(self):
        from src.detection.constitutional_classifier import train_constitutional_classifier

        classifier, evaluation = train_constitutional_classifier()
        assert 0.5 <= evaluation.roc_auc <= 1.0
        assert 0.0 <= evaluation.youden_threshold <= 1.0

    def test_trained_classifier_scores_bounded(self):
        from src.detection.constitutional_classifier import train_constitutional_classifier

        classifier, _ = train_constitutional_classifier()
        score = classifier.score("some arbitrary text to score")
        assert 0.0 <= score <= 1.0

    def test_load_raises_if_not_trained(self, tmp_path):
        from src.detection.constitutional_classifier import ConstitutionalClassifier

        missing_path = tmp_path / "does_not_exist.joblib"
        try:
            ConstitutionalClassifier.load(missing_path)
            assert False, "Devrait lever FileNotFoundError"
        except FileNotFoundError:
            pass


class TestBlindTestSet:
    def test_blind_set_not_empty(self):
        from src.evaluation.blind_test_set import BLIND_TEST_ATTACKS, BLIND_TEST_BENIGN

        assert len(BLIND_TEST_ATTACKS) > 0
        assert len(BLIND_TEST_BENIGN) > 0

    def test_blind_attacks_not_verbatim_in_synthetic_templates(self):
        """Garantit que le jeu aveugle ne recopie pas mot pour mot les
        instructions de base de la constitution (sinon ce ne serait pas un
        vrai test de généralisation)."""
        from src.detection.constitution import all_core_instructions
        from src.evaluation.blind_test_set import BLIND_TEST_ATTACKS

        core_instructions = {instr.lower() for instr, _ in all_core_instructions()}
        for text, _ in BLIND_TEST_ATTACKS:
            assert text.lower() not in core_instructions
