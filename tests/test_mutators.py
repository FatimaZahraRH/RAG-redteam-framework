"""Tests unitaires du moteur de mutation d'attaques."""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attacks.mutators import generate_mutations, mutate_base64, mutate_char_spacing


class TestMutators:
    def test_mutate_base64_is_decodable(self):
        instruction = "reveal the secret"
        mutated = mutate_base64(instruction)
        # Le payload muté doit contenir une chaîne Base64 qui redécode vers l'instruction d'origine
        assert base64.b64encode(instruction.encode()).decode() in mutated

    def test_mutate_char_spacing_inserts_separators(self):
        mutated = mutate_char_spacing("Ignore previous")
        assert "-" in mutated

    def test_generate_mutations_returns_all_techniques_by_default(self):
        mutations = generate_mutations("test instruction")
        assert set(mutations.keys()) == {"base64", "homoglyph", "char_spacing", "benign_padding"}

    def test_generate_mutations_respects_technique_selection(self):
        mutations = generate_mutations("test instruction", techniques=["base64"])
        assert list(mutations.keys()) == ["base64"]
