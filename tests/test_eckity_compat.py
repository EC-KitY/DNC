import ast
import random
import runpy
from importlib.metadata import version
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from eckity.creators import GAIntVectorCreator
from eckity.evaluators import SimpleIndividualEvaluator
from eckity.genetic_encodings.ga import IntVector
from eckity.genetic_operators import IntVectorOnePointMutation

import eckity_dnc
from eckity_dnc import (
    DeepNeuralCrossover,
    DeepNeuralCrossoverConfig,
    DNCFitnessEvaluator,
)

ROOT = Path(__file__).parents[1]
RUNNER_FILES = [
    ROOT / "bpp_runner.py",
    ROOT / "dnc_runner_eckity.py",
    ROOT / "eckity_demo.py",
    ROOT / "eckity_runner.py",
]


def test_release_metadata_and_public_imports():
    assert version("eckity") == "0.4.2"
    assert version("eckity-dnc") == "0.2.0"
    assert DNCFitnessEvaluator is not None
    assert DeepNeuralCrossover is not None
    assert DeepNeuralCrossoverConfig is not None

    legacy_creator = "GAInteger" + "StringVectorCreator"
    source_files = [
        ROOT / "README.md",
        *ROOT.glob("*.py"),
        *(ROOT / "src" / "eckity_dnc").rglob("*.py"),
    ]
    assert eckity_dnc.__all__ == [
        "DNCFitnessEvaluator",
        "DeepNeuralCrossover",
        "DeepNeuralCrossoverConfig",
    ]
    assert not hasattr(eckity_dnc, legacy_creator)
    for path in source_files:
        assert legacy_creator not in path.read_text(encoding="utf-8")

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    dev_dependencies = project["project"]["optional-dependencies"]["dev"]
    assert any(dependency.lower().startswith("scipy") for dependency in dependencies)
    assert any(dependency.lower().startswith("tqdm") for dependency in dev_dependencies)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "running_mean_decay" not in readme
    assert "EC-KitY 0.4.2 imports it" in readme
    assert "RELEASING.md" not in readme


class RecordingEvaluator(SimpleIndividualEvaluator):
    def evaluate_individual(self, individual):
        self.individual = individual
        return float(sum(individual.vector))


def test_dnc_converts_native_creator_vectors_for_evaluation():
    evaluator = RecordingEvaluator()
    fitness_evaluator = DNCFitnessEvaluator(evaluator)
    creator = GAIntVectorCreator(length=2, bounds=(0, 1))
    crossover = DeepNeuralCrossover(
        probability=1.0,
        population_size=2,
        dnc_config=DeepNeuralCrossoverConfig(
            embedding_dim=2,
            sequence_length=2,
            num_embeddings=2,
            batch_size=2,
            use_device="cpu",
        ),
        individual_evaluator=fitness_evaluator,
        vector_creator=creator,
    )

    assert crossover.get_fitness_from_vector([0, 1]) == 1.0
    assert isinstance(evaluator.individual, IntVector)
    assert evaluator.individual.vector == [0, 1]
    assert evaluator.individual.fitness.higher_is_better


def test_runners_use_native_integer_uniform_mutation():
    legacy_class = "IntVector" + "UniformMutation"
    legacy_helper = "uniform_cell_" + "selector"
    source_files = [
        ROOT / "bpp_runner.py",
        ROOT / "dnc_aux.py",
        ROOT / "dnc_runner_eckity.py",
    ]

    for path in source_files:
        source = path.read_text(encoding="utf-8")
        assert legacy_class not in source
        assert legacy_helper not in source
    for path in (ROOT / "bpp_runner.py", ROOT / "dnc_runner_eckity.py"):
        assert "IntVectorOnePointMutation" in path.read_text(encoding="utf-8")

    creator = GAIntVectorCreator(length=8, bounds=(0, 5))
    original = [0, 1, 2, 3, 4, 5, 0, 1]
    first, second = creator.create_individuals(2, higher_is_better=True)
    first.set_vector(original.copy())
    second.set_vector(original.copy())
    mutation = IntVectorOnePointMutation(
        probability=1.0,
        probability_for_each=1.0,
    )

    assert mutation.cell_selector(first) == list(range(first.size()))
    random.seed(7)
    succeeded, _ = mutation.attempt_operator([first], 0)
    random.seed(7)
    mutation.attempt_operator([second], 0)

    assert succeeded
    assert first.vector == second.vector
    assert first.vector != original
    assert all(0 <= value <= 5 for value in first.vector)


def test_runners_do_not_use_legacy_selection_tuples_or_direction_keywords():
    for path in RUNNER_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            assert not (
                call_name in {"TournamentSelection", "ElitismSelection"}
                and any(keyword.arg == "higher_is_better" for keyword in node.keywords)
            )
            for keyword in node.keywords:
                if keyword.arg == "selection_methods" and isinstance(
                    keyword.value, ast.List
                ):
                    assert all(
                        not isinstance(item, ast.Tuple) for item in keyword.value.elts
                    )


def test_legacy_runner_is_safe_to_import(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    runpy.run_path(str(ROOT / "runner.py"), run_name="runner_import_test")


def test_one_max_demo_uses_bit_bounds():
    tree = ast.parse((ROOT / "eckity_demo.py").read_text(encoding="utf-8"))
    creator_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", ""))
        == "GABitStringVectorCreator"
    ]
    assert len(creator_calls) == 1
    bounds = next(
        keyword.value
        for keyword in creator_calls[0].keywords
        if keyword.arg == "bounds"
    )
    assert [element.value for element in bounds.elts] == [0, 1]
