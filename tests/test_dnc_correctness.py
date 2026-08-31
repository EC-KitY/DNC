import random

import numpy as np
import pytest
import torch
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators import GAIntVectorCreator
from eckity.evaluators import SimpleIndividualEvaluator
from eckity.genetic_operators import IntVectorOnePointMutation
from eckity.genetic_operators.selections.tournament_selection import (
    TournamentSelection,
)
from eckity.subpopulation import Subpopulation

from eckity_dnc import (
    DeepNeuralCrossover,
    DeepNeuralCrossoverConfig,
    DNCFitnessEvaluator,
)
from eckity_dnc.neural_crossover import NeuralCrossover
from eckity_dnc.wrapper import NeuralCrossoverWrapper


class CountingEvaluator(SimpleIndividualEvaluator):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.last_direction = None

    def evaluate_individual(self, individual):
        self.calls += 1
        self.last_direction = individual.fitness.higher_is_better
        return float(sum(individual.vector))


def make_operator(population_size=2, length=2, higher_is_better=True, probability=1.0):
    base_evaluator = CountingEvaluator()
    evaluator = DNCFitnessEvaluator(base_evaluator)
    creator = GAIntVectorCreator(length=length, bounds=(0, 1))
    operator = DeepNeuralCrossover(
        probability=probability,
        population_size=population_size,
        dnc_config=DeepNeuralCrossoverConfig(
            embedding_dim=2,
            sequence_length=length,
            num_embeddings=2,
            batch_size=max(population_size, 2),
            use_device="cpu",
            higher_is_better=higher_is_better,
        ),
        individual_evaluator=evaluator,
        vector_creator=creator,
    )
    return operator, creator, evaluator, base_evaluator


def test_dnc_requires_shared_fitness_evaluator_adapter():
    creator = GAIntVectorCreator(length=1, bounds=(0, 1))
    with pytest.raises(TypeError, match="DNCFitnessEvaluator"):
        DeepNeuralCrossover(
            probability=1.0,
            population_size=2,
            dnc_config=DeepNeuralCrossoverConfig(2, 1, 2),
            individual_evaluator=CountingEvaluator(),
            vector_creator=creator,
        )


def test_fitness_adapter_reuses_evaluated_fitness_and_respects_invalidation():
    base_evaluator = CountingEvaluator()
    evaluator = DNCFitnessEvaluator(base_evaluator)
    creator = GAIntVectorCreator(length=1, bounds=(0, 1))
    individual = creator.create_individuals(1, higher_is_better=True)[0]
    individual.set_vector([1])

    assert evaluator.evaluate_individual(individual) == 1.0
    assert base_evaluator.calls == 1

    individual.fitness.set_fitness(7.0)
    assert evaluator.evaluate_individual(individual) == 7.0
    assert base_evaluator.calls == 1

    mutation = IntVectorOnePointMutation(
        probability=1.0,
        probability_for_each=1.0,
    )
    mutation.apply_operator([individual])
    evaluator.evaluate_individual(individual)
    assert base_evaluator.calls == 2


def test_minimization_direction_reaches_temporary_individual_and_rewards():
    operator, _, _, base_evaluator = make_operator(higher_is_better=False)

    operator.get_fitness_from_vector([0, 1])
    assert base_evaluator.last_direction is False

    wrapper = operator.dnc_wrapper
    fitness_values = torch.tensor([1.0, 10.0])
    assert torch.equal(wrapper._fitness_to_rewards(fitness_values), -fitness_values)

    logits = torch.zeros(2, requires_grad=True)
    loss = -torch.mean(
        torch.log_softmax(logits, dim=0)
        * wrapper._fitness_to_rewards(fitness_values)
    )
    loss.backward()
    updated_logits = logits.detach() - 0.1 * logits.grad
    assert updated_logits.argmax().item() == 0


def test_maximization_rewards_preserve_fitness_values():
    wrapper = NeuralCrossoverWrapper(
        embedding_dim=2,
        sequence_length=1,
        num_embeddings=2,
        get_fitness_function=lambda vector: float(sum(vector)),
        higher_is_better=True,
    )
    fitness_values = torch.tensor([1.0, 10.0])
    assert torch.equal(wrapper._fitness_to_rewards(fitness_values), fitness_values)


class RecordingDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x, ref, previous_hidden):
        self.inputs.append(x.detach().clone())
        probabilities = torch.zeros(
            (ref.shape[0], ref.shape[1]), device=ref.device
        )
        probabilities[:, -1] = 1.0
        return probabilities, previous_hidden


def test_decoder_receives_selected_gene_embedding():
    model = NeuralCrossover(
        input_size=2,
        hidden_size=2,
        n_embeddings=4,
        ind_length=2,
        n_parents=2,
        device="cpu",
    )
    decoder = RecordingDecoder()
    model.decoder = decoder
    with torch.no_grad():
        model.embedding.embedding.weight.copy_(
            torch.tensor(
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
            )
        )

    parents = torch.tensor([[[0, 1]], [[2, 3]]], dtype=torch.long)
    model(parents, epsilon_greedy=0.0)

    assert torch.equal(decoder.inputs[1], torch.tensor([[[2.0, 2.0]]]))


def test_single_value_genome_with_two_parents_does_not_index_parent_as_gene():
    torch.manual_seed(0)
    model = NeuralCrossover(
        input_size=2,
        hidden_size=2,
        n_embeddings=1,
        ind_length=3,
        n_parents=2,
        device="cpu",
    )
    parents = torch.zeros((2, 8, 3), dtype=torch.long)

    _, children = model(parents, epsilon_greedy=1.0)
    assert children.shape == (8, 3)


@pytest.mark.parametrize("crossover_mask", [0.0, 1.0])
def test_odd_population_preserves_unpaired_individual(monkeypatch, crossover_mask):
    operator, creator, _, _ = make_operator(population_size=3, probability=0.5)
    individuals = creator.create_individuals(3, higher_is_better=True)
    vectors = [[0, 0], [1, 1], [0, 1]]
    for individual, vector in zip(individuals, vectors):
        individual.set_vector(vector)

    monkeypatch.setattr(
        np.random, "uniform", lambda size: np.full(size, crossover_mask)
    )
    result = operator.apply(individuals)

    assert len(result) == 3
    assert result[-1].vector.tolist() == vectors[-1]


def test_tiny_evolution_delegates_each_required_fitness_only_once():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    operator, creator, evaluator, base_evaluator = make_operator(
        population_size=2,
        length=1,
    )
    evolution = SimpleEvolution(
        population=Subpopulation(
            creators=creator,
            population_size=2,
            evaluator=evaluator,
            higher_is_better=True,
            elitism_rate=0.0,
            operators_sequence=[operator],
            selection_methods=[TournamentSelection(tournament_size=2)],
        ),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=2,
        random_seed=7,
    )

    evolution.evolve()

    assert base_evaluator.calls == 6
