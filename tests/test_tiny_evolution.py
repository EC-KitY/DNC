import random

import numpy as np
import torch
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators import GAIntVectorCreator
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.selections.tournament_selection import (
    TournamentSelection,
)
from eckity.subpopulation import Subpopulation

from eckity_dnc import (
    DeepNeuralCrossover,
    DeepNeuralCrossoverConfig,
    DNCFitnessEvaluator,
)


class OneMaxEvaluator(SimpleIndividualEvaluator):
    def evaluate_individual(self, individual):
        return float(sum(individual.vector))


def test_tiny_bit_vector_evolves_and_trains_dnc():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    evaluator = DNCFitnessEvaluator(OneMaxEvaluator())
    creator = GAIntVectorCreator(length=4, bounds=(0, 1))
    config = DeepNeuralCrossoverConfig(
        embedding_dim=2,
        sequence_length=4,
        num_embeddings=2,
        batch_size=2,
        learning_rate=1e-3,
        epsilon_greedy=0.1,
        use_device="cpu",
        n_parents=2,
    )
    crossover = DeepNeuralCrossover(
        probability=1.0,
        population_size=4,
        dnc_config=config,
        individual_evaluator=evaluator,
        vector_creator=creator,
    )
    evolution = SimpleEvolution(
        population=Subpopulation(
            creators=creator,
            population_size=4,
            evaluator=evaluator,
            higher_is_better=True,
            elitism_rate=0.0,
            operators_sequence=[crossover],
            selection_methods=[TournamentSelection(tournament_size=2)],
        ),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=3,
        random_seed=7,
    )

    evolution.evolve()

    individuals = evolution.population.sub_populations[0].individuals
    best_fitness = float(evolution.best_of_run_.get_pure_fitness())
    assert evolution.generation_num == 3
    assert len(individuals) == 4
    assert 0.0 <= best_fitness <= 4.0
    assert all(
        len(individual.vector) == 4 and set(individual.vector) <= {0, 1}
        for individual in individuals
    )
    assert crossover.dnc_wrapper.optimizer.state
    assert all(
        torch.isfinite(parameter).all()
        for parameter in crossover.dnc_wrapper.neural_crossover.parameters()
    )
