import numpy as np
from eckity.creators import GAIntVectorCreator
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.genetic_operator import GeneticOperator

from .fitness_evaluator import DNCFitnessEvaluator
from .wrapper import NeuralCrossoverWrapper


class DeepNeuralCrossoverConfig:
    def __init__(self, embedding_dim, sequence_length, num_embeddings,
                 batch_size=32, load_weights_path=None, freeze_weights=False, learning_rate=1e-3, epsilon_greedy=0.1,
                 use_scheduler=False, use_device='cpu', adam_decay=0, clip_grads=False, n_parents=2, higher_is_better=True, scheduling_threshold=0):
        if scheduling_threshold < 0:
            raise ValueError("scheduling_threshold must be greater than or equal to 0")

        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.num_embeddings = num_embeddings
        self.batch_size = batch_size
        self.load_weights_path = load_weights_path
        self.freeze_weights = freeze_weights
        self.learning_rate = learning_rate
        self.epsilon_greedy = epsilon_greedy
        self.use_scheduler = use_scheduler
        self.use_device = use_device
        self.adam_decay = adam_decay
        self.clip_grads = clip_grads
        self.n_parents = n_parents
        self.higher_is_better = higher_is_better
        self.scheduling_threshold = scheduling_threshold


class DeepNeuralCrossover(GeneticOperator):
    def __init__(self, probability: float, population_size: int, dnc_config: DeepNeuralCrossoverConfig,
                 individual_evaluator: SimpleIndividualEvaluator, vector_creator: GAIntVectorCreator,
                 events=None):
        assert 0 < probability <= 1, "Probability must be between 0 and 1."
        assert population_size > 0, "Population size must be greater than 0."
        if not isinstance(individual_evaluator, DNCFitnessEvaluator):
            raise TypeError(
                "individual_evaluator must be a DNCFitnessEvaluator shared with "
                "the EC-KitY subpopulation"
            )

        self.individuals = None
        self.applied_individuals = None
        self.vector_creator = vector_creator
        self.individual_evaluator = individual_evaluator
        self.crossover_probability = probability
        self.dnc_wrapper = NeuralCrossoverWrapper(**dnc_config.__dict__,
                                                  get_fitness_function=self.get_fitness_from_vector)

        # all individuals are passed to the operator, this is done for optimization purposes (vectorization)
        super().__init__(probability=1.0, arity=population_size, events=events)

    def apply(self, individuals):
        population = np.array([ind.vector for ind in individuals], dtype='int32')
        self.dnc_wrapper.trained = False  # Reset training state for each application
        pairs_to_cross, crossover_masks = self.get_pairs_to_crossover(population)
        crossed_parents_pairs = self.dnc_wrapper.cross_pairs_with_fitness(pairs_to_cross)
        next_gen = []

        for child1, child2, cross_mask in zip(population[::2], population[1::2], crossover_masks):
            if cross_mask < self.crossover_probability:
                next_gen += crossed_parents_pairs.pop(0)
            else:
                next_gen += [(child1.copy(), None), (child2.copy(), None)]

        if len(population) % 2 != 0:
            next_gen.append((population[-1].copy(), None))

        for ind, (vector, fitness) in zip(individuals, next_gen):
            ind.set_vector(vector)
            if fitness is not None:
                ind.fitness.set_fitness(fitness)

        return individuals

    def get_pairs_to_crossover(self, population):
        """
        Selects pairs of individuals to perform crossover on.
        """
        pairs_to_cross = []

        crossover_masks = np.random.uniform(size=len(population) // 2)
        extended_crossover_parent_indexes = np.random.randint(low=0, high=len(population),
                                                              size=(
                                                                  len(population) // 2, self.dnc_wrapper.n_parents - 2))
        crossover_index = 0
        for child1, child2, cross_mask in zip(population[::2], population[1::2], crossover_masks):
            if cross_mask < self.crossover_probability:
                current_tuple = [child1.copy(), child2.copy()]
                if self.dnc_wrapper.n_parents > 2:
                    current_tuple += [population[extended_crossover_parent_indexes[crossover_index, i]].copy() for i in
                                      range(extended_crossover_parent_indexes.shape[1])]
                pairs_to_cross.append(current_tuple)
            crossover_index += 1

        return pairs_to_cross, crossover_masks

    def get_fitness_from_vector(self, vector):
        ind = self.vector_creator.type(
            length=self.vector_creator.length,
            bounds=self.vector_creator.bounds,
            fitness=self.vector_creator.fitness_type(
                higher_is_better=self.dnc_wrapper.higher_is_better
            ),
            update_parents=self.vector_creator.update_parents,
        )
        ind.set_vector(vector)
        return self.individual_evaluator.evaluate_individual(ind)
