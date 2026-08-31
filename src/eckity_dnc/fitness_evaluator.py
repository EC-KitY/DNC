from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator


class DNCFitnessEvaluator(SimpleIndividualEvaluator):
    """Reuse fitness values already computed by the DNC operator."""

    def __init__(self, evaluator: SimpleIndividualEvaluator):
        super().__init__()
        if not isinstance(evaluator, SimpleIndividualEvaluator):
            raise TypeError("evaluator must be a SimpleIndividualEvaluator")
        self.evaluator = evaluator

    def evaluate_individual(self, individual):
        if individual.fitness.is_fitness_evaluated():
            return individual.fitness.get_pure_fitness()
        return self.evaluator.evaluate_individual(individual)
