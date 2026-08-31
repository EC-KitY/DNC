import json
import os

import numpy as np
import torch

from eckity_dnc.wrapper import NeuralCrossoverWrapper
from selection_ga import SelectionGA


def get_bin_packing_fitness(individual, fitness_dict, penalty=100):
    global item_weights, bin_capacity

    if tuple(individual) in fitness_dict:
        return fitness_dict[tuple(individual)]

    fitness = 0
    bin_capacities = np.zeros(n_items)
    legal_solution = True

    for item_index, bin_index in enumerate(individual):
        bin_capacities[bin_index] += item_weights[item_index]

        if bin_capacities[bin_index] > bin_capacity:
            legal_solution = False
            fitness -= penalty

    if legal_solution:
        utilized_bins = bin_capacities[bin_capacities > 0]
        fitness = ((bin_capacities / bin_capacity) ** 2).sum() / len(utilized_bins)

    fitness_dict[tuple(individual)] = fitness
    return fitness


def main():
    global item_weights, bin_capacity, n_items

    with open('./datasets/hard_parsed.json', 'r') as dataset_file:
        datasets_json = json.load(dataset_file)
    path_to_exp = './experiments/bin_packing/DNC'
    dataset_name = 'BPP_14'
    item_weights = np.array(datasets_json[dataset_name]['items'])
    bin_capacity = datasets_json[dataset_name]['max_bin_weight']
    n_items = len(item_weights)
    n_parents = 2
    print(dataset_name, n_items)

    output_folder = os.path.join(path_to_exp, dataset_name)
    os.makedirs(output_folder, exist_ok=True)

    params_dict = {
        'n_generations': 6000,
        'population_size': 100,
        'crossover_prob': 0.5,
        'mutation_prob': 0.5,
        'ind_length': n_items,
        'save_every_n_generations': 5,
        'min_selection_val': 0,
        'max_selection_val': n_items - 1,
        'flip_mutation_prob': 0.1,
        'tournament_size': 5,
        'save_population_info': False,
        'save_fitness_info': False,
        'elitism': False,
        'n_parents': n_parents,
    }

    torch.manual_seed(4242)
    ga_class = SelectionGA(**params_dict, random_state=42)
    ncs = NeuralCrossoverWrapper(
        embedding_dim=64,
        sequence_length=n_items,
        num_embeddings=180 + 1,
        get_fitness_function=lambda ind: get_bin_packing_fitness(
            ind, ga_class.fitness_dict
        ),
        batch_size=2048,
        freeze_weights=True,
        load_weights_path=None,
        learning_rate=1e-4,
        epsilon_greedy=0.3,
        use_scheduler=False,
        use_device='cpu',
        n_parents=n_parents,
    )
    ga_class.fit(output_folder, get_bin_packing_fitness, crossover_func=ncs.cross_pairs)


if __name__ == '__main__':
    main()
