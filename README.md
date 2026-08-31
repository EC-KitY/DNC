# Deep Neural Crossover for EC-KitY

`eckity-dnc` provides the Deep Neural Crossover (DNC) genetic operator for [EC-KitY](https://github.com/EC-KitY/EC-KitY).

DNC is described in **“Deep Neural Crossover: A Multi-Parent Operator That Leverages Gene Correlations”** ([paper](https://doi.org/10.1145/3638529.3654020)).

## Installation

```bash
pip install eckity-dnc
```

Installing `eckity-dnc` also installs its EC-KitY, PyTorch, NumPy, and SciPy dependencies. SciPy is declared here because EC-KitY 0.4.2 imports it without declaring it as a dependency.

## Usage

Import the public API from `eckity_dnc`:

```python
from eckity.creators import GAIntVectorCreator
from eckity_dnc import (
    DNCFitnessEvaluator,
    DeepNeuralCrossover,
    DeepNeuralCrossoverConfig,
)
```

Create the vector creator and DNC operator:

```python
population_size = 100
individual_length = 160
number_of_gene_values = 161

individual_creator = GAIntVectorCreator(
    length=individual_length,
    bounds=(0, number_of_gene_values - 1),
)

fitness_evaluator = DNCFitnessEvaluator(your_eckity_evaluator)

dnc_config = DeepNeuralCrossoverConfig(
    embedding_dim=64,
    sequence_length=individual_length,
    num_embeddings=number_of_gene_values,
    batch_size=1024,
    learning_rate=1e-4,
    use_device="cpu",
    n_parents=2,
    epsilon_greedy=0.3,
)

dnc_operator = DeepNeuralCrossover(
    probability=0.8,
    population_size=population_size,
    dnc_config=dnc_config,
    individual_evaluator=fitness_evaluator,
    vector_creator=individual_creator,
)
```

Add `dnc_operator` to the EC-KitY subpopulation's `operators_sequence`, and use the same `fitness_evaluator` instance as the subpopulation's evaluator. The wrapped evaluator must inherit from EC-KitY's `SimpleIndividualEvaluator` and evaluate vectors created by `individual_creator`. Sharing the adapter lets EC-KitY reuse fitness values that DNC computed while training instead of evaluating unchanged crossover children twice.

See [`dnc_runner_eckity.py`](dnc_runner_eckity.py) for a complete bin-packing example using tournament selection and mutation.

## Compatibility

- Python 3.10 or newer
- EC-KitY 0.4.x (tested with 0.4.2)
- PyTorch 2.7.1 or newer (tested with 2.7.1)
- NumPy 2.0.2 or newer (tested with 2.0.2)
- SciPy 1.13.0 or newer (required by EC-KitY 0.4.2)

## Development

Install the package and development tools, then run the tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## License

This project is licensed under the GNU General Public License v3.0. See [`LICENSE`](LICENSE).
