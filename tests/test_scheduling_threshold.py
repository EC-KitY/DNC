import pytest
import torch

from eckity_dnc import DeepNeuralCrossoverConfig
from eckity_dnc.wrapper import BEFORE_TRAIN_EVENT_NAME, NeuralCrossoverWrapper


def make_config(**overrides):
    values = {
        "embedding_dim": 2,
        "sequence_length": 1,
        "num_embeddings": 2,
    }
    values.update(overrides)
    return DeepNeuralCrossoverConfig(**values)


def make_wrapper(**overrides):
    values = {
        "embedding_dim": 2,
        "sequence_length": 1,
        "num_embeddings": 2,
        "get_fitness_function": lambda vector: float(sum(vector)),
        "batch_size": 1,
    }
    values.update(overrides)
    return NeuralCrossoverWrapper(**values)


def stage_child_pair(wrapper, first_child_fitness, second_child_fitness):
    assert len(first_child_fitness) == len(second_child_fitness)
    pair_count = len(first_child_fitness)
    wrapper.acc_batch_length += pair_count

    for fitness_values in (first_child_fitness, second_child_fitness):
        wrapper.batch_stack_fitness_values.append(
            torch.tensor(fitness_values, dtype=torch.float32)
        )
        wrapper.sampled_action_space.append(
            torch.full(
                (pair_count, 1, 2),
                0.5,
                dtype=torch.float32,
                requires_grad=True,
            )
        )
        wrapper.sampled_solutions.append(
            torch.zeros((pair_count, 1), dtype=torch.long)
        )


def capture_training_fitness(wrapper):
    observed_fitness = []
    original_get_batch_and_clear = wrapper.get_batch_and_clear

    def capture_batch():
        batch = original_get_batch_and_clear()
        observed_fitness.append(batch[0].flatten().tolist())
        return batch

    wrapper.get_batch_and_clear = capture_batch
    return observed_fitness


def register_before_train(wrapper):
    before_train_calls = []
    wrapper.register(
        BEFORE_TRAIN_EVENT_NAME,
        lambda publisher, event: before_train_calls.append(event),
    )
    return before_train_calls


def test_scheduling_threshold_defaults_to_zero():
    assert make_config().scheduling_threshold == 0


def test_negative_scheduling_threshold_is_rejected_by_config():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        make_config(scheduling_threshold=-0.1)


def test_negative_scheduling_threshold_is_rejected_by_wrapper():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        make_wrapper(scheduling_threshold=-0.1)


def test_zero_threshold_never_suppresses_training():
    wrapper = make_wrapper(scheduling_threshold=0)
    wrapper.best_of_gen = torch.tensor(10.0)
    before_train_calls = register_before_train(wrapper)
    stage_child_pair(wrapper, [10.0], [10.0])

    wrapper.run_epoch()

    assert wrapper.trained
    assert len(before_train_calls) == 1


def test_zero_threshold_preserves_a_complete_overfull_batch():
    wrapper = make_wrapper(batch_size=2, scheduling_threshold=0)
    stage_child_pair(wrapper, [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
    observed_fitness = capture_training_fitness(wrapper)

    wrapper.run_epoch()

    assert observed_fitness == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    assert wrapper.trained


def test_positive_threshold_retains_the_latest_pair_preserving_window():
    wrapper = make_wrapper(batch_size=4, scheduling_threshold=0.1)
    stage_child_pair(wrapper, [1.0, 2.0], [11.0, 12.0])
    stage_child_pair(wrapper, [3.0, 4.0, 5.0], [13.0, 14.0, 15.0])
    observed_fitness = capture_training_fitness(wrapper)

    wrapper.run_epoch()

    assert observed_fitness == [[2.0, 12.0, 3.0, 4.0, 5.0, 13.0, 14.0, 15.0]]
    assert wrapper.trained


def test_positive_threshold_handles_a_single_overfull_batch():
    wrapper = make_wrapper(batch_size=2, scheduling_threshold=0.1)
    stage_child_pair(wrapper, list(range(10)), list(range(10, 20)))
    observed_fitness = capture_training_fitness(wrapper)

    wrapper.run_epoch()

    assert observed_fitness == [[8.0, 9.0, 18.0, 19.0]]
    assert wrapper.trained


@pytest.mark.parametrize(
    ("higher_is_better", "best_of_gen", "new_fitness"),
    [
        (True, 10.0, [10.25, 10.5]),
        (False, 10.0, [9.75, 9.5]),
    ],
)
def test_positive_threshold_suppresses_small_absolute_changes(
    higher_is_better, best_of_gen, new_fitness
):
    wrapper = make_wrapper(
        scheduling_threshold=1.0,
        higher_is_better=higher_is_better,
    )
    wrapper.best_of_gen = torch.tensor(best_of_gen)
    before_train_calls = register_before_train(wrapper)
    stage_child_pair(wrapper, [new_fitness[0]], [new_fitness[1]])

    wrapper.run_epoch()

    assert not before_train_calls
    assert not wrapper.trained


def training_decision_for_scale(scale):
    wrapper = make_wrapper(scheduling_threshold=0.1)
    wrapper.best_of_gen = torch.tensor(1.0 * scale)
    before_train_calls = register_before_train(wrapper)
    stage_child_pair(wrapper, [0.0 * scale], [1.05 * scale])

    wrapper.run_epoch()

    return bool(before_train_calls)


def test_threshold_remains_absolute_in_fitness_units():
    assert not training_decision_for_scale(1.0)
    assert training_decision_for_scale(100.0)
