from .neural_crossover import NeuralCrossover
import torch
from numpy import stack as np_stack
from eckity.before_after_publisher import BeforeAfterPublisher

BEFORE_TRAIN_EVENT_NAME = 'before_train'
AFTER_TRAIN_EVENT_NAME = 'after_train'

class NeuralCrossoverWrapper(BeforeAfterPublisher):
    def __init__(self, embedding_dim, sequence_length, num_embeddings, get_fitness_function, running_mean_decay=0.99,
                 batch_size=32, load_weights_path=None, freeze_weights=False, learning_rate=1e-3, epsilon_greedy=0.1,
                 use_scheduler=False, use_device='cpu', adam_decay=0, clip_grads=False, n_parents=2, scheduling_threshold=0, higher_is_better = True, events=None, event_names=None):
        if scheduling_threshold < 0:
            raise ValueError("scheduling_threshold must be greater than or equal to 0")

        ext_events_names = event_names if event_names is not None else []
        if events is None:
            # Initialize events dictionary with event names as keys and subscribers as values
            ext_events_names.extend([BEFORE_TRAIN_EVENT_NAME, AFTER_TRAIN_EVENT_NAME])
        super().__init__(events, ext_events_names)
        self.device = use_device
        self.neural_crossover = NeuralCrossover(embedding_dim, embedding_dim, num_embeddings, sequence_length,
                                                n_parents=n_parents, device=use_device).to(
            self.device)
        self.running_mean_decay = running_mean_decay
        self.optimizer = torch.optim.Adam(self.neural_crossover.parameters(), lr=learning_rate, weight_decay=adam_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10)
        self.get_fitness_function = get_fitness_function
        self.batch_size = batch_size
        self.n_parents = n_parents
        self.batch_stack_fitness_values = []
        self.sampled_action_space = []
        self.sampled_solutions = []
        self.load_weights_path = load_weights_path
        self.freeze_weights = freeze_weights
        self.epsilon_greedy = epsilon_greedy
        self.use_scheduler = use_scheduler
        self.clip_grads = clip_grads
        self.acc_batch_length = 0
        self.trained = False
        self.scheduling_threshold = scheduling_threshold
        self.best_of_gen = None
        self.higher_is_better = higher_is_better

        if self.load_weights_path is not None:
            self.neural_crossover.load_state_dict(torch.load(self.load_weights_path))

    def get_batch_and_clear(self):
        """
        Returns the batch of parents and fitness values and clears the batch.
        """
        fitness_values = torch.cat(self.batch_stack_fitness_values, dim=0).unsqueeze(1).to(self.device)
        sampled_action_space = torch.cat(self.sampled_action_space, dim=0).to(self.device)
        sampled_solutions = torch.cat(self.sampled_solutions, dim=0).to(self.device)

        self.clear_stacks()

        return fitness_values, sampled_action_space, sampled_solutions

    def clear_stacks(self):
        """
        Clears the batch stacks.
        """
        self.batch_stack_fitness_values.clear()
        self.sampled_action_space.clear()
        self.sampled_solutions.clear()

    def _retain_latest_batch_window(self):
        """
        Retains the newest ``batch_size`` parent pairs and both children produced
        for each pair. This window is used only when scheduling is enabled.
        """
        if self.acc_batch_length <= self.batch_size:
            return

        stack_lengths = {
            len(self.batch_stack_fitness_values),
            len(self.sampled_action_space),
            len(self.sampled_solutions),
        }
        stack_length = len(self.batch_stack_fitness_values)
        if len(stack_lengths) != 1 or stack_length == 0 or stack_length % 2 != 0:
            raise RuntimeError("DNC training stacks must contain aligned child pairs")

        remaining_pairs = self.batch_size
        retained_fitness_values = []
        retained_action_space = []
        retained_solutions = []

        for stack_index in range(stack_length - 2, -1, -2):
            first_child_size = self.batch_stack_fitness_values[stack_index].shape[0]
            second_child_size = self.batch_stack_fitness_values[stack_index + 1].shape[0]
            if first_child_size != second_child_size:
                raise RuntimeError("DNC child batches must have matching sizes")

            pairs_to_keep = min(first_child_size, remaining_pairs)
            slice_start = first_child_size - pairs_to_keep
            retained_fitness_values[0:0] = [
                self.batch_stack_fitness_values[stack_index][slice_start:],
                self.batch_stack_fitness_values[stack_index + 1][slice_start:],
            ]
            retained_action_space[0:0] = [
                self.sampled_action_space[stack_index][slice_start:],
                self.sampled_action_space[stack_index + 1][slice_start:],
            ]
            retained_solutions[0:0] = [
                self.sampled_solutions[stack_index][slice_start:],
                self.sampled_solutions[stack_index + 1][slice_start:],
            ]
            remaining_pairs -= pairs_to_keep
            if remaining_pairs == 0:
                break

        self.batch_stack_fitness_values = retained_fitness_values
        self.sampled_action_space = retained_action_space
        self.sampled_solutions = retained_solutions
        self.acc_batch_length = self.batch_size - remaining_pairs

    def run_epoch(self):
        """
        Performs one step of training on the neural crossover.
        """
        if self.freeze_weights:
            self.clear_stacks()
            return

        if self.acc_batch_length < self.batch_size or self.acc_batch_length <= 0:
            return
        # ------ scheduling threshold ------
        if self.scheduling_threshold > 0:
            self._retain_latest_batch_window()

        best_func_torch = torch.max if self.higher_is_better else torch.min
        best_batch_fitness = best_func_torch(torch.cat(self.batch_stack_fitness_values, dim=0).unsqueeze(1))

        if self.best_of_gen is not None and self.scheduling_threshold > 0:
            if abs(best_batch_fitness - self.best_of_gen) < self.scheduling_threshold:
                return

        self.publish(BEFORE_TRAIN_EVENT_NAME)
        if(self.best_of_gen is None):
            self.best_of_gen = best_batch_fitness
        else:
            best_func = max if self.higher_is_better else min
            self.best_of_gen = best_func(best_batch_fitness, self.best_of_gen)
        # ------ scheduling threshold ------

        self.acc_batch_length = 0
        fitness_values, sampled_action_space, sampled_solutions = self.get_batch_and_clear()
        self.optimizer.zero_grad()
        sampled_solutions_proba = torch.gather(sampled_action_space, 2, sampled_solutions.unsqueeze(2)).squeeze(-1).to(
            self.device)
        loss = -torch.mean(
            torch.log(sampled_solutions_proba) * (fitness_values.type(torch.DoubleTensor)).to(self.device))

        loss.backward()

        if self.clip_grads:
            torch.nn.utils.clip_grad_norm_(self.neural_crossover.parameters(), 1.0)

        self.optimizer.step()

        if self.use_scheduler:
            self.scheduler.step(loss)

        self.publish(AFTER_TRAIN_EVENT_NAME)
        self.trained = True
        # print(f'loss: {loss}, reward: {torch.mean(fitness_values.type(torch.DoubleTensor))}')

    def combine_parents_uniform(self, parents_matrix):
        """
        Uses the neural crossover to select the crossover points from the parents.
        """
        if self.freeze_weights:
            self.neural_crossover.eval()

        parents_matrix = parents_matrix.to(self.device)

        attention_values, selected_crossovers_indices = self.neural_crossover(parents_matrix,
                                                                              epsilon_greedy=self.epsilon_greedy)
        self.sampled_action_space.append(attention_values)
        self.sampled_solutions.append(selected_crossovers_indices)
        return torch.gather(parents_matrix.permute(1, 2, 0), dim=2,
                            index=selected_crossovers_indices.unsqueeze(-1)).squeeze(-1)

    def update_batch_stack(self, fitness_values):
        """
        Updates the batch stack.
        """
        self.batch_stack_fitness_values.append(fitness_values)

    def get_crossover(self, parents_matrix):
        """
        Uses the neural crossover to select the crossover points from the parents.
        Then performs one step of training on the neural crossover.
        :param parents_matrix: parents to crossover
        :return: resulting crossover individuals
        """
        parents_matrix = torch.Tensor(parents_matrix).type(torch.LongTensor)

        selected_crossover_func = self.combine_parents_uniform

        child1 = selected_crossover_func(parents_matrix)
        child2 = selected_crossover_func(parents_matrix)

        child1_fitness_values = [self.get_fitness_function(child) for child in
                                 child1.detach().cpu().numpy()]

        child2_fitness_values = [self.get_fitness_function(child) for child in
                                 child2.detach().cpu().numpy()]

        child1_fitness_values = torch.Tensor(child1_fitness_values).type(torch.FloatTensor)
        child2_fitness_values = torch.Tensor(child2_fitness_values).type(torch.FloatTensor)

        self.update_batch_stack(child1_fitness_values)
        self.update_batch_stack(child2_fitness_values)
        self.run_epoch()

        return child1.detach().cpu().numpy().tolist(), child2.detach().cpu().numpy()

    def cross_pairs(self, parents_pairs):
        if len(parents_pairs) == 0:
            return []

        parents_grouped = list(zip(*parents_pairs))

        parents_matrix_np = np_stack(parents_grouped)
        parents_matrix = torch.from_numpy(parents_matrix_np)
        # parents_matrix = torch.cat([torch.unsqueeze(torch.tensor(group), 0) for group in parents_grouped], dim=0)

        self.acc_batch_length += parents_matrix.shape[1]
        child1, child2 = self.get_crossover(parents_matrix)
        return list(zip(child1, child2))

    def save_weights(self, path):
        torch.save(self.neural_crossover.state_dict(), path)
