from numpy.core._multiarray_umath import ndarray
import os
from time import time as t
import matplotlib.pyplot as plt

if 'AI' in os.getcwd():
    from solvers.constructive_algorithms import *
    from solvers.local_search import *
    from solvers.two_opt_with_candidate import twoOpt_with_cl
else:
    from AI2022MA.solvers.constructive_algorithms import *
    from AI2022MA.solvers.local_search import *
    from AI2022MA.solvers.two_opt_with_candidate import twoOpt_with_cl


class Solver_TSP:
    solution: ndarray
    found_length: float
    available_initializers = {"random": Random_Initializer.random_method,
                              "nearest_neighbors": Nearest_Neighbor.nn,
                              # "best_nn": nearest_neighbor.best_nn,
                              "multi_fragment": Multi_Fragment.mf
                              }

    available_improvements = {"2-opt": TwoOpt.local_search,
                              "2.5-opt": TwoDotFiveOpt.local_search,
                              # "2-opt_cl": twoOpt_with_cl
                              }

    def __init__(self, initializer, seed_=0, stop_run_after=180):
        self.initializer = initializer[0]
        self.methods_name = [initializer[0]]
        self.methods = [initializer[1]]
        self.name_method = "initialized with " + self.initializer
        self.solved = False
        self.seed = seed_
        self.max_time = stop_run_after

    def bind(self, local_or_meta):
        self.methods.append(local_or_meta[1])
        self.methods_name.append(local_or_meta[0])
        self.name_method += ", improved with " + local_or_meta[0]

    def pop(self):
        self.methods.pop()
        self.name_method = self.name_method[::-1][self.name_method[::-1].find("improved"[::-1]) + len("improved") + 2:][
                           ::-1]

    def __call__(self, instance_, verbose=False, return_value=False):
        self.instance = instance_
        self.solved = False
        self.ls_calls = 0
        if self.seed > 0:
            np.random.seed(self.seed)
        if verbose:
            print(f"###  solving with {self.methods} ####")
        start = t()
        self.solution, self.found_length = self.methods[0](instance_.dist_matrix)
        for i in range(1, len(self.methods)):
            for data_ret in self.methods[i](self.solution, self.found_length, self.instance.dist_matrix):
                # self.solution, new_len, ls = data_ret
                self.solution, self.found_length, ls, end_condition = data_ret
                if t() - start > self.max_time or end_condition:
                    break
            self.ls_calls += ls
        end = t()
        self.time_to_solve = np.around(end - start, 3)
        self.solved = True
        self._gap()
        if verbose:
            print(f"###  solution found with {self.gap} % gap  in {self.time_to_solve} seconds ####")
            print(f"the total length for the solution found is {self.found_length}",
                  f"while the optimal length is {self.instance.best_sol}",
                  f"the gap is {self.gap}%",
                  f"the number of LS calls are {self.ls_calls}",
                  f"the solution is found in {self.time_to_solve} seconds", sep="\n")

        if return_value:
            return self.solution

    def plot_solution(self):
        assert self.solved, "You can't plot the solution, you need to solve it first!"
        plt.figure(figsize=(8, 8))
        self._gap()
        plt.title(f"{self.instance.name} solved with {self.name_method} solver, gap {self.gap}")
        ordered_points = self.instance.points[np.hstack([self.solution, self.solution[0]])]
        plt.plot(ordered_points[:, 1], ordered_points[:, 2], 'b-')
        for i, txt in enumerate(np.arange(self.instance.nPoints)):  # tour_found[:-1]
            plt.annotate(txt, (self.instance.points[i, 1], self.instance.points[i, 2]))
        plt.show()

    def check_if_solution_is_valid(self, solution):
        rights_values = np.sum([self.check_validation(i, solution) for i in np.arange(self.instance.nPoints)])

        if rights_values == self.instance.nPoints:
            return True
        else:
            return False

    def check_validation(self, node, solution):
        if np.sum(solution == node) == 1:
            return 1
        else:
            return 0

    def evaluate_solution(self, return_value=False):
        total_length = 0
        starting_node = self.solution[0]
        from_node = starting_node
        for node in self.solution[1:]:
            total_length += self.instance.dist_matrix[from_node, node]
            from_node = node

        total_length += self.instance.dist_matrix[from_node, starting_node]
        self.found_length = total_length
        if return_value:
            return total_length

    def _gap(self):
        self.evaluate_solution(return_value=False)
        self.gap = np.round(((self.found_length - self.instance.best_sol) / self.instance.best_sol) * 100, 2)
