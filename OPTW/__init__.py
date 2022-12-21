import numpy as np
import random
import matplotlib.pyplot as plt
import os
from os import path
import AI2022MA.OPTW.op_utils.op as u_o
import AI2022MA.OPTW.op_utils.instance as u_i


class Env:
    cl_dim: int = 64

    def __init__(self, n_nodes=50, seed=None, from_file=False, instance_number=0, verbose=False):
        self.x = None
        self.adj = None
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.sim_counter = 0
        self.name = None
        self.verbose = verbose
        if from_file:
            base_dir = path.dirname(path.dirname(path.abspath(__file__)))
            files_path = path.join(base_dir, "OPTW/instances")
            x_path = os.path.join(files_path, f"instances/instance{instance_number:04}.csv")
            adj_path = os.path.join(files_path, f"adjs/adj-instance{instance_number:04}.csv")
            self.x, self.adj, self.instance_name = u_i.read_instance(x_path, adj_path)
            self.n_nodes = len(self.x)
            t_max = int(self.x[0, 6])
            print(self.x.shape, self.adj.shape, self.instance_name, t_max, self.n_nodes)
            self.x[self.x[:, 3] > t_max - self.adj[np.arange(self.n_nodes), 0], 3] = \
                t_max - self.adj[np.arange(self.n_nodes), 0]
            self.x[self.x[:, 4] > t_max - self.adj[np.arange(self.n_nodes), 0], 4] = \
                t_max - self.adj[np.arange(self.n_nodes), 0]
        else:
            assert n_nodes is not None, 'if no file is given, n_nodes is required'
            self.n_nodes = n_nodes
            self.instance_name = ''
            self.x, self.adj = u_i.make_instance(self.n_nodes, seed=self.seed)

    # def get_features(self):
    #     tw, max_t = self._normalize_features(self.x[:, 3:5])
    #     features = np.concatenate([self._normalize_pos(self.x[:, 1:3]), tw,
    #                                np.expand_dims(np.array(self.adj[0] / max_t), axis=-1),
    #                                self.x[:, 6, None] / max_t, self.x[:, 5, None]], axis=-1)
    #     # np.expand_dims(np.array([self.n_nodes / 200 for _ in range(self.n_nodes)]),
    #     #                axis=-1)],
    #     #                           axis=-1)
    #     return features

    @staticmethod
    def _normalize_features(x):
        max_x = np.max(x)
        min_x = np.min(x)
        return (x - min_x) / (max_x - min_x), max_x

    @staticmethod
    def _normalize_pos(x):
        centro = np.mean(x, axis=0)
        new_pos = x - centro
        # print(new_pos)
        max_value = np.max(new_pos)
        # print(max_value)
        return new_pos / max_value

    def check_solution(self, sol):

        # assert len(sol) == len(self.x) + 1, 'len(sol) = ' + str(len(sol)) + ', n_nodes+1 = ' + str(len(self.x) + 1)
        # assert len(sol) == len(set(sol)) + 1
        self.sim_counter += 1
        self.name = f'tour{self.sim_counter:03}'
        tour_time, rewards, feas = u_o.tour_check(sol, self.x, self.adj, self.verbose)
        return tour_time, rewards, feas


def plot_instance(X, tour=None):
    for i in range(len(X)):
        plt.plot(X[i][1], X[i][2], "bo")
        plt.annotate(f"{int(X[i][0])}", (X[i][1], X[i][2]))
    if tour:
        for c1, c2 in zip(tour[:-1], tour[1:]):
            # print(X[[c1, c2]][1], X[c1][1], X[c2][1])
            plt.plot(X[[c1, c2], 1], X[[c1, c2], 2], "g-")

    plt.plot(X[0][1], X[0][2], "ro")
    plt.title(f"total time {X[0][6]}")
    plt.show()


if __name__ == '__main__':
    for instance in range(3):
        # env = Env(n_nodes=55, seed=1235)
        env = Env(from_file=True, instance_number=instance)
        X = np.array(env.x)
        # print(X)
        # print(env.get_features())
        # print(type(env.x))
        # print(env.adj)
        # print(np.sum(np.array(env.x), axis=0))
        sol = [0, 40, 11, 42, 0]
        # print(X.shape)
        plot_instance(X, sol)
        # ext_sol = [j for j in range(50) if j not in sol]
        # sol = sol + ext_sol
        # print(sol)
        # for _ in range(10):
        print(env.check_solution(sol))
        #     print(env.x)
