import os
import numpy as np
from typing import List
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
from numpy.core._multiarray_umath import ndarray
from scipy.sparse.csgraph import minimum_spanning_tree


class TSP_Instance_Creator:
    nPoints: int
    best_sol: float
    name: str
    lines: List[str]
    dist_matrix: ndarray
    points: ndarray

    def __init__(self, mode, name_problem=False, seed=1, dimension=False):
        self.problems = ['att532.tsp', 'fl1577.tsp', 'pr439.tsp', 'ch130.tsp', 'rat783.tsp',
                         'd198.tsp', 'kroA100.tsp', 'u1060.tsp', 'lin318.tsp',
                         'eil76.tsp', 'pcb442.tsp']

        if mode == 'random':
            self.seed = seed
            np.random.seed(self.seed)
            self.random_2D_instance(dimension=dimension)
            self.print_best = f'LB_sol: {self.LB}'

        elif mode == "standard":
            if not name_problem:
                name_problem = np.random.choice(self.problems)
            self.read_instance(name_problem)
            self.print_best = f'best_sol: {self.best_sol}'

    def read_instance(self, name_tsp):
        # read raw data
        folder = "problems/TSP/"
        if "AI" not in os.getcwd():
            folder = "AI2022MA/problems/TSP/"
        file_object = open(f"{folder}{name_tsp}")
        data = file_object.read()
        file_object.close()
        self.lines = data.splitlines()

        # store data set information
        self.name = self.lines[0].split(' ')[2]
        self.nPoints = np.int(self.lines[3].split(' ')[2])
        self.best_sol = np.float(self.lines[5].split(' ')[2])

        # read all data points and store them
        self.points = np.zeros((self.nPoints, 3))
        for i in range(self.nPoints):
            line_i = self.lines[7 + i].split(' ')
            self.points[i, 0] = int(line_i[0])
            self.points[i, 1] = line_i[1]
            self.points[i, 2] = line_i[2]

        self.create_dist_matrix()
        self.exist_opt = False
        if [name for name in ["eil76", "kroA100"] if name in name_tsp]:
            self.exist_opt = True
            file_object = open(f"{folder}{name_tsp.replace('.tsp', '.opt.tour')}")
            data = file_object.read()
            file_object.close()
            lines = data.splitlines()

            # read all data points and store them
            self.optimal_tour = np.zeros(self.nPoints, dtype=np.int)
            for i in range(self.nPoints):
                line_i = lines[5 + i].split(' ')
                self.optimal_tour[i] = int(line_i[0]) - 1

    def print_info(self):
        print('name: ' + self.name)
        print('nPoints: ' + str(self.nPoints))
        print(self.print_best)

    def plot_data(self):
        plt.figure(figsize=(8, 8))
        plt.title(self.name + " optimal")
        plt.scatter(self.points[:, 1], self.points[:, 2])
        for i, txt in enumerate(np.arange(self.nPoints)):  # tour_found[:-1]
            plt.annotate(txt, (self.points[i, 1], self.points[i, 2]))
        plt.show()

    def plot_optimal_solution(self):
        # assert self.name in ["eil76", "kroA100"], f"the solution is not available for {self.name}"
        if self.exist_opt:
            plt.figure(figsize=(8, 8))
            plt.title(self.name)
            plt.scatter(self.points[:, 1], self.points[:, 2])
            for i, txt in enumerate(np.arange(self.nPoints)):  # tour_found[:-1]
                plt.annotate(txt, (self.points[i, 1], self.points[i, 2]))

            for i, j in zip(self.optimal_tour[:-1], self.optimal_tour[1:]):
                plt.plot([self.points[i, 1], self.points[j, 1]],
                         [self.points[i, 2], self.points[j, 2]], 'b-')
            plt.plot([self.points[self.optimal_tour[0], 1], self.points[self.optimal_tour[-1], 1]],
                     [self.points[self.optimal_tour[0], 2], self.points[self.optimal_tour[-1], 2]], 'b-')
            plt.show()

    @staticmethod
    def distance_euc(zi, zj):
        xi, xj = zi[0], zj[0]
        yi, yj = zi[1], zj[1]
        return round(np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 0)

    def create_dist_matrix(self):
        self.dist_matrix = np.zeros((self.nPoints, self.nPoints))
        for i in range(self.nPoints):
            for j in range(i, self.nPoints):
                self.dist_matrix[i, j] = self.distance_euc(self.points[i][1:3], self.points[j][1:3])
        self.dist_matrix += self.dist_matrix.T

    def random_2D_instance(self, dimension=False):
        self.name = f"random {self.seed}"
        if not dimension:
            dimension = np.random.randint(100, 300)
        self.nPoints = dimension
        self.points = np.zeros((self.nPoints, 3))
        for i in range(self.nPoints):
            a, b = np.random.uniform(0, 1000, size=2)
            self.points[i, 0] = i
            self.points[i, 1] = a
            self.points[i, 2] = b

        self.create_dist_matrix()
        self.exist_opt = False
        self.LB = np.sum(create_MST(self.dist_matrix))
        return


def create_MST(dist_tensor):
    """

    @param dist_tensor:
    @return:
    """
    X = csr_matrix(dist_tensor)
    Tcsr = minimum_spanning_tree(X)
    mst = Tcsr.toarray()
    # mst += np.transpose(mst)
    return mst
