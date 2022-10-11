import os
import numpy as np
from numpy.core._multiarray_umath import ndarray
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

distributions = ["uncorrelated",
                 "weakly_correlated",
                 "strongly_correlated",
                 "inverse_weakly_correlated",
                 "inverse_strongly_correlated",
                 # "subset_sum",
                 "multiple_strongly_correlated",
                 "multiple_inverse_strongly_correlated",
                 # "profit_ceiling",
                 "circle"]


# Creating a Knapsack problem instance.
# > This class creates a new instance of the Knapsack Problem
class KnapsackInstanceCreator:
    n_items: int
    distribution: str
    capacity: int
    item_weights: ndarray
    item_profits: ndarray
    existing_distributions = distributions

    def __init__(self, mode, seed=1, dimension=50):
        """
        The constructor of the class.

        :param mode: This is the mode of the model. It can be either 'train' or 'test'
        :param seed: The random seed used to initialize the random number generator, defaults to 1 (optional)
        :param dimension: The dimension of the embedding space, defaults to 50 (optional)
        """
        # print(mode)
        self.seed_ = seed
        np.random.seed(self.seed_)
        self.n_items = dimension
        if mode == "random":
            self.my_random(dimension=dimension)
        else:
            self.read_data(mode)
        self.distribution = mode

    def read_data(self, name_type):
        """
        This function reads in the data from the file and returns a list of lists.

        :param name_type: The name of the data type you want to read
        """
        assert name_type in self.existing_distributions, f"the distribution {name_type} does not exits"
        folder = "problems/KP/"

        if "AI" not in os.getcwd():
            folder = "AI2022MA/problems/KP/"

        files_distr = [file_ for file_ in os.listdir(folder) if name_type in file_]
        # print(files_distr)
        file_object = np.random.choice(files_distr, 1)[0]
        # print(f"{folder}{file_object}")
        file_object = open(f"{folder}{file_object}")
        data = file_object.read()
        file_object.close()
        lines = data.splitlines()

        self.n_items = int(lines[0])
        self.capacity = int(lines[1])

        self.item_weights = np.zeros(self.n_items, np.int)
        self.item_profits = np.zeros(self.n_items, np.int)
        for i in range(self.n_items):
            line_i = lines[3 + i].split(' ')
            self.item_profits[i] = int(line_i[0])
            self.item_weights[i] = int(line_i[1])
        if name_type in ["inverse_strongly_correlated",
                         "inverse_weakly_correlated",
                         "multiple_inverse_strongly_correlated"]:
            max_weight = np.max(self.item_weights)
            self.item_weights = max_weight - self.item_weights

        if name_type == "circle":
            ray = (np.max(self.item_weights) - np.min(self.item_weights)) / 2
            # ray_2 = (np.max(self.item_profits) - np.min(self.item_profits)) / 2
            # # ray = np.max([ray_1, ray_2])
            # ray = ray_1
            centre_a = np.median(self.item_weights)
            centre_b = np.median(self.item_profits)
            # print(ray, centre_a, centre_b)
            tot_el = self.item_weights.shape[0]
            new_profit = np.zeros(tot_el * 2)
            new_weight = np.zeros(tot_el * 2)
            for el in range(tot_el):
                x = self.item_weights[el]
                up = x >= centre_a
                delta_ = np.abs(ray ** 2 - (x - centre_a) ** 2)
                new_weight[el] = (centre_b + np.sqrt(delta_)) / 50
                new_weight[el + tot_el] = (centre_b - np.sqrt(delta_)) / 50
                new_profit[el] = self.item_profits[el]
                new_profit[el + tot_el] = self.item_profits[el]
            self.item_profits = new_profit
            self.item_weights = new_weight

    def my_random(self, dimension=50):
        """
        It generates a random number between 0 and 1.

        :param dimension: The number of dimensions of the vector space, defaults to 50 (optional)
        """
        mean = [300, 400]
        cov = [[8, 100], [100, 13]]
        features, true_labels = make_blobs(n_samples=dimension,
                                           centers=3,
                                           cluster_std=1.75,
                                           random_state=43)
        max_value = np.max(np.abs(features)) + 0.1
        self.item_weights, self.item_profits = np.round(np.array(features[:, 0] + max_value)), \
                                               np.round(np.array(features[:, 1] + max_value))
        # self.item_weights, self.item_profits = np.random.multivariate_normal(mean, cov , dimension).astype(np.int).T
        num_items_prob = np.random.choice(np.arange(1, dimension // 2), 1)[0]
        self.capacity = int(np.mean(self.item_weights) * num_items_prob)

    def plot_data_scatter(self):
        """
        This function plots the data points in the scatter plot.
        """
        plt.figure(figsize=(8, 8))
        plt.title(self.distribution)
        plt.scatter(self.item_profits, self.item_weights)
        plt.xlabel("profit values")
        plt.ylabel("weight values")
        # for i in range(self.n_items):  # tour_found[:-1]
        #     plt.annotate(i, (self.item_profits[i], self.item_weights[i]))

        plt.show()

    def plot_data_distribution(self):
        """
        It plots the cumulative distribution of the weight and profit of the items,
        and shows the percentage of the weight that can be collected with the given capacity
        """
        preferability = self.item_profits / (self.item_weights + 1e-16)
        greedy_sort = np.argsort(preferability)
        # greedy_sort_profits = np.argsort(self.item_profits)
        weight_plot = normalize(self.item_weights, index_sort=greedy_sort)
        profit_plot = normalize(self.item_profits, index_sort=greedy_sort)
        cum_weight = np.cumsum(self.item_weights[greedy_sort])
        # print(weight_plot)
        # print(profit_plot)
        # print(self.capacity, cum_weight)
        arg_where = np.where(cum_weight >= self.capacity)[0][0]
        # print(arg_where)
        capacity_plot = arg_where / len(self.item_weights)
        # print(f"collected {capacity_plot * 100}% of the weight")
        plt.figure(figsize=(8, 8))
        plt.hist(weight_plot, 50, density=True, histtype='step',
                 cumulative=True, label='weight cumulative', color='blue')
        plt.hist(profit_plot, 50, density=True, histtype='step',
                 cumulative=True, label='profit cumulative', color='green')
        plt.plot(np.linspace(0, 1, 10), np.ones(10) * capacity_plot, color='orange')
        plt.legend()
        plt.show()

    def plot_solution(self, solution):
        """
        It plots the solution to the problem

        :param solution: a list of lists, where each list is a list of the indices of the nodes in the order they are
        visited
        """
        plt.figure(figsize=(8, 8))
        plt.title(self.distribution)
        plt.scatter(self.item_profits, self.item_weights)
        plt.scatter(self.item_profits[solution],
                    self.item_weights[solution], c="red")
        plt.xlabel("profit values")
        plt.ylabel("weight values")
        plt.show()


def normalize(array_, index_sort):
    """
    It takes an array and an index, and returns the array sorted by the index

    :param array_: the array to be normalized
    :param index_sort: the index of the column to sort by
    """
    return (np.max(array_) - array_[index_sort]) / (np.max(array_) - np.min(array_))
