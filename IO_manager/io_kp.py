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


class KP_Instance_Creator:
    nItems: int
    distribution: str
    capacity: int
    volume_items: ndarray
    profit_items: ndarray
    existing_distributions = distributions

    def __init__(self, mode, seed=1, dimension=50):
        print(mode)
        self.seed_ = seed
        np.random.seed(self.seed_)
        self.nItems = dimension
        if mode == "random":
            self.my_random(dimension=dimension)
        else:
            self.read_data(mode)
        self.distribution = mode

    def read_data(self, name_type):
        assert name_type in self.existing_distributions, f"the distribution {name_type} does not exits"
        folder = "problems/KP/"
        if "AI" not in os.getcwd():
            folder = "AI2021MA/problems/KP/"
        files_distr = [file_ for file_ in os.listdir(folder) if name_type in file_]
        # print(files_distr)
        file_object = np.random.choice(files_distr, 1)[0]
        # print(f"{folder}{file_object}")
        file_object = open(f"{folder}{file_object}")
        data = file_object.read()
        file_object.close()
        lines = data.splitlines()

        self.nItems = int(lines[0])
        self.capacity = int(lines[1])

        self.volume_items = np.zeros(self.nItems, np.int)
        self.profit_items = np.zeros(self.nItems, np.int)
        for i in range(self.nItems):
            line_i = lines[3 + i].split(' ')
            self.profit_items[i] = int(line_i[0])
            self.volume_items[i] = int(line_i[1])
        if name_type in ["inverse_strongly_correlated",
                         "inverse_weakly_correlated",
                         "multiple_inverse_strongly_correlated"]:
            max_volume = np.max(self.volume_items)
            self.volume_items = max_volume - self.volume_items

        if name_type == "circle":
            ray = (np.max(self.volume_items) - np.min(self.volume_items)) / 2
            # ray_2 = (np.max(self.profit_items) - np.min(self.profit_items)) / 2
            # # ray = np.max([ray_1, ray_2])
            # ray = ray_1
            centre_a = np.median(self.volume_items)
            centre_b = np.median(self.profit_items)
            # print(ray, centre_a, centre_b)
            tot_el = self.volume_items.shape[0]
            new_profit = np.zeros(tot_el * 2)
            new_volume = np.zeros(tot_el * 2)
            for el in range(tot_el):
                x = self.volume_items[el]
                up = x >= centre_a
                delta_ = np.abs(ray ** 2 - (x - centre_a) ** 2)
                new_volume[el] = (centre_b + np.sqrt(delta_)) / 50
                new_volume[el + tot_el] = (centre_b - np.sqrt(delta_)) / 50
                new_profit[el] = self.profit_items[el]
                new_profit[el + tot_el] = self.profit_items[el]
            self.profit_items = new_profit
            self.volume_items = new_volume

    def my_random(self, dimension=50):
        mean = [300, 400]
        cov = [[8, 100], [100, 13]]
        features, true_labels = make_blobs(n_samples=dimension,
                                           centers=3,
                                           cluster_std=1.75,
                                           random_state=42)
        max_value = np.max(np.abs(features)) + 0.1
        self.volume_items, self.profit_items = np.round(np.array(features[:, 0] + max_value)), \
                                               np.round(np.array(features[:, 1] + max_value))
        # self.volume_items, self.profit_items = np.random.multivariate_normal(mean, cov , dimension).astype(np.int).T
        num_items_prob = np.random.choice(np.arange(1, dimension // 2), 1)[0]
        self.capacity = int(np.mean(self.volume_items) * num_items_prob)

    def plot_data_scatter(self):
        plt.figure(figsize=(8, 8))
        plt.title(self.distribution)
        plt.scatter(self.profit_items, self.volume_items)
        plt.xlabel("profit values")
        plt.ylabel("volume values")
        # for i in range(self.nItems):  # tour_found[:-1]
        #     plt.annotate(i, (self.profit_items[i], self.volume_items[i]))

        plt.show()

    def plot_data_distribution(self):
        greedy_sort_vol = np.argsort(self.volume_items)
        greedy_sort_profits = np.argsort(self.profit_items)
        volume_plot = normalize(self.volume_items, index_sort=greedy_sort_vol)
        profit_plot = normalize(self.profit_items, index_sort=greedy_sort_profits)
        cum_volume = np.cumsum(volume_plot)
        # print(self.capacity, cum_volume)
        arg_where = np.where(cum_volume >= self.capacity)[0][0]
        capacity_plot = arg_where / len(self.volume_items)
        # print(f"collected {capacity_plot * 100}% of the volume")
        print(volume_plot)
        print(profit_plot)
        plt.hist(volume_plot, 50, density=True, histtype='step',
                 cumulative=True, label='volume comulative', color='blue')
        plt.hist(profit_plot, 50, density=True, histtype='step',
                 cumulative=True, label='profit comulative', color='green')
        plt.plot(np.linspace(0, 1, 10), np.ones(10) * capacity_plot, color='orange')
        plt.legend()
        plt.show()

    def plot_solution(self, solution):
        plt.figure(figsize=(8, 8))
        plt.title(self.distribution)
        plt.scatter(self.profit_items, self.volume_items)
        plt.scatter(self.profit_items[solution],
                    self.volume_items[solution], c="red")
        plt.xlabel("profit values")
        plt.ylabel("volume values")
        plt.show()


def normalize(array_, index_sort):
    return (np.max(array_) - array_[index_sort]) / (np.max(array_) - np.min(array_))
