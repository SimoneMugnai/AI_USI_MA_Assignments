import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

random_matrix = {}


def DJ1(x):
    return np.sum(x ** 2)


def DJ2(x):
    return 100 * ((x[0] ** 2 - x[1]) ** 2) + ((1 - x[0]) ** 2)


def DJ3(x):
    return np.sum(np.ceil(x)) + 24


def DJ4(x):
    if str(list(x)) in random_matrix.keys():
        r = random_matrix[str(list(x))]
    else:
        r = np.random.randn()
        random_matrix[str(list(x))] = r
        random_matrix[str(list(x[::-1]))] = r

    return np.sum(x ** 4) + r


a_up = np.tile([-32, -16, 0, 16, 32], 5)
a_down = np.transpose(np.tile([-32, -16, 0, 16, 32], (5, 1))).flatten()
a = np.stack([a_up, a_down])


def DJ5(x):
    d = 0.002
    for i in range(25):
        d += 1 / (i + 1 + (x[0] - a[0][i]) ** 6 + (x[1] - a[1][i]) ** 6)
    return 1 / d


class De_Jong:
    dimension: int
    function_name: str
    range: (float, float)
    resolution_factor: float
    num_bits: int
    funcs = {1: ('De Jong 1', DJ1, 5.12, 0.01, 3),
             2: ('De Jong 2', DJ2, 2.048, 0.001, 2),
             3: ('De Jong 3', DJ3, 5.12, 0.01, 4),
             4: ('De Jong 4', DJ4, 1.28, 0.01, 30),
             5: ('De Jong 5', DJ5, 65.536, 0.001, 2)}

    def __init__(self, func_number, dimension_in=False):
        assert isinstance(func_number, int), f'func_number needs to be an integer!!!'
        assert func_number < 6 and func_number > 0, 'func_number needs to be a number between 1 and 5'
        self.function_name, self.fun, min_, self.resolution_factor, self.dimension = self.funcs[func_number]
        self.digits_after = len(str(self.resolution_factor)[2:])
        self.range = (-min_, min_ - self.resolution_factor)
        self.num_bits = int(np.log2(int(2 * (min_ / self.resolution_factor))))
        if dimension_in:
            self.dimension = dimension_in

    def decode_(self, x):
        return x

    def evaluate(self, x_e, gray_=False):
        assert len(x_e[0]) == self.dimension, 'the dimension does not match with the problem'
        if gray_:
            self.decode = self.gray_decode
        else:
            self.decode = self.decode_
        fitness_pop_list = []
        for i in range(len(x_e)):
            pos = []
            for dim in range(self.dimension):
                pos.append(self.decode(x_e[i][dim]))
            fitness_pop_list.append(self.fun(np.array(pos)))
        return np.array(fitness_pop_list)

    def gray_encode(self, n_f):
        scale = int(1 / self.resolution_factor)
        n = int(n_f * scale - self.range[0] * scale)
        val = n ^ n >> 1
        r_val = f"{val:>b}"
        pad = "0" * (self.num_bits - len(r_val))
        return pad + r_val

    def gray_decode(self, n_s):
        n = int(n_s, 2)
        m = n >> 1
        while m:
            n ^= m
            m >>= 1
        n_f = np.around(self.range[0] + self.resolution_factor * n, self.digits_after)
        return n_f

    def plot(self):
        samples = int(1 / self.resolution_factor)
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        x = np.linspace(self.range[0], self.range[1], samples)
        y = np.linspace(self.range[0], self.range[1], samples)

        X, Y = np.meshgrid(x, y)
        z_ = np.stack([X.flatten(), Y.flatten()], axis=-1)
        Z_ = np.apply_along_axis(self.fun, -1, z_)
        Z = np.resize(Z_, (samples, samples))

        # ax.plot_wireframe(X, Y, Z, color='green')
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        cset = ax.contour(X, Y, Z, zdir='z', cmap=cm.coolwarm)
        cset = ax.contour(X, Y, Z, zdir='x', cmap=cm.coolwarm)
        cset = ax.contour(X, Y, Z, zdir='y', cmap=cm.coolwarm)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y) ')

        plt.show()
