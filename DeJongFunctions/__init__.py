import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def de_jong_1(x):
    """
    It takes a vector of numbers and returns the sum of the squares of those numbers.

    :param x: A numpy array of size (n,).
    :return:  The sum of the squares of the elements of x.
    """
    return np.sum(x ** 2)


def de_jong_2(x):
    """
    It's a paraboloid with a minimum at (1, 1).

    :param x: The input vector.
    :return:  The value of the function at the point x.
    """
    return 100 * ((x[0] ** 2 - x[1]) ** 2) + ((1 - x[0]) ** 2)


def de_jong_3(x):
    """
    It returns the sum of the ceiling of each element in the input vector, plus 24.

    :param x: A numpy array of size 2.
    :return:  The sum of the ceiling of each element in the array.
    """
    return np.sum(np.ceil(x)) + 24


def de_jong_4(x):
    """
    It returns the sum of the fourth powers of the elements of the input vector, plus a random number.

    :param x: The input vector.
    :return:  The sum of the elements of x to the power of 4, plus a random number.
    """
    random_matrix = {}
    if str(list(x)) in random_matrix.keys():
        r = random_matrix[str(list(x))]
    else:
        r = np.random.randn()
        random_matrix[str(list(x))] = r
        random_matrix[str(list(x[::-1]))] = r

    return np.sum(x ** 4) + r


def de_jong_5(x):
    """
    It's a sum of 25 terms, each of which is the inverse of the sum of the square of the distance between the input and
    a point in a 5x5 grid, plus the inverse of the term's index.

    :param x: The input vector.
    :return:  The value of the function at the point x.
    """

    a_up = np.tile([-32, -16, 0, 16, 32], 5)
    a_down = np.transpose(np.tile([-32, -16, 0, 16, 32], (5, 1))).flatten()
    a = np.stack([a_up, a_down])

    d = 0.002
    for i in range(25):
        d += 1 / (i + 1 + (x[0] - a[0][i]) ** 6 + (x[1] - a[1][i]) ** 6)
    return 1 / d


# It's a class that defines a function that takes in a vector of length 2 and returns a scalar
class DeJong:
    dimension: int
    function_name: str
    range: (float, float)
    resolution_factor: float
    num_bits: int
    funcs = {1: ('De Jong 1', de_jong_1, 5.12, 0.01, 3),
             2: ('De Jong 2', de_jong_2, 2.048, 0.001, 2),
             3: ('De Jong 3', de_jong_3, 5.12, 0.01, 4),
             4: ('De Jong 4', de_jong_4, 1.28, 0.01, 30),
             5: ('De Jong 5', de_jong_5, 65.536, 0.001, 2)}

    def __init__(self, func_number, dimension_in=False):
        """
        The function takes in a function number and an optional dimension. It then sets the function name, function,
        minimum value, resolution factor, and dimension. It also sets the number of digits after the decimal point,
        the range, and the number of bits.

        :param func_number:  The number of the function you want to use.
        :param dimension_in: The dimension of the function. If not specified, it will be set to the default value,
                             defaults to False (optional).
        """
        assert isinstance(func_number, int), f'func_number needs to be an integer!!!'
        assert 6 > func_number > 0, 'func_number needs to be a number between 1 and 5'
        self.function_name, self.fun, min_, self.resolution_factor, self.dimension = self.funcs[func_number]
        self.digits_after = len(str(self.resolution_factor)[2:])
        self.range = (-min_, min_)
        self.num_bits = int(np.log2(int(2 * (min_ / self.resolution_factor))))
        if dimension_in:
            self.dimension = dimension_in

    def decode_(self, x):
        """
        The function takes in a string and returns the string.

        :param x: The input to the encoder.
        :return:  The input x is being returned.
        """
        return x

    def evaluate(self, x_e, gray_=False):
        """
        The function takes in a list of binary strings, and returns a list of fitness values.

        :param x_e:   The encoded population.
        :param gray_: Whether to use gray code or not, defaults to False (optional).
        :return:      The fitness of the population.
        """
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
        """
        The function takes a float value and converts it to a binary string.

        :param n_f: The number to encode.
        :return:    The gray code of the number.
        """
        scale = int(1 / self.resolution_factor)
        n = int(np.round(n_f * scale - self.range[0] * scale))
        val = n ^ n >> 1
        r_val = f"{val:>b}"
        pad = "0" * (self.num_bits - len(r_val))
        return pad + r_val

    def gray_decode(self, n_s):
        """
        The function takes a binary string as input, converts it to an integer, and then uses the bitwise XOR operator
        to convert the integer to its Gray code equivalent.

        :param n_s: The binary string of the gray code.
        :return:    The decoded value of the gray code.
        """
        n = int(n_s, 2)
        m = n >> 1
        while m:
            n ^= m
            m >>= 1
        n_f = np.around(self.range[0] + self.resolution_factor * n, self.digits_after)
        return n_f

    def plot(self):
        """
        We create a 3D plot of the function, with the x and y axes being the input variables and the z axis being the
        output of the function.
        """
        samples = int(1 / self.resolution_factor)
        plt.figure()
        ax = plt.axes(projection="3d")

        x = np.linspace(self.range[0], self.range[1], samples)
        y = np.linspace(self.range[0], self.range[1], samples)

        xx, yy = np.meshgrid(x, y)
        z = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        zz = np.apply_along_axis(self.fun, -1, z)
        zz = np.resize(zz, (samples, samples))

        # ax.plot_wireframe(X, Y, Z, color='green')
        ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.contour(xx, yy, zz, zdir='z', cmap=cm.coolwarm)
        ax.contour(xx, yy, zz, zdir='x', cmap=cm.coolwarm)
        ax.contour(xx, yy, zz, zdir='y', cmap=cm.coolwarm)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y) ')

        plt.show()
