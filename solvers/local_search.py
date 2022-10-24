import numpy as np

class TwoOpt:

    @staticmethod
    def step2opt(solution, matrix_dist, distance):
        seq_length = len(solution)
        tsp_sequence = np.array(solution)
        uncrosses = 0
        for i in range(1, seq_length - 2):
            for j in range(i, seq_length - 2):
                new_distance = distance + TwoOpt.gain(i - 1, j, tsp_sequence, matrix_dist)
                if new_distance < distance:
                    uncrosses += 1
                    new_tsp_sequence = TwoOpt.swap2opt(tsp_sequence, i - 1, j)
                    tsp_sequence = np.copy(new_tsp_sequence)
                    # print(new_distance)
                    # print(tsp_sequence)
                    distance = new_distance
                    return tsp_sequence, distance, 1
        return tsp_sequence, distance, uncrosses

    @staticmethod
    def swap2opt(tsp_sequence, i, j):
        new_tsp_sequence = np.copy(tsp_sequence)
        final_index = j + 1  # if j+1 < len(tsp_sequence) else -1
        new_tsp_sequence[i:final_index] = np.flip(tsp_sequence[i:final_index], axis=0)  # flip or swap ?
        return new_tsp_sequence

    @staticmethod
    def gain(i, j, tsp_sequence, matrix_dist):
        old_link_len = (matrix_dist[tsp_sequence[i], tsp_sequence[i - 1]] + matrix_dist[
            tsp_sequence[j], tsp_sequence[j + 1]])
        changed_links_len = (matrix_dist[tsp_sequence[j], tsp_sequence[i - 1]] + matrix_dist[
            tsp_sequence[i], tsp_sequence[j + 1]])
        return - old_link_len + changed_links_len

    @staticmethod
    def local_search(solution, actual_len, matrix_dist):
        new_tsp_sequence = np.copy(np.array(solution))
        uncross = 0
        while True:
            # input()
            new_tsp_sequence, new_reward, uncr_ = TwoOpt.step2opt(new_tsp_sequence, matrix_dist, actual_len)
            new_tsp_sequence = np.roll(new_tsp_sequence, np.random.randint(len(new_tsp_sequence)))
            uncross += uncr_
            if new_reward < actual_len:
                actual_len = new_reward
                yield new_tsp_sequence, actual_len, 0, False
            else:
                return new_tsp_sequence, actual_len, 1, True


def twoOpt(solution, actual_len, matrix_dist):
    for data in TwoOpt.local_search(solution, actual_len, matrix_dist):
        data = data
        if data[3]:
            return data[0], data[1]
    return data[0], data[1]


class TwoDotFiveOpt:

    @staticmethod
    def step2dot5opt(solution, matrix_dist, distance):
        seq_length = len(solution) - 2
        tsp_sequence = np.array(solution)
        uncrosses = 0
        for i in range(1, seq_length - 1):
            for j in range(i + 1, seq_length):
                # 2opt swap
                twoOpt_tsp_sequence = TwoOpt.swap2opt(tsp_sequence, i, j)
                twoOpt_len = distance + TwoOpt.gain(i, j, tsp_sequence, matrix_dist)
                # node shift 1
                first_shift_tsp_sequence = TwoDotFiveOpt.shift1(tsp_sequence, i, j)
                first_shift_len = distance + TwoDotFiveOpt.shift_gain1(i, j, tsp_sequence, matrix_dist)
                # node shift 2
                second_shift_tsp_sequence = TwoDotFiveOpt.shift2(tsp_sequence, i, j)
                second_shift_len = distance + TwoDotFiveOpt.shift_gain2(i, j, tsp_sequence, matrix_dist)

                best_len, best_method = min([twoOpt_len, first_shift_len, second_shift_len]), np.argmin(
                    [twoOpt_len, first_shift_len, second_shift_len])
                sequences = [twoOpt_tsp_sequence, first_shift_tsp_sequence, second_shift_tsp_sequence]
                if best_len < distance:
                    uncrosses += 1
                    tsp_sequence = sequences[best_method]
                    distance = best_len
                    # print(distance, best_method, [twoOpt_len, first_shift_len, second_shift_len])
        return tsp_sequence, distance, uncrosses

    @staticmethod
    def shift1(tsp_sequence, i, j):
        new_tsp_sequence = np.concatenate(
            [tsp_sequence[:i], tsp_sequence[i + 1: j + 1], [tsp_sequence[i]], tsp_sequence[j + 1:]])
        return new_tsp_sequence

    @staticmethod
    def shift_gain1(i, j, tsp_sequence, matrix_dist):
        old_link_len = (matrix_dist[tsp_sequence[i], tsp_sequence[i - 1]] +
                        matrix_dist[tsp_sequence[i], tsp_sequence[i + 1]] +
                        matrix_dist[tsp_sequence[j], tsp_sequence[j + 1]])
        changed_links_len = (matrix_dist[tsp_sequence[i - 1], tsp_sequence[i + 1]] +
                             matrix_dist[tsp_sequence[i], tsp_sequence[j]]
                             + matrix_dist[tsp_sequence[i], tsp_sequence[j + 1]])
        return - old_link_len + changed_links_len

    @staticmethod
    def shift2(tsp_sequence, i, j):
        new_tsp_sequence = np.concatenate(
            [tsp_sequence[:i], [tsp_sequence[j]], tsp_sequence[i: j], tsp_sequence[j + 1:]])
        return new_tsp_sequence

    @staticmethod
    def shift_gain2(i, j, tsp_sequence, matrix_dist):
        old_link_len = (matrix_dist[tsp_sequence[i], tsp_sequence[i - 1]] + matrix_dist[
            tsp_sequence[j], tsp_sequence[j - 1]] + matrix_dist[tsp_sequence[j], tsp_sequence[j + 1]])
        changed_links_len = (
                matrix_dist[tsp_sequence[j], tsp_sequence[i - 1]] + matrix_dist[tsp_sequence[i], tsp_sequence[j]] +
                matrix_dist[tsp_sequence[j - 1], tsp_sequence[j + 1]])
        return - old_link_len + changed_links_len

    @staticmethod
    def local_search(solution, actual_len, matrix_dist):
        new_tsp_sequence = np.copy(np.array(solution))
        uncross = 0
        while True:
            new_tsp_sequence, new_len, uncr_ = TwoDotFiveOpt.step2dot5opt(new_tsp_sequence, matrix_dist, actual_len)
            uncross += uncr_
            # print(new_len, uncross)
            if new_len < actual_len:
                actual_len = new_len
                yield new_tsp_sequence, new_len, 0, False
            else:
                return new_tsp_sequence, new_len, 1, True


def compute_length(solution, dist_matrix):
    total_length = 0
    starting_node = solution[0]
    from_node = starting_node
    for node in solution[1:]:
        total_length += dist_matrix[from_node, node]
        from_node = node
    total_length += dist_matrix[from_node, starting_node]
    return total_length

