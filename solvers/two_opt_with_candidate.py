import numpy as np


class TwoOpt_CL:

    @staticmethod
    def step2opt(solution, matrix_dist, distance, cand_list, N):
        tsp_sequence = np.copy(solution)
        uncrosses = 0
        ite = 0
        for i in range(1, N-2):
            # ite += 1
            # if ite > 11:
            #     break
            for j_ in cand_list[i]:
                el1 = el2 = 0

                j = np.argwhere(tsp_sequence == j_)[0][0]

                if j > i:
                    el1, el2 = i, j
                elif j < i:
                    el1, el2 = j, i
                elif j == i:
                    el1, el2 = j, i

                if el2 == N-1:
                    break
                if el1 == 0:
                    break


                improve = False
                sol_lens = [10000000 for _ in range(4)]
                indices = [[] for _ in range(4)]
                case = 0

                if np.abs(el1 - el2) >= 1:
                    new_distance = distance + TwoOpt_CL.gain(el1, el1 + 1, el2, el2 + 1, tsp_sequence, matrix_dist)
                    if new_distance < distance:
                        improve = True
                        case = 0
                        sol_lens[case] = new_distance
                        indices[case] = [el1 + 1, el2]

                if np.abs(el1 - el2) >= 0:
                    new_distance = distance + TwoOpt_CL.gain(el1 -1, el1, el2, el2 + 1, tsp_sequence, matrix_dist)
                    if new_distance < distance:
                        improve = True
                        case = 1
                        sol_lens[case] = new_distance
                        indices[case] = [el1, el2]

                if np.abs(el1 - el2) >= 1:
                    new_distance = distance + TwoOpt_CL.gain(el1 - 1, el1, el2 - 1, el2, tsp_sequence, matrix_dist)
                    if new_distance < distance:
                        improve = True
                        case = 2
                        sol_lens[case] = new_distance
                        indices[case] = [el1, el2 - 1]

                if np.abs(el1 - el2) >= 2:
                    new_distance = distance + TwoOpt_CL.gain(el1, el1+1,  el2 -1, el2, tsp_sequence, matrix_dist)
                    if new_distance < distance:
                        improve = True
                        case = 3
                        sol_lens[case] = new_distance
                        indices[case] = [el1 + 1, el2 - 1]

                if improve:
                    uncrosses += 1
                    best_case = np.argmin(sol_lens)
                    ind_ = indices[best_case]
                    new_tsp_sequence = TwoOpt_CL.swap2opt(tsp_sequence, ind_[0], ind_[1])
                    # print(tsp_sequence)
                    # print(distance)
                    # print()
                    tsp_sequence = np.copy(new_tsp_sequence)
                    # print(ind_)
                    # print(case)
                    # print(tsp_sequence)
                    distance = sol_lens[best_case]
                    # print(distance)
                    # print()

        return tsp_sequence, distance, uncrosses

    @staticmethod
    def gain(i, ip, j, jp, tsp_sequence, matrix_dist):
        old_link_len = (matrix_dist[tsp_sequence[i], tsp_sequence[ip]] +
                        matrix_dist[tsp_sequence[j], tsp_sequence[jp]])
        changed_links_len = (matrix_dist[tsp_sequence[jp], tsp_sequence[ip]] +
                             matrix_dist[tsp_sequence[j], tsp_sequence[i]])
        return - old_link_len + changed_links_len

    @staticmethod
    def swap2opt(tsp_sequence, i, j):
        new_tsp_sequence = np.copy(tsp_sequence)
        new_tsp_sequence[i:j+1] = np.flip(tsp_sequence[i:j+1], axis=0)
        return new_tsp_sequence


    @staticmethod
    def local_search(solution, actual_len,  matrix_dist, CL):
        new_tsp_sequence = np.copy(np.array(solution))
        N = len(solution)
        uncross = 0
        # ite = 0
        while True:
            new_tsp_sequence, new_reward, uncr_ = TwoOpt_CL.step2opt(new_tsp_sequence, matrix_dist, actual_len, CL, N)
            uncross += uncr_
            if new_reward < actual_len:
                # print(actual_len, new_reward)
                actual_len = np.copy(new_reward)
                yield new_tsp_sequence, actual_len, 0, False
            else:
                yield new_tsp_sequence, actual_len, 1, True
                # ite += 1
                # # print(new_tsp_sequence)
                # int_ = np.random.randint(1, N)
                # new_tsp_sequence = np.roll(new_tsp_sequence, int_)
                # print(new_tsp_sequence)


def twoOpt_with_cl(solution, actual_len, matrix_dist, CL):
    for data in TwoOpt_CL.local_search(solution, actual_len, matrix_dist, CL):
        if data[3]:
            return data[0], data[1]

