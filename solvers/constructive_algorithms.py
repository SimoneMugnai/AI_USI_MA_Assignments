import numpy as np


class Random_Initializer:
    @staticmethod
    def random_method(dist_matrix):
        n = int(dist_matrix.shape[0])
        solution = np.random.choice(np.arange(n), size=n, replace=False)
        return solution, compute_length(solution, dist_matrix)


class Nearest_Neighbor:
    @staticmethod
    def nn(dist_matrix, starting_node=0):
        dist_matrix = np.copy(dist_matrix)
        n = int(dist_matrix.shape[0])
        node = starting_node
        tour = [node]
        for _ in range(n - 1):
            for new_node in np.argsort(dist_matrix[node]):
                if new_node not in tour:
                    tour.append(new_node)
                    node = new_node
                    break
        return tour, compute_length(tour, dist_matrix)

    @staticmethod
    def best_nn(dist_matrix):
        solutions, lens = [], []
        for start in range(dist_matrix.shape[0]):
            new_solution = Nearest_Neighbor.nn(dist_matrix, starting_node=start)
            solutions.append(new_solution)
            lens.append(compute_length(new_solution, dist_matrix))

        solution = solutions[np.argmin(lens)]
        return solution


class Multi_Fragment:
    @staticmethod
    def check_if_available(n1, n2, sol):
        if len(sol[str(n1)]) < 2 and len(sol[str(n2)]) < 2:
            return True
        else:
            return False

    @staticmethod
    def check_if_not_close(edge_to_append, sol):
        n1, n2 = edge_to_append
        from_city = n2
        if len(sol[str(from_city)]) == 0:
            return True
        partial_tour = [from_city]
        end = False
        iteration = 0
        while not end:
            if len(sol[str(from_city)]) == 1:
                if from_city == n1:
                    return_value = False
                    end = True
                elif iteration > 1:
                    # print(f"iteration {iteration}, elements in partial {len(partial_tour)}",
                    #       f"from city {from_city}")
                    return_value = True
                    end = True
                else:
                    from_city = sol[str(from_city)][0]
                    partial_tour.append(from_city)
                    iteration += 1
            else:
                # print(from_city, partial_tour, sol[str(from_city)])
                for node_connected in sol[str(from_city)]:
                    # print(node_connected)
                    if node_connected not in partial_tour:
                        from_city = node_connected
                        partial_tour.append(node_connected)
                        # print(node_connected, sol[str(from_city)])
                        iteration += 1
        return return_value

    @staticmethod
    def create_solution(start_sol, sol, n):
        assert len(start_sol) == 2, "too many cities with just one link"
        end = False
        n1, n2 = start_sol
        from_city = n2
        sol_list = [n1, n2]
        iterazione = 0
        while not end:
            for node_connected in sol[str(from_city)]:
                iterazione += 1
                if node_connected not in sol_list:
                    from_city = node_connected
                    sol_list.append(node_connected)
                    # print(f"next {node_connected}",
                    #       f"possible {sol[str(from_city)]}",
                    #       f"last tour {sol_list[-5:]}")
                if iterazione > 300:
                    if len(sol_list) == n:
                        end = True
        # sol_list.append(n1)
        return sol_list

    @staticmethod
    def mf(dist_matrix):
        mat = np.copy(dist_matrix)
        mat = np.triu(mat)
        mat[mat == 0] = 100000
        num_cit = dist_matrix.shape[0]
        solution = {str(i): [] for i in range(num_cit)}
        start_list = [i for i in range(num_cit)]
        inside = 0
        for el in np.argsort(mat.flatten()):
            node1, node2 = el // num_cit, el % num_cit
            possible_edge = [node1, node2]
            if Multi_Fragment.check_if_available(node1, node2,
                                                 solution):
                if Multi_Fragment.check_if_not_close(possible_edge, solution):
                    # print("entered", inside)
                    solution[str(node1)].append(node2)
                    solution[str(node2)].append(node1)
                    if len(solution[str(node1)]) == 2:
                        start_list.remove(node1)
                    if len(solution[str(node2)]) == 2:
                        start_list.remove(node2)
                    inside += 1
                    # print(node1, node2, inside)
                    if inside == num_cit - 1:
                        # print(f"reconstruct the solution from {start_list}",
                        #       f"neighbours of these two nodes {[solution[str(i)] for i in start_list]}")
                        solution = Multi_Fragment.create_solution(start_list, solution, num_cit)
                        return solution, compute_length(solution, dist_matrix)


def compute_length(solution, dist_matrix):
    total_length = 0
    starting_node = solution[0]
    from_node = starting_node
    for node in solution[1:]:
        total_length += dist_matrix[from_node, node]
        from_node = node
    total_length += dist_matrix[from_node, starting_node]
    return total_length


def nn(dist_matrix, starting_node=0):
    dist_matrix = np.copy(dist_matrix)
    n = int(dist_matrix.shape[0])
    node = starting_node
    tour = [node]
    for _ in range(n - 1):
        for new_node in np.argsort(dist_matrix[node]):
            if new_node not in tour:
                tour.append(new_node)
                node = new_node
                break
    return tour, compute_length(tour, dist_matrix)
