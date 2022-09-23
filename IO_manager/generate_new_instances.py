from concorde.tsp import TSPSolver
from IO_manager.io_tsp import TSP_Instance_Creator


def generate_instance():
    for dim in [10, 20, 30]:
        ic = TSP_Instance_Creator('random', seed=123, dimension=dim)
        solver = TSPSolver.from_data(ic.points[:, 1], ic.points[:, 2], norm="EUC_2D")
        solution = solver.solve()
        save_new_tsplib(f"myTSP_dim{dim}.tsp", ic.points, solution)


def save_new_tsplib(name, pos, sol):
    optimal_sol = sol.tour
    best_sol = sol.optimal_value
    folder = "problems/TSP/"
    file_object = open(f"{folder}{name}", 'w')
    file_object.write(f"NAME : {name[:-4]}\n")
    file_object.write(f"COMMENT : generated for test\n")
    file_object.write(f"TYPE : TSP\n")
    file_object.write(f"DIMENSION : {pos.shape[0]}\n")
    file_object.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
    file_object.write(f"BEST_KNOWN : {int(best_sol)}\n")
    file_object.write("NODE_COORD_SECTION\n")
    for i, p in enumerate(optimal_sol):
        file_object.write(f"{i + 1} {pos[p, 1]} {pos[p, 2]}\n")

    file_object.write(f"EOF")
    file_object.close()
