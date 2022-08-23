from typing import List, Tuple
from .graph import Ant, Graph
from multiprocessing import Pool
from random import choices
import numpy as np


class ACO:
    def __init__(self, pop: int, rho: float, alpha: float, beta: float, its: int, filepath: str):
        """Creates an Ant Colony

        Args:
            pop (int): Number of ants
            rho (float): Evaporation rate
            alpha (float): How much importance to give to the pheromone
            beta (float): How much importance to give to the greedy heuristic
            its (int): Maximum number of iterations
            filepath (str): Path to the file describing the graph
        """
        self.pop = pop
        self.alpha = alpha
        self.beta = beta
        self.its = its
        self.graph = Graph(rho, filepath)

    def start(self) -> List[Tuple[float, float, float, float]]:
        """Runs the ant colony optimization

        Returns:
            List[Tuple[float, float, float, float]]: List of 4-tuples (best so far, iteration best, iteration worst, iteration mean)
            per iteration.
        """
        P = [Ant(self.alpha, self.beta) for _ in range(self.pop)]
        log = []
        best = (0, [])

        with Pool() as pool:
            for it in range(self.its):
                P = pool.map(self.worker_ant, P)
                itbest = max(P, key=lambda a: a.get_path_length(self.graph))
                itbest = (itbest.get_path_length(self.graph), itbest.path)
                best = itbest if itbest[0] > best[0] else best
                self.graph.update_pheromones(choices((best[1], itbest[1]), [it / self.its, 1 - (it / self.its)])[0])
                log.append(
                    (
                        float(best[0]),
                        float(itbest[0]),
                        float(min(P, key=lambda a: a.get_path_length(self.graph)).get_path_length(self.graph)),
                        float(np.mean(np.array(list(map(lambda a: a.get_path_length(self.graph), P)))))
                    )
                )

        return log

    def worker_ant(self, ant: Ant) -> Ant:
        """Helper function to use with multiprocessing.Pool, just calls Ant.create_path(/1), and returns the ant.
        """
        ant.create_path(self.graph)
        return ant
