from typing import List
import numpy as np
from numpy.random import randint, choice


class Graph:
    def __init__(self, rho: float, filepath: str = ""):
        """Creates a directed graph with edges weighted by distance and pheromones

        Args:
            rho (float): Evaporation rate
            filepath (str, optional): Path to the file that describes the graph. Defaults to "".
        """
        self.rho = rho

        if filepath:
            self.load_file(filepath)


    def fn(self) -> int:
        """Executes a furthest neighbor heuristic starting from a random node of the graph.

        Returns:
            int: Path length
        """
        a = Ant(0, 1)
        a.create_path(self)
        return a.get_path_length(self)

    def load_file(self, filepath: str):
        """Loads the graph

        Args:
            filepath (str): Path to the file that describes the graph.
        """
        with open(filepath, "r") as fp:
            lines = list(map(lambda line: tuple(map(int, line.split())), fp.readlines()))
            self.n_vertices = np.max(np.array(lines).flatten())
            self.adj_matrix = np.zeros(shape=(self.n_vertices, self.n_vertices), dtype=np.longfloat)
            self.pheromone = np.zeros(shape=(self.n_vertices, self.n_vertices), dtype=np.longfloat)
            for o, d, w in lines:
                self.adj_matrix[o-1][d-1] = w
            # initial pheromone
            self.t_0 = self.fn()/(self.rho)
            self.pheromone[self.adj_matrix != 0] = self.t_0
            # Average number of edges per vertex
            self.avg = len(np.nonzero(self.adj_matrix))/self.n_vertices

    def get_path_length(self, path: List[int]) -> int:
        """Computes the path length of path

        Args:
            path (List[int]): List of vertices

        Returns:
            int: Path length
        """
        return np.sum(self.adj_matrix[path[:-1], path[1:]])

    def update_pheromones(self, path: List[int]):
        """Evaporates the pheromones and updates the pheromones along the path

        Args:
            path (List[int]): Path to increase pheromone concentration
        """
        self.t_max = self.get_path_length(path)/(self.rho)
        a = 0.05**(1/(self.n_vertices))
        self.t_min = self.t_max * ((1 - a)/((self.avg - 1)*a))
        self.pheromone = self.pheromone * (1 - self.rho)
        self.pheromone[path[:-1], path[1:]] += self.get_path_length(path)
        self.pheromone[self.pheromone < self.t_min] = self.t_min
        self.pheromone[self.pheromone > self.t_max] = self.t_max

    def reset_pheromones(self):
        """Reset the pheromones to the initial value
        """
        self.pheromone[self.adj_matrix != 0] = self.t_0







class Ant:
    def __init__(self, alpha: float, beta: float):
        """Creates an Ant
        """
        self.alpha = alpha
        self.beta = beta
        self.path: List[int] = []

    def get_valid_neighbors(self, v: int, graph: Graph) -> np.ndarray:
        """Returns the neighbors of v that were not visited

        Args:
            v (int): Vertex to check neighborhood
            graph (Graph): Graph containing v

        Returns:
            np.ndarray: Array of valid neighbors
        """
        return np.arange(graph.n_vertices)[(self.visited == 0) & (graph.adj_matrix[v] != 0)]

    def create_path(self, graph: Graph):
        """Create the path and set self.path to it

        Args:
            graph (Graph): Graph to create the path on
        """
        self.path: List[int] = []
        curv = randint(0, graph.n_vertices)
        self.visited = np.zeros(graph.n_vertices)
        self.visited[curv] = 1
        self.path.append(curv)
        not_path = self.get_valid_neighbors(curv, graph)
        N: np.ndarray = graph.adj_matrix[curv, not_path]
        W: np.ndarray = graph.pheromone[curv, not_path]
        while len(N):
            d = np.sum((N**self.beta) * (W**self.alpha))
            if not d or d == np.inf:
                P = np.ones(len(W))/len(W)
            else:
                P: np.ndarray = ((W**self.alpha)*(N**self.beta))  / d

            curv = int(choice(not_path, p=P.astype(np.float64)))
            self.path.append(curv)
            self.visited[curv] = 1
            not_path = self.get_valid_neighbors(curv, graph)
            N: np.ndarray = graph.adj_matrix[curv, not_path]
            W: np.ndarray = graph.pheromone[curv, not_path]

    def get_path_length(self, graph: Graph) -> int:
        """Computes the self.path length

        Args:
            graph (Graph): Graph where the path is

        Returns:
            int: Path length
        """
        return np.sum(graph.adj_matrix[self.path[:-1], self.path[1:]])