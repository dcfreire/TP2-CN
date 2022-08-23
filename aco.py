from .graph import Ant, Graph
from multiprocessing import Pool
from random import choices
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
import numpy as np




class ACO:
    def __init__(self, pop: int, rho: float, alpha: float, beta: float, its: int, filepath: str):
        P = [Ant(alpha, beta) for _ in range(pop)]
        self.graph = Graph(rho, filepath)
        best = (0, [])
        fig  = plt.figure(figsize=(12,8))
        with Pool() as pool:
            for it in range(its):
                P = pool.map(self.worker_ant, P)
                itbest = max(P, key=lambda a: a.get_path_length(self.graph))
                itbest = (itbest.get_path_length(self.graph), itbest.path)
                best = itbest if itbest[0] > best[0] else best
                print(best[0])
                self.graph.update_pheromones(choices((best[1], itbest[1]),[it/its, 1-(it/its)])[0])
                g = nx.from_numpy_matrix(self.graph.adj_matrix, create_using=nx.DiGraph)
                sm = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=np.max(self.graph.pheromone)), cmap=plt.cm.jet)
                my_pos = nx.spring_layout(g, seed = 100)
                ax = plt.subplot(1,1,1)
                nx.draw(g, pos=my_pos, node_size=200, with_labels=True, edge_color=list(map(sm.to_rgba, self.graph.pheromone[np.nonzero(self.graph.pheromone)].flatten())), edge_cmap=plt.cm.jet)
                plt.text(0.11, 1, f"it: {it} - best: {best[0]}")
                plt.colorbar(sm)
                plt.savefig(f"figs/{it}.png")

        fig  = plt.figure(figsize=(12,8))
        def animate(frame):
            im = plt.imread(f"figs/{frame}.png")
            plt.imshow(im)

        ani = animation.FuncAnimation(fig, animate, frames=600, interval=(1))
        ani.save("test.mp4")
        self.best = best



    def worker_ant(self, ant):
        ant.create_path(self.graph)
        return ant
