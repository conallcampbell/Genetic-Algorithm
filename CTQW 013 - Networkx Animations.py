import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
 
# Create a simple graph
G = nx.cycle_graph(5)
pos = nx.spring_layout(G)
 
values = np.random.rand(100,5)
 
# Example function to update values
class GraphAnimator:
 
    def __init__(self):
        pass  
 
    def animate(self, graph, values, duration = 10):
        self.graph = graph
        self.pos = nx.spring_layout(G)
        self.fig, self.ax = plt.subplots()
        self.ani = FuncAnimation(self.fig, self.update, frames=values, repeat=False, interval=int(duration * 1000 / values.shape[0]))
        plt.show()
 
    # Animation update function
    def update(self, frame):
        print(frame)
        self.ax.clear()
        node_colors = [plt.cm.viridis(value) for value in frame]
        nx.draw(self.graph, pos=self.pos, node_color=node_colors, with_labels=True, ax=self.ax)