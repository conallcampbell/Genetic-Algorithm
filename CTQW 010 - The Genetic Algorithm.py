#The Genetic Algorithm

#The aim of this script is to consider a known target network
#and a population of randomly generated complex quantum networks,
#and to use a genetic algorithm on my population to reconstruct the topology
#of the target quantum network by comparing the fidelity between the sink population

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.linalg import expm
import random

#Population size
#number of networks I wish to consider
population_size = 100

#genes
#all different possible entries in my network
genes = np.array([[0] , [1]])

#target
#the random complex network we want to reconstruct

# number of nodes
n = 15
# each node is joined with its m nearest neighbours in a ring topology
m = 2
# probability of edge rewiring
p = 0.01
# node that contains the single excitation
# *note: this starts at 0 so k = 2 is the 3rd node*
k = 3
# node that contains the sink
s = 8


def target(n , m , p):
    return nx.watts_strogatz_graph(n , m , p , seed = None)
def adjacency_matrix_target(n , m , p):
    return nx.to_numpy_array(target(n , m , p , seed = None)).astype("complex")



#Class for population of networks
class Individual:
    def __init__(self , n , m , p , seed):
        self.n = n
        self.m = m
        self.p = round(random.uniform(0 , 0.1) , 3)
        self.seed = seed

    def adjacency_matrix(n , m , p , seed):
        return nx.to_numpy_array(Individual(n , m , p , seed = None)).astype("complex")



#Class for fidelity of sink outputs
class Sink_Output:
    #the parameters we'll need for the sink output are the node containing the sink,
    #the initial state of the system, the time evolution operator, the current state
    #of the system at time t, and the fidelity between the sink population for the
    #target state and the individual of the population
    def __init__(self , sink , initial_state , U , current_state , fidelity_sinkoutput):
        self.sink = sink
        self.initial_state = initial_state
        self.U = U
        self.current_state = current_state
        self.fidelity_sinkoutput = fidelity_sinkoutput

