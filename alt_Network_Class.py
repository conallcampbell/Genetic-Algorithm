# Frozen Network class
# this takes any input matrix and the target matrix and calculates the associated distance
# building the class this way ensures that the individual and its corresponding distance are saved together
import numpy as np
from scipy.linalg import expm


def kron(*matrices):
    result = np.array([[1]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


class NetworkClass:
    # initialising the class
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.nodes = self.adj_matrix.shape[0]
        self.distance = None

    # assuring that there are no sinks in the adjacency matrices
    @property
    def adjacency_matrix(self):
        return np.real(self.mat)

    # ensuring that each element of my class is indeed an adjacency matrix of a network
    @adjacency_matrix.setter
    def adjacency_matrix(self, mat):
        self.mat = mat

    # attaching the sink to node [i , i]
    def set_sink(self, i):
        mat = np.array(self.adj_matrix.copy(), dtype=complex)
        mat[i , i] -= 1j
        return mat
 
    # defining the initial state
    def set_excitation(self, e):
        rho0 = kron(np.zeros((self.nodes, 1), dtype=complex), np.conjugate(np.zeros((self.nodes, 1), dtype = complex).T))
        rho0[e, e] = 1
        return rho0

    # defining the unitary evolution
    def evolution(self, timestep, num_steps):
        evolution = []
        state = self.set_excitation(1)
        # evolution_data = np.zeros(self.nodes)
        # setting the excitation initially at node 1
        sink_operators = []
        for i in range(self.nodes):
            sink = self.set_sink(i)  
            sink_operator = expm(complex(0, -1) * sink * timestep)
            sink_operators.append(sink_operator)
        
        sink_states = [state for i in range(self.nodes)]
        for i in range(num_steps):
            timestep_data = []
            for j in range(self.nodes):
                sink_operator = sink_operators[j]
                # evolving the state that has the sink located at node i
                sink_states[j] = sink_operator @ sink_states[j] @ np.conjugate(sink_operator.T)
                evolution_result = 1 - np.trace(sink_states[j]).real         
                timestep_data.append(evolution_result)
            evolution.append(timestep_data)
        self.sink_data = np.array(evolution)
    

    # defining the distance function
    def calculate_distance(self, target_data, timestep, num_steps):
        self.evolution(timestep, num_steps)
        self.distance = np.sum(abs(target_data - self.sink_data))