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

class alt_Frozen_Network:
    # initialising the class
    def __init__(self , adj_matrix , target , times):
        self.adj_matrix = adj_matrix
        self.target = target
        self.times = times
        self.nodes = self.adj_matrix.shape[0]
        self.distance = self.calculate_distance

    # assuring that there are no sinks in the adjacency matrices
    @property
    def adjacency_matrix(self):
        return np.real(self.mat)
    
    # ensuring that each element of my class is indeed an adjacency matrix of a network
    @adjacency_matrix.setter
    def adjacency_matrix(self , mat):
        self.mat = mat

    # attaching the sink to node [i , i]
    def set_sink(self , i):
        mat = np.array(self.adj_matrix.copy(), dtype=complex)
        mat[i , i] -= 1j
        return mat

    # defining the initial state
    def set_excitation(self , e):
        rho0 = kron(np.zeros((self.nodes , 1) , dtype = complex) , np.conjugate(np.zeros((self.nodes , 1) , dtype = complex).T))
        rho0[e , e] = 1
        return rho0
    
    # defining the unitary evolution
    def evolution(self):

        evolution = []
        state = self.set_excitation(1)
        # evolution_data = np.zeros(self.nodes)
        #  setting the excitation initially at node 1

        for t in self.times:
            sink_operators = []
            for i in range(self.nodes):
                sink = self.set_sink(i)  
                sink_operator = expm(complex(0, -1) * sink * t)
                sink_operators.append(sink_operator)
        
            sink_states = [state for i in range(self.nodes)]
            timestep_data = []
            for i in range(self.nodes):
                sink_operator = sink_operators[i]
                 # evolving the state that has the sink located at node i
                sink_states[i] = sink_operator @ sink_states[i] @ np.conjugate(sink_operator.T)
                evolution_result = 1 - np.trace(sink_states[i]).real
                timestep_data.append(evolution_result)
            evolution.append(timestep_data)
        return np.array(evolution)



    # defining the distance function
    def calculate_distance(self):       
        indiv = self.evolution()
        individual_distance_sum = np.sum(abs(self.target - indiv))
        return individual_distance_sum