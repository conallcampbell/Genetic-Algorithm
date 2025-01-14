import numpy as np
from scipy.linalg import expm

##########################################
#kronecker product code for more than 2 inputs
##########################################
def kron(*matrices):
    result = np.array([[1]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result

##########################################
# defining the timestep
##########################################
ts = np.linspace(0 , 10 , 100)

##########################################
# Defining the network class that considers any size of adjacency matrix 
# and returns all the sink outputs for that specific matrix
##########################################
class Network:
    # saving the adjacency matrices of the population to the class
    def __init__(self, adjacency_mat):
        self.mat = adjacency_mat
        self.nodes = self.mat.shape[0]

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
        mat = self.mat.copy()
        mat[i , i] -= 1j
        return mat
    
    # defining the initial state
    def set_excitation(self , e):
        rho0 = kron(np.zeros((self.nodes , 1)) , np.conjugate(np.zeros((self.nodes , 1)).T))
        rho0[e , e] = 1
        return rho0
    
    # defining the unitary evolution
    def evolution(self , t):
        # setting the excitation initially at node 1
        state = self.set_excitation(1)
        evolution_data = np.zeros(self.nodes)
        for i in range(self.nodes):
            sink = self.set_sink(i)
            
            # evolving the state that has the sink located at node i
            evolution_result = (

                                                1 - np.trace(
                                                                (expm(complex(0 , -1) * sink * t))
                                                                @ state
                                                                @ (np.conjugate(expm(complex(0 , -1) * sink * t).T))
                                                            ).real
                                            
                                            )
            evolution_data[i] = evolution_result
        return evolution_data
    
    def __str__(self):
        s = f"Nodes: {self.nodes}"
        s += str(self.mat)
        return s
    
    # Getting the output of the class
    def simulate(self):
        adjacency_str = str(self.mat)
        sink_outputs = []
        for t in ts:
            sink_output_data = self.evolution(t)
            sink_outputs.append(sink_output_data)
        return sink_outputs