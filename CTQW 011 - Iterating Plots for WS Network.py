import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.linalg import expm
from scipy.linalg import sqrtm
import random

##########################################
#kronecker product code for more than 2 inputs
def kron(*matrices):
    result = np.array([[1]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result

##########################################
#Defining the necessary parameters for the network
# number of nodes
n = 7
# each node is joined with its m nearest neighbours in a ring topology
m = 2
# probability of edge rewiring
p = 0.01

##########################################
#Building the network
# Building a Watts-Strogatz network
def G(n , m , p):
    return nx.watts_strogatz_graph(n , m , p , seed = None)

network1 = G(n , m , p)
network2 = G(n , m + 2 , 0.02)
network3 = G(n , m + 1 , 0.04)

def network1_adjmatrices_list():
    return [np.array(nx.to_numpy_array(network1).astype("complex")) for _ in range(n)]
def network2_adjmatrices_list():
    return [np.array(nx.to_numpy_array(network2).astype("complex")) for _ in range(n)]
def network3_adjmatrices_list():
    return [np.array(nx.to_numpy_array(network3).astype("complex")) for _ in range(n)]

##########################################
#Mutating the network
##########################################
#A mutation will correspond to adding/removing a connection between nodes
#therefore, this will result in changing a 0 to a 1/a 1 to a 0 in the adjacency matrix
#we must be careful, however, since an adjacency matrix is symmetric
#thus, if we change element [i , j] to 0, then we must also change element [j , i]
##########################################
#However, we only want to mutate at most 2 node connections,
#so we should consider the case where we add a single connection and take away a mutation
#we would want this to happen 1 in 5 times for our GA but this can be resolved later

def mutated_network():
    matrices_list1 = network1_adjmatrices_list()
    mutated_matrices = []
    for matrix in matrices_list1:
        mutated_matrix = np.copy(matrix)
        for i in range(n):
            for j in range(i):
                if random.random() < p:
                    el = mutated_matrix[i, j]
                    mutated_matrix[i, j] = np.real(el) % 2 + np.imag(el)*0j
                    mutated_matrix[j , i] = mutated_matrix[i , j]
                    mutated_matrices.append(np.copy(mutated_matrix))
    return mutated_matrices


#try to build mutation of target network for GREAT comparison
#adjacency matrices are symmetric so flipping the value of element [i,j] flips the [j,i] element too

##########################################
# Adding the Sink
# We add -1j to the diagonal element for our node containing the sink
# we want to add a sink to each different node for each different copy
# we've made of the network, i.e., for each different network_adjmatriceslist(n)
def sink_node_network1():
    matrices_list1 = network1_adjmatrices_list()
    for i in range(n):
        matrices_list1[i][i , i] -= 1j
    return matrices_list1

#produce a list of n copies with sink at each different node
def mutated_sink_node():
    mutated = mutated_network()
    for i in range(n):
        mutated[i][i , i] -= 1j
    return mutated

def sink_node_network2():
    matrices_list2 = network2_adjmatrices_list(n)
    for i in range(n):
        matrices_list2[i][i , i] -= 1j
    return matrices_list2

def sink_node_network3():
    matrices_list3 = network3_adjmatrices_list(n)
    for i in range(n):
        matrices_list3[i][i , i] -= 1j
    return matrices_list3

##########################################
# Initial state of my n node network, where the single excitation
# exists in the k^th node
k = 1
rho0 = np.zeros((n,1))
rho0[k] = 1
state = kron(rho0 , np.conjugate(rho0.T))

##########################################
# Evolution
# We observe how this changes as a function of time
# thus, we choose a time-evolution operator U = exp(-iHt) for H = A

def Unitary_operator_network1(t):
    return [expm(complex(0 , -1) * sink_node_network1(n)[i] * t) for i in range(n)]

def Unitary_operator_network2(t):
    return [expm(complex(0 , -1) * sink_node_network2(n)[i] * t) for i in range(n)]

def Unitary_operator_network3(t):
    return [expm(complex(0 , -1) * sink_node_network3(n)[i] * t) for i in range(n)]

# Defining the evolution of the state

def Evolution_of_network1(t):
    return [Unitary_operator_network1(t)[i] @ state @ np.conjugate(Unitary_operator_network1(t)[i].T) for i in range(n)]

def Evolution_of_network2(t):
    return [Unitary_operator_network2(t)[i] @ state @ np.conjugate(Unitary_operator_network2(t)[i].T) for i in range(n)]

def Evolution_of_network3(t):
    return [Unitary_operator_network3(t)[i] @ state @ np.conjugate(Unitary_operator_network3(t)[i].T) for i in range(n)]



##########################################
## Plotting
# Now we make a plot of the evolution of the diagonal elements of rho1
ts = np.linspace(0, 10, 100)

Excitation_Evolution_network1 = [[Evolution_of_network1(t)[i][i,i] for t in ts] for i in range(n)]
Excitation_Evolution_network2 = [[Evolution_of_network2(t)[i][i,i] for t in ts] for i in range(n)]
Excitation_Evolution_network3 = [[Evolution_of_network3(t)[i][i,i] for t in ts] for i in range(n)]

figure , ax2 = plt.subplots(1 , 1)
ax2.set_xlabel("t")
ax2.set_ylabel("Excitation Evolution")
for i in range(n):
    ax2.plot(ts , Excitation_Evolution_network1[i])
    # ax2.plot(ts , Excitation_Evolution_network2[i])
plt.legend(["Target Network"])
plt.show()

figure , ax3 = plt.subplots(1 , 1)
ax3.set_xlabel("t")
ax3.set_ylabel("Excitation Evolution")
for i in range(n):
    ax3.plot(ts , Excitation_Evolution_network2[i])
    # ax3.plot(ts , Excitation_Evolution_network3[i])
plt.legend(["Target Network" ])
plt.show()