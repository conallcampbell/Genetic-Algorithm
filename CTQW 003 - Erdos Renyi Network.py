# Simulating the evolution of an Erdos-Renyi network
# An Erdos-Renyi network is a random graph with n nodes and m edges

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.linalg import expm

#kronecker product code for more than 2 inputs
def kron(*matrices):
    result = np.array([[1]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result



#Defining the necessary parameters for the code

# number of nodes
n = 29
# probability of edge creation
p = 0.44
# node that contains the single excitation
# *note: this starts at 0 so m = 2 is the 3rd node*
k = 14
# node that contains the sink
s = 3



##########################################
# Building an Erdos-Renyi network
def G(n , p):
    return nx.erdos_renyi_graph(n , p)
# We want to consider 2 different sinks separately
# so we want to change the variable name to network
network = G(n , p)
nx.draw(network , with_labels = True)


##########################################
# Adjacency Matrix
# Building the adjacency matrix of the network
A = nx.to_numpy_array(network).astype("complex")
B = nx.to_numpy_array(network).astype("complex")

# we add the command .astype("complex") to specify that our array is to be converted to complex numbers
# this is because nx.to_numpy_array outputs a numpy array with only real entries
# therefore, we must specify that we want to start dealing with complex numbers if we want to include a sink

##########################################
# Adding the Sink
# We add -1j to the diagonal element for our node containing the sink
A[s,s]-=15j
B[s,s]-=1j


##########################################
# Evolution
# We choose an initial state for the system, I want to build this so that my input state will change with the dimension of my adjacency matrix

# Initial state of my n node network, where the single excitation exists in the k^th node
rho0 = np.zeros((n,1))
rho0[k] = 1
state = kron(rho0 , np.conjugate(rho0.T))

# We observe how this changes as a function of time
# thus, we choose a time-evolution operator U = exp(-iHt) for H = A
def UA(t):
    return expm(complex(0 , -1) * A * t)
def UB(t):
    return expm(complex(0 , -1) * B * t)
# Defining the evolution of the state
def rho1A(t):
    return UA(t) @ state @ np.conjugate(UA(t).T)
def rho1B(t):
    return UB(t) @ state @ np.conjugate(UB(t).T)



##########################################
## Plotting
# Now we make a plot of the evolution of the diagonal elements of rho1
ts = np.linspace(0, 10000, 100)
excitation_evolution1 = [rho1A(t)[14,14] for t in ts]
excitation_evolution2 = [rho1B(t)[14,14] for t in ts]
excitation_evolution3 = [rho1A(t)[3,3] for t in ts]
excitation_evolution4 = [rho1B(t)[3,3] for t in ts]

figure , ax1 = plt.subplots(1,1)
ax1.set_xlabel("t")
ax1.set_ylabel("Excitation Evolution")
ax1.plot(ts , excitation_evolution1)
ax1.plot(ts , excitation_evolution2)
ax1.plot(ts , excitation_evolution3)
ax1.plot(ts , excitation_evolution4)
plt.legend(["Strong Sink - Excitation Node","Weak Sink - Excitation Node","Strong Sink - Sink Node","Weak Sink - Sink Node"])
plt.show()