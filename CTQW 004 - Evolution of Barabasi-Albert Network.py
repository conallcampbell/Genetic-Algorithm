# Simulating the evolution of a Barabasi-Albert network
# A Barabasi-Albert network is a graph of n nodes that is grown by attaching new nodes
# each with m edges that are preferentially attached to existing nodes with high degree

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
n = 25
# number of edges to attach from a new node to existing nodes
m = 13
# node that contains the single excitation
# *note: this starts at 0 so k = 2 is the 3rd node*
k = 3
# node that contains the sink
s = 5


##########################################
# Building a Barabasi-Albert network
def G(n , m):
    return nx.barabasi_albert_graph(n , m , seed = 13 , initial_graph = None)
# We want to consider 2 different sinks separately
# so we want to change the variable name to network
network = G(n , m)
#drawing the network
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



# Initial state of my n node network, where the single excitation exists in the k^th node
rho0 = np.zeros((n,1))
rho0[k] = 1
state = kron(rho0 , np.conjugate(rho0.T))



##########################################
# Evolution
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
ts = np.linspace(0, 10, 5000)
sink_population1 = [1-np.trace(rho1A(t)) for t in ts]
sink_population2 = [1-np.trace(rho1B(t)) for t in ts]

figure , ax1 = plt.subplots(1,1)
ax1.set_xlabel("t")
ax1.set_ylabel("Sink Population")
ax1.plot(ts , sink_population1)
ax1.plot(ts , sink_population2)
plt.legend(["Strong Sink","Weak Sink"])
plt.show()