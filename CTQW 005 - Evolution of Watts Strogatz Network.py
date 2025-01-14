# Simulating the evolution of a Watts-Strogatz network
# A Watts-Strogatz networrk creates a ring over n nodes, then each node is joined to its
# k nearest neighbours (or k-1 if k is odd)
# watts_strogatz_graph( n = number of nodes , k = number of nearest neighbours , p = probability of edge reqiring ,  seed = indicator of Random Number Generation state)



import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.linalg import expm
from scipy.linalg import sqrtm



#kronecker product code for more than 2 inputs
def kron(*matrices):
    result = np.array([[1]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result



#Defining the necessary parameters for the code

# number of nodes
n = 7
# each node is joined with its m nearest neighbours in a ring topology
m = 2
# probability of edge rewiring
p = 0.01
# node that contains the single excitation
# *note: this starts at 0 so k = 2 is the 3rd node*
k = 3
# node that contains the sink
s = 4


##########################################
#Building the network
# Building a Watts-Strogatz network
def G(n , m , p):
    return nx.watts_strogatz_graph(n , m , p , seed = None)
# We want to consider 2 different sinks separately
# so we want to change the variable name to network
network = G(n , m , p)
network1 = G(n , 2 , 0.05)
network2 = G(n , 4 , 0.02)
network3 = G(n , 3 , 0.04)
#drawing the network
#nx.draw(network , with_labels = True)



##########################################
# Adjacency Matrix
# Building the adjacency matrix of the network
A = nx.to_numpy_array(network).astype("complex")
B = nx.to_numpy_array(network).astype("complex")
#unrelated to QZE work
#comparing the fidelity between two different unlike networks
network1_adjmatrix = nx.to_numpy_array(network1).astype("complex")
network2_adjmatrix = nx.to_numpy_array(network2).astype("complex")
network3_adjmatrix = nx.to_numpy_array(network3).astype("complex")


# we add the command .astype("complex") to specify that our array is to be converted to complex numbers
# this is because nx.to_numpy_array outputs a numpy array with only real entries
# therefore, we must specify that we want to start dealing with complex numbers if we want to include a sink

##########################################
# Adding the Sink
# We add -1j to the diagonal element for our node containing the sink



for s in range(n):
    network1_adjmatrix[s,s]-=1j
    network2_adjmatrix[s,s]-=1j
    network3_adjmatrix[s,s]-=1j





# A[s,s]-=15j
# B[s,s]-=1j

#adding a sink to the adjacency matrices of the like and unlike networks
network1_adjmatrix[s,s]-=1j
network2_adjmatrix[s,s]-=1j
network3_adjmatrix[s,s]-=1j

fidelity_adjmatrix_12 = (np.trace(sqrtm(sqrtm(network2_adjmatrix , 1/2) @ network1_adjmatrix @  sqrtm(network2_adjmatrix , 1/2), 1/2)))**2
fidelity_adjmatrix_13 = (np.trace(sqrtm(sqrtm(network3_adjmatrix , 1/2) @ network1_adjmatrix @  sqrtm(network3_adjmatrix , 1/2), 1/2)))**2



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
def U_network1(t):
    return expm(complex(0 , -1) *network1_adjmatrix *t)
def U_network2(t):
    return expm(complex(0 , -1) *network2_adjmatrix *t)
def U_network3(t):
    return expm(complex(0 , -1) *network3_adjmatrix *t)
# Defining the evolution of the state
def rho1A(t):
    return UA(t) @ state @ np.conjugate(UA(t).T)
def rho1B(t):
    return UB(t) @ state @ np.conjugate(UB(t).T)
def rho_network1(t):
    return U_network1(t) @ state @ np.conjugate(U_network1(t).T)
def rho_network2(t):
    return U_network2(t) @ state @ np.conjugate(U_network2(t).T)
def rho_network3(t):
    return U_network3(t) @ state @ np.conjugate(U_network3(t).T)



##########################################
## Plotting
# Now we make a plot of the evolution of the diagonal elements of rho1
ts = np.linspace(0, 10, 100)

# excitation_evolution1 = [rho1A(t)[14,14] for t in ts]
# excitation_evolution2 = [rho1B(t)[14,14] for t in ts]
# excitation_evolution3 = [rho1A(t)[3,3] for t in ts]
# excitation_evolution4 = [rho1B(t)[3,3] for t in ts]

# excitation_evolution_network1 = [rho_network1(t)[1,1] for t in ts]
# excitation_evolution_network2 = [rho_network2(t)[1,1] for t in ts]
# excitation_evolution_network3 = [rho_network3(t)[1,1] for t in ts]

excitation_evolution_network1 = [[rho_network1(t)[i,i] for t in ts] for i in range(n)]
excitation_evolution_network2 = [[rho_network2(t)[i,i] for t in ts] for i in range(n)]
excitation_evolution_network3 = [[rho_network3(t)[i,i] for t in ts] for i in range(n)]


# figure , ax1 = plt.subplots(1 , 1)
# ax1.set_xlabel("t")
# ax1.set_ylabel("Excitation Evolution")
# ax1.plot(ts , excitation_evolution1)
# ax1.plot(ts , excitation_evolution2)
# ax1.plot(ts , excitation_evolution3)
# ax1.plot(ts , excitation_evolution4)
# plt.legend(["Strong Sink - Excitation Node","Weak Sink - Excitation Node","Strong Sink - Sink Node","Weak Sink - Sink Node"])
# plt.show()


figure , ax2 = plt.subplots(1 , 1)
ax2.set_xlabel("t")
ax2.set_ylabel("Excitation Evolution")
for i in range(n):
    ax2.plot(ts , excitation_evolution_network1[i])
    ax2.plot(ts , excitation_evolution_network2[i])
plt.legend(["Target Network" , "Bad Individual node [i]"])
plt.show()


figure , ax3 = plt.subplots(1 , 1)
ax3.set_xlabel("t")
ax3.set_ylabel("Excitation Evolution")
for i in range(n):
    ax3.plot(ts , excitation_evolution_network1[i])
    ax3.plot(ts , excitation_evolution_network3[i])
plt.legend(["Target Network" , "Good Individual"])
plt.show()
