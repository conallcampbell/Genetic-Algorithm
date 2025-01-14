import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.linalg import sqrtm

##########################################
# kronecker product
##########################################

def kron(*matrices):
    result = np.array([[1]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result

##########################################
# defining initial state
##########################################

n = 15
k = 1
rho0 = np.zeros((n , 1))
rho0[k] = 1
state = kron(rho0 , np.conjugate(rho0.T))
ts = np.linspace(0 , 1 , 10)

##########################################
# defining physical properties
##########################################
##########################################
# fidelity
##########################################

def fidelity(mat1 , mat2):
    return np.real((np.trace(sqrtm(sqrtm(mat1) @ mat2 @ sqrtm(mat1))))**2)

##########################################
# coherence
##########################################

def coherence(mat):
    sum = []
    for i in range(len(ts)):
        sum.append(np.abs(np.sum(mat[i][0]) - np.trace(mat[i][0])))
    return sum

##########################################
# population
##########################################

def populations(mat1 , mat2):
    pop = [[] for _ in range(len(ts))]
    for i in range(len(ts)):
        for j in range(n):
            pop[i].append(np.abs(np.real(mat1[i][0][j][j] - mat2[i][0][j][j])))
    return pop


##########################################
##########################################
#target individual
target1 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest1 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target1cc = nx.node_connected_component(nx.from_numpy_array(np.real(target1)) , 1)
fittest1cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest1)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr1 = target1 == fittest1
# Count the number of False values
false_count1 = np.sum(arr1 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t1f1_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target1)) , nx.from_numpy_array(np.real(fittest1)))
# WL graph isomorphism test
t1_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target1)))
f1_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest1)))
# degree distribution
t1_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target1))), key=lambda x: x[1], reverse=True)]
f1_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest1))), key=lambda x: x[1], reverse=True)]
# number of connections
t1_connections = np.sum(t1_dd)
f1_connections = np.sum(f1_dd)
# distance
distance1 = 0.27378673371202955
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest1_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest1 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest1 * t).T))

            ]
def fittest1_evolution_data():
    data = []
    for t in ts:
        data.append(fittest1_evolution(t))
    return data
#target
def target1_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target1 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target1 * t).T))

            ]
def target1_evolution_data():
    data = []
    for t in ts:
        data.append(target1_evolution(t))
    return data
# fidelity results
fidelity1_tab = []
for i in range(len(ts)):
    fidelity1_tab.append(fidelity(target1_evolution_data()[i][0] , fittest1_evolution_data()[i][0]))
fidelity1_tab = np.round(fidelity1_tab , decimals = 6)
# coherence results
t1_coherence = coherence(target1_evolution_data())
f1_coherence = coherence(fittest1_evolution_data())
t1f1_coherence = []
for i in range(len(ts)):
     t1f1_coherence.append(np.abs(t1_coherence[i] - f1_coherence[i]))
# population results
pop1 = []
for i in range(len(ts)):
     pop1.append(np.sum(populations(target1_evolution_data() , fittest1_evolution_data())[i]))

##########################################
##########################################
#target individual
target2 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest2 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target2cc = nx.node_connected_component(nx.from_numpy_array(np.real(target2)) , 1)
fittest2cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest2)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr2 = target2 == fittest2
# Count the number of False values
false_count2 = np.sum(arr2 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t2f2_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target2)) , nx.from_numpy_array(np.real(fittest2)))
# WL graph isomorphism test
t2_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target2)))
f2_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest2)))
# degree distribution
t2_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target2))), key=lambda x: x[1], reverse=True)]
f2_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest2))), key=lambda x: x[1], reverse=True)]
# number of connections
t2_connections = np.sum(t2_dd)
f2_connections = np.sum(f2_dd)
# distance
distance2 = 0.46265378793055456
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest2_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest2 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest2 * t).T))

            ]
def fittest2_evolution_data():
    data = []
    for t in ts:
        data.append(fittest2_evolution(t))
    return data
#target
def target2_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target2 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target2 * t).T))

            ]
def target2_evolution_data():
    data = []
    for t in ts:
        data.append(target2_evolution(t))
    return data
# fidelity results
fidelity2_tab = []
for i in range(len(ts)):
    fidelity2_tab.append(fidelity(target2_evolution_data()[i][0] , fittest2_evolution_data()[i][0]))
fidelity2_tab = np.round(fidelity2_tab , decimals = 6)
# coherence results
t2_coherence = coherence(target2_evolution_data())
f2_coherence = coherence(fittest2_evolution_data())
t2f2_coherence = []
for i in range(len(ts)):
     t2f2_coherence.append(np.abs(t2_coherence[i] - f2_coherence[i]))
# population results
pop2 = []
for i in range(len(ts)):
     pop2.append(np.sum(populations(target2_evolution_data() , fittest2_evolution_data())[i]))


##########################################
##########################################
#target individual
target3 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest3 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target3cc = nx.node_connected_component(nx.from_numpy_array(np.real(target3)) , 1)
fittest3cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest3)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr3 = target3 == fittest3
# Count the number of False values
false_count3 = np.sum(arr3 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t3f3_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target3)) , nx.from_numpy_array(np.real(fittest3)))
# WL graph isomorphism test
t3_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target3)))
f3_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest3)))
# degree distribution
t3_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target3))), key=lambda x: x[1], reverse=True)]
f3_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest3))), key=lambda x: x[1], reverse=True)]
# number of connections
t3_connections = np.sum(t3_dd)
f3_connections = np.sum(f3_dd)
# distance
distance3 = 0.20827209169662309
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest3_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest3 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest3 * t).T))

            ]
def fittest3_evolution_data():
    data = []
    for t in ts:
        data.append(fittest3_evolution(t))
    return data
#target
def target3_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target3 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target3 * t).T))

            ]
def target3_evolution_data():
    data = []
    for t in ts:
        data.append(target3_evolution(t))
    return data
# fidelity results
fidelity3_tab = []
for i in range(len(ts)):
    fidelity3_tab.append(fidelity(target3_evolution_data()[i][0] , fittest3_evolution_data()[i][0]))
fidelity3_tab = np.round(fidelity3_tab , decimals = 6)
# coherence results
t3_coherence = coherence(target3_evolution_data())
f3_coherence = coherence(fittest3_evolution_data())
t3f3_coherence = []
for i in range(len(ts)):
     t3f3_coherence.append(np.abs(t3_coherence[i] - f3_coherence[i]))
# population results
pop3 = []
for i in range(len(ts)):
     pop3.append(np.sum(populations(target3_evolution_data() , fittest3_evolution_data())[i]))


##########################################
##########################################
#target individual
target4 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest4 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target4cc = nx.node_connected_component(nx.from_numpy_array(np.real(target4)) , 1)
fittest4cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest4)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr4 = target4 == fittest4
# Count the number of False values
false_count4 = np.sum(arr4 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t4f4_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target4)) , nx.from_numpy_array(np.real(fittest4)))
# WL graph isomorphism test
t4_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target4)))
f4_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest4)))
# degree distribution
t4_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target4))), key=lambda x: x[1], reverse=True)]
f4_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest4))), key=lambda x: x[1], reverse=True)]
# number of connections
t4_connections = np.sum(t4_dd)
f4_connections = np.sum(f4_dd)
# distance
distance4 = 0.37001473537632823
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest4_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest4 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest4 * t).T))

            ]
def fittest4_evolution_data():
    data = []
    for t in ts:
        data.append(fittest4_evolution(t))
    return data
#target
def target4_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target4 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target4 * t).T))

            ]
def target4_evolution_data():
    data = []
    for t in ts:
        data.append(target4_evolution(t))
    return data
# fidelity results
fidelity4_tab = []
for i in range(len(ts)):
    fidelity4_tab.append(fidelity(target4_evolution_data()[i][0] , fittest4_evolution_data()[i][0]))
fidelity4_tab = np.round(fidelity4_tab , decimals = 6)
# coherence results
t4_coherence = coherence(target4_evolution_data())
f4_coherence = coherence(fittest4_evolution_data())
t4f4_coherence = []
for i in range(len(ts)):
     t4f4_coherence.append(np.abs(t4_coherence[i] - f4_coherence[i]))
# population results
pop4 = []
for i in range(len(ts)):
     pop4.append(np.sum(populations(target4_evolution_data() , fittest4_evolution_data())[i]))

##########################################
##########################################
#target individual
target5 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest5 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target5cc = nx.node_connected_component(nx.from_numpy_array(np.real(target5)) , 1)
fittest5cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest5)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr5 = target5 == fittest5
# Count the number of False values
false_count5 = np.sum(arr5 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t5f5_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target5)) , nx.from_numpy_array(np.real(fittest5)))
# WL graph isomorphism test
t5_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target5)))
f5_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest5)))
# degree distribution
t5_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target5))), key=lambda x: x[1], reverse=True)]
f5_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest5))), key=lambda x: x[1], reverse=True)]
# number of connections
t5_connections = np.sum(t5_dd)
f5_connections = np.sum(f5_dd)
# distance
distance5 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest5_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest5 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest5 * t).T))

            ]
def fittest5_evolution_data():
    data = []
    for t in ts:
        data.append(fittest5_evolution(t))
    return data
#target
def target5_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target5 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target5 * t).T))

            ]
def target5_evolution_data():
    data = []
    for t in ts:
        data.append(target5_evolution(t))
    return data
# fidelity results
fidelity5_tab = []
for i in range(len(ts)):
    fidelity5_tab.append(fidelity(target5_evolution_data()[i][0] , fittest5_evolution_data()[i][0]))
fidelity5_tab = np.round(fidelity5_tab , decimals = 6)
# coherence results
t5_coherence = coherence(target5_evolution_data())
f5_coherence = coherence(fittest5_evolution_data())
t5f5_coherence = []
for i in range(len(ts)):
     t5f5_coherence.append(np.abs(t5_coherence[i] - f5_coherence[i]))
# population results
pop5 = []
for i in range(len(ts)):
     pop5.append(np.sum(populations(target5_evolution_data() , fittest5_evolution_data())[i]))

##########################################
##########################################
#target individual
target6 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest6 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target6cc = nx.node_connected_component(nx.from_numpy_array(np.real(target6)) , 1)
fittest6cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest6)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr6 = target6 == fittest6
# Count the number of False values
false_count6 = np.sum(arr6 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t6f6_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target6)) , nx.from_numpy_array(np.real(fittest6)))
# WL graph isomorphism test
t6_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target6)))
f6_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest6)))
# degree distribution
t6_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target6))), key=lambda x: x[1], reverse=True)]
f6_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest6))), key=lambda x: x[1], reverse=True)]
# number of connections
t6_connections = np.sum(t6_dd)
f6_connections = np.sum(f6_dd)
# distance
distance6 = 0.07877938514930394
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest6_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest6 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest6 * t).T))

            ]
def fittest6_evolution_data():
    data = []
    for t in ts:
        data.append(fittest6_evolution(t))
    return data
#target
def target6_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target6 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target6 * t).T))

            ]
def target6_evolution_data():
    data = []
    for t in ts:
        data.append(target6_evolution(t))
    return data
# fidelity results
fidelity6_tab = []
for i in range(len(ts)):
    fidelity6_tab.append(fidelity(target6_evolution_data()[i][0] , fittest6_evolution_data()[i][0]))
fidelity6_tab = np.round(fidelity6_tab , decimals = 6)
# coherence results
t6_coherence = coherence(target6_evolution_data())
f6_coherence = coherence(fittest6_evolution_data())
t6f6_coherence = []
for i in range(len(ts)):
     t6f6_coherence.append(np.abs(t6_coherence[i] - f6_coherence[i]))
# population results
pop6 = []
for i in range(len(ts)):
     pop6.append(np.sum(populations(target6_evolution_data() , fittest6_evolution_data())[i]))

##########################################
##########################################
#target individual
target7 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest7 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target7cc = nx.node_connected_component(nx.from_numpy_array(np.real(target7)) , 1)
fittest7cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest7)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr7 = target7 == fittest7
# Count the number of False values
false_count7 = np.sum(arr7 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t7f7_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target7)) , nx.from_numpy_array(np.real(fittest7)))
# WL graph isomorphism test
t7_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target7)))
f7_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest7)))
# degree distribution
t7_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target7))), key=lambda x: x[1], reverse=True)]
f7_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest7))), key=lambda x: x[1], reverse=True)]
# number of connections
t7_connections = np.sum(t7_dd)
f7_connections = np.sum(f7_dd)
# distance
distance7 = 0.08360224312951392
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest7_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest7 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest7 * t).T))

            ]
def fittest7_evolution_data():
    data = []
    for t in ts:
        data.append(fittest7_evolution(t))
    return data
#target
def target7_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target7 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target7 * t).T))

            ]
def target7_evolution_data():
    data = []
    for t in ts:
        data.append(target7_evolution(t))
    return data
# fidelity results
fidelity7_tab = []
for i in range(len(ts)):
    fidelity7_tab.append(fidelity(target7_evolution_data()[i][0] , fittest7_evolution_data()[i][0]))
fidelity7_tab = np.round(fidelity7_tab , decimals = 6)
# coherence results
t7_coherence = coherence(target7_evolution_data())
f7_coherence = coherence(fittest7_evolution_data())
t7f7_coherence = []
for i in range(len(ts)):
     t7f7_coherence.append(np.abs(t7_coherence[i] - f7_coherence[i]))
# population results
pop7 = []
for i in range(len(ts)):
     pop7.append(np.sum(populations(target7_evolution_data() , fittest7_evolution_data())[i]))

##########################################
##########################################
#target individual
target8 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest8 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target8cc = nx.node_connected_component(nx.from_numpy_array(np.real(target8)) , 1)
fittest8cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest8)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr8 = target8 == fittest8
# Count the number of False values
false_count8 = np.sum(arr8 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t8f8_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target8)) , nx.from_numpy_array(np.real(fittest8)))
# WL graph isomorphism test
t8_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target8)))
f8_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest8)))
# degree distribution
t8_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target8))), key=lambda x: x[1], reverse=True)]
f8_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest8))), key=lambda x: x[1], reverse=True)]
# number of connections
t8_connections = np.sum(t8_dd)
f8_connections = np.sum(f8_dd)
# distance
distance8 = 0.0011961152608347403
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest8_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest8 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest8 * t).T))

            ]
def fittest8_evolution_data():
    data = []
    for t in ts:
        data.append(fittest8_evolution(t))
    return data
#target
def target8_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target8 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target8 * t).T))

            ]
def target8_evolution_data():
    data = []
    for t in ts:
        data.append(target8_evolution(t))
    return data
# fidelity results
fidelity8_tab = []
for i in range(len(ts)):
    fidelity8_tab.append(fidelity(target8_evolution_data()[i][0] , fittest8_evolution_data()[i][0]))
fidelity8_tab = np.round(fidelity8_tab , decimals = 6)
# coherence results
t8_coherence = coherence(target8_evolution_data())
f8_coherence = coherence(fittest8_evolution_data())
t8f8_coherence = []
for i in range(len(ts)):
     t8f8_coherence.append(np.abs(t8_coherence[i] - f8_coherence[i]))
# population results
pop8 = []
for i in range(len(ts)):
     pop8.append(np.sum(populations(target8_evolution_data() , fittest8_evolution_data())[i]))

##########################################
##########################################
#target individual
target9 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest9 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target9cc = nx.node_connected_component(nx.from_numpy_array(np.real(target9)) , 1)
fittest9cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest9)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr9 = target9 == fittest9
# Count the number of False values
false_count9 = np.sum(arr9 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t9f9_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target9)) , nx.from_numpy_array(np.real(fittest9)))
# WL graph isomorphism test
t9_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target9)))
f9_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest9)))
# degree distribution
t9_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target9))), key=lambda x: x[1], reverse=True)]
f9_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest9))), key=lambda x: x[1], reverse=True)]
# number of connections
t9_connections = np.sum(t9_dd)
f9_connections = np.sum(f9_dd)
# distance
distance9 = 0.30051584955527944
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest9_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest9 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest9 * t).T))

            ]
def fittest9_evolution_data():
    data = []
    for t in ts:
        data.append(fittest9_evolution(t))
    return data
#target
def target9_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target9 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target9 * t).T))

            ]
def target9_evolution_data():
    data = []
    for t in ts:
        data.append(target9_evolution(t))
    return data
# fidelity results
fidelity9_tab = []
for i in range(len(ts)):
    fidelity9_tab.append(fidelity(target9_evolution_data()[i][0] , fittest9_evolution_data()[i][0]))
fidelity9_tab = np.round(fidelity9_tab , decimals = 6)
# coherence results
t9_coherence = coherence(target9_evolution_data())
f9_coherence = coherence(fittest9_evolution_data())
t9f9_coherence = []
for i in range(len(ts)):
     t9f9_coherence.append(np.abs(t9_coherence[i] - f9_coherence[i]))
# population results
pop9 = []
for i in range(len(ts)):
     pop9.append(np.sum(populations(target9_evolution_data() , fittest9_evolution_data())[i]))


##########################################
##########################################
#target individual
target10 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest10 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target10cc = nx.node_connected_component(nx.from_numpy_array(np.real(target10)) , 1)
fittest10cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest10)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr10 = target10 == fittest10
# Count the number of False values
false_count10 = np.sum(arr10 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t10f10_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target10)) , nx.from_numpy_array(np.real(fittest10)))
# WL graph isomorphism test
t10_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target10)))
f10_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest10)))
# degree distribution
t10_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target10))), key=lambda x: x[1], reverse=True)]
f10_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest10))), key=lambda x: x[1], reverse=True)]
# number of connections
t10_connections = np.sum(t10_dd)
f10_connections = np.sum(f10_dd)
# distance
distance10 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest10_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest10 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest10 * t).T))

            ]
def fittest10_evolution_data():
    data = []
    for t in ts:
        data.append(fittest10_evolution(t))
    return data
#target
def target10_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target10 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target10 * t).T))

            ]
def target10_evolution_data():
    data = []
    for t in ts:
        data.append(target10_evolution(t))
    return data
# fidelity results
fidelity10_tab = []
for i in range(len(ts)):
    fidelity10_tab.append(fidelity(target10_evolution_data()[i][0] , fittest10_evolution_data()[i][0]))
fidelity10_tab = np.round(fidelity10_tab , decimals = 6)
# coherence results
t10_coherence = coherence(target10_evolution_data())
f10_coherence = coherence(fittest10_evolution_data())
t10f10_coherence = []
for i in range(len(ts)):
     t10f10_coherence.append(np.abs(t10_coherence[i] - f10_coherence[i]))
# population results
pop10 = []
for i in range(len(ts)):
     pop10.append(np.sum(populations(target10_evolution_data() , fittest10_evolution_data())[i]))






##########################################
##########################################
#target individual
target11 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest11 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target11cc = nx.node_connected_component(nx.from_numpy_array(np.real(target11)) , 1)
fittest11cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest11)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr11 = target11 == fittest11
# Count the number of False values
false_count11 = np.sum(arr11 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t11f11_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target11)) , nx.from_numpy_array(np.real(fittest11)))
# WL graph isomorphism test
t11_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target11)))
f11_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest11)))
# degree distribution
t11_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target11))), key=lambda x: x[1], reverse=True)]
f11_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest11))), key=lambda x: x[1], reverse=True)]
# number of connections
t11_connections = np.sum(t11_dd)
f11_connections = np.sum(f11_dd)
# distance
distance11 = 3.7683651958886344e-06
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest11_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest11 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest11 * t).T))

            ]
def fittest11_evolution_data():
    data = []
    for t in ts:
        data.append(fittest11_evolution(t))
    return data
#target
def target11_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target11 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target11 * t).T))

            ]
def target11_evolution_data():
    data = []
    for t in ts:
        data.append(target11_evolution(t))
    return data
# fidelity results
fidelity11_tab = []
for i in range(len(ts)):
    fidelity11_tab.append(fidelity(target11_evolution_data()[i][0] , fittest11_evolution_data()[i][0]))
fidelity11_tab = np.round(fidelity11_tab , decimals = 6)
# coherence results
t11_coherence = coherence(target11_evolution_data())
f11_coherence = coherence(fittest11_evolution_data())
t11f11_coherence = []
for i in range(len(ts)):
     t11f11_coherence.append(np.abs(t11_coherence[i] - f11_coherence[i]))
# population results
pop11 = []
for i in range(len(ts)):
     pop11.append(np.sum(populations(target11_evolution_data() , fittest11_evolution_data())[i]))

##########################################
##########################################
#target individual
target12 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest12 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target12cc = nx.node_connected_component(nx.from_numpy_array(np.real(target12)) , 1)
fittest12cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest12)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr12 = target12 == fittest12
# Count the number of False values
false_count12 = np.sum(arr12 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t12f12_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target12)) , nx.from_numpy_array(np.real(fittest12)))
# WL graph isomorphism test
t12_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target12)))
f12_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest12)))
# degree distribution
t12_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target12))), key=lambda x: x[1], reverse=True)]
f12_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest12))), key=lambda x: x[1], reverse=True)]
# number of connections
t12_connections = np.sum(t12_dd)
f12_connections = np.sum(f12_dd)
# distance
distance12 = 0.07553306672780247
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest12_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest12 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest12 * t).T))

            ]
def fittest12_evolution_data():
    data = []
    for t in ts:
        data.append(fittest12_evolution(t))
    return data
#target
def target12_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target12 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target12 * t).T))

            ]
def target12_evolution_data():
    data = []
    for t in ts:
        data.append(target12_evolution(t))
    return data
# fidelity results
fidelity12_tab = []
for i in range(len(ts)):
    fidelity12_tab.append(fidelity(target12_evolution_data()[i][0] , fittest12_evolution_data()[i][0]))
fidelity12_tab = np.round(fidelity12_tab , decimals = 6)
# coherence results
t12_coherence = coherence(target12_evolution_data())
f12_coherence = coherence(fittest12_evolution_data())
t12f12_coherence = []
for i in range(len(ts)):
     t12f12_coherence.append(np.abs(t12_coherence[i] - f12_coherence[i]))
# population results
pop12 = []
for i in range(len(ts)):
     pop12.append(np.sum(populations(target12_evolution_data() , fittest12_evolution_data())[i]))


##########################################
##########################################
#target individual
target13 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest13 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target13cc = nx.node_connected_component(nx.from_numpy_array(np.real(target13)) , 1)
fittest13cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest13)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr13 = target13 == fittest13
# Count the number of False values
false_count13 = np.sum(arr13 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t13f13_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target13)) , nx.from_numpy_array(np.real(fittest13)))
# WL graph isomorphism test
t13_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target13)))
f13_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest13)))
# degree distribution
t13_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target13))), key=lambda x: x[1], reverse=True)]
f13_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest13))), key=lambda x: x[1], reverse=True)]
# number of connections
t13_connections = np.sum(t13_dd)
f13_connections = np.sum(f13_dd)
# distance
distance13 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest13_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest13 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest13 * t).T))

            ]
def fittest13_evolution_data():
    data = []
    for t in ts:
        data.append(fittest13_evolution(t))
    return data
#target
def target13_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target13 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target13 * t).T))

            ]
def target13_evolution_data():
    data = []
    for t in ts:
        data.append(target13_evolution(t))
    return data
# fidelity results
fidelity13_tab = []
for i in range(len(ts)):
    fidelity13_tab.append(fidelity(target13_evolution_data()[i][0] , fittest13_evolution_data()[i][0]))
fidelity13_tab = np.round(fidelity13_tab , decimals = 6)
# coherence results
t13_coherence = coherence(target13_evolution_data())
f13_coherence = coherence(fittest13_evolution_data())
t13f13_coherence = []
for i in range(len(ts)):
     t13f13_coherence.append(np.abs(t13_coherence[i] - f13_coherence[i]))
# population results
pop13 = []
for i in range(len(ts)):
     pop13.append(np.sum(populations(target13_evolution_data() , fittest13_evolution_data())[i]))


##########################################
##########################################
#target individual
target14 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest14 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target14cc = nx.node_connected_component(nx.from_numpy_array(np.real(target14)) , 1)
fittest14cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest14)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr14 = target14 == fittest14
# Count the number of False values
false_count14 = np.sum(arr14 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t14f14_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target14)) , nx.from_numpy_array(np.real(fittest14)))
# WL graph isomorphism test
t14_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target14)))
f14_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest14)))
# degree distribution
t14_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target14))), key=lambda x: x[1], reverse=True)]
f14_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest14))), key=lambda x: x[1], reverse=True)]
# number of connections
t14_connections = np.sum(t14_dd)
f14_connections = np.sum(f14_dd)
# distance
distance14 = 0.4731560642140401
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest14_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest14 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest14 * t).T))

            ]
def fittest14_evolution_data():
    data = []
    for t in ts:
        data.append(fittest14_evolution(t))
    return data
#target
def target14_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target14 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target14 * t).T))

            ]
def target14_evolution_data():
    data = []
    for t in ts:
        data.append(target14_evolution(t))
    return data
# fidelity results
fidelity14_tab = []
for i in range(len(ts)):
    fidelity14_tab.append(fidelity(target14_evolution_data()[i][0] , fittest14_evolution_data()[i][0]))
fidelity14_tab = np.round(fidelity14_tab , decimals = 6)
# coherence results
t14_coherence = coherence(target14_evolution_data())
f14_coherence = coherence(fittest14_evolution_data())
t14f14_coherence = []
for i in range(len(ts)):
     t14f14_coherence.append(np.abs(t14_coherence[i] - f14_coherence[i]))
# population results
pop14 = []
for i in range(len(ts)):
     pop14.append(np.sum(populations(target14_evolution_data() , fittest14_evolution_data())[i]))

##########################################
##########################################
#target individual
target15 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest15 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target15cc = nx.node_connected_component(nx.from_numpy_array(np.real(target15)) , 1)
fittest15cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest15)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr15 = target15 == fittest15
# Count the number of False values
false_count15 = np.sum(arr15 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t15f15_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target15)) , nx.from_numpy_array(np.real(fittest15)))
# WL graph isomorphism test
t15_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target15)))
f15_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest15)))
# degree distribution
t15_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target15))), key=lambda x: x[1], reverse=True)]
f15_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest15))), key=lambda x: x[1], reverse=True)]
# number of connections
t15_connections = np.sum(t15_dd)
f15_connections = np.sum(f15_dd)
# distance
distance15 = 0.5884517942267272
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest15_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest15 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest15 * t).T))

            ]
def fittest15_evolution_data():
    data = []
    for t in ts:
        data.append(fittest15_evolution(t))
    return data
#target
def target15_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target15 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target15 * t).T))

            ]
def target15_evolution_data():
    data = []
    for t in ts:
        data.append(target15_evolution(t))
    return data
# fidelity results
fidelity15_tab = []
for i in range(len(ts)):
    fidelity15_tab.append(fidelity(target15_evolution_data()[i][0] , fittest15_evolution_data()[i][0]))
fidelity15_tab = np.round(fidelity15_tab , decimals = 6)
# coherence results
t15_coherence = coherence(target15_evolution_data())
f15_coherence = coherence(fittest15_evolution_data())
t15f15_coherence = []
for i in range(len(ts)):
     t15f15_coherence.append(np.abs(t15_coherence[i] - f15_coherence[i]))
# population results
pop15 = []
for i in range(len(ts)):
     pop15.append(np.sum(populations(target15_evolution_data() , fittest15_evolution_data())[i]))

##########################################
##########################################
#target individual
target16 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest16 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target16cc = nx.node_connected_component(nx.from_numpy_array(np.real(target16)) , 1)
fittest16cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest16)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr16 = target16 == fittest16
# Count the number of False values
false_count16 = np.sum(arr16 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t16f16_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target16)) , nx.from_numpy_array(np.real(fittest16)))
# WL graph isomorphism test
t16_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target16)))
f16_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest16)))
# degree distribution
t16_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target16))), key=lambda x: x[1], reverse=True)]
f16_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest16))), key=lambda x: x[1], reverse=True)]
# number of connections
t16_connections = np.sum(t16_dd)
f16_connections = np.sum(f16_dd)
# distance
distance16 = 0.010786835317397436
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest16_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest16 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest16 * t).T))

            ]
def fittest16_evolution_data():
    data = []
    for t in ts:
        data.append(fittest16_evolution(t))
    return data
#target
def target16_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target16 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target16 * t).T))

            ]
def target16_evolution_data():
    data = []
    for t in ts:
        data.append(target16_evolution(t))
    return data
# fidelity results
fidelity16_tab = []
for i in range(len(ts)):
    fidelity16_tab.append(fidelity(target16_evolution_data()[i][0] , fittest16_evolution_data()[i][0]))
fidelity16_tab = np.round(fidelity16_tab , decimals = 16)
# coherence results
t16_coherence = coherence(target16_evolution_data())
f16_coherence = coherence(fittest16_evolution_data())
t16f16_coherence = []
for i in range(len(ts)):
     t16f16_coherence.append(np.abs(t16_coherence[i] - f16_coherence[i]))
# population results
pop16 = []
for i in range(len(ts)):
     pop16.append(np.sum(populations(target16_evolution_data() , fittest16_evolution_data())[i]))



##########################################
##########################################
#target individual
target17 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest17 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target17cc = nx.node_connected_component(nx.from_numpy_array(np.real(target17)) , 1)
fittest17cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest17)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr17 = target17 == fittest17
# Count the number of False values
false_count17 = np.sum(arr17 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t17f17_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target17)) , nx.from_numpy_array(np.real(fittest17)))
# WL graph isomorphism test
t17_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target17)))
f17_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest17)))
# degree distribution
t17_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target17))), key=lambda x: x[1], reverse=True)]
f17_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest17))), key=lambda x: x[1], reverse=True)]
# number of connections
t17_connections = np.sum(t17_dd)
f17_connections = np.sum(f17_dd)
# distance
distance17 = 0.08644204975830727
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest17_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest17 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest17 * t).T))

            ]
def fittest17_evolution_data():
    data = []
    for t in ts:
        data.append(fittest17_evolution(t))
    return data
#target
def target17_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target17 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target17 * t).T))

            ]
def target17_evolution_data():
    data = []
    for t in ts:
        data.append(target17_evolution(t))
    return data
# fidelity results
fidelity17_tab = []
for i in range(len(ts)):
    fidelity17_tab.append(fidelity(target17_evolution_data()[i][0] , fittest17_evolution_data()[i][0]))
fidelity17_tab = np.round(fidelity17_tab , decimals = 17)
# coherence results
t17_coherence = coherence(target17_evolution_data())
f17_coherence = coherence(fittest17_evolution_data())
t17f17_coherence = []
for i in range(len(ts)):
     t17f17_coherence.append(np.abs(t17_coherence[i] - f17_coherence[i]))
# population results
pop17 = []
for i in range(len(ts)):
     pop17.append(np.sum(populations(target17_evolution_data() , fittest17_evolution_data())[i]))




##########################################
##########################################
#target individual
target18 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest18 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target18cc = nx.node_connected_component(nx.from_numpy_array(np.real(target18)) , 1)
fittest18cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest18)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr18 = target18 == fittest18
# Count the number of False values
false_count18 = np.sum(arr18 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t18f18_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target18)) , nx.from_numpy_array(np.real(fittest18)))
# WL graph isomorphism test
t18_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target18)))
f18_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest18)))
# degree distribution
t18_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target18))), key=lambda x: x[1], reverse=True)]
f18_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest18))), key=lambda x: x[1], reverse=True)]
# number of connections
t18_connections = np.sum(t18_dd)
f18_connections = np.sum(f18_dd)
# distance
distance18 = 0.3734711971524768
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest18_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest18 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest18 * t).T))

            ]
def fittest18_evolution_data():
    data = []
    for t in ts:
        data.append(fittest18_evolution(t))
    return data
#target
def target18_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target18 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target18 * t).T))

            ]
def target18_evolution_data():
    data = []
    for t in ts:
        data.append(target18_evolution(t))
    return data
# fidelity results
fidelity18_tab = []
for i in range(len(ts)):
    fidelity18_tab.append(fidelity(target18_evolution_data()[i][0] , fittest18_evolution_data()[i][0]))
fidelity18_tab = np.round(fidelity18_tab , decimals = 18)
# coherence results
t18_coherence = coherence(target18_evolution_data())
f18_coherence = coherence(fittest18_evolution_data())
t18f18_coherence = []
for i in range(len(ts)):
     t18f18_coherence.append(np.abs(t18_coherence[i] - f18_coherence[i]))
# population results
pop18 = []
for i in range(len(ts)):
     pop18.append(np.sum(populations(target18_evolution_data() , fittest18_evolution_data())[i]))
    





##########################################
##########################################
#target individual
target19 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest19 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target19cc = nx.node_connected_component(nx.from_numpy_array(np.real(target19)) , 1)
fittest19cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest19)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr19 = target19 == fittest19
# Count the number of False values
false_count19 = np.sum(arr19 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t19f19_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target19)) , nx.from_numpy_array(np.real(fittest19)))
# WL graph isomorphism test
t19_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target19)))
f19_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest19)))
# degree distribution
t19_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target19))), key=lambda x: x[1], reverse=True)]
f19_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest19))), key=lambda x: x[1], reverse=True)]
# number of connections
t19_connections = np.sum(t19_dd)
f19_connections = np.sum(f19_dd)
# distance
distance19 = 0.04225035454420423
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest19_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest19 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest19 * t).T))

            ]
def fittest19_evolution_data():
    data = []
    for t in ts:
        data.append(fittest19_evolution(t))
    return data
#target
def target19_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target19 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target19 * t).T))

            ]
def target19_evolution_data():
    data = []
    for t in ts:
        data.append(target19_evolution(t))
    return data
# fidelity results
fidelity19_tab = []
for i in range(len(ts)):
    fidelity19_tab.append(fidelity(target19_evolution_data()[i][0] , fittest19_evolution_data()[i][0]))
fidelity19_tab = np.round(fidelity19_tab , decimals = 19)
# coherence results
t19_coherence = coherence(target19_evolution_data())
f19_coherence = coherence(fittest19_evolution_data())
t19f19_coherence = []
for i in range(len(ts)):
     t19f19_coherence.append(np.abs(t19_coherence[i] - f19_coherence[i]))
# population results
pop19 = []
for i in range(len(ts)):
     pop19.append(np.sum(populations(target19_evolution_data() , fittest19_evolution_data())[i]))





##########################################
##########################################
#target individual
target20 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest20 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target20cc = nx.node_connected_component(nx.from_numpy_array(np.real(target20)) , 1)
fittest20cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest20)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr20 = target20 == fittest20
# Count the number of False values
false_count20 = np.sum(arr20 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t20f20_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target20)) , nx.from_numpy_array(np.real(fittest20)))
# WL graph isomorphism test
t20_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target20)))
f20_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest20)))
# degree distribution
t20_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target20))), key=lambda x: x[1], reverse=True)]
f20_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest20))), key=lambda x: x[1], reverse=True)]
# number of connections
t20_connections = np.sum(t20_dd)
f20_connections = np.sum(f20_dd)
# distance
distance20 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest20_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest20 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest20 * t).T))

            ]
def fittest20_evolution_data():
    data = []
    for t in ts:
        data.append(fittest20_evolution(t))
    return data
#target
def target20_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target20 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target20 * t).T))

            ]
def target20_evolution_data():
    data = []
    for t in ts:
        data.append(target20_evolution(t))
    return data
# fidelity results
fidelity20_tab = []
for i in range(len(ts)):
    fidelity20_tab.append(fidelity(target20_evolution_data()[i][0] , fittest20_evolution_data()[i][0]))
fidelity20_tab = np.round(fidelity20_tab , decimals = 20)
# coherence results
t20_coherence = coherence(target20_evolution_data())
f20_coherence = coherence(fittest20_evolution_data())
t20f20_coherence = []
for i in range(len(ts)):
     t20f20_coherence.append(np.abs(t20_coherence[i] - f20_coherence[i]))
# population results
pop20 = []
for i in range(len(ts)):
     pop20.append(np.sum(populations(target20_evolution_data() , fittest20_evolution_data())[i]))




##########################################
##########################################
#target individual
target21 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest21 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target21cc = nx.node_connected_component(nx.from_numpy_array(np.real(target21)) , 1)
fittest21cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest21)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr21 = target21 == fittest21
# Count the number of False values
false_count21 = np.sum(arr21 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t21f21_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target21)) , nx.from_numpy_array(np.real(fittest21)))
# WL graph isomorphism test
t21_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target21)))
f21_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest21)))
# degree distribution
t21_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target21))), key=lambda x: x[1], reverse=True)]
f21_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest21))), key=lambda x: x[1], reverse=True)]
# number of connections
t21_connections = np.sum(t21_dd)
f21_connections = np.sum(f21_dd)
# distance
distance21 = 0.4043712856649919
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest21_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest21 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest21 * t).T))

            ]
def fittest21_evolution_data():
    data = []
    for t in ts:
        data.append(fittest21_evolution(t))
    return data
#target
def target21_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target21 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target21 * t).T))

            ]
def target21_evolution_data():
    data = []
    for t in ts:
        data.append(target21_evolution(t))
    return data
# fidelity results
fidelity21_tab = []
for i in range(len(ts)):
    fidelity21_tab.append(fidelity(target21_evolution_data()[i][0] , fittest21_evolution_data()[i][0]))
fidelity21_tab = np.round(fidelity21_tab , decimals = 21)
# coherence results
t21_coherence = coherence(target21_evolution_data())
f21_coherence = coherence(fittest21_evolution_data())
t21f21_coherence = []
for i in range(len(ts)):
     t21f21_coherence.append(np.abs(t21_coherence[i] - f21_coherence[i]))
# population results
pop21 = []
for i in range(len(ts)):
     pop21.append(np.sum(populations(target21_evolution_data() , fittest21_evolution_data())[i]))





##########################################
##########################################
#target individual
target22 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest22 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target22cc = nx.node_connected_component(nx.from_numpy_array(np.real(target22)) , 1)
fittest22cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest22)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr22 = target22 == fittest22
# Count the number of False values
false_count22 = np.sum(arr22 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t22f22_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target22)) , nx.from_numpy_array(np.real(fittest22)))
# WL graph isomorphism test
t22_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target22)))
f22_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest22)))
# degree distribution
t22_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target22))), key=lambda x: x[1], reverse=True)]
f22_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest22))), key=lambda x: x[1], reverse=True)]
# number of connections
t22_connections = np.sum(t22_dd)
f22_connections = np.sum(f22_dd)
# distance
distance22 = 0.28860780724784707
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest22_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest22 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest22 * t).T))

            ]
def fittest22_evolution_data():
    data = []
    for t in ts:
        data.append(fittest22_evolution(t))
    return data
#target
def target22_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target22 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target22 * t).T))

            ]
def target22_evolution_data():
    data = []
    for t in ts:
        data.append(target22_evolution(t))
    return data
# fidelity results
fidelity22_tab = []
for i in range(len(ts)):
    fidelity22_tab.append(fidelity(target22_evolution_data()[i][0] , fittest22_evolution_data()[i][0]))
fidelity22_tab = np.round(fidelity22_tab , decimals = 22)
# coherence results
t22_coherence = coherence(target22_evolution_data())
f22_coherence = coherence(fittest22_evolution_data())
t22f22_coherence = []
for i in range(len(ts)):
     t22f22_coherence.append(np.abs(t22_coherence[i] - f22_coherence[i]))
# population results
pop22 = []
for i in range(len(ts)):
     pop22.append(np.sum(populations(target22_evolution_data() , fittest22_evolution_data())[i]))




##########################################
##########################################
#target individual
target23 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest23 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target23cc = nx.node_connected_component(nx.from_numpy_array(np.real(target23)) , 1)
fittest23cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest23)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr23 = target23 == fittest23
# Count the number of False values
false_count23 = np.sum(arr23 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t23f23_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target23)) , nx.from_numpy_array(np.real(fittest23)))
# WL graph isomorphism test
t23_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target23)))
f23_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest23)))
# degree distribution
t23_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target23))), key=lambda x: x[1], reverse=True)]
f23_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest23))), key=lambda x: x[1], reverse=True)]
# number of connections
t23_connections = np.sum(t23_dd)
f23_connections = np.sum(f23_dd)
# distance
distance23 = 0.1290223759372866
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest23_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest23 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest23 * t).T))

            ]
def fittest23_evolution_data():
    data = []
    for t in ts:
        data.append(fittest23_evolution(t))
    return data
#target
def target23_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target23 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target23 * t).T))

            ]
def target23_evolution_data():
    data = []
    for t in ts:
        data.append(target23_evolution(t))
    return data
# fidelity results
fidelity23_tab = []
for i in range(len(ts)):
    fidelity23_tab.append(fidelity(target23_evolution_data()[i][0] , fittest23_evolution_data()[i][0]))
fidelity23_tab = np.round(fidelity23_tab , decimals = 23)
# coherence results
t23_coherence = coherence(target23_evolution_data())
f23_coherence = coherence(fittest23_evolution_data())
t23f23_coherence = []
for i in range(len(ts)):
     t23f23_coherence.append(np.abs(t23_coherence[i] - f23_coherence[i]))
# population results
pop23 = []
for i in range(len(ts)):
     pop23.append(np.sum(populations(target23_evolution_data() , fittest23_evolution_data())[i]))





##########################################
##########################################
#target individual
target24 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest24 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target24cc = nx.node_connected_component(nx.from_numpy_array(np.real(target24)) , 1)
fittest24cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest24)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr24 = target24 == fittest24
# Count the number of False values
false_count24 = np.sum(arr24 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t24f24_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target24)) , nx.from_numpy_array(np.real(fittest24)))
# WL graph isomorphism test
t24_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target24)))
f24_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest24)))
# degree distribution
t24_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target24))), key=lambda x: x[1], reverse=True)]
f24_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest24))), key=lambda x: x[1], reverse=True)]
# number of connections
t24_connections = np.sum(t24_dd)
f24_connections = np.sum(f24_dd)
# distance
distance24 = 0.15242126055525285
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest24_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest24 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest24 * t).T))

            ]
def fittest24_evolution_data():
    data = []
    for t in ts:
        data.append(fittest24_evolution(t))
    return data
#target
def target24_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target24 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target24 * t).T))

            ]
def target24_evolution_data():
    data = []
    for t in ts:
        data.append(target24_evolution(t))
    return data
# fidelity results
fidelity24_tab = []
for i in range(len(ts)):
    fidelity24_tab.append(fidelity(target24_evolution_data()[i][0] , fittest24_evolution_data()[i][0]))
fidelity24_tab = np.round(fidelity24_tab , decimals = 24)
# coherence results
t24_coherence = coherence(target24_evolution_data())
f24_coherence = coherence(fittest24_evolution_data())
t24f24_coherence = []
for i in range(len(ts)):
     t24f24_coherence.append(np.abs(t24_coherence[i] - f24_coherence[i]))
# population results
pop24 = []
for i in range(len(ts)):
     pop24.append(np.sum(populations(target24_evolution_data() , fittest24_evolution_data())[i]))





##########################################
##########################################
#target individual
target25 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest25 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target25cc = nx.node_connected_component(nx.from_numpy_array(np.real(target25)) , 1)
fittest25cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest25)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr25 = target25 == fittest25
# Count the number of False values
false_count25 = np.sum(arr25 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t25f25_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target25)) , nx.from_numpy_array(np.real(fittest25)))
# WL graph isomorphism test
t25_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target25)))
f25_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest25)))
# degree distribution
t25_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target25))), key=lambda x: x[1], reverse=True)]
f25_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest25))), key=lambda x: x[1], reverse=True)]
# number of connections
t25_connections = np.sum(t25_dd)
f25_connections = np.sum(f25_dd)
# distance
distance25 = 0.36253683203276277
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest25_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest25 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest25 * t).T))

            ]
def fittest25_evolution_data():
    data = []
    for t in ts:
        data.append(fittest25_evolution(t))
    return data
#target
def target25_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target25 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target25 * t).T))

            ]
def target25_evolution_data():
    data = []
    for t in ts:
        data.append(target25_evolution(t))
    return data
# fidelity results
fidelity25_tab = []
for i in range(len(ts)):
    fidelity25_tab.append(fidelity(target25_evolution_data()[i][0] , fittest25_evolution_data()[i][0]))
fidelity25_tab = np.round(fidelity25_tab , decimals = 25)
# coherence results
t25_coherence = coherence(target25_evolution_data())
f25_coherence = coherence(fittest25_evolution_data())
t25f25_coherence = []
for i in range(len(ts)):
     t25f25_coherence.append(np.abs(t25_coherence[i] - f25_coherence[i]))
# population results
pop25 = []
for i in range(len(ts)):
     pop25.append(np.sum(populations(target25_evolution_data() , fittest25_evolution_data())[i]))




##########################################
##########################################
#target individual
target26 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest26 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target26cc = nx.node_connected_component(nx.from_numpy_array(np.real(target26)) , 1)
fittest26cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest26)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr26 = target26 == fittest26
# Count the number of False values
false_count26 = np.sum(arr26 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t26f26_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target26)) , nx.from_numpy_array(np.real(fittest26)))
# WL graph isomorphism test
t26_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target26)))
f26_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest26)))
# degree distribution
t26_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target26))), key=lambda x: x[1], reverse=True)]
f26_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest26))), key=lambda x: x[1], reverse=True)]
# number of connections
t26_connections = np.sum(t26_dd)
f26_connections = np.sum(f26_dd)
# distance
distance26 = 0.4952825464025835
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest26_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest26 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest26 * t).T))

            ]
def fittest26_evolution_data():
    data = []
    for t in ts:
        data.append(fittest26_evolution(t))
    return data
#target
def target26_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target26 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target26 * t).T))

            ]
def target26_evolution_data():
    data = []
    for t in ts:
        data.append(target26_evolution(t))
    return data
# fidelity results
fidelity26_tab = []
for i in range(len(ts)):
    fidelity26_tab.append(fidelity(target26_evolution_data()[i][0] , fittest26_evolution_data()[i][0]))
fidelity26_tab = np.round(fidelity26_tab , decimals = 26)
# coherence results
t26_coherence = coherence(target26_evolution_data())
f26_coherence = coherence(fittest26_evolution_data())
t26f26_coherence = []
for i in range(len(ts)):
     t26f26_coherence.append(np.abs(t26_coherence[i] - f26_coherence[i]))
# population results
pop26 = []
for i in range(len(ts)):
     pop26.append(np.sum(populations(target26_evolution_data() , fittest26_evolution_data())[i]))





##########################################
##########################################
#target individual
target27 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest27 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target27cc = nx.node_connected_component(nx.from_numpy_array(np.real(target27)) , 1)
fittest27cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest27)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr27 = target27 == fittest27
# Count the number of False values
false_count27 = np.sum(arr27 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t27f27_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target27)) , nx.from_numpy_array(np.real(fittest27)))
# WL graph isomorphism test
t27_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target27)))
f27_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest27)))
# degree distribution
t27_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target27))), key=lambda x: x[1], reverse=True)]
f27_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest27))), key=lambda x: x[1], reverse=True)]
# number of connections
t27_connections = np.sum(t27_dd)
f27_connections = np.sum(f27_dd)
# distance
distance27 = 0.6556495128451959
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest27_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest27 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest27 * t).T))

            ]
def fittest27_evolution_data():
    data = []
    for t in ts:
        data.append(fittest27_evolution(t))
    return data
#target
def target27_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target27 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target27 * t).T))

            ]
def target27_evolution_data():
    data = []
    for t in ts:
        data.append(target27_evolution(t))
    return data
# fidelity results
fidelity27_tab = []
for i in range(len(ts)):
    fidelity27_tab.append(fidelity(target27_evolution_data()[i][0] , fittest27_evolution_data()[i][0]))
fidelity27_tab = np.round(fidelity27_tab , decimals = 27)
# coherence results
t27_coherence = coherence(target27_evolution_data())
f27_coherence = coherence(fittest27_evolution_data())
t27f27_coherence = []
for i in range(len(ts)):
     t27f27_coherence.append(np.abs(t27_coherence[i] - f27_coherence[i]))
# population results
pop27 = []
for i in range(len(ts)):
     pop27.append(np.sum(populations(target27_evolution_data() , fittest27_evolution_data())[i]))
     




##########################################
##########################################
#target individual
target28 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest28 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target28cc = nx.node_connected_component(nx.from_numpy_array(np.real(target28)) , 1)
fittest28cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest28)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr28 = target28 == fittest28
# Count the number of False values
false_count28 = np.sum(arr28 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t28f28_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target28)) , nx.from_numpy_array(np.real(fittest28)))
# WL graph isomorphism test
t28_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target28)))
f28_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest28)))
# degree distribution
t28_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target28))), key=lambda x: x[1], reverse=True)]
f28_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest28))), key=lambda x: x[1], reverse=True)]
# number of connections
t28_connections = np.sum(t28_dd)
f28_connections = np.sum(f28_dd)
# distance
distance28 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest28_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest28 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest28 * t).T))

            ]
def fittest28_evolution_data():
    data = []
    for t in ts:
        data.append(fittest28_evolution(t))
    return data
#target
def target28_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target28 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target28 * t).T))

            ]
def target28_evolution_data():
    data = []
    for t in ts:
        data.append(target28_evolution(t))
    return data
# fidelity results
fidelity28_tab = []
for i in range(len(ts)):
    fidelity28_tab.append(fidelity(target28_evolution_data()[i][0] , fittest28_evolution_data()[i][0]))
fidelity28_tab = np.round(fidelity28_tab , decimals = 28)
# coherence results
t28_coherence = coherence(target28_evolution_data())
f28_coherence = coherence(fittest28_evolution_data())
t28f28_coherence = []
for i in range(len(ts)):
     t28f28_coherence.append(np.abs(t28_coherence[i] - f28_coherence[i]))
# population results
pop28 = []
for i in range(len(ts)):
     pop28.append(np.sum(populations(target28_evolution_data() , fittest28_evolution_data())[i]))




##########################################
##########################################
#target individual
target29 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest29 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target29cc = nx.node_connected_component(nx.from_numpy_array(np.real(target29)) , 1)
fittest29cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest29)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr29 = target29 == fittest29
# Count the number of False values
false_count29 = np.sum(arr29 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t29f29_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target29)) , nx.from_numpy_array(np.real(fittest29)))
# WL graph isomorphism test
t29_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target29)))
f29_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest29)))
# degree distribution
t29_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target29))), key=lambda x: x[1], reverse=True)]
f29_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest29))), key=lambda x: x[1], reverse=True)]
# number of connections
t29_connections = np.sum(t29_dd)
f29_connections = np.sum(f29_dd)
# distance
distance29 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest29_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest29 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest29 * t).T))

            ]
def fittest29_evolution_data():
    data = []
    for t in ts:
        data.append(fittest29_evolution(t))
    return data
#target
def target29_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target29 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target29 * t).T))

            ]
def target29_evolution_data():
    data = []
    for t in ts:
        data.append(target29_evolution(t))
    return data
# fidelity results
fidelity29_tab = []
for i in range(len(ts)):
    fidelity29_tab.append(fidelity(target29_evolution_data()[i][0] , fittest29_evolution_data()[i][0]))
fidelity29_tab = np.round(fidelity29_tab , decimals = 29)
# coherence results
t29_coherence = coherence(target29_evolution_data())
f29_coherence = coherence(fittest29_evolution_data())
t29f29_coherence = []
for i in range(len(ts)):
     t29f29_coherence.append(np.abs(t29_coherence[i] - f29_coherence[i]))
# population results
pop29 = []
for i in range(len(ts)):
     pop29.append(np.sum(populations(target29_evolution_data() , fittest29_evolution_data())[i]))





##########################################
##########################################
#target individual
target30 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest30 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target30cc = nx.node_connected_component(nx.from_numpy_array(np.real(target30)) , 1)
fittest30cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest30)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr30 = target30 == fittest30
# Count the number of False values
false_count30 = np.sum(arr30 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t30f30_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target30)) , nx.from_numpy_array(np.real(fittest30)))
# WL graph isomorphism test
t30_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target30)))
f30_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest30)))
# degree distribution
t30_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target30))), key=lambda x: x[1], reverse=True)]
f30_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest30))), key=lambda x: x[1], reverse=True)]
# number of connections
t30_connections = np.sum(t30_dd)
f30_connections = np.sum(f30_dd)
# distance
distance30 = 0.43864738980831897
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest30_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest30 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest30 * t).T))

            ]
def fittest30_evolution_data():
    data = []
    for t in ts:
        data.append(fittest30_evolution(t))
    return data
#target
def target30_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target30 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target30 * t).T))

            ]
def target30_evolution_data():
    data = []
    for t in ts:
        data.append(target30_evolution(t))
    return data
# fidelity results
fidelity30_tab = []
for i in range(len(ts)):
    fidelity30_tab.append(fidelity(target30_evolution_data()[i][0] , fittest30_evolution_data()[i][0]))
fidelity30_tab = np.round(fidelity30_tab , decimals = 30)
# coherence results
t30_coherence = coherence(target30_evolution_data())
f30_coherence = coherence(fittest30_evolution_data())
t30f30_coherence = []
for i in range(len(ts)):
     t30f30_coherence.append(np.abs(t30_coherence[i] - f30_coherence[i]))
# population results
pop30 = []
for i in range(len(ts)):
     pop30.append(np.sum(populations(target30_evolution_data() , fittest30_evolution_data())[i]))





##########################################
##########################################
#target individual
target31 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest31 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target31cc = nx.node_connected_component(nx.from_numpy_array(np.real(target31)) , 1)
fittest31cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest31)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr31 = target31 == fittest31
# Count the number of False values
false_count31 = np.sum(arr31 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t31f31_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target31)) , nx.from_numpy_array(np.real(fittest31)))
# WL graph isomorphism test
t31_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target31)))
f31_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest31)))
# degree distribution
t31_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target31))), key=lambda x: x[1], reverse=True)]
f31_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest31))), key=lambda x: x[1], reverse=True)]
# number of connections
t31_connections = np.sum(t31_dd)
f31_connections = np.sum(f31_dd)
# distance
distance31 = 0.12873467488093426
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest31_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest31 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest31 * t).T))

            ]
def fittest31_evolution_data():
    data = []
    for t in ts:
        data.append(fittest31_evolution(t))
    return data
#target
def target31_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target31 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target31 * t).T))

            ]
def target31_evolution_data():
    data = []
    for t in ts:
        data.append(target31_evolution(t))
    return data
# fidelity results
fidelity31_tab = []
for i in range(len(ts)):
    fidelity31_tab.append(fidelity(target31_evolution_data()[i][0] , fittest31_evolution_data()[i][0]))
fidelity31_tab = np.round(fidelity31_tab , decimals = 31)
# coherence results
t31_coherence = coherence(target31_evolution_data())
f31_coherence = coherence(fittest31_evolution_data())
t31f31_coherence = []
for i in range(len(ts)):
     t31f31_coherence.append(np.abs(t31_coherence[i] - f31_coherence[i]))
# population results
pop31 = []
for i in range(len(ts)):
     pop31.append(np.sum(populations(target31_evolution_data() , fittest31_evolution_data())[i]))





##########################################
##########################################
#target individual
target32 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest32 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target32cc = nx.node_connected_component(nx.from_numpy_array(np.real(target32)) , 1)
fittest32cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest32)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr32 = target32 == fittest32
# Count the number of False values
false_count32 = np.sum(arr32 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t32f32_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target32)) , nx.from_numpy_array(np.real(fittest32)))
# WL graph isomorphism test
t32_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target32)))
f32_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest32)))
# degree distribution
t32_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target32))), key=lambda x: x[1], reverse=True)]
f32_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest32))), key=lambda x: x[1], reverse=True)]
# number of connections
t32_connections = np.sum(t32_dd)
f32_connections = np.sum(f32_dd)
# distance
distance32 = 0.31354632383979497
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest32_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest32 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest32 * t).T))

            ]
def fittest32_evolution_data():
    data = []
    for t in ts:
        data.append(fittest32_evolution(t))
    return data
#target
def target32_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target32 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target32 * t).T))

            ]
def target32_evolution_data():
    data = []
    for t in ts:
        data.append(target32_evolution(t))
    return data
# fidelity results
fidelity32_tab = []
for i in range(len(ts)):
    fidelity32_tab.append(fidelity(target32_evolution_data()[i][0] , fittest32_evolution_data()[i][0]))
fidelity32_tab = np.round(fidelity32_tab , decimals = 32)
# coherence results
t32_coherence = coherence(target32_evolution_data())
f32_coherence = coherence(fittest32_evolution_data())
t32f32_coherence = []
for i in range(len(ts)):
     t32f32_coherence.append(np.abs(t32_coherence[i] - f32_coherence[i]))
# population results
pop32 = []
for i in range(len(ts)):
     pop32.append(np.sum(populations(target32_evolution_data() , fittest32_evolution_data())[i]))




##########################################
##########################################
#target individual
target33 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest33 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target33cc = nx.node_connected_component(nx.from_numpy_array(np.real(target33)) , 1)
fittest33cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest33)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr33 = target33 == fittest33
# Count the number of False values
false_count33 = np.sum(arr33 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t33f33_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target33)) , nx.from_numpy_array(np.real(fittest33)))
# WL graph isomorphism test
t33_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target33)))
f33_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest33)))
# degree distribution
t33_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target33))), key=lambda x: x[1], reverse=True)]
f33_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest33))), key=lambda x: x[1], reverse=True)]
# number of connections
t33_connections = np.sum(t33_dd)
f33_connections = np.sum(f33_dd)
# distance
distance33 = 0.0007224120661640798
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest33_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest33 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest33 * t).T))

            ]
def fittest33_evolution_data():
    data = []
    for t in ts:
        data.append(fittest33_evolution(t))
    return data
#target
def target33_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target33 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target33 * t).T))

            ]
def target33_evolution_data():
    data = []
    for t in ts:
        data.append(target33_evolution(t))
    return data
# fidelity results
fidelity33_tab = []
for i in range(len(ts)):
    fidelity33_tab.append(fidelity(target33_evolution_data()[i][0] , fittest33_evolution_data()[i][0]))
fidelity33_tab = np.round(fidelity33_tab , decimals = 33)
# coherence results
t33_coherence = coherence(target33_evolution_data())
f33_coherence = coherence(fittest33_evolution_data())
t33f33_coherence = []
for i in range(len(ts)):
     t33f33_coherence.append(np.abs(t33_coherence[i] - f33_coherence[i]))
# population results
pop33 = []
for i in range(len(ts)):
     pop33.append(np.sum(populations(target33_evolution_data() , fittest33_evolution_data())[i]))




##########################################
##########################################
#target individual
target34 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest34 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target34cc = nx.node_connected_component(nx.from_numpy_array(np.real(target34)) , 1)
fittest34cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest34)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr34 = target34 == fittest34
# Count the number of False values
false_count34 = np.sum(arr34 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t34f34_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target34)) , nx.from_numpy_array(np.real(fittest34)))
# WL graph isomorphism test
t34_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target34)))
f34_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest34)))
# degree distribution
t34_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target34))), key=lambda x: x[1], reverse=True)]
f34_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest34))), key=lambda x: x[1], reverse=True)]
# number of connections
t34_connections = np.sum(t34_dd)
f34_connections = np.sum(f34_dd)
# distance
distance34 = 0.3666602511912348
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest34_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest34 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest34 * t).T))

            ]
def fittest34_evolution_data():
    data = []
    for t in ts:
        data.append(fittest34_evolution(t))
    return data
#target
def target34_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target34 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target34 * t).T))

            ]
def target34_evolution_data():
    data = []
    for t in ts:
        data.append(target34_evolution(t))
    return data
# fidelity results
fidelity34_tab = []
for i in range(len(ts)):
    fidelity34_tab.append(fidelity(target34_evolution_data()[i][0] , fittest34_evolution_data()[i][0]))
fidelity34_tab = np.round(fidelity34_tab , decimals = 34)
# coherence results
t34_coherence = coherence(target34_evolution_data())
f34_coherence = coherence(fittest34_evolution_data())
t34f34_coherence = []
for i in range(len(ts)):
     t34f34_coherence.append(np.abs(t34_coherence[i] - f34_coherence[i]))
# population results
pop34 = []
for i in range(len(ts)):
     pop34.append(np.sum(populations(target34_evolution_data() , fittest34_evolution_data())[i]))




##########################################
##########################################
#target individual
target35 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest35 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target35cc = nx.node_connected_component(nx.from_numpy_array(np.real(target35)) , 1)
fittest35cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest35)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr35 = target35 == fittest35
# Count the number of False values
false_count35 = np.sum(arr35 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t35f35_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target35)) , nx.from_numpy_array(np.real(fittest35)))
# WL graph isomorphism test
t35_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target35)))
f35_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest35)))
# degree distribution
t35_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target35))), key=lambda x: x[1], reverse=True)]
f35_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest35))), key=lambda x: x[1], reverse=True)]
# number of connections
t35_connections = np.sum(t35_dd)
f35_connections = np.sum(f35_dd)
# distance
distance35 = 0.10267042302082618
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest35_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest35 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest35 * t).T))

            ]
def fittest35_evolution_data():
    data = []
    for t in ts:
        data.append(fittest35_evolution(t))
    return data
#target
def target35_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target35 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target35 * t).T))

            ]
def target35_evolution_data():
    data = []
    for t in ts:
        data.append(target35_evolution(t))
    return data
# fidelity results
fidelity35_tab = []
for i in range(len(ts)):
    fidelity35_tab.append(fidelity(target35_evolution_data()[i][0] , fittest35_evolution_data()[i][0]))
fidelity35_tab = np.round(fidelity35_tab , decimals = 35)
# coherence results
t35_coherence = coherence(target35_evolution_data())
f35_coherence = coherence(fittest35_evolution_data())
t35f35_coherence = []
for i in range(len(ts)):
     t35f35_coherence.append(np.abs(t35_coherence[i] - f35_coherence[i]))
# population results
pop35 = []
for i in range(len(ts)):
     pop35.append(np.sum(populations(target35_evolution_data() , fittest35_evolution_data())[i]))





##########################################
##########################################
#target individual
target36 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest36 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target36cc = nx.node_connected_component(nx.from_numpy_array(np.real(target36)) , 1)
fittest36cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest36)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr36 = target36 == fittest36
# Count the number of False values
false_count36 = np.sum(arr36 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t36f36_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target36)) , nx.from_numpy_array(np.real(fittest36)))
# WL graph isomorphism test
t36_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target36)))
f36_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest36)))
# degree distribution
t36_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target36))), key=lambda x: x[1], reverse=True)]
f36_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest36))), key=lambda x: x[1], reverse=True)]
# number of connections
t36_connections = np.sum(t36_dd)
f36_connections = np.sum(f36_dd)
# distance
distance36 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest36_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest36 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest36 * t).T))

            ]
def fittest36_evolution_data():
    data = []
    for t in ts:
        data.append(fittest36_evolution(t))
    return data
#target
def target36_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target36 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target36 * t).T))

            ]
def target36_evolution_data():
    data = []
    for t in ts:
        data.append(target36_evolution(t))
    return data
# fidelity results
fidelity36_tab = []
for i in range(len(ts)):
    fidelity36_tab.append(fidelity(target36_evolution_data()[i][0] , fittest36_evolution_data()[i][0]))
fidelity36_tab = np.round(fidelity36_tab , decimals = 36)
# coherence results
t36_coherence = coherence(target36_evolution_data())
f36_coherence = coherence(fittest36_evolution_data())
t36f36_coherence = []
for i in range(len(ts)):
     t36f36_coherence.append(np.abs(t36_coherence[i] - f36_coherence[i]))
# population results
pop36 = []
for i in range(len(ts)):
     pop36.append(np.sum(populations(target36_evolution_data() , fittest36_evolution_data())[i]))




##########################################
##########################################
#target individual
target37 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest37 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target37cc = nx.node_connected_component(nx.from_numpy_array(np.real(target37)) , 1)
fittest37cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest37)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr37 = target37 == fittest37
# Count the number of False values
false_count37 = np.sum(arr37 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t37f37_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target37)) , nx.from_numpy_array(np.real(fittest37)))
# WL graph isomorphism test
t37_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target37)))
f37_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest37)))
# degree distribution
t37_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target37))), key=lambda x: x[1], reverse=True)]
f37_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest37))), key=lambda x: x[1], reverse=True)]
# number of connections
t37_connections = np.sum(t37_dd)
f37_connections = np.sum(f37_dd)
# distance
distance37 = 0.7989136374395859
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest37_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest37 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest37 * t).T))

            ]
def fittest37_evolution_data():
    data = []
    for t in ts:
        data.append(fittest37_evolution(t))
    return data
#target
def target37_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target37 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target37 * t).T))

            ]
def target37_evolution_data():
    data = []
    for t in ts:
        data.append(target37_evolution(t))
    return data
# fidelity results
fidelity37_tab = []
for i in range(len(ts)):
    fidelity37_tab.append(fidelity(target37_evolution_data()[i][0] , fittest37_evolution_data()[i][0]))
fidelity37_tab = np.round(fidelity37_tab , decimals = 37)
# coherence results
t37_coherence = coherence(target37_evolution_data())
f37_coherence = coherence(fittest37_evolution_data())
t37f37_coherence = []
for i in range(len(ts)):
     t37f37_coherence.append(np.abs(t37_coherence[i] - f37_coherence[i]))
# population results
pop37 = []
for i in range(len(ts)):
     pop37.append(np.sum(populations(target37_evolution_data() , fittest37_evolution_data())[i]))




##########################################
##########################################
#target individual
target38 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest38 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target38cc = nx.node_connected_component(nx.from_numpy_array(np.real(target38)) , 1)
fittest38cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest38)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr38 = target38 == fittest38
# Count the number of False values
false_count38 = np.sum(arr38 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t38f38_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target38)) , nx.from_numpy_array(np.real(fittest38)))
# WL graph isomorphism test
t38_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target38)))
f38_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest38)))
# degree distribution
t38_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target38))), key=lambda x: x[1], reverse=True)]
f38_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest38))), key=lambda x: x[1], reverse=True)]
# number of connections
t38_connections = np.sum(t38_dd)
f38_connections = np.sum(f38_dd)
# distance
distance38 = 9.711389021727079e-05
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest38_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest38 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest38 * t).T))

            ]
def fittest38_evolution_data():
    data = []
    for t in ts:
        data.append(fittest38_evolution(t))
    return data
#target
def target38_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target38 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target38 * t).T))

            ]
def target38_evolution_data():
    data = []
    for t in ts:
        data.append(target38_evolution(t))
    return data
# fidelity results
fidelity38_tab = []
for i in range(len(ts)):
    fidelity38_tab.append(fidelity(target38_evolution_data()[i][0] , fittest38_evolution_data()[i][0]))
fidelity38_tab = np.round(fidelity38_tab , decimals = 38)
# coherence results
t38_coherence = coherence(target38_evolution_data())
f38_coherence = coherence(fittest38_evolution_data())
t38f38_coherence = []
for i in range(len(ts)):
     t38f38_coherence.append(np.abs(t38_coherence[i] - f38_coherence[i]))
# population results
pop38 = []
for i in range(len(ts)):
     pop38.append(np.sum(populations(target38_evolution_data() , fittest38_evolution_data())[i]))





##########################################
##########################################
#target individual
target39 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest39 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target39cc = nx.node_connected_component(nx.from_numpy_array(np.real(target39)) , 1)
fittest39cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest39)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr39 = target39 == fittest39
# Count the number of False values
false_count39 = np.sum(arr39 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t39f39_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target39)) , nx.from_numpy_array(np.real(fittest39)))
# WL graph isomorphism test
t39_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target39)))
f39_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest39)))
# degree distribution
t39_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target39))), key=lambda x: x[1], reverse=True)]
f39_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest39))), key=lambda x: x[1], reverse=True)]
# number of connections
t39_connections = np.sum(t39_dd)
f39_connections = np.sum(f39_dd)
# distance
distance39 = 0.5794535961948366
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest39_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest39 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest39 * t).T))

            ]
def fittest39_evolution_data():
    data = []
    for t in ts:
        data.append(fittest39_evolution(t))
    return data
#target
def target39_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target39 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target39 * t).T))

            ]
def target39_evolution_data():
    data = []
    for t in ts:
        data.append(target39_evolution(t))
    return data
# fidelity results
fidelity39_tab = []
for i in range(len(ts)):
    fidelity39_tab.append(fidelity(target39_evolution_data()[i][0] , fittest39_evolution_data()[i][0]))
fidelity39_tab = np.round(fidelity39_tab , decimals = 39)
# coherence results
t39_coherence = coherence(target39_evolution_data())
f39_coherence = coherence(fittest39_evolution_data())
t39f39_coherence = []
for i in range(len(ts)):
     t39f39_coherence.append(np.abs(t39_coherence[i] - f39_coherence[i]))
# population results
pop39 = []
for i in range(len(ts)):
     pop39.append(np.sum(populations(target39_evolution_data() , fittest39_evolution_data())[i]))




##########################################
##########################################
#target individual
target40 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest40 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target40cc = nx.node_connected_component(nx.from_numpy_array(np.real(target40)) , 1)
fittest40cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest40)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr40 = target40 == fittest40
# Count the number of False values
false_count40 = np.sum(arr40 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t40f40_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target40)) , nx.from_numpy_array(np.real(fittest40)))
# WL graph isomorphism test
t40_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target40)))
f40_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest40)))
# degree distribution
t40_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target40))), key=lambda x: x[1], reverse=True)]
f40_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest40))), key=lambda x: x[1], reverse=True)]
# number of connections
t40_connections = np.sum(t40_dd)
f40_connections = np.sum(f40_dd)
# distance
distance40 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest40_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest40 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest40 * t).T))

            ]
def fittest40_evolution_data():
    data = []
    for t in ts:
        data.append(fittest40_evolution(t))
    return data
#target
def target40_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target40 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target40 * t).T))

            ]
def target40_evolution_data():
    data = []
    for t in ts:
        data.append(target40_evolution(t))
    return data
# fidelity results
fidelity40_tab = []
for i in range(len(ts)):
    fidelity40_tab.append(fidelity(target40_evolution_data()[i][0] , fittest40_evolution_data()[i][0]))
fidelity40_tab = np.round(fidelity40_tab , decimals = 40)
# coherence results
t40_coherence = coherence(target40_evolution_data())
f40_coherence = coherence(fittest40_evolution_data())
t40f40_coherence = []
for i in range(len(ts)):
     t40f40_coherence.append(np.abs(t40_coherence[i] - f40_coherence[i]))
# population results
pop40 = []
for i in range(len(ts)):
     pop40.append(np.sum(populations(target40_evolution_data() , fittest40_evolution_data())[i]))





##########################################
##########################################
#target individual
target41 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest41 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target41cc = nx.node_connected_component(nx.from_numpy_array(np.real(target41)) , 1)
fittest41cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest41)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr41 = target41 == fittest41
# Count the number of False values
false_count41 = np.sum(arr41 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t41f41_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target41)) , nx.from_numpy_array(np.real(fittest41)))
# WL graph isomorphism test
t41_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target41)))
f41_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest41)))
# degree distribution
t41_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target41))), key=lambda x: x[1], reverse=True)]
f41_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest41))), key=lambda x: x[1], reverse=True)]
# number of connections
t41_connections = np.sum(t41_dd)
f41_connections = np.sum(f41_dd)
# distance
distance41 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest41_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest41 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest41 * t).T))

            ]
def fittest41_evolution_data():
    data = []
    for t in ts:
        data.append(fittest41_evolution(t))
    return data
#target
def target41_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target41 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target41 * t).T))

            ]
def target41_evolution_data():
    data = []
    for t in ts:
        data.append(target41_evolution(t))
    return data
# fidelity results
fidelity41_tab = []
for i in range(len(ts)):
    fidelity41_tab.append(fidelity(target41_evolution_data()[i][0] , fittest41_evolution_data()[i][0]))
fidelity41_tab = np.round(fidelity41_tab , decimals = 41)
# coherence results
t41_coherence = coherence(target41_evolution_data())
f41_coherence = coherence(fittest41_evolution_data())
t41f41_coherence = []
for i in range(len(ts)):
     t41f41_coherence.append(np.abs(t41_coherence[i] - f41_coherence[i]))
# population results
pop41 = []
for i in range(len(ts)):
     pop41.append(np.sum(populations(target41_evolution_data() , fittest41_evolution_data())[i]))




##########################################
##########################################
#target individual
target42 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest42 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target42cc = nx.node_connected_component(nx.from_numpy_array(np.real(target42)) , 1)
fittest42cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest42)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr42 = target42 == fittest42
# Count the number of False values
false_count42 = np.sum(arr42 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t42f42_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target42)) , nx.from_numpy_array(np.real(fittest42)))
# WL graph isomorphism test
t42_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target42)))
f42_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest42)))
# degree distribution
t42_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target42))), key=lambda x: x[1], reverse=True)]
f42_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest42))), key=lambda x: x[1], reverse=True)]
# number of connections
t42_connections = np.sum(t42_dd)
f42_connections = np.sum(f42_dd)
# distance
distance42 = 0.31433207579380673
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest42_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest42 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest42 * t).T))

            ]
def fittest42_evolution_data():
    data = []
    for t in ts:
        data.append(fittest42_evolution(t))
    return data
#target
def target42_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target42 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target42 * t).T))

            ]
def target42_evolution_data():
    data = []
    for t in ts:
        data.append(target42_evolution(t))
    return data
# fidelity results
fidelity42_tab = []
for i in range(len(ts)):
    fidelity42_tab.append(fidelity(target42_evolution_data()[i][0] , fittest42_evolution_data()[i][0]))
fidelity42_tab = np.round(fidelity42_tab , decimals = 42)
# coherence results
t42_coherence = coherence(target42_evolution_data())
f42_coherence = coherence(fittest42_evolution_data())
t42f42_coherence = []
for i in range(len(ts)):
     t42f42_coherence.append(np.abs(t42_coherence[i] - f42_coherence[i]))
# population results
pop42 = []
for i in range(len(ts)):
     pop42.append(np.sum(populations(target42_evolution_data() , fittest42_evolution_data())[i]))





##########################################
##########################################
#target individual
target43 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest43 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target43cc = nx.node_connected_component(nx.from_numpy_array(np.real(target43)) , 1)
fittest43cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest43)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr43 = target43 == fittest43
# Count the number of False values
false_count43 = np.sum(arr43 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t43f43_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target43)) , nx.from_numpy_array(np.real(fittest43)))
# WL graph isomorphism test
t43_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target43)))
f43_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest43)))
# degree distribution
t43_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target43))), key=lambda x: x[1], reverse=True)]
f43_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest43))), key=lambda x: x[1], reverse=True)]
# number of connections
t43_connections = np.sum(t43_dd)
f43_connections = np.sum(f43_dd)
# distance
distance43 = 0.44011854718679766
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest43_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest43 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest43 * t).T))

            ]
def fittest43_evolution_data():
    data = []
    for t in ts:
        data.append(fittest43_evolution(t))
    return data
#target
def target43_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target43 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target43 * t).T))

            ]
def target43_evolution_data():
    data = []
    for t in ts:
        data.append(target43_evolution(t))
    return data
# fidelity results
fidelity43_tab = []
for i in range(len(ts)):
    fidelity43_tab.append(fidelity(target43_evolution_data()[i][0] , fittest43_evolution_data()[i][0]))
fidelity43_tab = np.round(fidelity43_tab , decimals = 43)
# coherence results
t43_coherence = coherence(target43_evolution_data())
f43_coherence = coherence(fittest43_evolution_data())
t43f43_coherence = []
for i in range(len(ts)):
     t43f43_coherence.append(np.abs(t43_coherence[i] - f43_coherence[i]))
# population results
pop43 = []
for i in range(len(ts)):
     pop43.append(np.sum(populations(target43_evolution_data() , fittest43_evolution_data())[i]))





##########################################
##########################################
#target individual
target44 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest44 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target44cc = nx.node_connected_component(nx.from_numpy_array(np.real(target44)) , 1)
fittest44cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest44)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr44 = target44 == fittest44
# Count the number of False values
false_count44 = np.sum(arr44 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t44f44_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target44)) , nx.from_numpy_array(np.real(fittest44)))
# WL graph isomorphism test
t44_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target44)))
f44_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest44)))
# degree distribution
t44_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target44))), key=lambda x: x[1], reverse=True)]
f44_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest44))), key=lambda x: x[1], reverse=True)]
# number of connections
t44_connections = np.sum(t44_dd)
f44_connections = np.sum(f44_dd)
# distance
distance44 = 0.2839443221141438
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest44_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest44 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest44 * t).T))

            ]
def fittest44_evolution_data():
    data = []
    for t in ts:
        data.append(fittest44_evolution(t))
    return data
#target
def target44_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target44 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target44 * t).T))

            ]
def target44_evolution_data():
    data = []
    for t in ts:
        data.append(target44_evolution(t))
    return data
# fidelity results
fidelity44_tab = []
for i in range(len(ts)):
    fidelity44_tab.append(fidelity(target44_evolution_data()[i][0] , fittest44_evolution_data()[i][0]))
fidelity44_tab = np.round(fidelity44_tab , decimals = 44)
# coherence results
t44_coherence = coherence(target44_evolution_data())
f44_coherence = coherence(fittest44_evolution_data())
t44f44_coherence = []
for i in range(len(ts)):
     t44f44_coherence.append(np.abs(t44_coherence[i] - f44_coherence[i]))
# population results
pop44 = []
for i in range(len(ts)):
     pop44.append(np.sum(populations(target44_evolution_data() , fittest44_evolution_data())[i]))





##########################################
##########################################
#target individual
target45 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
# fittest individual
fittest45 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target45cc = nx.node_connected_component(nx.from_numpy_array(np.real(target45)) , 1)
fittest45cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest45)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr45 = target45 == fittest45
# Count the number of False values
false_count45 = np.sum(arr45 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t45f45_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target45)) , nx.from_numpy_array(np.real(fittest45)))
# WL graph isomorphism test
t45_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target45)))
f45_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest45)))
# degree distribution
t45_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target45))), key=lambda x: x[1], reverse=True)]
f45_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest45))), key=lambda x: x[1], reverse=True)]
# number of connections
t45_connections = np.sum(t45_dd)
f45_connections = np.sum(f45_dd)
# distance
distance45 = 0.010344922660180278
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest45_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest45 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest45 * t).T))

            ]
def fittest45_evolution_data():
    data = []
    for t in ts:
        data.append(fittest45_evolution(t))
    return data
#target
def target45_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target45 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target45 * t).T))

            ]
def target45_evolution_data():
    data = []
    for t in ts:
        data.append(target45_evolution(t))
    return data
# fidelity results
fidelity45_tab = []
for i in range(len(ts)):
    fidelity45_tab.append(fidelity(target45_evolution_data()[i][0] , fittest45_evolution_data()[i][0]))
fidelity45_tab = np.round(fidelity45_tab , decimals = 45)
# coherence results
t45_coherence = coherence(target45_evolution_data())
f45_coherence = coherence(fittest45_evolution_data())
t45f45_coherence = []
for i in range(len(ts)):
     t45f45_coherence.append(np.abs(t45_coherence[i] - f45_coherence[i]))
# population results
pop45 = []
for i in range(len(ts)):
     pop45.append(np.sum(populations(target45_evolution_data() , fittest45_evolution_data())[i]))





##########################################
##########################################
#target individual
target46 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest46 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target46cc = nx.node_connected_component(nx.from_numpy_array(np.real(target46)) , 1)
fittest46cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest46)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr46 = target46 == fittest46
# Count the number of False values
false_count46 = np.sum(arr46 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t46f46_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target46)) , nx.from_numpy_array(np.real(fittest46)))
# WL graph isomorphism test
t46_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target46)))
f46_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest46)))
# degree distribution
t46_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target46))), key=lambda x: x[1], reverse=True)]
f46_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest46))), key=lambda x: x[1], reverse=True)]
# number of connections
t46_connections = np.sum(t46_dd)
f46_connections = np.sum(f46_dd)
# distance
distance46 = 0.01106843482365727
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest46_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest46 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest46 * t).T))

            ]
def fittest46_evolution_data():
    data = []
    for t in ts:
        data.append(fittest46_evolution(t))
    return data
#target
def target46_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target46 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target46 * t).T))

            ]
def target46_evolution_data():
    data = []
    for t in ts:
        data.append(target46_evolution(t))
    return data
# fidelity results
fidelity46_tab = []
for i in range(len(ts)):
    fidelity46_tab.append(fidelity(target46_evolution_data()[i][0] , fittest46_evolution_data()[i][0]))
fidelity46_tab = np.round(fidelity46_tab , decimals = 46)
# coherence results
t46_coherence = coherence(target46_evolution_data())
f46_coherence = coherence(fittest46_evolution_data())
t46f46_coherence = []
for i in range(len(ts)):
     t46f46_coherence.append(np.abs(t46_coherence[i] - f46_coherence[i]))
# population results
pop46 = []
for i in range(len(ts)):
     pop46.append(np.sum(populations(target46_evolution_data() , fittest46_evolution_data())[i]))





##########################################
##########################################
#target individual
target47 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest47 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target47cc = nx.node_connected_component(nx.from_numpy_array(np.real(target47)) , 1)
fittest47cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest47)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr47 = target47 == fittest47
# Count the number of False values
false_count47 = np.sum(arr47 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t47f47_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target47)) , nx.from_numpy_array(np.real(fittest47)))
# WL graph isomorphism test
t47_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target47)))
f47_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest47)))
# degree distribution
t47_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target47))), key=lambda x: x[1], reverse=True)]
f47_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest47))), key=lambda x: x[1], reverse=True)]
# number of connections
t47_connections = np.sum(t47_dd)
f47_connections = np.sum(f47_dd)
# distance
distance47 = 0.6388751119587021
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest47_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest47 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest47 * t).T))

            ]
def fittest47_evolution_data():
    data = []
    for t in ts:
        data.append(fittest47_evolution(t))
    return data
#target
def target47_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target47 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target47 * t).T))

            ]
def target47_evolution_data():
    data = []
    for t in ts:
        data.append(target47_evolution(t))
    return data
# fidelity results
fidelity47_tab = []
for i in range(len(ts)):
    fidelity47_tab.append(fidelity(target47_evolution_data()[i][0] , fittest47_evolution_data()[i][0]))
fidelity47_tab = np.round(fidelity47_tab , decimals = 47)
# coherence results
t47_coherence = coherence(target47_evolution_data())
f47_coherence = coherence(fittest47_evolution_data())
t47f47_coherence = []
for i in range(len(ts)):
     t47f47_coherence.append(np.abs(t47_coherence[i] - f47_coherence[i]))
# population results
pop47 = []
for i in range(len(ts)):
     pop47.append(np.sum(populations(target47_evolution_data() , fittest47_evolution_data())[i]))





##########################################
##########################################
#target individual
target48 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest48 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target48cc = nx.node_connected_component(nx.from_numpy_array(np.real(target48)) , 1)
fittest48cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest48)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr48 = target48 == fittest48
# Count the number of False values
false_count48 = np.sum(arr48 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t48f48_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target48)) , nx.from_numpy_array(np.real(fittest48)))
# WL graph isomorphism test
t48_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target48)))
f48_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest48)))
# degree distribution
t48_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target48))), key=lambda x: x[1], reverse=True)]
f48_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest48))), key=lambda x: x[1], reverse=True)]
# number of connections
t48_connections = np.sum(t48_dd)
f48_connections = np.sum(f48_dd)
# distance
distance48 = 0.007111216270113796
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest48_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest48 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest48 * t).T))

            ]
def fittest48_evolution_data():
    data = []
    for t in ts:
        data.append(fittest48_evolution(t))
    return data
#target
def target48_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target48 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target48 * t).T))

            ]
def target48_evolution_data():
    data = []
    for t in ts:
        data.append(target48_evolution(t))
    return data
# fidelity results
fidelity48_tab = []
for i in range(len(ts)):
    fidelity48_tab.append(fidelity(target48_evolution_data()[i][0] , fittest48_evolution_data()[i][0]))
fidelity48_tab = np.round(fidelity48_tab , decimals = 48)
# coherence results
t48_coherence = coherence(target48_evolution_data())
f48_coherence = coherence(fittest48_evolution_data())
t48f48_coherence = []
for i in range(len(ts)):
     t48f48_coherence.append(np.abs(t48_coherence[i] - f48_coherence[i]))
# population results
pop48 = []
for i in range(len(ts)):
     pop48.append(np.sum(populations(target48_evolution_data() , fittest48_evolution_data())[i]))





##########################################
##########################################
#target individual
target49 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest49 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target49cc = nx.node_connected_component(nx.from_numpy_array(np.real(target49)) , 1)
fittest49cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest49)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr49 = target49 == fittest49
# Count the number of False values
false_count49 = np.sum(arr49 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t49f49_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target49)) , nx.from_numpy_array(np.real(fittest49)))
# WL graph isomorphism test
t49_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target49)))
f49_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest49)))
# degree distribution
t49_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target49))), key=lambda x: x[1], reverse=True)]
f49_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest49))), key=lambda x: x[1], reverse=True)]
# number of connections
t49_connections = np.sum(t49_dd)
f49_connections = np.sum(f49_dd)
# distance
distance49 = 0.4931698554045536
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest49_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest49 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest49 * t).T))

            ]
def fittest49_evolution_data():
    data = []
    for t in ts:
        data.append(fittest49_evolution(t))
    return data
#target
def target49_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target49 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target49 * t).T))

            ]
def target49_evolution_data():
    data = []
    for t in ts:
        data.append(target49_evolution(t))
    return data
# fidelity results
fidelity49_tab = []
for i in range(len(ts)):
    fidelity49_tab.append(fidelity(target49_evolution_data()[i][0] , fittest49_evolution_data()[i][0]))
fidelity49_tab = np.round(fidelity49_tab , decimals = 49)
# coherence results
t49_coherence = coherence(target49_evolution_data())
f49_coherence = coherence(fittest49_evolution_data())
t49f49_coherence = []
for i in range(len(ts)):
     t49f49_coherence.append(np.abs(t49_coherence[i] - f49_coherence[i]))
# population results
pop49 = []
for i in range(len(ts)):
     pop49.append(np.sum(populations(target49_evolution_data() , fittest49_evolution_data())[i]))






##########################################
##########################################
#target individual
target50 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]])
# fittest individual
fittest50 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target50cc = nx.node_connected_component(nx.from_numpy_array(np.real(target50)) , 1)
fittest50cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest50)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr50 = target50 == fittest50
# Count the number of False values
false_count50 = np.sum(arr50 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t50f50_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target50)) , nx.from_numpy_array(np.real(fittest50)))
# WL graph isomorphism test
t50_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target50)))
f50_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest50)))
# degree distribution
t50_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target50))), key=lambda x: x[1], reverse=True)]
f50_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest50))), key=lambda x: x[1], reverse=True)]
# number of connections
t50_connections = np.sum(t50_dd)
f50_connections = np.sum(f50_dd)
# distance
distance50 = 0.2794718662879955
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest50_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest50 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest50 * t).T))

            ]
def fittest50_evolution_data():
    data = []
    for t in ts:
        data.append(fittest50_evolution(t))
    return data
#target
def target50_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target50 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target50 * t).T))

            ]
def target50_evolution_data():
    data = []
    for t in ts:
        data.append(target50_evolution(t))
    return data
# fidelity results
fidelity50_tab = []
for i in range(len(ts)):
    fidelity50_tab.append(fidelity(target50_evolution_data()[i][0] , fittest50_evolution_data()[i][0]))
fidelity50_tab = np.round(fidelity50_tab , decimals = 50)
# coherence results
t50_coherence = coherence(target50_evolution_data())
f50_coherence = coherence(fittest50_evolution_data())
t50f50_coherence = []
for i in range(len(ts)):
     t50f50_coherence.append(np.abs(t50_coherence[i] - f50_coherence[i]))
# population results
pop50 = []
for i in range(len(ts)):
     pop50.append(np.sum(populations(target50_evolution_data() , fittest50_evolution_data())[i]))











##########################################
##########################################
# analysis of network properties
##########################################
##########################################
# table of asking if connected component of target is the same as connected component of fittest individual
connected_component_tab = np.array([target1cc == fittest1cc , target2cc == fittest2cc , target3cc == fittest3cc , target4cc == fittest4cc , target5cc == fittest5cc , 
                                    target6cc == fittest6cc , target7cc == fittest7cc , target8cc == fittest8cc , target9cc == fittest9cc , target10cc == fittest10cc , 
                                    target11cc == fittest11cc , target12cc == fittest12cc , target13cc == fittest13cc , target14cc == fittest14cc , target15cc == fittest15cc , 
                                    target16cc == fittest16cc , target17cc == fittest17cc , target18cc == fittest18cc , target19cc == fittest19cc , target20cc == fittest20cc ,
                                    target21cc == fittest21cc , target22cc == fittest22cc , target23cc == fittest23cc , target24cc == fittest24cc , target25cc == fittest25cc , 
                                    target26cc == fittest26cc , target27cc == fittest27cc , target28cc == fittest28cc , target29cc == fittest29cc , target30cc == fittest30cc ,
                                    target31cc == fittest31cc , target32cc == fittest32cc , target33cc == fittest33cc , target34cc == fittest34cc , target35cc == fittest35cc , 
                                    target36cc == fittest36cc , target37cc == fittest37cc , target38cc == fittest38cc , target39cc == fittest39cc , target40cc == fittest40cc ,
                                    target41cc == fittest41cc , target42cc == fittest42cc , target43cc == fittest43cc , target44cc == fittest44cc , target45cc == fittest45cc , 
                                    target46cc == fittest46cc , target47cc == fittest47cc , target48cc == fittest48cc , target49cc == fittest49cc , target50cc == fittest50cc])

connected_component15_tab = np.array([ True,  True,  True,  True,  True,  True, False,  True,  True,
        True, False, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True])
connected_component_true15Nodes = np.sum(connected_component15_tab == True)
# table showing number of different connections between target and fittest arrays
false_count_tab = np.array([false_count1 , false_count2 , false_count3 , false_count4 , false_count5 , false_count6 , false_count7 , false_count8 , false_count9 , false_count10 , 
                            false_count11 , false_count12 , false_count13 , false_count14 , false_count15 , false_count16 , false_count17 , false_count18 , false_count19 , false_count20 , 
                            false_count21 , false_count22 , false_count23 , false_count24 , false_count25 , false_count26 , false_count27 , false_count28 , false_count29 , false_count30 , 
                            false_count31 , false_count32 , false_count33 , false_count34 , false_count35 , false_count36 , false_count37 , false_count38 , false_count39 , false_count40 , 
                            false_count41 , false_count42 , false_count43 , false_count44 , false_count45 , false_count46 , false_count47 , false_count48 , false_count49 , false_count50])

false_count15_tab = np.array([35, 42, 16, 43, 15, 22, 36, 12, 39,  6,  7, 21,  8, 39, 54,  2, 12,
       51, 30, 20, 50, 37, 15, 19, 40, 52, 50, 11, 24, 44, 22, 39,  7, 43,
       22,  0, 59,  9, 59,  4,  2, 32, 49, 41,  6,  8, 50,  6, 52, 34])
false_count_15Nodes = np.sum(false_count15_tab == 0)

# table showing how many true isomorphisms were found of target network
isomorphism_tab = np.array([t1f1_iso , t2f2_iso , t3f3_iso , t4f4_iso , t5f5_iso , t6f6_iso , t7f7_iso , t8f8_iso , t9f9_iso , t10f10_iso , 
                            t11f11_iso , t12f12_iso , t13f13_iso , t14f14_iso , t15f15_iso , t16f16_iso , t17f17_iso , t18f18_iso , t19f19_iso , t20f20_iso , 
                            t21f21_iso , t22f22_iso , t23f23_iso , t24f24_iso , t25f25_iso , t26f26_iso , t27f27_iso , t28f28_iso , t29f29_iso , t30f30_iso , 
                            t31f31_iso , t32f32_iso , t33f33_iso , t34f34_iso , t35f35_iso , t36f36_iso , t37f37_iso , t38f38_iso , t39f39_iso , t40f40_iso , 
                            t41f41_iso , t42f42_iso , t43f43_iso , t44f44_iso , t45f45_iso , t46f46_iso , t47f47_iso , t48f48_iso , t49f49_iso , t50f50_iso])


isomorphism15_tab = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False,  True, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False,  True,
       False, False, False, False,  True, False, False, False,  True,
       False, False,  True, False, False])
true_count_15Nodes = np.sum(isomorphism15_tab == True)

# table showing how many times the degree distribution of the fittest network matched that of the target network
degree_distribution_table = np.array([t1_dd == f1_dd , t2_dd == f2_dd , t3_dd == f3_dd , t4_dd == f4_dd , t5_dd == f5_dd , t6_dd == f6_dd , t7_dd == f7_dd , t8_dd == f8_dd , t9_dd == f9_dd , t10_dd == f10_dd , 
                                      t11_dd == f11_dd , t12_dd == f12_dd , t13_dd == f13_dd , t14_dd == f14_dd , t15_dd == f15_dd , t16_dd == f16_dd , t17_dd == f17_dd , t18_dd == f18_dd , t19_dd == f19_dd , t20_dd == f20_dd , 
                                      t21_dd == f21_dd , t22_dd == f22_dd , t23_dd == f23_dd , t24_dd == f24_dd , t25_dd == f25_dd , t26_dd == f26_dd , t27_dd == f27_dd , t28_dd == f28_dd , t29_dd == f29_dd , t30_dd == f30_dd , 
                                      t31_dd == f31_dd , t32_dd == f32_dd , t33_dd == f33_dd , t34_dd == f34_dd , t35_dd == f35_dd , t36_dd == f36_dd , t37_dd == f37_dd , t38_dd == f38_dd , t39_dd == f39_dd , t40_dd == f40_dd , 
                                      t41_dd == f41_dd , t42_dd == f42_dd , t43_dd == f43_dd , t44_dd == f44_dd , t45_dd == f45_dd , t46_dd == f46_dd , t47_dd == f47_dd , t48_dd == f48_dd , t49_dd == f49_dd , t50_dd == f50_dd])


degree_distribution15_table = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False,  True, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False,  True,
       False, False, False, False,  True, False, False, False,  True,
        True, False,  True, False, False])
degree_distribution_15Nodes_true = np.sum(degree_distribution15_table == True)


# table showing how many times the target and fittest networks shared the same number of connections
connections_tab = np.array([t1_connections == f1_connections , t2_connections == f2_connections , t3_connections == f3_connections , t4_connections == f4_connections , t5_connections == f5_connections , 
                            t6_connections == f6_connections , t7_connections == f7_connections , t8_connections == f8_connections , t9_connections == f9_connections , t10_connections == f10_connections , 
                            t11_connections == f11_connections , t12_connections == f12_connections , t13_connections == f13_connections , t14_connections == f14_connections , t15_connections == f15_connections , 
                            t16_connections == f16_connections , t17_connections == f17_connections , t18_connections == f18_connections , t19_connections == f19_connections , t20_connections == f20_connections , 
                            t21_connections == f21_connections , t22_connections == f22_connections , t23_connections == f23_connections , t24_connections == f24_connections , t25_connections == f25_connections , 
                            t26_connections == f26_connections , t27_connections == f27_connections , t28_connections == f28_connections , t29_connections == f29_connections , t30_connections == f30_connections , 
                            t31_connections == f31_connections , t32_connections == f32_connections , t33_connections == f33_connections , t34_connections == f34_connections , t35_connections == f35_connections , 
                            t36_connections == f36_connections , t37_connections == f37_connections , t38_connections == f38_connections , t39_connections == f39_connections , t40_connections == f40_connections , 
                            t41_connections == f41_connections , t42_connections == f42_connections , t43_connections == f43_connections , t44_connections == f44_connections , t45_connections == f45_connections , 
                            t46_connections == f46_connections , t47_connections == f47_connections , t48_connections == f48_connections , t49_connections == f49_connections , t50_connections == f50_connections])

connections15_tab = np.array([False, False, False, False, False,  True, False,  True, False,
       False, False, False, False, False, False,  True,  True, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False, False, False,  True,  True,
       False, False, False, False,  True, False, False, False,  True,
        True, False,  True, False,  True])
connections_15Nodes_true = np.sum(connections15_tab == True)

##########################################
##########################################
# analysis of physical properties
##########################################
##########################################
# fidelity analysis
fidelity_tab = np.array([fidelity1_tab , fidelity2_tab , fidelity3_tab , fidelity4_tab , fidelity5_tab , fidelity6_tab , fidelity7_tab , fidelity8_tab , fidelity9_tab , fidelity10_tab , 
                         fidelity11_tab , fidelity12_tab , fidelity13_tab , fidelity14_tab , fidelity15_tab , fidelity16_tab , fidelity17_tab , fidelity18_tab , fidelity19_tab , fidelity20_tab ,
                         fidelity21_tab , fidelity22_tab , fidelity23_tab , fidelity24_tab , fidelity25_tab , fidelity26_tab , fidelity27_tab , fidelity28_tab , fidelity29_tab , fidelity30_tab , 
                         fidelity31_tab , fidelity32_tab , fidelity33_tab , fidelity34_tab , fidelity35_tab , fidelity36_tab , fidelity37_tab , fidelity38_tab , fidelity39_tab , fidelity40_tab , 
                         fidelity41_tab , fidelity42_tab , fidelity43_tab , fidelity44_tab , fidelity45_tab , fidelity46_tab , fidelity47_tab , fidelity48_tab , fidelity49_tab , fidelity50_tab]).T


fidelity_tab15 = np.array([[1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        , 1.        , 1.        ],
       [0.976858  , 0.952521  , 0.977127  , 0.975738  , 1.        ,
        0.999458  , 0.999849  , 1.        , 0.988039  , 1.        ,
        1.        , 0.999924  , 1.        , 0.987328  , 0.92737   ,
        0.99993454, 0.99993242, 0.96170279, 0.99999966, 1.        ,
        0.98749175, 0.99966714, 0.99992443, 0.99970464, 0.97531805,
        0.96383209, 0.9516817 , 1.        , 1.        , 0.96301144,
        0.99970162, 0.96575728, 1.        , 0.97526675, 0.9993727 ,
        1.        , 0.87411342, 1.        , 0.88195979, 1.        ,
        1.        , 0.96359607, 0.90893698, 0.97586128, 0.99999987,
        0.99999986, 0.93069232, 0.999935  , 0.91630061, 0.9755286 ],
       [0.922825  , 0.829652  , 0.924315  , 0.907899  , 1.        ,
        0.993943  , 0.997636  , 1.        , 0.954909  , 1.        ,
        1.        , 0.998777  , 1.        , 0.94707   , 0.729634  ,
        0.99934332, 0.99924723, 0.83782948, 0.9999779 , 1.        ,
        0.94816859, 0.99511969, 0.99882036, 0.99569471, 0.90323441,
        0.86477304, 0.81976148, 1.        , 1.        , 0.85510945,
        0.99552281, 0.88884523, 1.00000001, 0.90042696, 0.99235849,
        1.        , 0.59279037, 1.00000001, 0.58793021, 1.        ,
        1.        , 0.86233326, 0.70694363, 0.90927794, 0.99999129,
        0.99999118, 0.7692377 , 0.99936235, 0.69740082, 0.9058708 ],
       [0.865819  , 0.671967  , 0.859522  , 0.809205  , 1.        ,
        0.983433  , 0.988479  , 1.        , 0.900829  , 1.        ,
        1.        , 0.993811  , 1.        , 0.880609  , 0.473854  ,
        0.99852759, 0.99791898, 0.63313512, 0.99976482, 1.        ,
        0.87802668, 0.978599  , 0.99427286, 0.98125629, 0.79797603,
        0.72436495, 0.64258764, 1.        , 1.        , 0.69770666,
        0.979639  , 0.80691544, 1.00000001, 0.7734715 , 0.97565268,
        1.        , 0.33190852, 1.        , 0.26466854, 1.        ,
        1.        , 0.72058609, 0.50441911, 0.81326895, 0.99994044,
        0.99993872, 0.59855382, 0.99862781, 0.43051858, 0.80689899],
       [0.816018  , 0.512538  , 0.774362  , 0.693267  , 1.        ,
        0.977505  , 0.965471  , 1.        , 0.821439  , 1.        ,
        1.        , 0.980572  , 1.        , 0.800128  , 0.249769  ,
        0.99843656, 0.99689409, 0.40165696, 0.99879251, 1.        ,
        0.77500434, 0.94458728, 0.98295149, 0.95158849, 0.69019654,
        0.56307401, 0.47265831, 1.        , 1.        , 0.5397687 ,
        0.94458865, 0.72015989, 1.00000002, 0.59620745, 0.95887617,
        1.        , 0.19141159, 0.9999999 , 0.05796993, 1.00000001,
        1.        , 0.57297293, 0.33472238, 0.701053  , 0.99982421,
        0.99981766, 0.45865645, 0.99859915, 0.21542716, 0.70663314],
       [0.759912  , 0.370671  , 0.667627  , 0.569083  , 1.        ,
        0.975099  , 0.921388  , 0.999999  , 0.724183  , 1.        ,
        1.        , 0.953475  , 1.        , 0.717819  , 0.1077    ,
        0.99808359, 0.99536401, 0.21824748, 0.99585912, 1.        ,
        0.64342391, 0.89444162, 0.96156415, 0.90741394, 0.60846626,
        0.39794894, 0.35213068, 1.        , 1.        , 0.43250948,
        0.88804827, 0.61495425, 1.00000001, 0.39103934, 0.94644804,
        1.        , 0.15692336, 0.99999885, 0.01179881, 1.00000001,
        1.        , 0.44730026, 0.19387904, 0.58515179, 0.99963235,
        0.99962598, 0.34394423, 0.99818772, 0.10572661, 0.6307226 ],
       [0.683088  , 0.260103  , 0.574888  , 0.445383  , 1.        ,
        0.959666  , 0.851094  , 0.999996  , 0.627484  , 1.        ,
        1.        , 0.907093  , 1.        , 0.634548  , 0.03919   ,
        0.99635215, 0.99055209, 0.12727669, 0.98898807, 1.        ,
        0.49642985, 0.83353483, 0.92802286, 0.85445564, 0.56106347,
        0.2491259 , 0.29457523, 1.        , 1.        , 0.39730969,
        0.81436659, 0.51633334, 0.9999999 , 0.2095002 , 0.91992456,
        1.        , 0.15355918, 0.99999308, 0.05037547, 1.        ,
        1.        , 0.35334098, 0.10434679, 0.47969814, 0.99926922,
        0.99928284, 0.24633185, 0.99656863, 0.09206233, 0.59052603],
       [0.594165  , 0.190085  , 0.524841  , 0.336192  , 1.        ,
        0.928027  , 0.754453  , 0.999983  , 0.537569  , 1.        ,
        0.999999  , 0.838045  , 1.        , 0.548965  , 0.013192  ,
        0.99510504, 0.98178958, 0.12424918, 0.97533494, 1.        ,
        0.35537294, 0.76229267, 0.88258412, 0.80038382, 0.53938538,
        0.13587489, 0.28686852, 1.        , 1.        , 0.42215193,
        0.73212591, 0.46604001, 0.99999938, 0.10214472, 0.86395063,
        1.        , 0.14168836, 0.99996969, 0.08330434, 1.00000001,
        1.        , 0.28947698, 0.08269651, 0.39765429, 0.99865084,
        0.9986995 , 0.17769112, 0.99576591, 0.129646  , 0.58363219],
       [0.52033   , 0.160022  , 0.498758  , 0.25449   , 1.        ,
        0.899139  , 0.638694  , 0.999941  , 0.442299  , 1.        ,
        0.999994  , 0.746781  , 1.        , 0.470971  , 0.00566   ,
        0.99561076, 0.97070794, 0.17114195, 0.95108252, 1.        ,
        0.24170026, 0.67258992, 0.8285345 , 0.75182726, 0.52965164,
        0.06509043, 0.29953133, 1.        , 1.        , 0.4718347 ,
        0.65044618, 0.45838873, 0.99999703, 0.07511812, 0.80005516,
        1.        , 0.13648413, 0.99989365, 0.08164745, 1.00000002,
        1.        , 0.25439455, 0.1085012 , 0.34361762, 0.99786865,
        0.99796775, 0.15165867, 0.99644201, 0.16784971, 0.5979661 ],
       [0.482495  , 0.160337  , 0.461774  , 0.201887  , 1.        ,
        0.87997   , 0.5184    , 0.999826  , 0.336003  , 1.        ,
        0.999976  , 0.638277  , 1.        , 0.412773  , 0.003944  ,
        0.99549126, 0.95604009, 0.230638  , 0.91147648, 1.        ,
        0.16444213, 0.55928954, 0.77189027, 0.71236733, 0.51641239,
        0.03146116, 0.29481568, 1.        , 1.        , 0.50443713,
        0.57767397, 0.44826127, 0.99998846, 0.09246028, 0.75803202,
        1.        , 0.15831888, 0.99968521, 0.04939708, 1.00000001,
        1.        , 0.24725737, 0.15380679, 0.30942941, 0.99694684,
        0.9971687 , 0.16235529, 0.99608462, 0.17412998, 0.61639872]])
# Assuming `fidelity_tab` is a NumPy array with shape (10, N)
# where each row corresponds to a time step and each column to a fidelity value.

# Number of time steps
time_steps = fidelity_tab.shape[0]
# Define bins for fidelity quality (0 to 1)
bins = np.linspace(0, 1, 20)

# Create a figure for the histograms
plt.figure(figsize=(15, 10))

for i in range(time_steps):
    # Create a 2x5 grid of subplots for 10 time steps
    plt.subplot(2, 5, i + 1)
    plt.hist(fidelity_tab[i] , bins=bins , color='blue' , alpha = 0.75 , edgecolor='black')
    plt.title(f"Time Step {i + 1}")
    plt.xlabel("Fidelity Quality")
    plt.ylabel("Count")

plt.tight_layout()
plt.show()


# coherence analysis
coherence_tab = np.array([t1f1_coherence , t2f2_coherence , t3f3_coherence , t4f4_coherence , t5f5_coherence , t6f6_coherence , t7f7_coherence , t8f8_coherence , t9f9_coherence , t10f10_coherence , 
                          t11f11_coherence , t12f12_coherence , t13f13_coherence , t14f14_coherence , t15f15_coherence , t16f16_coherence , t17f17_coherence , t18f18_coherence , t19f19_coherence , t20f20_coherence , 
                          t21f21_coherence , t22f22_coherence , t23f23_coherence , t24f24_coherence , t25f25_coherence , t26f26_coherence , t27f27_coherence , t28f28_coherence , t29f29_coherence , t30f30_coherence , 
                          t31f31_coherence , t32f32_coherence , t33f33_coherence , t34f34_coherence , t35f35_coherence , t36f36_coherence , t37f37_coherence , t38f38_coherence , t39f39_coherence , t40f40_coherence , 
                          t41f41_coherence , t42f42_coherence , t43f43_coherence , t44f44_coherence , t45f45_coherence , t46f46_coherence , t47f47_coherence , t48f48_coherence , t49f49_coherence , t50f50_coherence]).T



coherence_tab15 = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
       [2.22097441e-04, 2.85722217e-02, 4.46363274e-02, 2.66852206e-02,
        0.00000000e+00, 6.30825644e-08, 2.37920372e-02, 9.83675227e-08,
        8.06367718e-02, 0.00000000e+00, 3.38895578e-12, 4.11590738e-04,
        0.00000000e+00, 3.30697859e-02, 1.22302973e-01, 1.11022302e-16,
        2.27245537e-04, 1.57614733e-01, 1.57962380e-04, 0.00000000e+00,
        3.36113888e-02, 3.63297077e-02, 1.01375976e-03, 2.54726599e-02,
        1.10811749e-01, 2.20213180e-02, 8.14324288e-02, 0.00000000e+00,
        0.00000000e+00, 1.27752170e-01, 4.92317762e-02, 3.34710936e-02,
        1.11022302e-16, 1.15390419e-02, 6.74606484e-04, 0.00000000e+00,
        6.80361288e-02, 1.28603003e-07, 2.00525331e-02, 0.00000000e+00,
        0.00000000e+00, 1.20580509e-02, 4.52087372e-02, 3.31848199e-02,
        2.22044605e-16, 1.34665060e-05, 7.64698575e-03, 3.33066907e-16,
        1.11259844e-01, 8.70573935e-03],
       [1.70741669e-03, 1.43716545e-01, 5.81904151e-02, 1.08859783e-01,
        0.00000000e+00, 5.48742978e-06, 8.50782475e-02, 6.11748845e-06,
        1.60153564e-01, 0.00000000e+00, 8.16475887e-10, 6.23476427e-03,
        0.00000000e+00, 7.34742502e-02, 3.57265636e-01, 1.33226763e-15,
        3.32188117e-03, 3.71475383e-01, 2.20642465e-03, 0.00000000e+00,
        9.67396833e-02, 1.31917873e-01, 1.55178948e-02, 1.10393067e-01,
        4.33896852e-01, 6.02938349e-02, 2.71767728e-01, 0.00000000e+00,
        0.00000000e+00, 4.22747326e-01, 1.94605754e-01, 1.87222036e-01,
        1.11022302e-16, 4.00370160e-02, 6.91900308e-03, 0.00000000e+00,
        1.99875693e-01, 7.84338247e-06, 5.74836903e-02, 0.00000000e+00,
        0.00000000e+00, 4.43785289e-02, 2.40721336e-01, 9.55100090e-02,
        1.11022302e-16, 1.23320168e-04, 1.01921596e-01, 1.11022302e-15,
        3.28180699e-01, 6.49592712e-03],
       [2.64854361e-03, 3.42440993e-01, 2.11837572e-02, 1.62675954e-01,
        0.00000000e+00, 9.36702744e-05, 1.58431225e-01, 6.64155891e-05,
        4.53658297e-02, 0.00000000e+00, 1.88207953e-08, 2.87958626e-02,
        0.00000000e+00, 1.31177048e-01, 4.82933716e-01, 2.22044605e-16,
        1.21000071e-02, 2.69444200e-01, 8.81192117e-03, 0.00000000e+00,
        1.03259521e-01, 2.24295101e-01, 7.29584932e-02, 2.74251063e-01,
        9.08294181e-01, 5.66236923e-02, 4.47094893e-01, 0.00000000e+00,
        0.00000000e+00, 6.77623196e-01, 4.25790025e-01, 1.23865675e-01,
        0.00000000e+00, 8.79085936e-02, 1.45124286e-02, 0.00000000e+00,
        2.00597594e-01, 8.23587595e-05, 3.09588684e-01, 0.00000000e+00,
        0.00000000e+00, 8.27545497e-02, 5.72183705e-01, 1.25913906e-01,
        5.55111512e-16, 1.19733076e-03, 3.74799861e-01, 1.77635684e-15,
        4.66839992e-01, 1.85641984e-02],
       [3.66447239e-02, 4.81973691e-01, 4.12733701e-03, 2.32493152e-02,
        0.00000000e+00, 1.51410018e-04, 2.15225212e-01, 3.48759705e-04,
        1.17884696e-01, 0.00000000e+00, 1.60196918e-07, 7.99432236e-02,
        0.00000000e+00, 1.44390183e-01, 4.69937782e-01, 1.55431223e-15,
        1.71404358e-02, 3.03778121e-02, 1.94476208e-02, 0.00000000e+00,
        3.13047765e-03, 2.02713069e-01, 2.07821413e-01, 4.14470590e-01,
        1.37396745e+00, 2.99384223e-02, 4.96909870e-01, 0.00000000e+00,
        0.00000000e+00, 7.06877938e-01, 7.16902948e-01, 1.32205515e-01,
        1.11022302e-16, 1.98984136e-01, 5.80047760e-03, 0.00000000e+00,
        1.14275912e-01, 4.11940156e-04, 4.06523432e-01, 0.00000000e+00,
        0.00000000e+00, 1.00041893e-01, 4.37630943e-01, 1.15330618e-01,
        1.11022302e-16, 1.53174860e-03, 7.27941470e-01, 1.33226763e-15,
        5.28776530e-01, 7.04101264e-03],
       [1.18352980e-01, 3.91791875e-01, 1.41535282e-01, 6.84213845e-02,
        0.00000000e+00, 9.39105414e-04, 2.37123150e-01, 1.21868185e-03,
        5.78395430e-02, 0.00000000e+00, 7.57922411e-07, 1.64890898e-01,
        0.00000000e+00, 1.64916936e-01, 4.88667860e-01, 2.66453526e-15,
        2.17553132e-03, 4.67478383e-02, 2.81058089e-02, 0.00000000e+00,
        1.65134349e-01, 2.35603455e-02, 4.43633945e-01, 4.19364916e-01,
        1.60591043e+00, 1.89873157e-01, 3.93195570e-01, 0.00000000e+00,
        0.00000000e+00, 2.68057810e-01, 1.02105535e+00, 1.77223214e-01,
        2.22044605e-16, 4.48952389e-01, 1.18124296e-02, 0.00000000e+00,
        6.83112988e-01, 1.34717184e-03, 2.36711580e-02, 0.00000000e+00,
        0.00000000e+00, 7.20533722e-02, 3.38471471e-01, 9.55307078e-02,
        1.11022302e-16, 1.46744567e-03, 8.62683828e-01, 1.55431223e-15,
        6.63366607e-01, 5.74026382e-02],
       [2.17439352e-01, 1.12196030e-01, 3.26934776e-02, 1.95236186e-02,
        0.00000000e+00, 2.73263967e-03, 2.24406266e-01, 3.26510840e-03,
        1.61947153e-01, 0.00000000e+00, 2.32503491e-06, 2.77489075e-01,
        0.00000000e+00, 7.72832594e-02, 2.92552573e-01, 4.44089210e-16,
        1.90178789e-02, 2.73525538e-01, 2.62496746e-02, 0.00000000e+00,
        2.24793856e-01, 1.53884399e-01, 7.80065062e-01, 3.10166805e-01,
        9.48583645e-01, 3.45644624e-01, 2.09100889e-01, 0.00000000e+00,
        0.00000000e+00, 1.83462167e-01, 9.09269233e-01, 3.63588452e-02,
        1.11022302e-16, 5.56300767e-01, 2.86967378e-03, 0.00000000e+00,
        1.03652014e+00, 3.30703803e-03, 1.39745716e-01, 0.00000000e+00,
        0.00000000e+00, 1.58870012e-02, 1.37930298e-01, 6.72585759e-02,
        9.99200722e-16, 2.06926708e-03, 1.96181150e-01, 2.33146835e-15,
        2.79053083e-01, 7.17067498e-02],
       [3.31820676e-02, 8.15647627e-02, 8.22671808e-02, 6.47363861e-02,
        0.00000000e+00, 4.93153090e-04, 1.95314375e-01, 7.23047545e-03,
        2.15099346e-01, 0.00000000e+00, 4.65521693e-06, 2.75439676e-01,
        0.00000000e+00, 1.08318639e-01, 2.50083992e-02, 2.66453526e-15,
        5.98832763e-03, 6.70918737e-01, 1.03516054e-02, 0.00000000e+00,
        7.44781488e-02, 6.77163134e-02, 1.18817199e+00, 1.08779108e-01,
        6.41270560e-01, 3.63454242e-01, 8.40703992e-02, 0.00000000e+00,
        0.00000000e+00, 4.61290757e-02, 4.41497146e-01, 5.05129796e-02,
        1.11022302e-16, 3.57133346e-01, 2.70771750e-02, 0.00000000e+00,
        1.13837749e+00, 6.53120965e-03, 2.12754205e-01, 0.00000000e+00,
        0.00000000e+00, 2.94101148e-03, 3.33898986e-03, 1.97025733e-03,
        8.88178420e-16, 4.08560536e-03, 3.94644522e-02, 1.44328993e-15,
        2.86777258e-02, 3.00718166e-02],
       [1.42030758e-01, 2.68202376e-02, 1.72493321e-01, 4.76228448e-01,
        0.00000000e+00, 8.42929516e-03, 1.77046651e-01, 1.38333393e-02,
        2.26921375e-02, 0.00000000e+00, 4.06866048e-06, 4.50077379e-01,
        0.00000000e+00, 1.24754929e-02, 9.18387909e-03, 2.66453526e-15,
        3.38528792e-02, 4.54448439e-01, 1.08673640e-02, 0.00000000e+00,
        1.89620436e-01, 2.90703734e-01, 1.61582720e+00, 1.31033680e-01,
        5.57143038e-01, 1.42448134e-01, 1.43379113e-01, 0.00000000e+00,
        0.00000000e+00, 1.95322736e-01, 1.59104669e-01, 5.31259281e-01,
        2.22044605e-16, 1.04259947e+00, 2.11837451e-02, 0.00000000e+00,
        7.42066965e-01, 1.07428133e-02, 4.92910920e-01, 0.00000000e+00,
        0.00000000e+00, 5.51405124e-02, 2.90679068e-01, 8.35235770e-02,
        6.66133815e-16, 4.74382995e-03, 4.15379624e-01, 1.44328993e-15,
        1.35339186e-01, 1.27058965e-01],
       [6.63409379e-02, 1.08797972e-01, 2.79923804e-01, 3.45575842e-01,
        0.00000000e+00, 2.40052560e-03, 1.93076725e-01, 2.35121639e-02,
        1.29839560e-01, 0.00000000e+00, 1.20105879e-05, 5.87359736e-01,
        0.00000000e+00, 2.11699664e-04, 2.31667537e-01, 3.55271368e-15,
        3.35808983e-02, 6.50171858e-02, 1.58140330e-02, 0.00000000e+00,
        3.16095523e-01, 5.20363771e-01, 2.00155889e+00, 3.44289294e-01,
        4.82698241e-01, 8.92667016e-02, 3.97003938e-01, 0.00000000e+00,
        0.00000000e+00, 5.07160103e-01, 7.50606168e-01, 1.10121295e-01,
        0.00000000e+00, 1.27262462e+00, 2.12090996e-02, 0.00000000e+00,
        3.53428970e-01, 1.48690211e-02, 1.27606277e-03, 0.00000000e+00,
        0.00000000e+00, 1.28719800e-01, 1.04569717e-01, 4.60004096e-02,
        5.55111512e-16, 4.84300873e-03, 1.38819064e-01, 2.55351296e-15,
        7.40142579e-02, 6.30547224e-02]])
# Assuming `coherence_tab` is a NumPy array with shape (10, N)
# where each row corresponds to a time step and each column to a 'difference in coherence' value.

# Number of time steps
time_steps = coherence_tab.shape[0]
# Define bins for fidelity quality (0 to 1)
bins = np.linspace(0, 1, 20)

# Create a figure for the histograms
plt.figure(figsize=(15, 10))

for i in range(time_steps):
    # Create a 2x5 grid of subplots for 10 time steps
    plt.subplot(2, 5, i + 1)
    plt.hist(coherence_tab[i] , bins=bins , color='red' , alpha = 0.75 , edgecolor='black')
    plt.title(f"Time Step {i + 1}")
    plt.xlabel("Coherence Quality")
    plt.ylabel("Count")

plt.tight_layout()
plt.show()


# population analysis
population_tab = np.array([pop1 , pop2 , pop3 , pop4 , pop5 , pop6 , pop7 , pop8 , pop9 , pop10 , 
                           pop11 , pop12 , pop13 , pop14 , pop15 , pop16 , pop17 , pop18 , pop19 , pop20 , 
                           pop21 , pop22 , pop23 , pop24 , pop25 , pop26 , pop27 , pop28 , pop29 , pop30 , 
                           pop31 , pop32 , pop33 , pop34 , pop35 , pop36 , pop37 , pop38 , pop39 , pop40 , 
                           pop41 , pop42 , pop43 , pop44 , pop45 , pop46 , pop47 , pop48 , pop49 , pop50]).T


population_tab15 = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00],
       [2.26225496e-02, 4.54887481e-02, 2.12010257e-02, 2.36280552e-02,
        0.00000000e+00, 2.06333248e-03, 1.24018001e-04, 2.10519833e-10,
        2.24101195e-02, 0.00000000e+00, 1.29272309e-16, 7.47124152e-05,
        0.00000000e+00, 2.24210121e-02, 6.79277670e-02, 4.40091580e-04,
        6.32857813e-04, 4.47831221e-02, 1.10194449e-06, 0.00000000e+00,
        2.24195887e-02, 3.30644341e-03, 9.94973049e-05, 2.22201614e-04,
        2.35315280e-02, 4.54845303e-02, 4.61099728e-02, 0.00000000e+00,
        0.00000000e+00, 4.38399004e-02, 2.22493790e-04, 4.04934088e-02,
        1.28169903e-10, 2.37063554e-02, 2.29821694e-03, 0.00000000e+00,
        1.45025867e-01, 2.34660735e-13, 1.45769052e-01, 0.00000000e+00,
        0.00000000e+00, 3.93360102e-02, 1.03603513e-01, 2.29077573e-02,
        1.57238512e-04, 1.60498326e-04, 6.63136317e-02, 5.21377373e-04,
        8.69177506e-02, 2.09785176e-02],
       [6.85771895e-02, 1.39674189e-01, 4.94772737e-02, 7.67727279e-02,
        0.00000000e+00, 2.06500108e-02, 1.84542167e-03, 5.27518728e-08,
        6.40752343e-02, 0.00000000e+00, 4.93727907e-13, 1.12548586e-03,
        0.00000000e+00, 6.55218758e-02, 2.05191272e-01, 3.71836167e-03,
        6.24248772e-03, 1.29893167e-01, 6.55390766e-05, 0.00000000e+00,
        6.72582861e-02, 3.59292287e-02, 1.49416109e-03, 3.26002725e-03,
        8.06663291e-02, 1.40369574e-01, 1.47844882e-01, 0.00000000e+00,
        0.00000000e+00, 1.18733871e-01, 3.27862206e-03, 7.91352324e-02,
        3.21369836e-08, 8.23105338e-02, 2.27990976e-02, 0.00000000e+00,
        3.29697652e-01, 2.36344322e-10, 3.31899944e-01, 0.00000000e+00,
        0.00000000e+00, 6.88137780e-02, 2.27353074e-01, 7.23020681e-02,
        1.05094731e-03, 1.16464554e-03, 1.86133930e-01, 4.34931654e-03,
        2.31245788e-01, 4.97891921e-02],
       [9.31099388e-02, 1.87281849e-01, 1.08167315e-01, 1.24592772e-01,
        0.00000000e+00, 4.08447969e-02, 8.25023986e-03, 1.30447529e-06,
        9.82059843e-02, 0.00000000e+00, 6.24824002e-11, 5.13857562e-03,
        0.00000000e+00, 8.79261668e-02, 2.63325886e-01, 4.24756318e-03,
        1.27612914e-02, 1.61486119e-01, 6.59453100e-04, 0.00000000e+00,
        9.36193029e-02, 8.61087393e-02, 6.79766724e-03, 1.42216939e-02,
        1.35242530e-01, 1.93885067e-01, 2.15645721e-01, 0.00000000e+00,
        0.00000000e+00, 1.16266497e-01, 1.44314045e-02, 1.21683762e-01,
        7.95506512e-07, 1.36523229e-01, 4.49268264e-02, 0.00000000e+00,
        2.54959784e-01, 1.32432332e-08, 2.35125046e-01, 0.00000000e+00,
        0.00000000e+00, 1.49625008e-01, 1.36405254e-01, 1.04761836e-01,
        6.12354014e-04, 9.47325762e-04, 2.08558489e-01, 5.03543724e-03,
        2.33251910e-01, 7.99125457e-02],
       [8.51014434e-02, 1.83826011e-01, 1.16703794e-01, 1.72971416e-01,
        0.00000000e+00, 1.11365487e-02, 2.17351850e-02, 1.23912391e-05,
        1.54108890e-01, 0.00000000e+00, 1.90501480e-09, 1.39604502e-02,
        0.00000000e+00, 1.75849878e-01, 2.31058223e-01, 5.59875473e-03,
        7.28837320e-03, 2.12091313e-01, 3.10175567e-03, 0.00000000e+00,
        1.78824715e-01, 5.83256941e-02, 1.84523347e-02, 3.61108047e-02,
        1.45087797e-01, 3.40181206e-01, 1.70693075e-01, 0.00000000e+00,
        0.00000000e+00, 1.54529104e-01, 3.72712294e-02, 2.51327136e-01,
        7.56666541e-06, 1.30052688e-01, 1.74951484e-02, 0.00000000e+00,
        3.44519665e-01, 2.25879700e-07, 3.75794119e-01, 0.00000000e+00,
        0.00000000e+00, 1.58057379e-01, 2.91253907e-01, 9.22650059e-02,
        6.56994846e-04, 7.89470376e-04, 1.78007810e-01, 4.82187096e-03,
        1.36093019e-01, 1.51690382e-01],
       [1.42740954e-01, 2.35530733e-01, 9.65355418e-02, 1.84410624e-01,
        0.00000000e+00, 6.18757876e-02, 4.19106062e-02, 6.92111516e-05,
        1.50406670e-01, 0.00000000e+00, 2.65047297e-08, 2.76859647e-02,
        0.00000000e+00, 2.03635326e-01, 2.38356639e-01, 1.15376121e-02,
        3.32036066e-02, 1.93882338e-01, 9.34054212e-03, 0.00000000e+00,
        2.45224139e-01, 1.09539491e-01, 3.68823621e-02, 6.50492410e-02,
        1.82511340e-01, 2.44699487e-01, 2.32760304e-01, 0.00000000e+00,
        0.00000000e+00, 2.67610525e-01, 6.93790199e-02, 2.07843769e-01,
        4.23302929e-05, 1.03722362e-01, 7.02009500e-02, 0.00000000e+00,
        2.94659117e-01, 1.99699159e-06, 3.05039333e-01, 0.00000000e+00,
        0.00000000e+00, 1.16279114e-01, 4.08347411e-01, 1.36561351e-01,
        7.63216146e-03, 7.28168879e-03, 3.03663916e-01, 8.18117985e-03,
        2.11814358e-01, 1.18798109e-01],
       [1.54827547e-01, 1.90371240e-01, 1.75181292e-01, 2.09931381e-01,
        0.00000000e+00, 8.42291695e-02, 6.44149039e-02, 2.74726854e-04,
        1.59061257e-01, 0.00000000e+00, 2.23662503e-07, 4.33944431e-02,
        0.00000000e+00, 2.47405504e-01, 3.56726433e-01, 2.21038104e-03,
        5.64531438e-02, 2.99056592e-01, 2.07475126e-02, 0.00000000e+00,
        1.95932921e-01, 2.15297008e-01, 5.94767040e-02, 8.86418407e-02,
        2.48079511e-01, 2.09416203e-01, 3.93399255e-01, 0.00000000e+00,
        0.00000000e+00, 3.00900880e-01, 1.01174909e-01, 1.58231986e-01,
        1.68310957e-04, 1.88794980e-01, 1.33312550e-01, 0.00000000e+00,
        3.87693733e-01, 1.15988971e-05, 3.38710823e-01, 0.00000000e+00,
        0.00000000e+00, 2.34475852e-01, 1.91303164e-01, 1.40741848e-01,
        1.10639962e-02, 1.16918332e-02, 3.01253478e-01, 3.18750946e-03,
        2.74794763e-01, 1.10553462e-01],
       [1.85895410e-01, 3.00795713e-01, 1.65981187e-01, 2.69636308e-01,
        0.00000000e+00, 4.23322591e-02, 9.29784708e-02, 8.57221027e-04,
        2.05388030e-01, 0.00000000e+00, 1.33226508e-06, 5.48519184e-02,
        0.00000000e+00, 4.14859581e-01, 4.05690445e-01, 1.82509147e-04,
        5.62816805e-02, 4.42921132e-01, 3.79487706e-02, 0.00000000e+00,
        2.40637838e-01, 2.54726249e-01, 8.10582943e-02, 1.02068905e-01,
        3.30643981e-01, 4.00345630e-01, 4.80323664e-01, 0.00000000e+00,
        0.00000000e+00, 2.11719897e-01, 1.44676000e-01, 3.59224918e-01,
        5.26040332e-04, 3.38394655e-01, 1.01338281e-01, 0.00000000e+00,
        5.63741317e-01, 5.02167584e-05, 3.89223160e-01, 0.00000000e+00,
        0.00000000e+00, 2.28832352e-01, 3.05302074e-01, 1.85876088e-01,
        1.06230634e-03, 3.11076949e-03, 2.32749807e-01, 2.53306243e-03,
        3.29199770e-01, 2.45599587e-01],
       [2.17009795e-01, 4.45849923e-01, 1.91181458e-01, 3.81809318e-01,
        0.00000000e+00, 6.42450907e-02, 1.53258547e-01, 2.23253667e-03,
        3.70540060e-01, 0.00000000e+00, 6.13213297e-06, 9.54368373e-02,
        0.00000000e+00, 4.28618614e-01, 4.56065326e-01, 1.62540539e-02,
        6.62202644e-02, 3.42754261e-01, 6.13383478e-02, 0.00000000e+00,
        3.59135364e-01, 2.58109596e-01, 9.58296214e-02, 1.47483749e-01,
        3.65314561e-01, 2.88190170e-01, 5.72636375e-01, 0.00000000e+00,
        0.00000000e+00, 3.95627615e-01, 2.07870690e-01, 3.30843713e-01,
        1.37189898e-03, 4.30285034e-01, 5.77601320e-02, 0.00000000e+00,
        7.23078924e-01, 1.74724829e-04, 3.48933008e-01, 0.00000000e+00,
        0.00000000e+00, 1.68693898e-01, 3.71340638e-01, 3.21997853e-01,
        1.83155191e-03, 9.52535712e-03, 4.43018185e-01, 1.40644520e-02,
        4.76673056e-01, 3.01852218e-01],
       [4.02901011e-01, 5.78871914e-01, 2.93625466e-01, 5.16004525e-01,
        0.00000000e+00, 1.14912135e-01, 2.48438023e-01, 5.04321785e-03,
        3.32753446e-01, 0.00000000e+00, 2.31190198e-05, 1.95988757e-01,
        0.00000000e+00, 3.48778340e-01, 8.43085415e-01, 2.20781932e-02,
        1.43087672e-01, 4.26687611e-01, 9.19555744e-02, 0.00000000e+00,
        3.83972570e-01, 2.44991243e-01, 1.00060903e-01, 2.34079380e-01,
        4.06284220e-01, 3.02539022e-01, 6.33807230e-01, 0.00000000e+00,
        0.00000000e+00, 6.44304746e-01, 2.76281581e-01, 4.59878233e-01,
        3.10156778e-03, 4.86883927e-01, 1.88605606e-01, 0.00000000e+00,
        6.35409807e-01, 5.12804961e-04, 4.39091009e-01, 0.00000000e+00,
        0.00000000e+00, 4.39700478e-01, 3.14374001e-01, 4.24797916e-01,
        2.13836587e-02, 1.79577824e-02, 6.09655168e-01, 1.25156641e-02,
        5.48410051e-01, 2.78277031e-01]])
# Assuming `population_tab` is a NumPy array with shape (10, N)
# where each row corresponds to a time step and each column to a 'difference in population' value.

# Number of time steps
time_steps = population_tab.shape[0]
# Define bins for fidelity quality (0 to 1)
bins = np.linspace(0, 1, 20)

# Create a figure for the histograms
plt.figure(figsize=(15, 10))

for i in range(time_steps):
    # Create a 2x5 grid of subplots for 10 time steps
    plt.subplot(2, 5, i + 1)
    plt.hist(population_tab[i] , bins=bins , color='green' , alpha = 0.75 , edgecolor='black')
    plt.title(f"Time Step {i + 1}")
    plt.xlabel("Population Quality")
    plt.ylabel("Count")

plt.tight_layout()
plt.show()