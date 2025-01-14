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

n = 10
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



#target individual
target1 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest1 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t1f1_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target1)) , nx.from_numpy_array(np.real(fittest1)))
# degree distribution
t1_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target1))), key=lambda x: x[1], reverse=True)]
f1_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest1))), key=lambda x: x[1], reverse=True)]
# number of connections
t1_connections = np.sum(t1_dd)
f1_connections = np.sum(f1_dd)
# distance
distance1 = 0.0015324438802052365
##########################################
##########################################
# physical properties
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
target2 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest2 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t2f2_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target2)) , nx.from_numpy_array(np.real(fittest2)))
# degree distribution
t2_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target2))), key=lambda x: x[1], reverse=True)]
f2_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest2))), key=lambda x: x[1], reverse=True)]
# number of connections
t2_connections = np.sum(t2_dd)
f2_connections = np.sum(f2_dd)
# distance
distance2 = 0.0
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
target3 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest3 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t3f3_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target3)) , nx.from_numpy_array(np.real(fittest3)))
# degree distribution
t3_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target3))), key=lambda x: x[1], reverse=True)]
f3_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest3))), key=lambda x: x[1], reverse=True)]
# number of connections
t3_connections = np.sum(t3_dd)
f3_connections = np.sum(f3_dd)
# distance
distance3 = 0.09481102114748097
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
target4 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest4 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t4f4_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target4)) , nx.from_numpy_array(np.real(fittest4)))
# degree distribution
t4_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target4))), key=lambda x: x[1], reverse=True)]
f4_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest4))), key=lambda x: x[1], reverse=True)]
# number of connections
t4_connections = np.sum(t4_dd)
f4_connections = np.sum(f4_dd)
# distance
distance4 = 0.02995304182146208
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
target5 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest5 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t5f5_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target5)) , nx.from_numpy_array(np.real(fittest5)))
# degree distribution
t5_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target5))), key=lambda x: x[1], reverse=True)]
f5_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest5))), key=lambda x: x[1], reverse=True)]
# number of connections
t5_connections = np.sum(t5_dd)
f5_connections = np.sum(f5_dd)
# distance
distance5 = 0.08796785548235508
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
target6 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest6 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t6f6_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target6)) , nx.from_numpy_array(np.real(fittest6)))
# degree distribution
t6_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target6))), key=lambda x: x[1], reverse=True)]
f6_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest6))), key=lambda x: x[1], reverse=True)]
# number of connections
t6_connections = np.sum(t6_dd)
f6_connections = np.sum(f6_dd)
# distance
distance6 = 0.5919312618382147
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
target7 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest7 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t7f7_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target7)) , nx.from_numpy_array(np.real(fittest7)))
# degree distribution
t7_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target7))), key=lambda x: x[1], reverse=True)]
f7_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest7))), key=lambda x: x[1], reverse=True)]
# number of connections
t7_connections = np.sum(t7_dd)
f7_connections = np.sum(f7_dd)
# distance
distance7 = 0.2589855251527645
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
target8 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest8 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t8f8_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target8)) , nx.from_numpy_array(np.real(fittest8)))
# degree distribution
t8_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target8))), key=lambda x: x[1], reverse=True)]
f8_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest8))), key=lambda x: x[1], reverse=True)]
# number of connections
t8_connections = np.sum(t8_dd)
f8_connections = np.sum(f8_dd)
# distance
distance8 = 0.10461972872334868
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
target9 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest9 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t9f9_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target9)) , nx.from_numpy_array(np.real(fittest9)))
# degree distribution
t9_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target9))), key=lambda x: x[1], reverse=True)]
f9_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest9))), key=lambda x: x[1], reverse=True)]
# number of connections
t9_connections = np.sum(t9_dd)
f9_connections = np.sum(f9_dd)
# distance
distance9 = 0.0
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
# target found in initial population
##########################################
##########################################
#target individual
target10 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest10 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t10f10_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target10)) , nx.from_numpy_array(np.real(fittest10)))
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
target11 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest11 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t11f11_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target11)) , nx.from_numpy_array(np.real(fittest11)))
# degree distribution
t11_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target11))), key=lambda x: x[1], reverse=True)]
f11_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest11))), key=lambda x: x[1], reverse=True)]
# number of connections
t11_connections = np.sum(t11_dd)
f11_connections = np.sum(f11_dd)
# distance
distance11 = 0.414109416343868
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
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest12 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t12f12_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target12)) , nx.from_numpy_array(np.real(fittest12)))
# degree distribution
t12_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target12))), key=lambda x: x[1], reverse=True)]
f12_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest12))), key=lambda x: x[1], reverse=True)]
# number of connections
t12_connections = np.sum(t12_dd)
f12_connections = np.sum(f12_dd)
# distance
distance12 = 0.0014867588195573989
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
target13 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest13 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t13f13_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target13)) , nx.from_numpy_array(np.real(fittest13)))
# degree distribution
t13_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target13))), key=lambda x: x[1], reverse=True)]
f13_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest13))), key=lambda x: x[1], reverse=True)]
# number of connections
t13_connections = np.sum(t13_dd)
f13_connections = np.sum(f13_dd)
# distance
distance13 = 0.05980227225440404
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
target14 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest14 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t14f14_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target14)) , nx.from_numpy_array(np.real(fittest14)))
# degree distribution
t14_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target14))), key=lambda x: x[1], reverse=True)]
f14_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest14))), key=lambda x: x[1], reverse=True)]
# number of connections
t14_connections = np.sum(t14_dd)
f14_connections = np.sum(f14_dd)
# distance
distance14 = 9.325873406851315e-15
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
target15 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest15 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t15f15_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target15)) , nx.from_numpy_array(np.real(fittest15)))
# degree distribution
t15_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target15))), key=lambda x: x[1], reverse=True)]
f15_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest15))), key=lambda x: x[1], reverse=True)]
# number of connections
t15_connections = np.sum(t15_dd)
f15_connections = np.sum(f15_dd)
# distance
distance15 = 0.455261813501477
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
target16 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest16 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t16f16_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target16)) , nx.from_numpy_array(np.real(fittest16)))
# degree distribution
t16_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target16))), key=lambda x: x[1], reverse=True)]
f16_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest16))), key=lambda x: x[1], reverse=True)]
# number of connections
t16_connections = np.sum(t16_dd)
f16_connections = np.sum(f16_dd)
# distance
distance16 = 0.04570653423977766
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
fidelity16_tab = np.round(fidelity16_tab , decimals = 6)
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
target17 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest17 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t17f17_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target17)) , nx.from_numpy_array(np.real(fittest17)))
# degree distribution
t17_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target17))), key=lambda x: x[1], reverse=True)]
f17_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest17))), key=lambda x: x[1], reverse=True)]
# number of connections
t17_connections = np.sum(t17_dd)
f17_connections = np.sum(f17_dd)
# distance
distance17 = 0.28718883906500103
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
fidelity17_tab = np.round(fidelity17_tab , decimals = 6)
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
target18 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest18 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t18f18_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target18)) , nx.from_numpy_array(np.real(fittest18)))
# degree distribution
t18_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target18))), key=lambda x: x[1], reverse=True)]
f18_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest18))), key=lambda x: x[1], reverse=True)]
# number of connections
t18_connections = np.sum(t18_dd)
f18_connections = np.sum(f18_dd)
# distance
distance18 = 0.0
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
fidelity18_tab = np.round(fidelity18_tab , decimals = 6)
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
target19 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest19 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t19f19_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target19)) , nx.from_numpy_array(np.real(fittest19)))
# degree distribution
t19_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target19))), key=lambda x: x[1], reverse=True)]
f19_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest19))), key=lambda x: x[1], reverse=True)]
# number of connections
t19_connections = np.sum(t19_dd)
f19_connections = np.sum(f19_dd)
# distance
distance19 = 0.0
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
fidelity19_tab = np.round(fidelity19_tab , decimals = 6)
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
target20 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest20 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t20f20_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target20)) , nx.from_numpy_array(np.real(fittest20)))
# degree distribution
t20_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target20))), key=lambda x: x[1], reverse=True)]
f20_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest20))), key=lambda x: x[1], reverse=True)]
# number of connections
t20_connections = np.sum(t20_dd)
f20_connections = np.sum(f20_dd)
# distance
distance20 = 0.19994388669602947
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
fidelity20_tab = np.round(fidelity20_tab , decimals = 6)
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
target21 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest21 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t21f21_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target21)) , nx.from_numpy_array(np.real(fittest21)))
# degree distribution
t21_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target21))), key=lambda x: x[1], reverse=True)]
f21_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest21))), key=lambda x: x[1], reverse=True)]
# number of connections
t21_connections = np.sum(t21_dd)
f21_connections = np.sum(f21_dd)
# distance
distance21 = 0.0
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
fidelity21_tab = np.round(fidelity21_tab , decimals = 6)
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
target22 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest22 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t22f22_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target22)) , nx.from_numpy_array(np.real(fittest22)))
# degree distribution
t22_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target22))), key=lambda x: x[1], reverse=True)]
f22_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest22))), key=lambda x: x[1], reverse=True)]
# number of connections
t22_connections = np.sum(t22_dd)
f22_connections = np.sum(f22_dd)
# distance
distance22 = 0.008980069658976353
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
fidelity22_tab = np.round(fidelity22_tab , decimals = 6)
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
target23 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest23 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t23f23_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target23)) , nx.from_numpy_array(np.real(fittest23)))
# degree distribution
t23_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target23))), key=lambda x: x[1], reverse=True)]
f23_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest23))), key=lambda x: x[1], reverse=True)]
# number of connections
t23_connections = np.sum(t23_dd)
f23_connections = np.sum(f23_dd)
# distance
distance23 = 0.0
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
fidelity23_tab = np.round(fidelity23_tab , decimals = 6)
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
target24 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest24 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t24f24_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target24)) , nx.from_numpy_array(np.real(fittest24)))
# degree distribution
t24_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target24))), key=lambda x: x[1], reverse=True)]
f24_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest24))), key=lambda x: x[1], reverse=True)]
# number of connections
t24_connections = np.sum(t24_dd)
f24_connections = np.sum(f24_dd)
# distance
distance24 = 0.2804911963353518
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
fidelity24_tab = np.round(fidelity24_tab , decimals = 6)
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
target25 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest25 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t25f25_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target25)) , nx.from_numpy_array(np.real(fittest25)))
# degree distribution
t25_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target25))), key=lambda x: x[1], reverse=True)]
f25_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest25))), key=lambda x: x[1], reverse=True)]
# number of connections
t25_connections = np.sum(t25_dd)
f25_connections = np.sum(f25_dd)
# distance
distance25 = 0.0
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
fidelity25_tab = np.round(fidelity25_tab , decimals = 6)
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
target26 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest26 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t26f26_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target26)) , nx.from_numpy_array(np.real(fittest26)))
# degree distribution
t26_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target26))), key=lambda x: x[1], reverse=True)]
f26_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest26))), key=lambda x: x[1], reverse=True)]
# number of connections
t26_connections = np.sum(t26_dd)
f26_connections = np.sum(f26_dd)
# distance
distance26 = 0.0
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
fidelity26_tab = np.round(fidelity26_tab , decimals = 6)
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
target27 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest27 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t27f27_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target27)) , nx.from_numpy_array(np.real(fittest27)))
# degree distribution
t27_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target27))), key=lambda x: x[1], reverse=True)]
f27_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest27))), key=lambda x: x[1], reverse=True)]
# number of connections
t27_connections = np.sum(t27_dd)
f27_connections = np.sum(f27_dd)
# distance
distance27 = 0.0
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
fidelity27_tab = np.round(fidelity27_tab , decimals = 6)
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
target28 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest28 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t28f28_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target28)) , nx.from_numpy_array(np.real(fittest28)))
# degree distribution
t28_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target28))), key=lambda x: x[1], reverse=True)]
f28_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest28))), key=lambda x: x[1], reverse=True)]
# number of connections
t28_connections = np.sum(t28_dd)
f28_connections = np.sum(f28_dd)
# distance
distance28 = 0.038070708738333514
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
fidelity28_tab = np.round(fidelity28_tab , decimals = 6)
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
target29 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest29 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t29f29_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target29)) , nx.from_numpy_array(np.real(fittest29)))
# degree distribution
t29_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target29))), key=lambda x: x[1], reverse=True)]
f29_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest29))), key=lambda x: x[1], reverse=True)]
# number of connections
t29_connections = np.sum(t29_dd)
f29_connections = np.sum(f29_dd)
# distance
distance29 = 0.1904039743532535
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
fidelity29_tab = np.round(fidelity29_tab , decimals = 6)
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
target30 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest30 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t30f30_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target30)) , nx.from_numpy_array(np.real(fittest30)))
# degree distribution
t30_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target30))), key=lambda x: x[1], reverse=True)]
f30_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest30))), key=lambda x: x[1], reverse=True)]
# number of connections
t30_connections = np.sum(t30_dd)
f30_connections = np.sum(f30_dd)
# distance
distance30 = 0.08484606086682456
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
fidelity30_tab = np.round(fidelity30_tab , decimals = 6)
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
target31 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest31 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t31f31_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target31)) , nx.from_numpy_array(np.real(fittest31)))
# degree distribution
t31_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target31))), key=lambda x: x[1], reverse=True)]
f31_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest31))), key=lambda x: x[1], reverse=True)]
# number of connections
t31_connections = np.sum(t31_dd)
f31_connections = np.sum(f31_dd)
# distance
distance31 = 0.19996226673213935
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
fidelity31_tab = np.round(fidelity31_tab , decimals = 6)
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
target32 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest32 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t32f32_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target32)) , nx.from_numpy_array(np.real(fittest32)))
# degree distribution
t32_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target32))), key=lambda x: x[1], reverse=True)]
f32_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest32))), key=lambda x: x[1], reverse=True)]
# number of connections
t32_connections = np.sum(t32_dd)
f32_connections = np.sum(f32_dd)
# distance
distance32 = 0.0
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
fidelity32_tab = np.round(fidelity32_tab , decimals = 6)
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
target33 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest33 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
#measuring the similarity between two graphs through graph_edit_distance
t33f33_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target33)) , nx.from_numpy_array(np.real(fittest33)))
# degree distribution
t33_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target33))), key=lambda x: x[1], reverse=True)]
f33_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest33))), key=lambda x: x[1], reverse=True)]
# number of connections
t33_connections = np.sum(t33_dd)
f33_connections = np.sum(f33_dd)
# distance
distance33 = 0.0
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
fidelity33_tab = np.round(fidelity33_tab , decimals = 6)
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
target34 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest34 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t34f34_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target34)) , nx.from_numpy_array(np.real(fittest34)))
# degree distribution
t34_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target34))), key=lambda x: x[1], reverse=True)]
f34_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest34))), key=lambda x: x[1], reverse=True)]
# number of connections
t34_connections = np.sum(t34_dd)
f34_connections = np.sum(f34_dd)
# distance
distance34 = 0.0544510296327152
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
fidelity34_tab = np.round(fidelity34_tab , decimals = 6)
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
target35 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest35 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t35f35_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target35)) , nx.from_numpy_array(np.real(fittest35)))
# degree distribution
t35_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target35))), key=lambda x: x[1], reverse=True)]
f35_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest35))), key=lambda x: x[1], reverse=True)]
# number of connections
t35_connections = np.sum(t35_dd)
f35_connections = np.sum(f35_dd)
# distance
distance35 = 0.22757894509475596
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
fidelity35_tab = np.round(fidelity35_tab , decimals = 6)
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
target36 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest36 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t36f36_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target36)) , nx.from_numpy_array(np.real(fittest36)))
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
fidelity36_tab = np.round(fidelity36_tab , decimals = 6)
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
target37 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest37 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t37f37_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target37)) , nx.from_numpy_array(np.real(fittest37)))
# degree distribution
t37_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target37))), key=lambda x: x[1], reverse=True)]
f37_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest37))), key=lambda x: x[1], reverse=True)]
# number of connections
t37_connections = np.sum(t37_dd)
f37_connections = np.sum(f37_dd)
# distance
distance37 = 0.2246676375705725
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
fidelity37_tab = np.round(fidelity37_tab , decimals = 6)
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
target38 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest38 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t38f38_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target38)) , nx.from_numpy_array(np.real(fittest38)))
# degree distribution
t38_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target38))), key=lambda x: x[1], reverse=True)]
f38_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest38))), key=lambda x: x[1], reverse=True)]
# number of connections
t38_connections = np.sum(t38_dd)
f38_connections = np.sum(f38_dd)
# distance
distance38 = 0.2886211099468675
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
fidelity38_tab = np.round(fidelity38_tab , decimals = 6)
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
target39 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest39 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t39f39_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target39)) , nx.from_numpy_array(np.real(fittest39)))
# degree distribution
t39_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target39))), key=lambda x: x[1], reverse=True)]
f39_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest39))), key=lambda x: x[1], reverse=True)]
# number of connections
t39_connections = np.sum(t39_dd)
f39_connections = np.sum(f39_dd)
# distance
distance39 = 0.0
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
fidelity39_tab = np.round(fidelity39_tab , decimals = 6)
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
target40 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest40 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t40f40_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target40)) , nx.from_numpy_array(np.real(fittest40)))
# degree distribution
t40_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target40))), key=lambda x: x[1], reverse=True)]
f40_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest40))), key=lambda x: x[1], reverse=True)]
# number of connections
t40_connections = np.sum(t40_dd)
f40_connections = np.sum(f40_dd)
# distance
distance40 = 0.09106886977293349
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
fidelity40_tab = np.round(fidelity40_tab , decimals = 6)
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
target41 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest41 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t41f41_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target41)) , nx.from_numpy_array(np.real(fittest41)))
# degree distribution
t41_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target41))), key=lambda x: x[1], reverse=True)]
f41_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest41))), key=lambda x: x[1], reverse=True)]
# number of connections
t41_connections = np.sum(t41_dd)
f41_connections = np.sum(f41_dd)
# distance
distance41 = 0.018497301264533306
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
fidelity41_tab = np.round(fidelity41_tab , decimals = 6)
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
target42 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest42 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t42f42_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target42)) , nx.from_numpy_array(np.real(fittest42)))
# degree distribution
t42_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target42))), key=lambda x: x[1], reverse=True)]
f42_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest42))), key=lambda x: x[1], reverse=True)]
# number of connections
t42_connections = np.sum(t42_dd)
f42_connections = np.sum(f42_dd)
# distance
distance42 = 0.31297815353408465
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
fidelity42_tab = np.round(fidelity42_tab , decimals = 6)
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
target43 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest43 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
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
#measuring the similarity between two graphs through graph_edit_distance
t43f43_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target43)) , nx.from_numpy_array(np.real(fittest43)))
# degree distribution
t43_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target43))), key=lambda x: x[1], reverse=True)]
f43_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest43))), key=lambda x: x[1], reverse=True)]
# number of connections
t43_connections = np.sum(t43_dd)
f43_connections = np.sum(f43_dd)
# distance
distance43 = 0.014204895227422853
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
fidelity43_tab = np.round(fidelity43_tab , decimals = 6)
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
target44 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest44 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
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
#measuring the similarity between two graphs through graph_edit_distance
t44f44_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target44)) , nx.from_numpy_array(np.real(fittest44)))
# degree distribution
t44_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target44))), key=lambda x: x[1], reverse=True)]
f44_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest44))), key=lambda x: x[1], reverse=True)]
# number of connections
t44_connections = np.sum(t44_dd)
f44_connections = np.sum(f44_dd)
# distance
distance44 = 0.10850738731807785
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
fidelity44_tab = np.round(fidelity44_tab , decimals = 6)
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
target45 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest45 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
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
#measuring the similarity between two graphs through graph_edit_distance
t45f45_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target45)) , nx.from_numpy_array(np.real(fittest45)))
# degree distribution
t45_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target45))), key=lambda x: x[1], reverse=True)]
f45_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest45))), key=lambda x: x[1], reverse=True)]
# number of connections
t45_connections = np.sum(t45_dd)
f45_connections = np.sum(f45_dd)
# distance
distance45 = 0.15104647336458177
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
fidelity45_tab = np.round(fidelity45_tab , decimals = 6)
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
target46 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest46 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
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
#measuring the similarity between two graphs through graph_edit_distance
t46f46_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target46)) , nx.from_numpy_array(np.real(fittest46)))
# degree distribution
t46_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target46))), key=lambda x: x[1], reverse=True)]
f46_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest46))), key=lambda x: x[1], reverse=True)]
# number of connections
t46_connections = np.sum(t46_dd)
f46_connections = np.sum(f46_dd)
# distance
distance46 = 0.11438064090897604
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
fidelity46_tab = np.round(fidelity46_tab , decimals = 6)
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
target47 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest47 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
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
#measuring the similarity between two graphs through graph_edit_distance
t47f47_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target47)) , nx.from_numpy_array(np.real(fittest47)))
# degree distribution
t47_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target47))), key=lambda x: x[1], reverse=True)]
f47_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest47))), key=lambda x: x[1], reverse=True)]
# number of connections
t47_connections = np.sum(t47_dd)
f47_connections = np.sum(f47_dd)
# distance
distance47 = 0.0
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
fidelity47_tab = np.round(fidelity47_tab , decimals = 6)
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
target48 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest48 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
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
#measuring the similarity between two graphs through graph_edit_distance
t48f48_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target48)) , nx.from_numpy_array(np.real(fittest48)))
# degree distribution
t48_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target48))), key=lambda x: x[1], reverse=True)]
f48_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest48))), key=lambda x: x[1], reverse=True)]
# number of connections
t48_connections = np.sum(t48_dd)
f48_connections = np.sum(f48_dd)
# distance
distance48 = 0.03672752732879825
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
fidelity48_tab = np.round(fidelity48_tab , decimals = 6)
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
target49 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest49 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
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
#measuring the similarity between two graphs through graph_edit_distance
t49f49_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target49)) , nx.from_numpy_array(np.real(fittest49)))
# degree distribution
t49_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target49))), key=lambda x: x[1], reverse=True)]
f49_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest49))), key=lambda x: x[1], reverse=True)]
# number of connections
t49_connections = np.sum(t49_dd)
f49_connections = np.sum(f49_dd)
# distance
distance49 = 0.049150553629848104
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
fidelity49_tab = np.round(fidelity49_tab , decimals = 6)
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
target50 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest50 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
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
# measuring the similarity between two graphs through graph_edit_distance
t50f50_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target50)) , nx.from_numpy_array(np.real(fittest50)))
# degree distribution
t50_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target50))), key=lambda x: x[1], reverse=True)]
f50_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest50))), key=lambda x: x[1], reverse=True)]
# number of connections
t50_connections = np.sum(t50_dd)
f50_connections = np.sum(f50_dd)
# distance
distance50 = 0.1586489165028555
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
fidelity50_tab = np.round(fidelity50_tab , decimals = 6)
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
#target individual
target51 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest51 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target51cc = nx.node_connected_component(nx.from_numpy_array(np.real(target51)) , 1)
fittest51cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest51)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr51 = target51 == fittest51
# Count the number of False values
false_count51 = np.sum(arr51 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t51f51_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target51)) , nx.from_numpy_array(np.real(fittest51)))
# WL graph isomorphism test
t51_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target51)))
f51_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest51)))
# measuring the similarity between two graphs through graph_edit_distance
t51f51_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target51)) , nx.from_numpy_array(np.real(fittest51)))
# degree distribution
t51_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target51))), key=lambda x: x[1], reverse=True)]
f51_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest51))), key=lambda x: x[1], reverse=True)]
# number of connections
t51_connections = np.sum(t51_dd)
f51_connections = np.sum(f51_dd)
# distance
distance51 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest51_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest51 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest51 * t).T))

            ]
def fittest51_evolution_data():
    data = []
    for t in ts:
        data.append(fittest51_evolution(t))
    return data
#target
def target51_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target51 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target51 * t).T))

            ]
def target51_evolution_data():
    data = []
    for t in ts:
        data.append(target51_evolution(t))
    return data
# fidelity results
fidelity51_tab = []
for i in range(len(ts)):
    fidelity51_tab.append(fidelity(target51_evolution_data()[i][0] , fittest51_evolution_data()[i][0]))
fidelity51_tab = np.round(fidelity51_tab , decimals = 6)
# coherence results
t51_coherence = coherence(target51_evolution_data())
f51_coherence = coherence(fittest51_evolution_data())
t51f51_coherence = []
for i in range(len(ts)):
     t51f51_coherence.append(np.abs(t51_coherence[i] - f51_coherence[i]))
# population results
pop51 = []
for i in range(len(ts)):
     pop51.append(np.sum(populations(target51_evolution_data() , fittest51_evolution_data())[i]))


##########################################
##########################################
#target individual
target52 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest52 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target52cc = nx.node_connected_component(nx.from_numpy_array(np.real(target52)) , 1)
fittest52cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest52)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr52 = target52 == fittest52
# Count the number of False values
false_count52 = np.sum(arr52 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t52f52_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target52)) , nx.from_numpy_array(np.real(fittest52)))
# WL graph isomorphism test
t52_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target52)))
f52_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest52)))
# measuring the similarity between two graphs through graph_edit_distance
t52f52_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target52)) , nx.from_numpy_array(np.real(fittest52)))
# degree distribution
t52_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target52))), key=lambda x: x[1], reverse=True)]
f52_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest52))), key=lambda x: x[1], reverse=True)]
# number of connections
t52_connections = np.sum(t52_dd)
f52_connections = np.sum(f52_dd)
# distance
distance52 = 0.3998011677522252
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest52_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest52 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest52 * t).T))

            ]
def fittest52_evolution_data():
    data = []
    for t in ts:
        data.append(fittest52_evolution(t))
    return data
#target
def target52_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target52 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target52 * t).T))

            ]
def target52_evolution_data():
    data = []
    for t in ts:
        data.append(target52_evolution(t))
    return data
# fidelity results
fidelity52_tab = []
for i in range(len(ts)):
    fidelity52_tab.append(fidelity(target52_evolution_data()[i][0] , fittest52_evolution_data()[i][0]))
fidelity52_tab = np.round(fidelity52_tab , decimals = 6)
# coherence results
t52_coherence = coherence(target52_evolution_data())
f52_coherence = coherence(fittest52_evolution_data())
t52f52_coherence = []
for i in range(len(ts)):
     t52f52_coherence.append(np.abs(t52_coherence[i] - f52_coherence[i]))
# population results
pop52 = []
for i in range(len(ts)):
     pop52.append(np.sum(populations(target52_evolution_data() , fittest52_evolution_data())[i]))



##########################################
##########################################
#target individual
target53 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest53 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target53cc = nx.node_connected_component(nx.from_numpy_array(np.real(target53)) , 1)
fittest53cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest53)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr53 = target53 == fittest53
# Count the number of False values
false_count53 = np.sum(arr53 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t53f53_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target53)) , nx.from_numpy_array(np.real(fittest53)))
# WL graph isomorphism test
t53_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target53)))
f53_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest53)))
#measuring the similarity between two graphs through graph_edit_distance
t53f53_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target53)) , nx.from_numpy_array(np.real(fittest53)))
# degree distribution
t53_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target53))), key=lambda x: x[1], reverse=True)]
f53_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest53))), key=lambda x: x[1], reverse=True)]
# number of connections
t53_connections = np.sum(t53_dd)
f53_connections = np.sum(f53_dd)
# distance
distance53 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest53_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest53 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest53 * t).T))

            ]
def fittest53_evolution_data():
    data = []
    for t in ts:
        data.append(fittest53_evolution(t))
    return data
#target
def target53_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target53 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target53 * t).T))

            ]
def target53_evolution_data():
    data = []
    for t in ts:
        data.append(target53_evolution(t))
    return data
# fidelity results
fidelity53_tab = []
for i in range(len(ts)):
    fidelity53_tab.append(fidelity(target53_evolution_data()[i][0] , fittest53_evolution_data()[i][0]))
fidelity53_tab = np.round(fidelity53_tab , decimals = 6)
# coherence results
t53_coherence = coherence(target53_evolution_data())
f53_coherence = coherence(fittest53_evolution_data())
t53f53_coherence = []
for i in range(len(ts)):
     t53f53_coherence.append(np.abs(t53_coherence[i] - f53_coherence[i]))
# population results
pop53 = []
for i in range(len(ts)):
     pop53.append(np.sum(populations(target53_evolution_data() , fittest53_evolution_data())[i]))



##########################################
##########################################
#target individual
target54 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest54 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target54cc = nx.node_connected_component(nx.from_numpy_array(np.real(target54)) , 1)
fittest54cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest54)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr54 = target54 == fittest54
# Count the number of False values
false_count54 = np.sum(arr54 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t54f54_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target54)) , nx.from_numpy_array(np.real(fittest54)))
# WL graph isomorphism test
t54_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target54)))
f54_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest54)))
#measuring the similarity between two graphs through graph_edit_distance
t54f54_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target54)) , nx.from_numpy_array(np.real(fittest54)))
# degree distribution
t54_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target54))), key=lambda x: x[1], reverse=True)]
f54_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest54))), key=lambda x: x[1], reverse=True)]
# number of connections
t54_connections = np.sum(t54_dd)
f54_connections = np.sum(f54_dd)
# distance
distance54 = 0.006764108438414351
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest54_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest54 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest54 * t).T))

            ]
def fittest54_evolution_data():
    data = []
    for t in ts:
        data.append(fittest54_evolution(t))
    return data
#target
def target54_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target54 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target54 * t).T))

            ]
def target54_evolution_data():
    data = []
    for t in ts:
        data.append(target54_evolution(t))
    return data
# fidelity results
fidelity54_tab = []
for i in range(len(ts)):
    fidelity54_tab.append(fidelity(target54_evolution_data()[i][0] , fittest54_evolution_data()[i][0]))
fidelity54_tab = np.round(fidelity54_tab , decimals = 6)
# coherence results
t54_coherence = coherence(target54_evolution_data())
f54_coherence = coherence(fittest54_evolution_data())
t54f54_coherence = []
for i in range(len(ts)):
     t54f54_coherence.append(np.abs(t54_coherence[i] - f54_coherence[i]))
# population results
pop54 = []
for i in range(len(ts)):
     pop54.append(np.sum(populations(target54_evolution_data() , fittest54_evolution_data())[i]))


##########################################
##########################################
#target individual
target55 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest55 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target55cc = nx.node_connected_component(nx.from_numpy_array(np.real(target55)) , 1)
fittest55cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest55)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr55 = target55 == fittest55
# Count the number of False values
false_count55 = np.sum(arr55 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t55f55_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target55)) , nx.from_numpy_array(np.real(fittest55)))
# WL graph isomorphism test
t55_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target55)))
f55_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest55)))
#measuring the similarity between two graphs through graph_edit_distance
t55f55_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target55)) , nx.from_numpy_array(np.real(fittest55)))
# degree distribution
t55_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target55))), key=lambda x: x[1], reverse=True)]
f55_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest55))), key=lambda x: x[1], reverse=True)]
# number of connections
t55_connections = np.sum(t55_dd)
f55_connections = np.sum(f55_dd)
# distance
distance55 = 0.04523044611624871
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest55_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest55 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest55 * t).T))

            ]
def fittest55_evolution_data():
    data = []
    for t in ts:
        data.append(fittest55_evolution(t))
    return data
#target
def target55_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target55 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target55 * t).T))

            ]
def target55_evolution_data():
    data = []
    for t in ts:
        data.append(target55_evolution(t))
    return data
# fidelity results
fidelity55_tab = []
for i in range(len(ts)):
    fidelity55_tab.append(fidelity(target55_evolution_data()[i][0] , fittest55_evolution_data()[i][0]))
fidelity55_tab = np.round(fidelity55_tab , decimals = 6)
# coherence results
t55_coherence = coherence(target55_evolution_data())
f55_coherence = coherence(fittest55_evolution_data())
t55f55_coherence = []
for i in range(len(ts)):
     t55f55_coherence.append(np.abs(t55_coherence[i] - f55_coherence[i]))
# population results
pop55 = []
for i in range(len(ts)):
     pop55.append(np.sum(populations(target55_evolution_data() , fittest55_evolution_data())[i]))


##########################################
##########################################
#target individual
target56 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest56 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target56cc = nx.node_connected_component(nx.from_numpy_array(np.real(target56)) , 1)
fittest56cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest56)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr56 = target56 == fittest56
# Count the number of False values
false_count56 = np.sum(arr56 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t56f56_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target56)) , nx.from_numpy_array(np.real(fittest56)))
# WL graph isomorphism test
t56_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target56)))
f56_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest56)))
#measuring the similarity between two graphs through graph_edit_distance
t56f56_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target56)) , nx.from_numpy_array(np.real(fittest56)))
# degree distribution
t56_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target56))), key=lambda x: x[1], reverse=True)]
f56_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest56))), key=lambda x: x[1], reverse=True)]
# number of connections
t56_connections = np.sum(t56_dd)
f56_connections = np.sum(f56_dd)
# distance
distance56 = 0.03513501725322321
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest56_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest56 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest56 * t).T))

            ]
def fittest56_evolution_data():
    data = []
    for t in ts:
        data.append(fittest56_evolution(t))
    return data
#target
def target56_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target56 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target56 * t).T))

            ]
def target56_evolution_data():
    data = []
    for t in ts:
        data.append(target56_evolution(t))
    return data
# fidelity results
fidelity56_tab = []
for i in range(len(ts)):
    fidelity56_tab.append(fidelity(target56_evolution_data()[i][0] , fittest56_evolution_data()[i][0]))
fidelity56_tab = np.round(fidelity56_tab , decimals = 6)
# coherence results
t56_coherence = coherence(target56_evolution_data())
f56_coherence = coherence(fittest56_evolution_data())
t56f56_coherence = []
for i in range(len(ts)):
     t56f56_coherence.append(np.abs(t56_coherence[i] - f56_coherence[i]))
# population results
pop56 = []
for i in range(len(ts)):
     pop56.append(np.sum(populations(target56_evolution_data() , fittest56_evolution_data())[i]))



##########################################
##########################################
#target individual
target57 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest57 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target57cc = nx.node_connected_component(nx.from_numpy_array(np.real(target57)) , 1)
fittest57cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest57)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr57 = target57 == fittest57
# Count the number of False values
false_count57 = np.sum(arr57 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t57f57_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target57)) , nx.from_numpy_array(np.real(fittest57)))
# WL graph isomorphism test
t57_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target57)))
f57_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest57)))
#measuring the similarity between two graphs through graph_edit_distance
t57f57_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target57)) , nx.from_numpy_array(np.real(fittest57)))
# degree distribution
t57_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target57))), key=lambda x: x[1], reverse=True)]
f57_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest57))), key=lambda x: x[1], reverse=True)]
# number of connections
t57_connections = np.sum(t57_dd)
f57_connections = np.sum(f57_dd)
# distance
distance57 = 0.04397456425793911
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest57_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest57 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest57 * t).T))

            ]
def fittest57_evolution_data():
    data = []
    for t in ts:
        data.append(fittest57_evolution(t))
    return data
#target
def target57_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target57 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target57 * t).T))

            ]
def target57_evolution_data():
    data = []
    for t in ts:
        data.append(target57_evolution(t))
    return data
# fidelity results
fidelity57_tab = []
for i in range(len(ts)):
    fidelity57_tab.append(fidelity(target57_evolution_data()[i][0] , fittest57_evolution_data()[i][0]))
fidelity57_tab = np.round(fidelity57_tab , decimals = 6)
# coherence results
t57_coherence = coherence(target57_evolution_data())
f57_coherence = coherence(fittest57_evolution_data())
t57f57_coherence = []
for i in range(len(ts)):
     t57f57_coherence.append(np.abs(t57_coherence[i] - f57_coherence[i]))
# population results
pop57 = []
for i in range(len(ts)):
     pop57.append(np.sum(populations(target57_evolution_data() , fittest57_evolution_data())[i]))




##########################################
##########################################
#target individual
target58 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest58 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target58cc = nx.node_connected_component(nx.from_numpy_array(np.real(target58)) , 1)
fittest58cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest58)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr58 = target58 == fittest58
# Count the number of False values
false_count58 = np.sum(arr58 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t58f58_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target58)) , nx.from_numpy_array(np.real(fittest58)))
# WL graph isomorphism test
t58_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target58)))
f58_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest58)))
#measuring the similarity between two graphs through graph_edit_distance
t58f58_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target58)) , nx.from_numpy_array(np.real(fittest58)))
# degree distribution
t58_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target58))), key=lambda x: x[1], reverse=True)]
f58_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest58))), key=lambda x: x[1], reverse=True)]
# number of connections
t58_connections = np.sum(t58_dd)
f58_connections = np.sum(f58_dd)
# distance
distance58 = 0.04191685713754545
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest58_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest58 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest58 * t).T))

            ]
def fittest58_evolution_data():
    data = []
    for t in ts:
        data.append(fittest58_evolution(t))
    return data
#target
def target58_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target58 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target58 * t).T))

            ]
def target58_evolution_data():
    data = []
    for t in ts:
        data.append(target58_evolution(t))
    return data
# fidelity results
fidelity58_tab = []
for i in range(len(ts)):
    fidelity58_tab.append(fidelity(target58_evolution_data()[i][0] , fittest58_evolution_data()[i][0]))
fidelity58_tab = np.round(fidelity58_tab , decimals = 6)
# coherence results
t58_coherence = coherence(target58_evolution_data())
f58_coherence = coherence(fittest58_evolution_data())
t58f58_coherence = []
for i in range(len(ts)):
     t58f58_coherence.append(np.abs(t58_coherence[i] - f58_coherence[i]))
# population results
pop58 = []
for i in range(len(ts)):
     pop58.append(np.sum(populations(target58_evolution_data() , fittest58_evolution_data())[i]))



##########################################
##########################################
#target individual
target59 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest59 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target59cc = nx.node_connected_component(nx.from_numpy_array(np.real(target59)) , 1)
fittest59cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest59)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr59 = target59 == fittest59
# Count the number of False values
false_count59 = np.sum(arr59 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t59f59_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target59)) , nx.from_numpy_array(np.real(fittest59)))
# WL graph isomorphism test
t59_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target59)))
f59_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest59)))
#measuring the similarity between two graphs through graph_edit_distance
t59f59_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target59)) , nx.from_numpy_array(np.real(fittest59)))
# degree distribution
t59_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target59))), key=lambda x: x[1], reverse=True)]
f59_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest59))), key=lambda x: x[1], reverse=True)]
# number of connections
t59_connections = np.sum(t59_dd)
f59_connections = np.sum(f59_dd)
# distance
distance59 = 0.2611401221118911
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest59_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest59 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest59 * t).T))

            ]
def fittest59_evolution_data():
    data = []
    for t in ts:
        data.append(fittest59_evolution(t))
    return data
#target
def target59_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target59 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target59 * t).T))

            ]
def target59_evolution_data():
    data = []
    for t in ts:
        data.append(target59_evolution(t))
    return data
# fidelity results
fidelity59_tab = []
for i in range(len(ts)):
    fidelity59_tab.append(fidelity(target59_evolution_data()[i][0] , fittest59_evolution_data()[i][0]))
fidelity59_tab = np.round(fidelity59_tab , decimals = 6)
# coherence results
t59_coherence = coherence(target59_evolution_data())
f59_coherence = coherence(fittest59_evolution_data())
t59f59_coherence = []
for i in range(len(ts)):
     t59f59_coherence.append(np.abs(t59_coherence[i] - f59_coherence[i]))
# population results
pop59 = []
for i in range(len(ts)):
     pop59.append(np.sum(populations(target59_evolution_data() , fittest59_evolution_data())[i]))








##########################################
##########################################
#target individual
target60 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest60 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target60cc = nx.node_connected_component(nx.from_numpy_array(np.real(target60)) , 1)
fittest60cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest60)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr60 = target60 == fittest60
# Count the number of False values
false_count60 = np.sum(arr60 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t60f60_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target60)) , nx.from_numpy_array(np.real(fittest60)))
# WL graph isomorphism test
t60_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target60)))
f60_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest60)))
# measuring the similarity between two graphs through graph_edit_distance
t60f60_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target60)) , nx.from_numpy_array(np.real(fittest60)))
# degree distribution
t60_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target60))), key=lambda x: x[1], reverse=True)]
f60_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest60))), key=lambda x: x[1], reverse=True)]
# number of connections
t60_connections = np.sum(t60_dd)
f60_connections = np.sum(f60_dd)
# distance
distance60 = 0.08486789477855361
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest60_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest60 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest60 * t).T))

            ]
def fittest60_evolution_data():
    data = []
    for t in ts:
        data.append(fittest60_evolution(t))
    return data
#target
def target60_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target60 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target60 * t).T))

            ]
def target60_evolution_data():
    data = []
    for t in ts:
        data.append(target60_evolution(t))
    return data
# fidelity results
fidelity60_tab = []
for i in range(len(ts)):
    fidelity60_tab.append(fidelity(target60_evolution_data()[i][0] , fittest60_evolution_data()[i][0]))
fidelity60_tab = np.round(fidelity60_tab , decimals = 6)
# coherence results
t60_coherence = coherence(target60_evolution_data())
f60_coherence = coherence(fittest60_evolution_data())
t60f60_coherence = []
for i in range(len(ts)):
     t60f60_coherence.append(np.abs(t60_coherence[i] - f60_coherence[i]))
# population results
pop60 = []
for i in range(len(ts)):
     pop60.append(np.sum(populations(target60_evolution_data() , fittest60_evolution_data())[i]))


##########################################
##########################################
#target individual
target61 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest61 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target61cc = nx.node_connected_component(nx.from_numpy_array(np.real(target61)) , 1)
fittest61cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest61)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr61 = target61 == fittest61
# Count the number of False values
false_count61 = np.sum(arr61 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t61f61_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target61)) , nx.from_numpy_array(np.real(fittest61)))
# WL graph isomorphism test
t61_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target61)))
f61_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest61)))
# measuring the similarity between two graphs through graph_edit_distance
t61f61_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target61)) , nx.from_numpy_array(np.real(fittest61)))
# degree distribution
t61_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target61))), key=lambda x: x[1], reverse=True)]
f61_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest61))), key=lambda x: x[1], reverse=True)]
# number of connections
t61_connections = np.sum(t61_dd)
f61_connections = np.sum(f61_dd)
# distance
distance61 = 0.22233989389927622
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest61_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest61 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest61 * t).T))

            ]
def fittest61_evolution_data():
    data = []
    for t in ts:
        data.append(fittest61_evolution(t))
    return data
#target
def target61_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target61 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target61 * t).T))

            ]
def target61_evolution_data():
    data = []
    for t in ts:
        data.append(target61_evolution(t))
    return data
# fidelity results
fidelity61_tab = []
for i in range(len(ts)):
    fidelity61_tab.append(fidelity(target61_evolution_data()[i][0] , fittest61_evolution_data()[i][0]))
fidelity61_tab = np.round(fidelity61_tab , decimals = 6)
# coherence results
t61_coherence = coherence(target61_evolution_data())
f61_coherence = coherence(fittest61_evolution_data())
t61f61_coherence = []
for i in range(len(ts)):
     t61f61_coherence.append(np.abs(t61_coherence[i] - f61_coherence[i]))
# population results
pop61 = []
for i in range(len(ts)):
     pop61.append(np.sum(populations(target61_evolution_data() , fittest61_evolution_data())[i]))


##########################################
##########################################
#target individual
target62 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest62 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target62cc = nx.node_connected_component(nx.from_numpy_array(np.real(target62)) , 1)
fittest62cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest62)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr62 = target62 == fittest62
# Count the number of False values
false_count62 = np.sum(arr62 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t62f62_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target62)) , nx.from_numpy_array(np.real(fittest62)))
# WL graph isomorphism test
t62_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target62)))
f62_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest62)))
# measuring the similarity between two graphs through graph_edit_distance
t62f62_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target62)) , nx.from_numpy_array(np.real(fittest62)))
# degree distribution
t62_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target62))), key=lambda x: x[1], reverse=True)]
f62_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest62))), key=lambda x: x[1], reverse=True)]
# number of connections
t62_connections = np.sum(t62_dd)
f62_connections = np.sum(f62_dd)
# distance
distance62 = 0.2507741810277382
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest62_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest62 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest62 * t).T))

            ]
def fittest62_evolution_data():
    data = []
    for t in ts:
        data.append(fittest62_evolution(t))
    return data
#target
def target62_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target62 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target62 * t).T))

            ]
def target62_evolution_data():
    data = []
    for t in ts:
        data.append(target62_evolution(t))
    return data
# fidelity results
fidelity62_tab = []
for i in range(len(ts)):
    fidelity62_tab.append(fidelity(target62_evolution_data()[i][0] , fittest62_evolution_data()[i][0]))
fidelity62_tab = np.round(fidelity62_tab , decimals = 6)
# coherence results
t62_coherence = coherence(target62_evolution_data())
f62_coherence = coherence(fittest62_evolution_data())
t62f62_coherence = []
for i in range(len(ts)):
     t62f62_coherence.append(np.abs(t62_coherence[i] - f62_coherence[i]))
# population results
pop62 = []
for i in range(len(ts)):
     pop62.append(np.sum(populations(target62_evolution_data() , fittest62_evolution_data())[i]))



##########################################
##########################################
#target individual
target63 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest63 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target63cc = nx.node_connected_component(nx.from_numpy_array(np.real(target63)) , 1)
fittest63cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest63)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr63 = target63 == fittest63
# Count the number of False values
false_count63 = np.sum(arr63 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t63f63_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target63)) , nx.from_numpy_array(np.real(fittest63)))
# WL graph isomorphism test
t63_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target63)))
f63_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest63)))
#measuring the similarity between two graphs through graph_edit_distance
t63f63_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target63)) , nx.from_numpy_array(np.real(fittest63)))
# degree distribution
t63_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target63))), key=lambda x: x[1], reverse=True)]
f63_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest63))), key=lambda x: x[1], reverse=True)]
# number of connections
t63_connections = np.sum(t63_dd)
f63_connections = np.sum(f63_dd)
# distance
distance63 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest63_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest63 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest63 * t).T))

            ]
def fittest63_evolution_data():
    data = []
    for t in ts:
        data.append(fittest63_evolution(t))
    return data
#target
def target63_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target63 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target63 * t).T))

            ]
def target63_evolution_data():
    data = []
    for t in ts:
        data.append(target63_evolution(t))
    return data
# fidelity results
fidelity63_tab = []
for i in range(len(ts)):
    fidelity63_tab.append(fidelity(target63_evolution_data()[i][0] , fittest63_evolution_data()[i][0]))
fidelity63_tab = np.round(fidelity63_tab , decimals = 6)
# coherence results
t63_coherence = coherence(target63_evolution_data())
f63_coherence = coherence(fittest63_evolution_data())
t63f63_coherence = []
for i in range(len(ts)):
     t63f63_coherence.append(np.abs(t63_coherence[i] - f63_coherence[i]))
# population results
pop63 = []
for i in range(len(ts)):
     pop63.append(np.sum(populations(target63_evolution_data() , fittest63_evolution_data())[i]))



##########################################
##########################################
#target individual
target64 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest64 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target64cc = nx.node_connected_component(nx.from_numpy_array(np.real(target64)) , 1)
fittest64cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest64)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr64 = target64 == fittest64
# Count the number of False values
false_count64 = np.sum(arr64 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t64f64_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target64)) , nx.from_numpy_array(np.real(fittest64)))
# WL graph isomorphism test
t64_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target64)))
f64_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest64)))
#measuring the similarity between two graphs through graph_edit_distance
t64f64_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target64)) , nx.from_numpy_array(np.real(fittest64)))
# degree distribution
t64_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target64))), key=lambda x: x[1], reverse=True)]
f64_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest64))), key=lambda x: x[1], reverse=True)]
# number of connections
t64_connections = np.sum(t64_dd)
f64_connections = np.sum(f64_dd)
# distance
distance64 = 0.04638580872022835
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest64_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest64 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest64 * t).T))

            ]
def fittest64_evolution_data():
    data = []
    for t in ts:
        data.append(fittest64_evolution(t))
    return data
#target
def target64_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target64 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target64 * t).T))

            ]
def target64_evolution_data():
    data = []
    for t in ts:
        data.append(target64_evolution(t))
    return data
# fidelity results
fidelity64_tab = []
for i in range(len(ts)):
    fidelity64_tab.append(fidelity(target64_evolution_data()[i][0] , fittest64_evolution_data()[i][0]))
fidelity64_tab = np.round(fidelity64_tab , decimals = 6)
# coherence results
t64_coherence = coherence(target64_evolution_data())
f64_coherence = coherence(fittest64_evolution_data())
t64f64_coherence = []
for i in range(len(ts)):
     t64f64_coherence.append(np.abs(t64_coherence[i] - f64_coherence[i]))
# population results
pop64 = []
for i in range(len(ts)):
     pop64.append(np.sum(populations(target64_evolution_data() , fittest64_evolution_data())[i]))


##########################################
##########################################
#target individual
target65 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest65 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target65cc = nx.node_connected_component(nx.from_numpy_array(np.real(target65)) , 1)
fittest65cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest65)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr65 = target65 == fittest65
# Count the number of False values
false_count65 = np.sum(arr65 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t65f65_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target65)) , nx.from_numpy_array(np.real(fittest65)))
# WL graph isomorphism test
t65_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target65)))
f65_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest65)))
#measuring the similarity between two graphs through graph_edit_distance
t65f65_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target65)) , nx.from_numpy_array(np.real(fittest65)))
# degree distribution
t65_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target65))), key=lambda x: x[1], reverse=True)]
f65_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest65))), key=lambda x: x[1], reverse=True)]
# number of connections
t65_connections = np.sum(t65_dd)
f65_connections = np.sum(f65_dd)
# distance
distance65 = 0.024870755958226853
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest65_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest65 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest65 * t).T))

            ]
def fittest65_evolution_data():
    data = []
    for t in ts:
        data.append(fittest65_evolution(t))
    return data
#target
def target65_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target65 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target65 * t).T))

            ]
def target65_evolution_data():
    data = []
    for t in ts:
        data.append(target65_evolution(t))
    return data
# fidelity results
fidelity65_tab = []
for i in range(len(ts)):
    fidelity65_tab.append(fidelity(target65_evolution_data()[i][0] , fittest65_evolution_data()[i][0]))
fidelity65_tab = np.round(fidelity65_tab , decimals = 6)
# coherence results
t65_coherence = coherence(target65_evolution_data())
f65_coherence = coherence(fittest65_evolution_data())
t65f65_coherence = []
for i in range(len(ts)):
     t65f65_coherence.append(np.abs(t65_coherence[i] - f65_coherence[i]))
# population results
pop65 = []
for i in range(len(ts)):
     pop65.append(np.sum(populations(target65_evolution_data() , fittest65_evolution_data())[i]))


##########################################
##########################################
#target individual
target66 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest66 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target66cc = nx.node_connected_component(nx.from_numpy_array(np.real(target66)) , 1)
fittest66cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest66)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr66 = target66 == fittest66
# Count the number of False values
false_count66 = np.sum(arr66 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t66f66_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target66)) , nx.from_numpy_array(np.real(fittest66)))
# WL graph isomorphism test
t66_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target66)))
f66_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest66)))
#measuring the similarity between two graphs through graph_edit_distance
t66f66_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target66)) , nx.from_numpy_array(np.real(fittest66)))
# degree distribution
t66_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target66))), key=lambda x: x[1], reverse=True)]
f66_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest66))), key=lambda x: x[1], reverse=True)]
# number of connections
t66_connections = np.sum(t66_dd)
f66_connections = np.sum(f66_dd)
# distance
distance66 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest66_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest66 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest66 * t).T))

            ]
def fittest66_evolution_data():
    data = []
    for t in ts:
        data.append(fittest66_evolution(t))
    return data
#target
def target66_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target66 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target66 * t).T))

            ]
def target66_evolution_data():
    data = []
    for t in ts:
        data.append(target66_evolution(t))
    return data
# fidelity results
fidelity66_tab = []
for i in range(len(ts)):
    fidelity66_tab.append(fidelity(target66_evolution_data()[i][0] , fittest66_evolution_data()[i][0]))
fidelity66_tab = np.round(fidelity66_tab , decimals = 6)
# coherence results
t66_coherence = coherence(target66_evolution_data())
f66_coherence = coherence(fittest66_evolution_data())
t66f66_coherence = []
for i in range(len(ts)):
     t66f66_coherence.append(np.abs(t66_coherence[i] - f66_coherence[i]))
# population results
pop66 = []
for i in range(len(ts)):
     pop66.append(np.sum(populations(target66_evolution_data() , fittest66_evolution_data())[i]))



##########################################
##########################################
#target individual
target67 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest67 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target67cc = nx.node_connected_component(nx.from_numpy_array(np.real(target67)) , 1)
fittest67cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest67)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr67 = target67 == fittest67
# Count the number of False values
false_count67 = np.sum(arr67 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t67f67_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target67)) , nx.from_numpy_array(np.real(fittest67)))
# WL graph isomorphism test
t67_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target67)))
f67_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest67)))
#measuring the similarity between two graphs through graph_edit_distance
t67f67_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target67)) , nx.from_numpy_array(np.real(fittest67)))
# degree distribution
t67_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target67))), key=lambda x: x[1], reverse=True)]
f67_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest67))), key=lambda x: x[1], reverse=True)]
# number of connections
t67_connections = np.sum(t67_dd)
f67_connections = np.sum(f67_dd)
# distance
distance67 = 0.3554050641502493
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest67_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest67 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest67 * t).T))

            ]
def fittest67_evolution_data():
    data = []
    for t in ts:
        data.append(fittest67_evolution(t))
    return data
#target
def target67_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target67 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target67 * t).T))

            ]
def target67_evolution_data():
    data = []
    for t in ts:
        data.append(target67_evolution(t))
    return data
# fidelity results
fidelity67_tab = []
for i in range(len(ts)):
    fidelity67_tab.append(fidelity(target67_evolution_data()[i][0] , fittest67_evolution_data()[i][0]))
fidelity67_tab = np.round(fidelity67_tab , decimals = 6)
# coherence results
t67_coherence = coherence(target67_evolution_data())
f67_coherence = coherence(fittest67_evolution_data())
t67f67_coherence = []
for i in range(len(ts)):
     t67f67_coherence.append(np.abs(t67_coherence[i] - f67_coherence[i]))
# population results
pop67 = []
for i in range(len(ts)):
     pop67.append(np.sum(populations(target67_evolution_data() , fittest67_evolution_data())[i]))




##########################################
##########################################
#target individual
target68 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest68 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target68cc = nx.node_connected_component(nx.from_numpy_array(np.real(target68)) , 1)
fittest68cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest68)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr68 = target68 == fittest68
# Count the number of False values
false_count68 = np.sum(arr68 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t68f68_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target68)) , nx.from_numpy_array(np.real(fittest68)))
# WL graph isomorphism test
t68_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target68)))
f68_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest68)))
#measuring the similarity between two graphs through graph_edit_distance
t68f68_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target68)) , nx.from_numpy_array(np.real(fittest68)))
# degree distribution
t68_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target68))), key=lambda x: x[1], reverse=True)]
f68_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest68))), key=lambda x: x[1], reverse=True)]
# number of connections
t68_connections = np.sum(t68_dd)
f68_connections = np.sum(f68_dd)
# distance
distance68 = 0.056127401274446265
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest68_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest68 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest68 * t).T))

            ]
def fittest68_evolution_data():
    data = []
    for t in ts:
        data.append(fittest68_evolution(t))
    return data
#target
def target68_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target68 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target68 * t).T))

            ]
def target68_evolution_data():
    data = []
    for t in ts:
        data.append(target68_evolution(t))
    return data
# fidelity results
fidelity68_tab = []
for i in range(len(ts)):
    fidelity68_tab.append(fidelity(target68_evolution_data()[i][0] , fittest68_evolution_data()[i][0]))
fidelity68_tab = np.round(fidelity68_tab , decimals = 6)
# coherence results
t68_coherence = coherence(target68_evolution_data())
f68_coherence = coherence(fittest68_evolution_data())
t68f68_coherence = []
for i in range(len(ts)):
     t68f68_coherence.append(np.abs(t68_coherence[i] - f68_coherence[i]))
# population results
pop68 = []
for i in range(len(ts)):
     pop68.append(np.sum(populations(target68_evolution_data() , fittest68_evolution_data())[i]))



##########################################
##########################################
#target individual
target69 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest69 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target69cc = nx.node_connected_component(nx.from_numpy_array(np.real(target69)) , 1)
fittest69cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest69)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr69 = target69 == fittest69
# Count the number of False values
false_count69 = np.sum(arr69 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t69f69_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target69)) , nx.from_numpy_array(np.real(fittest69)))
# WL graph isomorphism test
t69_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target69)))
f69_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest69)))
#measuring the similarity between two graphs through graph_edit_distance
t69f69_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target69)) , nx.from_numpy_array(np.real(fittest69)))
# degree distribution
t69_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target69))), key=lambda x: x[1], reverse=True)]
f69_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest69))), key=lambda x: x[1], reverse=True)]
# number of connections
t69_connections = np.sum(t69_dd)
f69_connections = np.sum(f69_dd)
# distance
distance69 = 0.31431632160212686
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest69_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest69 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest69 * t).T))

            ]
def fittest69_evolution_data():
    data = []
    for t in ts:
        data.append(fittest69_evolution(t))
    return data
#target
def target69_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target69 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target69 * t).T))

            ]
def target69_evolution_data():
    data = []
    for t in ts:
        data.append(target69_evolution(t))
    return data
# fidelity results
fidelity69_tab = []
for i in range(len(ts)):
    fidelity69_tab.append(fidelity(target69_evolution_data()[i][0] , fittest69_evolution_data()[i][0]))
fidelity69_tab = np.round(fidelity69_tab , decimals = 6)
# coherence results
t69_coherence = coherence(target69_evolution_data())
f69_coherence = coherence(fittest69_evolution_data())
t69f69_coherence = []
for i in range(len(ts)):
     t69f69_coherence.append(np.abs(t69_coherence[i] - f69_coherence[i]))
# population results
pop69 = []
for i in range(len(ts)):
     pop69.append(np.sum(populations(target69_evolution_data() , fittest69_evolution_data())[i]))












##########################################
##########################################
#target individual
target70 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest70 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target70cc = nx.node_connected_component(nx.from_numpy_array(np.real(target70)) , 1)
fittest70cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest70)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr70 = target70 == fittest70
# Count the number of False values
false_count70 = np.sum(arr70 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t70f70_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target70)) , nx.from_numpy_array(np.real(fittest70)))
# WL graph isomorphism test
t70_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target70)))
f70_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest70)))
# measuring the similarity between two graphs through graph_edit_distance
t70f70_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target70)) , nx.from_numpy_array(np.real(fittest70)))
# degree distribution
t70_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target70))), key=lambda x: x[1], reverse=True)]
f70_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest70))), key=lambda x: x[1], reverse=True)]
# number of connections
t70_connections = np.sum(t70_dd)
f70_connections = np.sum(f70_dd)
# distance
distance70 = 0.028757068699452715
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest70_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest70 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest70 * t).T))

            ]
def fittest70_evolution_data():
    data = []
    for t in ts:
        data.append(fittest70_evolution(t))
    return data
#target
def target70_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target70 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target70 * t).T))

            ]
def target70_evolution_data():
    data = []
    for t in ts:
        data.append(target70_evolution(t))
    return data
# fidelity results
fidelity70_tab = []
for i in range(len(ts)):
    fidelity70_tab.append(fidelity(target70_evolution_data()[i][0] , fittest70_evolution_data()[i][0]))
fidelity70_tab = np.round(fidelity70_tab , decimals = 6)
# coherence results
t70_coherence = coherence(target70_evolution_data())
f70_coherence = coherence(fittest70_evolution_data())
t70f70_coherence = []
for i in range(len(ts)):
     t70f70_coherence.append(np.abs(t70_coherence[i] - f70_coherence[i]))
# population results
pop70 = []
for i in range(len(ts)):
     pop70.append(np.sum(populations(target70_evolution_data() , fittest70_evolution_data())[i]))


##########################################
##########################################
#target individual
target71 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest71 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target71cc = nx.node_connected_component(nx.from_numpy_array(np.real(target71)) , 1)
fittest71cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest71)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr71 = target71 == fittest71
# Count the number of False values
false_count71 = np.sum(arr71 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t71f71_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target71)) , nx.from_numpy_array(np.real(fittest71)))
# WL graph isomorphism test
t71_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target71)))
f71_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest71)))
# measuring the similarity between two graphs through graph_edit_distance
t71f71_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target71)) , nx.from_numpy_array(np.real(fittest71)))
# degree distribution
t71_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target71))), key=lambda x: x[1], reverse=True)]
f71_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest71))), key=lambda x: x[1], reverse=True)]
# number of connections
t71_connections = np.sum(t71_dd)
f71_connections = np.sum(f71_dd)
# distance
distance71 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest71_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest71 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest71 * t).T))

            ]
def fittest71_evolution_data():
    data = []
    for t in ts:
        data.append(fittest71_evolution(t))
    return data
#target
def target71_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target71 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target71 * t).T))

            ]
def target71_evolution_data():
    data = []
    for t in ts:
        data.append(target71_evolution(t))
    return data
# fidelity results
fidelity71_tab = []
for i in range(len(ts)):
    fidelity71_tab.append(fidelity(target71_evolution_data()[i][0] , fittest71_evolution_data()[i][0]))
fidelity71_tab = np.round(fidelity71_tab , decimals = 6)
# coherence results
t71_coherence = coherence(target71_evolution_data())
f71_coherence = coherence(fittest71_evolution_data())
t71f71_coherence = []
for i in range(len(ts)):
     t71f71_coherence.append(np.abs(t71_coherence[i] - f71_coherence[i]))
# population results
pop71 = []
for i in range(len(ts)):
     pop71.append(np.sum(populations(target71_evolution_data() , fittest71_evolution_data())[i]))


##########################################
##########################################
#target individual
target72 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest72 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target72cc = nx.node_connected_component(nx.from_numpy_array(np.real(target72)) , 1)
fittest72cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest72)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr72 = target72 == fittest72
# Count the number of False values
false_count72 = np.sum(arr72 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t72f72_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target72)) , nx.from_numpy_array(np.real(fittest72)))
# WL graph isomorphism test
t72_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target72)))
f72_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest72)))
# measuring the similarity between two graphs through graph_edit_distance
t72f72_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target72)) , nx.from_numpy_array(np.real(fittest72)))
# degree distribution
t72_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target72))), key=lambda x: x[1], reverse=True)]
f72_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest72))), key=lambda x: x[1], reverse=True)]
# number of connections
t72_connections = np.sum(t72_dd)
f72_connections = np.sum(f72_dd)
# distance
distance72 = 0.11098007466586213
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest72_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest72 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest72 * t).T))

            ]
def fittest72_evolution_data():
    data = []
    for t in ts:
        data.append(fittest72_evolution(t))
    return data
#target
def target72_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target72 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target72 * t).T))

            ]
def target72_evolution_data():
    data = []
    for t in ts:
        data.append(target72_evolution(t))
    return data
# fidelity results
fidelity72_tab = []
for i in range(len(ts)):
    fidelity72_tab.append(fidelity(target72_evolution_data()[i][0] , fittest72_evolution_data()[i][0]))
fidelity72_tab = np.round(fidelity72_tab , decimals = 6)
# coherence results
t72_coherence = coherence(target72_evolution_data())
f72_coherence = coherence(fittest72_evolution_data())
t72f72_coherence = []
for i in range(len(ts)):
     t72f72_coherence.append(np.abs(t72_coherence[i] - f72_coherence[i]))
# population results
pop72 = []
for i in range(len(ts)):
     pop72.append(np.sum(populations(target72_evolution_data() , fittest72_evolution_data())[i]))



##########################################
##########################################
#target individual
target73 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest73 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target73cc = nx.node_connected_component(nx.from_numpy_array(np.real(target73)) , 1)
fittest73cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest73)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr73 = target73 == fittest73
# Count the number of False values
false_count73 = np.sum(arr73 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t73f73_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target73)) , nx.from_numpy_array(np.real(fittest73)))
# WL graph isomorphism test
t73_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target73)))
f73_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest73)))
#measuring the similarity between two graphs through graph_edit_distance
t73f73_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target73)) , nx.from_numpy_array(np.real(fittest73)))
# degree distribution
t73_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target73))), key=lambda x: x[1], reverse=True)]
f73_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest73))), key=lambda x: x[1], reverse=True)]
# number of connections
t73_connections = np.sum(t73_dd)
f73_connections = np.sum(f73_dd)
# distance
distance73 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest73_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest73 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest73 * t).T))

            ]
def fittest73_evolution_data():
    data = []
    for t in ts:
        data.append(fittest73_evolution(t))
    return data
#target
def target73_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target73 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target73 * t).T))

            ]
def target73_evolution_data():
    data = []
    for t in ts:
        data.append(target73_evolution(t))
    return data
# fidelity results
fidelity73_tab = []
for i in range(len(ts)):
    fidelity73_tab.append(fidelity(target73_evolution_data()[i][0] , fittest73_evolution_data()[i][0]))
fidelity73_tab = np.round(fidelity73_tab , decimals = 6)
# coherence results
t73_coherence = coherence(target73_evolution_data())
f73_coherence = coherence(fittest73_evolution_data())
t73f73_coherence = []
for i in range(len(ts)):
     t73f73_coherence.append(np.abs(t73_coherence[i] - f73_coherence[i]))
# population results
pop73 = []
for i in range(len(ts)):
     pop73.append(np.sum(populations(target73_evolution_data() , fittest73_evolution_data())[i]))



##########################################
##########################################
#target individual
target74 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest74 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target74cc = nx.node_connected_component(nx.from_numpy_array(np.real(target74)) , 1)
fittest74cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest74)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr74 = target74 == fittest74
# Count the number of False values
false_count74 = np.sum(arr74 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t74f74_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target74)) , nx.from_numpy_array(np.real(fittest74)))
# WL graph isomorphism test
t74_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target74)))
f74_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest74)))
#measuring the similarity between two graphs through graph_edit_distance
t74f74_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target74)) , nx.from_numpy_array(np.real(fittest74)))
# degree distribution
t74_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target74))), key=lambda x: x[1], reverse=True)]
f74_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest74))), key=lambda x: x[1], reverse=True)]
# number of connections
t74_connections = np.sum(t74_dd)
f74_connections = np.sum(f74_dd)
# distance
distance74 = 0.044900229181278895
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest74_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest74 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest74 * t).T))

            ]
def fittest74_evolution_data():
    data = []
    for t in ts:
        data.append(fittest74_evolution(t))
    return data
#target
def target74_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target74 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target74 * t).T))

            ]
def target74_evolution_data():
    data = []
    for t in ts:
        data.append(target74_evolution(t))
    return data
# fidelity results
fidelity74_tab = []
for i in range(len(ts)):
    fidelity74_tab.append(fidelity(target74_evolution_data()[i][0] , fittest74_evolution_data()[i][0]))
fidelity74_tab = np.round(fidelity74_tab , decimals = 6)
# coherence results
t74_coherence = coherence(target74_evolution_data())
f74_coherence = coherence(fittest74_evolution_data())
t74f74_coherence = []
for i in range(len(ts)):
     t74f74_coherence.append(np.abs(t74_coherence[i] - f74_coherence[i]))
# population results
pop74 = []
for i in range(len(ts)):
     pop74.append(np.sum(populations(target74_evolution_data() , fittest74_evolution_data())[i]))


##########################################
##########################################
#target individual
target75 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest75 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target75cc = nx.node_connected_component(nx.from_numpy_array(np.real(target75)) , 1)
fittest75cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest75)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr75 = target75 == fittest75
# Count the number of False values
false_count75 = np.sum(arr75 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t75f75_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target75)) , nx.from_numpy_array(np.real(fittest75)))
# WL graph isomorphism test
t75_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target75)))
f75_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest75)))
#measuring the similarity between two graphs through graph_edit_distance
t75f75_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target75)) , nx.from_numpy_array(np.real(fittest75)))
# degree distribution
t75_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target75))), key=lambda x: x[1], reverse=True)]
f75_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest75))), key=lambda x: x[1], reverse=True)]
# number of connections
t75_connections = np.sum(t75_dd)
f75_connections = np.sum(f75_dd)
# distance
distance75 = 0.14356443712705003
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest75_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest75 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest75 * t).T))

            ]
def fittest75_evolution_data():
    data = []
    for t in ts:
        data.append(fittest75_evolution(t))
    return data
#target
def target75_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target75 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target75 * t).T))

            ]
def target75_evolution_data():
    data = []
    for t in ts:
        data.append(target75_evolution(t))
    return data
# fidelity results
fidelity75_tab = []
for i in range(len(ts)):
    fidelity75_tab.append(fidelity(target75_evolution_data()[i][0] , fittest75_evolution_data()[i][0]))
fidelity75_tab = np.round(fidelity75_tab , decimals = 6)
# coherence results
t75_coherence = coherence(target75_evolution_data())
f75_coherence = coherence(fittest75_evolution_data())
t75f75_coherence = []
for i in range(len(ts)):
     t75f75_coherence.append(np.abs(t75_coherence[i] - f75_coherence[i]))
# population results
pop75 = []
for i in range(len(ts)):
     pop75.append(np.sum(populations(target75_evolution_data() , fittest75_evolution_data())[i]))


##########################################
##########################################
#target individual
target76 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest76 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target76cc = nx.node_connected_component(nx.from_numpy_array(np.real(target76)) , 1)
fittest76cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest76)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr76 = target76 == fittest76
# Count the number of False values
false_count76 = np.sum(arr76 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t76f76_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target76)) , nx.from_numpy_array(np.real(fittest76)))
# WL graph isomorphism test
t76_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target76)))
f76_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest76)))
#measuring the similarity between two graphs through graph_edit_distance
t76f76_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target76)) , nx.from_numpy_array(np.real(fittest76)))
# degree distribution
t76_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target76))), key=lambda x: x[1], reverse=True)]
f76_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest76))), key=lambda x: x[1], reverse=True)]
# number of connections
t76_connections = np.sum(t76_dd)
f76_connections = np.sum(f76_dd)
# distance
distance76 = 0.14403619612984664
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest76_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest76 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest76 * t).T))

            ]
def fittest76_evolution_data():
    data = []
    for t in ts:
        data.append(fittest76_evolution(t))
    return data
#target
def target76_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target76 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target76 * t).T))

            ]
def target76_evolution_data():
    data = []
    for t in ts:
        data.append(target76_evolution(t))
    return data
# fidelity results
fidelity76_tab = []
for i in range(len(ts)):
    fidelity76_tab.append(fidelity(target76_evolution_data()[i][0] , fittest76_evolution_data()[i][0]))
fidelity76_tab = np.round(fidelity76_tab , decimals = 6)
# coherence results
t76_coherence = coherence(target76_evolution_data())
f76_coherence = coherence(fittest76_evolution_data())
t76f76_coherence = []
for i in range(len(ts)):
     t76f76_coherence.append(np.abs(t76_coherence[i] - f76_coherence[i]))
# population results
pop76 = []
for i in range(len(ts)):
     pop76.append(np.sum(populations(target76_evolution_data() , fittest76_evolution_data())[i]))



##########################################
##########################################
#target individual
target77 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest77 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target77cc = nx.node_connected_component(nx.from_numpy_array(np.real(target77)) , 1)
fittest77cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest77)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr77 = target77 == fittest77
# Count the number of False values
false_count77 = np.sum(arr77 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t77f77_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target77)) , nx.from_numpy_array(np.real(fittest77)))
# WL graph isomorphism test
t77_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target77)))
f77_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest77)))
#measuring the similarity between two graphs through graph_edit_distance
t77f77_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target77)) , nx.from_numpy_array(np.real(fittest77)))
# degree distribution
t77_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target77))), key=lambda x: x[1], reverse=True)]
f77_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest77))), key=lambda x: x[1], reverse=True)]
# number of connections
t77_connections = np.sum(t77_dd)
f77_connections = np.sum(f77_dd)
# distance
distance77 = 8.881784197001252e-15
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest77_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest77 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest77 * t).T))

            ]
def fittest77_evolution_data():
    data = []
    for t in ts:
        data.append(fittest77_evolution(t))
    return data
#target
def target77_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target77 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target77 * t).T))

            ]
def target77_evolution_data():
    data = []
    for t in ts:
        data.append(target77_evolution(t))
    return data
# fidelity results
fidelity77_tab = []
for i in range(len(ts)):
    fidelity77_tab.append(fidelity(target77_evolution_data()[i][0] , fittest77_evolution_data()[i][0]))
fidelity77_tab = np.round(fidelity77_tab , decimals = 6)
# coherence results
t77_coherence = coherence(target77_evolution_data())
f77_coherence = coherence(fittest77_evolution_data())
t77f77_coherence = []
for i in range(len(ts)):
     t77f77_coherence.append(np.abs(t77_coherence[i] - f77_coherence[i]))
# population results
pop77 = []
for i in range(len(ts)):
     pop77.append(np.sum(populations(target77_evolution_data() , fittest77_evolution_data())[i]))




##########################################
##########################################
#target individual
target78 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest78 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target78cc = nx.node_connected_component(nx.from_numpy_array(np.real(target78)) , 1)
fittest78cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest78)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr78 = target78 == fittest78
# Count the number of False values
false_count78 = np.sum(arr78 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t78f78_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target78)) , nx.from_numpy_array(np.real(fittest78)))
# WL graph isomorphism test
t78_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target78)))
f78_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest78)))
#measuring the similarity between two graphs through graph_edit_distance
t78f78_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target78)) , nx.from_numpy_array(np.real(fittest78)))
# degree distribution
t78_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target78))), key=lambda x: x[1], reverse=True)]
f78_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest78))), key=lambda x: x[1], reverse=True)]
# number of connections
t78_connections = np.sum(t78_dd)
f78_connections = np.sum(f78_dd)
# distance
distance78 = 0.20388877428734142
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest78_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest78 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest78 * t).T))

            ]
def fittest78_evolution_data():
    data = []
    for t in ts:
        data.append(fittest78_evolution(t))
    return data
#target
def target78_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target78 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target78 * t).T))

            ]
def target78_evolution_data():
    data = []
    for t in ts:
        data.append(target78_evolution(t))
    return data
# fidelity results
fidelity78_tab = []
for i in range(len(ts)):
    fidelity78_tab.append(fidelity(target78_evolution_data()[i][0] , fittest78_evolution_data()[i][0]))
fidelity78_tab = np.round(fidelity78_tab , decimals = 6)
# coherence results
t78_coherence = coherence(target78_evolution_data())
f78_coherence = coherence(fittest78_evolution_data())
t78f78_coherence = []
for i in range(len(ts)):
     t78f78_coherence.append(np.abs(t78_coherence[i] - f78_coherence[i]))
# population results
pop78 = []
for i in range(len(ts)):
     pop78.append(np.sum(populations(target78_evolution_data() , fittest78_evolution_data())[i]))



##########################################
##########################################
#target individual
target79 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest79 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target79cc = nx.node_connected_component(nx.from_numpy_array(np.real(target79)) , 1)
fittest79cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest79)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr79 = target79 == fittest79
# Count the number of False values
false_count79 = np.sum(arr79 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t79f79_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target79)) , nx.from_numpy_array(np.real(fittest79)))
# WL graph isomorphism test
t79_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target79)))
f79_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest79)))
#measuring the similarity between two graphs through graph_edit_distance
t79f79_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target79)) , nx.from_numpy_array(np.real(fittest79)))
# degree distribution
t79_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target79))), key=lambda x: x[1], reverse=True)]
f79_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest79))), key=lambda x: x[1], reverse=True)]
# number of connections
t79_connections = np.sum(t79_dd)
f79_connections = np.sum(f79_dd)
# distance
distance79 = 0.0013830426750687241
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest79_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest79 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest79 * t).T))

            ]
def fittest79_evolution_data():
    data = []
    for t in ts:
        data.append(fittest79_evolution(t))
    return data
#target
def target79_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target79 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target79 * t).T))

            ]
def target79_evolution_data():
    data = []
    for t in ts:
        data.append(target79_evolution(t))
    return data
# fidelity results
fidelity79_tab = []
for i in range(len(ts)):
    fidelity79_tab.append(fidelity(target79_evolution_data()[i][0] , fittest79_evolution_data()[i][0]))
fidelity79_tab = np.round(fidelity79_tab , decimals = 6)
# coherence results
t79_coherence = coherence(target79_evolution_data())
f79_coherence = coherence(fittest79_evolution_data())
t79f79_coherence = []
for i in range(len(ts)):
     t79f79_coherence.append(np.abs(t79_coherence[i] - f79_coherence[i]))
# population results
pop79 = []
for i in range(len(ts)):
     pop79.append(np.sum(populations(target79_evolution_data() , fittest79_evolution_data())[i]))


##########################################
##########################################
#target individual
target80 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest80 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target80cc = nx.node_connected_component(nx.from_numpy_array(np.real(target80)) , 1)
fittest80cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest80)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr80 = target80 == fittest80
# Count the number of False values
false_count80 = np.sum(arr80 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t80f80_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target80)) , nx.from_numpy_array(np.real(fittest80)))
# WL graph isomorphism test
t80_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target80)))
f80_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest80)))
# measuring the similarity between two graphs through graph_edit_distance
t80f80_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target80)) , nx.from_numpy_array(np.real(fittest80)))
# degree distribution
t80_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target80))), key=lambda x: x[1], reverse=True)]
f80_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest80))), key=lambda x: x[1], reverse=True)]
# number of connections
t80_connections = np.sum(t80_dd)
f80_connections = np.sum(f80_dd)
# distance
distance80 = 0.18351455800611283
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest80_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest80 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest80 * t).T))

            ]
def fittest80_evolution_data():
    data = []
    for t in ts:
        data.append(fittest80_evolution(t))
    return data
#target
def target80_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target80 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target80 * t).T))

            ]
def target80_evolution_data():
    data = []
    for t in ts:
        data.append(target80_evolution(t))
    return data
# fidelity results
fidelity80_tab = []
for i in range(len(ts)):
    fidelity80_tab.append(fidelity(target80_evolution_data()[i][0] , fittest80_evolution_data()[i][0]))
fidelity80_tab = np.round(fidelity80_tab , decimals = 6)
# coherence results
t80_coherence = coherence(target80_evolution_data())
f80_coherence = coherence(fittest80_evolution_data())
t80f80_coherence = []
for i in range(len(ts)):
     t80f80_coherence.append(np.abs(t80_coherence[i] - f80_coherence[i]))
# population results
pop80 = []
for i in range(len(ts)):
     pop80.append(np.sum(populations(target80_evolution_data() , fittest80_evolution_data())[i]))

##########################################
##########################################
#target individual
target81 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest81 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target81cc = nx.node_connected_component(nx.from_numpy_array(np.real(target81)) , 1)
fittest81cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest81)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr81 = target81 == fittest81
# Count the number of False values
false_count81 = np.sum(arr81 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t81f81_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target81)) , nx.from_numpy_array(np.real(fittest81)))
# WL graph isomorphism test
t81_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target81)))
f81_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest81)))
# measuring the similarity between two graphs through graph_edit_distance
t81f81_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target81)) , nx.from_numpy_array(np.real(fittest81)))
# degree distribution
t81_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target81))), key=lambda x: x[1], reverse=True)]
f81_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest81))), key=lambda x: x[1], reverse=True)]
# number of connections
t81_connections = np.sum(t81_dd)
f81_connections = np.sum(f81_dd)
# distance
distance81 = 0.11238423338147341
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest81_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest81 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest81 * t).T))

            ]
def fittest81_evolution_data():
    data = []
    for t in ts:
        data.append(fittest81_evolution(t))
    return data
#target
def target81_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target81 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target81 * t).T))

            ]
def target81_evolution_data():
    data = []
    for t in ts:
        data.append(target81_evolution(t))
    return data
# fidelity results
fidelity81_tab = []
for i in range(len(ts)):
    fidelity81_tab.append(fidelity(target81_evolution_data()[i][0] , fittest81_evolution_data()[i][0]))
fidelity81_tab = np.round(fidelity81_tab , decimals = 6)
# coherence results
t81_coherence = coherence(target81_evolution_data())
f81_coherence = coherence(fittest81_evolution_data())
t81f81_coherence = []
for i in range(len(ts)):
     t81f81_coherence.append(np.abs(t81_coherence[i] - f81_coherence[i]))
# population results
pop81 = []
for i in range(len(ts)):
     pop81.append(np.sum(populations(target81_evolution_data() , fittest81_evolution_data())[i]))


##########################################
##########################################
#target individual
target82 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest82 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target82cc = nx.node_connected_component(nx.from_numpy_array(np.real(target82)) , 1)
fittest82cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest82)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr82 = target82 == fittest82
# Count the number of False values
false_count82 = np.sum(arr82 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t82f82_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target82)) , nx.from_numpy_array(np.real(fittest82)))
# WL graph isomorphism test
t82_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target82)))
f82_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest82)))
# measuring the similarity between two graphs through graph_edit_distance
t82f82_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target82)) , nx.from_numpy_array(np.real(fittest82)))
# degree distribution
t82_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target82))), key=lambda x: x[1], reverse=True)]
f82_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest82))), key=lambda x: x[1], reverse=True)]
# number of connections
t82_connections = np.sum(t82_dd)
f82_connections = np.sum(f82_dd)
# distance
distance82 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest82_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest82 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest82 * t).T))

            ]
def fittest82_evolution_data():
    data = []
    for t in ts:
        data.append(fittest82_evolution(t))
    return data
#target
def target82_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target82 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target82 * t).T))

            ]
def target82_evolution_data():
    data = []
    for t in ts:
        data.append(target82_evolution(t))
    return data
# fidelity results
fidelity82_tab = []
for i in range(len(ts)):
    fidelity82_tab.append(fidelity(target82_evolution_data()[i][0] , fittest82_evolution_data()[i][0]))
fidelity82_tab = np.round(fidelity82_tab , decimals = 6)
# coherence results
t82_coherence = coherence(target82_evolution_data())
f82_coherence = coherence(fittest82_evolution_data())
t82f82_coherence = []
for i in range(len(ts)):
     t82f82_coherence.append(np.abs(t82_coherence[i] - f82_coherence[i]))
# population results
pop82 = []
for i in range(len(ts)):
     pop82.append(np.sum(populations(target82_evolution_data() , fittest82_evolution_data())[i]))



##########################################
##########################################
#target individual
target83 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest83 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target83cc = nx.node_connected_component(nx.from_numpy_array(np.real(target83)) , 1)
fittest83cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest83)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr83 = target83 == fittest83
# Count the number of False values
false_count83 = np.sum(arr83 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t83f83_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target83)) , nx.from_numpy_array(np.real(fittest83)))
# WL graph isomorphism test
t83_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target83)))
f83_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest83)))
#measuring the similarity between two graphs through graph_edit_distance
t83f83_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target83)) , nx.from_numpy_array(np.real(fittest83)))
# degree distribution
t83_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target83))), key=lambda x: x[1], reverse=True)]
f83_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest83))), key=lambda x: x[1], reverse=True)]
# number of connections
t83_connections = np.sum(t83_dd)
f83_connections = np.sum(f83_dd)
# distance
distance83 = 0.08602798474927231
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest83_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest83 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest83 * t).T))

            ]
def fittest83_evolution_data():
    data = []
    for t in ts:
        data.append(fittest83_evolution(t))
    return data
#target
def target83_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target83 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target83 * t).T))

            ]
def target83_evolution_data():
    data = []
    for t in ts:
        data.append(target83_evolution(t))
    return data
# fidelity results
fidelity83_tab = []
for i in range(len(ts)):
    fidelity83_tab.append(fidelity(target83_evolution_data()[i][0] , fittest83_evolution_data()[i][0]))
fidelity83_tab = np.round(fidelity83_tab , decimals = 6)
# coherence results
t83_coherence = coherence(target83_evolution_data())
f83_coherence = coherence(fittest83_evolution_data())
t83f83_coherence = []
for i in range(len(ts)):
     t83f83_coherence.append(np.abs(t83_coherence[i] - f83_coherence[i]))
# population results
pop83 = []
for i in range(len(ts)):
     pop83.append(np.sum(populations(target83_evolution_data() , fittest83_evolution_data())[i]))



##########################################
##########################################
#target individual
target84 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest84 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target84cc = nx.node_connected_component(nx.from_numpy_array(np.real(target84)) , 1)
fittest84cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest84)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr84 = target84 == fittest84
# Count the number of False values
false_count84 = np.sum(arr84 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t84f84_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target84)) , nx.from_numpy_array(np.real(fittest84)))
# WL graph isomorphism test
t84_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target84)))
f84_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest84)))
#measuring the similarity between two graphs through graph_edit_distance
t84f84_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target84)) , nx.from_numpy_array(np.real(fittest84)))
# degree distribution
t84_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target84))), key=lambda x: x[1], reverse=True)]
f84_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest84))), key=lambda x: x[1], reverse=True)]
# number of connections
t84_connections = np.sum(t84_dd)
f84_connections = np.sum(f84_dd)
# distance
distance84 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest84_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest84 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest84 * t).T))

            ]
def fittest84_evolution_data():
    data = []
    for t in ts:
        data.append(fittest84_evolution(t))
    return data
#target
def target84_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target84 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target84 * t).T))

            ]
def target84_evolution_data():
    data = []
    for t in ts:
        data.append(target84_evolution(t))
    return data
# fidelity results
fidelity84_tab = []
for i in range(len(ts)):
    fidelity84_tab.append(fidelity(target84_evolution_data()[i][0] , fittest84_evolution_data()[i][0]))
fidelity84_tab = np.round(fidelity84_tab , decimals = 6)
# coherence results
t84_coherence = coherence(target84_evolution_data())
f84_coherence = coherence(fittest84_evolution_data())
t84f84_coherence = []
for i in range(len(ts)):
     t84f84_coherence.append(np.abs(t84_coherence[i] - f84_coherence[i]))
# population results
pop84 = []
for i in range(len(ts)):
     pop84.append(np.sum(populations(target84_evolution_data() , fittest84_evolution_data())[i]))


##########################################
##########################################
#target individual
target85 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest85 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target85cc = nx.node_connected_component(nx.from_numpy_array(np.real(target85)) , 1)
fittest85cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest85)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr85 = target85 == fittest85
# Count the number of False values
false_count85 = np.sum(arr85 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t85f85_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target85)) , nx.from_numpy_array(np.real(fittest85)))
# WL graph isomorphism test
t85_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target85)))
f85_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest85)))
#measuring the similarity between two graphs through graph_edit_distance
t85f85_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target85)) , nx.from_numpy_array(np.real(fittest85)))
# degree distribution
t85_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target85))), key=lambda x: x[1], reverse=True)]
f85_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest85))), key=lambda x: x[1], reverse=True)]
# number of connections
t85_connections = np.sum(t85_dd)
f85_connections = np.sum(f85_dd)
# distance
distance85 = 0.1284277890412524
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest85_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest85 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest85 * t).T))

            ]
def fittest85_evolution_data():
    data = []
    for t in ts:
        data.append(fittest85_evolution(t))
    return data
#target
def target85_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target85 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target85 * t).T))

            ]
def target85_evolution_data():
    data = []
    for t in ts:
        data.append(target85_evolution(t))
    return data
# fidelity results
fidelity85_tab = []
for i in range(len(ts)):
    fidelity85_tab.append(fidelity(target85_evolution_data()[i][0] , fittest85_evolution_data()[i][0]))
fidelity85_tab = np.round(fidelity85_tab , decimals = 6)
# coherence results
t85_coherence = coherence(target85_evolution_data())
f85_coherence = coherence(fittest85_evolution_data())
t85f85_coherence = []
for i in range(len(ts)):
     t85f85_coherence.append(np.abs(t85_coherence[i] - f85_coherence[i]))
# population results
pop85 = []
for i in range(len(ts)):
     pop85.append(np.sum(populations(target85_evolution_data() , fittest85_evolution_data())[i]))


##########################################
##########################################
#target individual
target86 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest86 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target86cc = nx.node_connected_component(nx.from_numpy_array(np.real(target86)) , 1)
fittest86cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest86)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr86 = target86 == fittest86
# Count the number of False values
false_count86 = np.sum(arr86 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t86f86_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target86)) , nx.from_numpy_array(np.real(fittest86)))
# WL graph isomorphism test
t86_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target86)))
f86_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest86)))
#measuring the similarity between two graphs through graph_edit_distance
t86f86_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target86)) , nx.from_numpy_array(np.real(fittest86)))
# degree distribution
t86_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target86))), key=lambda x: x[1], reverse=True)]
f86_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest86))), key=lambda x: x[1], reverse=True)]
# number of connections
t86_connections = np.sum(t86_dd)
f86_connections = np.sum(f86_dd)
# distance
distance86 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest86_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest86 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest86 * t).T))

            ]
def fittest86_evolution_data():
    data = []
    for t in ts:
        data.append(fittest86_evolution(t))
    return data
#target
def target86_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target86 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target86 * t).T))

            ]
def target86_evolution_data():
    data = []
    for t in ts:
        data.append(target86_evolution(t))
    return data
# fidelity results
fidelity86_tab = []
for i in range(len(ts)):
    fidelity86_tab.append(fidelity(target86_evolution_data()[i][0] , fittest86_evolution_data()[i][0]))
fidelity86_tab = np.round(fidelity86_tab , decimals = 6)
# coherence results
t86_coherence = coherence(target86_evolution_data())
f86_coherence = coherence(fittest86_evolution_data())
t86f86_coherence = []
for i in range(len(ts)):
     t86f86_coherence.append(np.abs(t86_coherence[i] - f86_coherence[i]))
# population results
pop86 = []
for i in range(len(ts)):
     pop86.append(np.sum(populations(target86_evolution_data() , fittest86_evolution_data())[i]))



##########################################
##########################################
#target individual
target87 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest87 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target87cc = nx.node_connected_component(nx.from_numpy_array(np.real(target87)) , 1)
fittest87cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest87)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr87 = target87 == fittest87
# Count the number of False values
false_count87 = np.sum(arr87 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t87f87_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target87)) , nx.from_numpy_array(np.real(fittest87)))
# WL graph isomorphism test
t87_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target87)))
f87_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest87)))
#measuring the similarity between two graphs through graph_edit_distance
t87f87_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target87)) , nx.from_numpy_array(np.real(fittest87)))
# degree distribution
t87_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target87))), key=lambda x: x[1], reverse=True)]
f87_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest87))), key=lambda x: x[1], reverse=True)]
# number of connections
t87_connections = np.sum(t87_dd)
f87_connections = np.sum(f87_dd)
# distance
distance87 = 0.20758632094470453
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest87_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest87 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest87 * t).T))

            ]
def fittest87_evolution_data():
    data = []
    for t in ts:
        data.append(fittest87_evolution(t))
    return data
#target
def target87_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target87 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target87 * t).T))

            ]
def target87_evolution_data():
    data = []
    for t in ts:
        data.append(target87_evolution(t))
    return data
# fidelity results
fidelity87_tab = []
for i in range(len(ts)):
    fidelity87_tab.append(fidelity(target87_evolution_data()[i][0] , fittest87_evolution_data()[i][0]))
fidelity87_tab = np.round(fidelity87_tab , decimals = 6)
# coherence results
t87_coherence = coherence(target87_evolution_data())
f87_coherence = coherence(fittest87_evolution_data())
t87f87_coherence = []
for i in range(len(ts)):
     t87f87_coherence.append(np.abs(t87_coherence[i] - f87_coherence[i]))
# population results
pop87 = []
for i in range(len(ts)):
     pop87.append(np.sum(populations(target87_evolution_data() , fittest87_evolution_data())[i]))




##########################################
##########################################
#target individual
target88 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest88 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target88cc = nx.node_connected_component(nx.from_numpy_array(np.real(target88)) , 1)
fittest88cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest88)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr88 = target88 == fittest88
# Count the number of False values
false_count88 = np.sum(arr88 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t88f88_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target88)) , nx.from_numpy_array(np.real(fittest88)))
# WL graph isomorphism test
t88_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target88)))
f88_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest88)))
#measuring the similarity between two graphs through graph_edit_distance
t88f88_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target88)) , nx.from_numpy_array(np.real(fittest88)))
# degree distribution
t88_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target88))), key=lambda x: x[1], reverse=True)]
f88_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest88))), key=lambda x: x[1], reverse=True)]
# number of connections
t88_connections = np.sum(t88_dd)
f88_connections = np.sum(f88_dd)
# distance
distance88 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest88_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest88 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest88 * t).T))

            ]
def fittest88_evolution_data():
    data = []
    for t in ts:
        data.append(fittest88_evolution(t))
    return data
#target
def target88_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target88 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target88 * t).T))

            ]
def target88_evolution_data():
    data = []
    for t in ts:
        data.append(target88_evolution(t))
    return data
# fidelity results
fidelity88_tab = []
for i in range(len(ts)):
    fidelity88_tab.append(fidelity(target88_evolution_data()[i][0] , fittest88_evolution_data()[i][0]))
fidelity88_tab = np.round(fidelity88_tab , decimals = 6)
# coherence results
t88_coherence = coherence(target88_evolution_data())
f88_coherence = coherence(fittest88_evolution_data())
t88f88_coherence = []
for i in range(len(ts)):
     t88f88_coherence.append(np.abs(t88_coherence[i] - f88_coherence[i]))
# population results
pop88 = []
for i in range(len(ts)):
     pop88.append(np.sum(populations(target88_evolution_data() , fittest88_evolution_data())[i]))



##########################################
##########################################
#target individual
target89 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest89 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target89cc = nx.node_connected_component(nx.from_numpy_array(np.real(target89)) , 1)
fittest89cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest89)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr89 = target89 == fittest89
# Count the number of False values
false_count89 = np.sum(arr89 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t89f89_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target89)) , nx.from_numpy_array(np.real(fittest89)))
# WL graph isomorphism test
t89_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target89)))
f89_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest89)))
#measuring the similarity between two graphs through graph_edit_distance
t89f89_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target89)) , nx.from_numpy_array(np.real(fittest89)))
# degree distribution
t89_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target89))), key=lambda x: x[1], reverse=True)]
f89_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest89))), key=lambda x: x[1], reverse=True)]
# number of connections
t89_connections = np.sum(t89_dd)
f89_connections = np.sum(f89_dd)
# distance
distance89 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest89_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest89 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest89 * t).T))

            ]
def fittest89_evolution_data():
    data = []
    for t in ts:
        data.append(fittest89_evolution(t))
    return data
#target
def target89_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target89 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target89 * t).T))

            ]
def target89_evolution_data():
    data = []
    for t in ts:
        data.append(target89_evolution(t))
    return data
# fidelity results
fidelity89_tab = []
for i in range(len(ts)):
    fidelity89_tab.append(fidelity(target89_evolution_data()[i][0] , fittest89_evolution_data()[i][0]))
fidelity89_tab = np.round(fidelity89_tab , decimals = 6)
# coherence results
t89_coherence = coherence(target89_evolution_data())
f89_coherence = coherence(fittest89_evolution_data())
t89f89_coherence = []
for i in range(len(ts)):
     t89f89_coherence.append(np.abs(t89_coherence[i] - f89_coherence[i]))
# population results
pop89 = []
for i in range(len(ts)):
     pop89.append(np.sum(populations(target89_evolution_data() , fittest89_evolution_data())[i]))


##########################################
##########################################
#target individual
target90 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest90 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target90cc = nx.node_connected_component(nx.from_numpy_array(np.real(target90)) , 1)
fittest90cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest90)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr90 = target90 == fittest90
# Count the number of False values
false_count90 = np.sum(arr90 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t90f90_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target90)) , nx.from_numpy_array(np.real(fittest90)))
# WL graph isomorphism test
t90_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target90)))
f90_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest90)))
# measuring the similarity between two graphs through graph_edit_distance
t90f90_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target90)) , nx.from_numpy_array(np.real(fittest90)))
# degree distribution
t90_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target90))), key=lambda x: x[1], reverse=True)]
f90_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest90))), key=lambda x: x[1], reverse=True)]
# number of connections
t90_connections = np.sum(t90_dd)
f90_connections = np.sum(f90_dd)
# distance
distance90 = 0.03298089547598293
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest90_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest90 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest90 * t).T))

            ]
def fittest90_evolution_data():
    data = []
    for t in ts:
        data.append(fittest90_evolution(t))
    return data
#target
def target90_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target90 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target90 * t).T))

            ]
def target90_evolution_data():
    data = []
    for t in ts:
        data.append(target90_evolution(t))
    return data
# fidelity results
fidelity90_tab = []
for i in range(len(ts)):
    fidelity90_tab.append(fidelity(target90_evolution_data()[i][0] , fittest90_evolution_data()[i][0]))
fidelity90_tab = np.round(fidelity90_tab , decimals = 6)
# coherence results
t90_coherence = coherence(target90_evolution_data())
f90_coherence = coherence(fittest90_evolution_data())
t90f90_coherence = []
for i in range(len(ts)):
     t90f90_coherence.append(np.abs(t90_coherence[i] - f90_coherence[i]))
# population results
pop90 = []
for i in range(len(ts)):
     pop90.append(np.sum(populations(target90_evolution_data() , fittest90_evolution_data())[i]))


##########################################
##########################################
#target individual
target91 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest91 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target91cc = nx.node_connected_component(nx.from_numpy_array(np.real(target91)) , 1)
fittest91cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest91)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr91 = target91 == fittest91
# Count the number of False values
false_count91 = np.sum(arr91 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t91f91_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target91)) , nx.from_numpy_array(np.real(fittest91)))
# WL graph isomorphism test
t91_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target91)))
f91_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest91)))
# measuring the similarity between two graphs through graph_edit_distance
t91f91_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target91)) , nx.from_numpy_array(np.real(fittest91)))
# degree distribution
t91_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target91))), key=lambda x: x[1], reverse=True)]
f91_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest91))), key=lambda x: x[1], reverse=True)]
# number of connections
t91_connections = np.sum(t91_dd)
f91_connections = np.sum(f91_dd)
# distance
distance91 = 0.45424141030282406
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest91_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest91 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest91 * t).T))

            ]
def fittest91_evolution_data():
    data = []
    for t in ts:
        data.append(fittest91_evolution(t))
    return data
#target
def target91_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target91 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target91 * t).T))

            ]
def target91_evolution_data():
    data = []
    for t in ts:
        data.append(target91_evolution(t))
    return data
# fidelity results
fidelity91_tab = []
for i in range(len(ts)):
    fidelity91_tab.append(fidelity(target91_evolution_data()[i][0] , fittest91_evolution_data()[i][0]))
fidelity91_tab = np.round(fidelity91_tab , decimals = 6)
# coherence results
t91_coherence = coherence(target91_evolution_data())
f91_coherence = coherence(fittest91_evolution_data())
t91f91_coherence = []
for i in range(len(ts)):
     t91f91_coherence.append(np.abs(t91_coherence[i] - f91_coherence[i]))
# population results
pop91 = []
for i in range(len(ts)):
     pop91.append(np.sum(populations(target91_evolution_data() , fittest91_evolution_data())[i]))


##########################################
##########################################
#target individual
target92 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest92 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target92cc = nx.node_connected_component(nx.from_numpy_array(np.real(target92)) , 1)
fittest92cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest92)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr92 = target92 == fittest92
# Count the number of False values
false_count92 = np.sum(arr92 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t92f92_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target92)) , nx.from_numpy_array(np.real(fittest92)))
# WL graph isomorphism test
t92_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target92)))
f92_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest92)))
# measuring the similarity between two graphs through graph_edit_distance
t92f92_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target92)) , nx.from_numpy_array(np.real(fittest92)))
# degree distribution
t92_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target92))), key=lambda x: x[1], reverse=True)]
f92_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest92))), key=lambda x: x[1], reverse=True)]
# number of connections
t92_connections = np.sum(t92_dd)
f92_connections = np.sum(f92_dd)
# distance
distance92 = 0.3041322338503185
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest92_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest92 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest92 * t).T))

            ]
def fittest92_evolution_data():
    data = []
    for t in ts:
        data.append(fittest92_evolution(t))
    return data
#target
def target92_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target92 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target92 * t).T))

            ]
def target92_evolution_data():
    data = []
    for t in ts:
        data.append(target92_evolution(t))
    return data
# fidelity results
fidelity92_tab = []
for i in range(len(ts)):
    fidelity92_tab.append(fidelity(target92_evolution_data()[i][0] , fittest92_evolution_data()[i][0]))
fidelity92_tab = np.round(fidelity92_tab , decimals = 6)
# coherence results
t92_coherence = coherence(target92_evolution_data())
f92_coherence = coherence(fittest92_evolution_data())
t92f92_coherence = []
for i in range(len(ts)):
     t92f92_coherence.append(np.abs(t92_coherence[i] - f92_coherence[i]))
# population results
pop92 = []
for i in range(len(ts)):
     pop92.append(np.sum(populations(target92_evolution_data() , fittest92_evolution_data())[i]))



##########################################
##########################################
#target individual
target93 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest93 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target93cc = nx.node_connected_component(nx.from_numpy_array(np.real(target93)) , 1)
fittest93cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest93)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr93 = target93 == fittest93
# Count the number of False values
false_count93 = np.sum(arr93 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t93f93_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target93)) , nx.from_numpy_array(np.real(fittest93)))
# WL graph isomorphism test
t93_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target93)))
f93_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest93)))
#measuring the similarity between two graphs through graph_edit_distance
t93f93_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target93)) , nx.from_numpy_array(np.real(fittest93)))
# degree distribution
t93_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target93))), key=lambda x: x[1], reverse=True)]
f93_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest93))), key=lambda x: x[1], reverse=True)]
# number of connections
t93_connections = np.sum(t93_dd)
f93_connections = np.sum(f93_dd)
# distance
distance93 = 0.002365231615459429
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest93_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest93 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest93 * t).T))

            ]
def fittest93_evolution_data():
    data = []
    for t in ts:
        data.append(fittest93_evolution(t))
    return data
#target
def target93_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target93 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target93 * t).T))

            ]
def target93_evolution_data():
    data = []
    for t in ts:
        data.append(target93_evolution(t))
    return data
# fidelity results
fidelity93_tab = []
for i in range(len(ts)):
    fidelity93_tab.append(fidelity(target93_evolution_data()[i][0] , fittest93_evolution_data()[i][0]))
fidelity93_tab = np.round(fidelity93_tab , decimals = 6)
# coherence results
t93_coherence = coherence(target93_evolution_data())
f93_coherence = coherence(fittest93_evolution_data())
t93f93_coherence = []
for i in range(len(ts)):
     t93f93_coherence.append(np.abs(t93_coherence[i] - f93_coherence[i]))
# population results
pop93 = []
for i in range(len(ts)):
     pop93.append(np.sum(populations(target93_evolution_data() , fittest93_evolution_data())[i]))



##########################################
##########################################
#target individual
target94 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest94 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target94cc = nx.node_connected_component(nx.from_numpy_array(np.real(target94)) , 1)
fittest94cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest94)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr94 = target94 == fittest94
# Count the number of False values
false_count94 = np.sum(arr94 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t94f94_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target94)) , nx.from_numpy_array(np.real(fittest94)))
# WL graph isomorphism test
t94_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target94)))
f94_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest94)))
#measuring the similarity between two graphs through graph_edit_distance
t94f94_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target94)) , nx.from_numpy_array(np.real(fittest94)))
# degree distribution
t94_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target94))), key=lambda x: x[1], reverse=True)]
f94_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest94))), key=lambda x: x[1], reverse=True)]
# number of connections
t94_connections = np.sum(t94_dd)
f94_connections = np.sum(f94_dd)
# distance
distance94 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest94_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest94 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest94 * t).T))

            ]
def fittest94_evolution_data():
    data = []
    for t in ts:
        data.append(fittest94_evolution(t))
    return data
#target
def target94_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target94 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target94 * t).T))

            ]
def target94_evolution_data():
    data = []
    for t in ts:
        data.append(target94_evolution(t))
    return data
# fidelity results
fidelity94_tab = []
for i in range(len(ts)):
    fidelity94_tab.append(fidelity(target94_evolution_data()[i][0] , fittest94_evolution_data()[i][0]))
fidelity94_tab = np.round(fidelity94_tab , decimals = 6)
# coherence results
t94_coherence = coherence(target94_evolution_data())
f94_coherence = coherence(fittest94_evolution_data())
t94f94_coherence = []
for i in range(len(ts)):
     t94f94_coherence.append(np.abs(t94_coherence[i] - f94_coherence[i]))
# population results
pop94 = []
for i in range(len(ts)):
     pop94.append(np.sum(populations(target94_evolution_data() , fittest94_evolution_data())[i]))


##########################################
##########################################
#target individual
target95 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest95 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target95cc = nx.node_connected_component(nx.from_numpy_array(np.real(target95)) , 1)
fittest95cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest95)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr95 = target95 == fittest95
# Count the number of False values
false_count95 = np.sum(arr95 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t95f95_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target95)) , nx.from_numpy_array(np.real(fittest95)))
# WL graph isomorphism test
t95_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target95)))
f95_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest95)))
#measuring the similarity between two graphs through graph_edit_distance
t95f95_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target95)) , nx.from_numpy_array(np.real(fittest95)))
# degree distribution
t95_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target95))), key=lambda x: x[1], reverse=True)]
f95_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest95))), key=lambda x: x[1], reverse=True)]
# number of connections
t95_connections = np.sum(t95_dd)
f95_connections = np.sum(f95_dd)
# distance
distance95 = 0.1353774793132435
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest95_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest95 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest95 * t).T))

            ]
def fittest95_evolution_data():
    data = []
    for t in ts:
        data.append(fittest95_evolution(t))
    return data
#target
def target95_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target95 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target95 * t).T))

            ]
def target95_evolution_data():
    data = []
    for t in ts:
        data.append(target95_evolution(t))
    return data
# fidelity results
fidelity95_tab = []
for i in range(len(ts)):
    fidelity95_tab.append(fidelity(target95_evolution_data()[i][0] , fittest95_evolution_data()[i][0]))
fidelity95_tab = np.round(fidelity95_tab , decimals = 6)
# coherence results
t95_coherence = coherence(target95_evolution_data())
f95_coherence = coherence(fittest95_evolution_data())
t95f95_coherence = []
for i in range(len(ts)):
     t95f95_coherence.append(np.abs(t95_coherence[i] - f95_coherence[i]))
# population results
pop95 = []
for i in range(len(ts)):
     pop95.append(np.sum(populations(target95_evolution_data() , fittest95_evolution_data())[i]))


##########################################
##########################################
#target individual
target96 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest96 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target96cc = nx.node_connected_component(nx.from_numpy_array(np.real(target96)) , 1)
fittest96cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest96)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr96 = target96 == fittest96
# Count the number of False values
false_count96 = np.sum(arr96 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t96f96_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target96)) , nx.from_numpy_array(np.real(fittest96)))
# WL graph isomorphism test
t96_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target96)))
f96_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest96)))
#measuring the similarity between two graphs through graph_edit_distance
t96f96_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target96)) , nx.from_numpy_array(np.real(fittest96)))
# degree distribution
t96_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target96))), key=lambda x: x[1], reverse=True)]
f96_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest96))), key=lambda x: x[1], reverse=True)]
# number of connections
t96_connections = np.sum(t96_dd)
f96_connections = np.sum(f96_dd)
# distance
distance96 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest96_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest96 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest96 * t).T))

            ]
def fittest96_evolution_data():
    data = []
    for t in ts:
        data.append(fittest96_evolution(t))
    return data
#target
def target96_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target96 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target96 * t).T))

            ]
def target96_evolution_data():
    data = []
    for t in ts:
        data.append(target96_evolution(t))
    return data
# fidelity results
fidelity96_tab = []
for i in range(len(ts)):
    fidelity96_tab.append(fidelity(target96_evolution_data()[i][0] , fittest96_evolution_data()[i][0]))
fidelity96_tab = np.round(fidelity96_tab , decimals = 6)
# coherence results
t96_coherence = coherence(target96_evolution_data())
f96_coherence = coherence(fittest96_evolution_data())
t96f96_coherence = []
for i in range(len(ts)):
     t96f96_coherence.append(np.abs(t96_coherence[i] - f96_coherence[i]))
# population results
pop96 = []
for i in range(len(ts)):
     pop96.append(np.sum(populations(target96_evolution_data() , fittest96_evolution_data())[i]))



##########################################
##########################################
#target individual
target97 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
# fittest individual
fittest97 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target97cc = nx.node_connected_component(nx.from_numpy_array(np.real(target97)) , 1)
fittest97cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest97)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr97 = target97 == fittest97
# Count the number of False values
false_count97 = np.sum(arr97 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t97f97_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target97)) , nx.from_numpy_array(np.real(fittest97)))
# WL graph isomorphism test
t97_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target97)))
f97_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest97)))
#measuring the similarity between two graphs through graph_edit_distance
t97f97_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target97)) , nx.from_numpy_array(np.real(fittest97)))
# degree distribution
t97_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target97))), key=lambda x: x[1], reverse=True)]
f97_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest97))), key=lambda x: x[1], reverse=True)]
# number of connections
t97_connections = np.sum(t97_dd)
f97_connections = np.sum(f97_dd)
# distance
distance97 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest97_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest97 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest97 * t).T))

            ]
def fittest97_evolution_data():
    data = []
    for t in ts:
        data.append(fittest97_evolution(t))
    return data
#target
def target97_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target97 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target97 * t).T))

            ]
def target97_evolution_data():
    data = []
    for t in ts:
        data.append(target97_evolution(t))
    return data
# fidelity results
fidelity97_tab = []
for i in range(len(ts)):
    fidelity97_tab.append(fidelity(target97_evolution_data()[i][0] , fittest97_evolution_data()[i][0]))
fidelity97_tab = np.round(fidelity97_tab , decimals = 6)
# coherence results
t97_coherence = coherence(target97_evolution_data())
f97_coherence = coherence(fittest97_evolution_data())
t97f97_coherence = []
for i in range(len(ts)):
     t97f97_coherence.append(np.abs(t97_coherence[i] - f97_coherence[i]))
# population results
pop97 = []
for i in range(len(ts)):
     pop97.append(np.sum(populations(target97_evolution_data() , fittest97_evolution_data())[i]))




##########################################
##########################################
#target individual
target98 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest98 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target98cc = nx.node_connected_component(nx.from_numpy_array(np.real(target98)) , 1)
fittest98cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest98)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr98 = target98 == fittest98
# Count the number of False values
false_count98 = np.sum(arr98 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t98f98_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target98)) , nx.from_numpy_array(np.real(fittest98)))
# WL graph isomorphism test
t98_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target98)))
f98_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest98)))
#measuring the similarity between two graphs through graph_edit_distance
t98f98_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target98)) , nx.from_numpy_array(np.real(fittest98)))
# degree distribution
t98_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target98))), key=lambda x: x[1], reverse=True)]
f98_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest98))), key=lambda x: x[1], reverse=True)]
# number of connections
t98_connections = np.sum(t98_dd)
f98_connections = np.sum(f98_dd)
# distance
distance98 = 0.14535754912682086
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest98_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest98 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest98 * t).T))

            ]
def fittest98_evolution_data():
    data = []
    for t in ts:
        data.append(fittest98_evolution(t))
    return data
#target
def target98_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target98 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target98 * t).T))

            ]
def target98_evolution_data():
    data = []
    for t in ts:
        data.append(target98_evolution(t))
    return data
# fidelity results
fidelity98_tab = []
for i in range(len(ts)):
    fidelity98_tab.append(fidelity(target98_evolution_data()[i][0] , fittest98_evolution_data()[i][0]))
fidelity98_tab = np.round(fidelity98_tab , decimals = 6)
# coherence results
t98_coherence = coherence(target98_evolution_data())
f98_coherence = coherence(fittest98_evolution_data())
t98f98_coherence = []
for i in range(len(ts)):
     t98f98_coherence.append(np.abs(t98_coherence[i] - f98_coherence[i]))
# population results
pop98 = []
for i in range(len(ts)):
     pop98.append(np.sum(populations(target98_evolution_data() , fittest98_evolution_data())[i]))



##########################################
##########################################
#target individual
target99 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest99 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target99cc = nx.node_connected_component(nx.from_numpy_array(np.real(target99)) , 1)
fittest99cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest99)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr99 = target99 == fittest99
# Count the number of False values
false_count99 = np.sum(arr99 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t99f99_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target99)) , nx.from_numpy_array(np.real(fittest99)))
# WL graph isomorphism test
t99_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target99)))
f99_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest99)))
#measuring the similarity between two graphs through graph_edit_distance
t99f99_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target99)) , nx.from_numpy_array(np.real(fittest99)))
# degree distribution
t99_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target99))), key=lambda x: x[1], reverse=True)]
f99_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest99))), key=lambda x: x[1], reverse=True)]
# number of connections
t99_connections = np.sum(t99_dd)
f99_connections = np.sum(f99_dd)
# distance
distance99 = 0.0
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest99_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest99 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest99 * t).T))

            ]
def fittest99_evolution_data():
    data = []
    for t in ts:
        data.append(fittest99_evolution(t))
    return data
#target
def target99_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target99 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target99 * t).T))

            ]
def target99_evolution_data():
    data = []
    for t in ts:
        data.append(target99_evolution(t))
    return data
# fidelity results
fidelity99_tab = []
for i in range(len(ts)):
    fidelity99_tab.append(fidelity(target99_evolution_data()[i][0] , fittest99_evolution_data()[i][0]))
fidelity99_tab = np.round(fidelity99_tab , decimals = 6)
# coherence results
t99_coherence = coherence(target99_evolution_data())
f99_coherence = coherence(fittest99_evolution_data())
t99f99_coherence = []
for i in range(len(ts)):
     t99f99_coherence.append(np.abs(t99_coherence[i] - f99_coherence[i]))
# population results
pop99 = []
for i in range(len(ts)):
     pop99.append(np.sum(populations(target99_evolution_data() , fittest99_evolution_data())[i]))








##########################################
##########################################
#target individual
target100 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
# fittest individual
fittest100 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])
##########################################
##########################################
# network properties
##########################################
##########################################
# finding the connections related to the component the excitation is initially injected into
target100cc = nx.node_connected_component(nx.from_numpy_array(np.real(target100)) , 1)
fittest100cc = nx.node_connected_component(nx.from_numpy_array(np.real(fittest100)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr100 = target100 == fittest100
# Count the number of False values
false_count100 = np.sum(arr100 == False) // 2
# is the fittest individual an isomorphism of the target matrix?
t100f100_iso = nx.is_isomorphic(nx.from_numpy_array(np.real(target100)) , nx.from_numpy_array(np.real(fittest100)))
# WL graph isomorphism test
t100_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target100)))
f100_iso = nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest100)))
#measuring the similarity between two graphs through graph_edit_distance
t100f100_ged = nx.graph_edit_distance(nx.from_numpy_array(np.real(target100)) , nx.from_numpy_array(np.real(fittest100)))
# degree distribution
t100_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(target100))), key=lambda x: x[1], reverse=True)]
f100_dd = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(fittest100))), key=lambda x: x[1], reverse=True)]
# number of connections
t100_connections = np.sum(t100_dd)
f100_connections = np.sum(f100_dd)
# distance
distance100 = 1.0103029524088925e-14
##########################################
##########################################
# physical properties
##########################################
##########################################
##########################################
#Temporal Evolution
##########################################
#fittest individual
def fittest100_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * fittest100 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * fittest100 * t).T))

            ]
def fittest100_evolution_data():
    data = []
    for t in ts:
        data.append(fittest100_evolution(t))
    return data
#target
def target100_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * target100 * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target100 * t).T))

            ]
def target100_evolution_data():
    data = []
    for t in ts:
        data.append(target100_evolution(t))
    return data
# fidelity results
fidelity100_tab = []
for i in range(len(ts)):
    fidelity100_tab.append(fidelity(target100_evolution_data()[i][0] , fittest100_evolution_data()[i][0]))
fidelity100_tab = np.round(fidelity100_tab , decimals = 6)
# coherence results
t100_coherence = coherence(target100_evolution_data())
f100_coherence = coherence(fittest100_evolution_data())
t100f100_coherence = []
for i in range(len(ts)):
     t100f100_coherence.append(np.abs(t100_coherence[i] - f100_coherence[i]))
# population results
pop100 = []
for i in range(len(ts)):
     pop100.append(np.sum(populations(target100_evolution_data() , fittest100_evolution_data())[i]))









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
                                    target46cc == fittest46cc , target47cc == fittest47cc , target48cc == fittest48cc , target49cc == fittest49cc , target50cc == fittest50cc , 
                                    target51cc == fittest51cc , target52cc == fittest52cc , target53cc == fittest53cc , target54cc == fittest54cc , target55cc == fittest55cc , 
                                    target56cc == fittest56cc , target57cc == fittest57cc , target58cc == fittest58cc , target59cc == fittest59cc , target60cc == fittest60cc , 
                                    target61cc == fittest61cc , target62cc == fittest62cc , target63cc == fittest63cc , target64cc == fittest64cc , target65cc == fittest65cc , 
                                    target66cc == fittest66cc , target67cc == fittest67cc , target68cc == fittest68cc , target69cc == fittest69cc , target70cc == fittest70cc , 
                                    target71cc == fittest71cc , target72cc == fittest72cc , target73cc == fittest73cc , target74cc == fittest74cc , target75cc == fittest75cc , 
                                    target76cc == fittest76cc , target77cc == fittest77cc , target78cc == fittest78cc , target79cc == fittest79cc , target80cc == fittest80cc , 
                                    target81cc == fittest81cc , target82cc == fittest82cc , target83cc == fittest83cc , target84cc == fittest84cc , target85cc == fittest85cc , 
                                    target86cc == fittest86cc , target87cc == fittest87cc , target88cc == fittest88cc , target89cc == fittest89cc , target90cc == fittest90cc , 
                                    target91cc == fittest91cc , target92cc == fittest92cc , target93cc == fittest93cc , target94cc == fittest94cc , target95cc == fittest95cc , 
                                    target96cc == fittest96cc , target97cc == fittest97cc , target98cc == fittest98cc , target99cc == fittest99cc , target100cc == fittest100cc])

connected_component10_tab = np.array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True, False,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True])
connected_component_true10Nodes = np.sum(connected_component10_tab == True)
# table showing number of different connections between target and fittest arrays
false_count_tab = np.array([false_count1 , false_count2 , false_count3 , false_count4 , false_count5 , false_count6 , false_count7 , false_count8 , false_count9 , false_count10 , 
                            false_count11 , false_count12 , false_count13 , false_count14 , false_count15 , false_count16 , false_count17 , false_count18 , false_count19 , false_count20 , 
                            false_count21 , false_count22 , false_count23 , false_count24 , false_count25 , false_count26 , false_count27 , false_count28 , false_count29 , false_count30 , 
                            false_count31 , false_count32 , false_count33 , false_count34 , false_count35 , false_count36 , false_count37 , false_count38 , false_count39 , false_count40 , 
                            false_count41 , false_count42 , false_count43 , false_count44 , false_count45 , false_count46 , false_count47 , false_count48 , false_count49 , false_count50 , 
                            false_count51 , false_count52 , false_count53 , false_count54 , false_count55 , false_count56 , false_count57 , false_count58 , false_count59 , false_count60 , 
                            false_count61 , false_count62 , false_count63 , false_count64 , false_count65 , false_count66 , false_count67 , false_count68 , false_count69 , false_count70 , 
                            false_count71 , false_count72 , false_count73 , false_count74 , false_count75 , false_count76 , false_count77 , false_count78 , false_count79 , false_count80 , 
                            false_count81 , false_count82 , false_count83 , false_count84 , false_count85 , false_count86 , false_count87 , false_count88 , false_count89 , false_count90 , 
                            false_count91 , false_count92 , false_count93 , false_count94 , false_count95 , false_count96 , false_count97 , false_count98 , false_count99 , false_count100])

false_count10_tab = np.array([ 8, 10, 14,  2, 13, 24, 17, 12,  3,  0, 16,  4,  4,  4, 19,  8, 20,
        5,  0, 13,  5,  8,  3, 14,  0,  0, 11,  2, 11,  8, 12,  3,  6,  6,
       10, 16, 12, 14,  8,  4,  8, 17, 11, 14,  9, 12,  0,  2,  8, 15,  0,
       22,  0,  7, 10,  9,  4,  4, 17, 12, 11, 12,  0,  6,  4,  0, 14,  2,
       16,  2,  9,  6,  4,  2,  7, 16,  4,  8, 13, 16,  6,  0, 14,  1,  6,
        2, 14,  0,  0, 14, 20, 12,  6,  6, 13,  8,  0, 12,  0,  4])
false_count_10Nodes = np.sum(false_count10_tab == 0)
# table showing how many true isomorphisms were found of target network
isomorphism_tab = np.array([t1f1_iso , t2f2_iso , t3f3_iso , t4f4_iso , t5f5_iso , t6f6_iso , t7f7_iso , t8f8_iso , t9f9_iso , t10f10_iso , 
                            t11f11_iso , t12f12_iso , t13f13_iso , t14f14_iso , t15f15_iso , t16f16_iso , t17f17_iso , t18f18_iso , t19f19_iso , t20f20_iso , 
                            t21f21_iso , t22f22_iso , t23f23_iso , t24f24_iso , t25f25_iso , t26f26_iso , t27f27_iso , t28f28_iso , t29f29_iso , t30f30_iso , 
                            t31f31_iso , t32f32_iso , t33f33_iso , t34f34_iso , t35f35_iso , t36f36_iso , t37f37_iso , t38f38_iso , t39f39_iso , t40f40_iso , 
                            t41f41_iso , t42f42_iso , t43f43_iso , t44f44_iso , t45f45_iso , t46f46_iso , t47f47_iso , t48f48_iso , t49f49_iso , t50f50_iso , 
                            t51f51_iso , t52f52_iso , t53f53_iso , t54f54_iso , t55f55_iso , t56f56_iso , t57f57_iso , t58f58_iso , t59f59_iso , t60f60_iso , 
                            t61f61_iso , t62f62_iso , t63f63_iso , t64f64_iso , t65f65_iso , t66f66_iso , t67f67_iso , t68f68_iso , t69f69_iso , t70f70_iso , 
                            t71f71_iso , t72f72_iso , t73f73_iso , t74f74_iso , t75f75_iso , t76f76_iso , t77f77_iso , t78f78_iso , t79f79_iso , t80f80_iso , 
                            t81f81_iso , t82f82_iso , t83f83_iso , t84f84_iso , t85f85_iso , t86f86_iso , t87f87_iso , t88f88_iso , t89f89_iso , t90f90_iso , 
                            t91f91_iso , t92f92_iso , t93f93_iso , t94f94_iso , t95f95_iso , t96f96_iso , t97f97_iso , t98f98_iso , t99f99_iso , t100f100_iso])

isomorphism10_tab = np.array([ True, False, False, False, False, False, False, False, False,
        True, False,  True,  True,  True, False, False, False, False,
        True, False, False,  True, False, False,  True,  True, False,
        True, False,  True, False, False,  True,  True, False, False,
       False, False,  True, False,  True, False, False, False, False,
       False,  True,  True, False, False,  True, False,  True, False,
       False, False,  True,  True, False, False, False, False,  True,
        True, False,  True, False, False, False,  True, False,  True,
       False,  True, False, False,  True, False, False, False, False,
        True, False, False, False, False, False,  True,  True, False,
       False, False,  True, False, False,  True,  True, False,  True,
        True])
true_count_10Nodes = np.sum(isomorphism10_tab == True)


# table of graph edit distance for target and fittest individuals
graph_edit_distance_tab = np.array([t1f1_ged , t2f2_ged , t3f3_ged , t4f4_ged , t5f5_ged , t6f6_ged , t7f7_ged , t8f8_ged , t9f9_ged , t10f10_ged , 
                                    t11f11_ged , t12f12_ged , t13f13_ged , t14f14_ged , t15f15_ged , t16f16_ged , t17f17_ged , t18f18_ged , t19f19_ged , t20f20_ged , 
                                    t21f21_ged , t22f22_ged , t23f23_ged , t24f24_ged , t25f25_ged , t26f26_ged , t27f27_ged , t28f28_ged , t29f29_ged , t30f30_ged , 
                                    t31f31_ged , t32f32_ged , t33f33_ged , t34f34_ged , t35f35_ged , t36f36_ged , t37f37_ged , t38f38_ged , t39f39_ged , t40f40_ged , 
                                    t41f41_ged , t42f42_ged , t43f43_ged , t44f44_ged , t45f45_ged , t46f46_ged , t47f47_ged , t48f48_ged , t49f49_ged , t50f50_ged , 
                                    t51f51_ged , t52f52_ged , t53f53_ged , t54f54_ged , t55f55_ged , t56f56_ged , t57f57_ged , t58f58_ged , t59f59_ged , t60f60_ged , 
                                    t61f61_ged , t62f62_ged , t63f63_ged , t64f64_ged , t65f65_ged , t66f66_ged , t67f67_ged , t68f68_ged , t69f69_ged , t70f70_ged , 
                                    t71f71_ged , t72f72_ged , t73f73_ged , t74f74_ged , t75f75_ged , t76f76_ged , t77f77_ged , t78f78_ged , t79f79_ged , t80f80_ged , 
                                    t81f81_ged , t82f82_ged , t83f83_ged , t84f84_ged , t85f85_ged , t86f86_ged , t87f87_ged , t88f88_ged , t89f89_ged , t90f90_ged , 
                                    t91f91_ged , t92f92_ged , t93f93_ged , t94f94_ged , t95f95_ged , t96f96_ged , t97f97_ged , t98f98_ged , t99f99_ged , t100f100_ged])


graph_edit_distance10_tab = np.array([ 0.,  4.,  6.,  2.,  3.,  8.,  5.,  2.,  3.,  0.,  6.,  0.,  0.,
        0.,  5.,  4.,  6.,  1.,  0.,  5.,  3.,  0.,  1.,  8.,  0.,  0.,
       11.,  0.,  5.,  0.,  6.,  1.,  0.,  0.,  4.,  4.,  6.,  6.,  0.,
        2.,  0.,  5.,  7.,  2.,  3.,  6.,  0.,  0.,  2.,  7.,  0.,  8.,
        0.,  5.,  2.,  5.,  0.,  0.,  7.,  4.,  5.,  8.,  0.,  0.,  2.,
        0.,  8.,  2.,  2.,  0.,  3.,  0.,  4.,  0.,  7.,  4.,  0.,  4.,
        3.,  6.,  2.,  0.,  4.,  1.,  4.,  2.,  2.,  0.,  0.,  4.,  8.,
        6.,  0.,  2.,  9.,  0.,  0.,  6.,  0.,  0.])


# table showing how many times the degree distribution of the fittest network matched that of the target network
degree_distribution_table = np.array([t1_dd == f1_dd , t2_dd == f2_dd , t3_dd == f3_dd , t4_dd == f4_dd , t5_dd == f5_dd , t6_dd == f6_dd , t7_dd == f7_dd , t8_dd == f8_dd , t9_dd == f9_dd , t10_dd == f10_dd , 
                                      t11_dd == f11_dd , t12_dd == f12_dd , t13_dd == f13_dd , t14_dd == f14_dd , t15_dd == f15_dd , t16_dd == f16_dd , t17_dd == f17_dd , t18_dd == f18_dd , t19_dd == f19_dd , t20_dd == f20_dd , 
                                      t21_dd == f21_dd , t22_dd == f22_dd , t23_dd == f23_dd , t24_dd == f24_dd , t25_dd == f25_dd , t26_dd == f26_dd , t27_dd == f27_dd , t28_dd == f28_dd , t29_dd == f29_dd , t30_dd == f30_dd , 
                                      t31_dd == f31_dd , t32_dd == f32_dd , t33_dd == f33_dd , t34_dd == f34_dd , t35_dd == f35_dd , t36_dd == f36_dd , t37_dd == f37_dd , t38_dd == f38_dd , t39_dd == f39_dd , t40_dd == f40_dd , 
                                      t41_dd == f41_dd , t42_dd == f42_dd , t43_dd == f43_dd , t44_dd == f44_dd , t45_dd == f45_dd , t46_dd == f46_dd , t47_dd == f47_dd , t48_dd == f48_dd , t49_dd == f49_dd , t50_dd == f50_dd , 
                                      t51_dd == f51_dd , t52_dd == f52_dd , t53_dd == f53_dd , t54_dd == f54_dd , t55_dd == f55_dd , t56_dd == f56_dd , t57_dd == f57_dd , t58_dd == f58_dd , t59_dd == f59_dd , t60_dd == f60_dd , 
                                      t61_dd == f61_dd , t62_dd == f62_dd , t63_dd == f63_dd , t64_dd == f64_dd , t65_dd == f65_dd , t66_dd == f66_dd , t67_dd == f67_dd , t68_dd == f68_dd , t69_dd == f69_dd , t70_dd == f70_dd , 
                                      t71_dd == f71_dd , t72_dd == f72_dd , t73_dd == f73_dd , t74_dd == f74_dd , t75_dd == f75_dd , t76_dd == f76_dd , t77_dd == f77_dd , t78_dd == f78_dd , t79_dd == f79_dd , t80_dd == f80_dd , 
                                      t81_dd == f81_dd , t82_dd == f82_dd , t83_dd == f83_dd , t84_dd == f84_dd , t85_dd == f85_dd , t86_dd == f86_dd , t87_dd == f87_dd , t88_dd == f88_dd , t89_dd == f89_dd , t90_dd == f90_dd , 
                                      t91_dd == f91_dd , t92_dd == f92_dd , t93_dd == f93_dd , t94_dd == f94_dd , t95_dd == f95_dd , t96_dd == f96_dd , t97_dd == f97_dd , t98_dd == f98_dd , t99_dd == f99_dd , t100_dd == f100_dd])


degree_distribution10_table = np.array([ True, False, False, False, False, False, False, False, False,
        True, False,  True,  True,  True, False,  True, False, False,
        True, False, False,  True, False,  True,  True,  True, False,
        True, False,  True, False, False,  True,  True, False, False,
       False, False,  True, False,  True, False, False, False, False,
       False,  True,  True,  True, False,  True, False,  True, False,
       False, False,  True,  True, False, False, False, False,  True,
        True,  True,  True, False,  True,  True,  True, False,  True,
       False,  True, False, False,  True, False, False, False, False,
        True, False, False,  True, False, False,  True,  True, False,
       False, False,  True, False, False,  True,  True, False,  True,
        True])
degree_distribution_10Nodes_true = np.sum(degree_distribution10_table == True)
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
                            t46_connections == f46_connections , t47_connections == f47_connections , t48_connections == f48_connections , t49_connections == f49_connections , t50_connections == f50_connections , 
                            t51_connections == f51_connections , t52_connections == f52_connections , t53_connections == f53_connections , t54_connections == f54_connections , t55_connections == f55_connections , 
                            t56_connections == f56_connections , t57_connections == f57_connections , t58_connections == f58_connections , t59_connections == f59_connections , t60_connections == f60_connections , 
                            t61_connections == f61_connections , t62_connections == f62_connections , t63_connections == f63_connections , t64_connections == f64_connections , t65_connections == f65_connections , 
                            t66_connections == f66_connections , t67_connections == f67_connections , t68_connections == f68_connections , t69_connections == f69_connections , t70_connections == f70_connections , 
                            t71_connections == f71_connections , t72_connections == f72_connections , t73_connections == f73_connections , t74_connections == f74_connections , t75_connections == f75_connections , 
                            t76_connections == f76_connections , t77_connections == f77_connections , t78_connections == f78_connections , t79_connections == f79_connections , t80_connections == f80_connections , 
                            t81_connections == f81_connections , t82_connections == f82_connections , t83_connections == f83_connections , t84_connections == f84_connections , t85_connections == f85_connections , 
                            t86_connections == f86_connections , t87_connections == f87_connections , t88_connections == f88_connections , t89_connections == f89_connections , t90_connections == f90_connections , 
                            t91_connections == f91_connections , t92_connections == f92_connections , t93_connections == f93_connections , t94_connections == f94_connections , t95_connections == f95_connections , 
                            t96_connections == f96_connections , t97_connections == f97_connections , t98_connections == f98_connections , t99_connections == f99_connections , t100_connections == f100_connections])

connections10_tab = np.array([ True, False,  True,  True, False, False, False,  True, False,
        True, False,  True,  True,  True, False,  True, False, False,
        True, False, False,  True, False,  True,  True,  True, False,
        True, False,  True,  True, False,  True,  True, False, False,
       False, False,  True,  True,  True, False, False,  True, False,
       False,  True,  True,  True, False,  True, False,  True, False,
       False, False,  True,  True, False,  True, False, False,  True,
        True,  True,  True, False,  True,  True,  True, False,  True,
       False,  True, False, False,  True,  True, False, False,  True,
        True, False, False,  True,  True,  True,  True,  True,  True,
       False, False,  True, False, False,  True,  True,  True,  True,
        True])
connections_10Nodes_true = np.sum(connections10_tab == True)
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
                         fidelity41_tab , fidelity42_tab , fidelity43_tab , fidelity44_tab , fidelity45_tab , fidelity46_tab , fidelity47_tab , fidelity48_tab , fidelity49_tab , fidelity50_tab , 
                         fidelity51_tab , fidelity52_tab , fidelity53_tab , fidelity54_tab , fidelity55_tab , fidelity56_tab , fidelity57_tab , fidelity58_tab , fidelity59_tab , fidelity60_tab , 
                         fidelity61_tab , fidelity62_tab , fidelity63_tab , fidelity64_tab , fidelity65_tab , fidelity66_tab , fidelity67_tab , fidelity68_tab , fidelity69_tab , fidelity70_tab , 
                         fidelity71_tab , fidelity72_tab , fidelity73_tab , fidelity74_tab , fidelity75_tab , fidelity76_tab , fidelity77_tab , fidelity78_tab , fidelity79_tab , fidelity80_tab , 
                         fidelity81_tab , fidelity82_tab , fidelity83_tab , fidelity84_tab , fidelity85_tab , fidelity86_tab , fidelity87_tab , fidelity88_tab , fidelity89_tab , fidelity90_tab , 
                         fidelity91_tab , fidelity92_tab , fidelity93_tab , fidelity94_tab , fidelity95_tab , fidelity96_tab , fidelity97_tab , fidelity98_tab , fidelity99_tab , fidelity100_tab]).T



fidelity10_tab = np.array([[1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      , 1.      , 1.      ,
        1.      , 1.      , 1.      , 1.      ],
       [1.      , 1.      , 0.999776, 0.999924, 0.999925, 0.951135,
        0.975014, 0.999928, 1.      , 1.      , 0.975327, 1.      ,
        1.      , 1.      , 0.976159, 1.      , 0.986263, 1.      ,
        1.      , 0.999628, 1.      , 1.      , 1.      , 0.999772,
        1.      , 1.      , 1.      , 0.999929, 0.999925, 0.999924,
        0.999783, 1.      , 1.      , 0.999998, 0.999924, 1.      ,
        0.975453, 0.975863, 1.      , 0.999925, 0.999929, 0.976673,
        1.      , 0.999926, 0.999853, 0.999927, 1.      , 0.999929,
        0.999999, 0.99955 , 1.      , 0.963267, 1.      , 1.      ,
        0.999925, 1.      , 0.999928, 0.999927, 0.975545, 0.999926,
        0.98826 , 0.999486, 1.      , 0.999926, 1.      , 1.      ,
        0.975792, 1.      , 0.976459, 0.999929, 1.      , 0.976243,
        1.      , 0.999929, 0.999815, 0.999737, 1.      , 0.976747,
        1.      , 0.999777, 0.999997, 1.      , 0.999886, 1.      ,
        0.999926, 1.      , 0.999568, 1.      , 1.      , 0.999999,
        0.975392, 0.999   , 1.      , 1.      , 0.999771, 1.      ,
        1.      , 0.999778, 1.      , 1.      ],
       [1.      , 1.      , 0.996637, 0.998819, 0.998829, 0.81171 ,
        0.898065, 0.999004, 1.      , 1.      , 0.902343, 1.      ,
        0.999998, 1.      , 0.913695, 0.999993, 0.930995, 1.      ,
        1.      , 0.994469, 1.      , 0.999994, 1.      , 0.996404,
        1.      , 1.      , 1.      , 0.999067, 0.998871, 0.998784,
        0.997026, 1.      , 1.      , 0.999856, 0.998812, 1.      ,
        0.904271, 0.909853, 1.      , 0.998878, 0.999102, 0.921076,
        0.99999 , 0.998914, 0.997875, 0.998945, 1.      , 0.999085,
        0.999966, 0.993142, 1.      , 0.857235, 1.      , 1.      ,
        0.998833, 0.99999 , 0.99903 , 0.998971, 0.905362, 0.998943,
        0.95952 , 0.992651, 1.      , 0.998945, 0.999989, 1.      ,
        0.908972, 0.999994, 0.917753, 0.999062, 1.      , 0.914854,
        1.      , 0.999085, 0.997265, 0.995972, 1.      , 0.921651,
        1.      , 0.996668, 0.999853, 1.      , 0.998199, 1.      ,
        0.998896, 1.      , 0.994167, 1.      , 1.      , 0.999933,
        0.904319, 0.986813, 0.999993, 1.      , 0.996346, 1.      ,
        1.      , 0.996731, 1.      , 1.      ],
       [0.999998, 1.      , 0.984701, 0.994256, 0.994378, 0.608354,
        0.772249, 0.996103, 1.      , 1.      , 0.789082, 1.      ,
        0.999958, 1.      , 0.832801, 0.99992 , 0.809661, 1.      ,
        1.      , 0.975725, 1.      , 0.999946, 1.      , 0.982428,
        1.      , 1.      , 1.      , 0.996666, 0.9948  , 0.994094,
        0.988375, 1.      , 1.      , 0.998641, 0.994186, 1.      ,
        0.797647, 0.819009, 1.      , 0.99488 , 0.996958, 0.863962,
        0.999895, 0.995244, 0.990945, 0.995485, 1.      , 0.996797,
        0.999686, 0.968124, 1.      , 0.697222, 1.      , 0.999999,
        0.994416, 0.999888, 0.996339, 0.995805, 0.800932, 0.995536,
        0.927986, 0.969233, 1.      , 0.995567, 0.999893, 1.      ,
        0.816578, 0.999932, 0.848435, 0.996615, 1.      , 0.837491,
        1.      , 0.996804, 0.98785 , 0.981092, 1.      , 0.863617,
        0.999999, 0.984968, 0.998603, 1.      , 0.991085, 1.      ,
        0.995071, 1.      , 0.977937, 1.      , 1.      , 0.999276,
        0.802333, 0.951825, 0.999919, 1.      , 0.981698, 1.      ,
        1.      , 0.9857  , 1.      , 1.      ],
       [0.999981, 1.      , 0.95846 , 0.982859, 0.983563, 0.3925  ,
        0.620173, 0.991479, 1.      , 1.      , 0.659182, 0.999995,
        0.999641, 1.      , 0.751103, 0.999536, 0.625146, 1.      ,
        1.      , 0.939309, 1.      , 0.999766, 1.      , 0.948274,
        1.      , 1.      , 1.      , 0.993692, 0.985577, 0.983215,
        0.974211, 1.      , 1.      , 0.994122, 0.982485, 1.      ,
        0.679615, 0.724087, 1.      , 0.985971, 0.994745, 0.827214,
        0.999447, 0.987735, 0.977535, 0.988417, 1.      , 0.994057,
        0.998679, 0.911255, 1.      , 0.512979, 1.      , 0.999993,
        0.983704, 0.99938 , 0.992417, 0.990306, 0.684549, 0.989106,
        0.903912, 0.925732, 1.      , 0.989313, 0.999545, 1.      ,
        0.723904, 0.999654, 0.785305, 0.993488, 1.      , 0.762734,
        1.      , 0.994118, 0.967716, 0.946595, 1.      , 0.819196,
        0.999993, 0.959431, 0.993924, 1.      , 0.972836, 1.      ,
        0.986946, 1.      , 0.954075, 1.      , 1.      , 0.996242,
        0.697993, 0.904043, 0.999531, 1.      , 0.943868, 1.      ,
        1.      , 0.963537, 1.      , 1.      ],
       [0.999895, 1.      , 0.916672, 0.961193, 0.963889, 0.228237,
        0.480589, 0.987132, 1.      , 1.      , 0.550971, 0.999974,
        0.998277, 1.      , 0.675437, 0.998193, 0.424732, 1.      ,
        1.      , 0.89524 , 1.      , 0.99936 , 1.      , 0.887856,
        1.      , 1.      , 1.      , 0.992398, 0.970049, 0.966111,
        0.958402, 1.      , 1.      , 0.983901, 0.959867, 1.      ,
        0.580749, 0.640562, 1.      , 0.971348, 0.99456 , 0.807568,
        0.998059, 0.976883, 0.959656, 0.97733 , 1.      , 0.99268 ,
        0.996489, 0.817571, 1.      , 0.342931, 1.      , 0.99996 ,
        0.964153, 0.997693, 0.989386, 0.984424, 0.585578, 0.981003,
        0.88571 , 0.872128, 1.      , 0.981846, 0.998775, 1.      ,
        0.656358, 0.998827, 0.729325, 0.991931, 1.      , 0.699124,
        1.      , 0.992981, 0.93591 , 0.888479, 1.      , 0.782915,
        0.999958, 0.918527, 0.983243, 1.      , 0.937312, 1.      ,
        0.974541, 1.      , 0.934583, 1.      , 1.      , 0.987096,
        0.608962, 0.870581, 0.998152, 1.      , 0.87099 , 1.      ,
        1.      , 0.933239, 1.      , 1.      ],
       [0.999584, 1.      , 0.863722, 0.926788, 0.934625, 0.161848,
        0.385569, 0.985007, 1.      , 1.      , 0.499972, 0.999897,
        0.994285, 1.      , 0.603375, 0.994563, 0.268843, 1.      ,
        1.      , 0.863018, 1.      , 0.998714, 1.      , 0.804269,
        1.      , 1.      , 1.      , 0.993718, 0.948437, 0.946629,
        0.941559, 1.      , 1.      , 0.967422, 0.923168, 1.      ,
        0.524782, 0.574072, 1.      , 0.95176 , 0.996379, 0.781114,
        0.994755, 0.964701, 0.940922, 0.961494, 1.      , 0.992578,
        0.993079, 0.696015, 1.      , 0.218324, 1.      , 0.999842,
        0.934683, 0.993347, 0.988729, 0.981043, 0.527539, 0.973787,
        0.863441, 0.82644 , 1.      , 0.97628 , 0.997566, 1.      ,
        0.628192, 0.996971, 0.674606, 0.993073, 1.      , 0.649414,
        1.      , 0.993573, 0.894197, 0.811919, 1.      , 0.740664,
        0.99983 , 0.865028, 0.965815, 1.      , 0.880267, 1.      ,
        0.959376, 1.      , 0.926951, 1.      , 1.      , 0.966285,
        0.526643, 0.865183, 0.994346, 1.      , 0.757863, 1.      ,
        1.      , 0.903207, 1.      , 1.      ],
       [0.998706, 1.      , 0.807286, 0.879024, 0.897569, 0.196577,
        0.343146, 0.984769, 1.      , 1.      , 0.514116, 0.999678,
        0.985459, 1.      , 0.531265, 0.986444, 0.186933, 1.      ,
        1.      , 0.851438, 1.      , 0.99791 , 1.      , 0.710146,
        1.      , 1.      , 1.      , 0.995737, 0.921815, 0.9291  ,
        0.916081, 1.      , 1.      , 0.946089, 0.870847, 1.      ,
        0.5205  , 0.522698, 1.      , 0.928987, 0.997167, 0.725578,
        0.988209, 0.953222, 0.921676, 0.938553, 1.      , 0.991007,
        0.98867 , 0.56731 , 1.      , 0.149265, 1.      , 0.999506,
        0.896304, 0.984003, 0.989242, 0.981596, 0.513636, 0.969028,
        0.829699, 0.801639, 1.      , 0.974852, 0.996036, 1.      ,
        0.635342, 0.993563, 0.622312, 0.99537 , 1.      , 0.618123,
        1.      , 0.993452, 0.845004, 0.731211, 1.      , 0.691106,
        0.999451, 0.803763, 0.94304 , 1.      , 0.802072, 1.      ,
        0.942882, 1.      , 0.92306 , 1.      , 1.      , 0.927899,
        0.43633 , 0.869169, 0.985568, 1.      , 0.611625, 1.      ,
        1.      , 0.88133 , 1.      , 1.      ],
       [0.996638, 1.      , 0.753831, 0.819588, 0.856853, 0.295598,
        0.341543, 0.983628, 1.      , 1.      , 0.562589, 0.999165,
        0.969659, 1.      , 0.462639, 0.970846, 0.175346, 1.      ,
        1.      , 0.847826, 1.      , 0.997023, 1.      , 0.620935,
        1.      , 1.      , 1.      , 0.99561 , 0.891179, 0.911977,
        0.870878, 1.      , 1.      , 0.920466, 0.803666, 1.      ,
        0.561572, 0.4818  , 1.      , 0.904859, 0.993794, 0.642704,
        0.976879, 0.942643, 0.896822, 0.904312, 1.      , 0.985639,
        0.983023, 0.454851, 1.      , 0.1249  , 1.      , 0.998705,
        0.852084, 0.966536, 0.987726, 0.984469, 0.524777, 0.965354,
        0.787046, 0.79688 , 1.      , 0.976784, 0.994212, 1.      ,
        0.658546, 0.988218, 0.584767, 0.996196, 1.      , 0.613879,
        1.      , 0.990368, 0.789312, 0.662514, 1.      , 0.65218 ,
        0.998513, 0.738473, 0.915887, 1.      , 0.709398, 1.      ,
        0.924086, 1.      , 0.903972, 1.      , 1.      , 0.86835 ,
        0.342745, 0.848914, 0.967988, 1.      , 0.451775, 1.      ,
        1.      , 0.86924 , 1.      , 1.      ],
       [0.992429, 1.      , 0.705219, 0.752255, 0.817692, 0.407621,
        0.363471, 0.978726, 1.      , 1.      , 0.591688, 0.99812 ,
        0.945487, 1.      , 0.40995 , 0.944525, 0.223737, 1.      ,
        1.      , 0.832593, 1.      , 0.996   , 1.      , 0.545959,
        1.      , 1.      , 1.      , 0.993022, 0.85691 , 0.885432,
        0.800885, 1.      , 1.      , 0.887578, 0.724882, 1.      ,
        0.629042, 0.446961, 1.      , 0.88057 , 0.98597 , 0.55887 ,
        0.959184, 0.930367, 0.860204, 0.854296, 1.      , 0.97804 ,
        0.975066, 0.375172, 1.      , 0.124816, 1.      , 0.997052,
        0.806442, 0.937468, 0.982115, 0.985788, 0.534125, 0.958614,
        0.746055, 0.800037, 1.      , 0.978357, 0.99184 , 1.      ,
        0.675061, 0.980856, 0.57642 , 0.994927, 1.      , 0.644885,
        1.      , 0.985544, 0.726124, 0.613242, 1.      , 0.646242,
        0.996479, 0.670076, 0.882076, 1.      , 0.614179, 1.      ,
        0.89824 , 1.      , 0.857945, 1.      , 1.      , 0.789208,
        0.26605 , 0.794846, 0.936707, 1.      , 0.303267, 1.      ,
        1.      , 0.860709, 1.      , 1.      ]])


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
                          t41f41_coherence , t42f42_coherence , t43f43_coherence , t44f44_coherence , t45f45_coherence , t46f46_coherence , t47f47_coherence , t48f48_coherence , t49f49_coherence , t50f50_coherence , 
                          t51f51_coherence , t52f52_coherence , t53f53_coherence , t54f54_coherence , t55f55_coherence , t56f56_coherence , t57f57_coherence , t58f58_coherence , t59f59_coherence , t60f60_coherence , 
                          t61f61_coherence , t62f62_coherence , t63f63_coherence , t64f64_coherence , t65f65_coherence , t66f66_coherence , t67f67_coherence , t68f68_coherence , t69f69_coherence , t70f70_coherence , 
                          t71f71_coherence , t72f72_coherence , t73f73_coherence , t74f74_coherence , t75f75_coherence , t76f76_coherence , t77f77_coherence , t78f78_coherence , t79f79_coherence , t80f80_coherence , 
                          t81f81_coherence , t82f82_coherence , t83f83_coherence , t84f84_coherence , t85f85_coherence , t86f86_coherence , t87f87_coherence , t88f88_coherence , t89f89_coherence , t90f90_coherence , 
                          t91f91_coherence , t92f92_coherence , t93f93_coherence , t94f94_coherence , t95f95_coherence , t96f96_coherence , t97f97_coherence , t98f98_coherence , t99f99_coherence , t100f100_coherence
                          ]).T


coherence10_tab = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
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
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
       [3.33066907e-16, 0.00000000e+00, 2.89591255e-04, 2.25642124e-04,
        2.11571491e-04, 1.21502381e-02, 1.34902350e-03, 1.64900295e-04,
        0.00000000e+00, 0.00000000e+00, 2.11611747e-02, 0.00000000e+00,
        2.22044605e-16, 2.22044605e-16, 3.71479082e-02, 3.52258757e-05,
        9.40079141e-02, 0.00000000e+00, 0.00000000e+00, 2.65722079e-02,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.34361779e-04,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.33066907e-16,
        2.25509857e-02, 1.11022302e-16, 9.13410590e-04, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 1.65658677e-03, 0.00000000e+00,
        2.18631749e-02, 2.25790508e-02, 0.00000000e+00, 4.50633747e-05,
        3.33066907e-16, 1.21635128e-02, 1.12225503e-04, 2.94109145e-04,
        8.70217958e-04, 2.38515328e-02, 0.00000000e+00, 0.00000000e+00,
        3.83966781e-05, 2.32906935e-04, 0.00000000e+00, 3.40873980e-02,
        0.00000000e+00, 1.26626879e-05, 2.53157359e-05, 1.00299805e-04,
        3.33066907e-16, 3.33066907e-16, 3.37346684e-04, 5.93477429e-04,
        8.01288698e-02, 7.12160568e-02, 0.00000000e+00, 1.11022302e-16,
        2.01550649e-05, 0.00000000e+00, 4.89895924e-02, 2.07120081e-07,
        4.23983899e-05, 1.11022302e-16, 0.00000000e+00, 2.22044605e-16,
        0.00000000e+00, 2.22044605e-16, 1.26681112e-02, 1.15380158e-02,
        0.00000000e+00, 7.18682601e-04, 1.25764557e-05, 3.71016049e-04,
        2.15496583e-04, 0.00000000e+00, 1.21579417e-02, 0.00000000e+00,
        2.84730439e-09, 0.00000000e+00, 1.97199183e-04, 0.00000000e+00,
        0.00000000e+00, 1.48108618e-04, 1.09136093e-01, 4.69403301e-02,
        0.00000000e+00, 0.00000000e+00, 2.23400305e-04, 0.00000000e+00,
        0.00000000e+00, 1.00868093e-03, 0.00000000e+00, 0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 4.51379189e-03, 3.46972971e-03,
        3.18470847e-03, 4.55313411e-02, 1.83411092e-02, 2.09446440e-03,
        0.00000000e+00, 0.00000000e+00, 4.69440976e-02, 2.22044605e-16,
        2.22044605e-16, 0.00000000e+00, 1.48637968e-01, 4.33626534e-04,
        3.23294832e-01, 0.00000000e+00, 0.00000000e+00, 1.23303847e-01,
        0.00000000e+00, 2.22044605e-16, 0.00000000e+00, 4.42415412e-03,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.22044605e-16,
        6.65609768e-02, 3.33066907e-16, 1.28844347e-02, 0.00000000e+00,
        0.00000000e+00, 1.11022302e-16, 2.49867068e-02, 0.00000000e+00,
        5.87681932e-02, 6.82132279e-02, 0.00000000e+00, 4.69882796e-04,
        0.00000000e+00, 4.57577864e-02, 1.69922108e-03, 4.21892525e-03,
        1.19695123e-02, 8.58489823e-02, 0.00000000e+00, 0.00000000e+00,
        1.41878582e-04, 3.34623182e-03, 0.00000000e+00, 1.04284573e-01,
        0.00000000e+00, 2.00373559e-04, 4.01235637e-04, 1.54310833e-03,
        1.11022302e-16, 3.33066907e-16, 5.10641909e-03, 7.70832906e-03,
        2.53947038e-01, 2.52340457e-01, 0.00000000e+00, 2.22044605e-16,
        1.16433478e-04, 0.00000000e+00, 1.90719540e-01, 1.19383657e-05,
        3.29001946e-04, 2.22044605e-16, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 4.44089210e-16, 5.30714835e-02, 3.72802749e-02,
        0.00000000e+00, 9.10958842e-03, 1.95341056e-04, 5.47973008e-03,
        2.88383217e-03, 0.00000000e+00, 4.65078906e-02, 0.00000000e+00,
        1.75956062e-07, 0.00000000e+00, 1.85081117e-03, 0.00000000e+00,
        0.00000000e+00, 2.17081833e-03, 4.12070632e-01, 1.61638250e-01,
        0.00000000e+00, 0.00000000e+00, 3.33777616e-03, 0.00000000e+00,
        0.00000000e+00, 1.30661915e-02, 0.00000000e+00, 0.00000000e+00],
       [1.11022302e-16, 0.00000000e+00, 2.18769883e-02, 1.64226344e-02,
        1.45652846e-02, 8.72931516e-02, 6.98256154e-02, 7.10363909e-03,
        0.00000000e+00, 0.00000000e+00, 4.77597884e-03, 2.22044605e-16,
        5.55111512e-16, 4.44089210e-16, 3.26536668e-01, 1.22738203e-03,
        5.59460591e-01, 0.00000000e+00, 0.00000000e+00, 3.07045251e-01,
        0.00000000e+00, 2.22044605e-16, 0.00000000e+00, 1.60085764e-02,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.33066907e-16,
        7.50901062e-02, 1.11022302e-16, 5.28871886e-02, 0.00000000e+00,
        0.00000000e+00, 4.44089210e-16, 1.14495825e-01, 0.00000000e+00,
        5.74523405e-02, 9.14355905e-02, 0.00000000e+00, 6.34273516e-04,
        4.44089210e-16, 8.83628748e-02, 7.84879572e-03, 1.76901226e-02,
        4.67042447e-02, 1.60983356e-01, 0.00000000e+00, 0.00000000e+00,
        1.63239593e-03, 1.41053110e-02, 0.00000000e+00, 1.36493668e-01,
        0.00000000e+00, 9.89840747e-04, 2.00426965e-03, 7.31170896e-03,
        3.33066907e-16, 4.44089210e-16, 2.31264912e-02, 2.70094369e-02,
        3.78736218e-01, 4.58340698e-01, 0.00000000e+00, 4.44089210e-16,
        5.22920338e-04, 0.00000000e+00, 4.06460597e-01, 1.13191110e-04,
        4.95618431e-04, 1.11022302e-16, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 2.22044605e-16, 1.19285350e-01, 5.61284723e-02,
        4.44089210e-16, 3.03750018e-02, 9.41110308e-04, 2.42745563e-02,
        1.07899407e-02, 0.00000000e+00, 9.75716255e-02, 0.00000000e+00,
        5.76402930e-06, 0.00000000e+00, 1.29989253e-03, 0.00000000e+00,
        0.00000000e+00, 9.43749335e-03, 8.31704472e-01, 2.85326877e-01,
        2.22044605e-16, 0.00000000e+00, 1.51165379e-02, 0.00000000e+00,
        0.00000000e+00, 4.51628255e-02, 0.00000000e+00, 1.11022302e-16],
       [0.00000000e+00, 0.00000000e+00, 6.50523347e-02, 4.71268890e-02,
        3.99490489e-02, 1.12517133e-01, 1.42845648e-01, 1.24085284e-02,
        0.00000000e+00, 0.00000000e+00, 1.89470295e-01, 0.00000000e+00,
        2.22044605e-16, 0.00000000e+00, 5.39045521e-01, 3.61835893e-04,
        5.95706823e-01, 0.00000000e+00, 0.00000000e+00, 5.30384059e-01,
        0.00000000e+00, 2.22044605e-16, 0.00000000e+00, 2.99331663e-02,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.22044605e-16,
        7.60028138e-03, 8.88178420e-16, 1.24778157e-01, 0.00000000e+00,
        0.00000000e+00, 1.11022302e-16, 3.13804897e-01, 0.00000000e+00,
        4.49110882e-03, 6.74924238e-02, 0.00000000e+00, 3.61440299e-03,
        3.33066907e-16, 1.14580096e-01, 2.18311666e-02, 4.22798340e-02,
        1.00482421e-01, 2.18382534e-01, 0.00000000e+00, 3.33066907e-16,
        8.51605615e-03, 3.42310947e-02, 0.00000000e+00, 6.55286698e-02,
        0.00000000e+00, 2.98773858e-03, 6.24197056e-03, 2.10241720e-02,
        8.88178420e-16, 2.22044605e-16, 6.02977052e-02, 4.82293985e-02,
        2.73823527e-01, 5.84828178e-01, 0.00000000e+00, 1.11022302e-16,
        3.72256345e-03, 0.00000000e+00, 6.59610389e-01, 4.80488942e-04,
        7.11169780e-03, 4.44089210e-16, 0.00000000e+00, 4.44089210e-16,
        0.00000000e+00, 3.33066907e-16, 1.87718554e-01, 4.77984183e-02,
        2.22044605e-16, 4.90961761e-02, 2.77448575e-03, 6.36381551e-02,
        2.20711473e-02, 0.00000000e+00, 1.58773992e-01, 0.00000000e+00,
        3.74816025e-05, 0.00000000e+00, 1.68303574e-02, 0.00000000e+00,
        0.00000000e+00, 2.37334977e-02, 1.23823185e+00, 3.64811641e-01,
        2.22044605e-16, 0.00000000e+00, 4.11096546e-02, 0.00000000e+00,
        0.00000000e+00, 7.63425361e-02, 0.00000000e+00, 4.44089210e-16],
       [0.00000000e+00, 0.00000000e+00, 1.46819801e-01, 1.01200278e-01,
        8.13733887e-02, 9.78995200e-02, 1.81967876e-01, 1.36684244e-02,
        0.00000000e+00, 0.00000000e+00, 4.75824700e-01, 4.44089210e-16,
        4.44089210e-16, 0.00000000e+00, 7.25172086e-01, 7.43063535e-03,
        6.99902377e-02, 0.00000000e+00, 0.00000000e+00, 4.33842075e-01,
        0.00000000e+00, 4.44089210e-16, 0.00000000e+00, 3.15160532e-02,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.22044605e-16,
        1.99595662e-01, 1.22124533e-15, 2.09939281e-01, 0.00000000e+00,
        0.00000000e+00, 2.22044605e-16, 6.34330504e-01, 0.00000000e+00,
        4.63602856e-02, 1.99198723e-02, 0.00000000e+00, 6.44876098e-03,
        0.00000000e+00, 9.62541224e-02, 4.52682092e-02, 6.96765075e-02,
        1.43440268e-01, 2.33210936e-01, 0.00000000e+00, 3.33066907e-16,
        1.68586890e-02, 5.87967866e-02, 0.00000000e+00, 1.19115745e-01,
        0.00000000e+00, 6.75034396e-03, 1.50231529e-02, 4.52974614e-02,
        2.22044605e-16, 2.22044605e-16, 1.08247789e-01, 4.84436081e-02,
        4.16606194e-02, 5.26035877e-01, 0.00000000e+00, 4.44089210e-16,
        9.02204511e-03, 0.00000000e+00, 8.99491337e-01, 1.20671341e-03,
        2.34819276e-02, 3.33066907e-16, 0.00000000e+00, 1.11022302e-16,
        0.00000000e+00, 8.88178420e-16, 2.08900785e-01, 5.02541451e-03,
        4.44089210e-16, 3.80877188e-02, 6.19207362e-03, 1.22214027e-01,
        2.99127508e-02, 0.00000000e+00, 2.24986275e-01, 0.00000000e+00,
        8.38039423e-05, 0.00000000e+00, 6.61582301e-02, 0.00000000e+00,
        0.00000000e+00, 4.17203087e-02, 1.47468376e+00, 3.14787189e-01,
        1.11022302e-16, 0.00000000e+00, 8.34729447e-02, 0.00000000e+00,
        0.00000000e+00, 5.88829608e-02, 0.00000000e+00, 2.22044605e-16],
       [2.22044605e-16, 0.00000000e+00, 2.76399799e-01, 3.90149615e-02,
        1.35570227e-01, 5.01878618e-02, 1.27554524e-01, 1.19896102e-02,
        0.00000000e+00, 0.00000000e+00, 7.26227517e-01, 2.22044605e-16,
        3.33066907e-16, 2.22044605e-16, 8.13246255e-01, 2.95138566e-02,
        4.08548837e-01, 0.00000000e+00, 0.00000000e+00, 1.55696951e-01,
        0.00000000e+00, 2.22044605e-16, 0.00000000e+00, 8.34740008e-03,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.33066907e-16,
        4.62972101e-01, 1.11022302e-15, 2.27708097e-02, 0.00000000e+00,
        0.00000000e+00, 2.22044605e-16, 1.03428810e+00, 0.00000000e+00,
        1.91047732e-02, 4.62739694e-03, 0.00000000e+00, 5.24298391e-02,
        3.33066907e-16, 2.66252977e-02, 7.69753475e-02, 8.29772117e-02,
        1.41367167e-01, 1.96496605e-01, 0.00000000e+00, 5.55111512e-16,
        1.30755322e-02, 7.84313818e-02, 0.00000000e+00, 3.60112454e-01,
        0.00000000e+00, 1.24032001e-02, 3.07173384e-02, 8.01546249e-02,
        0.00000000e+00, 3.33066907e-16, 1.39501224e-01, 1.72460455e-02,
        4.65735340e-02, 9.53898983e-02, 0.00000000e+00, 1.11022302e-16,
        1.08750367e-02, 0.00000000e+00, 1.07813417e+00, 1.84075586e-03,
        4.43954743e-02, 2.22044605e-16, 0.00000000e+00, 1.11022302e-16,
        0.00000000e+00, 9.99200722e-16, 1.29862594e-01, 6.30529025e-02,
        0.00000000e+00, 5.59281742e-03, 1.15004416e-02, 1.89253478e-01,
        2.77978875e-02, 0.00000000e+00, 2.57987136e-01, 0.00000000e+00,
        7.08557238e-05, 0.00000000e+00, 1.31023665e-01, 0.00000000e+00,
        0.00000000e+00, 3.98682850e-02, 1.41726835e+00, 1.56209548e-01,
        1.11022302e-16, 0.00000000e+00, 1.39758251e-01, 0.00000000e+00,
        0.00000000e+00, 3.44468872e-02, 0.00000000e+00, 2.22044605e-16],
       [1.11022302e-16, 0.00000000e+00, 4.56116063e-01, 1.84509862e-01,
        1.27262087e-01, 2.45059508e-02, 2.38396785e-02, 6.88267314e-03,
        0.00000000e+00, 0.00000000e+00, 4.92296558e-01, 3.33066907e-16,
        1.11022302e-16, 2.22044605e-16, 5.19111122e-01, 7.17062484e-02,
        2.22158595e-01, 0.00000000e+00, 0.00000000e+00, 1.18868700e-01,
        0.00000000e+00, 2.22044605e-16, 0.00000000e+00, 3.66305125e-02,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.66133815e-16,
        5.52529392e-01, 2.22044605e-16, 2.85422419e-01, 0.00000000e+00,
        0.00000000e+00, 1.11022302e-16, 1.41897584e+00, 0.00000000e+00,
        1.03899499e-01, 6.05437936e-02, 0.00000000e+00, 9.73142815e-02,
        5.55111512e-16, 6.10878619e-02, 1.12909049e-01, 6.57300151e-02,
        8.86768339e-02, 1.19357048e-01, 0.00000000e+00, 4.44089210e-16,
        1.04592130e-02, 8.74637742e-02, 0.00000000e+00, 4.78485276e-01,
        0.00000000e+00, 1.92017109e-02, 5.59679859e-02, 1.21974923e-01,
        6.66133815e-16, 7.77156117e-16, 1.20953881e-01, 2.41458878e-02,
        5.20107907e-03, 3.78507564e-02, 0.00000000e+00, 0.00000000e+00,
        2.99210990e-03, 0.00000000e+00, 1.17333207e+00, 8.90034197e-04,
        5.28132227e-02, 1.11022302e-16, 0.00000000e+00, 2.22044605e-16,
        0.00000000e+00, 7.77156117e-16, 6.55098782e-02, 1.38826317e-01,
        6.66133815e-16, 4.13629954e-02, 1.86944552e-02, 2.49087952e-01,
        1.42482884e-02, 0.00000000e+00, 1.15135955e-01, 0.00000000e+00,
        9.04737104e-04, 0.00000000e+00, 1.59860486e-01, 0.00000000e+00,
        0.00000000e+00, 4.37423482e-02, 8.50947387e-01, 1.75626989e-01,
        1.11022302e-16, 0.00000000e+00, 2.03293266e-01, 0.00000000e+00,
        0.00000000e+00, 1.62396308e-01, 0.00000000e+00, 1.11022302e-16],
       [1.11022302e-16, 0.00000000e+00, 6.74740687e-01, 3.57582868e-01,
        2.50548432e-01, 1.02938306e-01, 1.36774905e-01, 2.41409576e-02,
        0.00000000e+00, 0.00000000e+00, 1.02519490e-01, 2.22044605e-16,
        3.33066907e-16, 2.22044605e-16, 1.69944369e-01, 1.33557958e-01,
        1.81916583e-01, 0.00000000e+00, 0.00000000e+00, 1.86359366e-01,
        0.00000000e+00, 2.22044605e-16, 0.00000000e+00, 8.50558811e-02,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.11022302e-16,
        5.01775379e-01, 2.22044605e-16, 2.07647983e-01, 0.00000000e+00,
        0.00000000e+00, 2.22044605e-16, 1.54039585e+00, 0.00000000e+00,
        2.44114384e-01, 3.48864197e-02, 0.00000000e+00, 1.35660680e-01,
        5.55111512e-16, 1.04306186e-01, 1.47053651e-01, 1.41327628e-02,
        2.15493979e-02, 3.00783335e-02, 0.00000000e+00, 2.22044605e-16,
        3.54289002e-02, 9.33716641e-02, 0.00000000e+00, 5.55775942e-01,
        0.00000000e+00, 2.52409009e-02, 9.31663092e-02, 1.63392623e-01,
        3.33066907e-16, 4.44089210e-16, 4.38684756e-02, 3.71453617e-02,
        6.44630367e-02, 4.48074487e-01, 0.00000000e+00, 2.22044605e-16,
        1.21493068e-02, 0.00000000e+00, 1.20217766e+00, 4.23163316e-03,
        1.22939487e-02, 2.22044605e-16, 0.00000000e+00, 1.11022302e-16,
        0.00000000e+00, 2.22044605e-16, 3.25868353e-01, 2.10565484e-01,
        6.66133815e-16, 2.13298037e-03, 2.74087613e-02, 2.87879199e-01,
        3.20565920e-03, 0.00000000e+00, 5.66043327e-02, 0.00000000e+00,
        2.56165844e-03, 0.00000000e+00, 1.02896163e-01, 0.00000000e+00,
        0.00000000e+00, 1.76252689e-03, 3.64356955e-01, 2.40183707e-02,
        1.11022302e-16, 0.00000000e+00, 2.63651163e-01, 0.00000000e+00,
        0.00000000e+00, 2.24535537e-01, 0.00000000e+00, 8.88178420e-16],
       [1.11022302e-16, 0.00000000e+00, 6.98197172e-01, 4.21181082e-01,
        2.95915956e-01, 3.30923916e-01, 1.81663547e-01, 2.91118601e-02,
        0.00000000e+00, 0.00000000e+00, 3.07340132e-01, 2.22044605e-16,
        3.33066907e-16, 1.22124533e-15, 3.03711290e-01, 2.05177167e-01,
        3.39014794e-01, 0.00000000e+00, 0.00000000e+00, 1.75228002e-01,
        0.00000000e+00, 4.44089210e-16, 0.00000000e+00, 1.17147515e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        3.98216517e-01, 2.22044605e-16, 1.80944497e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 1.02020819e+00, 0.00000000e+00,
        2.71825104e-01, 1.63418055e-01, 0.00000000e+00, 1.40554064e-01,
        9.99200722e-16, 5.35322779e-02, 1.72660677e-01, 5.54983700e-02,
        6.24099278e-03, 3.51763958e-02, 0.00000000e+00, 1.11022302e-16,
        2.77002770e-02, 1.20389768e-01, 0.00000000e+00, 4.51569458e-01,
        0.00000000e+00, 2.75702674e-02, 1.43483662e-01, 1.94347154e-01,
        7.77156117e-16, 8.88178420e-16, 5.27389774e-02, 4.82866167e-03,
        2.12102792e-01, 7.87448631e-01, 0.00000000e+00, 0.00000000e+00,
        2.08325445e-02, 0.00000000e+00, 1.21134186e+00, 1.63958758e-02,
        1.67062173e-02, 8.88178420e-16, 0.00000000e+00, 8.88178420e-16,
        0.00000000e+00, 8.88178420e-16, 5.39059354e-01, 2.83585618e-01,
        2.22044605e-16, 6.32812913e-02, 3.69565423e-02, 2.97834657e-01,
        2.53099635e-02, 0.00000000e+00, 2.04629676e-01, 0.00000000e+00,
        3.60460499e-03, 0.00000000e+00, 3.11750888e-02, 0.00000000e+00,
        0.00000000e+00, 9.05679737e-02, 2.73574237e-02, 7.23196674e-02,
        0.00000000e+00, 0.00000000e+00, 3.06536782e-01, 0.00000000e+00,
        0.00000000e+00, 1.37854458e-01, 0.00000000e+00, 3.33066907e-16]])



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
                           pop41 , pop42 , pop43 , pop44 , pop45 , pop46 , pop47 , pop48 , pop49 , pop50 , 
                           pop51 , pop52 , pop53 , pop54 , pop55 , pop56 , pop57 , pop58 , pop59 , pop60 , 
                           pop61 , pop62 , pop63 , pop64 , pop65 , pop66 , pop67 , pop68 , pop69 , pop70 , 
                           pop71 , pop72 , pop73 , pop74 , pop75 , pop76 , pop77 , pop78 , pop79 , pop80 , 
                           pop81 , pop82 , pop83 , pop84 , pop85 , pop86 , pop87 , pop88 , pop89 , pop90 , 
                           pop91 , pop92 , pop93 , pop94 , pop95 , pop96 , pop97 , pop98 , pop99 , pop100 , ]).T



population10_tab = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
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
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
       [1.92213753e-10, 3.46944695e-18, 2.58163290e-05, 2.52877503e-05,
        2.56382654e-05, 6.80618908e-02, 2.22176418e-02, 3.58292862e-04,
        0.00000000e+00, 0.00000000e+00, 2.35930750e-02, 2.40000982e-10,
        3.87884203e-06, 1.38777878e-17, 2.37127392e-02, 7.29293376e-07,
        2.33441956e-02, 0.00000000e+00, 0.00000000e+00, 3.62064650e-04,
        0.00000000e+00, 9.36379704e-05, 0.00000000e+00, 9.87105598e-04,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.52790859e-04,
        1.24667351e-04, 1.16484227e-04, 5.51048859e-04, 0.00000000e+00,
        0.00000000e+00, 4.66003516e-04, 1.74562415e-04, 0.00000000e+00,
        2.25287831e-02, 2.30043865e-02, 0.00000000e+00, 1.71480963e-04,
        1.73535044e-05, 2.29218798e-02, 2.86732671e-07, 1.70773275e-04,
        4.76289675e-04, 5.04089113e-05, 0.00000000e+00, 4.42732076e-04,
        3.69081936e-04, 2.47655196e-04, 0.00000000e+00, 4.52507721e-02,
        0.00000000e+00, 6.24452029e-08, 7.40345011e-05, 5.35288213e-07,
        2.58852802e-04, 2.43949394e-05, 2.26019138e-02, 3.55487465e-04,
        2.27106526e-02, 3.47806792e-04, 0.00000000e+00, 1.21513537e-04,
        1.86676415e-04, 0.00000000e+00, 2.34669937e-02, 9.84918005e-05,
        2.30908025e-02, 1.15862175e-04, 0.00000000e+00, 2.17628675e-02,
        0.00000000e+00, 4.43804341e-04, 2.66721068e-04, 9.94856096e-05,
        1.90819582e-17, 2.19772182e-02, 2.26445337e-10, 2.94944402e-04,
        4.73798930e-04, 0.00000000e+00, 9.90742302e-05, 0.00000000e+00,
        3.05287334e-04, 0.00000000e+00, 1.20881381e-03, 0.00000000e+00,
        0.00000000e+00, 5.66792860e-07, 2.43177176e-02, 2.01220799e-03,
        1.02103591e-07, 0.00000000e+00, 4.82014270e-04, 0.00000000e+00,
        0.00000000e+00, 5.05283157e-04, 0.00000000e+00, 1.73472348e-18],
       [4.81652936e-08, 0.00000000e+00, 4.32070591e-04, 3.99108593e-04,
        4.20614813e-04, 2.08480732e-01, 6.32811204e-02, 4.74367101e-03,
        0.00000000e+00, 0.00000000e+00, 8.20902931e-02, 5.99392049e-08,
        2.02658114e-04, 6.93889390e-17, 8.36773226e-02, 4.42705852e-05,
        7.83517268e-02, 0.00000000e+00, 0.00000000e+00, 4.95662770e-03,
        2.08166817e-17, 1.16290768e-03, 0.00000000e+00, 1.33967329e-02,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.44746446e-03,
        1.88525212e-03, 1.42550104e-03, 7.35682386e-03, 0.00000000e+00,
        0.00000000e+00, 5.69823681e-03, 2.64170226e-03, 0.00000000e+00,
        6.74187386e-02, 7.36243786e-02, 0.00000000e+00, 2.45652742e-03,
        3.36468072e-05, 7.20604882e-02, 1.72380043e-05, 2.41377953e-03,
        6.23797356e-03, 7.85806251e-04, 0.00000000e+00, 5.41427581e-03,
        4.36773363e-03, 3.66948345e-03, 0.00000000e+00, 1.37493191e-01,
        0.00000000e+00, 3.94266310e-06, 1.08434655e-03, 3.27060045e-05,
        3.26264908e-03, 3.50400539e-04, 6.84535708e-02, 4.58751198e-03,
        6.94377311e-02, 5.20091125e-03, 0.00000000e+00, 1.69224083e-03,
        2.29413177e-03, 0.00000000e+00, 8.01198672e-02, 1.43332445e-03,
        7.43833532e-02, 1.38454065e-03, 0.00000000e+00, 5.72149531e-02,
        2.08166817e-17, 5.47286568e-03, 3.69511734e-03, 1.49131185e-03,
        3.46944695e-17, 6.00269585e-02, 5.66746414e-08, 4.26828053e-03,
        6.11468466e-03, 0.00000000e+00, 1.46811850e-03, 0.00000000e+00,
        3.82064674e-03, 0.00000000e+00, 1.56118036e-02, 0.00000000e+00,
        0.00000000e+00, 3.47467328e-05, 9.20165520e-02, 2.58803185e-02,
        6.08100170e-06, 0.00000000e+00, 6.56056066e-03, 0.00000000e+00,
        0.00000000e+00, 6.84161669e-03, 0.00000000e+00, 3.46944695e-17],
       [1.19111961e-06, 0.00000000e+00, 2.33370764e-03, 1.97436280e-03,
        2.20236110e-03, 2.84283359e-01, 7.22949590e-02, 1.72045425e-02,
        0.00000000e+00, 0.00000000e+00, 1.44811387e-01, 1.47396621e-06,
        1.59498135e-03, 6.93889390e-17, 1.50472104e-01, 4.61435557e-04,
        1.27871243e-01, 0.00000000e+00, 0.00000000e+00, 1.91617109e-02,
        0.00000000e+00, 3.73773978e-03, 0.00000000e+00, 5.08495327e-02,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.47024330e-02,
        8.67280513e-03, 4.48759889e-03, 2.69662671e-02, 0.00000000e+00,
        0.00000000e+00, 1.77380050e-02, 1.21779265e-02, 0.00000000e+00,
        8.72989297e-02, 1.09149499e-01, 0.00000000e+00, 1.02803781e-02,
        1.77531362e-03, 1.00004905e-01, 1.76461395e-04, 9.83911974e-03,
        2.20374947e-02, 3.77873832e-03, 0.00000000e+00, 1.68290249e-02,
        1.27225600e-02, 1.63091889e-02, 0.00000000e+00, 1.85028943e-01,
        0.00000000e+00, 4.38677065e-05, 4.71031459e-03, 3.44628682e-04,
        1.06867499e-02, 1.55280324e-03, 9.15768177e-02, 1.58022903e-02,
        9.19325115e-02, 2.34521212e-02, 0.00000000e+00, 6.64627724e-03,
        7.21973430e-03, 0.00000000e+00, 1.35262522e-01, 6.16690555e-03,
        1.09244372e-01, 3.99690054e-03, 0.00000000e+00, 4.99019662e-02,
        0.00000000e+00, 1.73354162e-02, 1.44628010e-02, 6.73046832e-03,
        6.93889390e-17, 5.99244124e-02, 1.39880903e-06, 1.81787129e-02,
        2.11994048e-02, 0.00000000e+00, 6.51965557e-03, 0.00000000e+00,
        1.23308431e-02, 0.00000000e+00, 5.39422027e-02, 0.00000000e+00,
        0.00000000e+00, 3.68225910e-04, 1.82975856e-01, 8.86455812e-02,
        6.11559175e-05, 0.00000000e+00, 2.49902440e-02, 0.00000000e+00,
        0.00000000e+00, 2.58092385e-02, 0.00000000e+00, 1.87350135e-16],
       [1.13161735e-05, 0.00000000e+00, 7.93324184e-03, 6.03679647e-03,
        7.19160504e-03, 2.03104475e-01, 3.90166051e-02, 3.24953312e-02,
        0.00000000e+00, 0.00000000e+00, 1.77996403e-01, 1.38917134e-05,
        4.82285562e-03, 5.89805982e-17, 1.87632954e-01, 2.28660048e-03,
        1.33302894e-01, 0.00000000e+00, 0.00000000e+00, 4.05392985e-02,
        0.00000000e+00, 5.70849868e-03, 0.00000000e+00, 1.04505026e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.34836314e-02,
        2.38833331e-02, 6.99148290e-03, 5.59113490e-02, 0.00000000e+00,
        0.00000000e+00, 2.54113703e-02, 3.36988420e-02, 0.00000000e+00,
        5.44584551e-02, 9.21422384e-02, 0.00000000e+00, 2.45204362e-02,
        8.09970134e-03, 6.18409623e-02, 8.48716464e-04, 2.23706666e-02,
        3.93895486e-02, 1.09417809e-02, 0.00000000e+00, 2.36630381e-02,
        1.58773030e-02, 4.27492723e-02, 0.00000000e+00, 1.36794122e-01,
        0.00000000e+00, 2.38091915e-04, 1.18457410e-02, 1.73411558e-03,
        1.61859096e-02, 4.45465311e-03, 6.34878867e-02, 2.69144591e-02,
        5.28853356e-02, 6.26061807e-02, 0.00000000e+00, 1.38561092e-02,
        1.05938813e-02, 0.00000000e+00, 1.49856702e-01, 1.53482511e-02,
        8.34145185e-02, 4.05846617e-03, 0.00000000e+00, 1.83061681e-02,
        0.00000000e+00, 2.55354573e-02, 3.03401428e-02, 1.78554961e-02,
        1.14491749e-16, 5.45187029e-02, 1.32543321e-05, 4.44953312e-02,
        3.74733287e-02, 0.00000000e+00, 1.70256932e-02, 0.00000000e+00,
        1.82575847e-02, 0.00000000e+00, 9.28691468e-02, 0.00000000e+00,
        0.00000000e+00, 1.86826023e-03, 2.56434606e-01, 1.49876581e-01,
        2.85378780e-04, 0.00000000e+00, 5.09277772e-02, 0.00000000e+00,
        0.00000000e+00, 5.18320719e-02, 0.00000000e+00, 8.32667268e-17],
       [6.32271396e-05, 0.00000000e+00, 2.26846503e-02, 1.41014860e-02,
        1.79340358e-02, 1.32912895e-01, 1.37923510e-01, 3.60670245e-02,
        0.00000000e+00, 0.00000000e+00, 1.62549116e-01, 7.68062246e-05,
        5.33329424e-03, 1.61762964e-16, 1.67378896e-01, 7.40461780e-03,
        8.36654028e-02, 0.00000000e+00, 0.00000000e+00, 6.09895568e-02,
        0.00000000e+00, 4.16176858e-03, 0.00000000e+00, 1.45586946e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.77310786e-02,
        4.87452378e-02, 7.90503202e-03, 7.84668864e-02, 0.00000000e+00,
        0.00000000e+00, 1.60392525e-02, 6.91413728e-02, 0.00000000e+00,
        1.24596213e-01, 6.25611175e-02, 0.00000000e+00, 4.03500400e-02,
        1.72096451e-02, 1.29337072e-01, 2.61931632e-03, 3.35834981e-02,
        4.44230304e-02, 2.32880632e-02, 0.00000000e+00, 1.16359743e-02,
        7.56445209e-03, 8.12973737e-02, 0.00000000e+00, 1.31232372e-01,
        0.00000000e+00, 8.66215805e-04, 2.08938829e-02, 5.72748076e-03,
        8.13906759e-03, 1.06240133e-02, 6.65323264e-02, 2.82804587e-02,
        8.71977006e-02, 1.21408486e-01, 0.00000000e+00, 1.66732375e-02,
        7.13311975e-03, 0.00000000e+00, 1.00888115e-01, 2.69110506e-02,
        1.30137286e-01, 3.82950697e-03, 0.00000000e+00, 9.82398457e-02,
        0.00000000e+00, 1.55008554e-02, 3.77279182e-02, 3.38296277e-02,
        4.05057932e-16, 1.40999582e-01, 7.38196165e-05, 7.58391738e-02,
        3.87773893e-02, 0.00000000e+00, 3.20375790e-02, 0.00000000e+00,
        2.72425434e-02, 0.00000000e+00, 9.34635342e-02, 0.00000000e+00,
        0.00000000e+00, 6.23987486e-03, 2.78185887e-01, 1.26363585e-01,
        8.36364201e-04, 0.00000000e+00, 6.33650398e-02, 0.00000000e+00,
        0.00000000e+00, 6.68617510e-02, 0.00000000e+00, 7.97972799e-17],
       [2.51129059e-04, 0.00000000e+00, 5.63736136e-02, 2.76231749e-02,
        3.71812314e-02, 3.25167672e-01, 1.73868096e-01, 2.92885108e-02,
        0.00000000e+00, 0.00000000e+00, 1.58919132e-01, 3.01066963e-04,
        9.39474249e-03, 2.35868183e-16, 2.07549747e-01, 1.80334325e-02,
        8.69514725e-02, 0.00000000e+00, 0.00000000e+00, 7.77396136e-02,
        0.00000000e+00, 2.87460869e-04, 0.00000000e+00, 1.33486276e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.44915082e-03,
        8.13945969e-02, 2.42458719e-02, 1.01573922e-01, 0.00000000e+00,
        0.00000000e+00, 1.76105021e-03, 1.21455317e-01, 0.00000000e+00,
        1.37313580e-01, 1.29058247e-01, 0.00000000e+00, 4.80340956e-02,
        1.80533188e-02, 2.32503027e-01, 5.89721683e-03, 3.37000813e-02,
        4.96822419e-02, 4.66695621e-02, 0.00000000e+00, 1.39953800e-02,
        1.05515368e-02, 1.22199597e-01, 0.00000000e+00, 1.80416393e-01,
        0.00000000e+00, 2.43078388e-03, 2.71324940e-02, 1.42890467e-02,
        1.71571489e-02, 2.25000933e-02, 1.29526357e-01, 3.83508386e-02,
        1.57668738e-01, 1.85416325e-01, 0.00000000e+00, 5.51045305e-03,
        5.47848766e-03, 0.00000000e+00, 1.38402566e-01, 3.53945004e-02,
        1.99674791e-01, 1.93624117e-02, 0.00000000e+00, 1.12794015e-01,
        0.00000000e+00, 9.58397980e-03, 3.61939292e-02, 5.11436104e-02,
        1.34359754e-16, 1.62820520e-01, 2.92141267e-04, 9.87792491e-02,
        3.11820943e-02, 0.00000000e+00, 4.69188435e-02, 0.00000000e+00,
        4.27122560e-02, 0.00000000e+00, 7.84014970e-02, 0.00000000e+00,
        0.00000000e+00, 1.57946232e-02, 3.24971266e-01, 7.03765460e-02,
        1.71258673e-03, 0.00000000e+00, 5.95669264e-02, 0.00000000e+00,
        0.00000000e+00, 7.66042135e-02, 0.00000000e+00, 2.57579331e-16],
       [7.84418451e-04, 0.00000000e+00, 1.13841582e-01, 4.76177504e-02,
        6.67898518e-02, 3.85339340e-01, 1.06272888e-01, 7.10646100e-02,
        0.00000000e+00, 0.00000000e+00, 2.21788768e-01, 9.25389220e-04,
        5.15148448e-02, 8.84708973e-17, 2.26605405e-01, 3.59815434e-02,
        1.86159031e-01, 0.00000000e+00, 0.00000000e+00, 1.27582427e-01,
        0.00000000e+00, 4.97294239e-04, 0.00000000e+00, 1.42621488e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.67021370e-02,
        1.16436413e-01, 1.00854474e-01, 1.19268459e-01, 0.00000000e+00,
        0.00000000e+00, 2.11993868e-02, 1.93695717e-01, 0.00000000e+00,
        1.15499734e-01, 1.97499845e-01, 0.00000000e+00, 5.38494812e-02,
        1.46220817e-03, 3.03778169e-01, 1.04684782e-02, 4.69029998e-02,
        1.39602421e-01, 1.01045214e-01, 0.00000000e+00, 2.59918541e-02,
        3.56410574e-02, 1.57139066e-01, 0.00000000e+00, 1.87792503e-01,
        0.00000000e+00, 5.66386659e-03, 2.38449185e-02, 2.89800275e-02,
        4.69026225e-02, 4.15216677e-02, 1.63381995e-01, 5.28187740e-02,
        2.05301305e-01, 2.54580702e-01, 0.00000000e+00, 2.42693783e-02,
        1.21004840e-02, 0.00000000e+00, 2.49019443e-01, 3.39287580e-02,
        2.13395149e-01, 3.38332547e-02, 0.00000000e+00, 2.82063240e-02,
        0.00000000e+00, 2.51669902e-02, 1.03059330e-01, 9.00781041e-02,
        1.30104261e-16, 1.14948876e-01, 9.09007489e-04, 9.64958227e-02,
        7.36447658e-02, 0.00000000e+00, 6.50653972e-02, 0.00000000e+00,
        7.22951633e-02, 0.00000000e+00, 1.81081375e-01, 0.00000000e+00,
        0.00000000e+00, 3.26267398e-02, 2.84237065e-01, 2.26205036e-01,
        2.40202548e-03, 0.00000000e+00, 7.83481665e-02, 0.00000000e+00,
        0.00000000e+00, 1.15296536e-01, 0.00000000e+00, 2.16840434e-16],
       [2.04639238e-03, 0.00000000e+00, 1.94863893e-01, 7.42161150e-02,
        1.07103441e-01, 4.17030133e-01, 2.20637671e-01, 1.01627542e-01,
        0.00000000e+00, 0.00000000e+00, 3.12747233e-01, 2.36802358e-03,
        1.15438376e-01, 1.24900090e-16, 3.47619335e-01, 6.11636938e-02,
        3.20553744e-01, 0.00000000e+00, 0.00000000e+00, 1.86786516e-01,
        2.22044605e-16, 1.00036658e-02, 0.00000000e+00, 2.06546658e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.02339425e-02,
        1.47040240e-01, 2.11519741e-01, 2.00544805e-01, 0.00000000e+00,
        0.00000000e+00, 9.82485572e-02, 2.71283476e-01, 0.00000000e+00,
        1.75244909e-01, 2.32663077e-01, 0.00000000e+00, 7.99476312e-02,
        3.72421750e-02, 3.07879482e-01, 1.71215661e-02, 8.74669641e-02,
        2.80614541e-01, 1.78477941e-01, 0.00000000e+00, 5.42664443e-03,
        9.24106460e-02, 1.78112462e-01, 0.00000000e+00, 3.28500363e-01,
        0.00000000e+00, 1.14375918e-02, 3.43673655e-02, 4.98373083e-02,
        6.31057080e-02, 6.46020866e-02, 2.77614017e-01, 1.06441699e-01,
        1.94208341e-01, 3.40140531e-01, 0.00000000e+00, 6.47701781e-02,
        4.10180795e-02, 0.00000000e+00, 3.64127778e-01, 1.76897009e-02,
        1.38890702e-01, 3.72550703e-02, 0.00000000e+00, 1.11863173e-01,
        2.22044605e-16, 1.22004980e-02, 2.48227531e-01, 1.81756210e-01,
        8.39606162e-16, 1.24175423e-01, 2.36223340e-03, 1.91682965e-01,
        1.43943853e-01, 0.00000000e+00, 1.20006769e-01, 0.00000000e+00,
        1.40600965e-01, 0.00000000e+00, 2.74179204e-01, 0.00000000e+00,
        0.00000000e+00, 5.74049397e-02, 4.11016710e-01, 3.41882563e-01,
        1.49402310e-03, 0.00000000e+00, 1.71288495e-01, 0.00000000e+00,
        0.00000000e+00, 2.25886646e-01, 0.00000000e+00, 2.70616862e-16],
       [4.63469672e-03, 0.00000000e+00, 2.92979050e-01, 1.06536175e-01,
        1.56500192e-01, 6.14343401e-01, 3.87296390e-01, 1.10796312e-01,
        0.00000000e+00, 0.00000000e+00, 4.50995818e-01, 5.24188236e-03,
        1.69827950e-01, 2.77555756e-16, 5.01636870e-01, 9.10314845e-02,
        4.66719359e-01, 0.00000000e+00, 0.00000000e+00, 3.01439155e-01,
        0.00000000e+00, 2.47171217e-02, 0.00000000e+00, 2.86406321e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.69008221e-02,
        1.76853869e-01, 2.93073392e-01, 3.95036962e-01, 0.00000000e+00,
        0.00000000e+00, 1.97144105e-01, 3.41758380e-01, 0.00000000e+00,
        2.52949721e-01, 4.82350238e-01, 0.00000000e+00, 1.36629174e-01,
        6.45405287e-02, 3.97848982e-01, 2.92932947e-02, 1.37774099e-01,
        4.58952545e-01, 2.68455558e-01, 0.00000000e+00, 2.90478540e-02,
        1.37780023e-01, 2.25195108e-01, 0.00000000e+00, 5.79560814e-01,
        0.00000000e+00, 2.05543352e-02, 8.01755050e-02, 7.45300989e-02,
        6.55395519e-02, 8.25150790e-02, 5.62371480e-01, 1.78158657e-01,
        2.75260339e-01, 3.87026802e-01, 0.00000000e+00, 9.65740323e-02,
        6.95517272e-02, 0.00000000e+00, 4.51927910e-01, 5.66676207e-02,
        2.55006340e-01, 2.85778764e-02, 0.00000000e+00, 2.22405884e-01,
        0.00000000e+00, 1.56956432e-02, 4.26731129e-01, 2.92993887e-01,
        2.70616862e-16, 2.20382416e-01, 5.33053904e-03, 3.25828382e-01,
        2.39345485e-01, 0.00000000e+00, 2.17340021e-01, 0.00000000e+00,
        2.76223145e-01, 0.00000000e+00, 3.87150339e-01, 0.00000000e+00,
        0.00000000e+00, 8.82824567e-02, 5.46402758e-01, 3.74657062e-01,
        3.53895349e-03, 0.00000000e+00, 2.29462327e-01, 0.00000000e+00,
        0.00000000e+00, 3.14049746e-01, 0.00000000e+00, 4.85722573e-16]])



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