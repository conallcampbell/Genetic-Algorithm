# Genetic Algorithm
import numpy as np
import random
import networkx as nx
from scipy.linalg import expm
from scipy.linalg import sqrtm
from tqdm import tqdm
import matplotlib.pyplot as plt
# importing the frozen network and genetic evolution classes
from alt_Frozen_Network_class import alt_Frozen_Network as alt_Frozen_Network_class
from Genetic_Class import Genetic_Evolution

def kron(*matrices):
    result = np.array([[1]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result

##########################################
#number of networks I wish to consider
pop_size = 200
##########################################
##########################################
#network parameters
##########################################
# number of nodes
n = 20
# each node is joined with its m nearest neighbours in a ring topology
m = 2
# probability of edge rewiring
p = 0.01
# probability of edge creation
prob = np.random.random()
# Building a Watts-Strogatz network
def G(n , m , p):
    return nx.watts_strogatz_graph(n , m , p , seed = None)
def F(n , prob):
    return nx.erdos_renyi_graph(n , prob)

##########################################
#target
##########################################
target = F(n , prob)
def adjacency_matrix_target():
    return [np.array(nx.to_numpy_array(target).astype("complex")) for _ in range(n)]
##########################################
#Initial state
##########################################
# we place the excitation at node k = 1 for the target and all individuals

k = 1
rho0 = np.zeros((n , 1))
rho0[k] = 1
state = kron(rho0 , np.conjugate(rho0.T))

##########################################
# Adding the sink to the target list
##########################################
def target_sink():
    mat_list = adjacency_matrix_target()
    for i in range(n):
        mat_list[i][i , i] -= 1j
    return mat_list
target_sink = target_sink()

# mattest = adjacency_matrix_target()[0] - np.array([[0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]])


##########################################
#Unitary Evolution
##########################################
#defining the time step
ts = np.linspace(0 , 5 , 20)
#target
def evolution_target(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
    return [
            
                1-(np.trace
                    ((expm(complex(0 , -1) * target_sink[i] * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * target_sink[i] * t).T))
                  )
                ).real
            
            for i in range(n)
            ]
def target_data():
    data = []
    for t in ts:
        data.append(evolution_target(t))
    return data

# Frozen_Network_Class(mattest , target_data() , ts).calculate_distance()

# Generating the population of networks
def pop(pop_size, n):
    return [nx.to_numpy_array(F(n, prob)).astype("complex") for _ in range(pop_size)]
population = pop(pop_size, n)

# def alt_pop(input_matrix):
#     size = input_matrix.shape[0]
#     new_matrices = []

#     for i in range(size):
#         for j in range(i + 1 , size):

#             new_matrix = input_matrix.copy()

#             new_matrix[i , j] = 1 - new_matrix[i , j]
#             new_matrix[j , i] = new_matrix[i , j]

#             new_matrices.append(new_matrix)
    
#     return new_matrices
# nu_pop = alt_pop(adjacency_matrix_target()[0])
# nu_distances = []
# for i in range(len(nu_pop)):
#     nu_distances.append(alt_Frozen_Network_class(nu_pop[i] , target_data() , ts).calculate_distance())



# def genetic_distance(pop , start_index = 0):
#     dist = []
#     for i in range(start_index , pop_size):
#         dist.append([Frozen_Network_Class(pop[i] , target_data() , ts).calculate_distance() , pop[i]])
#     dist = sorted(dist, key = lambda x: x[0])
#     return [d[0] for d in dist], [d[1] for d in dist]
def alt_genetic_distance(pop , start_index = 0):
    dist = []
    for i in range(start_index , pop_size):
        dist.append([alt_Frozen_Network_class(pop[i] , target_data() , ts).calculate_distance() , pop[i]])
    dist = sorted(dist , key = lambda x: x[0])
    return [d[0] for d in dist] , [d[1] for d in dist]

def fit_half(pop):
    return pop[:pop_size // 2]

# distances = [genetic_distance(population)[0]]
alt_distances = [alt_genetic_distance(population)[0]]
ordered_pop = [alt_genetic_distance(population)[1]]
found_it = []
if alt_distances[0][0] < 10**-3:
    found_it.append(fit_half(ordered_pop)[0])
    print("Found fit individual")
else:
    print("Continue to Genetic Evolution")

# timestep = 0.1
# num_steps = 100
# alt_dist = []
# for i in range(pop_size):
#     network_instance = alt_Frozen_Network_class(population[i])
#     network_instance.calculate_distance(target_data() , timestep , num_steps)
#     alt_dist.append(network_instance.distance)

# def fidelity(matrix1 , matrix2):
#     return (np.trace(sqrtm(sqrtm(matrix1) @ matrix2 @ sqrtm(matrix1))))**2

# fidelity_list = []
# for i in range(pop_size):
#     fidelities = np.real(np.sqrt(fidelity(adjacency_matrix_target()[0] , population[i])*np.conjugate(fidelity(adjacency_matrix_target()[0] , population[i]))))
#     fidelity_list.append(fidelities)
# dist_plot = []
# for i in range(pop_size):
#     dist_plot.append(Frozen_Network_Class(population[i] , target_data() , ts).calculate_distance())
# matrix_diff = []
# for i in range(pop_size):
#     matrix_diff.append(np.sum(abs(adjacency_matrix_target()[0] - population[i])))
# classical_fidelity = []
# for i in range(pop_size):
#     classical_fidelity.append(np.real((np.sum(np.sqrt(adjacency_matrix_target()[0].flatten()*population[i].flatten())))**2))
# trace_distance = []
# for i in range(pop_size):
#     trace_distance.append(0.5 * np.real(np.trace(sqrtm(np.conjugate((adjacency_matrix_target()[0] - population[i]).T)@(adjacency_matrix_target()[0] - population[i])))))


# new_dist = sorted(alt_dist)
# dist = alt_NetworkClass(population).calculate_distance(target_data() , timestep , num_steps)

mutation1 = Genetic_Evolution(fit_half(alt_genetic_distance(population)[1]) , alt_distances[0][:len(alt_distances[0])//2]).mutation1()
mutation2 = Genetic_Evolution(fit_half(alt_genetic_distance(population)[1]) , alt_distances[0][:len(alt_distances[0])//2]).mutation2()
new_pop = Genetic_Evolution(fit_half(alt_genetic_distance(population)[1]) , alt_distances[0][:len(alt_distances[0])//2]).new_population()

distances = []
best_individual = []
for j in tqdm(range(1000)):

    fit_individual = []
    if j == 0:
        new_distances, new_pop = alt_genetic_distance(new_pop)
    else:
        new_distances, new_population = alt_genetic_distance(new_pop , start_index = pop_size // 4)
        new_distances = sorted(distances[-1][:pop_size // 4] + new_distances)
        new_pop = new_pop[:pop_size//4] + new_population

    distances.append(new_distances)
    best_individual.append(new_pop[0])

    
    if min(distances[j]) < 10**-3:
        fit_individual.append(new_pop[0])
        print("fit individual found")
        break
    else:
        new_pop = Genetic_Evolution(fit_half(alt_genetic_distance(new_pop)[1]) , distances[j][:len(distances[j])//2]).new_population()
        print(new_distances[0], np.mean(new_distances))
else:
    print("No fit individual found within iteration limit")

# plotting the distance as a function of the number of times the algorithm is iterated
distance_list = np.array(distances)

dist = []
for i in range(len(distances)):
    dist.append(distances[i][0])
plt.plot(range(len(distances)) , dist)
plt.xlabel('Iteration')
plt.ylabel('Minimum Distance')
plt.title('Minimum Distance vs Iteration of GA')
plt.show()

# plt.scatter(dist_plot , trace_distance)
# plt.xlabel('distance')
# plt.ylabel('trace distance')
# plt.show()

# plt.scatter(dist_plot , classical_fidelity)
# plt.xlabel('distance')
# plt.ylabel('fidelity')
# plt.show()


# for i in range(pop_size):
#     plt.plot(range(len(distance_list)) , distance_list[: , i])        
# plt.xlabel('Iteration')
# plt.ylabel('Distance')
# plt.title('Distance vs Iteration of GA')
# plt.show()