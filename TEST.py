# Genetic Algorithm
import numpy as np
import math
import random
import networkx as nx
from networkx import components
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

#creating the write-out function which will allow us to save the best and average distances after each iteration to a new file in the directory
#This write data to a file, parameter x is the list of x values, parameter y is the list of y values,
#filename is the name of the file where the data will be saved and mode  is the file writing mode ('w' for write and 'a' for append)
def write_out(x , y , filename , mode='a'):
    with open(filename , mode) as file:
        file.write(f"{x} {y}\n")
    file.close()


##########################################
#number of networks I wish to consider
pop_size = 400
##########################################
##########################################
#network parameters
##########################################
# number of nodes
n = 20
# each node is joined with its m nearest neighbours in a ring topology
m = 2
# probability of edge rewiring
p = 0.1
# probability of edge creation
prob = random.random()
# exponential probability distribution used for sampling the initial population
y = random.random()
probability = -1/100 * np.log(1 - random.random() + random.random()*math.e**-100)
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
##########################################
#Unitary Evolution
##########################################
#defining the time step
ts = np.linspace(0 , 1 , 10)
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
    return [nx.to_numpy_array(F(n, random.random())).astype("complex") for _ in range(pop_size)]
population = pop(pop_size, n)


def alt_genetic_distance(pop , start_index = 0):
    dist = []
    for i in range(start_index , pop_size):
        dist.append([alt_Frozen_Network_class(pop[i] , target_data() , ts).calculate_distance() , pop[i]])
    dist = sorted(dist , key = lambda x: x[0])
    return [d[0] for d in dist] , [d[1] for d in dist]

def fit_half(pop):
    return pop[:pop_size // 2]

alt_distances = [alt_genetic_distance(population)[0]]
ordered_pop = [alt_genetic_distance(population)[1]]
found_it = []
if alt_distances[0][0] == 0:
    found_it.append(fit_half(ordered_pop[0])[0])
    print("Found fit individual")
else:
    print("Continue to Genetic Evolution")


# If we find target network in initial population then run the following line of code
np.array([np.sum(np.real(adjacency_matrix_target()[0]))/2 , np.sum(np.real(found_it[0]))/2])
adjacency_matrix_target()[0] == found_it[0]
alt_Frozen_Network_class(found_it[0] , target_data() , ts).calculate_distance()


new_pop = Genetic_Evolution(fit_half(alt_genetic_distance(population)[1]) , alt_distances[0][:len(alt_distances[0])//2]).new_population()

distances = []
best_individual = []
number_of_connections = []
new_pop_set = []
best_distance = []
mean_distance = []
fittest_individual = []
connections = []
degree_distribution = []
found_it = []

# considering the degree distribution of the target network
target_degrees = sorted(dict(nx.degree(target)).items() , key = lambda x: x[1] , reverse = True)
target_degrees
target_degrees_only = [degree for node , degree in target_degrees]
target_degrees_only

adjacency_matrix_target()[0]

# initialising extreme mutation count
# this track will be used to track how many times an extreme mutation has been implemented
# if this number reaches 3 a new population of individuals will be injected, see inside loop
extreme_mutation_count = 0

for j in tqdm(range(100)):

    fit_individual = []
    if j == 0:
        new_distances, new_pop = alt_genetic_distance(new_pop)
    else:
        new_distances, new_population = alt_genetic_distance(new_pop , start_index = pop_size // 4)
        # new_distances = sorted(distances[-1][:pop_size // 4] + new_distances)
        # new_pop = new_pop[:pop_size//4] + new_population

        # Combine distances and population from current and new generations
        combined = list(zip(distances[-1][:pop_size // 4] + new_distances, new_pop[:pop_size // 4] + new_population))
        # Sort by distances
        combined.sort(key=lambda x: x[0])
        # Unpack sorted distances and population
        new_distances, new_pop = zip(*combined)
        # Convert to lists (if needed)
        new_distances = list(new_distances)
        new_pop = list(new_pop)

    distances.append(new_distances)
    best_individual.append(new_pop[0])
    number_of_connections.append(np.array([np.sum(np.real(adjacency_matrix_target()[0]))/2 , np.sum(np.real(new_pop[0]))/2 , np.sum(np.real(new_pop[1]))/2 , np.sum(np.real(new_pop[2]))/2]))
    new_pop_set.append(np.array(new_pop[:pop_size // 4]))
    degree_distribution.append([degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(new_pop[0]))), key=lambda x: x[1], reverse=True)])
    #Calculating the best and mean distances
    best_distance = min(distances[j])
    mean_distance = np.mean(new_distances)
    fittest_individual = new_pop[0]
    connections = np.array([np.sum(np.real(adjacency_matrix_target()[0]))/2 , np.sum(np.real(new_pop[0]))/2 , np.sum(np.real(new_pop[1]))/2 , np.sum(np.real(new_pop[2]))/2 , np.sum(np.real(new_pop[3]))/2])
    degree_dist = [degree for node , degree in sorted(nx.degree(nx.from_numpy_array(np.real(new_pop[0]))), key=lambda x: x[1], reverse=True)]


    #Saving the best and mean distances to their respective files
    write_out(j , best_distance , "best_distances_20NODES.dat")
    write_out(j , mean_distance , "mean_distances_20NODES.dat")
    write_out(j , fittest_individual , "fittest_individual_20NODES.dat")
    write_out(j , connections , "connections_20NODES.dat")
    write_out(j , degree_dist , "degree_dist_20NODEES.dat")
    
    # what to do if the algorithm finds the fittest individual
    # else rerun code
    if min(distances[j]) == 0:
        fit_individual.append(new_pop[0])
        print("fit individual found")
        break
    else:
        new_pop = Genetic_Evolution(fit_half(alt_genetic_distance(new_pop)[1]) , distances[j][:len(distances[j])//2]).new_population()
        print(len(new_pop))
    # implementing extreme mutations if population scores aren't improving
    if j > 5:
        if (new_pop_set[j-5] == new_pop_set[j]).all():
            new_pop = Genetic_Evolution(fit_half(alt_genetic_distance(new_pop)[1]) , distances[j][:len(distances[j])//2]).EXTREME_new_population()
            # if this is performed we add +1 to the extreme_mutation_count
            extreme_mutation_count += 1
            # printing when this if performed
            print("Extreme mutation performed")
            print(extreme_mutation_count)
    # the case of reaching 3 in extreme_mutation_count means we should inject a new population of individuals
    if extreme_mutation_count == 5:
        # inject a new population of individuals and keeping old unmutated parents
        new_pop = new_pop[:int(pop_size * 0.75)] + pop(pop_size//4 , n)
        # resetting the extreme_mutation_count to zero
        extreme_mutation_count = 0
        print("New population of individuals injected")


    # if target_degrees_only == degree_distribution[j]:
    #     fit_individual.append(new_pop[0])
    #     print("same degree distribution found")
    #     break

    # if nx.is_isomorphic(target , nx.from_numpy_array(new_pop[0])) == True:
    #     found_it.append(new_pop[0])
    #     print("Isomorphism of target network found")
    #     break
        
else:
    print("No fit individual found within iteration limit")


alt_Frozen_Network_class(best_individual[len(best_individual) - 1] , target_data() , ts).calculate_distance()
alt_Frozen_Network_class(new_pop[0] , target_data() , ts).calculate_distance()


#  RELABELLNIG MAP


# If fittest and target degrees do not match then use the following map to relabel the degree distributions

fittest_degrees = sorted(nx.degree(nx.from_numpy_array(np.real(best_individual[len(best_individual) - 1]))), key=lambda x: x[1], reverse=True)
fittest_degrees
target_degrees

# Create a mapping from fittest nodes to target nodes
relabel_map = {fittest_node: target_node for (fittest_node, _), (target_node, _) in zip(fittest_degrees, target_degrees)}
# Get the adjacency matrix of the fittest individual
fittest_adj_matrix = np.real(best_individual[len(best_individual) - 1])
# Create a new adjacency matrix with relabeled nodes
n = best_individual[len(best_individual) - 1].shape[0]  # Assuming the adjacency matrix is square (n x n)
relabelled_adj_matrix = np.zeros_like(best_individual[len(best_individual) - 1])
# Apply the relabeling to the adjacency matrix
for i in range(n):
    for j in range(n):
        new_i = relabel_map[i]  # Get the new node label for row i
        new_j = relabel_map[j]  # Get the new node label for column j
        relabelled_adj_matrix[new_i, new_j] = best_individual[len(best_individual) - 1][i, j]
# number of different connections to target network
arr2 = relabelled_adj_matrix == np.real(adjacency_matrix_target()[0])
false_count2 = np.sum(arr2 == False) // 2
print(f'The number of False connections for the rearranged individual is: {false_count2}')

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(target , 1)
nx.node_connected_component(nx.from_numpy_array(relabelled_adj_matrix) , 1)

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(target , nx.from_numpy_array(np.real(best_individual[len(best_individual) - 1])))
nx.is_isomorphic(target , nx.from_numpy_array(relabelled_adj_matrix))
nx.is_isomorphic(nx.from_numpy_array(np.real(best_individual[len(best_individual) - 1])) , nx.from_numpy_array(relabelled_adj_matrix))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(target)
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(best_individual[len(best_individual) - 1])))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(relabelled_adj_matrix))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(target , nx.from_numpy_array(np.real(best_individual[len(best_individual) - 1])))
nx.graph_edit_distance(target , nx.from_numpy_array(relabelled_adj_matrix))





# Comparing the physical properties of the networks





# Fidelity

# Beginning with the initial state of the network as we inject the single excitation into node 1
# We consider the fidelity between the target state and the fittest individual

##########################################
#Temporal Evolution
##########################################

#fittest individual
def fittest_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * best_individual[len(best_individual) - 1] * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * best_individual[len(best_individual) - 1] * t).T))

            ]
def fittest_evolution_data():
    data = []
    for t in ts:
        data.append(fittest_evolution(t))
    return data

#target
def target_evolution(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
        return [
            
                (expm(complex(0 , -1) * adjacency_matrix_target()[0] * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * adjacency_matrix_target()[0] * t).T))

            ]
def target_evolution_data():
    data = []
    for t in ts:
        data.append(target_evolution(t))
    return data

# are my states physical?
# trace = []
# for i in range(len(ts)):
#     trace.append(np.trace(fittest_evolution_data()[i][0]))
# eigenvalues = []
# for i in range(len(ts)):
#     eigenvalues.append(np.linalg.eigvals(fittest_evolution_data()[i][0]))
# hermitian = []
# for i in range(len(ts)):
#     arr = fittest_evolution_data()[i][0] == np.conjugate(fittest_evolution_data()[i][0].T)
#     hermitian.append(np.sum(arr == False) // 2)

# fittest_evolution_data()[1][0] == np.conjugate(fittest_evolution_data()[1][0].T)

# defining the fidelity between 2 quantum states
def fidelity(mat1 , mat2):
    return np.real((np.trace(sqrtm(sqrtm(mat1) @ mat2 @ sqrtm(mat1))))**2)
# fidelity results
fidelity_tab2 = []
for i in range(len(ts)):
    fidelity_tab2.append(fidelity(target_evolution_data()[i][0] , fittest_evolution_data()[i][0]))
# Plot of fidelity
figure , ax1 = plt.subplots(1,1)
ax1.set_xlabel("t")
ax1.set_ylabel("Fidelity")
ax1.plot(ts , fidelity_tab2)
plt.show()

# defining coherences
def coherence(mat):
    sum = []
    for i in range(len(ts)):
        sum.append(np.abs(np.sum(mat[i][0]) - np.trace(mat[i][0])))
    return sum
coherence(target_evolution_data())
coherence(fittest_evolution_data())
# Plot of coherence
figure , ax1 = plt.subplots(1,1)
ax1.set_xlabel("t")
ax1.set_ylabel("Coherences")
ax1.plot(ts , coherence(target_evolution_data()) , label = "Target Network")
ax1.plot(ts , coherence(fittest_evolution_data()) , label = "Fittest Network")
ax1.legend(title = "Network" , bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
plt.show()

# defining populations
def populations(mat1 , mat2):
    pop = [[] for _ in range(len(mat1[0][0]))]
    for i in range(len(mat1)):
        for j in range(len(mat1[0][0])):
            pop[i].append(np.abs(np.real(mat1[j][0][i][i] - mat2[j][0][i][i])))
    return pop
pop = populations(target_evolution_data() , fittest_evolution_data())
pop_target = populations(target_evolution_data())
pop_fittest = populations(fittest_evolution_data())
# plot of populations
figure , ax1 = plt.subplots(1,1)
ax1.set_xlabel("t")
ax1.set_ylabel("Populations")
for indx , node_pop in enumerate(pop):
    ax1.plot(ts , node_pop , label = f"Node {indx}")
ax1.legend(title = "Nodes", bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
plt.show()
figure.savefig("10 Nodes 16.png" , bbox_inches = 'tight' , dpi = 1000)
# figure , ax1 = plt.subplots(1,1)
# ax1.set_xlabel("t")
# ax1.set_ylabel("Populations")
# for node_pop in pop_target:
#     ax1.plot(ts , node_pop)
# plt.show()