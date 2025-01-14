import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.linalg import expm
from scipy.linalg import sqrtm
import random
import copy

##########################################
#kronecker product code for more than 2 inputs
##########################################
def kron(*matrices):
    result = np.array([[1]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


##########################################
#Population size
##########################################
#number of networks I wish to consider
pop_size = 4
##########################################
#network parameters
##########################################
# number of nodes
n = 12
# each node is joined with its m nearest neighbours in a ring topology
m = 2
# probability of edge rewiring
p = 0.01
# Building a Watts-Strogatz network
def G(n , m , p):
    return nx.watts_strogatz_graph(n , m , p , seed = None)



##########################################
#target
##########################################
#the random complex network we want to reconstruct
target = G(n , m , p)
#n adjacency matrices of target network
def adjacency_matrix_target():
    return [np.array(nx.to_numpy_array(target).astype("complex")) for _ in range(n)]
##########################################
#population of matrices
##########################################
#this is a test run to work out my algorithm with mutations
#I will generate 4 networks with differing probabilities for edge rewiring
#ask random for an exponential distribution
def pop(pop_size, n, m):
    return [G(n, m, round(random.uniform(0.0, 0.5), 2)) for _ in range(pop_size)]
population = pop(pop_size, n, m)

def adjmat(population, n):
    return [[copy.deepcopy(np.array(nx.to_numpy_array(graph).astype("complex"))) for _ in range(n)] for graph in population]
adj_matrices = adjmat(population, n)




##########################################
#Initial state
##########################################
# we place the excitation at node k = 6 for the target and all individuals
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
# Adding the sink to the target list
##########################################
def individuals_sink():
    lst_copy = copy.deepcopy(adj_matrices)
    for i in range(pop_size):
        for k in range(n):
            lst_copy[i][k][k , k] -= 1j
    return lst_copy
individual_sink = individuals_sink()



##########################################
#Unitary Evolution
##########################################
#defining the time step
ts = np.linspace(0 , 10 , 100)
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
for i in range(n):
    import sys
    np.savetxt(sys.stdout, target_sink[i], fmt="%.3f")
    print("---")

for t in ts:
    print(evolution_target(t))
#individuals
def evolution_individuals(t):
    return [
            [
                
                    1-(np.trace(
                        (expm(complex(0 , -1) * individual_sink[k][i] * t))
                        @ state
                        @ (np.conjugate(expm(complex(0 , -1) * individual_sink[k][i] * t).T))
                      )
                    ).real
                
             for i in range(n)
             ]
            for k in range(pop_size)
           ]



##########################################
#Fitness function
##########################################
#Using the data for the node evolutions above, we now build a fitness function
#this measures the distance between the excitation's evolution in the node for each node of the network, for the sink located at each node
#We require that if the overall distance between the plots is less than 10^-6 then we can make a fairly good reconstruction of the network's topology
def distance():
    distances = []
    for j in range(pop_size):
        individual_distance_sum = 0
        for t in ts:
            targ = evolution_target(t)
            indiv = evolution_individuals(t)[j]
            individual_distance_sum_t = sum(abs(targ[i] - indiv[i]) for i in range(n))
            individual_distance_sum += individual_distance_sum_t
        distances.append(individual_distance_sum)
    return distances
distances = distance()

#defining the fitness function
#where if the distance between the target network and the individual is less than 10**-6 then we save this network's adjacency matrix
#otherwise, we consider the best half of our population of individuals

def fitness():

    dist = distance()

    #indexing the population so that when we sort our distances from lowest to highest
    #we retain the original index so that we can recall its adjacency matrix if algorithm is successful
    sorted_indices = sorted(range(len(dist)) , key = lambda i: dist[i])
    #defining the best current individual of my population as the one with the lowest distance
    #which has been sorted using the sorted_indices variable above
    best_individual = sorted_indices[0]

    #Testing if the best individual passes the benchmark
    if dist[best_individual] < 10**-3:
        #If successful, then we return its adjacency matrix
        return adj_matrices[best_individual]
    
    #If unsuccessful, then we want the best half of our population
    else:
        #creating the best half
        best_half_of_population = sorted_indices[:len(sorted_indices)//2]
        selected_individuals = []
        #converting each adjacency matrix as a NumPy array
        for i in best_half_of_population:
            selected_individuals.append(np.array(adj_matrices[i][0]))
        #returining the best half
        return selected_individuals
fitnesses = fitness()



    
# dist = [3.2 , 2.8 , 4.5 , 2.1]
# sortedindex = sorted(range(len(dist)) , key = lambda i:dist[i])
# best_indiv = sortedindex[0]
# dist[best_indiv]

##########################################
#Evolution function
##########################################
#For all the elements of new-pop we mate every other parent with each other
#i.e. parent 2 mates with parents 1 and 3, parent 9 mates with parents 8 and 10


def offsprings():

    new_pop = fitness()
    offspring_tab = []
    

    for j in range(len(new_pop)):
        

        size = new_pop[j].shape[0]
        offspring_l1 = np.zeros((size , size) , dtype = np.complex128)

        index1 = j
        index2 = (j + 1) % len(new_pop)


        #populating the lower triangular matrix randomly from parent matrices
        for n in range(size):
            for m in range(n):
                if np.random.random() < 0.5:
                    offspring_l1[n , m] = new_pop[index1][n , m]
                else:
                    offspring_l1[n , m] = new_pop[index2][n , m]
            
        #populating the upper triangular matrix of the offspring
        offspring_u1 = np.conjugate(offspring_l1.T)
        #forming the offspring matrix
        offspring1 = offspring_l1 + offspring_u1
        offspring_tab.append(offspring1)
        
    return offspring_tab
resulting_offsprings = offsprings()
##########################################
#new population
##########################################

def new_population():
    new_pop = []
    fit = fitness()
    kid = resulting_offsprings
    for i in range(len(fit)):
        new_pop.append(fit[i])
    for j in range(len(kid)):
        new_pop.append(kid[j])
    return new_pop
new_pop = new_population()


##########################################
#Mutation function
##########################################
def mutated_network():
    mutated_matrices = []
    mutation_probability = 0.1

    mutation_start_index = pop_size // 4

    for k in range(mutation_start_index, pop_size):
        matrix = new_pop[k].copy()



        for i in range(n):
            for j in range(i):
                x=random.random()
                
                if x < mutation_probability:
                    
                    element = matrix[i , j]

                    mutated_real_part = int(np.real(element)) ^ 1
                    mutated_element = complex(mutated_real_part , np.imag(element))

                    matrix[i , j] = mutated_element
                    matrix[j , i] = matrix[i , j]
                    



        # new_pop[k] = matrix
        mutated_matrices.append(matrix)

    return mutated_matrices

mutated_matrices = mutated_network()


# Saving our evolved population that contains all the unmutated parents
# and the mutated parents that remained and all the mutated offspring
def evolved_population():
    return new_pop[:len(new_pop) // 4] + mutated_matrices
evolved_pop = evolved_population()

#Beginning to iterate the algorithm to test if the evolved population performs better than the original
def adjmat_evolution_iteration1(pop_size , n):
    return [[copy.deepcopy(np.array(evolved_pop[i])) for _ in range(n)] for i in range(pop_size)]
evolved_adj_matrices = adjmat_evolution_iteration1(pop_size , n)

#Adding sinks to all copies of the individuals
def evolved_individuals_sink_():
    lst_copy = copy.deepcopy(evolved_adj_matrices)
    for i in range(pop_size):
        for k in range(n):
            lst_copy[i][k][k , k] -= 1j
    return lst_copy

# Evolving the evolved population using time evolution operator
def evolution_evolved_individuals(t):
    return [
            [
                
                    1-(np.trace(
                        (expm(complex(0 , -1) * evolved_individuals_sink_()[k][i] * t))
                        @ state
                        @ (np.conjugate(expm(complex(0 , -1) * evolved_individuals_sink_()[k][i] * t).T))
                      )
                    ).real
                
             for i in range(n)
             ]
            for k in range(pop_size)
           ]

# Considering the fitness of our evolved population
def evolved_distance():
    distances = []
    for j in range(pop_size):
        individual_distance_sum = 0
        for t in ts:
            targ = evolution_target(t)
            indiv = evolution_evolved_individuals(t)[j]
            individual_distance_sum_t = sum(abs(targ[i] - indiv[i]) for i in range(n))
            individual_distance_sum += individual_distance_sum_t
        distances.append(individual_distance_sum)
    return distances
evolved_distances = evolved_distance()