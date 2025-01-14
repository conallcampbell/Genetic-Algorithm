import networkx as nx
import numpy as np
import random
from scipy.linalg import expm
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
# Population size
##########################################
pop_size = 4
##########################################

##########################################
# Network parameters
##########################################
n = 5
m = 2
p = 0.01
def G(n , m , p):
    return nx.watts_strogatz_graph(n , m , p , seed = None)
##########################################
# Target
##########################################
target = G(n , m , p)
target_matrix = nx.to_numpy_array(target).astype("complex")

t4 = np.array([[0.0 , 1.0 , 1.0 , 0.0] , [1.0 , 0.0 , 0.0 , 1.0] , [1.0 , 0.0 , 0.0 , 1.0] , [0.0 , 1.0 , 1.0 , 0.0]] , dtype=complex)
t5 = np.array([[0.0 , 1.0 , 0.0 , 1.0 , 1.0] , [1.0 , 0.0 , 1.0 , 0.0 , 0.0] , [0.0 , 1.0 , 0.0 , 1.0 , 1.0] , [1.0 , 0.0 , 1.0 , 0.0 , 0.0] , [1.0 , 0.0 , 1.0 , 0.0 , 0.0]] , dtype=complex)
##########################################
# Population
##########################################
def pop(pop_size , n , m):
    return [G(n , m , round(random.uniform(0.0 , 0.5) , 2)) for _ in range(pop_size)]
population = pop(pop_size , n , m)
def adj_mat(population):
    return [nx.to_numpy_array(graph).astype("complex") for graph in population]
adjacency_matrices = adj_mat(population)
##########################################
# defining the timestep
##########################################
ts = np.linspace(0 , 10 , 100)

##########################################
# Defining the network class that considers any size of adjacency matrix 
# and returns all the sink outputs for that specific matrix
##########################################
targ_test = np.array([[0.0 , 1.0 , 1.0 , 0.0] , [1.0 , 0.0 , 0.0 , 1.0] , [1.0 , 0.0 , 0.0 , 1.0] , [0.0 , 1.0 , 1.0 , 0.0]] , dtype=complex)
adjm_test = np.array([[0.0 , 1.0 , 0.0 , 0.0] , [1.0 , 0.0 , 0.0 , 1.0] , [0.0 , 0.0 , 0.0 , 1.0] , [0.0 , 1.0 , 1.0 , 0.0]] , dtype=complex)


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

t4_data = Network(t4)
new_data = t4_data.simulate()

t5_data = Network(t5)
new_data = t5_data.simulate()


target_data = Network(target_matrix)
new_data = target_data.simulate()












def adjacency_matrix_targ_test():
    return [copy.deepcopy((targ_test)) for _ in range(n)]

def targ_test_sink_with_sinks():
    mat_list = adjacency_matrix_targ_test()
    for i in range(n):
        mat_list[i][i, i] -= 1j
    return mat_list

targ_test_sink = targ_test_sink_with_sinks()

# initial state
k = 1
rho0 = np.zeros((n , 1))
rho0[k] = 1
state = kron(rho0 , np.conjugate(rho0.T))

ts = np.linspace(0 , 10 , 100)
def evolution_targ_test(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
    return [
            
                1-(np.trace
                    ((expm(complex(0 , -1) * targ_test_sink[i] * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * targ_test_sink[i] * t).T))
                  )
                ).real
            
            for i in range(n)
            ]
for i in range(n):
    import sys
    np.savetxt(sys.stdout, targ_test_sink[i], fmt="%.3f")
    print("---")


def evol_targ_test():
    data = []
    for t in ts:
        data.append(evolution_targ_test(t))
    return data
evolve_targ_test = np.array(evol_targ_test())





def adjacency_matrix_adjm_test():
    return [copy.deepcopy((adjm_test)) for _ in range(n)]

def adjm_test_sink_with_sinks():
    mat_list = adjacency_matrix_adjm_test()
    for i in range(n):
        mat_list[i][i, i] -= 1j
    return mat_list

adjm_test_sink = adjm_test_sink_with_sinks()

# initial state
k = 1
rho0 = np.zeros((n , 1))
rho0[k] = 1
state = kron(rho0 , np.conjugate(rho0.T))

ts = np.linspace(0 , 10 , 100)
def evolution_adjm_test(t):
    #expmatrix = expm(complex(0 , -1) * target_sink()[i] * t)
    return [
            
                1-(np.trace
                    ((expm(complex(0 , -1) * adjm_test_sink[i] * t))
                    @ state
                    @ (np.conjugate(expm(complex(0 , -1) * adjm_test_sink[i] * t).T))
                  )
                ).real
            
            for i in range(n)
            ]
for i in range(n):
    import sys
    np.savetxt(sys.stdout, adjm_test_sink[i], fmt="%.3f")
    print("---")


def evol_adjm_test():
    data = []
    for t in ts:
        data.append(evolution_adjm_test(t))
    return data
evolve_adjm_test = np.array(evol_adjm_test())


def alt_distance():
    distances = []
    targ = evolve_targ_test
    adjm = evolve_adjm_test
    distance_sum = np.sum(abs(targ - adjm))
    distances.append(distance_sum)
    return distances


class Distance:
    def __init__(self , target , network):
        self.target = target
        self.network = network

    # defining the distance between the individual and target networks
    def distance(self):
        distances = []
        targ = Network(self.target).simulate()

        for j in range(pop_size):
            # initialise individual_distance_sum for each network
            indiv = Network(self.network[j]).simulate()
            individual_distance_sum = sum(abs(targ[i][k] - indiv[i][k]) for i in range(len(ts)) for k in range(n))
            distances.append(individual_distance_sum)
        return distances




target_data = Network(target_matrix)

network_data = [Network(graph) for graph in adjacency_matrices]

distances = [Distance(target_data , network) for network in network_data]

class Fitness_Evaluator:
    # this class will determine if any of the individuals from the population are fit
    # therefore within this class will be contained the break function if a fit individual is found
    # otherwise
    # we arrange the population from fittest to weakest and eliminate the worst half of the population
    
    # saving the adjacency matrices of the population to the class
    def __init__(self, target , network):
        self.target = target
        self.network = network

    # organising the list of distances from best to worst
    def listed_individuals(self):
        for j in range(pop_size):
            # listing the population of adjacency matrices and their corresponding
            # distances to the target matrix
            self.indiv[j] = Network(self.network[j]).simulate()
            distance = Distance(self.target , self.indiv[j]).simulate()
            # sorting the distances from best to worst
            sorted_list = sorted(range(len(pop_size)) , key = lambda i: distance[i])
            return sorted_list
    
    #finding the fittest individual or returing the fit half of the population
    def evaluate_fitness(self):
        sorted_indices = self.listed_individuals()
        for j in range(pop_size):
            if distance[sorted_indices[0]] < 10**-3:
                return adjacency_matrices[sorted_indices[0]]




    # sorting the distances
    def listed_individuals(self):
        self.sorted_list = sorted(range(len(pop_size)) , key = lambda i: self.distances[i])
        return self.sorted_list
    # finding the fittest individual
    def evaluate_fitness(self):
        sorted_indices = self.listed_individuals()
        if self.distances[sorted_indices[0]] < 10**-3:
            return adjacency_matrices[sorted_indices[0]]
        # killing half the population
        else:
            best_half = sorted_indices[:len(sorted_indices)//2]
            selected_individuals = [self.adjacency_matrices[i] for i in best_half]
            return selected_individuals

class Fitness_Evaluator:
    # this takes the ouput from the Fitness_Evaluator class and evolves the population
    # by performing mating
    def __init__(self , fitness_evaluator):
        self.fitness_evaluator = fitness_evaluator

    # Mating procedure
    def mating(self):
        offspring_tab = []
        for j in range(len(self.fitness_evaluator)):
            size = self.fitness_evaluator[j].shape[0]
            offspring_l = np.zeros((size , size) , dtype = np.complex128)
            index1 = j
            index2 = (j + 1) % len(self.fitness_evaluator)

            for n in range(size):
                for m in range(n):
                    if np.random.random() < 0.5:
                        offspring_l[n , m] = self.fitness_evaluator[index1][n , m]
                    else:
                        offspring_l[n , m] = self.fitness_evaluator[index2][n , m]
            
            # populating the lower triangular matrix
            offspring_u = np.conjugate(offspring_l.T)
            # forming the complete offspring matrix
            offspring = offspring_l + offspring_u
            offspring_tab.append(offspring)
        
        return offspring_tab
    

class Mutation:
    # taking the output of the mating class, combined with the fit population
    # to create the x-men , i.e.,  my mutated population that will be fed back
    # into the network class and evaluated all over again
    def __init__(self , fitness_evaluator , offspring):
        self.fit = fitness_evaluator
        self.off = offspring
        self.pop = population

    def mutated(self):
        mutated_matrices = []
        mutation_probability = 0.1
        mutation_start_index = pop_size//4

        for k in range(mutation_start_index , pop_size):
            matrix = self.pop[k].copy()
            for i in range(self.n):
                for j in range(i):
                    x = np.random.random()
                    if x < mutation_probability:
                        element = matrix[i , j]

                        mutated_real_part = int(np.real(element)) ^ 1
                        mutated_element = complex(mutated_real_part , np.imag(element))

                        matrix[i , j] = mutated_element
                        matrix[j , i] = matrix[i , j]
            
                mutated_matrices.append(matrix)
        return mutated_matrices
    



# potential code for evolution
class OffspringGenerator:
    def __init__(self, fitness_results):
        self.fitness_results = fitness_results

    def generate_offsprings(self):
        offspring_tab = []
        for j in range(len(self.fitness_results)):
            size = self.fitness_results[j].shape[0]
            offspring_l1 = np.zeros((size, size), dtype=np.complex128)
            index1 = j
            index2 = (j + 1) % len(self.fitness_results)

            for n in range(size):
                for m in range(n):
                    if np.random.random() < 0.5:
                        offspring_l1[n, m] = self.fitness_results[index1][n, m]
                    else:
                        offspring_l1[n, m] = self.fitness_results[index2][n, m]

            offspring_u1 = np.conjugate(offspring_l1.T)
            offspring1 = offspring_l1 + offspring_u1
            offspring_tab.append(offspring1)
        return offspring_tab


class PopulationEvolver:
    def __init__(self, fitness_results, resulting_offsprings):
        self.fitness_results = fitness_results
        self.resulting_offsprings = resulting_offsprings

    def generate_new_population(self):
        new_pop = []
        for i in range(len(self.fitness_results)):
            new_pop.append(self.fitness_results[i])
        for j in range(len(self.resulting_offsprings)):
            new_pop.append(self.resulting_offsprings[j])
        return new_pop


class MutationOperator:
    def __init__(self, new_pop, pop_size, n):
        self.new_pop = new_pop
        self.pop_size = pop_size
        self.n = n

    def mutate_network(self):
        mutated_matrices = []
        mutation_probability = 0.1
        mutation_start_index = self.pop_size // 4

        for k in range(mutation_start_index, self.pop_size):
            matrix = self.new_pop[k].copy()
            for i in range(self.n):
                for j in range(i):
                    x = random.random()
                    if x < mutation_probability:
                        element = matrix[i, j]
                        mutated_real_part = int(np.real(element)) ^ 1
                        mutated_element = complex(mutated_real_part, np.imag(element))
                        matrix[i, j] = mutated_element
                        matrix[j, i] = matrix[i, j]
            mutated_matrices.append(matrix)
        return mutated_matrices


class PopulationManager:
    def __init__(self, new_pop, mutated_matrices):
        self.new_pop = new_pop
        self.mutated_matrices = mutated_matrices

    def evolve_population(self):
        return self.new_pop[:len(self.new_pop) // 4] + self.mutated_matrices