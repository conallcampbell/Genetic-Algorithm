##########################################
# Genetic Evolution Class performs mating and mutation procedure
# on an the fittest half of the population
##########################################
import numpy as np
import math
import random

class Genetic_Evolution:
    def __init__(self , population , distances):
        self.population = population
        self.distances = distances

    def offsprings(self):
        # insert generalised form of mating procedure here
        offspring_tab = []

        for m in range(len(self.population)):

            size = self.population[m].shape[0]
            # generating a lower triangular matrix of mated matrix
            offspring_l = np.zeros((size , size)).astype("complex")
            index1 = m
            index2 = (m + 1) % len(self.population)

            total_inverse_distance = 1/(self.distances[m]) + 1/(self.distances[(m + 1) % len(self.population)])
            mating_probability1 = 1 / (total_inverse_distance * self.distances[m])

            for n in range(size):
                for k in range(n):
                    if np.random.random() < mating_probability1:
                        offspring_l[n , k] = self.population[index1][n , k]
                    else:
                        offspring_l[n , k] = self.population[index2][n , k]
            # populating the upper triangular matrix
            offspring_u = np.conjugate(offspring_l.T)
            # forming the complete offspring matrix
            offspring = offspring_l + offspring_u
            offspring_tab.append(offspring)
        return offspring_tab

    def mated_population(self):
        # make list of parents + offsprings
        mated_pop = []
        for i in range(len(self.population)):
            mated_pop.append(self.population[i])
        for j in range(len(self.offsprings())):
            mated_pop.append(self.offsprings()[j])
        return mated_pop

    def mutation1(self):
        # mutation function that flips connections between nodes
        mutated_matrices = []
        y = random.random()
        mutation_probability = -1/100 * np.log(1-y+y*math.e**-100)
        mated_pop = self.mated_population()
        for k in range(len(mated_pop) // 4 , len(mated_pop)):
            matrix = mated_pop[k].copy()
            for i in range(len(matrix)):
                for j in range(i):
                    x = random.random()
                    if x < mutation_probability:
                        element = matrix[i , j]
                        mutated_real_part = int(np.real(element)) ^ 1
                        mutated_element = complex(mutated_real_part , np.imag(element))
                        matrix[i , j] = mutated_element
                        matrix[j , i] = matrix[i , j]
            mutated_matrices.append(matrix)
        return mutated_matrices
    
    def mutation2(self):
        # considers n pairs of nodes (2 at a time) and flips the connections between them
        # empty list of matrices that will be the output of this function
        mutated_matrices = []
        # calling on the set of mutated matrices from the previous function
        mutated_pop = self.mutation1()

        # all matrices in the population have already been mutated
        # so like before we generate a random number from an exponential distribution for each network
        # then if this random number x is less than the mutation probability we perform this secondary mutation
        y = random.random()
        mutation_probability = -1/100 * np.log(1-y+y*math.e**-100)

        # starting the mutation process
        for m in range(len(mutated_pop)):
            # defining the size of the matrices, telling us how many nodes are in the network
            size = mutated_pop[m].shape[0]
            # defining a blank matrix which we will mutate
            matrix = mutated_pop[m].copy()

            # generating a random number x
            x = random.random()
            if x < mutation_probability:
                # considering any 2 different connections of network m
                # connection 1 is any connection in the lower triangular (excluding diagonal)
                while True:
                    i1 , j1 = random.randint(0 , size - 1) , random.randint(0 , size - 1)
                    if i1 < j1:
                        break
                # connection 2 is any connection DIFFERENT from connection 1
                # again, connection 2 is in the lower triangular (excluding diagonal)
                while True:
                    i2 , j2 = random.randint(0 , size - 1) , random.randint(0 , size - 1)
                    if i2 < j2 and (i1 , j1) != (i2 , j2) and matrix[i1 , j1] != matrix[i2 , j2]:
                        break
                
                # swapping the connections if they are different
                # swapping the upper triangular part
                matrix[i1 , j1] , matrix[i2 , j2] = (matrix[i1 , j1].real + 1) % 2 + 0j , (matrix[i2 , j2].real + 1) % 2 + 0j
                # swapping the lower triangular part (which ensures symmetry)
                matrix[j1 , i1] , matrix[j2 , i2] = (matrix[j1 , i1].real + 1) % 2 + 0j, (matrix[j2 , i2].real + 1) % 2 + 0j

            # we save this new matrix and append to our list of newly mutated networks
            mutated_matrices.append(matrix)
        return mutated_matrices
    
    def EXTREME_MUTATION(self):
        # mutation function that flips connections between nodes
        mutated_matrices = []
        mutation_probability = 0.45
        mated_pop = self.mated_population()
        for k in range(len(mated_pop) // 4 , len(mated_pop)):
            matrix = mated_pop[k].copy()
            for i in range(len(matrix)):
                for j in range(i):
                    x = random.random()
                    if x < mutation_probability:
                        element = matrix[i , j]
                        mutated_real_part = int(np.real(element)) ^ 1
                        mutated_element = complex(mutated_real_part , np.imag(element))
                        matrix[i , j] = mutated_element
                        matrix[j , i] = matrix[i , j]
            mutated_matrices.append(matrix)
        return mutated_matrices

    def new_population(self):
        # make a list of unmutated and mutated networks
        mated_pop = self.mated_population()
        return mated_pop[:len(mated_pop) // 4] + self.mutation1()
    
    def EXTREME_new_population(self):
        # making a list of unmutated and mutated networks
        # this time we have performed an extreme mutation
        mated_pop = self.mated_population()
        return mated_pop[:len(mated_pop) // 4] + self.EXTREME_MUTATION()