##########################################
# Genetic Evolution Class performs mating and mutation procedure
# on an the fittest half of the population
##########################################
import numpy as np
import random

from network import Frozen_Network


class Genetic_Evolution:
    def __init__(self, population, mutation_probability=0.01):
        self.population = population
        self.mutation_probability = mutation_probability

    def mate_population(self):
        # insert generalised form of mating procedure here
        offspring_tab = []

        for i, parent1 in enumerate(self.population):
            size = parent1.adjacency_matrix.shape[0]
            parent2 = self.population[(i + 1) % len(self.population)]
            # generating a lower triangular matrix of mated matrix
            offspring_l = np.zeros((size, size)).astype("complex")

            for n in range(size):
                for m in range(n):
                    if np.random.random() < 0.5:
                        offspring_l[n , m] = parent1.adjacency_matrix[n , m]
                    else:
                        offspring_l[n , m] = parent2.adjacency_matrix[n , m]

            # populating the upper triangular matrix
            offspring_u = np.conjugate(offspring_l.T)
            # forming the complete offspring matrix
            offspring = offspring_l + offspring_u
            offspring_tab.append(Frozen_Network(offspring))
        self.population += offspring_tab

    def mutate_population(self, fraction):
        # insert mutation code here
        for network in sorted(self.population, key=lambda x: x.distance if x.distance else 10000000)[int(len(self.population) * (1-fraction)):]:
            self.mutate(network)

    def mutate(self, network):
        matrix = network.adjacency_matrix
        n_con = np.sum(matrix)//2
        n_nocon = (matrix.shape[0]**2 - matrix.shape[0])//2 - n_con
        for i in range(len(matrix)):
            for j in range(i):
                con_val = matrix[i][j]
                if con_val == 1:
                    mutation_factor = n_nocon / n_con
                else:
                    mutation_factor = n_con/n_nocon
                x = random.random()
                if x < self.mutation_probability * mutation_factor:
                    element = matrix[i , j]
                    matrix[i , j] = (element + 1)%2
                    matrix[j , i] = matrix[i , j]
        network.adjacency_matrix = matrix

    def new_population(self):
        # make a list of unmutated and mutated networks
        self.mate_population()
        self.mutate_population(0.99)
        return self.population