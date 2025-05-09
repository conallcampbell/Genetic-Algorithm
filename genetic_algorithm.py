##########################################
# Genetic Evolution Class performs mating and mutation procedure
# on an the fittest half of the population
##########################################
import numpy as np
import random
import math

from network import Frozen_Network
import networkx as nx


def erdos_renyi(n, p):
    return nx.erdos_renyi_graph(n, p, seed=None)


def randomERNetwork(n):
    G = erdos_renyi(n, round(random.uniform(0, 1), 2))

    while not nx.is_connected(G):
        G = erdos_renyi(n, round(random.uniform(0, 1), 2))

    return Frozen_Network(nx.to_numpy_array(G).astype("complex"))


class Genetic_Evolution:
    def __init__(
        self,
        population,
        pop_size,
        mask,
        mutation_probability=0.01,
        flip_probability=0.1,
        mutation_fraction=0.9,
        retain_frac=0.1,
        injection_frac=0,
    ):
        self.old_population = population
        self.new_population = self.old_population[: int(pop_size * retain_frac)]
        self.mutation_probability = mutation_probability
        self.flip_probability = flip_probability
        self.mutation_fraction = mutation_fraction
        self.injection_fraction = injection_frac
        self.pop_size = pop_size
        self.mask = mask

    def mate_population(self):
        # insert generalised form of mating procedure here

        while len(self.new_population) < self.pop_size:
            for parent1 in self.old_population:
                parent2 = self.old_population[
                    int(np.random.rand() * len(self.old_population))
                ]

                while parent1 == parent2:
                    parent2 = self.old_population[
                        int(np.random.rand() * len(self.old_population))
                    ]

                size = parent1.adjacency_matrix.shape[0]
                # generating a lower triangular matrix of mated matrix
                offspring_l = np.zeros((size, size)).astype("complex")

                total_inverse_distance = 1 / (parent1.distance) + 1 / (parent2.distance)
                mating_probability1 = 1 / (total_inverse_distance * parent1.distance)

                for n in range(size):
                    for m in range(n):
                        if np.random.random() < mating_probability1:
                            offspring_l[n, m] = parent1.adjacency_matrix[n, m]
                        else:
                            offspring_l[n, m] = parent2.adjacency_matrix[n, m]

                # populating the upper triangular matrix
                offspring_u = np.conjugate(offspring_l.T)
                # forming the complete offspring matrix
                offspring = offspring_l + offspring_u

                self.new_population.append(Frozen_Network(offspring))
                if len(self.new_population) == self.pop_size:
                    break

    def mutate_population(self):
        # insert mutation code here
        for i, network in enumerate(self.new_population):
            if i > int(self.pop_size * (1 - self.mutation_fraction)):
                # if np.random.rand() < 0.5:
                self.new_population[i] = self.alterMutate(network)

    def alterMutate(self, network):
        matrix = network.adjacency_matrix
        y = random.random()
        mutation_probability = -self.mutation_probability * np.log(
            1 - y + y * math.e**-100
        )
        for i in range(len(matrix)):
            for j in range(i):
                if self.mask[i][j] == -1:
                    x = np.random.rand()
                    if x < mutation_probability:
                        element = matrix[i, j]
                        matrix[i, j] = (element + 1) % 2
                        matrix[j, i] = matrix[i, j]

        return Frozen_Network(matrix)

    def inject_population(self):
        for i in range(int(self.pop_size * self.injection_fraction)):
            network = randomERNetwork(self.old_population[0].nodes)
            network.adjacency_matrix[self.mask != -1] = self.mask[self.mask != -1]
            self.new_population.append(network)

    def new_pop(self):
        # make a list of unmutated and mutated networks
        self.inject_population()
        self.mate_population()
        self.mutate_population()
        return self.new_population
