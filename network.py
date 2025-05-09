# Frozen Network class
# this takes any input matrix and the target matrix and calculates the associated distance
# building the class this way ensures that the individual and its corresponding distance are saved together
import numpy as np
import scipy
from scipy.fft import fft
from scipy.linalg import expm, sqrtm
import scipy.linalg
import networkx as nx
import tensorflow as tf


def kron(*matrices):
    result = np.array([[1]])
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


class Frozen_Network:
    # initialising the class
    def __init__(self, adj_matrix):
        self.adjacency_matrix = adj_matrix
        self.nodes = self.adjacency_matrix.shape[0]
        self.distance = None
        self.est_adj_distance = None
        self.evolution_data = None

    # assuring that there are no sinks in the adjacency matrices
    @property
    def adjacency_matrix(self):
        return np.real(self.mat)

    # ensuring that each element of my class is indeed an adjacency matrix of a network
    @adjacency_matrix.setter
    def adjacency_matrix(self, mat):
        self.mat = mat
        self.distance = None
        self.est_adj_distance = None

    @property
    def num_connections(self):
        return np.sum(self.adjacency_matrix) // 2

    # attaching the sink to node [i , i]
    def set_sink(self, i):
        mat = self.adjacency_matrix.copy().astype(complex)
        mat[i, i] -= 1j
        return mat

    # defining the initial state
    def set_excitation(self, e):
        rho0 = np.zeros(self.nodes, dtype=complex)
        rho0[e] = 1
        return rho0

    # defining the unitary evolution
    def evolution(self, timestep, num_steps, pairs):
        self.sink_data = []
        operators = set([pair[1] for pair in pairs])
        # evolution_data = np.zeros(self.nodes)
        # setting the excitation initially at node 1
        sinks = np.array([self.set_sink(i) for i in operators])
        sink_operators = np.array([expm(-1j * s * timestep) for s in sinks])

        for i, sink in enumerate(operators):
            evolution = []
            sink_operator = sink_operators[i]
            sink_states = [
                self.set_excitation(pair[0]) for pair in pairs if pair[1] == sink
            ]
            for j in range(num_steps):
                timestep_data = []
                for k, state in enumerate(sink_states):
                    # evolving the state that has the sink located at node i
                    sink_states[k] = sink_operator @ state
                    evolution_result = 1 - np.vdot(sink_states[k], sink_states[k]).real
                    timestep_data.append(evolution_result)
                evolution.append(timestep_data)
            self.sink_data.append(np.array(evolution))

    def fullEvolution(self, timestep, num_steps):

        # evolution_data = np.zeros(self.nodes)
        # setting the excitation initially at node 1

        state = self.set_excitation(1)
        state = np.outer(state, state)

        operator = expm(complex(0, -1) * self.adjacency_matrix * timestep)

        self.evolution_data = [state]

        for i in range(num_steps):

            # evolving the state that has the sink located at node i
            state = operator @ state @ np.conjugate(operator.T)
            self.evolution_data.append(state)

    # defining the distance function
    def calculate_distance(self, target_data):
        self.distance = np.sum(
            [
                np.sum(np.abs(target_data[i] - self.sink_data[i]))
                for i in range(len(self.sink_data))
            ]
        )

    def evaluate(self, target, ts, num_steps):
        if self.evolution_data is None:
            self.fullEvolution(ts, num_steps)
        if target.evolution_data is None:
            target.fullEvolution(ts, num_steps)

        self.fidelity_diff = []
        self.coherence_diff = []
        self.pop_diff = []

        for i, state in enumerate(self.evolution_data):
            state2 = target.evolution_data[i]
            self.fidelity_diff.append(
                np.real((np.trace(sqrtm(sqrtm(state) @ state2 @ sqrtm(state)))) ** 2)
            )
            diff = state - state2
            self.coherence_diff.append(np.sum(np.abs(diff)) - np.trace(np.abs(diff)))
            self.pop_diff.append(np.diag(state) - np.diag(state2))

        self.is_isomorphic = nx.is_isomorphic(
            nx.Graph(self.adjacency_matrix), nx.Graph(target.adjacency_matrix)
        )
        self.matching_degree = sorted(
            [val[1] for val in nx.degree(nx.Graph(self.adjacency_matrix))]
        ) == sorted([val[1] for val in nx.degree(nx.Graph(target.adjacency_matrix))])

    def calculate_ML_data(self, target_network):
        self.calculate_distance(target_network.sink_data)
        target_network.calculate_variation_data()
        self.calculate_variation_data()
        diff = self.sink_data - target_network.sink_data
        # var_diff = self.variational_data - target_network.variational_data
        # self.data_ML = [np.mean(np.abs(diff)), np.var(diff), np.std(diff), np.mean(var_diff), np.var(var_diff), np.std(var_diff)]
        # self.data_ML = np.array(self.data_ML)

        self.data_ML = []
        self.data_ML = np.abs(self.sink_data - target_network.sink_data).flatten()
        # self.data_ML.append(np.sum(self.sink_data - target_network.sink_data,axis=1).flatten())
        # self.data_ML.append((self.variational_data - target_network.variational_data).flatten())
        # self.data_ML.append(np.sum(self.variational_data - target_network.variational_data,axis=1).flatten())
        # self.data_ML.append(np.diff(self.sink_data - target_network.sink_data, axis=0).flatten())
        # self.data_ML.append(np.sum(np.diff(self.sink_data - target_network.sink_data, axis=0),axis=1).flatten())
        # self.data_ML.append(np.cumsum(self.sink_data - target_network.sink_data, axis=0).flatten())
        # self.data_ML.append(np.sum(np.cumsum(self.sink_data - target_network.sink_data, axis=0),axis=1).flatten())
        # self.data_ML = np.concatenate(self.data_ML)

    def calculate_variation_data(self):
        diff = []

        for i in range(self.sink_data.shape[1]):
            new_data = []
            for j in range(self.sink_data.shape[0] - 1):
                new_data.append(self.sink_data[j + 1, i] - self.sink_data[j, i])
            diff.append(new_data)

        self.variational_data = np.array(diff).T

    def calculate_frequency_data(self):

        freqs = []
        for i in range(self.sink_data.shape[1]):
            y = self.diff.T[i]
            yf = fft(y)[: self.sink_data.shape[0] // 2]
            freqs.append(yf)

        self.frequency_data = np.abs(np.array(freqs)).T

    def estimate_adjacency_distance(self, estimator):
        self.est_adj_distance = estimator(self.piecewise_distance)

    def calculate_adjacency_distance(self, adjacency_matrix):
        self.adj_distance = np.sum(np.abs(adjacency_matrix - self.adjacency_matrix)) / 2

    def calculate_operator_distance(self, adjacency_matrix):
        self.operator_distance = np.sum(
            np.abs(
                scipy.linalg.expm(-1j * self.adjacency_matrix)
                - scipy.linalg.expm(-1j * adjacency_matrix)
            )
        )


class Population:

    def __init__(self, networks):
        self.networks = networks

    def evolvePopulation(self, ts, num_steps, pairs, batch_size=2000):
        for network in self.networks:
            network.sink_data = []
        sink_nodes = set([pair[1] for pair in pairs])
        dim = self.networks[0].nodes
        sinks = []
        for network in self.networks:
            for i in sink_nodes:
                sinks.append(-1j * network.set_sink(i) * ts)

        x = tf.convert_to_tensor(np.array(sinks), dtype=tf.complex128)

        gpu_results = []

        index = 0
        while index < len(sinks):
            mats = x[index : min(index + batch_size, len(sinks))]
            if mats.shape[0] < batch_size:
                mats = tf.concat(
                    [
                        mats,
                        tf.zeros(
                            (batch_size - mats.shape[0], dim, dim), dtype=tf.complex128
                        ),
                    ],
                    axis=0,
                )
            gpu_results.append(batch_expm_tf(mats))
            index += batch_size

        operators = tf.concat(gpu_results, axis=0)
        operators = operators[: len(sinks)]

        results_np = []
        for i, sink_node in enumerate(sink_nodes):
            ops = operators[i :: len(sink_nodes)]

            sink_states = []
            for pair in pairs:
                if pair[1] == sink_node:
                    state = np.zeros(dim).astype("complex")
                    state[pair[0]] = 1
                    state = tf.reshape(
                        tf.convert_to_tensor(state, dtype=tf.complex128), (1, dim, 1)
                    )
                    sink_states.append(state)
            results = []
            for state in sink_states:
                for k in range(num_steps):
                    state = tf.matmul(ops, state)
                    results.append(tf.reduce_sum(tf.math.conj(state) * state, axis=1))
            results_np.append(tf.concat(results, axis=1).numpy())

        for j, network in enumerate(self.networks):
            for i, sink in enumerate(sink_nodes):
                network.sink_data.append(
                    np.real((1 - results_np[i][j]).T.reshape(-1, num_steps).T)
                )

    def evaluate(self, target):
        for network in self.networks:
            network.calculate_distance(target.sink_data)
            network.calculate_adjacency_distance(target.adjacency_matrix)


@tf.function(reduce_retracing=True)
def batch_expm_tf(matrices):
    return tf.linalg.expm(matrices)
