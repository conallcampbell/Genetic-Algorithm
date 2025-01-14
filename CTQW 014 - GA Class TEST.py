class Network:

    def __init__(self, adjacency_mat):
        self.mat = adjacency_mat
        self.nodes = self.mat.shape[0]

    @property
    def adjacency_matrix(self):
        for i in range(self.nodes):
            if np.real(self.mat)[i][i] == 1:
                self.mat[i][i] -= 1
        self.mat = np.real(mat)

        return self.mat
    
    @adjacency_matrix.setter
    def adjacency_matrix(self, mat):
        self.mat = mat

    def set_sink(self, node):
        self.mat = np.real(mat)
        self.mat[node, node] -= 1j
        
    def set_excitation(self, node):
        for i in range(self.nodes):
            if np.real(self.mat)[i][i] == 1:
                self.mat[i][i] -= 1

        self.mat[node][node] += 1

    def generate_output(self, excitation, sink):
        self.set_sink(sink)
        self.set_excitation(excitation)
        return self.evolution()
    
    def evolution(self):
        return 
    
    def generate_complete_output(self, excitation):
        outputs = []

        for node in range(self.nodes):
            outputs.append(self.generate_output(excitation, node))

        return outputs


class FitnessEvaluator:
    
    def __init__(self, target_network):
        self.network = target_network


    def compare(self, network):
        network1.generate_complete_outputs(1)
        network2.generate_complete_outputs(1)


class GeneticController:

    def __init__(self):
        pass

    def mate(self, network1, network2):
        new_network = self.join(network1, network2)
        mutated_network = self.mutate(new_network)
        return mutated_network

    def join()
        
    def mutate()