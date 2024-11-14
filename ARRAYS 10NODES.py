import numpy as np
import networkx as nx

target1 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
0.+0.j, 1.+0.j],
[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
0.+0.j, 1.+0.j],
[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
0.+0.j, 0.+0.j],
[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
1.+0.j, 0.+0.j],
[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
0.+0.j, 0.+0.j],
[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
0.+0.j, 0.+0.j],
[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
0.+0.j, 1.+0.j],
[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
1.+0.j, 0.+0.j],
[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
0.+0.j, 0.+0.j],
[1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
0.+0.j, 0.+0.j]])

fittest1 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target1)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest1)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target1 == fittest1
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target1)) , nx.from_numpy_array(np.real(fittest1)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target1)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest1)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target1)) , nx.from_numpy_array(np.real(fittest1)))


target2 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

fittest2 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target2)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest2)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target2 == fittest2
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target2)) , nx.from_numpy_array(np.real(fittest2)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target2)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest2)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target2)) , nx.from_numpy_array(np.real(fittest2)))


target3 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest3 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target3)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest3)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target3 == fittest3
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target3)) , nx.from_numpy_array(np.real(fittest3)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target3)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest3)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target3)) , nx.from_numpy_array(np.real(fittest3)))


target4 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])

fittest4 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target4)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest4)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target4 == fittest4
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target4)) , nx.from_numpy_array(np.real(fittest4)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target4)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest4)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target4)) , nx.from_numpy_array(np.real(fittest4)))


target5 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest5 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target5)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest5)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target5 == fittest5
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target5)) , nx.from_numpy_array(np.real(fittest5)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target5)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest5)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target5)) , nx.from_numpy_array(np.real(fittest5)))


target6 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest6 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target6)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest6)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target6 == fittest6
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target6)) , nx.from_numpy_array(np.real(fittest6)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target6)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest6)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target6)) , nx.from_numpy_array(np.real(fittest6)))


target7 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

fittest7 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target7)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest7)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target7 == fittest7
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target7)) , nx.from_numpy_array(np.real(fittest7)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target7)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest7)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target7)) , nx.from_numpy_array(np.real(fittest7)))


target8 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest8 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target8)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest8)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target8 == fittest8
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target8)) , nx.from_numpy_array(np.real(fittest8)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target8)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest8)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target8)) , nx.from_numpy_array(np.real(fittest8)))


target9 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest9 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target9)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest9)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target9 == fittest9
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target9)) , nx.from_numpy_array(np.real(fittest9)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target9)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest9)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target9)) , nx.from_numpy_array(np.real(fittest9)))


target10 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest10 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target10)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest10)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target10 == fittest10
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target10)) , nx.from_numpy_array(np.real(fittest10)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target10)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest10)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target10)) , nx.from_numpy_array(np.real(fittest10)))


target11 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest11 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target11)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest11)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target11 == fittest11
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target11)) , nx.from_numpy_array(np.real(fittest11)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target11)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest11)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target11)) , nx.from_numpy_array(np.real(fittest11)))


target12 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest12 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target12)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest12)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target12 == fittest12
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target12)) , nx.from_numpy_array(np.real(fittest12)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target12)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest12)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target12)) , nx.from_numpy_array(np.real(fittest12)))


target13 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

fittest13 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target13)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest13)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target13 == fittest13
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target13)) , nx.from_numpy_array(np.real(fittest13)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target13)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest13)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target13)) , nx.from_numpy_array(np.real(fittest13)))


target14 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest14 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target14)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest14)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target14 == fittest14
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target14)) , nx.from_numpy_array(np.real(fittest14)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target14)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest14)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target14)) , nx.from_numpy_array(np.real(fittest14)))


target15 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest15 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target15)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest15)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target15 == fittest15
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target15)) , nx.from_numpy_array(np.real(fittest15)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target15)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest15)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target15)) , nx.from_numpy_array(np.real(fittest15)))


target16 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

fittest16 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target16)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest16)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target16 == fittest16
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target16)) , nx.from_numpy_array(np.real(fittest16)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target16)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest16)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target16)) , nx.from_numpy_array(np.real(fittest16)))

target17 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest17 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target17)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest17)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target17 == fittest17
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target17)) , nx.from_numpy_array(np.real(fittest17)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target17)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest17)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target17)) , nx.from_numpy_array(np.real(fittest17)))

target18 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest18 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

target18 == fittest18

target19 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

fittest19 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target19)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest19)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target19 == fittest19
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target19)) , nx.from_numpy_array(np.real(fittest19)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target19)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest19)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target19)) , nx.from_numpy_array(np.real(fittest19)))


target20 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
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

fittest20 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target20)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest20)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target20 == fittest20
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target20)) , nx.from_numpy_array(np.real(fittest20)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target20)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest20)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target20)) , nx.from_numpy_array(np.real(fittest20)))

target21 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])

fittest21 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target21)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest21)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target21 == fittest21
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target21)) , nx.from_numpy_array(np.real(fittest21)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target21)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest21)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target21)) , nx.from_numpy_array(np.real(fittest21)))


target22 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

fittest22 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target22)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest22)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target22 == fittest22
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target22)) , nx.from_numpy_array(np.real(fittest22)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target22)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest22)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target22)) , nx.from_numpy_array(np.real(fittest22)))


target23 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest23 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target23)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest23)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target23 == fittest23
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target23)) , nx.from_numpy_array(np.real(fittest23)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target23)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest23)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target23)) , nx.from_numpy_array(np.real(fittest23)))


target24 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest24 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target24)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest24)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target24 == fittest24
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target24)) , nx.from_numpy_array(np.real(fittest24)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target24)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest24)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target24)) , nx.from_numpy_array(np.real(fittest24)))


target25 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest25 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target25)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest25)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target25 == fittest25
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target25)) , nx.from_numpy_array(np.real(fittest25)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target25)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest25)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target25)) , nx.from_numpy_array(np.real(fittest25)))


target26 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest26 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

target26 == fittest26

target27 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest27 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target27)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest27)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target27 == fittest27
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target27)) , nx.from_numpy_array(np.real(fittest27)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target27)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest27)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target27)) , nx.from_numpy_array(np.real(fittest27)))


target28 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest28 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target28)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest28)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target28 == fittest28
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target28)) , nx.from_numpy_array(np.real(fittest28)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target28)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest28)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target28)) , nx.from_numpy_array(np.real(fittest28)))


target29 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

fittest29 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target29)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest29)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target29 == fittest29
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target29)) , nx.from_numpy_array(np.real(fittest29)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target29)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest29)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target29)) , nx.from_numpy_array(np.real(fittest29)))


target30 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest30 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target30)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest30)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target30 == fittest30
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target30)) , nx.from_numpy_array(np.real(fittest30)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target30)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest30)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target30)) , nx.from_numpy_array(np.real(fittest30)))


target31 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest31 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

np.array([np.sum(np.real(target31))/2 , np.sum(np.real(fittest31))/2])

target32 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
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

fittest32 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

np.array([np.sum(np.real(target32))/2 , np.sum(np.real(fittest32))/2])

target33 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest33 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target33)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest33)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target33 == fittest33
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target33)) , nx.from_numpy_array(np.real(fittest33)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target33)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest33)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target33)) , nx.from_numpy_array(np.real(fittest33)))


target34 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest34 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target34)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest34)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target34 == fittest34
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target34)) , nx.from_numpy_array(np.real(fittest34)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target34)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest34)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target34)) , nx.from_numpy_array(np.real(fittest34)))


target35 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])

fittest35 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target35)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest35)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target35 == fittest35
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target35)) , nx.from_numpy_array(np.real(fittest35)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target35)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest35)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target35)) , nx.from_numpy_array(np.real(fittest35)))


target36 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest36 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target36)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest36)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target36 == fittest36
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target36)) , nx.from_numpy_array(np.real(fittest36)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target36)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest36)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target36)) , nx.from_numpy_array(np.real(fittest36)))


target37 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])

fittest37 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target37)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest37)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target37 == fittest37
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target37)) , nx.from_numpy_array(np.real(fittest37)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target37)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest37)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target37)) , nx.from_numpy_array(np.real(fittest37)))


target38 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest38 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])


# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target38)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest38)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target38 == fittest38
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target38)) , nx.from_numpy_array(np.real(fittest38)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target38)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest38)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target38)) , nx.from_numpy_array(np.real(fittest38)))


target39 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])

fittest39 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target39)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest39)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target39 == fittest39
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target39)) , nx.from_numpy_array(np.real(fittest39)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target39)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest39)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target39)) , nx.from_numpy_array(np.real(fittest39)))


target40 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
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

fittest40 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
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

target40 == fittest40


target41 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])

fittest41 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target41)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest41)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target41 == fittest41
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target41)) , nx.from_numpy_array(np.real(fittest41)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target41)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest41)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target41)) , nx.from_numpy_array(np.real(fittest41)))


target42 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

fittest42 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target42)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest42)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target42 == fittest42
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target42)) , nx.from_numpy_array(np.real(fittest42)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target42)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest42)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target42)) , nx.from_numpy_array(np.real(fittest42)))


target43 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest43 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target43)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest43)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target43 == fittest43
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target43)) , nx.from_numpy_array(np.real(fittest43)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target43)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest43)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target43)) , nx.from_numpy_array(np.real(fittest43)))


target44 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

fittest44 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target44)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest44)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target44 == fittest44
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target44)) , nx.from_numpy_array(np.real(fittest44)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target44)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest44)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target44)) , nx.from_numpy_array(np.real(fittest44)))


target45 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest45 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target45)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest45)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target45 == fittest45
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target45)) , nx.from_numpy_array(np.real(fittest45)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target45)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest45)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target45)) , nx.from_numpy_array(np.real(fittest45)))


target46 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

fittest46 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

target46 == fittest46


target47 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])


fittest47 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target47)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest47)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target47 == fittest47
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target47)) , nx.from_numpy_array(np.real(fittest47)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target47)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest47)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target47)) , nx.from_numpy_array(np.real(fittest47)))


target48 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest48 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target48)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest48)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target48 == fittest48
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target48)) , nx.from_numpy_array(np.real(fittest48)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target48)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest48)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target48)) , nx.from_numpy_array(np.real(fittest48)))


target49 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
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


fittest49 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
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

target49 == fittest49

target50 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest50 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target50)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest50)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target50 == fittest50
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target50)) , nx.from_numpy_array(np.real(fittest50)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target50)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest50)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target50)) , nx.from_numpy_array(np.real(fittest50)))


target51 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest51 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target51)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest51)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target51 == fittest51
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target51)) , nx.from_numpy_array(np.real(fittest51)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target51)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest51)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target51)) , nx.from_numpy_array(np.real(fittest51)))


target52 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

fittest52 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target52)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest52)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target52 == fittest52
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target52)) , nx.from_numpy_array(np.real(fittest52)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target52)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest52)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target52)) , nx.from_numpy_array(np.real(fittest52)))


target53 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest53 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])


# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target53)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest53)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target53 == fittest53
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target53)) , nx.from_numpy_array(np.real(fittest53)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target53)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest53)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target53)) , nx.from_numpy_array(np.real(fittest53)))


target54 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest54 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target54)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest54)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target54 == fittest54
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target54)) , nx.from_numpy_array(np.real(fittest54)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target54)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest54)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target54)) , nx.from_numpy_array(np.real(fittest54)))


target55 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest55 = np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target55)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest55)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target55 == fittest55
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target55)) , nx.from_numpy_array(np.real(fittest55)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target55)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest55)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target55)) , nx.from_numpy_array(np.real(fittest55)))


target56 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

fittest56 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target56)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest56)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target56 == fittest56
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target56)) , nx.from_numpy_array(np.real(fittest56)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target56)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest56)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target56)) , nx.from_numpy_array(np.real(fittest56)))


target57 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j]])

fittest57 = np.array([[0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target57)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest57)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target57 == fittest57
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target57)) , nx.from_numpy_array(np.real(fittest57)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target57)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest57)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target57)) , nx.from_numpy_array(np.real(fittest57)))


target58 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest58 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target58)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest58)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target58 == fittest58
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target58)) , nx.from_numpy_array(np.real(fittest58)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target58)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest58)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target58)) , nx.from_numpy_array(np.real(fittest58)))


target59 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j]])

fittest59 = np.array([[0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j]])

# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target59)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest59)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target59 == fittest59
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target59)) , nx.from_numpy_array(np.real(fittest59)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target59)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest59)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target59)) , nx.from_numpy_array(np.real(fittest59)))


target60 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        1.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])

fittest60 = np.array([[0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j,
        1.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 1.+0.j],
       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j,
        1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
        0.+0.j, 0.+0.j],
       [1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
        0.+0.j, 0.+0.j]])


# finding the connections related to the component the excitation is initially injected into
nx.node_connected_component(nx.from_numpy_array(np.real(target60)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(fittest60)) , 1)

# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target60 == fittest60
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values for the fittest individual is: {false_count}')

# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target60)) , nx.from_numpy_array(np.real(fittest60)))

# WL graph isomorphism test
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(target60)))
nx.weisfeiler_lehman_graph_hash(nx.from_numpy_array(np.real(fittest60)))

# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target60)) , nx.from_numpy_array(np.real(fittest60)))
