import numpy as np

# Step 1: Read the file, remove square brackets, and merge every two lines
with open('target30.dat', 'r') as f:
    lines = f.readlines()
# Step 2: Clean the lines (remove square brackets) and merge every two lines
cleaned_lines = []
for i in range(0, len(lines), 3):  # Iterate in steps of 2 to merge every pair of lines
    line1 = lines[i].replace('[' , '').replace(']' , '').strip()
    line2 = lines[i+1].replace('[' , '').replace(']' , '').strip() if i+1 < len(lines) else ''
    line3 = lines[i+2].replace('[' , '').replace(']' , '').strip() if i+2 < len(lines) else ''
    merged_line = line1 + ' ' + line2 + ' ' + line3  # Merge the two lines with a space
    cleaned_lines.append(merged_line)
# Step 3: Save the cleaned and merged data back to a temporary file
with open('target30_cleaned.dat', 'w') as f:
    f.write("\n".join(cleaned_lines))
# Step 4: Now use np.genfromtxt to load the cleaned file
target30 = np.genfromtxt('target30_cleaned.dat', dtype=complex, delimiter=' ')



# Step 1: Read the file, remove square brackets, and merge every two lines
with open('30fittest.dat', 'r') as f:
    lines = f.readlines()
# Step 2: Clean the lines (remove square brackets) and merge every two lines
cleaned_lines = []
for i in range(0, len(lines), 3):  # Iterate in steps of 2 to merge every pair of lines
    line1 = lines[i].replace('[' , '').replace(']' , '').strip()
    line2 = lines[i+1].replace('[' , '').replace(']' , '').strip() if i+1 < len(lines) else ''
    line3 = lines[i+2].replace('[' , '').replace(']' , '').strip() if i+2 < len(lines) else ''
    merged_line = line1 + ' ' + line2 + ' ' + line3  # Merge the two lines with a space
    cleaned_lines.append(merged_line)
# Step 3: Save the cleaned and merged data back to a temporary file
with open('fittest30_cleaned.dat', 'w') as f:
    f.write("\n".join(cleaned_lines))
# Step 4: Now use np.genfromtxt to load the cleaned file
fittest30 = np.genfromtxt('fittest30_cleaned.dat', dtype=complex, delimiter=' ')



nx.draw(nx.from_numpy_array(np.real(target30)) , with_labels = True)
plt.show()
nx.draw(nx.from_numpy_array(np.real(fittest30)) , with_labels = True)
plt.show()

nx.node_connected_component(nx.from_numpy_array(np.real(fittest30)) , 1)
nx.node_connected_component(nx.from_numpy_array(np.real(target30)) , 1)
# which adjacency matrix elements differ between the target and fittest indiviudal
arr = target30 == fittest30
# Count the number of False values
false_count = np.sum(arr == False) // 2
# Print the result
print(f'The number of False values is: {false_count}')
# is the fittest individual an isomorphism of the target matrix?
nx.is_isomorphic(nx.from_numpy_array(np.real(target30)) , nx.from_numpy_array(np.real(fittest30)))
# measuring the similarity between two graphs through graph_edit_distance
nx.graph_edit_distance(nx.from_numpy_array(np.real(target20)) , nx.from_numpy_array(np.real(fittest20)))
