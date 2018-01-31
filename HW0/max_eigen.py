import numpy as np 

A = np.array([[1, 0], [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

max_eigenval_index = np.argmax(eigenvalues)

max_eigenvector = eigenvectors[max_eigenval_index]

print(max_eigenvector)