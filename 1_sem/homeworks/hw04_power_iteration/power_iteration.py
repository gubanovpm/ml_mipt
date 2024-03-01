import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE

    return [it_meth := [0, np.random.randn(len(data))]] and [ [temp := data.dot(it_meth[1])] and [it_meth := [float(it_meth[1].dot(temp)/it_meth[1].dot(it_meth[1])), temp/np.linalg.norm(temp)]][-1] for _ in range(0, num_steps) ][-1]