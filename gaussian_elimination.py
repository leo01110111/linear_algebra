#A Leo Wang Production
import numpy as np

def gaussian_elimination(A : np.matrix):
    """Requires an invertible matrix.
    Ensures that the result is in REF form"""
    
    #Base case
    if (A.shape == (1,1)):
        return A
    assert(A.shape[0] > 1)

    #Find non-zero element and puts it in 1,1
    col_0 = A[:, 0]

    #Find a non-zero row
    i = 0
    while (col_0[i] == 0):
        i+=1
    assert(A[i,0] != 0)

    #Swap row i with the 0th if i is not the 0th row
    if (i != 0):
        S : np.matrix = A.copy()
        A[0,:] = S[i,:]
        A[i,:] = S[0,:]
    
    #Eliminate the following non-zero entries in the 1st column through ERO
    for i in range(1, A.shape[0]):
        A[i,:] = A[i,:] - (A[i,0]/A[0,0])*A[0,:] #This is the scale and subtract ERO
    assert np.all(A[1:, 0] == 0)
    
    #Creates a minor matrix without the 0th row and column
    A_minor : np.matrix = np.matrix(np.delete(np.delete(A, 0, axis=0), 0, axis=1))
    #uses G elim
    A_minor_G = gaussian_elimination(A_minor)
    
    #Sitches back the top row and left column
    #print(f"A: {A}")
    A_top = np.vstack([A[0,1:], A_minor_G])
    #print(A_top)
    return np.hstack( [A[:,0], A_top])

def main():
    # identity matrix
    A = np.matrix([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])
    R = gaussian_elimination(A)
    print("====")
    print(A)
    print(R)

    # upper triangular
    A = np.matrix([[2, 3, 1],
                   [0, 5, 4],
                   [0, 0, 7]])
    R = gaussian_elimination(A)
    print("====")
    print(A)
    print(R)

    # full invertible 2x2
    A = np.matrix([[2, 1],
                   [4, 3]])
    R = gaussian_elimination(A)
    print("====")
    print(A)
    print(R)
    assert np.isclose(np.linalg.det(A), np.prod(np.diag(R)))  # determinant check

    # random invertible 3x3
    A = np.matrix([[1, 2, 3],
                   [0, 1, 4],
                   [5, 6, 0]])
    R = gaussian_elimination(A)
    print("====")
    print(A)
    print(R)
    assert np.isclose(np.linalg.det(A), np.prod(np.diag(R)))  # determinant check



if __name__ == "__main__":
    main()