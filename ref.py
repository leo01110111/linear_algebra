import numpy as np
def is_ref(A: np.ndarray) -> bool:
    """Check if matrix A is in Row Echelon Form (REF)."""
    rows, cols = A.shape
    last_pivot_col = -1
    zero_row_started = False

    for r in range(rows):
        row = A[r, :]
        
        # Find first nonzero (pivot) in this row
        nonzeros = np.nonzero(row)[0]
        if len(nonzeros) == 0:  # row of all zeros
            zero_row_started = True
            continue
        
        pivot_col = nonzeros[0]
        
        # Rule 1: zero rows must be at the bottom
        if zero_row_started:
            return False
        
        # Rule 2: pivot must be to the right of last pivot
        if pivot_col <= last_pivot_col:
            return False
        
        # Rule 3: below pivot must be all zero
        for rr in range(r+1, rows):
            if A[rr, pivot_col] != 0:
                return False
        
        last_pivot_col = pivot_col

    return True