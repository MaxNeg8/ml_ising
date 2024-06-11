"""
This script is meant for pre-processing the input to the neural networks. Currently, the preprocessing
does the following:
    
    * Map any configuration matrix M inside the set of configuration matrices accesible from
    M via any number of cyclic permutations of rows and/or columns P to the same element of P,
    thereby taking care of translational symmetry before passing inputs to the network
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def _cleanup(result, n):
    """
    Helper function to process the intermediate results of the _count function.
    Takes index pairs of starting/ending strings of zeros and converts them to
    pairs of numbers, the first indicating the starting index of a string and the
    second one indicating the number of zeros in the string. Also takes care of periodic
    boundary conditions by wrapping the last string of zeros around to the front if necessary.

    Parameters
    ----------
    result : list
        List of results in the above-described format

    n : int
        Lattice sites at edge (only square lattice)

    Returns
    -------
    to_return : list
        Converted results as a list with entries (starting_index, number_of_zeros)
    """
    if len(result) == 0:
        return [[0, 0]] # Numba bs for empty lists

    if len(result) == 1 and result[0][1] - result[0, 0] == n:
        return [[0, 0]] # If the entire row is zeros, skip it (or we get into trouble later)

    to_return = [[entry[0], entry[1] - entry[0]] for entry in result]

    # Wrap around if first and last spins are zero
    if len(to_return) > 1:
        if (to_return[0][0] == 0) and (to_return[-1][0] + to_return[-1][1] == n):
            to_return[-1][1] += to_return[0][1]
            to_return.pop(0)
    return to_return

@jit(nopython=True)
def _count(a):
    """
    Helper function that counts the number of consecutive zeros in the individual
    rows and columns of a configuration matrix and returns the result as two
    lists, one for the columns and one for the rows.

    Parameters
    ----------
    a : np.ndarray
        The square configuration matrix

    Returns
    -------
    row_results : list
        A list containing the positions and counts of strings of consecutive zeros
        inside the rows of the configuration matrix. It is a list of lists of pairs
        of values of format (starting_index, number_of_zeros)

    col_results : list
        A list containing the positions and counts of strings of consecutive zeros
        inside the columns of the configuration matrix. It is a list of lists of pairs
        of values of format (starting_index, number_of_zeros)
    """
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]

    row_results = []
    col_results = []

    padded = np.zeros((n+2, n+2)) # Pad with zeros to enable taking differences
    padded[1:-1, 1:-1] = np.logical_not(a) # We can't use np.pad because numba doesn't support it

    row_switch = np.abs(padded[1:-1, 1:] - padded[1:-1, :-1]) # We also can't use np.diff because numba doesn't support it
    col_switch = np.abs(padded[1:, 1:-1] - padded[:-1, 1:-1])

    # Build the results by noting the indices where 0 changes to 1 and vice-versa
    for i in range(n):
        row_results.append(np.where(row_switch[i] == 1)[0].reshape(-1, 2))
        col_results.append(np.where(col_switch[:, i] == 1)[0].reshape(-1, 2))

    row_results_cleanup = [] # More Numba bs because it can't overwrite entries
    col_results_cleanup = []

    # Clean up the results to have the correct format (see docstring of function _cleanup)
    for i in range(n):
        row_results_cleanup.append(_cleanup(row_results[i], n))
        col_results_cleanup.append(_cleanup(col_results[i], n))
    
    return row_results_cleanup, col_results_cleanup


def _find_largest_number_of_zeros(count_results):
    """
    Helper function that finds the largest number of consecutive zeros inside of the rows
    or columns of a configuration matrix, taking as an input one of the resulting lists
    of function _count. The function thus acts as a filter for the results of the _count
    function, only keeping those entries where the number of consecutive zeros is
    maximal with respect to the entire input array.
    Ideally, this will leave us with a single pair of numbers, but realistically, multiple
    rows/columns will have the same (maximal) number of consecutive zeros.

    Parameters
    ----------
    count_results : list
        One of the resulting lists (row or columns) from the _count function that will be
        filtered

    Returns
    -------
    to_return : list
        The results of the _count function filtered for the maximum number of consecutive
        zeros as described in the docstring above
    """
    largest = []
    maximum = 0 # Keep track of the largest number of consecutive zeros
    for row_col in count_results:
        row_col = np.array(row_col)
        largest.append(row_col[row_col[:, 1] == np.max(row_col[:, 1])])
        maximum = largest[-1][0, 1] if largest[-1][0, 1] > maximum else maximum

    to_return = []

    for row_col_index in range(len(largest)):
        if largest[row_col_index][0, 1] == maximum:
            to_return.append((row_col_index, largest[row_col_index][:, 0]))

    return to_return

def _array_to_number(array):
    """
    Helper function that converts an array of zeros and ones to a decimal number by
    interpreting the array as a binary number

    Parameters
    ----------
    array : np.ndarray
        The array representing the binary number

    Returns
    -------
    number : int
        The corresponding decimal number
    """
    return np.dot(array[::-1], 2**np.arange(len(array)))

def _take_lowest_largest_number_of_zeros(a, largest_number_of_zeros, row_col="row"):
    """
    Helper function that, given some input rows/cols along with the corresponding starting
    index/indices of strings of zeros, shifts each row/column such that the smallest binary
    number possible is produced. It then returns only those number pairs from the input
    that correspond to the lowest number achieved (ideally one, multiple if those generate
    the same number). The input data is taken from the output of function _find_largest_number_of_zeros

    Parameters
    ----------
    a : np.ndarray
        The square spin configuration

    largest_number_of_zeros : list
        The list coming from function _find_largest_number_of_zeros containing the starting
        indices and lengths of strings of zeros

    row_col : str
        Whether the input data corresponds to rows or columns (will determine for which axis the
        lowest numbers are computed). Either "row" or "col"
    
    Returns
    -------
    result : list
        The filtered input (filtered by the criterion described in the docstring above)
    """
    n = a.shape[0]
    lowest = 2**n-1 # The highest possible number for n rows/cols
    results = []

    for candidate_tuple in largest_number_of_zeros:
        row_col_index, candidate = candidate_tuple
        array = a[row_col_index] if row_col == "row" else a[:, row_col_index]
        for entry in candidate:
            number = _array_to_number(np.roll(array, -entry)) # Roll the candidate to the beginning of the string of zeros
            if number < lowest:
                lowest = number
                results = [(row_col_index, entry)] # Reset the results list, discarding all previous entries with higher numbers
            elif number == lowest:
                results.append((row_col_index, entry))

    return results

def _test_neighbours(a, lowest_largest_number_results, row_col="row"):
    """
    Helper function that again filters the output of function _take_lowest_largest_number_of_zeros. This time, only the
    entry survives whose neighbouring rows/cols with lower indices (starting with -1) have the lowest number. If this
    still produces multiple row/col candidates due to high symmetry of the matrix, all of those candidates are returned.

    Parameters
    ----------
    a : np.ndarray
        Square spin configuration

    lowest_largest_number_results : list
        The output from the function _take_lowest_largest_number_of_zeros that is to be filtered

    row_col : str
        String that determines whether the input data corresponds to the rows or the columns of the spin matrix.
        Either "row" or "col".

    Returns
    -------
    lowest_largest_number_results : list
        The results filtered according to the criterion described in the docstring above
    """
    n = a.shape[0]
    # Compute all rolls of the matrix a that correspond to the remaining input row/col candidates
    rolled_as = [np.roll(a, -entry[1], axis= 1 if row_col == "row" else 0) for entry in lowest_largest_number_results]
    skipping = []
    for i in range(n-1): # Test all n-1 lower neighbouring rows/cols until we find one row/col with a lower number than the others
        lowest = 2**n-1
        results = []
        for j, a in enumerate(rolled_as):
            if j in skipping:
                continue
            row_col_index = lowest_largest_number_results[j][0]
            array = a[row_col_index-i] if row_col == "row" else a[:, row_col_index-i]
            number = _array_to_number(array)
            if number < lowest:
                lowest = number
                skipping += results
                results = [j]
            elif number == lowest:
                results.append(j)
            else:
                skipping.append(j)
        if len(results) == 1:
            return [lowest_largest_number_results[results[0]]] # If there is only one lowest number, we stop
    return lowest_largest_number_results

def _bruteforce(a):
    """
    Helper function that bruteforces the lexicographically smallest matrix that can be generated through
    permutations of the matrix a. This is the last resort in our quest to map a set of permutated matrices
    to a single matrix, as it scales with N**2. There are algorithms that achieve this with complexity N log(N),
    but since we do lots of steps prior to this and don't have that large matrices to begin with, this simple
    algorithm suffices.

    Parameters
    ----------
    a : np.ndarray
        Square spin configuration matrix

    Returns
    -------
    smallest_matrix : np.ndarray
        The lexicographically smallest matrix that can be generated through cyclic shifts of rows and/or cols of the
        input matrix
    """
    n = a.shape[0]
    smallest_matrix = a.copy()
    smallest_matrix_string = "".join(map(str, a.flatten()))
    for i in range(n):
        for j in range(n): 
            a = np.roll(a, 1, axis=1)
            a_string = "".join(map(str, a.flatten()))
            if a_string < smallest_matrix_string:
                smallest_matrix = a.copy()
                smallest_matrix_string = a_string
        a = np.roll(a, 1, axis=0)
    return smallest_matrix


def preprocess(a):
    """
    Function that pre-processes a square spin configuration before it is handed to a neural network. It
    maps the set of all matrices that can be obtained by cyclic shifts of the input matrix to a single
    matrix, thereby forcing the network results to be invariant under cyclic shifts of rows and columns.

    It achieves this through the following procedure:
        1. Try to find the row with the largest number of consecutive zeros. If there is only one and the
        position of those zeros is unique within the row, shift the entire matrix horizontally to the
        starting position of the string of zeros of that row.
        2. If there are multiple rows with the same number of zeros or multiple strings of zeros with the
        same, maximal number of zeros inside of a column or multiple columns, we perform a series of filters
        on the previously obtained results. The first of those being that we only select those rows that,
        when shifted to the starting index of the individual strings of zeros, compute to the smallest
        binary number when each entry is viewed as a single bit.
        3. If this still does not produce unique results, we investigate the neighbouring rows of the
        candidates, starting with index -1 and going down. The first ith neighbour to have a smaller number than
        all the other candidates' ith neighbours is then selected.
        4. If, after all this, there are still multiple candidates, we employ a bruteforcing scheme where we
        comput the lexicographically smallest permutation of the matrix, ensuring a unique result

    Parameters
    ----------
    a : np.ndarray
        The square spin configuration matrix

    Returns
    -------
    result : np.ndarray
        A matrix that is the same for all input matrices a that can be generated from one another through cyclic shifts
        of rows and/or columns
    """
    row_shift = 0
    col_shift = 0

    row, col = _count(a)
    largest_number_of_zeros_row = _find_largest_number_of_zeros(row)
    lowest_largest_number_results_row = _take_lowest_largest_number_of_zeros(a, largest_number_of_zeros_row, "row")
    if len(lowest_largest_number_results_row) > 1:
        neighbour_results_row = _test_neighbours(a, lowest_largest_number_results_row, "row")
        if len(neighbour_results_row) > 1:
            return _bruteforce(a)
        else:
            row_shift = -neighbour_results_row[0][1]
    else:
        row_shift = -lowest_largest_number_results_row[0][1]

    largest_number_of_zeros_col = _find_largest_number_of_zeros(col)
    lowest_largest_number_results_col = _take_lowest_largest_number_of_zeros(a, largest_number_of_zeros_col, "col")
    if len(lowest_largest_number_results_col) > 1:
        neighbour_results_col = _test_neighbours(a, lowest_largest_number_results_col, "col")
        if len(neighbour_results_col) > 1:
            return _bruteforce(a)
        else:
            col_shift = -neighbour_results_col[0][1]
    else:
        col_shift = -lowest_largest_number_results_col[0][1]

    return np.roll(np.roll(a, col_shift, axis=0), row_shift, axis=1)
