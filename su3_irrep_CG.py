# SU(3) irreps Clebsch-Gordan. This works!
# Still, very rudimentary. Next step: generalized to SU(N)!

import copy

class Tableau(): # Young tableau for SU(3) irrep
    def __init__(self, n, m):
        self.idx1 = n
        self.idx2 = m
        self.dim = (n + 1) * (m + 1) * (n + m + 2) // 2
        self.bar = n < m

    def __repr__(self):
        index_1 = str(self.idx1)
        index_2 = str(self.idx2)
        display_dim = str(self.dim)
        if self.bar:
            display_dim += ' bar'
        description = "({idx1}, {idx2})[{dim}]".format(idx1=index_1, idx2=index_2, dim=display_dim)
        return description

    def matrix(self):
        init_matrix = []
        # First row
        if self.idx1 + self.idx2 > 0:
            init_matrix.append(['1'] * (self.idx1 + self.idx2))
        # Second row
        if self.idx2 > 0:
            init_matrix.append(['2'] * self.idx2)
        return init_matrix


# Add a box to tableau -- could be 'a' or 'b' (NOT USED)
def add_box_to_tableau(tableau, c='a'):
    result1 = tableau.matrix()
    number_of_rows = len(tableau.matrix())
    # result1: add c to the right of 1st row
    if number_of_rows == 0:
        result1.append([c])
    else:
        result1[0].append(c)
    result = [result1]
    # result2: add c to the right of 2nd row
    if number_of_rows >= 1:
        result2 = tableau.matrix()
        if number_of_rows == 1:
            result2.append([c])
            result.append(result2)
        elif len(result2[0]) > len(result2[1]):
            result2[1].append(c)
            result.append(result2)
    # result3: add c to 3rd row
    if number_of_rows >= 2:
        result3 = tableau.matrix()
        result3.append([c])
        result.append(result3)
    return result


# Add a box to the right of a specific row_index (NOTE: have to defind a deep copy first!)
def add_box_to_row(input_matrix, c, row_index):
    n_rows = len(input_matrix)
    result = copy.deepcopy(input_matrix)
    if row_index < n_rows:
        result[row_index].append(c)
    elif row_index == n_rows:
        result.append([c])
    return result

# Convert matrix back to Tableau
def to_tableau(matrix):
    number_of_rows = len(matrix)
    idx1 = 0
    idx2 = 0
    if number_of_rows == 1:
        idx1 = len(matrix[0])
    elif number_of_rows == 2:
        idx2 = len(matrix[1])
        idx1 = len(matrix[0]) - len(matrix[1])
    else:
        idx2 = len(matrix[1]) - len(matrix[-1])
        idx1 = len(matrix[0]) - len(matrix[1])
    return Tableau(idx1, idx2)


# Add a box to tableau and convert back (NOT USED)
def tab_add_c_then_to_tab(tableau, c='a'):
    lst =  add_a_box_to_tableau(tableau, c)
    return [to_tableau(item) for item in lst]


# number of rows check
def n_rows_check(matrix):
    if len(matrix) > 3:
        return False
    return True

# row length check
def row_length_check(matrix):
    if len(matrix) == 0:
        return True
    else:
        row_lengths = [len(matrix[i]) for i in range(len(matrix))]
        # row length check
        for i in range(len(matrix) - 1):
            if row_lengths[i] < row_lengths[i + 1]:
                return False
    return True

# 'b_count <= a_count' check
def b_a_check(matrix):
    flag = True
    lst_b_a = []
    for i in range(len(matrix)):
        lst_b_a += matrix[i][::-1]
    b_count = 0
    a_count = 0
    k = 0
    while flag and ( k < len(lst_b_a) ):
        if lst_b_a[k] == 'a':
            a_count += 1
        if lst_b_a[k] == 'b':
            b_count += 1
        flag = (b_count <= a_count)
        k += 1
    return flag

# no double 'a' or 'b' check (NOTE: need a deepcopy too, in order not to affect input matrix)
def no_double_check(matrix):
    cpy = copy.deepcopy(matrix)
    if len(cpy) == 0:
        return True
    col_num = len(cpy[0])
    row_num = len(cpy)
    row_lengths = [len(cpy[i]) for i in range(row_num)]
    # fill in 0's to form a full rectangle
    for i in range(1, row_num):
        cpy[i] += ['0'] * (col_num - row_lengths[i])
    for j in range(col_num):
        a_count = 0
        b_count = 0
        k = 0
        while (k < row_num) and (cpy[k][j] != '0'):
            if cpy[k][j] == 'a':
                a_count += 1
            if cpy[k][j] == 'b':
                b_count += 1
            if (a_count >= 2) or (b_count >= 2):
                return False
            k += 1
    return True

# overall check
def big_check(matrix):
    return (row_length_check(matrix) and b_a_check(matrix)) and (no_double_check(matrix) and n_rows_check(matrix))

# checked add_box_to_matrix_checked
def add_box_to_matrix_checked(matrix, c='a'):
    n_rows = len(matrix)
    lst = []
    for i in range(n_rows + 1):
        temp = add_box_to_row(matrix, c, i)
        if big_check(temp):
            lst.append(temp)
    return lst

# add box to a list of matrices checked, avoid duplication too
def add_box_to_matrix_list_checked(input_lst, c='a'):
    output = []
    for matrix in input_lst:
        temp1 = add_box_to_matrix_checked(matrix, c)
        for x in temp1:
            if x not in output:
                output.append(x)
    return output

# Finally, tensor "multiply" 2 tableaux!
def multiply_tableaux(tab1, tab2):
    num_a = tab2.idx1 + tab2.idx2
    num_b = tab2.idx2
    if num_a == 0:
        return [tab1]
    total_lst = [tab1.matrix()]
    for i in range(num_a):
        total_lst = add_box_to_matrix_list_checked(total_lst, 'a')
    for i in range(num_b):
        total_lst = add_box_to_matrix_list_checked(total_lst, 'b')
    return [to_tableau(matrix) for matrix in total_lst]


# some tests
irreps1 = [Tableau(0, 0), Tableau(1, 0), Tableau(0, 1), Tableau(2, 0), Tableau(0, 2)]
irreps2 = [Tableau(1, 1), Tableau(3, 0), Tableau(0, 3), Tableau(2, 1), Tableau(1, 2), Tableau(2, 2)]

print(multiply_tableaux(irreps2[0], irreps2[0]))

# other intermediate tests
"""
for irrep in irreps:
    print([to_tableau(item) for item in add_box_to_matrix_checked(irrep.matrix(), 'a')], end=';\n')

for irrep in irreps:
    lst = []
    for i in range( 1 + len(irrep.matrix()) ):
        temp = add_box_to_row(irrep.matrix(), 'a', i)
        if big_check(temp):
            lst.append(temp)
    print(lst)

trial1 = [[1, 'a'], ['a']]
trial2 = [[1, 'a'], [2, 'a']]
trial3 = [[1, 1, 'a'], [2, 'a', 'a']]
trial4 = [[1, 1, 'a'], [2, 'a', 'b']]
trial5 = []
trial6 = [[1, 1, 'a'], [2, 'b', 'b']]
trials = [trial1, trial2, trial3, trial4, trial5, trial6]
for trial in trials:
    print(no_double_check(trial), b_a_check(trial), row_length_check(trial))
"""
