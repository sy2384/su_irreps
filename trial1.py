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
            init_matrix.append([1] * (self.idx1 + self.idx2))
        # Second row
        if self.idx2 > 0:
            init_matrix.append([2] * self.idx2)
        return init_matrix

# Add a box to tableau -- could be 'a' or 'b'
def add_a_box_to_tableau(tableau, c='a'):
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

# Add a box to tableau and convert back
def tab_add_c_then_to_tab(tableau, c='a'):
    lst =  add_a_box_to_tableau(tableau, c)
    return [to_tableau(item) for item in lst]

# row length check
def row_length_check(matrix):
    row_lengths = [len(matrix[i]) for i in range(len(matrix))]
    # row length check
    for i in range(len(row_lengths) - 1):
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

# no double 'a' or 'b' check
def no_double_check(matrix):
    copy = matrix
    if len(copy) == 0:
        return True
    col_num = len(copy[0])
    row_num = len(copy)
    # fill in 0's to form a full rectangle
    for i in range(1, row_num):
        copy[i] += ( [0] * (col_num - len(matrix[i])) )
    a_count = 0
    b_count = 0
    for j in range(col_num):
        k = 0
        while (k < row_num) and (copy[k][j] != 0):
            if copy[k][j] == 'a':
                a_count += 1
            if copy[k][j] == 'b':
                b_count += 1
            if (a_count >= 2) or (b_count >= 2):
                return False
            k += 1
    return True

# big_check
def big_check(matrix):
    return (row_length_check(matrix) and b_a_check(matrix)) and no_double_check(matrix)

# One iteration for a given list of tableaux (unfinished, no check)
def add_a_box_list(input_lst, c='a'):
    lst1 = []
    for tab in input_lst:
        temp = tab_add_c_then_to_tab(tab, c)
        lst1 += temp
    return lst1

# Finally, tensor "multiply" 2 tableaux!
def multiply_tableaux(tab1, tab2):
    num_a = tab2.idx1 + tab2.idx2
    num_b = tab2.idx2
    if num_a == 0:
        return [tab1]
    CG_decomp = [tab1]
    for i in range(num_a):
        CG_decomp = add_a_box_list(CG_decomp, 'a')
    for i in range(num_b):
        CG_decomp = add_a_box_list(CG_decomp, 'b')
    return [thing for thing in CG_decomp]

# some tests
irreps = [Tableau(0, 0), Tableau(1, 0), Tableau(0, 1), Tableau(2, 0), Tableau(0, 2), Tableau(1, 1), Tableau(2, 1), Tableau(1, 2), Tableau(0, 3), Tableau(2, 2)]
"""
for irrep in irreps:
    print([tab_add_c_then_to_tab(irrep, 'a')])
"""
print(multiply_tableaux(irreps[1], irreps[2]))
