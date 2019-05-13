from su3_irrep_CG import Irrep, multiply_irreps
import csv


# Return as string the result of Clebsch-Gordan decomposition from the tensor product of two SU(3) irreps, or error message
def irrep_result_str(lst):
    message = "Need two pairs of nonnegative integers as Dynkin coefficients, try again!"
    if lst:
        tab1 = Irrep(lst[0], lst[1])
        tab2 = Irrep(lst[2], lst[3])
        result = multiply_irreps(tab1, tab2)
        rhs = ''
        for tab in result:
            rhs += str(tab) + ' \u2295 '
        rhs = rhs[:-3]
        message = str(tab1) + ' \u2297 ' + str(tab2) + " = " + rhs
    return message


# Write to file -- by default, input file = 'test_cases.csv', output file = 'result.txt'
def write_to_file(input_file='test_cases.csv', output_file='result.txt'):
    with open(input_file, newline='') as tests:
        tests_reader = csv.reader(tests)
        tests_input = []
        for row in tests_reader:
            input_row = []
            for item in row:
                try:
                    if int(item) < 0:
                        break
                    else:
                        input_row.append(int(item))
                except ValueError:
                    input_row = []
                    break
            if len(input_row) <= 3:
                input_row = []
            tests_input.append(input_row)
    with open(output_file, 'w') as results:
        for line in tests_input:
            results.write(irrep_result_str(line)+'\n')


write_to_file()
