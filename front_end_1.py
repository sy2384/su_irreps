from su3_irrep_CG import Tableau, multiply_tableaux

def enter_irrep(n):
    while True:
        flag = True
        irrep_str = input("Please enter SU(3) irrep {num} (n, m): ".format(num=str(n)))
        irrep = []
        for index_str in irrep_str.split(','):
            try:
                irrep.append(int(index_str))
            except ValueError:
                print("Need 2 Dynkin coefficients, try again!")
                flag = False
                break
        if flag:
            try:
                return Tableau(irrep[0], irrep[1])
            except IndexError:
                print("Need 2 Dynkin coefficients, try again!")

def output_irrep():
    tab1 = enter_irrep(1)
    tab2 = enter_irrep(2)
    result = multiply_tableaux(tab1, tab2)
    rhs = ''
    for tab in result:
        rhs += str(tab) + ' \u2295 '
    rhs = rhs[:-3]
    message = str(tab1) + ' \u2297 ' + str(tab2) + " = " + rhs
    print(message)

output_irrep()
