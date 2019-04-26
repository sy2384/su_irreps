class Tableau(): # A tableau for SU(3) irrep
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
            display_dim += " bar"
        description = "({idx1}, {idx2}), {dim}".format(idx1=index_1, idx2=index_2, dim=display_dim)
        return description

# some tests
irreps = [Tableau(1, 0), Tableau(0, 1), Tableau(2, 0), Tableau(0, 2), Tableau(1, 1), Tableau(2, 1), Tableau(1, 2), Tableau(0, 3), Tableau(2, 2)]
for irrep in irreps:
    print(irrep)
