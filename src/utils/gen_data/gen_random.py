import numpy as np


class RandomStateGenerator:
    @staticmethod
    def gen_unitary(n):
        n = 2 ** n
        H = np.random.rand(n, n)
        Q, R = np.linalg.qr(H)
        return Q

    # generate random product states
    # by first generating a random unitary matrix
    # and multiplying it onto a constant beginning vector
    # which gives us a random 1-qubit state
    # we then tensor product n of these together
    # this gives a 2^n dim vector representable by 2n pieces of info
    def make_dset(self, n_qubits):
        u = self.gen_unitary(1)
        prod = (u * np.array([1, 0]).T).sum(1)

        for i in range(n_qubits-1):
            u = self.gen_unitary(1)
            v = (u*np.array([1,0]).T).sum(1)

            prod = np.kron(prod, v)

        prod = prod * np.conj(prod).T

        return prod

