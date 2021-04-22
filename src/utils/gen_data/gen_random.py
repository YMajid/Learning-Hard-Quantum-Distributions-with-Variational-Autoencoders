import numpy as np

# Code for generating the random hard state, adapted from original authors

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
    def make_dset_easy(self, n_qubits):
        u = self.gen_unitary(1)
        prod = (u * np.array([1, 0]).T).sum(1)

        for i in range(n_qubits-1):
            u = self.gen_unitary(1)
            v = (u*np.array([1,0]).T).sum(1)

            prod = np.kron(prod, v)

        prod = prod * np.conj(prod).T

        return prod

    def make_dset_hard(self, n_qubits):
        # sample from 2^n dim complex unit sphere
        # via sampling 2^2n real unit sphere and then just making it complex
        n = n_qubits+1 # or any positive integer
        x = np.random.normal(size=(1, 2**n))
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
        x = x.reshape(2**n_qubits, 2)
        y = np.empty(x.shape[0], dtype=complex)
        y.real=x[:,0]
        y.imag=x[:,1]
        del x

        y /= np.linalg.norm(y) # normalize
        out = (y * np.conj(y).T).real # still a complex dtype so take .real

        return out

