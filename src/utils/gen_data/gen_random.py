import numpy as np


class RandomStateGenerator:
    @staticmethod
    def gen_unitary(n):
        n = 2 ** n
        H = np.random.rand(n, n)
        Q, R = np.linalg.qr(H)
        return Q
