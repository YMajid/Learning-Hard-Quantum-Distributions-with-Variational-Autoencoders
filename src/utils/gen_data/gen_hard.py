import numpy as np
import itertools


# Code for generating the hard state, adapted from original authors
class HardStateGenerator:
    def __init__(self, n, L):
        """
        Args:
            n, L: somehow combined to get number of qubits ¯\_(ツ)_/¯
        """
        self.n = n
        self.L = L

    # Get the factorial representation of the number s
    def get_factoradic(self, s):
        f = np.zeros(self.n)
        for i in range(self.n):
            k = self.n - i - 1
            if np.math.factorial(k) > s:
                f[i] = 0
            else:
                f[i] = s // np.math.factorial(k)
                s = s % np.math.factorial(k)
        return f

    # Find the permutation vector given the factoradic
    def get_permutation_vector(self, factoradic):
        seq = np.linspace(0, self.n - 1, self.n)
        perm_vec = np.zeros(self.n)
        for i in range(self.n):
            perm_vec[i] = int(seq[int(factoradic[i])])
            seq = np.delete(seq, int(factoradic[i]))
        return perm_vec

    # Map the permutation vector to a flattened permutation matrix
    def get_binary_permutation_matrix_flat(self, perm_vector):
        idm = np.eye(perm_vector.shape[0])
        return np.reshape(
            idm[perm_vector.astype("int"), :], (1, perm_vector.shape[0] ** 2)
        )

    # All permutations with repetition: Total permutations = L PermRep n**2
    def get_permutations_with_repetition(self):
        x = np.linspace(0, self.L - 1, self.L).astype(int)
        perms = [p for p in itertools.product(x, repeat=self.n ** 2)]
        return np.asarray(perms)

    def get_binary_permutation_matrices_flat(self):
        binary_permutation_matrices_flat = np.zeros(
            (np.math.factorial(self.n) - 1, self.n ** 2)
        )
        for s in range(np.math.factorial(self.n) - 1):
            factoradic = self.get_factoradic(s)
            permutation_vector = self.get_permutation_vector(factoradic)
            binary_permutation_matrix_flat = self.get_binary_permutation_matrix_flat(
                permutation_vector
            )
            binary_permutation_matrices_flat[s,
            :] = binary_permutation_matrix_flat
        return binary_permutation_matrices_flat

    def get_hard_distribution(self):
        P = []
        h = self.get_binary_permutation_matrices_flat()
        x = np.linspace(0, self.L - 1, self.L).astype(int)
        perms = itertools.product(x, repeat=self.n ** 2)

        for i, p in enumerate(perms):
            y = p
            omega = np.exp((2 * np.pi * np.complex(1j)) / self.L * 1.0)
            z = np.power(omega, y)
            q = np.sum(np.prod(np.power(z, h), 1))
            P.append(np.real(np.conj(q) * q) / (
                    self.L ** (self.n ** 2) * (np.math.factorial(self.n) - 1)
            ))
        return np.asarray(P)
