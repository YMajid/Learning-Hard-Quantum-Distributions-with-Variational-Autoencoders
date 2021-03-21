from quspin.operators import hamiltonian
import numpy as np


class EasyStateGenerator:

    # set total number of qubits
    @staticmethod
    def get_time_evolve_state(L, times):
        # L=8

        # constructing the coupling matrix J_ij

        # alpha is the exponent for the long-range decay
        alpha = 0.9  # used to be 0.75

        J_indx = []

        # define operator strings
        for i in range(L):
            for j in range(L):
                if i != j:
                    # d=np.min([np.abs(i-j),np.abs(i-j-L),np.abs(i-j+L)])
                    d = np.abs(i - j)

                    jval = 1.0 / d ** alpha
                    J_indx.append([jval * 0.5, i, j])

        xx_list = [["xx", J_indx]]
        yy_list = [["yy", J_indx]]

        H = 0.5 * hamiltonian(
            xx_list, [], N=L, dtype=np.float64, check_herm=False, check_symm=False
        )
        H += 0.5 * hamiltonian(
            yy_list, [], N=L, dtype=np.float64, check_herm=False, check_symm=False
        )

        # times=np.arange(0,5.01,1.0)

        # starting from a flat state
        psi = np.ones(2 ** L) / np.sqrt(2 ** L)

        ##    startin from a neel state
        ##    10101010..
        #    psi=np.zeros(2**L)
        #
        #    neelindex=0
        #    for i in range (0,L,2):
        #        	neelindex+=1<<i
        #
        #    psi[neelindex]=1.0

        vf = H.evolve(
            psi, 0, times, solver_name="vode", iterate=False, atol=1.0e-6, rtol=1.0e-6
        )
        return vf
