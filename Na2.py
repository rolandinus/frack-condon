import molecule
import unit_conv as uc
import numpy as np

class Na2(molecule.Molecule):
    m1 = uc.molecular_weight_to_me(45.9795)/2
    m2 = m1
    M = m1/2
    #m2 = 3.81754E-26/9.1094e-31
  #  m = uc.g_to_me(2*3.8175458e-23)/2

    def __init__(self, state, J):
        if state == "X":
            Ed = uc.cm_to_hatree(6022.0)
            Re = uc.angst_to_a0(3.0787)
            ome = 2 * np.pi * uc.wavenumber_to_frequency(159.20)
            min_electronic_energy = 0
            full_name = '$X^1Σ_g^+$'

        elif state == "A":
            Ed = uc.cm_to_hatree(8310) #
            Re = uc.angst_to_a0(3.6384)
            #ome = 2 * np.pi * 1 / uc.cm_to_a0(1 / 117.323)
            #ome = 2 * np.pi * 1 / uc.m2nat(1 / 11732.3)
            ome = 2 * np.pi * uc.wavenumber_to_frequency(117.323)
            min_electronic_energy = uc.cm_to_hatree(14680)
            full_name = '$A^1Σ_u^+$'

        elif state == "B":
            Re = uc.angst_to_a0(3.4226)
            Ed = uc.cm_to_hatree(2671)  # uc.cm_to_hatree(self.termval[0][3])) # from hessel und kusch chem phys 1978
            #ome = 2 * np.pi * 1 / uc.cm_to_a0(1 / 124.09)
            ome = 2 * np.pi * uc.wavenumber_to_frequency(124.09)
            # https: // aip.scitation.org / doi / pdf / 10.1063 / 1.452019
            min_electronic_energy = uc.cm_to_hatree(20320)
            full_name = '$B^1Π_u$'
        super().__init__("Na2", state, J, Na2.m1, Na2.m2, Re, ome, Ed, min_electronic_energy, full_name)

