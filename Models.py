import numpy as np
import scipy.integrate as integrate


class AlphaModel:
    def __init__(self, Omega_p, Omega_c, gamma_p, gamma_c, Gamma_e, Q_E, n_0, sigma_z, sigma_r, texp, C_3, C_6, N_i):
        self.Omega_p = Omega_p*2*np.pi
        self.Omega_c = Omega_c*2*np.pi
        self.gamma_p = gamma_p*2*np.pi
        self.gamma_c = gamma_c*2*np.pi
        self.Gamma_e = Gamma_e*2*np.pi
        self.gamma_gR = self.gamma_p + self.gamma_c
        self.gamma_eg = self.Gamma_e/2 + self.gamma_p
        self.n_0 = n_0
        self.N_i = N_i
        self.sigma_z = sigma_z
        self.sigma_r = sigma_r
        self.texp = texp
        self.C_3 = C_3
        self.C_6 = C_6
        self.Q_E = Q_E
        self.z_grid = np.linspace(-200, 200, 401)
        self.prefactor = 0.6827 * (2 * np.pi / 0.78)

        self.step_count = 0
        self.rejected_values = 0
        self.rejection_ratio = 1
        # self.rejected_values_oob=0
        # self.rejection_ratio_oob=1

        self.Omega_p_list = []
        self.Omega_c_list = []
        self.Omega_p_over_c_list = []
        self.alpha_list = []
        self.n_0_list = []
        self.min_counts_i_list = []
        self.min_counts_e_list = []
        self.n_counts_list = []
        self.min_counts_over_n_counts = []
        self.peak_f_bl_list = []

    def n_counts(self):
        return (self.Omega_p / 2.72) ** 2 * self.texp

    def rho_0(self):
        return self.Omega_p ** 2 / self.Omega_c ** 2

    def R_3(self):
        return (2 * self.C_3 * self.gamma_eg / (self.Omega_c ** 2 + self.gamma_eg * self.gamma_gR)) ** (1 / 3)

    def R_6(self):
        return (2 * self.C_6 * self.gamma_eg / (self.Omega_c ** 2 + self.gamma_eg * self.gamma_gR)) ** (1 / 6)

    def V_bl_3(self):
        return 4 / 3 * np.pi * self.R_3() ** 3

    def V_bl_6(self):
        return 4 / 3 * np.pi * self.R_6() ** 3

    def ground_state_density(self):
        return self.n_0 * np.exp(-0.5 * (self.z_grid / self.sigma_z) ** 2)

    def impurity_density(self):
        return self.N_i / (self.R_3() * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (self.z_grid / self.R_3()) ** 2)

    def f_bl_simple(self):
        return self.rho_0() * self.V_bl_6() * self.ground_state_density()

    def f_bl(self):
        zero_crossings = np.zeros(401)
        Ns = np.linspace(1, 2000, 2000)
        l1 = (self.Omega_p ** 2 + self.Omega_c ** 2) * (
                    self.Gamma_e * self.Omega_c ** 2 + self.gamma_eg * self.Omega_p ** 2)
        i = 0
        for d in self.ground_state_density():
            # print(d)
            Dint = self.C_6 * (4 / 3 * np.pi * d / Ns) ** 2
            l = 4 * self.gamma_eg * Dint ** 2 * (self.Gamma_e * self.gamma_eg + 2 * self.Omega_p ** 2)
            self_con = (1 - l / (l + l1)) * ((Ns - 1) * self.rho_0() + 2) - 1
            # print(self_con)
            # print(Ns[np.where(np.diff(np.signbit(self_con)))[0]][0]*self.rho_0())
            try:
                zero_crossings[i] = Ns[np.where(np.diff(np.signbit(self_con)))[0]][0] * self.rho_0()
            except:
                None
            i += 1
        # print('zero crossings',zero_crossings)
        return zero_crossings

    def f_ir(self):
        return 1 - (1 / (1 + self.impurity_density() * self.V_bl_3()))

    def N_bl_3(self):
        return self.V_bl_3() * self.ground_state_density()

    def N_bl_6(self):
        return self.V_bl_6() * self.ground_state_density()

    def chi_2_lvl(self):
        return self.prefactor / (self.gamma_eg + (self.Omega_p ** 2) / (2 * self.gamma_eg))
        #return self.prefactor * self.Gamma_e * self.gamma_eg / (self.gamma_eg ** 2 + 2 * self.Omega_p ** 2 * (
          #self.Gamma_e + self.gamma_eg) / self.Gamma_e)

    def chi_3_lvl(self):
        return self.prefactor / (self.gamma_eg + (self.Omega_c ** 2) / (4 * self.gamma_gR))
        #return self.chi0 * self.Gamma_e / (self.gamma_eg + (self.Omega_c ** 2) / self.gamma_gR)
        #return self.chi0 * self.gamma_gR / (self.gamma_gR * self.gamma_eg + self.Omega_c ** 2)

    def transmission_2lvl(self):
        return np.e ** (integrate.simps(- self.ground_state_density() * self.chi_2_lvl(), self.z_grid))

    def transmission_single_eit(self):
        return np.e ** (integrate.simps(- self.ground_state_density() * self.chi_3_lvl(), self.z_grid))

    def transmission_simple_eit(self):
        return np.e ** (integrate.simps(- self.ground_state_density() * (self.chi_3_lvl() + self.chi_2_lvl() *
                                        self.f_bl_simple()) / (1 + self.f_bl_simple()), self.z_grid))

    def transmission_eit(self):
        return np.e ** (integrate.simps(- self.ground_state_density() * (self.chi_3_lvl() + self.chi_2_lvl() *
                                        self.f_bl()) / (1 + self.f_bl()), self.z_grid))

    def integrand(self):
        return self.ground_state_density() * self.f_ir() * self.chi_2_lvl() * (
                (self.chi_3_lvl() / self.chi_2_lvl() + self.f_bl_simple()) / (1 + self.f_bl_simple()) - 1)

    def alpha(self):
        return np.e ** (integrate.simps(self.integrand(), self.z_grid))

    def min_counts_e(self):
        return 0.5 * (np.sqrt(
            16 * self.sigma_r ** 2 / (1 - self.alpha()) ** 2 + ((self.alpha() + 3) / (1 - self.alpha()) ** 2) ** 2) + (
                              self.alpha() + 3) / (1 - self.alpha()) ** 2)

    def min_counts_i(self):
        return self.alpha() * self.min_counts_e()

    def min_counts_raw(self):
        return self.min_counts_e() / self.Q_E * np.e ** (integrate.simps(
            self.ground_state_density() * (self.chi_3_lvl() + self.chi_2_lvl() * self.f_bl_simple()) / (
                    1 + self.f_bl_simple()), self.z_grid))

    def iterate(self, density):
        self.twolvl_list = []
        self.simple_eit_list = []
        self.single_eit_list = []
        self.eit_list = []
        for n in density:
            self.n_0 = n
            # print(self.n_0)

            # original_Omega_p = self.Omega_p
            self.twolvl_list.append(self.transmission_2lvl())
            # self.Omega_p = original_Omega_p*(self.transmission_2lvl()+1)/2
            # print(self.Omega_p)

            self.simple_eit_list.append(self.transmission_simple_eit())
            self.single_eit_list.append(self.transmission_single_eit())
            self.eit_list.append(self.transmission_eit())

            # self.Omega_p = original_Omega_p

        return self.twolvl_list, self.simple_eit_list, self.eit_list, self.single_eit_list

    def walk(self):

        self.step_count += 1
        old_Omega_p = self.Omega_p
        old_Omega_c = self.Omega_c
        old_n_0 = self.n_0
        old_count_ratio = self.min_counts_raw() / self.n_counts()
        self.n_0 = self.n_0 * 2 ** (np.random.uniform(-0.5, 0.5))
        self.Omega_c = self.Omega_c + np.random.uniform(-0.1, 0.1)
        self.Omega_p = self.Omega_c * np.random.uniform(0, 1)
        new_count_ratio = self.min_counts_raw() / self.n_counts()
        peak_f_bl = self.rho_0() * self.V_bl_6() * self.n_0

        if self.n_0 > 3 * 10 ** (-1) or self.n_0 < 10 ** (-3):
            self.n_0 = old_n_0

        elif self.Omega_c > 1 * self.Gamma_e or self.Omega_c < 0.01:
            self.Omega_c = old_Omega_c

        elif old_count_ratio > new_count_ratio or old_count_ratio > new_count_ratio * np.random.uniform(0, 1):
            # print("Accept")
            if new_count_ratio < 5000:
                self.Omega_p_list.append(self.Omega_p)
                self.Omega_c_list.append(self.Omega_c)
                self.n_0_list.append(self.n_0)
                self.alpha_list.append(-np.log(self.alpha()))
                self.min_counts_i_list.append(self.min_counts_i())
                self.min_counts_e_list.append(self.min_counts_e())
                self.n_counts_list.append(self.n_counts())
                self.Omega_p_over_c_list.append(self.Omega_p / self.Omega_c)
                self.peak_f_bl_list.append(peak_f_bl)
                self.min_counts_over_n_counts.append(1 / new_count_ratio)
            return None

        else:
            # print("Reject, Alpha is too big")
            self.rejected_values += 1
            self.Omega_p = old_Omega_p
            self.Omega_c = old_Omega_c
            self.n_0 = old_n_0
            self.rejection_ratio = self.rejected_values / self.step_count
            return None
