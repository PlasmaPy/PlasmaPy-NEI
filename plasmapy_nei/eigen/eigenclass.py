"""
Contains the class to provide access to eigenvalue data for the
ionization and recombination rates.
"""

__all__ = ["EigenData"]

import h5py
import numpy as np
from numpy import linalg as LA
import pkg_resources
import warnings

try:
    from plasmapy.particles import Particle, particle_input
except (ImportError, ModuleNotFoundError):
    from plasmapy.atomic import Particle, particle_input

max_atomic_number = 30  # TODO: double check if this is the correct number


def _get_equilibrium_charge_states(ioniz_rate, recomb_rate, natom):
    """
    Compute the equilibrium charge state distribution for the
    temperature specified using ionization and recombination rates.

    Parameters
    ----------
    ioniz_rate

    recomb_rate

    natom
        The atomic number.
    """

    # TODO: refactor this function with

    nstates = natom + 1
    conce = np.zeros(nstates)
    f = np.zeros(nstates + 1)
    c = np.zeros(nstates + 1)
    r = np.zeros(nstates + 1)

    # The start index is 1.
    for i in range(nstates):
        c[i + 1] = ioniz_rate[i]
        r[i + 1] = recomb_rate[i]

    # f[0] = 0.0 from initialization
    f[1] = 1.0

    f[2] = c[1] * f[1] / r[2]

    # The solution for hydrogen may be found analytically.
    if natom == 1:
        f[1] = 1.0 / (1.0 + c[1] / r[2])
        f[2] = c[1] * f[1] / r[2]
        conce[0:2] = f[1:3]
        return conce

    # for other elements

    for k in range(2, natom):
        f[k + 1] = (-c[k - 1] * f[k - 1] + (c[k] + r[k]) * f[k]) / r[k + 1]

    f[natom + 1] = c[natom] * f[natom] / r[natom + 1]

    f[1] = 1.0 / np.sum(f)

    f[2] = c[1] * f[1] / r[2]

    for k in range(2, natom):
        f[k + 1] = (-c[k - 1] * f[k - 1] + (c[k] + r[k]) * f[k]) / r[k + 1]

    f[natom + 1] = c[natom] * f[natom] / r[natom + 1]
    
    # normalize the distribution
    f = f / np.sum(f)

    conce[0:nstates] = f[1: nstates + 1]
    return conce


class EigenData:
    """
    Provides access to ionization and recombination rate data.

    Parameters
    ----------
    element : particle-like
        Representation of the element to access data for.

    Examples
    --------
    >>> eigendata = EigenData("He")
    >>> eigendata.nstates
    3
    """

    def _validate_element(self, element):

        # The following might not be needed if the check is in @particle_input
        if not element.is_category(require="element", exclude=["ion", "isotope"]):
            raise ValueError(f"{element} is not an element")

        if element.atomic_number > max_atomic_number:
            raise ValueError("Need an element")

        self._element = element

    def _load_data(self):
        """
        Retrieve data from the HDF5 file containing ionization and
        recombination rates.
        """
        filename = pkg_resources.resource_filename(
            "plasmapy_nei",
            "data/ionrecomb_rate.h5",  # from Chianti database, version 8.07
        )

        # TODO: enable different HDF5 files

        with h5py.File(filename, "r") as file:
            self._temperature_grid = file["te_gird"][:]  # TODO: fix typo in HDF5 file
            self._ntemp = self._temperature_grid.size
            c_ori = file["ioniz_rate"][:]
            r_ori = file["recomb_rate"][:]

        c_rate = np.zeros((self.ntemp, self.nstates))
        r_rate = np.zeros((self.ntemp, self.nstates))
        for temperature_index in range(self.ntemp):
            for i in range(self.nstates - 1):
                c_rate[temperature_index, i] = c_ori[
                    i, self.element.atomic_number - 1, temperature_index
                ]
            for i in range(1, self.nstates):
                r_rate[temperature_index, i] = r_ori[
                    i - 1, self.element.atomic_number - 1, temperature_index
                ]

        self._ionization_rate = np.ndarray(
            shape=(self.ntemp, self.nstates), dtype=np.float64
        )
        self._recombination_rate = np.ndarray(
            shape=(self.ntemp, self.nstates), dtype=np.float64
        )
        self._equilibrium_states = np.ndarray(
            shape=(self.ntemp, self.nstates), dtype=np.float64
        )
        self._eigenvalues = np.ndarray(
            shape=(self.ntemp, self.nstates), dtype=np.float64
        )

        self._eigenvectors = np.ndarray(
            shape=(self.ntemp, self.nstates, self.nstates), dtype=np.float64,
        )

        self._eigenvector_inverses = np.ndarray(
            shape=(self.ntemp, self.nstates, self.nstates), dtype=np.float64,
        )

        self._ionization_rate[:, :] = c_rate[:, :]
        self._recombination_rate[:, :] = r_rate[:, :]

        # Define the coefficients matrix A. The first dimension is
        # for elements, and the second number of equations.

        nequations = self.nstates
        coefficients_matrix = np.zeros(
            shape=(self.nstates, nequations), dtype=np.float64
        )

        # Enter temperature loop over the whole temperature grid

        for temperature_index in range(self.ntemp):

            # Ionization and recombination rate at Te(temperature_index)
            carr = c_rate[temperature_index, :]
            rarr = r_rate[temperature_index, :]

            eqi = _get_equilibrium_charge_states(
                carr, rarr, self.element.atomic_number,
            )

            for ion in range(1, self.nstates - 1):
                coefficients_matrix[ion, ion - 1] = carr[ion - 1]
                coefficients_matrix[ion, ion] = -(carr[ion] + rarr[ion])
                coefficients_matrix[ion, ion + 1] = rarr[ion + 1]

            # The first row
            coefficients_matrix[0, 0] = -carr[0]
            coefficients_matrix[0, 1] = rarr[1]

            # The last row
            coefficients_matrix[self.nstates - 1, self.nstates - 2] = carr[
                self.nstates - 2
            ]
            coefficients_matrix[self.nstates - 1, self.nstates - 1] = -rarr[
                self.nstates - 1
            ]

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = LA.eig(coefficients_matrix)

            # Rearrange the eigenvalues.
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Compute inverse of eigenvectors
            v_inverse = LA.inv(eigenvectors)

            # transpose the order to as same as the Fortran Version
            eigenvectors = eigenvectors.transpose()
            v_inverse = v_inverse.transpose()

            # Save eigenvalues and eigenvectors into arrays
            for j in range(self.nstates):
                self._eigenvalues[temperature_index, j] = eigenvalues[j]
                self._equilibrium_states[temperature_index, j] = eqi[j]
                for i in range(self.nstates):
                    self._eigenvectors[temperature_index, i, j] = eigenvectors[i, j]
                    self._eigenvector_inverses[temperature_index, i, j] = v_inverse[
                        i, j
                    ]

    @particle_input
    def __init__(self, element: Particle):

        self._validate_element(element)
        self._load_data()

    def _get_temperature_index(self, T_e):  # TODO: extract this to a function
        """Return the temperature index closest to a particular temperature."""
        T_e_array = self._temperature_grid

        # Check the temperature range
        T_e_grid_max = np.amax(T_e_array)
        T_e_grid_min = np.amin(T_e_array)

        if T_e == T_e_grid_max:
            return self._ntemp - 1
        elif T_e == T_e_grid_min:
            return 0
        elif T_e > T_e_grid_max:
            warnings.warn(
                f"Temperature exceeds the temperature grid "
                f"boundary: temperature index will be reset "
                f"to {self._ntemp - 1}",
                UserWarning,
            )
            return self._ntemp - 1
        elif T_e < T_e_grid_min:
            warnings.warn(
                f"Temperature is below the temperature grid "
                f"boundary: temperature index will be reset to "
                f"0.",
                UserWarning,
            )
            return 0

        # TODO: Add a test to check that the temperature grid is monotonic

        res = np.where(T_e_array >= T_e)
        res_ind = res[0]
        index = res_ind[0]
        dte_l = abs(T_e - T_e_array[index - 1])  # re-check the neighbor point
        dte_r = abs(T_e - T_e_array[index])
        if dte_l <= dte_r:
            index = index - 1
        return index

    @property
    def temperature(self):
        return self._temperature

    @property
    def temperature(self, T_e):
        self._temperature = T_e
        self._te_index = self._get_temperature_index(T_e)

    @property
    def temperature_grid(self):
        return self._temperature_grid

    @property
    def element(self) -> Particle:
        """The element corresponding to an instance of this class."""
        return self._element

    @property
    def nstates(self) -> int:
        """Number of charge states corresponding to the element."""
        return self.element.atomic_number + 1

    @property
    def ntemp(self) -> int:
        """Number of points in ``temperature_grid``."""
        return self._ntemp

    @property
    def ionization_rate(self):
        return self._ionization_rate

    @property
    def recombination_rate(self):
        return self._recombination_rate

    def eigenvalues(self, T_e=None, T_e_index=None):
        """
        Returns the eigenvalues for the ionization and recombination
        rates for the temperature specified in the class.
        """
        if T_e_index is not None:
            return self._eigenvalues[T_e_index, :]
        elif T_e is not None:
            T_e_index = self._get_temperature_index(T_e)
            return self._eigenvalues[T_e_index, :]
        elif self.temperature:
            return self._eigenvalues[self._te_index, :]
        else:
            raise AttributeError("The temperature has not been set.")

    def eigenvectors(self, T_e=None, T_e_index=None):
        """
        Returns the eigenvectors for the ionization and recombination
        rates for the temperature specified in the class.
        """
        if T_e_index:
            return self._eigenvectors[T_e_index, :, :]
        elif T_e:
            T_e_index = self._get_temperature_index(T_e)
            return self._eigenvectors[T_e_index, :, :]
        elif self.temperature:
            return self._eigenvectors[self._te_index, :, :]
        else:
            raise AttributeError("The temperature has not been set.")

    def eigenvector_inverses(self, T_e=None, T_e_index=None):
        """
        Returns the inverses of the eigenvectors for the ionization and
        recombination rates for the temperature specified in the class.
        """
        if T_e_index:
            return self._eigenvector_inverses[T_e_index, :, :]
        elif T_e:
            T_e_index = self._get_temperature_index(T_e)
            return self._eigenvector_inverses[T_e_index, :, :]
        elif self.temperature:
            return self._eigenvector_inverses[self._te_index, :, :]
        else:
            raise AttributeError("The temperature has not been set.")

    def equilibrium_state(self, T_e=None, T_e_index=None):
        """
        Return the equilibrium charge state distribution for the
        temperature specified in the class.
        """
        if T_e_index:
            return self._equilibrium_states[T_e_index, :]
        elif T_e:
            T_e_index = self._get_temperature_index(T_e)
            return self._equilibrium_states[T_e_index, :]
        elif self.temperature:
            return self._equilibrium_states[self._te_index, :]
        else:
            raise AttributeError("The temperature has not been set.")
