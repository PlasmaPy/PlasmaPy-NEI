"""Tests of the class to store package data."""

import numpy as np
import pytest

from plasmapy import particles

from plasmapy_nei.eigen import EigenData


@pytest.mark.parametrize("atomic_numb", np.arange(1, 30))
def test_instantiation(atomic_numb):
    try:
        element_symbol = particles.atomic_symbol(int(atomic_numb))
        EigenData(element=element_symbol)
    except Exception as exc:
        raise Exception(
            f"Problem with instantiation for atomic number={atomic_numb}."
        ) from exc


def time_advance_solver_for_testing(natom, te, ne, dt, f0, table):
    """Testing function for performing time advance calculations"""

    common_index = table._get_temperature_index(te)
    evals = table.eigenvalues(
        T_e_index=common_index
    )  # find eigenvalues on the chosen Te node
    evect = table.eigenvectors(T_e_index=common_index)
    evect_invers = table.eigenvector_inverses(T_e_index=common_index)

    # define the temperary diagonal matrix
    diagona_evals = np.zeros((natom + 1, natom + 1))
    for ii in range(natom + 1):
        diagona_evals[ii, ii] = np.exp(evals[ii] * dt * ne)

    # matrix operation
    matrix_1 = np.dot(diagona_evals, evect)
    matrix_2 = np.dot(evect_invers, matrix_1)

    # get ionic fraction at (time+dt)
    ft = np.dot(f0, matrix_2)

    # re-check the smallest value
    minconce = 1.0e-15
    for ii in np.arange(0, natom + 1, dtype=np.int):
        if abs(ft[ii]) <= minconce:
            ft[ii] = 0.0
    return ft


@pytest.mark.parametrize("natom", [1, 2, 6, 7, 8])
def test_reachequlibrium_state(natom):
    """
    Starting the random initial distribution, the charge states will reach
    to equilibrium cases after a long time.
    In this test, we set the ionization and recombination rates at
    Te0=2.0e6 K and plasma density ne0=1.0e+7. A random charge states
    distribution will be finally closed to equilibrium distribution at
    2.0e6K.
    """
    #
    # Initial conditions, set plasma temperature, density and dt
    #
    element_symbol = particles.atomic_symbol(int(natom))
    te0 = 1.0e6
    ne0 = 1.0e8

    # Start from any ionizaiont states, e.g., Te = 4.0d4 K,
    time = 0
    table = EigenData(element=natom)
    f0 = table.equilibrium_state(T_e=4.0e4)

    print("START test_reachequlibrium_state:")
    print("time_sta = ", time)
    print("INI: ", f0)
    print("Sum(f0) = ", np.sum(f0))

    # After time + dt:
    dt = 1.0e7
    ft = time_advance_solver_for_testing(natom, te0, ne0, time + dt, f0, table)

    print("time_end = ", time + dt)
    print("NEI:", ft)
    print("Sum(ft) = ", np.sum(ft))
    print("EI :", table.equilibrium_state(T_e=te0))
    print("End Test.\n")

    assert np.isclose(np.sum(ft), 1), "np.sum(ft) is not approximately 1"
    assert np.isclose(np.sum(f0), 1), "np.sum(f0) is not approximately 1"
    assert np.allclose(ft, table.equilibrium_state(T_e=te0))


def test_reachequlibrium_state_multisteps(natom=8):
    """
    Starting the random initial distribution, the charge states will reach
    to equilibrium cases after a long time (multiple steps).
    In this test, we set the ionization and recombination rates at
    Te0=2.0e6 K and plasma density ne0=1.0e+7. A random charge states
    distribution will be finally closed to equilibrium distribution at
    2.0e6K.
    """
    #
    # Initial conditions, set plasma temperature, density and dt
    #
    te0 = 1.0e6  # unit: K
    ne0 = 1.0e8  # unit: cm^-3

    # Start from any ionization state, e.g., Te = 4.0d4 K,
    time = 0
    table = EigenData(element=natom)
    f0 = table.equilibrium_state(T_e=4.0e4)

    # print(f"time_sta = ", time)
    # print(f"INI: ", f0)
    # print(f"Sum(f0) = ", np.sum(f0))

    # After time + dt:
    dt = 100000.0  # unit: second

    # Enter the time loop:
    for _ in range(100):
        ft = time_advance_solver_for_testing(natom, te0, ne0, time + dt, f0, table)
        f0 = np.copy(ft)
        time = time + dt

    # print(f"time_end = ", time + dt)
    # print(f"NEI:", ft)
    # print(f"Sum(ft) = ", np.sum(ft))

    assert np.isclose(np.sum(ft), 1)

    # print(f"EI :", table.equilibrium_state(T_e=te0))
    # print("End Test.\n")


# TODO: Test that appropriate exceptions are raised for invalid inputs
