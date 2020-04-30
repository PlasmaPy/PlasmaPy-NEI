"""Test_eigenvaluetable"""
import warnings
import numpy as np
from plasmapy import atomic
from ..eigenvaluetable import EigenData2
import pytest

#-------------------------------------------------------------------------------
# Set elements and temperary list as testing inputs
#-------------------------------------------------------------------------------
natom_list = np.arange(1, 27)
natom = 8

#------------------------------------------------------------------------------
# function: Time-Advance solover
#------------------------------------------------------------------------------
def func_solver_eigenval(natom, te, ne, dt, f0, table):
    """
        The testing function for performing time_advance calculations.
    """

    common_index = table._get_temperature_index(te)
    evals = table.eigenvalues(T_e_index=common_index) # find eigenvalues on the chosen Te node
    evect = table.eigenvectors(T_e_index=common_index)
    evect_invers = table.eigenvector_inverses(T_e_index=common_index)

    # define the temperary diagonal matrix
    diagona_evals = np.zeros((natom+1, natom+1))
    for ii in range(0, natom+1):
        diagona_evals[ii,ii] = np.exp(evals[ii]*dt*ne)

    # matirx operation
    matrix_1 = np.dot(diagona_evals, evect)
    matrix_2 = np.dot(evect_invers, matrix_1)

    # get ions fraction at (time+dt)
    ft = np.dot(f0, matrix_2)

    # re-check the smallest value
    minconce = 1.0e-15
    for ii in np.arange(0, natom+1, dtype=np.int):
        if (abs(ft[ii]) <= minconce):
            ft[ii] = 0.0
    return ft

# [Nick] Temporarily commenting this test out to check if it may be
# causing problems in the tests on Travis CI because of the dependence
# on ChiantiPy.  This might be causing the tests on Travis CI to stall,
# while still working when we run `pytest` or `python setup.py test`
# locally.

#@pytest.mark.parametrize('natom', natom_list)
#def test_equlibrium_state_vs_chiantipy(natom=8):
#    """
#        Test equilibrium states saved in EigenData2 and compare them with
#        Outputs from ChiantiPy.
#        Note:
#        This test requires ChiantiPy to be installed (see details
#        in: https://github.com/chianti-atomic/ChiantiPy).
#    """
#    try:
#        import ChiantiPy.core as ch
#    except ImportError:
#        warnings.warn('ChiantiPy is required in this test.', UserWarning)
#        return
#
#    temperatures = [1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8]
#    eqi_ch = ch.ioneq(natom)
#    eqi_ch.calculate(temperatures)
#    conce = eqi_ch.Ioneq
#
#    table_sta = EigenData2(element=natom)
#    for i in range(2):
#        ch_conce = conce[:, i]
#        table_conce = table_sta.equilibrium_state(T_e=temperatures[i])
#        assert ch_conce.all() == table_conce.all()
#    return

@pytest.mark.parametrize('natom', [1, 2, 6, 7, 8])
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
    element_symbol = atomic.atomic_symbol(int(natom))
    te0 = 1.0e+6
    ne0 = 1.0e+8

    # Start from any ionizaiont states, e.g., Te = 4.0d4 K,
    time = 0
    table = EigenData2(element=natom)
    f0 = table.equilibrium_state(T_e=4.0e4)

    print('START test_reachequlibrium_state:')
    print(f'time_sta = ', time)
    print(f'INI: ', f0)
    print(f'Sum(f0) = ', np.sum(f0))

    # After time + dt:
    dt = 1.0e+7
    ft = func_solver_eigenval(natom, te0, ne0, time+dt, f0, table)

    print(f'time_end = ', time+dt)
    print(f'NEI:', ft)
    print(f'Sum(ft) = ', np.sum(ft))
    print(f'EI :', table.equilibrium_state(T_e=te0))
    print("End Test.\n")

    assert np.isclose(np.sum(ft), 1), 'np.sum(ft) is not approximately 1'
    assert np.isclose(np.sum(f0), 1), 'np.sum(f0) is not approximately 1'
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
    element_symbol = atomic.atomic_symbol(int(natom))
    te0 = 1.0e+6 # unit: K
    ne0 = 1.0e+8 # unit: cm^-3

    # Start from any ionizaiont states, e.g., Te = 4.0d4 K,
    time = 0
    table = EigenData2(element=natom)
    f0 = table.equilibrium_state(T_e=4.0e+4)

    print('START test_reachequlibrium_state_multisteps:')
    print(f'time_sta = ', time)
    print(f'INI: ', f0)
    print(f'Sum(f0) = ', np.sum(f0))

    # After time + dt:
    dt = 100000.0 # unit: second

    # Enter the time loop:
    for it in range(100):
        ft = func_solver_eigenval(natom, te0, ne0, time+dt, f0, table)
        f0 = np.copy(ft)
        time = time+dt

    print(f'time_end = ', time+dt)
    print(f'NEI:', ft)
    print(f'Sum(ft) = ', np.sum(ft))

    assert np.isclose(np.sum(ft), 1)

    print(f"EI :", table.equilibrium_state(T_e=te0))
    print("End Test.\n")

# Temporarily test only lighter elements due to Travis CI delays

#@pytest.mark.parametrize('atomic_numb', np.arange(1, 27))
@pytest.mark.parametrize('atomic_numb', np.arange(1, 10))
def test_element_range(atomic_numb):
    """
    Function test_element_range:
        This function is used to test element including Hydrogen to Iron.
    """
    try:
        element_symbol = atomic.atomic_symbol(int(atomic_numb))
        eigen = EigenData2(element=element_symbol)
    except Exception as exc:
        raise Exception(f"Problem with atomic number={atomic_numb}.") from exc

    eigen=0  # attempt to clear up memory
