"""Tests for non-equilibrium ionization modeling classes."""

import astropy.units as u
try:
    from plasmapy.atomic import IonizationStates, particle_symbol
except ImportError:
    from plasmapy.particles import IonizationStates, particle_symbol
from plasmapy_nei.nei import NEI
from plasmapy_nei.eigen import EigenData
import numpy as np
import pytest

inputs_dict = {"H": [0.9, 0.1], "He": [0.5, 0.3, 0.2]}
abundances = {"H": 1, "He": 0.1}

time_array = np.array([0, 800]) * u.s
T_e_array = np.array([4e4, 6e4]) * u.K
n_array = np.array([1e9, 5e8]) * u.cm ** -3


@pytest.fixture(scope="module", params =
                         [
    ("basic", {
        "inputs": inputs_dict,
        "abundances": abundances,
        "T_e": T_e_array,
        "n": n_array,
        "time_input": time_array,
        "time_start": 0 * u.s,
        "time_max": 800 * u.s,
        "max_steps": 1,
        "dt": 800 * u.s,
        "adapt_dt": False,
        "verbose": True,
    }),
    ("T_e constant", {
        "inputs": inputs_dict,
        "abundances": abundances,
        "T_e": 1 * u.MK,
        "n": n_array,
        "time_input": time_array,
        "time_start": 0 * u.s,
        "time_max": 800 * u.s,
        "dt": 100 * u.s,
        "max_steps": 2,
        "adapt_dt": False,
        "verbose": True,
    }),
    ("n_e constant", {
        "inputs": inputs_dict,
        "abundances": abundances,
        "T_e": T_e_array,
        "n": 1e9 * u.cm ** -3,
        "time_input": time_array,
        "time_start": 0 * u.s,
        "time_max": 800 * u.s,
        "max_steps": 2,
        "adapt_dt": False,
        "dt": 100 * u.s,
        "verbose": True,
    }),
    ("T_e function", {
        "inputs": inputs_dict,
        "abundances": abundances,
        "T_e": lambda time: 1e4 * (1 + time / u.s) * u.K,
        "n": 1e15 * u.cm ** -3,
        "time_max": 800 * u.s,
        "max_steps": 2,
        "dt": 100 * u.s,
        "adapt_dt": False,
        "verbose": True,
    }),
    ("n function", {
        "inputs": inputs_dict,
        "abundances": abundances,
        "T_e": 6e4 * u.K,
        "n": lambda time: 1e9 * (1 + time / u.s) * u.cm ** -3,
        "time_start": 0 * u.s,
        "time_max": 800 * u.s,
        "adapt_dt": False,
        "dt": 200 * u.s,
        "verbose": True,
        "max_steps": 4,
    }),
    ("equil test cool", {
        "inputs": ["H", "He", "N"],
        "abundances": {"H": 1, "He": 0.1, "C": 1e-4, "N": 1e-4, "O": 1e-4, "Fe": 1e-4},
        "T_e": 10001.0 * u.K,
        "n": 1e13 * u.cm ** -3,
        "time_max": 2e6 * u.s,
        "tol": 1e-9,
        "adapt_dt": False,
        "dt": 5e5 * u.s,
        "max_steps": 4,
        "verbose": True,
    }),
    ("equil test hot", {
        "inputs": ["H", "He", "C"],
        "abundances": {
            "H": 1,
            "He": 0.1,
            "C": 1e-4,
            "N": 1e-4,
            "O": 1e-4,
            "Fe": 1e-4,
            "S": 2e-6,
        },
        "T_e": 7e6 * u.K,
        "n": 1e9 * u.cm ** -3,
        "time_max": 1e8 * u.s,
        "dt": 5e7 * u.s,
        "max_steps": 3,
        "adapt_dt": False,
        "verbose": True,
    }),
    ("equil test start far out of equil", {
        "inputs": {
            "H": [0.99, 0.01],
            "He": [0.5, 0.0, 0.5],
            "O": [0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2],
        },
        "abundances": {"H": 1, "He": 0.1, "O": 1e-4},
        "T_e": 3e6 * u.K,
        "n": 1e9 * u.cm ** -3,
        "dt": 1e6 * u.s,
        "time_start": 0 * u.s,
        "time_max": 1e6 * u.s,
        "adapt_dt": False,
        "verbose": True,
        "max_steps": 2,
    }),
    ("adapt dt", {
        "inputs": ["H", "He"],
        "abundances": {"H": 1, "He": 0.1},
        "T_e": lambda t: u.K * (1e6 + 1.3e4 * np.sin(t.value)),
        "n": 1e10 * u.cm ** -3,
        "max_steps": 300,
        "time_start": 0 * u.s,
        "time_max": 2 * np.pi * u.s,
        "adapt_dt": True,
    }),
])
def name_inputs_instance(request):
    test_name, dictionary = request.param
    return test_name, dictionary, NEI(**dictionary)


def test_time_start(name_inputs_instance):
    test_name, inputs, instance = name_inputs_instance
    if "time_start" in inputs.keys():
        assert inputs["time_start"] == instance.time_start
    elif "time_input" in inputs.keys():
        assert inputs["time_input"].min() == instance.time_start
    else:
        assert instance.time_start == 0 * u.s

def test_time_max(name_inputs_instance):
    test_name, inputs, instance = name_inputs_instance
    if "time_max" in inputs.keys():
        assert inputs["time_max"] == instance.time_max

def test_initial_type(name_inputs_instance):
    test_name, inputs, instance = name_inputs_instance
    assert isinstance(instance.initial, IonizationStates)

def test_n_input(name_inputs_instance):
    test_name, inputs, instance = name_inputs_instance
    actual = instance.n_input
    expected = inputs["n"]
    assert np.all(expected == actual)

def test_T_e_input(name_inputs_instance):
    test_name, inputs, instance = name_inputs_instance
    actual = instance.T_e_input
    expected = inputs["T_e"]
    assert np.all(expected == actual)

def test_electron_temperature(name_inputs_instance):
    test_name, inputs, instance = name_inputs_instance
    T_e_input = instance.T_e_input
    if isinstance(T_e_input, u.Quantity):
        if T_e_input.isscalar:
            assert instance.electron_temperature(instance.time_start) == T_e_input
            assert instance.electron_temperature(instance.time_max) == T_e_input
        else:
            for time, T_e in zip(instance.time_input, T_e_input):
                T_e_func = instance.electron_temperature(time)

                assert np.isclose(T_e.value, T_e_func.value)
    if callable(T_e_input):
        assert instance.T_e_input(
            instance.time_start
        ) == instance.electron_temperature(instance.time_start)

def test_initial_ionfracs(name_inputs_instance):
    test_name, inputs, instance = name_inputs_instance
    if not isinstance(inputs["inputs"], dict):
        pytest.skip("Test irrelevant")

        
    original_inputs = inputs["inputs"]
    original_elements = original_inputs.keys()

    for element in original_elements:
        assert np.allclose(
            original_inputs[element],
            instance.initial.ionic_fractions[
                particle_symbol(element)
            ],
        )

def test_simulate(name_inputs_instance):
    test_name, inputs, instance = name_inputs_instance
    instance.simulate()
    final = instance.final
    results = instance.results
    # Make sure the elements are equal to each other
    assert final.ionic_fractions.keys() == results.ionic_fractions.keys()
    assert final.abundances == results.abundances
    for elem in final.ionic_fractions.keys():
        assert np.allclose(
            results.ionic_fractions[elem][-1, :], final.ionic_fractions[elem]
        )

def test_simulation_end(name_inputs_instance):
    test_name, inputs, instance = name_inputs_instance
    time = instance.results.time
    end_time = time[-1]
    max_steps = instance.max_steps

    if np.isnan(end_time.value):
        raise Exception("End time is NaN.")

    got_to_max_steps = len(time) == instance.max_steps + 1
    got_to_time_max = np.isclose(time[-1].value, instance.time_max.value)

    if time.isscalar or len(time) == 1:
        raise Exception(f"The only element in results.time is {time}")

    if not got_to_max_steps and not got_to_time_max:
        print(f"time = {time}")
        print(f"max_steps = {max_steps}")
        print(f"time_max = {instance.time_max}")
        raise Exception("Problem with end time.")

def test_equilibration(name_inputs_instance):
    test_name, inputs, instance = name_inputs_instance
    if "equil" in test_name:
        pytest.skip("Test irrelevant")
    """
    Test that equilibration works.
    """
    T_e = instance.T_e_input
    if not (isinstance(T_e, u.Quantity) and T_e.isscalar):
        pytest.skip("This test can only be used for cases where T_e is constant.")
    equil_dict = instance.equil_ionic_fractions(T_e)
    for element in instance.elements:
        assert np.allclose(
            equil_dict[element], instance.results.ionic_fractions[element][-1, :]
        )

def test_initial_results(name_inputs_instance):
    test_name, inputs, instance = name_inputs_instance
    initial = instance.initial
    results = instance.results
    # Make sure that the elements are equal to each other
    assert initial.ionic_fractions.keys() == results.ionic_fractions.keys()
    assert initial.abundances == results.abundances
    for elem in initial.ionic_fractions.keys():  # TODO: enable initial.elements
        assert np.allclose(
            results.ionic_fractions[elem][0, :], initial.ionic_fractions[elem]
        )
