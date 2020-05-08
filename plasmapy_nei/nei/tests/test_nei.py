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


tests = {
    "basic": {
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
    },
    "T_e constant": {
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
    },
    "n_e constant": {
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
    },
    "T_e function": {
        "inputs": inputs_dict,
        "abundances": abundances,
        "T_e": lambda time: 1e4 * (1 + time / u.s) * u.K,
        "n": 1e15 * u.cm ** -3,
        "time_max": 800 * u.s,
        "max_steps": 2,
        "dt": 100 * u.s,
        "adapt_dt": False,
        "verbose": True,
    },
    "n function": {
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
    },
    "equil test cool": {
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
    },
    "equil test hot": {
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
    },
    "equil test start far out of equil": {
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
    },
    "adapt dt": {
        "inputs": ["H", "He"],
        "abundances": {"H": 1, "He": 0.1},
        "T_e": lambda t: u.K * (1e6 + 1.3e4 * np.sin(t.value)),
        "n": 1e10 * u.cm ** -3,
        "max_steps": 300,
        "time_start": 0 * u.s,
        "time_max": 2 * np.pi * u.s,
        "adapt_dt": True,
    },
}

test_names = list(tests.keys())


class TestNEI:
    @classmethod
    def setup_class(cls):
        cls.instances = {}

    @pytest.mark.parametrize("test_name", test_names)
    def test_instantiate(self, test_name):
        try:
            instance = NEI(**tests[test_name])
            self.instances[test_name] = instance
        except Exception as exc:
            raise Exception(f"Problem with test {test_name}") from exc

    @pytest.mark.parametrize("test_name", test_names)
    def test_time_start(self, test_name):
        instance = self.instances[test_name]
        if "time_start" in tests[test_name].keys():
            assert tests[test_name]["time_start"] == instance.time_start
        elif "time_input" in tests[test_name].keys():
            assert tests[test_name]["time_input"].min() == instance.time_start
        else:
            assert instance.time_start == 0 * u.s

    @pytest.mark.parametrize("test_name", test_names)
    def test_time_max(self, test_name):
        instance = self.instances[test_name]
        if "time_max" in tests[test_name].keys():
            assert tests[test_name]["time_max"] == instance.time_max

    @pytest.mark.parametrize("test_name", test_names)
    def test_initial_type(self, test_name):
        instance = self.instances[test_name]
        assert isinstance(instance.initial, IonizationStates)

    @pytest.mark.parametrize("test_name", test_names)
    def test_n_input(self, test_name):
        actual = self.instances[test_name].n_input
        expected = tests[test_name]["n"]
        if isinstance(expected, u.Quantity) and not expected.isscalar:
            assert all(expected == actual)
        else:
            assert expected == actual

    @pytest.mark.parametrize("test_name", test_names)
    def test_T_e_input(self, test_name):
        actual = self.instances[test_name].T_e_input
        expected = tests[test_name]["T_e"]
        if isinstance(expected, u.Quantity) and not expected.isscalar:
            assert all(expected == actual)
        else:
            assert expected == actual

    @pytest.mark.parametrize("test_name", test_names)
    def test_electron_temperature(self, test_name):
        instance = self.instances[test_name]
        T_e_input = instance.T_e_input
        if isinstance(T_e_input, u.Quantity):
            if T_e_input.isscalar:
                assert instance.electron_temperature(instance.time_start) == T_e_input
                assert instance.electron_temperature(instance.time_max) == T_e_input
            else:
                for time, T_e in zip(instance.time_input, T_e_input):
                    try:
                        T_e_func = instance.electron_temperature(time)
                    except Exception:
                        raise ValueError("Unable to find T_e from electron_temperature")

                    assert np.isclose(T_e.value, T_e_func.value)
        if callable(T_e_input):
            assert instance.T_e_input(
                instance.time_start
            ) == instance.electron_temperature(instance.time_start)

    @pytest.mark.parametrize(
        "test_name",
        [
            test_name
            for test_name in test_names
            if isinstance(tests[test_name]["inputs"], dict)
        ],
    )
    def test_initial_ionfracs(self, test_name):
        original_inputs = tests[test_name]["inputs"]
        original_elements = original_inputs.keys()

        for element in original_elements:
            assert np.allclose(
                original_inputs[element],
                self.instances[test_name].initial.ionic_fractions[
                    particle_symbol(element)
                ],
            )

    @pytest.mark.parametrize("test_name", test_names)
    def test_simulate(self, test_name):
        try:
            self.instances[test_name].simulate()
        except Exception as exc:
            raise ValueError(f"Unable to simulate for test: {test_name}") from exc

    @pytest.mark.parametrize("test_name", test_names)
    def test_simulation_end(self, test_name):
        instance = self.instances[test_name]
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

    @pytest.mark.parametrize(
        "test_name", [test_name for test_name in test_names if "equil" in test_name]
    )
    def test_equilibration(self, test_name):
        """
        Test that equilibration works.
        """
        instance = self.instances[test_name]
        T_e = instance.T_e_input
        assert (
            isinstance(T_e, u.Quantity) and T_e.isscalar
        ), "This test can only be used for cases where T_e is constant."
        equil_dict = instance.equil_ionic_fractions(T_e)
        for element in instance.elements:
            assert np.allclose(
                equil_dict[element], instance.results.ionic_fractions[element][-1, :]
            )

    @pytest.mark.parametrize("test_name", test_names)
    def test_initial_results(self, test_name):
        initial = self.instances[test_name].initial
        results = self.instances[test_name].results
        # Make sure that the elements are equal to each other
        assert initial.ionic_fractions.keys() == results.ionic_fractions.keys()
        assert initial.abundances == results.abundances
        for elem in initial.ionic_fractions.keys():  # TODO: enable initial.elements
            assert np.allclose(
                results.ionic_fractions[elem][0, :], initial.ionic_fractions[elem]
            )

    @pytest.mark.parametrize("test_name", test_names)
    def test_final_results(self, test_name):
        # initial = self.instances[test_name].initial
        final = self.instances[test_name].final
        results = self.instances[test_name].results
        # Make sure the elements are equal to each other
        assert final.ionic_fractions.keys() == results.ionic_fractions.keys()
        assert final.abundances == results.abundances
        for elem in final.ionic_fractions.keys():
            assert np.allclose(
                results.ionic_fractions[elem][-1, :], final.ionic_fractions[elem]
            )
