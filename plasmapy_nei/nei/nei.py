"""Contains classes to represent non-equilibrium ionization simulations."""

__all__ = ["NEI", "NEIError", "SimulationResults"]


import astropy.units as u
import numpy as np
import warnings

from plasmapy.particles import atomic_number, IonizationStateCollection
from scipy import interpolate, optimize
from typing import Callable, Dict, List, Optional, Union

from plasmapy_nei.eigen import eigen_data_dict, EigenData

# TODO: Allow this to keep track of velocity and position too, and
# eventually to have density and temperature be able to be functions of
# position.  (and more complicated expressions for density and
# temperature too)

# TODO: Expand Simulation docstring


# TODO: Include the methods in the original Visualize class which is a
#       subclass of NEI in the NEI-modeling/NEI repo. These were deleted
#       temporarily to make it possible to get the NEI class itself
#       adapted into this package.


# TODO: In this file and test_nei.py, there are a few places with
#       initial.ionic_fractions.keys(), where initial is an instance
#       of IonizationStateCollection.  This workaround exists because I forgot
#       to put in an `elements` attribute in IonizationStateCollection, and
#       should be corrected.


class NEIError(Exception):
    """For when there are errors in setting up or performing NEI simulations."""

    pass


class SimulationResults:
    """
    Results from a non-equilibrium ionization simulation.

    Parameters
    ----------
    initial: plasmapy.particles.IonizationStateCollection
        The ``IonizationStateCollection`` instance representing the ionization
        states of different elements and plasma properties as the
        initial conditions.

    n_init: astropy.units.Quantity
        The initial number density scaling factor.

    T_e_init: astropy.units.Quantity
        The initial electron temperature.

    max_steps: int
        The maximum number of time steps that the simulation can take
        before stopping.

    time_start: astropy.units.Quantity
        The time at the start of the simulation.

    """

    def __init__(
        self,
        initial: IonizationStateCollection,
        n_init: u.Quantity,
        T_e_init: u.Quantity,
        max_steps: int,
        time_start: u.Quantity,
    ):

        self._elements = list(initial.ionic_fractions.keys())
        self._abundances = initial.abundances
        self._max_steps = max_steps

        self._nstates = {elem: atomic_number(elem) + 1 for elem in self.elements}

        self._ionic_fractions = {
            elem: np.full((max_steps + 1, self.nstates[elem]), np.nan, dtype=np.float64)
            for elem in self.elements
        }

        self._number_densities = {
            elem: np.full((max_steps + 1, self.nstates[elem]), np.nan, dtype=np.float64)
            * u.cm ** -3
            for elem in self.elements
        }

        self._n_elem = {
            elem: np.full(max_steps + 1, np.nan) * u.cm ** -3 for elem in self.elements
        }

        self._n_e = np.full(max_steps + 1, np.nan) * u.cm ** -3
        self._T_e = np.full(max_steps + 1, np.nan) * u.K
        self._time = np.full(max_steps + 1, np.nan) * u.s

        self._index = 0

        self._assign(
            new_time=time_start,
            new_ionfracs=initial.ionic_fractions,
            new_n=n_init,
            new_T_e=T_e_init,
        )

    def _assign(
        self,
        new_time: u.Quantity,
        new_ionfracs: Dict[str, np.ndarray],
        new_n: u.Quantity,
        new_T_e: u.Quantity,
    ):
        """
        Store results from a time step of a non-equilibrium ionization
        time advance in the `~plasmapy_nei.classes.NEI` class.

        Parameters
        ----------
        new_time
            The time associated with this time step.

        new_ionfracs: dict
            The new ionization fractions for this time step.  The keys
            of this `dict` are the atomic symbols of the elements being
            tracked, and with the corresponding value being an
            ``numpy.ndarray`` representing the ionic fractions.  Each
            element's array must have a length of the atomic number plus
            one, and be normalized to one with all values between zero
            and one.

        new_n
            The new number density scaling factor for this time step.
            The number densities of each ionic species will be the
            product of this scaling factor, the element's abundance, and
            the ionic fraction given in ``new_ionfracs``.

        new_T_e
            The new electron temperature.

        """

        try:
            index = self._index
            elements = self.elements
            self._time[index] = new_time
            self._T_e[index] = new_T_e

            for elem in elements:
                self._ionic_fractions[elem][index, :] = new_ionfracs[elem][:]

            # Calculate elemental and ionic number densities
            n_elem = {elem: new_n * self.abundances[elem] for elem in elements}
            number_densities = {
                elem: n_elem[elem] * new_ionfracs[elem] for elem in elements
            }

            # Calculate the electron number density
            n_e = 0.0 * u.cm ** -3
            for elem in elements:
                integer_charges = np.linspace(
                    0, self.nstates[elem] - 1, self.nstates[elem]
                )
                n_e += np.sum(number_densities[elem] * integer_charges)

            # Assign densities
            self._n_e[index] = n_e
            for elem in elements:
                self._n_elem[elem][index] = n_elem[elem]
                self._number_densities[elem][index, :] = number_densities[elem]

        except Exception as exc:
            raise NEIError(
                "Unable to assign parameters to Simulation instance "
                f"for index {index} at time = {new_time}.  The "
                f"parameters are new_n = {new_n}, new_T_e = {new_T_e}, "
                f"and new_ionic_fractions = {new_ionfracs}."
            ) from exc
        finally:
            self._index += 1

    def _cleanup(self):
        """
        Clean up this class after the simulation is complete.

        This method removes the excess elements from each array that
        did not end up getting used for a time step in the simulation
        and sets the ``last_step`` attribute.

        """
        nsteps = self._index

        self._n_e = self._n_e[:nsteps]
        self._T_e = self._T_e[:nsteps]
        self._time = self._time[:nsteps]

        for element in self.elements:
            self._ionic_fractions[element] = self._ionic_fractions[element][0:nsteps, :]
            self._number_densities[element] = self._number_densities[element][
                0:nsteps, :
            ]

        self._last_step = nsteps - 1

        self._index = None

    @property
    def max_steps(self) -> int:
        """
        The maximum number of time steps allowed for this simulation.
        """
        return self._max_steps

    @property
    def last_step(self) -> int:
        """The time index of the last step."""
        return self._last_step

    @property
    def nstates(self) -> Dict[str, int]:
        """
        Return the dictionary containing atomic symbols as keys and the
        number of ionic species for the corresponding element as the
        value.
        """
        return self._nstates

    @property
    def elements(self) -> List[str]:
        """The elements modeled by this simulation."""
        return self._elements

    @property
    def abundances(self) -> Dict[str, float]:
        """
        The relative elemental abundances of the elements modeled in
        this simulation.

        The keys are the atomic symbols and the values are a `float`
        representing that element's elemental abundance.
        """
        return self._abundances

    @property
    def ionic_fractions(self) -> Dict[str, np.ndarray]:
        """
        Return the ionic fractions over the course of the simulation.

        The keys of this dictionary are atomic symbols.  The values are
        2D arrays where the first index refers to the time step and the
        second index refers to the integer charge.
        """
        return self._ionic_fractions

    @property
    def number_densities(self) -> Dict[str, u.Quantity]:
        """
        Return the number densities over the course of the simulation.

        The keys of ``number_densities`` are atomic symbols.  The values
        are 2D arrays with units of number density where the first index
        refers to the time step and the second index is the integer
        charge.

        """
        return self._number_densities

    @property
    def n_elem(self) -> Dict[str, u.Quantity]:
        """
        The number densities of each element over the course of the
        simulation.

        The keys of ``n_elem`` are atomic symbols.  The values are 1D
        arrays with units of number density where the index refers to
        the time step.

        """
        return self._n_elem

    @property
    def n_e(self) -> u.Quantity:
        """
        The electron number density over the course of the simulation in
        units of number density.

        The index of this array corresponds to the time step.
        """
        return self._n_e

    @property
    def T_e(self) -> u.Quantity:
        """
        The electron temperature over the course of the simulation in
        kelvin.

        The index of this array corresponds to the time step.
        """
        return self._T_e

    @property
    def time(self) -> u.Quantity:
        """
        The time for each time step over the course of the simulation
        in units of seconds.
        """
        return self._time


class NEI:
    r"""
    Perform and analyze a non-equilibrium ionization simulation.

    Parameters
    ----------
    inputs

    T_e: astropy.units.Quantity or callable
        The electron temperature, which may be a constant, an array of
        temperatures corresponding to the times in `time_input`, or a
        function that yields the temperature as a function of time.

    n: astropy.units.Quantity or callable
        The number density multiplicative factor.  The number density of
        each element will be ``n`` times the abundance given in
        ``abundances``.  For example, if ``abundance['H'] = 1``, then this
        will correspond to the number density of hydrogen (including
        neutral hydrogen and protons).  This factor may be a constant,
        an array of number densities over time, or a function that
        yields a number density as a function of time.

    time_input: astropy.units.Quantity, optional
        An array containing the times associated with ``n`` and ``T_e`` in
        units of time.

    time_start: astropy.units.Quantity, optional
        The start time for the simulation.  If density and/or
        temperature are given by arrays, then this argument must be
        greater than ``time_input[0]``.  If this argument is not supplied,
        then ``time_start`` defaults to ``time_input[0]`` (if given) and
        zero seconds otherwise.

    time_max: astropy.units.Quantity
        The maximum time for the simulation.  If density and/or
        temperature are given by arrays, then this argument must be less
        than ``time_input[-1]``.

    max_steps: `int`
        The maximum number of time steps to be taken during a
        simulation.

    dt: astropy.units.Quantity
        The time step.  If ``adapt_dt`` is `False`, then ``dt`` is the
        time step for the whole simulation.

    dt_max: astropy.units.Quantity
        The maximum time step to be used with an adaptive time step.

    dt_min: astropy.units.Quantity
        The minimum time step to be used with an adaptive time step.

    adapt_dt: `bool`
        If `True`, change the time step based on the characteristic
        ionization and recombination time scales and change in
        temperature.  Not yet implemented.

    safety_factor: `float` or `int`
        A multiplicative factor to multiply by the time step when
        ``adapt_dt`` is `True`.  Lower values improve accuracy, whereas
        higher values reduce computational time.  Not yet implemented.

    tol: float
        The absolute tolerance to be used in comparing ionic fractions.

    verbose: bool, optional
        A flag stating whether or not to print out information for every
        time step. Setting ``verbose`` to `True` is useful for testing.
        Defaults to `False`.

    abundances: dict

    Examples
    --------

    >>> import numpy as np
    >>> import astropy.units as u

    >>> inputs = {'H': [0.9, 0.1], 'He': [0.9, 0.099, 0.001]}
    >>> abund = {'H': 1, 'He': 0.085}
    >>> n = u.Quantity([1e9, 1e8], u.cm**-3)
    >>> T_e = np.array([10000, 40000]) * u.K
    >>> time = np.array([0, 300]) * u.s
    >>> dt = 0.25 * u.s

    The initial conditions can be accessed using the initial attribute.

    >>> sim = NEI(inputs=inputs, abundances=abund, n=n, T_e=T_e, time_input=time, adapt_dt=False, dt=dt)

    After having inputted all of the necessary information, we can run
    the simulation.

    >>> results = sim.simulate()

    The initial results are stored in the ``initial`` attribute.

    >>> sim.initial.ionic_fractions['H']
    array([0.9, 0.1])

    The final results can be access with the ``final`` attribute.

    >>> sim.final.ionic_fractions['H']
    array([0.16665179, 0.83334821])
    >>> sim.final.ionic_fractions['He']
    array([0.88685261, 0.11218358, 0.00096381])
    >>> sim.final.T_e
    <Quantity 40000. K>

    Both ``initial`` and ``final`` are instances of the ``IonizationStateCollection``
    class.

    Notes
    -----
    The ionization and recombination rates are from Chianti version
    8.7.  These rates include radiative and dielectronic recombination.
    Photoionization is not included.
    """

    def __init__(
        self,
        inputs,
        abundances: Union[Dict, str] = None,
        T_e: Union[Callable, u.Quantity] = None,
        n: Union[Callable, u.Quantity] = None,
        time_input: u.Quantity = None,
        time_start: u.Quantity = None,
        time_max: u.Quantity = None,
        max_steps: Union[int, np.integer] = 10000,
        tol: Union[int, float] = 1e-15,
        dt: u.Quantity = None,
        dt_max: u.Quantity = np.inf * u.s,
        dt_min: u.Quantity = 0 * u.s,
        adapt_dt: bool = None,
        safety_factor: Union[int, float] = 1,
        verbose: bool = False,
    ):

        try:

            self.time_input = time_input
            self.time_start = time_start
            self.time_max = time_max
            self.T_e_input = T_e
            self.n_input = n
            self.max_steps = max_steps
            self.dt_input = dt

            if self.dt_input is None:
                self._dt = self.time_max / max_steps
            else:
                self._dt = self.dt_input

            self.dt_min = dt_min
            self.dt_max = dt_max
            self.adapt_dt = adapt_dt
            self.safety_factor = safety_factor
            self.verbose = verbose

            T_e_init = self.electron_temperature(self.time_start)
            n_init = self.hydrogen_number_density(self.time_start)

            self.initial = IonizationStateCollection(
                inputs=inputs,
                abundances=abundances,
                T_e=T_e_init,
                n0=n_init,
                tol=tol,
            )

            self.tol = tol

            # TODO: Update IonizationStateCollection in PlasmaPy to have elements attribute

            self.elements = list(self.initial.ionic_fractions.keys())

            if "H" not in self.elements:
                raise NEIError("Must have H in elements")

            self.abundances = self.initial.abundances

            self._eigen_data_dict = eigen_data_dict

            if self.T_e_input is not None and not isinstance(inputs, dict):
                for element in self.initial.ionic_fractions.keys():
                    self.initial.ionic_fractions[element] = self.eigen_data_dict[
                        element
                    ].equilibrium_state(T_e_init.value)

            self._temperature_grid = self._eigen_data_dict[
                self.elements[0]
            ].temperature_grid

            self._get_temperature_index = self._eigen_data_dict[
                self.elements[0]
            ]._get_temperature_index

            self._results = None

        except Exception as e:
            raise NEIError(
                "Unable to create NEI object for:\n"
                f"     inputs = {inputs}\n"
                f" abundances = {abundances}\n"
                f"        T_e = {T_e}\n"
                f"          n = {n}\n"
                f" time_input = {time_input}\n"
                f" time_start = {time_start}\n"
                f"   time_max = {time_max}\n"
                f"  max_steps = {max_steps}\n"
            ) from e

    def equil_ionic_fractions(
        self,
        T_e: u.Quantity = None,
        time: u.Quantity = None,
    ) -> Dict[str, np.ndarray]:
        """
        Return the equilibrium ionic fractions for a temperature or at
        a given time.

        Parameters
        ----------
        T_e: astropy.units.Quantity, optional
            The electron temperature in units that can be converted to
            kelvin.

        time: astropy.units.Quantity, optional
            The time in units that can be converted to seconds.

        Returns
        -------
        equil_ionfracs: `dict`
            The equilibrium ionic fractions for the elements contained
            within this class

        Notes
        -----
        Only one of ``T_e`` and ``time`` may be included as an argument.
        If neither ``T_e`` or ``time`` is provided and the temperature
        for the simulation is given by a constant, the this method will
        assume that ``T_e`` is the temperature of the simulation.
        """

        if T_e is not None and time is not None:
            raise NEIError("Only one of T_e and time may be an argument.")

        if T_e is None and time is None:
            if self.T_e_input.isscalar:
                T_e = self.T_e_input
            else:
                raise NEIError

        try:
            T_e = T_e.to(u.K) if T_e is not None else None
            time = time.to(u.s) if time is not None else None
        except Exception as exc:
            raise NEIError("Invalid input to equilibrium_ionic_fractions.") from exc

        if time is not None:
            T_e = self.electron_temperature(time)

        if not T_e.isscalar:
            raise NEIError("Need scalar input for equil_ionic_fractions.")

        return {
            element: self.eigen_data_dict[element].equilibrium_state(T_e.value)
            for element in self.elements
        }

    @property
    def elements(self) -> List[str]:
        """A `list` of the elements."""
        return self._elements

    @elements.setter
    def elements(self, elements):
        # TODO: Update this
        self._elements = elements

    @property
    def abundances(self) -> Dict[str, Union[float, int]]:
        """Return the abundances."""
        return self._abundances

    @abundances.setter
    def abundances(self, abund: Dict[Union[str, int], Union[float, int]]):

        # TODO: Update initial, etc. when abundances is updated. The
        # checks within IonizationStateCollection will also be checks for

        # TODO: Update initial and other attributes when abundances is
        # updated.

        self._abundances = abund

    @property
    def tol(self) -> float:
        """
        The tolerance for comparisons between different ionization
        states.
        """
        return self._tol

    @tol.setter
    def tol(self, value: Union[float, int]):
        try:
            value = float(value)
        except Exception as exc:
            raise TypeError(f"Invalid tolerance: {value}") from exc
        if not 0 <= value < 1:
            raise ValueError("Need 0 <= tol < 1.")
        self._tol = value

    @property
    def time_input(self) -> u.s:
        return self._time_input

    @time_input.setter
    def time_input(self, times: u.s):
        if times is None:
            self._time_input = None
        elif isinstance(times, u.Quantity):
            if times.isscalar:
                raise ValueError("time_input must be an array.")
            try:
                times = times.to(u.s)
            except u.UnitConversionError:
                raise u.UnitsError("time_input must have units of seconds.") from None
            if not np.all(times[1:] > times[:-1]):
                raise ValueError("time_input must monotonically increase.")
            self._time_input = times
        else:
            raise TypeError("Invalid time_input.")

    @property
    def time_start(self) -> u.s:
        """The start time of the simulation."""
        return self._time_start

    @time_start.setter
    def time_start(self, time: u.s):
        if time is None:
            self._time_start = 0.0 * u.s
        elif isinstance(time, u.Quantity):
            if not time.isscalar:
                raise ValueError("time_start must be a scalar")
            try:
                time = time.to(u.s)
            except u.UnitConversionError:
                raise u.UnitsError("time_start must have units of seconds") from None
            if (
                hasattr(self, "_time_max")
                and self._time_max is not None
                and self._time_max <= time
            ):
                raise ValueError("Need time_start < time_max.")
            if self.time_input is not None and self.time_input.min() > time:
                raise ValueError("time_start must be less than min(time_input)")
            self._time_start = time
        else:
            raise TypeError("Invalid time_start.") from None

    @property
    def time_max(self) -> u.s:
        """The maximum time allowed for the simulation."""
        return self._time_max

    @time_max.setter
    def time_max(self, time: u.s):
        if time is None:
            self._time_max = (
                self.time_input[-1] if self.time_input is not None else np.inf * u.s
            )
        elif isinstance(time, u.Quantity):
            if not time.isscalar:
                raise ValueError("time_max must be a scalar")
            try:
                time = time.to(u.s)
            except u.UnitConversionError:
                raise u.UnitsError("time_max must have units of seconds") from None
            if (
                hasattr(self, "_time_start")
                and self._time_start is not None
                and self._time_start >= time
            ):
                raise ValueError("time_max must be greater than time_start")
            self._time_max = time
        else:
            raise TypeError("Invalid time_max.") from None

    @property
    def adapt_dt(self) -> Optional[bool]:
        """
        Return `True` if the time step is set to be adaptive, `False`
        if the time step is set to not be adapted, and `None` if this
        attribute was not set.
        """
        return self._adapt_dt

    @adapt_dt.setter
    def adapt_dt(self, choice: Optional[bool]):
        if choice is None:
            self._adapt_dt = self.dt_input is None
        elif choice is True or choice is False:
            self._adapt_dt = choice
        else:
            raise TypeError("Invalid value for adapt_dt")

    @property
    def dt_input(self) -> u.s:
        """Return the inputted time step."""
        return self._dt_input

    @dt_input.setter
    def dt_input(self, dt: u.s):
        if dt is None:
            self._dt_input = None
        elif isinstance(dt, u.Quantity):
            try:
                dt = dt.to(u.s)
                if dt > 0 * u.s:
                    self._dt_input = dt
            except (AttributeError, u.UnitConversionError):
                raise NEIError("Invalid dt.")

    @property
    def dt_min(self) -> u.s:
        """The minimum time step."""
        return self._dt_min

    @dt_min.setter
    def dt_min(self, value: u.s):
        if not isinstance(value, u.Quantity):
            raise TypeError("dt_min must be a Quantity.")
        try:
            value = value.to(u.s)
        except u.UnitConversionError as exc:
            raise u.UnitConversionError("Invalid units for dt_min.") from exc
        if (
            hasattr(self, "_dt_input")
            and self.dt_input is not None
            and self.dt_input < value
        ):
            raise ValueError("dt_min cannot exceed the inputted time step.")
        if hasattr(self, "_dt_max") and self.dt_max < value:
            raise ValueError("dt_min cannot exceed dt_max.")
        self._dt_min = value

    @property
    def dt_max(self) -> u.s:
        return self._dt_max

    @dt_max.setter
    def dt_max(self, value: u.s):
        if not isinstance(value, u.Quantity):
            raise TypeError("dt_max must be a Quantity.")
        try:
            value = value.to(u.s)
        except u.UnitConversionError as exc:
            raise u.UnitConversionError("Invalid units for dt_max.") from exc
        if (
            hasattr(self, "_dt_input")
            and self.dt_input is not None
            and self.dt_input > value
        ):
            raise ValueError("dt_max cannot be less the inputted time step.")
        if hasattr(self, "_dt_min") and self.dt_min > value:
            raise ValueError("dt_min cannot exceed dt_max.")
        self._dt_max = value

    @property
    def safety_factor(self):
        """
        The multiplicative factor that the time step is to be multiplied
        by when using an adaptive time step.
        """
        return self._safety_factor

    @safety_factor.setter
    def safety_factor(self, value):
        if not isinstance(value, (float, np.float64, np.integer, int)):
            raise TypeError
        if 1e-3 <= value <= 1e3:
            self._safety_factor = value
        else:
            raise NEIError("Invalid safety factor.")

    @property
    def verbose(self) -> bool:
        """
        Return `True` if verbose output during a simulation is
        requested, and `False` otherwise.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, choice: bool):
        self._verbose = choice

    @u.quantity_input
    def in_time_interval(self, time: u.s, buffer: u.s = 1e-9 * u.s):
        """
        Return `True` if the ``time`` is between ``time_start - buffer``
        and ``time_max + buffer`` , and `False` otherwise.

        Raises
        ------
        TypeError
            If ``time`` or ``buffer`` is not a ``astropy.units.Quantity``

        astropy.units.UnitsError
            If ``time`` or ``buffer`` is not in units of time.

        """
        return self.time_start - buffer <= time <= self.time_max + buffer

    @property
    def max_steps(self) -> int:
        """
        The maximum number of steps that a simulation will be allowed
        to take.
        """
        return self._max_steps

    @max_steps.setter
    def max_steps(self, n: int):
        if isinstance(n, (int, np.integer)) and 0 < n <= 1000000:
            self._max_steps = n
        else:
            raise TypeError(
                "max_steps must be an integer with 0 < max_steps <= 1000000"
            )

    @property
    def T_e_input(self) -> Union[u.Quantity, Callable]:
        """
        The temperature input.
        """
        return self._T_e_input

    @T_e_input.setter
    def T_e_input(self, T_e: Optional[Union[Callable, u.Quantity]]):
        """Set the input electron temperature."""
        if isinstance(T_e, u.Quantity):
            try:
                T_e = T_e.to(u.K, equivalencies=u.temperature_energy())
            except u.UnitConversionError:
                raise u.UnitsError("Invalid electron temperature.") from None
            if T_e.isscalar:
                self._T_e_input = T_e
                self._electron_temperature = lambda time: T_e
            else:
                if self._time_input is None:
                    raise TypeError("Must define time_input prior to T_e for an array.")
                time_input = self.time_input
                if len(time_input) != len(T_e):
                    raise ValueError("len(T_e) not equal to len(time_input).")
                f = interpolate.interp1d(
                    time_input.value,
                    T_e.value,
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                self._electron_temperature = lambda time: f(time.value) * u.K
                self._T_e_input = T_e
        elif callable(T_e):
            if self.time_start is not None:
                try:
                    T_e(self.time_start).to(u.K)
                    T_e(self.time_max).to(u.K)
                except Exception:
                    raise ValueError("Invalid electron temperature function.")
            self._T_e_input = T_e
            self._electron_temperature = T_e
        elif T_e is None:
            self._electron_temperature = lambda: None
        else:
            raise TypeError("Invalid T_e")

    def electron_temperature(self, time: u.Quantity) -> u.Quantity:
        try:
            if not self.in_time_interval(time):
                warnings.warn(
                    f"{time} is not in the simulation time interval:"
                    f"[{self.time_start}, {self.time_max}]. "
                    "May be extrapolating temperature."
                )
            T_e = self._electron_temperature(time.to(u.s))
            if np.isnan(T_e) or np.isinf(T_e) or T_e < 0 * u.K:
                raise NEIError(f"T_e = {T_e} at time = {time}.")
            return T_e
        except Exception as exc:
            raise NEIError(
                f"Unable to calculate a valid electron temperature for time {time}"
            ) from exc

    @property
    def n_input(self) -> u.Quantity:
        """The number density factor input."""
        if "H" in self.elements:
            return self._n_input
        else:
            raise ValueError

    @n_input.setter
    def n_input(self, n: u.Quantity):
        if isinstance(n, u.Quantity):
            try:
                n = n.to(u.cm ** -3)
            except u.UnitConversionError:
                raise u.UnitsError("Invalid hydrogen density.")
            if n.isscalar:
                self._n_input = n
                self.hydrogen_number_density = lambda time: n
            else:
                if self._time_input is None:
                    raise TypeError("Must define time_input prior to n for an array.")
                time_input = self.time_input
                if len(time_input) != len(n):
                    raise ValueError("len(n) is not equal to len(time_input).")
                f = interpolate.interp1d(
                    time_input.value,
                    n.value,
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                self._hydrogen_number_density = lambda time: f(time.value) * u.cm ** -3
                self._n_input = n
        elif callable(n):
            if self.time_start is not None:
                try:
                    n(self.time_start).to(u.cm ** -3)
                    n(self.time_max).to(u.cm ** -3)
                except Exception:
                    raise ValueError("Invalid number density function.")
            self._n_input = n
            self._hydrogen_number_density = n
        elif n is None:
            self._hydrogen_number_density = lambda: None
        else:
            raise TypeError("Invalid n.")

    def hydrogen_number_density(self, time: u.Quantity) -> u.Quantity:
        try:
            time = time.to(u.s)
        except (AttributeError, u.UnitsError):
            raise NEIError("Invalid time in hydrogen_density")
        return self._hydrogen_number_density(time)

    @property
    def eigen_data_dict(self) -> Dict[str, EigenData]:
        """
        Return a `dict` containing `~plasmapy_nei.eigen.EigenData` instances
        for each element.
        """
        return self._eigen_data_dict

    @property
    def initial(self) -> IonizationStateCollection:
        """
        Return the ionization states of the plasma at the beginning of
        the simulation.
        """
        return self._initial

    @initial.setter
    def initial(self, initial_states: IonizationStateCollection):
        if isinstance(initial_states, IonizationStateCollection):
            self._initial = initial_states
            self._elements = (
                initial_states.ionic_fractions.keys()
            )  # TODO IonizationStateCollection
        elif initial_states is None:
            self._ionstates = None
        else:
            raise TypeError("Expecting an IonizationStateCollection instance.")

    @property
    def results(self) -> SimulationResults:
        """
        Return the `~plasmapy_nei.nei.SimulationResults` class instance that
        corresponds to the simulation results.

        """
        if self._results is not None:
            return self._results
        else:
            raise AttributeError("The simulation has not yet been performed.")

    @property
    def final(self) -> IonizationStateCollection:
        """
        Return the ionization states of the plasma at the end of the
        simulation.
        """
        try:
            return self._final
        except AttributeError:
            raise NEIError("The simulation has not yet been performed.") from None

    def _initialize_simulation(self):

        self._results = SimulationResults(
            initial=self.initial,
            n_init=self.hydrogen_number_density(self.time_start),
            T_e_init=self.electron_temperature(self.time_start),
            max_steps=self.max_steps,
            time_start=self.time_start,
        )
        self._old_time = self.time_start.to(u.s)
        self._new_time = self.time_start.to(u.s)

    def simulate(self) -> SimulationResults:
        """
        Perform a non-equilibrium ionization simulation.

        Returns
        -------
        results: `~plasmapy_nei.classes.Simulation`
            The results from the simulation (which are also stored in
            the ``results`` attribute of the `~plasmapy_nei.nei.NEI`
            instance this method was called from.

        """

        self._initialize_simulation()

        for _ in range(self.max_steps):
            try:
                self.set_timestep()
                self.time_advance()
            except StopIteration:
                break
            except Exception as exc:
                raise NEIError("Unable to complete simulation.") from exc

        self._finalize_simulation()

        # Is there a way to use the inspect package or something similar
        # to only return self.results if it is in an expression where

        return self.results

    def _finalize_simulation(self):
        self._results._cleanup()

        final_ionfracs = {
            element: self.results.ionic_fractions[element][-1, :]
            for element in self.elements
        }

        self._final = IonizationStateCollection(
            inputs=final_ionfracs,
            abundances=self.abundances,
            n0=np.sum(self.results.number_densities["H"][-1, :]),  # modify this later?,
            T_e=self.results.T_e[-1],
            tol=1e-6,
        )

        if not np.isclose(self.time_max / u.s, self.results.time[-1] / u.s):
            warnings.warn(
                f"The simulation ended at {self.results.time[-1]}, "
                f"which is prior to time_max = {self.time_max}."
            )

    def _set_adaptive_timestep(self):
        """Adapt the time step."""

        t = self._new_time if hasattr(self, "_new_time") else self.t_start

        # We need to guess the timestep in order to narrow down what the
        # timestep should be.  If we are in the middle of a simulation,
        # we can use the old timestep as a reasonable guess.  If we are
        # simulation, then we can either use the inputted timestep or
        # estimate it from other inputs.

        dt_guess = self._dt or self._dt_input or self.time_max / self.max_steps

        # Make sure that dt_guess does not lead to a time that is out
        # of the domain.

        dt_guess = dt_guess if t + dt_guess <= self.time_max - t else self.time_max - t

        # The temperature may start out exactly at the boundary of a
        # bin, so we check what bin it is in just slightly after to
        # figure out which temperature bin the plasma is entering.

        T = self.electron_temperature(t + 1e-9 * dt_guess)

        # Find the boundaries to the temperature bin.

        index = self._get_temperature_index(T.to(u.K).value)
        T_nearby = np.array(self._temperature_grid[index - 1 : index + 2]) * u.K
        T_boundary = (T_nearby[:-1] + T_nearby[1:]) / 2

        # In order to use Brent's method, we must bound the root's
        # location.  Functions may change sharply or slowly, so we test
        # different times that are logarithmically spaced to find the
        # first one that is outside of the boundary.

        dt_spread = (
            np.geomspace(1e-9 * dt_guess.value, (self.time_max - t).value, num=100)
            * u.s
        )
        time_spread = t + dt_spread
        T_spread = [self.electron_temperature(time) for time in time_spread]
        in_range = [T_boundary[0] <= temp <= T_boundary[1] for temp in T_spread]

        # If all of the remaining temperatures are in the same bin, then
        # the temperature will be roughly constant for the rest of the
        # simulation.  Take one final long time step, unless it exceeds
        # dt_max.

        if all(in_range):
            new_dt = self.time_max - t
            self._dt = new_dt if new_dt <= self.dt_max else self.dt_max
            return

        # Otherwise, we need to find the first index in the spread that
        # corresponds to a temperature outside of the temperature bin
        # for this time step.

        first_false_index = in_range.index(False)

        # We need to figure out if the temperature is dropping so that
        # it crosses the low temperature boundary of the bin, or if it
        # is rising so that it crosses the high temperature of the bin.

        T_first_outside = self.electron_temperature(time_spread[first_false_index])

        if T_first_outside >= T_boundary[1]:
            boundary_index = 1
        elif T_first_outside <= T_boundary[0]:
            boundary_index = 0

        # Select the values for the time step in the spread just before
        # and after the temperature leaves the temperature bin as bounds
        # for the root finding method.

        dt_bounds = (dt_spread[first_false_index - 1 : first_false_index + 1]).value

        # Define a function for the difference between the temperature
        # and the temperature boundary as a function of the value of the
        # time step.

        T_val = lambda dtval: (
            self.electron_temperature(t + dtval * u.s) - T_boundary[boundary_index]
        ).value

        # Next we find the root.  This method should succeed as long as
        # the root is bracketed by dt_bounds.  Because astropy.units is
        # not fully compatible with SciPy, we temporarily drop units and
        # then reattach them.

        try:
            new_dt = (
                optimize.brentq(
                    T_val,
                    *dt_bounds,
                    xtol=1e-14,
                    maxiter=1000,
                    disp=True,
                )
                * u.s
            )
        except Exception as exc:
            raise NEIError(f"Unable to find new dt at t = {t}") from exc
        else:
            if np.isnan(new_dt.value):
                raise NEIError(f"new_dt = {new_dt}")

        # Enforce that the time step is in the interval [dt_min, dt_max].

        if new_dt < self.dt_min:
            new_dt = self.dt_min
        elif new_dt > self.dt_max:
            new_dt = self.dt_max

        # Store the time step as a private attribute so that it can be
        # used in the time advance.

        self._dt = new_dt.to(u.s)

    def set_timestep(self, dt: u.Quantity = None):
        """
        Set the time step for the next non-equilibrium ionization time
        advance.

        Parameters
        ----------
        dt: astropy.units.Quantity, optional
            The time step to be used for the next time advance.

        Notes
        -----
        If ``dt`` is not `None`, then the time step will be set to ``dt``.

        If ``dt`` is not set and the ``adapt_dt`` attribute of an
        `~plasmapy_nei.nei.NEI` instance is `True`, then this method will
        calculate the time step corresponding to how long it will be
        until the temperature rises or drops into the next temperature
        bin.  If this time step is between ``dtmin`` and ``dtmax``, then

        If ``dt`` is not set and the ``adapt_dt`` attribute is `False`,
        then this method will set the time step as what was inputted to
        the `~plasmapy_nei.nei.NEI` class upon instantiation in the
        ``dt`` argument or through the `~plasmapy_nei.nei.NEI` class's
        ``dt_input`` attribute.

        Raises
        ------
        ~plasmapy_nei.nei.NEIError
            If the time step cannot be set, for example if the ``dt``
            argument is invalid or the time step cannot be adapted.
        """

        if dt is not None:
            # Allow the time step to set as an argument to this method.
            try:
                dt = dt.to(u.s)
            except Exception as exc:
                raise NEIError(f"{dt} is not a valid time step.") from exc
            finally:
                self._dt = dt
        elif self.adapt_dt:
            try:
                self._set_adaptive_timestep()
            except Exception as exc:
                raise NEIError("Unable to adapt the time step.") from exc
        elif self.dt_input is not None:
            self._dt = self.dt_input
        else:
            raise NEIError("Unable to set the time step.")

        self._old_time = self._new_time
        self._new_time = self._old_time + self._dt

        if self._new_time > self.time_max:
            self._new_time = self.time_max
            self._dt = self._new_time - self._old_time

    def time_advance(self):
        """Advance the simulation by one time step."""
        # TODO: Expand docstring and include equations!

        # TODO: Fully implement units into this.

        step = self.results._index
        T_e = self.results.T_e[step - 1].value
        n_e = self.results.n_e[step - 1].value  # set average
        dt = self._dt.value

        if self.verbose:
            print(f"step={step}  T_e={T_e}  n_e={n_e}  dt={dt}")

        new_ionic_fractions = {}

        try:
            for elem in self.elements:
                nstates = self.results.nstates[elem]
                f0 = self.results._ionic_fractions[elem][self.results._index - 1, :]

                evals = self.eigen_data_dict[elem].eigenvalues(T_e=T_e)
                evect = self.eigen_data_dict[elem].eigenvectors(T_e=T_e)
                evect_inverse = self.eigen_data_dict[elem].eigenvector_inverses(T_e=T_e)

                diagonal_evals = np.zeros((nstates, nstates), dtype=np.float64)
                for ii in range(nstates):
                    diagonal_evals[ii, ii] = np.exp(evals[ii] * dt * n_e)

                matrix_1 = np.dot(diagonal_evals, evect)
                matrix_2 = np.dot(evect_inverse, matrix_1)

                ft = np.dot(f0, matrix_2)

                # Due to truncation errors in the solutions in the
                # eigenvalues and eigenvectors, there is a chance that
                # very slightly negative ionic fractions will arise.
                # These are not natural and will make the code grumpy.
                # For these reasons, the ionic fractions will be very
                # slightly unnormalized.  We set negative ionic
                # fractions to zero and renormalize.

                ft[np.where(ft < 0.0)] = 0.0
                new_ionic_fractions[elem] = ft / np.sum(ft)

        except Exception as exc:
            raise NEIError(f"Unable to do time advance for {elem}") from exc
        else:

            new_time = self.results.time[self.results._index - 1] + self._dt
            self.results._assign(
                new_time=new_time,
                new_ionfracs=new_ionic_fractions,
                new_T_e=self.electron_temperature(new_time),
                new_n=self.hydrogen_number_density(new_time),
            )

        if new_time >= self.time_max or np.isclose(new_time.value, self.time_max.value):
            raise StopIteration

    def save(self, filename: str = "nei.h5"):
        """
        Save the `~plasmapy_nei.nei.NEI` instance to an HDF5 file.  Not
        implemented.
        """
        raise NotImplementedError

    def index_to_time(self, index):
        """
        Returns the time value or array given the index/indices

        Parameters
        ----------
        index: array-like
               A value or array of values representing the index of
               the time array created by the simulation

        Returns
        -------
        get_time: astropy.units.Quantity
                  The time value associated with index input(s)
        """

        return self.results.time[index]

    def time_to_index(self, time):
        """
        Returns the closest index value or array for the given time(s)

        Parameters
        ----------
        time: array-like,
               A value or array of values representing the values of
               the time array created by the simulation

        Returns
        -------
        index: int or array-like,
                  The index value associated with the time input(s)
        """
        return (np.abs(self.results.time.value - time)).argmin()
