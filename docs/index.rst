PlasmaPy-NEI Documentation
==========================

The ``plasmapy_nei`` package is being developed as an affiliated package
of `PlasmaPy <https://docs.plasmapy.org>`_ designed to perform
non-equilibrium ionization modelling of plasma.  Use cases include the
solar wind and coronal mass ejections.

Non-equilibrium ionization
--------------------------

Plasma that is kept at a constant temperature will eventually reach
*ionization equilibrium*: a state where the ionization rate from an ion
or neutral atom with charge :math:`Z` balances the recombination rate
from an ion with charge :math:`Z+1`.  Ionization equilibrium is a valid
assumption when the changes in temperature are much longer than the
characteristic time scales for ionization and recombination.  The
assumption of ionization equilibrium is built into many analysis
techniques, such as differential emission measure (DEM) analyses of the
solar corona and other astrophysical plasmas.  This assumption, when
valid, greatly simplifies the interpretation of astrophysical spectra.

However, the temperature and density of solar and astrophysical plasma
often changes on time scales shorter than the time scale for ionization
and recombination.  As a consequence, the plasma ends up in a state of
*non-equilibrium ionization (NEI)*.  For an example in solar physics,
plasma in a coronal mass ejection (CME) drops in density while
propagating out of the solar corona.  At the lowest heights, the density
is high enough that ionization and recombination can keep up with the
temperature changes.  As the plasma moves away from the Sun, the density
drops and the ionization and recombination time scales exceed the
propagation time scales.  Eventually the charge states freeze out and
remain roughly constant for longer than it takes for the plasma to
depart the heliosphere.  Similarly, low-density plasma that is suddenly
heated by a supernova remnant shock wave will be out of ionization
equilibrium for some time.  In these situations, the charge state
distributions must be found by evolving the time-dependent ionization
equations.

The `plasmapy_nei` package is intended to enable students and
scientists to perform NEI models of laboratory, heliospheric, and
astrophysical plasma.  The early versions of this package will account
for collisional ionization, radiative recombination, and dielectronic
recombination.  The time advance is calculated using the eigenvalue
method.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   `plasmapy_nei.eigen` <eigen/index>
   `plasmapy_nei.nei` <nei/index>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
