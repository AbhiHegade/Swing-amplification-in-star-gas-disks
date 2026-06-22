# Swing-amplification-in-star-gas-disks

# Code description
The main source code is available in src/Vlasov_integrators_thick_disk.py.

The main classes are:

class Stellar_integrator - Use for integrating a pure stellar disk.

class Gas_integrator - Use for integrating a pure gaseous disk.

class Gas_plus_stars_integrator - Use for integrating a star - gas disk. Can also be used to run either the pure stellar (gas) disk by setting the gas (stellar) surface density to zero.

class two_fluid_integrator - Two fluid approximation to star - gas disk.

# Examples

Several examples are avaiable. In the Classic-Results we reproduce some important results from Binney 2019. Reproduce-Star-Disk-Paper directory contains code to reproduce all the results in the paper.
