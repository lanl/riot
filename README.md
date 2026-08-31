<p align="center">
  <img src="riot_logo.png" alt="Riot Logo" width="100%">
</p>

RIOT is a performance portable, multiphysics code targeting high
energy density physics, astrophysics, and related phenomena. It
builds on an ecosystem of open source capabilities, linked to below.

Preliminary documentation is available at
[https://lanl.github.io/riot](https://lanl.github.io/riot). It will be
further fleshed out soon. Some example inputs are provided as a
jumping off point.

# Physics
- Hydrodynamics
- Multi-material, multi-phase
- Arbitrary EOS (via <a href="https://github.com/lanl/singularity-eos">singularity-eos</a>)
- Pressure-temperature equilibrium closure for mixtures
- Multi-group P1 radiation transport
- Multi-group S<sub>n</sub> transport
- Arbitrary gray and multi-group opacities (via <a href="https://github.com/lanl/singularity-opac">singularity-opac</a>)
- Nuclear reactions
- Electron thermal conduction
- Ion thermal conduction
- Ionization
- Electron/ion nonequilibrium
- Laser energy deposition
- BHR turbulence model
- Prescribed energy sources
- Constant gravity

# Features
- Built on <a href="https://github.com/parthenon-hpc-lab/parthenon">parthenon</a>, which provides
    - Block-structured adaptive mesh refinement
    - Portable and highly performant execution across CPU and GPU architectures (leveraging <a href="https://github.com/kokkos/kokkos/kokkos">kokkos</a>)
    - Highly scalable
    - Native solvers, including geometric multigrid and Krylov subspace methods
    - Sparse representations of material data
    - Task parallel execution
- Adaptive sparse physics
- Flexible setup using Python
- Inline diagnostics
- Plugin infrastructure to allow for easy extensions

## Build instructions

For detailed build instructions, see the [build document](https://lanl.github.io/riot/building.html).

## Notes on python

The code builds by default with support for inline calls to Python, which requires Python be installed on the system with development components (this is typical) and that NumPy be installed.  This can be disabled at cmake configuration time via `-DRIOT_BUILD_PYTHON=OFF`.  When Python is invoked, it must be able to find not only the file(s) containing the functionality you have defined, but also any imported modules.  The recommended way of specifying search paths for these modules is by setting the environment variable `PYTHONPATH`.  Two often useful modules are `script/inputs/riot.py` and the singularity-eos python bindings that enable calls to equations of state, which are built along with riot and show up in the build directory within `singularity-eos/python` as `singularity_eos.cpython-*`.  Pointing your `PYTHONPATH` at these directories should enable their use in your own Python files.  Alternatively, after building riot, you can invoke cmake from your build directory to install into a specified location via

```
cmake --install . --component runtime --prefix path_to_install
```

which will put the riot executable into `path_to_install/bin` and `riot.py` and `singularity_eos.cpython-*` into `path_to_install/python`.  With this option, you can just point `PYTHONPATH` at `path_to_install/python`.

Note that if your Python version does not match the version used to build riot, you may encounter compatibility issues.  One potential indication of this is that you get import failures when trying to import singularity-eos.

## Formatting

The code is formatted with clang-format-20. To format the code run

```bash
CFM=/path/to/clang/format VERBOSE=1 ./script/format.sh
```

## Primary Contributors (alphabetical)
- Josh Dolence (CAI-2, LANL)
- Sam Jones (T-5, LANL)
- Chad Meyer (XCP-4, LANL)
- Jonah Miller (CAI-2, LANL)
- Patrick Mullen (CAI-2, LANL)
- Luke Roberts (CAI-2, LANL)

## 

LANL's Richard P. Feynman Center for Innovation assigned Riot number O5171.
