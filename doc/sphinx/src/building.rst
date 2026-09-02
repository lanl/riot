.. This file was made with the assistance of generative AI.

.. _building-doc:

Building ``riot``
===================

``riot`` builds on Linux with CMake. Windows users should build it in Windows
Subsystem for Linux (WSL); after WSL is installed, the Linux build steps are
the same as on Ubuntu.

Prerequisites
-------------

``riot`` requires CMake 3.26 or newer, a C++17-capable compiler, MPI, and a
parallel HDF5 installation. For Ubuntu, install the base dependencies with:

.. code-block:: bash

   sudo apt update
   sudo apt install -y build-essential libhdf5-mpich-dev git cmake curl \
     python3 python3-dev python3-venv

Clone ``riot`` and all required submodules:

.. code-block:: bash

   git clone --recursive git@github.com:lanl/riot.git
   cd riot

If the repository was cloned without submodules, initialize them before
configuring:

.. code-block:: bash

   git submodule update --init --recursive

Python environment
------------------

``riot``'s Python interface requires Python development files and NumPy. A virtual
environment keeps Python packages isolated from the system installation:

.. code-block:: bash

   python3 -m venv riot_venv
   . riot_venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install numpy

Additional Python packages, such as ``scipy`` and ``h5py``, may be installed
in this virtual environment when needed by an input deck. Do not install them
into a Conda or Micromamba environment that CMake will use to locate parallel
HDF5 unless that environment provides a compatible MPI-enabled HDF5 package.

Configure, build, and install
------------------------------

Create a separate build directory. The following configuration uses the active
virtual environment for Python and installs ``riot``'s runtime files into it:

.. code-block:: bash

   cmake -S . -B build \
     -DCMAKE_BUILD_TYPE=Release \
     -DPython_ROOT_DIR="$VIRTUAL_ENV" \
     -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV"
   cmake --build build --parallel 6
   cmake --install build

The executable is built at ``build/src/riot``. To run a Cartesian example:

.. code-block:: bash

   cd build/src
   cp ../../inputs/triple/triple.py .
   ./riot -i "$(python triple.py)"
   mpirun -np 4 ./riot -i "$(python triple.py)"

The final command runs the example with four MPI ranks.

Building in WSL
---------------

Install WSL and Ubuntu from an elevated Windows PowerShell prompt:

.. code-block:: powershell

   wsl --list --online
   wsl --install Ubuntu-24.04

Restart Windows if prompted, launch Ubuntu, and create a Linux username and
password. Then follow the Ubuntu instructions above inside the Ubuntu terminal.
Windows drives are available under ``/mnt``; for example, the ``C:`` drive is
mounted at ``/mnt/c``.

Micromamba alternative
----------------------

Micromamba can be used instead of a Python virtual environment. After creating
and activating an environment with MPI-enabled HDF5, configure ``riot`` with its
prefix:

.. code-block:: bash

   "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
   source ~/.bashrc
   micromamba create -n riot python=3.12 numpy matplotlib scipy h5py \
     'hdf5=*=mpi*' zlib cmake cxx-compiler c-compiler pkg-config
   micromamba activate riot

   cmake -S . -B build \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
     -DHDF5_ROOT="$CONDA_PREFIX" \
     -DHDF5_IS_PARALLEL=ON \
     -DHDF5_PREFER_PARALLEL=ON \
     -DPython_ROOT_DIR="$CONDA_PREFIX" \
     -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"

Build options
-------------

Pass options to CMake as ``-DOPTION=VALUE``. The options below are defined by
``riot``; additional Parthenon and Kokkos options are available from their
documentation.

.. list-table::
   :header-rows: 1
   :widths: 32 12 56

   * - Option
     - Default
     - Description
   * - ``CMAKE_BUILD_TYPE``
     - ``RelWithDebInfo``
     - Build configuration. Common alternatives are ``Debug`` and ``Release``.
   * - ``RIOT_ENABLE_UNIT_TESTS``
     - ``OFF``
     - Build ``riot`` unit tests.
   * - ``RIOT_ENABLE_REGRESSION_TESTS``
     - ``OFF``
     - Build ``riot`` regression tests.
   * - ``RIOT_REGRESSION_GOLD_VER``
     - ``20260830``
     - Version of the regression-gold GitHub Release asset to use.
   * - ``RIOT_REGRESSION_GOLD_HASH``
     - SHA-512 for the default gold archive
     - Expected SHA-512 of the selected regression-gold archive. Set this with
       ``RIOT_REGRESSION_GOLD_VER`` when selecting a different version.
   * - ``RIOT_REGRESSION_GOLD_SYNC``
     - ``ON``
     - Download and extract the selected regression-gold archive during CMake
       configuration when regression tests are enabled.
   * - ``RIOT_REGRESSION_GOLD_LOCAL``
     - unset
     - Path to a pre-downloaded regression-gold archive. When set, CMake uses
       this file instead of downloading from GitHub, after verifying it against
       ``RIOT_REGRESSION_GOLD_HASH``. Useful on systems where the build node has
       no internet access: fetch the archive ahead of time (for example on a
       login node) and point this option at it.
   * - ``RIOT_ENABLE_CUDA``
     - ``OFF``
     - Enable CUDA support in ``riot`` and its in-tree dependencies.
   * - ``RIOT_ENABLE_HDF5``
     - ``ON``
     - Enable HDF5 support. MPI builds require a parallel HDF5 installation.
   * - ``RIOT_ENABLE_MPI``
     - ``ON``
     - Enable MPI support.
   * - ``RIOT_ENABLE_OPENMP``
     - ``OFF``
     - Enable OpenMP support in ``riot`` and Parthenon.
   * - ``RIOT_ENABLE_WARNINGS``
     - ``OFF``
     - Enable compiler warnings.
   * - ``RIOT_ENABLE_SANITIZE``
     - ``OFF``
     - Enable the address sanitizer.
   * - ``RIOT_ENABLE_NDI``
     - ``OFF``
     - Enable NDI support for the ``ndi2spiner`` tool.
   * - ``RIOT_BUILD_NDI``
     - ``ON`` when NDI is enabled
     - Build the bundled NDI dependency. This option is available only when
       ``RIOT_ENABLE_NDI=ON``.
   * - ``RIOT_BUILD_CATCH2``
     - ``ON``
     - Build the bundled Catch2. Set to ``OFF`` to use a system Catch2 package.
   * - ``RIOT_BUILD_PARTHENON``
     - ``ON``
     - Build the bundled Parthenon. Set to ``OFF`` to use an installed package.
   * - ``RIOT_BUILD_SINGULARITY``
     - ``ON``
     - Build the bundled singularity-eos. Set to ``OFF`` to use an installed package.
   * - ``RIOT_BUILD_PYTHON``
     - ``ON``
     - Build ``riot``'s Python interface.
   * - ``RIOT_BUILD_CAD``
     - ``OFF``
     - Enable CAD-file support through OpenCASCADE.
   * - ``RIOT_MAX_MATERIALS``
     - ``16``
     - Maximum number of materials stored per mesh block.
   * - ``RIOT_MAX_STRONG``
     - ``16``
     - Maximum number of strong materials.
   * - ``RIOT_MAX_ADV``
     - ``32``
     - Maximum number of advected quantities.
   * - ``RIOT_MAX_GROUPS``
     - ``64``
     - Maximum number of radiation energy groups.
   * - ``PARTHENON_COORDINATES``
     - ``UniformCartesian``
     - Coordinate system: ``UniformCartesian``, ``UniformCylindrical``, or
       ``UniformSpherical``. Use a separate build directory for each choice.
   * - ``MACHINE_CFG``
     - unset
     - Path to a Parthenon machine-configuration file for compiler and platform
       defaults.
