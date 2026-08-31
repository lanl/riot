# Build RIOT on Windows with WSL

These instructions build RIOT in Ubuntu running under Windows Subsystem for Linux (WSL). They assume Ubuntu 24.04 and Bash.

## 1. Install WSL and Ubuntu

Open Windows PowerShell as an administrator, list available distributions, and install Ubuntu:

```powershell
wsl --list --online
wsl --install Ubuntu-24.04
```

Restart Windows if prompted, then launch Ubuntu and create a Linux username and password. Use the Linux terminal for the remaining steps.

## 2. Install system packages

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential libhdf5-mpich-dev git cmake curl
```

Install the Python development packages. Replace `3.12` with the version provided by your distribution if necessary (`python3 --version`):

```bash
sudo apt install -y \
  python3.12-dev python3.12-venv python3-pip \
  python3-numpy python3-numpy-dev libpython3.12-dev
```

## 3. Create a Python environment

Micromamba is convenient, but another environment manager may be used. To install Micromamba:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc
```

Create and activate an environment for RIOT:

```bash
micromamba create -n riot python=3.12 numpy matplotlib scipy h5py \
  'hdf5=*=mpi*' zlib cmake cxx-compiler c-compiler pkg-config
micromamba activate riot
```

Use the same Python version in the environment and in the development packages above.

## 4. Clone RIOT

Clone the repository with its submodules:

```bash
cd ~
git clone --recursive git@github.com:lanl/riot.git
cd riot
```

If SSH is not configured, use the repository's HTTPS URL instead. For SSH, add your public key to GitHub under **Settings > SSH and GPG keys**.

## 5. Configure and build

```bash
mkdir build
cd build
cmake .. \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DHDF5_ROOT="$CONDA_PREFIX" \
  -DHDF5_IS_PARALLEL=ON \
  -DHDF5_PREFER_PARALLEL=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 6
```

The default coordinate system is Cartesian. For another coordinate system, use a separate build directory and add one of these options during configuration:

```bash
-DPARTHENON_COORDINATES=UniformCylindrical
-DPARTHENON_COORDINATES=UniformSpherical
```

If CMake reports a version compatibility problem, install a supported version through your distribution or environment manager, remove only the contents of this build directory, and rerun configuration.

## 6. Install and test

Install runtime files into the active environment:

```bash
cmake --install . --prefix "$CONDA_PREFIX"
```

Run a sample problem from the build directory:

```bash
cd src
cp ../../inputs/triple/triple.py .
./riot -i "$(python triple.py)"
```

For an MPI run, use for example:

```bash
mpirun -np 4 ./riot -i "$(python triple.py)"
```

The `triple` example is Cartesian. Use an input appropriate to the selected coordinate system when testing other builds.

## 7. Access files on Windows

Windows drives are mounted under `/mnt`. For example:

```bash
cp /mnt/c/Users/<windows-user>/path/to/file.py .
```

