# Preliminary dependencies

You need build-essential for compilers, cmake for the configuration,
some MPI and some parallel HDF5. You also need git for version control. I do:

```bash
sudo apt install build-essential libhdf5-mpich-dev git cmake
```
You need a compiler with at least C++20 but most modern compilers on modern systems have

You also need Python if you want to use the Python bindings. I manage
mine with
[micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). But
pick your poison. The main python library required is numpy.

## Python virtualenv

Personally I use a Python virtual environment to isolate what I'm doing for riot from Python globally. I do something like:

```bash
python -m venv riot_venv
. riot_venv/bin/activate
pip install numpy
```

Be aware that sometimes the Python h5py library can interfere with
your parallel hdf5 build. So it's best to only have h5py installed
inside a virtual environment or conda environment that cmake can't see
when you're bulding riot.

# Configuring and building

Riot is built in two steps, the first is a configuration step where CMake is used to generate a Makefile. Create a build directory inside the riot directory:

```bash
mkdir -p riot/build
cd riot/build
```

and then call cmake as

```bash
cmake ..
```

If succesful, you can go on to build with

```bash
make -j6
```

and finally, you do need to install riot in your virtual environment
for the Python infrastructure to work:

```bash
make install
```
