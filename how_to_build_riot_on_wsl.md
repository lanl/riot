Getting Riot Locally on Windows Machine
========================================

# Step 1 (Install WSL):

in windows PowerShell type in the following command:
```
wsl --install
```

(This will likely require r-account permissions)

# Step 2 (Install Linux in WSL):

```
type wsl.exe --list --online
```

view available Linux distributions. I went with Ubuntu 24.04, but
other distributions might also work To install the distribution:

```
wsl.exe --install <distro>
```

ex:

```
wsl.exe --install Ubuntu-24.04
```

will need to come up with a username and password to use sudo


# Step 3 (Install dependencies):

type these commands (and type y when prompted with y/n):

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential libhdf5-mpich-dev git cmake
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc

sudo apt install \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    python3-numpy \
    python3-numpy-dev \
    libpython3.12-dev
```

NOTE: use python3.## for whatever version of python you are using (check with python3 --version)

NOTE: don't have to use micromamba but I do


# Step 4 (get ssh key for this linux terminal):

- go to GitHub.com and sign in
- click the profile picture circle in the top right
- select settings
- on the left side panel click 'SSH and GPG keys'
- click 'New SSH key'
- In the box it says 'Begins with 'ssh-rsa', etc.', your id # is from 'ssh-#######' and 'sk-ssh-#######@openssh.com'

back in the Linux terminal:
```bash
cd ~
ssh-keygen -t ####### -C "moniker@lanl.gov"
```
make a file and/or passphrase if you want
find the public key to copy with
```bash
cat ~/.ssh/id_#######.pub
```

copy the output and paste in GitHub in the box where it says 'Key', name the key and save

# Step 5 (clone RIOT from re-git or GitHub):

back in the linux terminal type (re-git version):
```bash
git clone --recursive git@github.com:lanl/riot.git
```

# Step 6 (create a virtual environment):
```bash
micromamba create -n <whatever you want to name your environment> python=3.12 numpy matplotlib scipy h5py hdf5 "hdf5=*=mpi*" zlib cmake cxx-compiler c-compiler pkg-config
```

NOTE: python= needs to be whatever version of python you are running

to activate type in
```bash
micromamba activate <whatever you named the environment>
```
once activated, you need to build riot into the environment.

# Step 7 (building riot):

type these commands:
```bash
mkdir build
cd build
cmake .. \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DHDF5_ROOT=$CONDA_PREFIX \
  -DHDF5_IS_PARALLEL=ON \
  -DHDF5_PREFER_PARALLEL=ON \
  -DCMAKE_BUILD_TYPE=Release
```

[NOTE: the default coordinates are cartesian, if you want to use cylindrical do:
```bash
cmake .. -DPARTHENON_COORDINATES=UniformCylindrical \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DHDF5_ROOT=$CONDA_PREFIX \
  -DHDF5_IS_PARALLEL=ON \
  -DHDF5_PREFER_PARALLEL=ON \
  -DCMAKE_BUILD_TYPE=Release
```
Or if you want to do spherical
```bash
cmake .. -DPARTHENON_COORDINATES=UniformSpherical \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DHDF5_ROOT=$CONDA_PREFIX \
  -DHDF5_IS_PARALLEL=ON \
  -DHDF5_PREFER_PARALLEL=ON \
  -DCMAKE_BUILD_TYPE=Release
```

Personally, I have different build directories for each (build_cart, build_cyl, build_sph) or you could re-make it each time you want to switch]

[NOTE: CMake version might have compatibility issues if its too new or too old (Needs to be between 3.26 and 3.50, so Ubuntu 22.04, Ubuntu 26.04, and Ubuntu will have this problem)
if this happens, in the build directory and type:
```bash
rm -rf *
cd ~
sudo apt remove --purge cmake cmake-data
sudo snap install cmake --channel=3.30/stable --classic
export PATH="/snap/bin:$PATH"
echo 'export PATH="/snap/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```
then go back to the build directory and try cmake .. etc. again]

then type in the build directory:
```bash
make -j6
make install
```

# Step 8 (copy stuff you need in your environment):

cd into `~/riot/build/src`
type in these commands (again change the python version to be the version you're using -- NOTE: python 3.14 will be 'python3.14t'):
```bash
cp ../../script/inputs/riot.py $CONDA_PREFIX/lib/python3.12/site-packages
cp ../singularity-eos/python/singularity_eos.cpython-312-x86_64-linux-gnu.so $CONDA_PREFIX/lib/python3.12/site-packages
```

# Step 9 (see if it works):

Copy a test problem and try it out.
In riot/build/src and with your python environment activated type:
```bash
cp ../../inputs/triple/triple.py .
./riot -i $(python triple.py)
```
It should run the simulation until completion with no errors. You can also run like (with 4 just being an example #)

```bash
mpirun -np 4 ./riot -i $(python triple.py)
```

NOTE: this is a cartesian example problem, will need to modify to test cylindrical, spherical, etc.

# Step 10 (grabbing local files you might need):

copy a file from your computer to riot/build/src with:
```bash
cp /mnt/c/Users/<your Z #>/path/to/file.py .
```
you will likely need materials for non-ideal gas input decks, so I would scp them from the hpc wherever you have them and then copy them to your Linux terminal:
```bash
cp /mnt/c/Users/<your Z #>/path/to/materials.sp5 .
```

If everything works, then yay.