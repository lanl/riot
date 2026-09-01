# ========================================================================================
#  AthenaXXX astrophysical plasma code
#  Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
#  Licensed under the 3-clause BSD License, see licenses/bsd_athenak.txt file for details
# ========================================================================================
# ========================================================================================
#  (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

# Functions for interfacing with Riot during testing

# Modules
import logging
import sys
import os
import subprocess
from timeit import default_timer as timer
from .log_pipe import LogPipe

# Global variables
riot_rel_path = "../"

riot_input_path = os.path.abspath("../script/inputs")
singularity_path = os.path.abspath("build/singularity-eos/python")
riotbuild_path = os.path.join(os.path.abspath("build"), "src")
env = os.environ.copy()
searchpaths = [riot_input_path, singularity_path, riotbuild_path]
existing = env.get("PYTHONPATH", "")
if existing:
    searchpaths.append(existing)
env["PYTHONPATH"] = os.pathsep.join(searchpaths)

# Cute use of or. If the environment variable is empty or an empty
# string, lazy evaluation gives us the second argument.
mpicommand = os.environ.get("RIOT_MPI_COMMAND") or "mpiexec"

# Function for compiling Riot
def make(cmake_args, make_nproc):
    logger = logging.getLogger("riot.make")
    out_log = LogPipe("riot.make", logging.INFO)
    current_dir = os.getcwd()
    try:
        subprocess.check_call(["mkdir", "build"], stdout=out_log)
        build_dir = current_dir + "/build/"
        os.chdir(build_dir)
        cmake_command = ["cmake", "../" + riot_rel_path] + cmake_args
        make_command = ["make", "-j" + str(make_nproc)]
        try:
            t0 = timer()
            logger.debug("Executing: " + " ".join(cmake_command))
            subprocess.check_call(cmake_command, stdout=out_log)
            logger.debug("Executing: " + " ".join(make_command))
            subprocess.check_call(make_command, stdout=out_log)
            logger.debug("Build took {0:.3g} seconds.".format(timer() - t0))
        except subprocess.CalledProcessError as err:
            logger.error("Something bad happened", exc_info=True)
            raise RiotError(
                "Return code {0} from command '{1}'".format(
                    err.returncode, " ".join(err.cmd)
                )
            )
        install_command = ["make", "install"]
        try:
            subprocess.check_call(install_command, stdout=out_log)
        except subprocess.CalledProcessError as err:
            logger.debug("Install failed, but it might be fine.", exc_info=True)
    finally:
        out_log.close()
        os.chdir(current_dir)


# Function for generating inputs from python scripts
def generate(input_script):
    logger = logging.getLogger("riot.generate")
    try:
        subprocess.check_call([sys.executable, "../inputs/" + input_script], env=env)
    except subprocess.CalledProcessError as err:
        logger.error("Something bad happened while generating inputs", exc_info=True)
        raise RiotError(
            "Return code {0} from command '{1}'".format(
                err.returncode, " ".join(err.cmd)
            )
        )


# Function for running Riot
def run(input_filename, arguments):
    out_log = LogPipe("riot.run", logging.INFO)
    current_dir = os.getcwd()
    exe_dir = current_dir + "/build/src/"
    os.chdir(exe_dir)
    try:
        input_filename_full = "../../" + riot_rel_path + "inputs/" + input_filename
        run_command = ["./riot", "-i", input_filename_full]
        try:
            cmd = run_command + arguments
            logging.getLogger("riot.run").debug("Executing: " + " ".join(cmd))
            subprocess.check_call(cmd, stdout=out_log, env=env)
        except subprocess.CalledProcessError as err:
            raise RiotError(
                "Return code {0} from command '{1}'".format(
                    err.returncode, " ".join(err.cmd)
                )
            )
    finally:
        out_log.close()
        os.chdir(current_dir)


# Function for running Riot with MPI
def mpirun(nproc, input_filename, arguments):
    out_log = LogPipe("riot.run", logging.INFO)
    current_dir = os.getcwd()
    exe_dir = current_dir + "/build/src/"
    os.chdir(exe_dir)
    try:
        input_filename_full = "../../" + riot_rel_path + "inputs/" + input_filename
        run_command = [mpicommand, "-n", str(nproc), "./riot", "-i", input_filename_full]
        try:
            cmd = run_command + arguments
            logging.getLogger("riot.run").debug("Executing: " + " ".join(cmd))
            subprocess.check_call(cmd, stdout=out_log, env=env)
        except subprocess.CalledProcessError as err:
            raise RiotError(
                "Return code {0} from command '{1}'".format(
                    err.returncode, " ".join(err.cmd)
                )
            )

    finally:
        out_log.close()
        os.chdir(current_dir)


# General exception class for these functions
class RiotError(RuntimeError):
    pass
