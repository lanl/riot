#!/bin/bash

# ========================================================================================
# (C) (or copyright) 2021-2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

set -euo pipefail

version="${RIOT_GOLD_VERSION:-$(date +%Y%m%d)}"
outname="${RIOT_GOLD_NAME:-riot_regression_gold_${version}.tgz}"
expected_outname="riot_regression_gold_${version}.tgz"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    echo "Usage: ${0} [--update-cmake] [test run directory]"
    echo
    echo "Builds a tarball of goldfiles."
    echo "Run only in the gold files directory."
    echo
    echo "If a path to a test run directory is optionally specified,"
    echo "files matching the names of the files in the current"
    echo "directory will be copied from that folder into this one"
    echo "and included in the tarball. If you are just updating"
    echo "the golds, you can just pass that in. If you are adding"
    echo "a new file or want to be targeted, copy the new file into"
    echo "this folder by hand and do not include this argument."
    echo
    echo "Options:"
    echo "  -h, --help    Show this help message"
    echo "  --update-cmake Update the root CMake gold version and SHA-512 after bundling"
    echo "Shell variables:"
    echo "  RIOT_GOLD_VERSION sets the goldfiles version."
    echo "     default is today's date in YYYYMMDD format."
    echo "     currently set to ${version}"
    echo "  RIOT_GOLD_NAME sets the tarball name."
    echo "     default is riot_regression_gold_[version].tgz."
    echo "     currently set to ${outname}"
}

update_cmake=false
test_run_directory=""
while (($# > 0)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --update-cmake)
            update_cmake=true
            ;;
        -*)
            echo "ERROR: Unknown option '$1'." >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "$test_run_directory" ]]; then
                echo "ERROR: Only one test run directory may be specified." >&2
                usage >&2
                exit 2
            fi
            test_run_directory="$1"
            ;;
    esac
    shift
done

if [[ -n "$test_run_directory" && ! -d "$test_run_directory" ]]; then
    echo "ERROR: Test run directory '$test_run_directory' does not exist or is not a directory." >&2
    exit 2
fi

if "$update_cmake" && [[ "$(basename -- "$outname")" != "$expected_outname" ]]; then
    echo "ERROR: --update-cmake requires an archive named '$expected_outname'." >&2
    exit 2
fi

for required_file in current_version README.md; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: Expected '$required_file' in the gold files directory." >&2
        exit 2
    fi
done

shopt -s nullglob
targetfiles=( *.phdf *.phdf.xdmf )
if ((${#targetfiles[@]} == 0)); then
    echo "ERROR: No .phdf or .phdf.xdmf gold files found." >&2
    exit 2
fi

# check if output tarball already exists
if [[ -f "$outname" ]]; then
  read -p "WARNING: Output file ${outname} already exists. Overwrite? [yN] " -n 1 -r
  echo    # (optional) move to a new line
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting"
    exit 1
  fi
fi

# check if contents of current_version file matches the version we're creating
curr_v=$(<current_version)
if [[ "${curr_v}" != "${version}" ]]; then
  read -p "WARNING: Contents of 'current_version' is '${curr_v}', but making tarball for version '${version}'. Update 'current_version' file? [yN] " -n 1 -r
  echo    # (optional) move to a new line
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting"
    exit 1
  else
    echo "Updating 'current_version' file"
    echo
    printf '%s\n' "$version" > current_version
  fi
fi

# check if version is present in README.md
if ! grep -Fq "${version}:" README.md; then
  read -p "WARNING: Contents of 'README.md' does not contain '${version}'. Update 'README.md' file? [yN] " -n 1 -r
  echo    # (optional) move to a new line
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting"
    exit 1
  else
    echo "Updating 'README.md' file"
    echo
    printf '%s\n' "- ${version}: corresponds to $(git rev-parse --short HEAD)." >> README.md
  fi
fi

if [[ -n "$test_run_directory" ]]; then
    echo "Copying dump files from the test run directory ${test_run_directory}."
    for f in "${targetfiles[@]}"; do
        echo "...${f}"
        if [[ -f "${test_run_directory}/${f}" ]]; then
            cp -- "${test_run_directory}/${f}" .
        fi
    done
fi

echo
echo "Adding the following files to tarball:"
ls -lah "${targetfiles[@]}" current_version README.md

tar czf "$outname" "${targetfiles[@]}" current_version README.md

echo "Created tarball $outname, SHA-512 hash:"
echo

sha512sum "$outname"

if "$update_cmake"; then
    cmake_file="${script_dir}/../../../../CMakeLists.txt"
    if [[ ! -f "$cmake_file" ]]; then
        echo "ERROR: Could not find root CMakeLists.txt at '$cmake_file'." >&2
        exit 1
    fi

    gold_hash=$(sha512sum "$outname" | awk '{print $1}')
    cmake_tmp=$(mktemp "${cmake_file}.tmp.XXXXXX")
    if ! awk -v version="$version" -v hash="$gold_hash" '
        BEGIN { version_found = 0; hash_found = 0; in_hash = 0 }
        /^set\(RIOT_REGRESSION_GOLD_VER / {
            sub(/set\(RIOT_REGRESSION_GOLD_VER [^ )]+/,
                "set(RIOT_REGRESSION_GOLD_VER " version)
            version_found = 1
            print
            next
        }
        /^set\(RIOT_REGRESSION_GOLD_HASH$/ {
            in_hash = 1
            print
            next
        }
        in_hash && /"SHA512=/ {
            sub(/"SHA512=[^"]*"/, "\"SHA512=" hash "\"")
            hash_found = 1
            in_hash = 0
        }
        { print }
        END {
            if (!version_found || !hash_found) {
                exit 1
            }
        }
    ' "$cmake_file" > "$cmake_tmp"; then
        rm -f "$cmake_tmp"
        echo "ERROR: Could not find the expected regression-gold settings in '$cmake_file'." >&2
        exit 1
    fi

    chmod --reference="$cmake_file" "$cmake_tmp"
    mv "$cmake_tmp" "$cmake_file"
    echo "Updated $cmake_file with gold version $version and its SHA-512 hash."
fi

echo "To publish your new gold file set, go to github.com/lanl/riot and"
echo "create a new release with tag regression-gold-${version}."
echo "Upload the created tarball as an additional file in the release."
echo "If you are not a maintainer, ask a maintainer to do this for you."
