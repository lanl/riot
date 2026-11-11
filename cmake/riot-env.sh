#!/usr/bin/env sh

if [ -n "${BASH_VERSION:-}" ]; then
  _riot_env_file=${BASH_SOURCE[0]}
elif [ -n "${ZSH_VERSION:-}" ]; then
  _riot_env_file=${(%):-%x}
else
  echo "riot-env.sh must be sourced from Bash or Zsh" >&2
  return 1 2>/dev/null || exit 1
fi

RIOT_ROOT=$(CDPATH= cd -- "$(dirname -- "${_riot_env_file}")" && pwd -P)
RIOT_SOURCE_DIR="${RIOT_ROOT}/share/riot/source"
RIOT_INPUT_DIR="${RIOT_SOURCE_DIR}/inputs"
RIOT_SCRIPT_DIR="${RIOT_SOURCE_DIR}/script"

case ":${PATH}:" in
  *":${RIOT_ROOT}/bin:"*) ;;
  *) PATH="${RIOT_ROOT}/bin:${PATH}" ;;
esac

_riot_python_path="${RIOT_ROOT}/python:${RIOT_SCRIPT_DIR}/inputs:${RIOT_SCRIPT_DIR}"
case ":${PYTHONPATH:-}:" in
  *":${_riot_python_path}:"*) ;;
  "::") PYTHONPATH=${_riot_python_path} ;;
  *) PYTHONPATH="${_riot_python_path}:${PYTHONPATH}" ;;
esac

export PATH PYTHONPATH RIOT_ROOT RIOT_SOURCE_DIR RIOT_INPUT_DIR RIOT_SCRIPT_DIR

unset _riot_env_file _riot_python_path
