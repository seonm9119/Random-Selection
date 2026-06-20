#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${project_dir}/.venv"
python_bin="${PYTHON_BIN:-python3}"
requirements_path="${RANDOM_SELECTION_REQUIREMENTS_PATH:-${project_dir}/requirements.txt}"
environment_stamp_path="${venv_dir}/.train-venv-environment"
required_python_modules=("torch" "torchvision" "numpy" "PIL" "yaml" "tqdm")

get_environment_fingerprint() {
  {
    echo "requirements_path=${requirements_path}"
    sha256sum "${requirements_path}"
    echo "required_python_modules=${required_python_modules[*]}"
  } | sha256sum | awk '{print $1}'
}

create_project_venv() {
  if [ ! -x "${venv_dir}/bin/python" ]; then
    "${python_bin}" -m venv "${venv_dir}"
  fi
}

install_dependencies_from_venv_script() {
  if [ ! -f "${requirements_path}" ]; then
    echo "requirements file not found: ${requirements_path}" >&2
    exit 1
  fi

  "${venv_dir}/bin/python" -m pip install --upgrade pip wheel
  "${venv_dir}/bin/python" -m pip install -r "${requirements_path}"
  get_environment_fingerprint > "${environment_stamp_path}"
}

is_project_venv_synced() {
  if [ ! -f "${environment_stamp_path}" ]; then
    return 1
  fi

  if [ "$(cat "${environment_stamp_path}")" != "$(get_environment_fingerprint)" ]; then
    return 1
  fi

  required_modules_csv="$(IFS=,; echo "${required_python_modules[*]}")"
  REQUIRED_MODULES="${required_modules_csv}" "${venv_dir}/bin/python" - <<'PY'
import os

missing_modules = []
required_modules = os.environ["REQUIRED_MODULES"].split(",")

for module_name in required_modules:
    try:
        __import__(module_name)
    except Exception:
        missing_modules.append(module_name)

raise SystemExit(1 if missing_modules else 0)
PY
}

sync_project_venv() {
  create_project_venv

  if is_project_venv_synced; then
    return
  fi

  install_dependencies_from_venv_script
}

print_usage() {
  cat <<USAGE
Usage: ./train_venv.sh [command]

Commands:
  setup     Create/sync ${project_dir}/.venv from requirements.txt
  env-info  Print the local venv Python executable and package versions

Environment:
  RANDOM_SELECTION_REQUIREMENTS_PATH Override requirements path
  PYTHON_BIN                     Python executable used to create venv
USAGE
}

print_environment_info() {
  sync_project_venv
  required_modules_csv="$(IFS=,; echo "${required_python_modules[*]}")"
  REQUIRED_MODULES="${required_modules_csv}" "${venv_dir}/bin/python" - <<PY
import os
import sys

print(f"python={sys.executable}")
print("venv=${venv_dir}")
print("requirements=${requirements_path}")

for module_name in os.environ["REQUIRED_MODULES"].split(","):
    module = __import__(module_name)
    version = getattr(module, "__version__", "unknown")
    print(f"{module_name}={version}")
PY
}

command_name="${1:-setup}"

case "${command_name}" in
  setup)
    sync_project_venv
    ;;
  env-info)
    print_environment_info
    ;;
  help|--help|-h)
    print_usage
    ;;
  *)
    print_usage
    exit 1
    ;;
esac
