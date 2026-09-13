#!/bin/bash
# Portable cloud setup for Colab / Kaggle / AutoDL.
# This script must be sourced from cloud_*.sh so PYTHON_BIN stays available:
#   source "$(dirname "$0")/setup_cloud.sh"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "setup_cloud.sh is meant to be sourced by cloud_run.sh/cloud_quick.sh" >&2
  exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  if ! "$PYTHON_BIN" -c 'import torch' >/dev/null 2>&1; then
    echo "ERROR: PYTHON_BIN=$PYTHON_BIN does not have torch." >&2
    return 1
  fi
else
  CANDIDATES=()
  if command -v python3 >/dev/null 2>&1; then
    CANDIDATES+=("$(command -v python3)")
  fi
  if command -v python >/dev/null 2>&1; then
    CANDIDATES+=("$(command -v python)")
  fi

  # command -v can miss Windows .exe entries in some shells; scan PATH manually.
  OLD_IFS=$IFS
  IFS=:
  for dir in $PATH; do
    [[ -n "$dir" ]] || dir=.
    for name in python3 python python3.exe python.exe; do
      if [[ -x "$dir/$name" ]]; then
        CANDIDATES+=("$dir/$name")
      fi
    done
  done
  IFS=$OLD_IFS

  for p in /root/miniconda3/bin/python /opt/conda/bin/python \
           "$HOME/miniconda3/bin/python" "$HOME/anaconda3/bin/python"; do
    if [[ -x "$p" ]]; then
      CANDIDATES+=("$p")
    fi
  done

  PYTHON_BIN=""
  for p in "${CANDIDATES[@]}"; do
    if "$p" -c 'import torch' >/dev/null 2>&1; then
      PYTHON_BIN="$p"
      break
    fi
  done

  if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: no Python with torch was found." >&2
    echo "Use a GPU runtime in Colab/Kaggle, or a PyTorch image in AutoDL." >&2
    return 1
  fi
fi

echo "Using Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "PyTorch: $("$PYTHON_BIN" -c 'import torch; print(torch.__version__, "| cuda:", torch.cuda.is_available())')"

if "$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  GPU_FLAG="--use_gpu"
else
  GPU_FLAG="--no_use_gpu"
fi

declare -A MODULE_TO_PACKAGE=(
  [einops]=einops
  [reformer_pytorch]=reformer-pytorch
  [sktime]=sktime
  [datasets]=datasets
  [huggingface_hub]=huggingface_hub
  [sympy]=sympy
  [pywt]=PyWavelets
  [patoolib]=patool
  [tqdm]=tqdm
  [numpy]=numpy
  [pandas]=pandas
  [scipy]=scipy
  [sklearn]=scikit-learn
  [matplotlib]=matplotlib
)

MISSING=()
for mod in "${!MODULE_TO_PACKAGE[@]}"; do
  if ! "$PYTHON_BIN" -c "import ${mod}" >/dev/null 2>&1; then
    MISSING+=("${MODULE_TO_PACKAGE[$mod]}")
  fi
done

if [[ "${#MISSING[@]}" -gt 0 ]]; then
  echo "Installing missing dependencies: ${MISSING[*]}"
  "$PYTHON_BIN" -m pip install -q "${MISSING[@]}"
fi

echo "Environment OK."
return 0
