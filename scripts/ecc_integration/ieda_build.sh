#!/usr/bin/env bash
set -e
export HOME=/home/lxq
export MAMBA_ROOT_PREFIX=/home/lxq/AiEDA/micromamba
eval "$(/home/lxq/.local/bin/micromamba shell hook -s bash)"
micromamba activate ecc
export CC CXX
export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd /home/lxq/AiEDA/iEDA
echo "CXX=$CXX"; $CXX --version | head -1
MODE="${1:-build}"
if [ "$MODE" = "clean" ]; then rm -rf build; fi
echo "=== configure (iEDA, shared libs / BUILD_STATIC_LIB=OFF) ==="
cmake -S . -B build -G Ninja \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$PWD/bin" \
  -DCMAKE_BUILD_TYPE=Release -DCMD_BUILD=ON \
  -DBUILD_STATIC_LIB=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -Wl,-rpath-link,$CONDA_PREFIX/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -Wl,-rpath-link,$CONDA_PREFIX/lib" 2>&1 | tail -6
echo "=== build iEDA ==="
cmake --build build -j"$(nproc)" --target iEDA
echo "BUILD_EXIT=$?"
ls -la bin/iEDA 2>&1 || echo "iEDA NOT produced"
