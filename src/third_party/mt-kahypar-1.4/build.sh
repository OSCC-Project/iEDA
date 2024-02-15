#!/bin/bash

function get_num_cores {
  if [[ $(uname) == "Linux" ]]; then grep -c ^processor /proc/cpuinfo; fi
  if [[ $(uname) == "Darwin" ]]; then sysctl -n hw.ncpu; fi
}

ROOT=${PWD}
if [ -d .git ]; then
  # Mt-KaHyPar is build from a git repository
  git submodule update --init --recursive
else
  # Mt-KaHyPar is build from a release archive
  # which does not include submodules
  ./scripts/checkout_submodules.sh
fi

CMAKE_COMMANDS=$1
if [ ! -f build/Makefile ]; then
  mkdir -p build
fi

cd build && cmake .. -DCMAKE_BUILD_TYPE=Release $CMAKE_COMMANDS && cd ${ROOT}
cmake  --build build --parallel "$(get_num_cores)" --target MtKaHyPar

