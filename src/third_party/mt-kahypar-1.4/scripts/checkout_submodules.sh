#!/bin/bash
source scripts/submodule_heads.sh
ROOT=${PWD}

# Initialize GOOGLETEST
[ ! "$(ls -A external_tools/googletest)" ] &&
git clone https://github.com/google/googletest.git external_tools/googletest &&
cd external_tools/googletest && git checkout ${GOOGLETEST_HEAD} && cd ${ROOT}

# Initialize WHFC
[ ! "$(ls -A external_tools/WHFC)" ] &&
git clone https://github.com/larsgottesbueren/WHFC.git external_tools/WHFC &&
cd external_tools/WHFC && git checkout ${WHFC_HEAD} && cd ${ROOT}

# Initialize PYBIND11
[ ! "$(ls -A python/pybind11)" ] &&
git clone https://github.com/pybind/pybind11.git python/pybind11 &&
cd python/pybind11 && git checkout ${PYBIND11_HEAD} && cd ${ROOT}

# Initialize GROWT
[ ! "$(ls -A external_tools/growt)" ] &&
git clone --depth=1 --recursive https://github.com/TooBiased/growt.git external_tools/growt &&
cd external_tools/growt && git checkout ${GROWT_HEAD} && cd ${ROOT}

# Initialize KAHYPAR-SHARED-RESOURCES
[ ! "$(ls -A external_tools/kahypar-shared-resources)" ] &&
git clone https://github.com/kahypar/kahypar-shared-resources.git external_tools/kahypar-shared-resources &&
cd external_tools/kahypar-shared-resources && git checkout ${KAHYPAR_SHARED_RESOURCES_HEAD} && cd ${ROOT}
