#!/usr/bin/env bash
# ***************************************************************************************
# Copyright (c) 2023-2025 Peng Cheng Laboratory
# Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
# Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
#
# iEDA is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.
# ***************************************************************************************

set -e

# variables
IEDA_WORKSPACE=$(cd "$(dirname "$0")";pwd)
BINARY_TARGET="iEDA"
BINARY_DIR="${IEDA_WORKSPACE}/bin"
BUILD_DIR="${IEDA_WORKSPACE}/build"
CPP_COMPILER_PATH="g++-10"
RUN_IEDA="OFF"
NO_BUILD="OFF"
DEL_BUILD="OFF"
INSTALL_DEP="OFF"
BUILD_THREADS=""

# cmake defines
D_CMD_BUILD="-DCMD_BUILD=ON"
D_SANITIZER="-DSANITIZER=OFF"
D_CPP_COMPILER="-DCMAKE_CXX_COMPILER:FILEPATH=${CPP_COMPILER_PATH}"
D_BINARY_DIR="-DCMAKE_RUNTIME_OUTPUT_DIRECTORY:FILEPATH=${BINARY_DIR}"
G_BUILD_GENERATOR=""

# pretty print
clear="\e[0m"
bold="\e[1m"
underline="\e[4m"
red="\e[31m"
yellow="\e[33m"

# functions
help_msg_exit()
{
echo -e "build.sh: Build iEDA executable binary"
echo -e "Usage:"
echo -e "  ${bold}bash build.sh${clear} [-h] [-n] [-r] [-b] [-c] [-d] [-i] "
echo -e "                [-b ${underline}binary path${clear}] [-c ${underline}compiler path${clear}]"
echo -e "                [-j ${underline}num${clear}] [-i apt|nonit-apt|docker]"
echo -e "Options:"
echo -e "  ${bold}-h${clear} display this help and exit"
echo -e "  ${bold}-n${clear} do not build iEDA (default OFF)"
echo -e "  ${bold}-d${clear} delete build directory, (default OFF)"
echo -e "  ${bold}-r${clear} run iEDA after build (default OFF)"
echo -e "  ${bold}-j${clear} job threads for building iEDA (default -j128)"
echo -e "  ${bold}-b${clear} iEDA binary path (default at ${BINARY_DIR})"
echo -e "  ${bold}-c${clear} compiler(g++ version >= 10) path (default at \"$(which ${CPP_COMPILER_PATH})\")"
echo -e "  ${bold}-i${clear} apt-get install (root permission) dependencies before build (default OFF)"
exit $1;
}

build_ieda()
{
  if [[ ${DEL_BUILD} == "ON" ]]; then
    rm -rf $BUILD_DIR
  fi
  check_build
  # --graphviz=foo.dot
  cmake -S$IEDA_WORKSPACE -B$BUILD_DIR $D_SANITIZER $D_CMD_BUILD  $D_CPP_COMPILER $D_BINARY_DIR $G_BUILD_GENERATOR
  cmake --build $BUILD_DIR $BUILD_THREADS --target $BINARY_TARGET
}

check_build()
{
  check_gcc_version ${CPP_COMPILER_PATH}
  check_cmake
  set_build_generator_ninja
}

check_gcc_version()
{
  if ! command_exists $1; then
    echo -e "${red}ERROR: Compiler \"$1\" not found!
       Please install or set g++(>=10) path 
       by ${bold}bash build.sh -c ${underline}compiler path${clear}"
    help_msg_exit 1
  fi

  CPP_COMPILER_VERSION=$($1 --version | grep g++ | awk '{print $4+0}')
  # echo "g++ version: ${CPP_COMPILER_VERSION}"
  if [[ ${CPP_COMPILER_VERSION} < 10.0 ]]; then
    echo -e "${red}ERROR: minimum g++ version: 10${clear}"
    exit 1
  fi
}

check_cmake()
{
  if ! command_exists cmake ; then
    echo -e "${red}ERROR: command \"cmake\" not found!
       Please install or set cmake(>=3.11) path to env \$PATH${clear}"
    help_msg_exit 1
  fi
}

set_build_generator_ninja()
{
  if command_exists ninja; then
    G_BUILD_GENERATOR="-G Ninja"
  fi
}

# for init developer environment, need sudo
install_dependencies_apt()
{
  if command_exists apt-get; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y \
      g++-10 cmake ninja-build \
      tcl-dev libgflags-dev libgoogle-glog-dev libboost-all-dev libgtest-dev flex\
      libeigen3-dev libyaml-cpp-dev libunwind-dev libmetis-dev libgmp-dev bison rustc cargo
  else
    echo -e "${red}apt-get not found, pleas make sure you were running on Debian-Based Linux distribution${clear}"
    exit 1
  fi
}

install_dependencies()
{
  if [[ $INSTALL_DEP == "apt" ]]; then
    sys_requirement_warning
    install_dependencies_apt
  elif [[ $INSTALL_DEP == "nonit-apt" ]]; then
    install_dependencies_apt
  elif [[ $INSTALL_DEP == "docker" ]]; then
    install_docker_experimental
  else
    echo "unknown arg $INSTALL_DEP"
    help_msg_exit 1
  fi
}

# sudo
opt_install_dependencies()
{
  INSTALL_DEP=$1
}

command_exists() {
	command -v "$@" > /dev/null 2>&1
}

read_continue_or_exit()
{
  while true; do
    read -p "Continue? (y/n) " answer
    case $answer in
        [Yy]* ) break  ;;
        [Nn]* ) exit 1 ;;
        * ) echo "invalid answer";;
    esac
  done
}

# experimental function, (may) need sudo
install_docker_experimental()
{
  if command_exists docker; then
    echo -e "${yellow}Warning:"
    echo -e "  Docker exists, this installation may cause problems.${clear}"
    read_continue_or_exit
  fi

  echo -e "${yellow}Warning:\n  Experimental option, caution with sudo!\n\
  See https://docs.docker.com/engine/install for manual installation${clear}"
  read_continue_or_exit

  export DOWNLOAD_URL="https://mirrors.tuna.tsinghua.edu.cn/docker-ce"
  if command_exists curl; then
    curl -fsSL https://get.docker.com/ | sh
  elif command_exists wget; then
    wget -O- https://get.docker.com/ | sh
  else
    echo -e "${red}ERROR: please install curl or wget first${clear}"
    exit 1;
  fi
}

# TODO
run_ieda()
{
  ${BINARY_DIR}/iEDA -script ${IEDA_WORKSPACE}/scripts/hello.tcl
}

sys_requirement_warning()
{
  echo -e "${yellow}Warning:"
  echo -e "  iEDA had only been tested on Debian-Based Linux distribution (Debian 11, Ubuntu 20.04)"
  echo -e "  We recommend using Docker image (based on Debian 11): iedaopensource/base:latest"
  echo -e "  Continue the script may cause problems.${clear}"
  read_continue_or_exit
}

perf_report_svg()
{
  rm -rf perf_report
  mkdir perf_report
  for PROF_REPORT in *.prof; do
    pprof --svg iEDA ${PROF_REPORT} > perf_report/${PROF_REPORT%.prof}.svg
  done
}

opt_no_build()
{
  NO_BUILD="ON"
}

opt_binary_dir()
{
  echo "change CMAKE_RUNTIME_OUTPUT_DIRECTORY from ${BINARY_DIR} to $1"
  BINARY_DIR=$1
  D_BINARY_DIR="-DCMAKE_RUNTIME_OUTPUT_DIRECTORY:FILEPATH=${BINARY_DIR}"
}

opt_compiler_path()
{
  check_gcc_version $1
  echo "change CMAKE_CXX_COMPILER from ${CPP_COMPILER_PATH} to $1"
  CPP_COMPILER_PATH=$1
  D_CPP_COMPILER="-DCMAKE_CXX_COMPILER:FILEPATH=${CPP_COMPILER_PATH}"
}

opt_run_ieda()
{
  RUN_IEDA="ON"
}

opt_jenkins()
{
  echo "jenkins do not support task: ${OPTARG}"
  help_msg_exit 1
}

# opt_dockerbuild()
# {
#   # docker tag local-image:tagname new-repo:tagname
#   # docker push new-repo:tagname
# }

opt_thread_num()
{
  BUILD_THREADS="-j ${OPTARG}"
}

opt_del_build()
{
  DEL_BUILD="ON"
}

opt_build_target()
{
  BINARY_TARGET=${OPTARG}
}

# invalid args
if [[ $1 != "" ]] && [[ $1 != -* ]]; then
  help_msg_exit 1
fi

while getopts j:t:b:c:dnhi:r opt; do
  case "${opt}" in
    j) opt_thread_num $OPTARG     ;;
    b) opt_binary_dir $OPTARG     ;;
    t) opt_build_target $OPTARG   ;;
    c) opt_compiler_path $OPTARG  ;;
    i) opt_install_dependencies $OPTARG ;;
    r) opt_run_ieda               ;;
    n) opt_no_build               ;;
    d) opt_del_build              ;;
    h) help_msg_exit 0            ;;
    *) help_msg_exit 1            ;;
  esac
done

if [[ ${INSTALL_DEP} != "OFF" ]]; then
  install_dependencies $INSTALL_DEP
fi

if [[ ${NO_BUILD} == "OFF" ]]; then
  build_ieda
fi

if [[ ${RUN_IEDA} == "ON" ]]; then
  run_ieda
fi
