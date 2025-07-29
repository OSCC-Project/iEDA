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
C_COMPILER_PATH="gcc-10"
DRY_RUN="OFF"
RUN_IEDA="OFF"
NO_BUILD="OFF"
DEL_BUILD="OFF"
INSTALL_DEP="OFF"
NON_INTERACTIVE="OFF"
BUILD_THREADS="$(nproc)"

CMAKE_OPTIONS=(
  "-DCMAKE_BUILD_TYPE=Release"
  "-DCMD_BUILD=ON"
  "-DBUILD_STATIC_LIB=${BUILD_STATIC_LIB:-ON}"
)
  # "-DBUILD_PYTHON=${BUILD_PYTHON:-OFF}"
  # "-DBUILD_GUI=${BUILD_GUI:-OFF}"
  # "-DCOMPATIBILITY_MODE=${COMPATIBILITY_MODE:-OFF}"
  # "-DUSE_PROFILER=${USE_PROFILER:-OFF}"
  # "-DSANITIZER=${SANITIZER:-OFF}"
  # "-DUSE_GPU=${USE_GPU:-OFF}"
G_BUILD_GENERATOR=""

# pretty print
clear="\e[0m"
bold="\e[1m"
underline="\e[4m"
red="\e[31m"
yellow="\e[33m"
green="\e[32m"

# functions
help_msg_exit()
{
echo -e "build.sh: Build iEDA executable binary"
echo -e "Usage:"
echo -e "  ${bold}bash build.sh${clear} [-h] [-n] [-r] [-b] [-d] [-i] [-p] "
echo -e "                [-g] [-s] [-P] [-G] [-C] [-D] [-y]"
echo -e "                [-b ${underline}binary path${clear}] [-j ${underline}num${clear}] [-i apt|docker]"
echo -e "Options:"
echo -e "  ${bold}-h${clear} display this help and exit"
echo -e "  ${bold}-n${clear} do not build iEDA (default OFF)"
echo -e "  ${bold}-d${clear} delete all build artifacts including cmake and rust, (default OFF)"
echo -e "  ${bold}-r${clear} run iEDA hello test after build (default OFF)"
echo -e "  ${bold}-j${clear} job threads for building iEDA (default ${BUILD_THREADS} (num of cores))"
echo -e "  ${bold}-b${clear} iEDA binary path (default at ${BINARY_DIR})"
echo -e "  ${bold}-i${clear} apt-get install (root/sudo required) dependencies before build (default OFF)"
echo -e "  ${bold}-p${clear} build Python bindings (default OFF)"
echo -e "  ${bold}-g${clear} enable GUI components (default OFF)"
echo -e "  ${bold}-s${clear} enable address sanitizer (default OFF)"
echo -e "  ${bold}-P${clear} enable performance profiling (default OFF)"
echo -e "  ${bold}-G${clear} enable GPU acceleration (default OFF)"
echo -e "  ${bold}-C${clear} enable compatibility mode (disable optimizations, default OFF)"
echo -e "  ${bold}-D${clear} dry-run mode (show cmake build commands)"
echo -e "  ${bold}-y${clear} auto confirm all actions, non-interactive mode (defaults: OFF)"
exit "$1";
}

build_ieda()
{
  check_build

  local cmake_config=(
    cmake -S "$IEDA_WORKSPACE" -B "$BUILD_DIR"
    "-DCMAKE_CXX_COMPILER=$CPP_COMPILER_PATH"
    "-DCMAKE_C_COMPILER=$C_COMPILER_PATH"
    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$BINARY_DIR"
    "${CMAKE_OPTIONS[@]}"
    "$G_BUILD_GENERATOR"
  )

  local cmake_build=(
    cmake --build "$BUILD_DIR" -j "$BUILD_THREADS" --target "$BINARY_TARGET"
  )

  echo -e "${bold}CMake config commands:${clear}"
  echo "${cmake_config[@]}"
  echo -e "${bold}CMake build commands:${clear}"
  echo "${cmake_build[@]}"

  if [[ $DRY_RUN == "ON" ]]; then
    return 0
  fi

  "${cmake_config[@]}"
  "${cmake_build[@]}"
}

check_build()
{
  check_compiler_version ${C_COMPILER_PATH}
  check_compiler_version ${CPP_COMPILER_PATH}
  check_cmake
  set_build_generator_ninja
  export CC=${C_COMPILER_PATH}
  export CXX=${CPP_COMPILER_PATH}
}

check_compiler_version() {
  local compiler_path=$1
  local compiler_name=$(basename "$compiler_path")
  local min_major=10
  local min_minor=0

  if ! command -v "$compiler_path" &> /dev/null; then
    echo -e "${red}ERROR: Compiler \"$compiler_path\" not found!${clear}"
    echo -e "Please install or specify a valid compiler path using:"
    echo -e "  ${bold}bash build.sh -c ${underline}/path/to/gcc-or-g++${clear}"
    exit 1
  fi

  local version_str=$("$compiler_path" --version | grep -E -m1 '(gcc|g\+\+)' | head -1)
  local version_num=$(echo "$version_str" | 
    grep -oP '(?<= )\d+\.\d+(?=\.)?' | 
    head -1)

  if ! [[ "$version_num" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo -e "${red}ERROR: Failed to detect $compiler_name version from:${clear}"
    echo "  $version_str"
    exit 1
  fi

  local major=$(echo "$version_num" | cut -d. -f1)
  local minor=$(echo "$version_num" | cut -d. -f2)

  if (( major > min_major )) || \
     (( major == min_major && minor >= min_minor )); then
    echo -e "${green}Validated $compiler_name version: ${version_num}${clear}"
  else
    echo -e "${red}ERROR: Minimum required $compiler_name version: ${min_major}.${min_minor} (found ${version_num})${clear}"
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
    G_BUILD_GENERATOR="-GNinja"
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
      libeigen3-dev libunwind-dev libmetis-dev libgmp-dev bison rustc cargo\
      libhwloc-dev libcairo2-dev libcurl4-openssl-dev libtbb-dev git
    exit 0
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
  elif [[ $INSTALL_DEP == "mirror" ]]; then
    sed -i \
        -e 's@//archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' \
        -e 's@//security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' \
        /etc/apt/sources.list
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
  if [[ $NON_INTERACTIVE == "ON" ]]; then
    return 0
  fi

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
    echo -e "  Docker exists, try \`docker pull iedaopensource/base:latest\` instead${clear}"
    exit 1;
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

# hello_test
run_ieda()
{
  "${BINARY_DIR}"/iEDA -script "${IEDA_WORKSPACE}"/scripts/hello.tcl
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
    pprof --svg iEDA "${PROF_REPORT}" > perf_report/"${PROF_REPORT%.prof}".svg
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
}

opt_run_ieda()
{
  RUN_IEDA="ON"
}

opt_thread_num()
{
  BUILD_THREADS="-j ${OPTARG}"
}

opt_del_build()
{
  DEL_BUILD="ON"
}

perform_clean()
{
  echo -e "${yellow}Cleaning all build artifacts...${clear}"

  local cmake_build_dir="$BUILD_DIR"
  local rust_target_dirs=$(find "$IEDA_WORKSPACE/src" -type d -name "target" \
    -exec test -f "{}/../Cargo.toml" \; -print 2>/dev/null)

  local delete_list=()
  [[ -d "$cmake_build_dir" ]] && delete_list+=("$cmake_build_dir (CMake build)")
  [[ -n "$rust_target_dirs" ]] && while IFS= read -r dir; do
    delete_list+=("$dir (Rust build)")
  done <<< "$rust_target_dirs"

  if [[ ${#delete_list[@]} -eq 0 ]]; then
    echo -e "${green}No build artifacts found, nothing to clean.${clear}"
    return 0
  fi

  echo -e "${bold}Will delete the following directories:${clear}"
  for item in "${delete_list[@]}"; do
    echo -e "  ${red}[-]${clear} $item"
  done

  if [[ $NON_INTERACTIVE == "ON" ]]; then
    [[ -d "$cmake_build_dir" ]] && rm -rf "$cmake_build_dir"
    [[ -n "$rust_target_dirs" ]] && xargs -I{} rm -rf {} <<< "$rust_target_dirs"
  else
    read -p $'\nAre you sure to delete these? [y/N] ' confirm
    [[ $confirm == [yY] ]] || return 0

    echo -e "\n${yellow}Starting deletion...${clear}"
    [[ -d "$cmake_build_dir" ]] && rm -rf "$cmake_build_dir" && echo "Deleted: $cmake_build_dir"
    [[ -n "$rust_target_dirs" ]] && while IFS= read -r dir; do
      rm -rf "$dir" && echo "Deleted: $dir"
    done <<< "$rust_target_dirs"
  fi

  echo -e "${green}Cleanup completed.${clear}"
}

opt_build_target()
{
  BINARY_TARGET=${OPTARG}
}

opt_dry_run()
{
  DRY_RUN="ON"
}

opt_non_interactive()
{
  NON_INTERACTIVE="ON"
}

# invalid args
if [[ $1 != "" ]] && [[ $1 != -* ]]; then
  help_msg_exit 1
fi

while getopts j:b:t:i:rndDypx opt; do
  case "${opt}" in
    j) opt_thread_num "$OPTARG"     ;;
    b) opt_binary_dir "$OPTARG"     ;;
    t) opt_build_target "$OPTARG"   ;;
    i) opt_install_dependencies "$OPTARG" ;;
    r) opt_run_ieda               ;;
    n) opt_no_build               ;;
    d) opt_del_build              ;;
    D) opt_dry_run                ;;
    y) opt_non_interactive        ;;
    p) CMAKE_OPTIONS+=("-DBUILD_PYTHON=ON") ;;
    g) CMAKE_OPTIONS+=("-DBUILD_GUI=ON")    ;;
    s) CMAKE_OPTIONS+=("-DSANITIZER=ON")    ;;
    P) CMAKE_OPTIONS+=("-DUSE_PROFILER=ON") ;;
    G) CMAKE_OPTIONS+=("-DUSE_GPU=ON")      ;;
    C) CMAKE_OPTIONS+=("-DCOMPATIBILITY_MODE=ON") ;;
    x) CMAKE_OPTIONS+=("-DCCLOUD_WORKAROUND=ON") ;;
    h) help_msg_exit 0            ;;
    *) help_msg_exit 1            ;;
  esac
done

if [[ $DEL_BUILD == "ON" ]]; then
  perform_clean
fi

if [[ ${INSTALL_DEP} != "OFF" ]]; then
  install_dependencies "$INSTALL_DEP"
fi

if [[ ${NO_BUILD} == "OFF" ]]; then
  build_ieda
fi

if [[ ${RUN_IEDA} == "ON" ]]; then
  run_ieda
fi
