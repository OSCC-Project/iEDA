#!/bin/bash

IFS='.' read -ra version <<< "$1"

for (( i=${#version[*]}; i<4; i++ ))
do
   version[$i]=0
done

versionstring="${version[0]}.${version[1]}.${version[2]}.${version[3]}"
cmakefile=`dirname $0`/../CMakeLists.txt
dummyconfig=`dirname $0`/../src/papilo/Config.hpp

sed -i "s/^set(PAPILO_VERSION_MAJOR .*/set(PAPILO_VERSION_MAJOR ${version[0]})/g" $cmakefile
sed -i "s/^set(PAPILO_VERSION_MINOR .*/set(PAPILO_VERSION_MINOR ${version[1]})/g" $cmakefile
sed -i "s/^set(PAPILO_VERSION_PATCH .*/set(PAPILO_VERSION_PATCH ${version[2]})/g" $cmakefile

sed -i "s/^#define PAPILO_VERSION_MAJOR .*/#define PAPILO_VERSION_MAJOR ${version[0]}/g" $dummyconfig
sed -i "s/^#define PAPILO_VERSION_MINOR .*/#define PAPILO_VERSION_MINOR ${version[1]}/g" $dummyconfig
sed -i "s/^#define PAPILO_VERSION_PATCH .*/#define PAPILO_VERSION_PATCH ${version[2]}/g" $dummyconfig
sed -i "s/^#define PAPILO_VERSION_TWEAK .*/#define PAPILO_VERSION_TWEAK ${version[3]}/g" $dummyconfig
