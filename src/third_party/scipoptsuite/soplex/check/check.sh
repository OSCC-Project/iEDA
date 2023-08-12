#!/bin/bash
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
#*                                                                           *#
#*                  This file is part of the class library                   *#
#*       SoPlex --- the Sequential object-oriented simPlex.                  *#
#*                                                                           *#
#*  Copyright 1996-2022 Zuse Institute Berlin                                *#
#*                                                                           *#
#*  Licensed under the Apache License, Version 2.0 (the "License");          *#
#*  you may not use this file except in compliance with the License.         *#
#*  You may obtain a copy of the License at                                  *#
#*                                                                           *#
#*      http://www.apache.org/licenses/LICENSE-2.0                           *#
#*                                                                           *#
#*  Unless required by applicable law or agreed to in writing, software      *#
#*  distributed under the License is distributed on an "AS IS" BASIS,        *#
#*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *#
#*  See the License for the specific language governing permissions and      *#
#*  limitations under the License.                                           *#
#*                                                                           *#
#*  You should have received a copy of the Apache-2.0 license                *#
#*  along with SoPlex; see the file LICENSE. If not email soplex@zib.de.     *#
#*                                                                           *#
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#

# Solves a number of standard settings with a short testset
# Called by 'make check' from soplex root

EXECUTABLE="${1}"
BINNAME=$(basename "${EXECUTABLE}")
HOST=$(uname -n | sed 's/\(.zib.de\)//g')
BINNAME="${BINNAME}.${HOST}"

OUTPUTDIR=results/quick

SOPLEX_BOOST_SUPPORT="$(../${EXECUTABLE} --solvemode=2 check/instances/galenet.mps 2>&1 | grep 'rational solve without Boost not defined' )"
if [[ "${SOPLEX_BOOST_SUPPORT}" =~ "rational solve without Boost not defined" ]]
then
    SETTINGSLIST=(default devex steep)
else
    SETTINGSLIST=(default devex steep exact)
fi

if ! test -f ../settings/default.set
then
    touch ../settings/default.set
fi

# Solve with the different settings
for SETTINGS in ${SETTINGSLIST[@]}
do
    ./test.sh quick "${EXECUTABLE}" "${SETTINGS}" 60 "${OUTPUTDIR}"
done

echo
echo 'Summary:'
for SETTINGS in ${SETTINGSLIST[@]}
do
    echo
    grep 'Results' -A1 ${OUTPUTDIR}'/check.quick.'${BINNAME}'.'${SETTINGS}'.res'
    echo 'check/'${OUTPUTDIR}'/check.quick.'${BINNAME}'.'${SETTINGS}'.res'
done

# Evalulate the results
