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

# configures environment variables for test runs both on the cluster and locally
# to be invoked inside a check(_cluster.sh) script
# This script cancels the process if required variables are not correctly set

# new environment variables defined by this script:
#    SOPLEXPATH - absolute path to invocation working directory
#    EXECUTABLE - absolute path to executable
#    FULLTSTNAME - path to .test file
#    SOLUFILE - .solu file for this test set, for parsing optimal solution values
#    SETTINGSFILE - absolute path to settings file

# get current SOPLEX path
SOPLEXPATH=$(pwd -P)

EXECUTABLE="${SOPLEXPATH}/../${EXECUTABLE}"

# search for test file in check/instancedata/testsets and in check/testset
if test -f "instancedata/testsets/${TSTNAME}.test"
then
    FULLTSTNAME="${SOPLEXPATH}/instancedata/testsets/${TSTNAME}.test"
elif test -f "testset/${TSTNAME}.test"
then
    FULLTSTNAME="${SOPLEXPATH}/testset/${TSTNAME}.test"
else
    echo "Skipping test: no ${TSTNAME}.test file found in testset/ or instancedata/testsets/"
    exit 1
fi

SETTINGSFILE="${SOPLEXPATH}/../settings/${SETTINGS}.set"
# Abort if files are missing
if ! test -f "${SETTINGSFILE}"
then
    if [ "${SETTINGSFILE}" == "${SOPLEXPATH}/../settings/default.set" ];
    then
        touch "${SETTINGSFILE}"
    else
        echo "Settings file not found: ${SETTINGSFILE}"
        exit 1
    fi
fi

# look for solufiles under the name of the test, the name of the test with everything after the first "_" or "-" stripped, and all;
# prefer more specific solufile names over general ones and the instance database solufiles over those in testset/
SOLUFILE=""
for f in "${TSTNAME}" ${TSTNAME%%_*} ${TSTNAME%%-*} all
do
    for d in instancedata/testsets testset
    do
        if test -f "${SOPLEXPATH}/${d}/${f}.solu"
        then
            SOLUFILE="${SOPLEXPATH}/${d}/${f}.solu"
            break
        fi
    done
    if ! test "${SOLUFILE}" = ""
    then
        break
    fi
done

