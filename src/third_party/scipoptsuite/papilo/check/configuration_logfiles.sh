#!/usr/bin/env bash
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*                  This file is part of the program and library             *
#*         SCIP --- Solving Constraint Integer Programs                      *
#*                                                                           *
#*    Copyright (C) 2020-2022 Konrad-Zuse-Zentrum                            *
#*                            fuer Informationstechnik Berlin                *
#*                                                                           *
#*  SCIP is distributed under the terms of the ZIB Academic License.         *
#*                                                                           *
#*  You should have received a copy of the ZIB Academic License              *
#*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

### configures the right test output files such as the .eval, the .tmp and the .set
### files to run a test on.
### the invoking script should pass "init" as argument to signalize that
### files need to be reset

### environment variables declared in this script
### OUTFILE - the name of the (sticked together) output file
### ERRFILE - the name of the (sticked together) error file
### EVALFILE - evaluation file to glue single output and error files together
### OBJECTIVEVAL - the optimal or best-know objective value for this instance
### SHORTPROBNAME - the basename of ${INSTANCE} without file extension
### FILENAME - the basename of the local files (.out, .tmp, and .err)
### SKIPINSTANCE - should the instance be skipped because it was already evaluated in a previous setting?
### BASENAME - ${CHECKPATH}/${OUTPUTDIR}/${FILENAME} cf. FILENAME argument
### TMPFILE  - the batch file name to pass for solver instructions
### SETFILE  - the name of the settings file to save solver settings to


# checks if branch has something pending
function parse_git_dirty() {
  git diff --quiet --ignore-submodules HEAD 2>/dev/null; [ "$?" -eq 1 ] && echo "*"
}

# gets the current git branch
function parse_git_branch() {
  git branch --no-color 2> /dev/null | sed -e '/^[^*]/d' -e "s/* \(.*\)/\1$(parse_git_dirty)/"
}

# get last commit hash prepended
function parse_git_hash() {
  git rev-parse --short HEAD
}

function parse_git_time_stamp() {
  git log -1 --date=short --pretty=format:%ai
}

function parse_git_author() {
  git log -1 --pretty=format:'%an'
}

### environment variables passed as arguments to this script
INIT="${1}"                # should log files be initialized (this overwrite or copy/move some existing log files)
COUNT="${2}"               # the instance count as part of the filename
INSTANCE="${3}"            # the name of the instance
BINID="${4}"               # the ID of the binary to use
PERMUTE="${5}"             # the number of permutations to use - 0 for no permutation
SEEDS="${6}"               # the number of random seeds - 0 only default seeds
SETNAME="${7}"             # the name of the setting
TSTNAME="${8}"             # the name of the testset
CONTINUE="${9}"            # should test continue an existing run
QUEUE="${10}"            # the queue name
p="${11}"                # shift of the global permutation seed
s="${12}"                # shift of the global random seed
GLBSEEDSHIFT="${13}"     # the global seed shift
STARTPERM="${14}"        # the starting permutation

# common naming scheme for eval files
EVALFILE="${CHECKPATH}/${OUTPUTDIR}/check.${TSTNAME}.${BINID}.${QUEUE}.${SETNAME}"

# if seed is positive, add postfix
SEED=$((s + GLBSEEDSHIFT))
if test "${SEED}" -gt 0
then
    EVALFILE="${EVALFILE}-s${SEED}"
fi

# if permutation is positive, add postfix
PERM=$((p + STARTPERM))
if test "${PERM}" -gt 0
then
    EVALFILE="${EVALFILE}-p${PERM}"
fi

OUTFILE="${EVALFILE}.out"
ERRFILE="${EVALFILE}.err"

# add .eval extension to evalfile
EVALFILE="${EVALFILE}.eval"

# create meta file
if test -e "${EVALFILE}"
then
    fname="${CHECKPATH}/${OUTPUTDIR}/$(basename ${EVALFILE} .eval).meta"
    if ! test -e "${fname}"
    then
        echo "@Permutation ${PERM}"                        > "${fname}"
        echo "@Seed ${SEED}"                              >> "${fname}"
        echo "@Settings ${SETNAME}"                       >> "${fname}"
        echo "@TstName ${TSTNAME}"                        >> "${fname}"
        echo "@BinName ${BINNAME}"                        >> "${fname}"
        echo "@NodeLimit ${NODELIMIT}"                    >> "${fname}"
        echo "@MemLimit ${MEMLIMIT}"                      >> "${fname}"
        echo "@FeasTol ${FEASTOL}"                        >> "${fname}"
        echo "@Queue ${QUEUE}"                            >> "${fname}"
        echo "@Exclusive ${EXCLUSIVE}"                    >> "${fname}"
        echo "@GitBranch $(parse_git_branch)"             >> "${fname}"
        echo "@GitHash $(parse_git_hash)"                 >> "${fname}"
        echo "@CommitTimestamp $(parse_git_time_stamp)"   >> "${fname}"
        echo "@Author $(parse_git_author)"                >> "${fname}"
        if [ "${CLUSTERBENCHMARK}" == "yes" ]; then
            echo "@QueueNode ${CB_QUEUENODE}"             >> "${fname}"
            echo "@ClusterBenchmarkID ${CB_ID}"           >> "${fname}"
        fi
    fi
fi

if test "${INSTANCE}" = "DONE"
then
    return
fi

# reset files if flag is set to 'init'
if test "${INIT}" = "true"
then
    #reset the eval file
    echo > "${EVALFILE}"

    #mv existing out and error files
    if test "${CONTINUE}" = "true"
    then
        MVORCP=cp
    else
        MVORCP=mv
    fi
    DATEINT=$(date +"%s")
    for FILE in OUTFILE ERRFILE
    do
        if test -e "${FILE}"
        then
            "${MVORCP}" "${FILE}" "${FILE}.old-${DATEINT}"
        fi
    done
fi


# filter all parseable file format extensions
SHORTPROBNAME=$(basename "${INSTANCE}" .gz)
for EXTENSION in .mps .lp .opb .gms .pip .zpl .cip .fzn .osil .wbo .cnf .difflist .cbf .dat-s
do
    SHORTPROBNAME=$(basename "${SHORTPROBNAME}" "${EXTENSION}")
done

# get objective value from solution file
# we do this here to have it available for all solvers, even though it is not really related to logfiles
if test -e "${SOLUFILE}"
then
    # get the objective value from the solution file: grep for the instance name and only use entries with an optimal or best known value;
    # if there are multiple entries for this instance in the solution file, sort them by objective value and take the objective value
    # written in the last line, i.e., the largest value;
    # as a double-check, we do the same again, but reverse the sorting to get the smallest value
    OBJECTIVEVAL=$(grep " ${SHORTPROBNAME} " "${SOLUFILE}" | grep -e =opt= -e =best= | sort -k 3 -g | tail -n 1 | awk '{print ${3}}')
    CHECKOBJECTIVEVAL=$(grep " ${SHORTPROBNAME} " "${SOLUFILE}" | grep -e =opt= -e =best= | sort -k 3 -g -r | tail -n 1 | awk '{print ${3}}')

    # if largest and smalles reference value given in the solution file differ by more than 1e-04, stop because of this inconsistency
    if awk -v n1="${OBJECTIVEVAL}" -v n2="${CHECKOBJECTIVEVAL}" 'BEGIN { exit (n1 <= n2 + 0.0001 && n2 <= n1 + 0.0001) }' /dev/null;
    then
        echo "Exiting test because objective value in solu file is inconsistent: ${OBJECTIVEVAL} vs. ${CHECKOBJECTIVEVAL}"
        exit
    fi
else
    OBJECTIVEVAL=""
fi
#echo "Reference value ${OBJECTIVEVAL} ${SOLUFILE}"

NEWSHORTPROBNAME=$(echo "${SHORTPROBNAME}" | cut -c1-25)
SHORTPROBNAME="${NEWSHORTPROBNAME}"

#define file name for temporary log file
FILENAME="${USER}.${TSTNAME}.${COUNT}_${SHORTPROBNAME}.${BINID}.${QUEUE}.${SETNAME}"

# if seed is positive, add postfix
if test "${SEED}" -gt 0
then
    FILENAME="${FILENAME}-s${SEED}"
fi

# if permutation is positive, add postfix
if test "${PERM}" -gt 0
then
    FILENAME="${FILENAME}-p${PERM}"
fi

SKIPINSTANCE="false"
# in case we want to continue we check if the job was already performed
if test "${CONTINUE}" = "true" && test -e "${OUTPUTDIR}/${FILENAME}.out"
then
    echo "skipping file ${INSTANCE} due to existing output file ${OUTPUTDIR}/${FILENAME}.out"
    SKIPINSTANCE="true"
fi

# configure global names TMPFILE (batch file) and SETFILE to save settings to
BASENAME="${CHECKPATH}/${OUTPUTDIR}/${FILENAME}"
TMPFILE="${BASENAME}.tmp"
SETFILESCIP="${BASENAME}.scip.set"
SETFILEPAPILO="${BASENAME}.papilo.set"

if ! test -f "${SETFILEPAPILO}"; then
    cp "${CHECKPATH}/../settings/${SETNAME}.set" "${SETFILEPAPILO}"
fi

# generate random seed for SCIP settings
if [ ! -f "${SETFILESCIP}" ]; then
    touch "${SETFILESCIP}"
    echo randomization/randomseedshift = "${SEED}" >> "${SETFILESCIP}"
fi

# even if we decide to skip this instance, we write the basename to the eval file
echo "${OUTPUTDIR}/${FILENAME}" >> "${EVALFILE}"
