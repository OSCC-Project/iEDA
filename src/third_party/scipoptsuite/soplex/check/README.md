# Testsuite

In general the execution of the testruns is the following:

- To start a testrun call `make test` for a local one, `make testcluster` for one on the cluster.
  (These are for running soplex tests but variants exist for qsoptex and perplex)
- The `Makefile` will call the corresponding `check.sh` or `check_cluster_<solver>.sh` script
  with the correct variables. These scripts will
  + configure the environment variables for local and cluster runs `. configuration_set.sh`, `. configuration_cluster.sh`
  + configure the test output files such as the .eval, the .tmp and the .set files.
  + cluster run scripts have a `waitcluster.sh` script in the loop that makes the scripts wait
    instead of overloading the cluster queue with jobs.
    + run `test.sh` on the instances and call `evaluation.py` or `evaluation.sh` for local tests or
    + `runcluster_<solver>.sh` to execute the tests with the solver, later `evalcheck_cluster.sh` has to be called to evaluate the logfiles which in turn calls `evaluation.py` or `evaluation.sh`.
  + the `evaluation.*` files check the solutions that are produced by the solvers and recorded in the logfiles.

Make targets for testing
----------

## local make targets

### solve a number of standard settings on a short testset

make check
  - `check.sh`
    - `. configuration_set.sh`
    - `test.sh`
      + `evaluate.sh`
      + `evaluate.py`

### solve a testset locally

make test
  - `test.sh`
    + `. configuration_set.sh`
    + `evaluate.sh`
    + `evaluate.py`

## cluster make targets

### solve a testset on the cluster with soplex

make testcluster
  - `check_cluster.sh`
    - `. configuration_cluster.sh`
    - `. configuration_set.sh`
    - `runcluster.sh`

### solve a testset on the cluster with perplex

make testclusterperplex
  - `check_cluster_perplex.sh`
    - `. configuration_cluster.sh`
    - `. configuration_set.sh`
    - `runcluster_perplex.sh`

### solve a testset on the cluster with qsoptex

make testclusterqsoptex
  - `check_cluster_qsoptex.sh`
    - `. configuration_cluster.sh`
    - `. configuration_set.sh`
    - `runcluster_qsoptex.sh`

## other Scripts

- `compare.py` compares several SoPlex .json files.
  The first argument is the default run that is compared to all other runs.
  Set values to be compared and respective shift values in arrays 'compareValues' and 'shift'.

# Files

## Bash Scripts

- `check.sh`
  Solves a number of standard settings with a short testset
  Called by `make check` from soplex root

- `test.sh`
  solve a given testset with given settings and time limit parameters:
    1: name of testset (has to be in check/testset)
    2: path to soplex executable
    3: name of settings (has to be in settings)
    4: time limit
    5: results directory

- `check_cluster*.sh`
  Call with "make testcluster", "make testclusterperplex" and "make testclusterqsoptex"
  The queue is passed via $QUEUE (possibly defined in a local makefile in soplex/make/local).
  For each run, we can specify the number of nodes reserved for a run via $PPN. If tests runs
  with valid time measurements should be executed, this number should be chosen in such a way
  that a job is run on a single computer, i.e., in general, $PPN should equal the number of cores
  of each computer. Of course, the value depends on the specific computer/queue.

- `. configuration_cluster.sh`
  Configures environment variables for cluster runs.
  It is to be invoked inside a `check_cluster*.sh` script.
  It calls `wakeup-slurm` to the respective queue.
  + `wakeup-slurm` END

- `. configuration_set.sh`
  Configures environment variables that are needed for test runs both on the cluster and locally.
  It is to be invoked inside a `check(_cluster)*.sh` script.

### Running

- `runcluster*.sh`
  Execute binary and write logfiles, gets called by `check_cluster*.sh`.

### Evaluation

- `evalcheck_cluster.sh`
  evaluate logfiles from testrun started by `make testcluster`. Calls
  + `evaluate.sh`
  + `evaluate.py`

- `evaluation.py`, `evaluation.sh`
  Simple evaluation script, called by `test.sh`

### helper

- `waitcluster.sh` END
  In order to not overload the cluster, no jobs are submitted if the queue is too full
  instead, this script waits until the queue load falls under a threshold and returns
  for the calling script to continue submitting jobs.

- `make_solu_file.sh`
  Shell script for generating .solu file from a .test file using CPLEX and perplex, calls
  - `make_solu_file.awk`
    Takes two files:  <CPLEX log> <Perplex log> (order is critical!)

## folders and other files

- `CMakeLists.txt` is part of the cmake system, this file's main purpose is to add tests.

- In the `testset/` directory reside the *.test and *.solu files to be specified via `TEST=short`.
  A `.test` file lists problem files, one file per line, absolute and relative paths to its location.
  A `.solu` file with the same basename as the `.test` file contains information about feasibility and best known objective value.
  It is optional for a testrun.

- In the `instances` direcotry are some example instances that can be solved with SCIP.
