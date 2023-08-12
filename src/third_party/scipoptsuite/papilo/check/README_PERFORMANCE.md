# How to run performance tests

In PaPILO you can run your performance tests locally or remote via Jenkins job
(https://cijenkins.zib.de/job/papilo_performance_run/). Locally you should run with `make test`, while Jenkins
tiggers `make testcluster`.

- `make testcluster` starts `check_cluster`, which send jobs on the slurm machine and upload the results on
  Rubberband. (There are special evaluation files and a special PaPILO solver interface).
- `make test` can do the same steps on your machine besides the upload. The steps are explained in the following

# How to install & execute locally

Required:

- [SCIP Repository](https://git.zib.de/integer/scip) is checked out and built
- [SOPLEX Repository](https://git.zib.de/integer/soplex) is checked out and built

Replace in the following `PATH-TO-XYZ` with the path to your local checked-out repository.

Generate build folder and build the project with Cmake (link SCIP & SOPLEX): (not necessary if you want only presolve)

```
  mkdir build
  cd build
  cmake -DSCIP_DIR=PATH-TO-SCIP -j3 ..
  cd ..
```

Run local test in the PaPILO directory:

```
  make test EXECUTABLE=build/bin/papilo TEST=short
```

# How to use IPET with PaPILO:

Required:

- Python 3
- clone the following repository [IPET](https://github.com/GregorCH/ipet)
- executing papilo locally should have generated `PATH_TO_OUT`=`check/results/check.*.default.out`
  and `check/results/check.*.default.err`
- access to Rubberband (where you can download the evalution files)

Steps:

- read the Readme and setup an Virtual environment, in example:
  ```
    cd ipet
    virtualenv -p python3 venv
    source venv/bin/activate
    pip3 install -r requirements.txt
  ```
- create the folder `.ipet/solver` in your home directory
- copy the file `check/ipet/PapiloSolver.py` to `~/.ipet/solvers`
- create the file `__init__.py` in `~/.ipet/solvers` with following content:

```
__all__ = []
from .PapiloSolver import PaPILOSolver
```
- generate log files by executing `make test` in the PaPILO directory.
- the log files are then in the folder `check/results`
- parse these file(s) with the following command: `ipet-parse -l check/results/filename.{err,out}`
- parsing should generate an testrunfile `check/results/filename.trn`
- choose your eval file in the check/ipet/ folder (f.e. `check/ipet/papilo_evaluation.xml`) or download it from Rubberband
- replace `RubberbandId` in the evaluation file with `LogFileName` (if not already happened)
- evaluate the testrunfile: `ipet-evaluate -t check/results/filename.trn -l -e check/ipet/papilo_evaluation.xml`
- exit virtual environment with `deactivate`
