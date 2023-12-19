#!/usr/bin/env bash
set -e

./IncTO_2023 -script ./script/contest_script/run_contest.tcl
./IncTO_2023 -script ./script/contest_script/run_evaluation.tcl
