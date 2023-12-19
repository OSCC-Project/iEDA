#!/bin/bash

# for i in 1 2 3 4 5 6
for i in 1
do
    ./run_case.sh \
    ../../result/input/case1/case1.def \
    ../../result/input/case1/case1.guide \
    ../../result/output/case1/case1.def \
    ../../result/output/case1/case1.guide \
    0.2
    > ./${i}.rpt
done