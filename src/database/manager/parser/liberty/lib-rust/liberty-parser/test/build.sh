#!/bin/bash

make test

#g++ -o test test.cpp -lliberty_parser
cp /home/taosimin/iEDA/src/database/manager/parser/liberty/lib-rust/liberty-parser/target/debug/libliberty_parser.so . -f
export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
./test