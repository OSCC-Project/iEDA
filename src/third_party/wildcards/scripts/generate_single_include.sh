#!/bin/sh

scripts/GenSingleHeader.py -Iinclude include/cx.hpp > single_include/cx_tmp.hpp
scripts/generate_header.sh > single_include/cx.hpp
gcc -fpreprocessed -dD -E -P single_include/cx_tmp.hpp >> single_include/cx.hpp 2> /dev/null
rm -f single_include/cx_tmp.hpp

scripts/GenSingleHeader.py -Iinclude include/wildcards.hpp > single_include/wildcards_tmp.hpp
scripts/generate_header.sh > single_include/wildcards.hpp
gcc -fpreprocessed -dD -E -P single_include/wildcards_tmp.hpp >> single_include/wildcards.hpp 2> /dev/null
rm -f single_include/wildcards_tmp.hpp
