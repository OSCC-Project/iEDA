#!/bin/python3
import os
import re
import sys
import argparse  # argument parsing
import time

#===========================================================
##   run contest script
#===========================================================
os.system('./iEDA_contest -script ./script/contest_script/run_contest.tcl')

#===========================================================
##   run evaluation script
#===========================================================
os.system('./iEDA_contest -script ./script/contest_script/run_evaluation.tcl')
