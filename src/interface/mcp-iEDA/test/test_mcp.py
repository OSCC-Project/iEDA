#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File : test_mcp.py
@Time : 2025/07/24 12:44:42
@Author : simin tao
@Version : 1.0
@Contact : taosm@pcl.ac.cn
@Desc : test mcp ieda.
'''


import unittest
import os
import sys

current_dir = os.path.split(os.path.abspath(__file__))[0]
root_dir = current_dir.rsplit("/", 1)[0]

sys.path.append(root_dir)

os.environ["iEDA"] = "/home/taosimin/iEDA24/iEDA/scripts/design/sky130_gcd/iEDA"
os.environ["WORKSPACE"] = "/home/taosimin/iEDA24/iEDA/scripts/design/sky130_gcd"

from src.mcp_ieda import get_ieda_path
from src.mcp_ieda.server import run_ieda


class TestRuniEDA(unittest.TestCase):
    def test_run(self):
        iEDA = get_ieda_path()
        run_ieda(iEDA, current_dir + "../../example/gcd/run_iEDA.tcl")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRuniEDA)
    unittest.TextTestRunner(verbosity=2).run(suite)