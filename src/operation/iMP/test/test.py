import sys

sys.path.append('/home/huangfuxing/Prog_cpp/iEDA/build/src/operation/iMP/ops/IO')
import py_dm


if __name__ == "__main__":
    a = 'dfa'
    py_dm.wrapperIdb(a)
    # print(a)