import os
import argparse


copyright = """// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
"""


def add_copyright(file_path):
    with open(file_path, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(copyright + content)
    f.close()


def get_all_cpp_files(dir_path, exclude_folder='', log_path=''):
    if log_path == None or log_path == '':
        log_path = os.path.join(dir_path, 'log.txt')
    files = os.listdir(dir_path)
    for file in files:
        if file == exclude_folder:
            continue
        full_path = os.path.join(dir_path, file)
        if os.path.isdir(full_path):
            get_all_cpp_files(full_path, exclude_folder, log_path)
        elif file.endswith('.cpp') or file.endswith('.cc') or file.endswith('.h') or file.endswith('.hh') or file.endswith('.hpp'):
            add_copyright(full_path)
        else:
            print(log_path)
            with open(log_path, 'a') as f:
                f.write(full_path + '\n')
            f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path')
    parser.add_argument('-e', '--exclude_folder')
    parser.add_argument('-l', '--log_path')
    args = parser.parse_args()
    get_all_cpp_files(args.path, args.exclude_folder, args.log_path)
