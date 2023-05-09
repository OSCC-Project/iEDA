// ***************************************************************************************
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
#pragma once

#include <fstream>
#include <iostream>
#include <string>

using std::string;

namespace ino {
using std::endl;
using std::ios;
using std::ofstream;
using std::string;

class Reporter {
 public:
  Reporter() = default;
  Reporter(string path) : _output_path(path) {}
  ~Reporter() = default;

  void reportTime(bool begin);

  void report(string info);

  ofstream &get_ofstream() {
    _outfile.open(_output_path, std::ios::app);
    return _outfile;
  }

 private:
  string   _output_path;
  ofstream _outfile;

  int _check_count = 1;
};
} // namespace ino
