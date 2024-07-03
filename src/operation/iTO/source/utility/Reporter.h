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
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ctime"
#include "ids.hpp"

using std::string;
using std::vector;

namespace ista {
class Net;
}

namespace ito {
using std::endl;
using std::ios;
using std::ofstream;
using std::setw;
using std::string;
using std::vector;

#define toRptInst Reporter::get_instance()

class Reporter
{
 public:
  static Reporter* get_instance();
  static void destroy_instance();

  void init(std::string path) { _output_path = path; }

  void reportTime(bool begin);

  void reportDRVResult(int slew_violations, int cap_violations, bool before);
  void reportSetupResult(std::vector<double> slack_store);
  void reportHoldResult(std::vector<double> timing_slacks_hold, std::vector<int> hold_vio_num, std::vector<int> insert_buf_num, double slack,
                        int insert_buf);

  void reportNetInfo(ista::Net* net, double cap_load_allowed_max);

  void report(const string info);

  std::ofstream& get_ofstream()
  {
    if (!_outfile.is_open()) {
      _outfile.open(_output_path, std::ios::app);
    }
    return _outfile;
  }

 private:
  static Reporter* _instance;

  std::string _output_path;
  std::ofstream _outfile;

  int _check_count = 1;

  Reporter() = default;
  ~Reporter() = default;
};
}  // namespace ito
