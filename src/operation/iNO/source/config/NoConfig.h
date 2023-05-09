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
#include <list>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::list;
using std::string;
using std::vector;

namespace ino {
class NoConfig {
 public:
  NoConfig() = default;
  ~NoConfig() = default;

  // setter
  void set_lef_files(vector<string> lefs) { _lef_files_path = lefs; }
  void set_def_file(string def) { _def_file_path = def; }
  void set_design_work_space(string work_space) { _design_work_space = work_space; }
  void set_sdc_file(string sdc) { _sdc_file_path = sdc; }
  void set_lib_files(vector<string> libs) { _lib_files_path = libs; }
  void set_output_def_file(string out) { _out_def_path = out; }
  void set_insert_buffer(string buf) { _insert_buffer = buf; }
  void set_max_fanout(int fanout) { _max_fanout = fanout; }
  void set_report_file(string report) { _report_file = report; }

  // getter
  const vector<string> &get_lef_files() const { return _lef_files_path; }
  const string         &get_def_file() const { return _def_file_path; }
  const string         &get_design_work_space() const { return _design_work_space; }
  const string         &get_sdc_file() const { return _sdc_file_path; }
  const vector<string> &get_lib_files() const { return _lib_files_path; }
  const string         &get_output_def_file() const { return _out_def_path; }
  const string         &get_insert_buffer() const { return _insert_buffer; }
  int                   get_max_fanout() const { return _max_fanout; }
  const string         &get_report_file() const { return _report_file; }

 private:
  // input
  vector<string> _lef_files_path;
  string         _def_file_path;
  string         _design_work_space;
  string         _sdc_file_path;
  vector<string> _lib_files_path;
  string         _report_file;

  string _insert_buffer;
  int    _max_fanout;

  // output
  string _out_def_path;
};

} // namespace ino
