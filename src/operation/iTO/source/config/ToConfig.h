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

#include <iostream>
#include <list>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::list;
using std::string;
using std::vector;

namespace ito {
class ToConfig {
 public:
  ToConfig() = default;
  ~ToConfig() = default;

  // setter
  void set_lef_files(const vector<string> lefs) { _lef_files_path = lefs; }
  void set_def_file(const string def) { _def_file_path = def; }
  void set_design_work_space(const string work_space) { _design_work_space = work_space; }
  void set_sdc_file(const string sdc) { _sdc_file_path = sdc; }
  void set_lib_files(const vector<string> libs) { _lib_files_path = libs; }
  void set_output_def_file(const string out) { _out_def_path = out; }
  void set_report_file(const string report) { _report_path = report; }
  void set_gds_file(const string gds) { _gds_path = gds; }

  void set_setup_target_slack(float slack_m) { this->_setup_target_slack = slack_m; }
  void set_hold_slack_margin(float slack_m) { this->_hold_target_slack = slack_m; }
  void set_max_buffer_percent(float percent) { _max_buffer_percent = percent; }
  void set_max_utilization(float util) { _max_utilization = util; }

  void set_fix_fanout(bool bo) { _fix_fanout = bo; }
  void set_optimize_drv(bool bo) { _opti_drv = bo; }
  void set_optimize_hold(bool bo) { _opti_hold = bo; }
  void set_optimize_setup(bool bo) { _opti_setup = bo; }

  void set_drv_insert_buffers(const vector<string> bufs) { _drv_insert_buffers = bufs; }
  void set_setup_insert_buffers(const vector<string> bufs) {
    _setup_insert_buffers = bufs;
  }
  void set_hold_insert_buffers(const vector<string> bufs) { _hold_insert_buffers = bufs; }

  void set_number_passes_allowed_decreasing_slack(int num) {
    _number_iter_allowed_decreasing_slack = num;
  }
  void set_rebuffer_max_fanout(int num) { _rebuffer_max_fanout = num; }
  void set_split_load_min_fanout(int num) { _split_load_min_fanout = num; }

  // getter
  const vector<string> &get_lef_files() const { return _lef_files_path; }
  const string         &get_def_file() const { return _def_file_path; }
  const string         &get_design_work_space() const { return _design_work_space; }
  const string         &get_sdc_file() const { return _sdc_file_path; }
  const vector<string> &get_lib_files() const { return _lib_files_path; }
  const string         &get_output_def_file() const { return _out_def_path; }
  const string         &get_report_file() const { return _report_path; }
  const string         &get_gds_file() const { return _gds_path; }

  float get_setup_target_slack() const { return _setup_target_slack; }
  float get_hold_target_slack() const { return _hold_target_slack; }
  float get_max_buffer_percent() const { return _max_buffer_percent; }
  float get_max_utilization() const { return _max_utilization; }

  bool get_fix_fanout() const { return _fix_fanout; }
  bool get_optimize_drv() const { return _opti_drv; }
  bool get_optimize_hold() const { return _opti_hold; }
  bool get_optimize_setup() const { return _opti_setup; }

  const vector<string> &get_drv_insert_buffers() const { return _drv_insert_buffers; }
  const vector<string> &get_setup_insert_buffers() const { return _setup_insert_buffers; }
  const vector<string> &get_hold_insert_buffers() const { return _hold_insert_buffers; }

  int get_number_passes_allowed_decreasing_slack() {
    return _number_iter_allowed_decreasing_slack;
  }
  int get_rebuffer_max_fanout() { return _rebuffer_max_fanout; }
  int get_split_load_min_fanout() { return _split_load_min_fanout; }

 private:
  // input
  vector<string> _lef_files_path;
  string         _def_file_path;
  string         _design_work_space;
  string         _sdc_file_path;
  vector<string> _lib_files_path;
  float          _setup_target_slack = 0.0;
  float          _hold_target_slack = 0.0;
  float          _max_buffer_percent = 0.2;
  float          _max_utilization = 0.8;

  bool _fix_fanout;
  bool _opti_drv;
  bool _opti_hold;
  bool _opti_setup;

  vector<string> _drv_insert_buffers;   // buffer for optimize Design Rule Violation
  vector<string> _setup_insert_buffers; // buffer for optimize Setup Violation
  vector<string> _hold_insert_buffers;  // buffer for optimize Hold Violation

  // the maximum number of times slack is allowed to get worse when fix setup
  int _number_iter_allowed_decreasing_slack = 50;
  int _rebuffer_max_fanout = 20;
  int _split_load_min_fanout = 8; // Nets with low fanout don't need to split loads.

  // output
  string _out_def_path;
  string _report_path;
  string _gds_path;
};

} // namespace ito
