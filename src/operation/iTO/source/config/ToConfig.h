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

enum class RoutingType : int { kHVTree = 0, kSteiner = 1, kShallowLight = 2 };

#define toConfig ToConfig::getInstance()
class ToConfig {
 public:
  static ToConfig *getInstance();

  // setter
  void set_lef_files(const vector<string> lefs) { _lef_files_path = lefs; }
  void set_def_file(const string def) { _def_file_path = def; }
  void set_design_work_space(const string work_space) { _design_work_space = work_space; }
  void set_sdc_file(const string sdc) { _sdc_file_path = sdc; }
  void set_lib_files(const vector<string> libs) { _lib_files_path = libs; }
  void set_output_def_file(const string out) { _out_def_path = out; }
  void set_report_file(const string report) { _report_path = report; }
  void set_gds_file(const string gds) { _gds_path = gds; }
  void set_routing_tree(const string tree)
  {
    if (tree == "flute") {
      _routing_tree = RoutingType::kSteiner;
    } else if (tree == "hvtree") {
      _routing_tree = RoutingType::kHVTree;
    } else if (tree == "shallow-light") {
      _routing_tree = RoutingType::kShallowLight;
    }
  }

  void set_setup_target_slack(float slack_m) { this->_setup_target_slack = slack_m; }
  void set_hold_target_slack(float slack_m) { this->_hold_target_slack = slack_m; }
  void set_max_insert_instance_percent(float percent) { _max_insert_instance_percent = percent; }
  void set_max_core_utilization(float util) { _max_core_utilization = util; }

  void set_fix_fanout(bool bo) { _fix_fanout = bo; }
  void set_optimize_drv(bool bo) { _opti_drv = bo; }
  void set_optimize_hold(bool bo) { _opti_hold = bo; }
  void set_optimize_setup(bool bo) { _opti_setup = bo; }

  void set_drv_insert_buffers(const vector<string> bufs) { _drv_insert_buffers = bufs; }
  void set_setup_insert_buffers(const vector<string> bufs) {
    _setup_insert_buffers = bufs;
  }
  void set_hold_insert_buffers(const vector<string> bufs) { _hold_insert_buffers = bufs; }

  void set_number_of_decreasing_slack_iter(int num) {
    _number_iter_allowed_decreasing_slack = num;
  }
  void set_max_allowed_buffering_fanout(int num) { _max_allowed_buffering_fanout = num; }
  void set_min_divide_fanout(int num) { _min_divide_fanout = num; }
  void set_optimize_endpoints_percent(float num) { _optimize_endpoints_percent = num; }
  void set_drv_optimize_iter_number(int num) { _drv_optimize_iter_number = num; }

  void set_drv_buffer_prefix(const string& prefix) { _drv_buffer_prefix = prefix; }
  void set_drv_net_prefix(const string& prefix) { _drv_net_prefix = prefix; }
  void set_hold_buffer_prefix(const string& prefix) { _hold_buffer_prefix = prefix; }
  void set_hold_net_prefix(const string& prefix) { _hold_net_prefix = prefix;}
  void set_setup_buffer_prefix(const string& prefix) { _setup_buffer_prefix = prefix;}
  void set_setup_net_prefix(const string& prefix) { _setup_net_prefix = prefix;}

  // getter
  const vector<string> &get_lef_files() const { return _lef_files_path; }
  const string         &get_def_file() const { return _def_file_path; }
  const string         &get_design_work_space() const { return _design_work_space; }
  const string         &get_sdc_file() const { return _sdc_file_path; }
  const vector<string> &get_lib_files() const { return _lib_files_path; }
  const string         &get_output_def_file() const { return _out_def_path; }
  const string         &get_report_file() const { return _report_path; }
  const string         &get_gds_file() const { return _gds_path; }
  const RoutingType    &get_routing_tree() const { return _routing_tree; }

  float get_setup_target_slack() const { return _setup_target_slack; }
  float get_hold_target_slack() const { return _hold_target_slack; }
  float get_max_insert_instance_percent() const { return _max_insert_instance_percent; }
  float get_max_core_utilization() const { return _max_core_utilization; }

  bool get_fix_fanout() const { return _fix_fanout; }
  bool get_optimize_drv() const { return _opti_drv; }
  bool get_optimize_hold() const { return _opti_hold; }
  bool get_optimize_setup() const { return _opti_setup; }

  const vector<string> &get_drv_insert_buffers() const { return _drv_insert_buffers; }
  const vector<string> &get_setup_insert_buffers() const { return _setup_insert_buffers; }
  const vector<string> &get_hold_insert_buffers() const { return _hold_insert_buffers; }

  int get_number_of_decreasing_slack_iter() {
    return _number_iter_allowed_decreasing_slack;
  }
  int get_max_allowed_buffering_fanout() { return _max_allowed_buffering_fanout; }
  int get_min_divide_fanout() { return _min_divide_fanout; }
  float get_optimize_endpoints_percent() { return _optimize_endpoints_percent; }
  int get_drv_optimize_iter_number() { return _drv_optimize_iter_number; }

  string get_drv_buffer_prefix() const { return _drv_buffer_prefix; }
  string get_drv_net_prefix() const { return _drv_net_prefix; }
  string get_hold_buffer_prefix() const { return _hold_buffer_prefix; }
  string get_hold_net_prefix() const { return _hold_net_prefix;}
  string get_setup_buffer_prefix() const { return _setup_buffer_prefix;}
  string get_setup_net_prefix() const { return _setup_net_prefix;}

 private:
  static ToConfig *_instance;

  // input
  vector<string> _lef_files_path;
  string         _def_file_path;
  string         _design_work_space;
  string         _sdc_file_path;
  vector<string> _lib_files_path;
  float          _setup_target_slack = 0.0;
  float          _hold_target_slack = 0.0;
  float          _max_insert_instance_percent = 0.2;
  float          _max_core_utilization = 0.8;

  bool _fix_fanout;
  bool _opti_drv;
  bool _opti_hold;
  bool _opti_setup;

  RoutingType _routing_tree = RoutingType::kSteiner;
  vector<string> _drv_insert_buffers;   // buffer for optimize Design Rule Violation
  vector<string> _setup_insert_buffers; // buffer for optimize Setup Violation
  vector<string> _hold_insert_buffers;  // buffer for optimize Hold Violation

  // specific names prefixes
  string _drv_buffer_prefix = "DRV_buffer_";
  string _drv_net_prefix = "DRV_net_";
  string _hold_buffer_prefix  = "hold_buffer_";
  string _hold_net_prefix = "hold_net_";
  string _setup_buffer_prefix = "setup_buffer_";
  string _setup_net_prefix = "setup_net_";

  // the maximum number of times slack is allowed to get worse when fix setup
  int _number_iter_allowed_decreasing_slack = 50;
  int _max_allowed_buffering_fanout = 20;
  int _min_divide_fanout = 8; // Nets with low fanout don't need to divide loads.
  float _optimize_endpoints_percent = 1.0; // Nets with low fanout don't need to divide loads.
  int _drv_optimize_iter_number = 1.0; // max iter number for optimize DRV

  // output
  string _out_def_path;
  string _report_path;
  string _gds_path;

  ToConfig() = default;
  ~ToConfig() = default;
};

} // namespace ito
