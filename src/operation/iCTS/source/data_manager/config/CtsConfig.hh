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
/**
 * @file CtsConfig.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <string>
#include <vector>

namespace icts {
struct Assign
{
  double max_net_len;  // max wire length of a cluster
  int max_fanout;      // max fanout of a cluster
  double max_cap;      // max cap of a cluster
  double skew_bound;
  double ratio;  // clustering margin
};
/**
 * @brief RC Pattern definition
 *
 */
enum class RCPattern
{
  kSingle,  // single layer
  kHV,      // first horizontal then vertical
  kVH,      // first vertical then horizontal
};
/**
 * @brief Layer Pattern definition
 *
 */
enum class LayerPattern
{
  kH,  // horizontal
  kV,  // vertical
  kNone,
};

class CtsConfig
{
 public:
  CtsConfig() {}
  CtsConfig(const CtsConfig& rhs) = default;
  ~CtsConfig() = default;
  // algorithm
  bool get_use_skew_tree_alg() const { return _use_skew_tree_alg; }
  double get_skew_bound() const { return _skew_bound; }
  double get_max_buf_tran() const { return _max_buf_tran; }
  double get_max_sink_tran() const { return _max_sink_tran; }
  double get_max_cap() const { return _max_cap; }
  int get_max_fanout() const { return _max_fanout; }
  double get_min_length() const { return _min_length; }
  double get_max_length() const { return _max_length; }
  const int& get_h_layer() const { return _h_layer; }
  const int& get_v_layer() const { return _v_layer; }
  std::vector<int> get_routing_layers() const { return _routing_layers; }
  std::vector<std::string> get_buffer_types() const { return _buffer_types; }
  std::string get_root_buffer_type() const { return _root_buffer_type; }
  bool is_root_buffer_required() const { return _root_buffer_required; }
  bool is_inherit_root() const { return _inherit_root; }
  bool is_break_long_wire() const { return _break_long_wire; }
  // level constraint
  std::vector<double> get_level_max_length() const { return _level_max_length; }
  std::vector<int> get_level_max_fanout() const { return _level_max_fanout; }
  std::vector<double> get_level_max_cap() const { return _level_max_cap; }
  std::vector<double> get_level_skew_bound() const { return _level_skew_bound; }
  std::vector<double> get_level_cluster_ratio() const { return _level_cluster_ratio; }
  const int& get_shift_level() const { return _shift_level; }
  const int& get_latency_opt_level() const { return _latency_opt_level; }
  const double& get_global_latency_opt_ratio() const { return _global_latency_opt_ratio; }
  const double& get_local_latency_opt_ratio() const { return _local_latency_opt_ratio; }
  // file
  const std::string& get_work_dir() const { return _work_dir; }
  const std::string& get_output_def_path() const { return _output_def_path; }
  const std::string& get_log_file() const { return _log_file; }
  const std::string& get_gds_file() const { return _gds_file; }
  const std::string& get_use_netlist_string() const { return _gds_file; }
  bool is_use_netlist() { return _use_netlist == "ON" ? true : false; }
  const std::vector<std::pair<std::string, std::string>> get_clock_netlist() const { return _net_list; }

  // algorithm
  void set_use_skew_tree_alg(const bool& use_skew_tree_alg) { _use_skew_tree_alg = use_skew_tree_alg; }
  void set_skew_bound(double skew_bound) { _skew_bound = skew_bound; }
  void set_max_buf_tran(double max_buf_tran) { _max_buf_tran = max_buf_tran; }
  void set_max_sink_tran(double max_sink_tran) { _max_sink_tran = max_sink_tran; }
  void set_max_cap(double max_cap) { _max_cap = max_cap; }
  void set_max_fanout(int max_fanout) { _max_fanout = max_fanout; }
  void set_min_length(double min_length) { _min_length = min_length; }
  void set_max_length(double max_length) { _max_length = max_length; }
  void set_h_layer(const int& h_layer) { _h_layer = h_layer; }
  void set_v_layer(const int& v_layer) { _v_layer = v_layer; }
  void set_routing_layers(const std::vector<int>& routing_layers) { _routing_layers = routing_layers; }
  void set_buffer_types(const std::vector<std::string>& types) { _buffer_types = types; }
  void set_root_buffer_type(const std::string& type) { _root_buffer_type = type; }
  void set_root_buffer_required(const bool& required) { _root_buffer_required = required; }
  void set_inherit_root(const bool& inherit_root) { _inherit_root = inherit_root; }
  void set_break_long_wire(const bool& break_long_wire) { _break_long_wire = break_long_wire; }
  // level constraint
  void set_level_max_length(const std::vector<double>& level_max_length) { _level_max_length = level_max_length; }
  void set_level_max_fanout(const std::vector<int>& level_max_fanout) { _level_max_fanout = level_max_fanout; }
  void set_level_max_cap(const std::vector<double>& level_max_cap) { _level_max_cap = level_max_cap; }
  void set_level_skew_bound(const std::vector<double>& level_skew_bound) { _level_skew_bound = level_skew_bound; }
  void set_level_cluster_ratio(const std::vector<double>& level_cluster_ratio) { _level_cluster_ratio = level_cluster_ratio; }
  void set_shift_level(const int& shift_level) { _shift_level = shift_level; }
  void set_latency_opt_level(const int& latency_opt_level) { _latency_opt_level = latency_opt_level; }
  void set_global_latency_opt_ratio(const double& global_latency_opt_ratio) { _global_latency_opt_ratio = global_latency_opt_ratio; }
  void set_local_latency_opt_ratio(const double& local_latency_opt_ratio) { _local_latency_opt_ratio = local_latency_opt_ratio; }

  // file
  void set_work_dir(const std::string& work_dir) { _work_dir = work_dir; }
  void set_output_def_path(const std::string& output_def_path) { _output_def_path = output_def_path; }
  void set_log_file(const std::string& file) { _log_file = file; }
  void set_gds_file(const std::string& file) { _gds_file = file; }
  void set_use_netlist(const std::string& use_netlist) { _use_netlist = use_netlist; }
  void set_netlist(const std::vector<std::pair<std::string, std::string>>& net_list) { _net_list = net_list; }

  // query
  Assign query_assign(const int& level) const
  {
    Assign assign;
    assign.max_net_len = query_max_length(level);
    assign.max_fanout = query_max_fanout(level);
    assign.max_cap = query_max_cap(level);
    assign.skew_bound = query_skew_bound(level);
    assign.ratio = query_cluster_ratio(level);
    return assign;
  }

  double query_max_length(const int& level) const
  {
    if (level <= 0 || _level_max_length.empty()) {
      return _max_length;
    }
    int n = _level_max_length.size();
    if (level - 1 >= n) {
      return _level_max_length.back();
    }
    return _level_max_length[level - 1];
  }
  int query_max_fanout(const int& level) const
  {
    if (level <= 0 || _level_max_fanout.empty()) {
      return _max_fanout;
    }
    int n = _level_max_fanout.size();
    if (level - 1 >= n) {
      return _level_max_fanout.back();
    }
    return _level_max_fanout[level - 1];
  }
  double query_max_cap(const int& level) const
  {
    if (level <= 0 || _level_max_cap.empty()) {
      return _max_cap;
    }
    int n = _level_max_cap.size();
    if (level - 1 >= n) {
      return _level_max_cap.back();
    }
    return _level_max_cap[level - 1];
  }

  double query_skew_bound(const int& level) const
  {
    if (level <= 0 || _level_skew_bound.empty()) {
      return _skew_bound;
    }
    int n = _level_skew_bound.size();
    if (level - 1 >= n) {
      return _level_skew_bound.back();
    }
    return _level_skew_bound[level - 1];
  }

  double query_cluster_ratio(const int& level) const
  {
    if (level <= 0 || _level_cluster_ratio.empty()) {
      return 0.9;
    }
    int n = _level_cluster_ratio.size();
    if (level - 1 >= n) {
      return _level_cluster_ratio.back();
    }
    return _level_cluster_ratio[level - 1];
  }

 private:
  // algorithm
  bool _use_skew_tree_alg = false;
  double _skew_bound = 0.04;
  double _max_buf_tran = 1.5;
  double _max_sink_tran = 1.5;
  double _max_cap = 1.5;
  int _max_fanout = 32;
  double _min_length = 50;
  double _max_length = 300;
  int _h_layer = 1;
  int _v_layer = 1;
  std::vector<int> _routing_layers;
  std::vector<std::string> _buffer_types;
  std::string _root_buffer_type;
  bool _root_buffer_required = false;
  bool _inherit_root = false;
  bool _break_long_wire = false;
  // level constraint
  std::vector<double> _level_max_length;
  std::vector<int> _level_max_fanout;
  std::vector<double> _level_max_cap;
  std::vector<double> _level_skew_bound;
  std::vector<double> _level_cluster_ratio;
  int _shift_level = 1;
  int _latency_opt_level = 1;
  double _global_latency_opt_ratio = 0.3;
  double _local_latency_opt_ratio = 0.4;

  // file
  std::string _work_dir = "./result/cts";
  std::string _output_def_path = "./result/cts/output";
  std::string _log_file = "./result/cts/cts.log";
  std::string _gds_file = "./result/cts/output/cts.gds";

  std::string _use_netlist = "OFF";
  std::vector<std::pair<std::string, std::string>> _net_list;
};
}  // namespace icts