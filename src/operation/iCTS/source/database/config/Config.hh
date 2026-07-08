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
 * @file Config.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-11
 * @brief Configuration for CTS
 */

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "LogFormat.hh"

namespace icts {

class SchemaWriter;

class Config
{
 public:
  Config() = default;
  ~Config() = default;

  Config(const Config& rhs) = default;
  Config(Config&& rhs) = default;
  auto operator=(const Config& rhs) -> Config& = default;
  auto operator=(Config&& rhs) -> Config& = default;

  // Initialize from config file
  auto init(const std::string& config_file) -> bool;

  // Reset to default values
  auto reset() -> void
  {
    _skew_bound = 0.04;
    _max_buf_tran = 1.5;
    _root_input_slew = 0.0;
    _max_sink_tran = 1.5;
    _max_cap = 1.5;
    _has_max_buf_tran = false;
    _has_max_cap = false;
    _max_fanout = 32;
    // Kept as a config placeholder for compatibility with existing JSONs.
    _max_length = 300;
    // Active wirelength lattice controls used by characterization.
    _wirelength_unit_um = 0.0;
    _wirelength_iterations = 3;
    _routing_layers.clear();
    _buffer_types.clear();
    _slew_steps = 15;
    _cap_steps = 15;
    _wire_width = 0.0;
    _char_buf_redundancy_pct = 0.0;
    _force_branch_buffer = false;
    _htree_depth_explore_window = 4;
    _htree_topology_tolerance = 0.1;
    _enable_analytical_htree = false;
    _enable_sink_clustering = true;
    _work_dir = "./result/cts";
    _log_file = "./result/cts/cts.log";
    _visualization_dir = "./result/cts/visualization";
    _statistics_dir = "./result/cts/statistics";
    _last_error.clear();
    _warnings.clear();
  }

  // algorithm
  auto get_skew_bound() const -> double { return _skew_bound; }
  auto get_max_buf_tran() const -> double { return _max_buf_tran; }
  auto get_root_input_slew() const -> double { return _root_input_slew; }
  auto get_max_sink_tran() const -> double { return _max_sink_tran; }
  auto get_max_cap() const -> double { return _max_cap; }
  auto has_max_buf_tran() const -> bool { return _has_max_buf_tran; }
  auto has_max_cap() const -> bool { return _has_max_cap; }
  auto get_max_length() const -> double { return _max_length; }
  auto get_wirelength_unit_um() const -> double { return _wirelength_unit_um; }
  auto get_wirelength_iterations() const -> unsigned { return _wirelength_iterations; }
  auto get_slew_steps() const -> unsigned { return _slew_steps; }
  auto get_cap_steps() const -> unsigned { return _cap_steps; }
  auto get_wire_width() const -> double { return _wire_width; }
  auto get_max_fanout() const -> unsigned { return _max_fanout; }
  auto get_routing_layers() const -> const std::vector<unsigned>& { return _routing_layers; }
  auto get_buffer_types() const -> const std::vector<std::string>& { return _buffer_types; }
  auto get_char_buf_redundancy_pct() const -> double { return _char_buf_redundancy_pct; }
  auto is_force_branch_buffer() const -> bool { return _force_branch_buffer; }
  auto get_htree_depth_explore_window() const -> unsigned { return _htree_depth_explore_window; }
  auto get_htree_topology_tolerance() const -> double { return _htree_topology_tolerance; }
  auto is_enable_analytical_htree() const -> bool { return _enable_analytical_htree; }
  auto is_enable_sink_clustering() const -> bool { return _enable_sink_clustering; }

  // file
  auto get_work_dir() const -> const std::string& { return _work_dir; }
  auto get_log_file() const -> const std::string& { return _log_file; }
  auto get_visualization_dir() const -> const std::string& { return _visualization_dir; }
  auto get_statistics_dir() const -> const std::string& { return _statistics_dir; }
  auto get_last_error() const -> const std::string& { return _last_error; }
  auto get_warnings() const -> const std::vector<std::string>& { return _warnings; }
  auto set_last_error(const std::string& error) -> void { _last_error = error; }

  // algorithm
  auto set_skew_bound(double skew_bound) -> void { _skew_bound = skew_bound; }
  auto set_max_buf_tran(double max_buf_tran) -> void
  {
    _max_buf_tran = max_buf_tran;
    _has_max_buf_tran = true;
  }
  auto set_root_input_slew(double root_input_slew) -> void { _root_input_slew = std::max(0.0, root_input_slew); }
  auto set_max_sink_tran(double max_sink_tran) -> void { _max_sink_tran = max_sink_tran; }
  auto set_max_cap(double max_cap) -> void
  {
    _max_cap = max_cap;
    _has_max_cap = true;
  }
  auto set_max_length(double max_length) -> void { _max_length = max_length; }
  auto set_wirelength_unit_um(double wirelength_unit_um) -> void { _wirelength_unit_um = wirelength_unit_um; }
  auto set_wirelength_iterations(unsigned wirelength_iterations) -> void { _wirelength_iterations = wirelength_iterations; }
  auto set_slew_steps(unsigned steps) -> void { _slew_steps = steps; }
  auto set_cap_steps(unsigned steps) -> void { _cap_steps = steps; }
  auto set_wire_width(double wire_width) -> void { _wire_width = wire_width; }
  auto set_max_fanout(unsigned max_fanout) -> void { _max_fanout = max_fanout; }
  auto set_routing_layers(const std::vector<unsigned>& routing_layers) -> void { _routing_layers = routing_layers; }
  auto set_buffer_types(const std::vector<std::string>& types) -> void { _buffer_types = types; }
  auto set_char_buf_redundancy_pct(double pct) -> void { _char_buf_redundancy_pct = pct; }
  auto set_force_branch_buffer(bool force_branch_buffer) -> void { _force_branch_buffer = force_branch_buffer; }
  auto set_htree_depth_explore_window(unsigned window) -> void { _htree_depth_explore_window = std::max(1U, window); }
  auto set_htree_topology_tolerance(double tolerance) -> void { _htree_topology_tolerance = std::max(0.0, tolerance); }
  auto set_enable_analytical_htree(bool enable_analytical_htree) -> void { _enable_analytical_htree = enable_analytical_htree; }
  auto set_enable_sink_clustering(bool enable_sink_clustering) -> void { _enable_sink_clustering = enable_sink_clustering; }

  // file
  auto set_work_dir(const std::string& work_dir) -> void { _work_dir = work_dir; }
  auto set_log_file(const std::string& file) -> void { _log_file = file; }
  auto set_visualization_dir(const std::string& dir) -> void { _visualization_dir = dir; }
  auto set_statistics_dir(const std::string& dir) -> void { _statistics_dir = dir; }

  // parse from json file
  auto parse(const std::string& json_file) -> bool;
  auto emitRuntimeConfigReport(SchemaWriter& reporter, const std::string& title) const -> void;

  auto buildRuntimeConfigRows() const -> logformat::TableRows;

 private:
  // algorithm
  double _skew_bound = 0.0;
  double _max_buf_tran = 0.0;
  double _root_input_slew = 0.0;
  double _max_sink_tran = 0.0;
  double _max_cap = 0.0;
  bool _has_max_buf_tran = false;
  bool _has_max_cap = false;
  double _max_length = 0.0;             // Placeholder knob (not step-based slicing).
  double _wirelength_unit_um = 0.0;     // Active base unit for wirelength lattice.
  unsigned _wirelength_iterations = 3;  // Active iteration count for wirelength lattice.
  unsigned _slew_steps = 15;
  unsigned _cap_steps = 15;
  double _wire_width = 0.0;
  unsigned _max_fanout = 32;
  std::vector<unsigned> _routing_layers;
  std::vector<std::string> _buffer_types;
  double _char_buf_redundancy_pct = 0.0;
  bool _force_branch_buffer = false;
  unsigned _htree_depth_explore_window = 4;
  double _htree_topology_tolerance = 0.1;
  bool _enable_analytical_htree = false;
  bool _enable_sink_clustering = true;

  // file
  std::string _work_dir = "./result/cts";
  std::string _log_file = "./result/cts/cts.log";
  std::string _visualization_dir = "./result/cts/visualization";
  std::string _statistics_dir = "./result/cts/statistics";

  std::string _last_error;
  std::vector<std::string> _warnings;
};

}  // namespace icts
