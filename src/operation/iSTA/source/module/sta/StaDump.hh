// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file StaDump.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief Dump the Sta data for debug.
 * @version 0.1
 * @date 2021-04-22
 */
#pragma once

#include <yaml-cpp/yaml.h>

#include "StaFunc.hh"

namespace ista {

/**
 * @brief The class for dump sta data in yaml text file for debug.
 *
 */
class StaDumpYaml : public StaFunc {
 public:
  unsigned operator()(StaVertex* the_vertex) override;
  unsigned operator()(StaArc* the_arc) override;
  unsigned operator()(StaGraph* the_graph) override;

  void printText(const char* file_name);

  void set_yaml_file_path(const char* yaml_file_path) {
    _yaml_file_path = yaml_file_path;
  }
  auto& get_yaml_file_path() { return _yaml_file_path; }

 protected:
  YAML::Node _node;
  std::string _yaml_file_path;
};

/**
 * @brief The class for dump delay data in yaml text file for training data.
 *
 */
class StaDumpDelayYaml : public StaDumpYaml {
 public:
  void set_analysis_mode(AnalysisMode analysis_mode) {
    _analysis_mode = analysis_mode;
  }
  AnalysisMode get_analysis_mode() override { return _analysis_mode; }

  void set_trans_type(TransType trans_type) { _trans_type = trans_type; }
  auto get_trans_type() { return _trans_type; }

  unsigned operator()(StaVertex* the_vertex) override;
  unsigned operator()(StaArc* the_arc) override;

 private:
  AnalysisMode _analysis_mode;
  TransType _trans_type;

  unsigned _node_id = 0;
  unsigned _arc_id = 0;
};

/**
 * @brief The class for dump sta data in graphviz format for GUI show.
 *
 */
class StaDumpGraphViz : public StaFunc {
 public:
  unsigned operator()(StaGraph* the_graph) override;
};

/**
 * @brief The class for dump timing data in memory for python call.
 * 
 */
class StaDumpTimingData : public StaFunc {
 public:
  unsigned operator()(StaArc* the_arc) override;

  void set_analysis_mode(AnalysisMode mode) { _analysis_mode = mode; }
  AnalysisMode get_analysis_mode() { return _analysis_mode; }
  void set_trans_type(TransType trans_type) { _trans_type = trans_type; }
  TransType get_trans_type() { return _trans_type; }

  auto get_wire_timing_datas() { return _wire_timing_datas; }

  private:
  std::vector<StaWireTimingData> _wire_timing_datas;

  AnalysisMode _analysis_mode;
  TransType _trans_type;
};

}  // namespace ista
