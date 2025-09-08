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
 * @file PwrDumpGraph.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class for dump the power graph data.
 * @version 0.1
 * @date 2023-04-04
 */

#pragma once

#include <yaml-cpp/yaml.h>

#include "json/json.hpp"

#include <sstream>

#include "core/PwrFunc.hh"
#include "core/PwrGraph.hh"

namespace ipower {

/**
 * @brief dump the power graph information.
 *
 */
class PwrDumpGraphYaml : public PwrFunc {
 public:
  unsigned operator()(PwrVertex* the_vertex) override;
  unsigned operator()(PwrArc* the_arc) override;
  unsigned operator()(PwrGraph* the_graph) override;

  void printText(const char* file_name);

 private:
  YAML::Node _node;
};

/**
 * @brief dump the power graphviz for debug.
 *
 */
class PwrDumpGraphViz : public PwrFunc {
 public:
  unsigned operator()(PwrArc* the_arc) override;

  void printText(const char* file_name);

 private:
  std::stringstream _ss;  //!< for print information to string stream.
};

/**
 * @brief dump the power graph json for power predict.
 * 
 */
class PwrDumpGraphJson : public PwrFunc {
 public:
  using json = nlohmann::ordered_json;
  PwrDumpGraphJson(json& json_file) : _json_file(json_file) {}
  ~PwrDumpGraphJson() override = default;

  unsigned operator()(PwrGraph* the_graph) override;

  json dumpNodeFeature(PwrGraph* the_graph);
  json dumpNodeNetPower(PwrGraph* the_graph);
  json dumpNodeInternalPower(PwrGraph* the_graph);
  json dumpInstInternalPower(PwrGraph* the_graph);

 private:
  json& _json_file;
};

}  // namespace ipower