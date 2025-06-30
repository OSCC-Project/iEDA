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
 * @file PwrDumpSeqGraph.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Dump the seq graph for debug.
 * @version 0.1
 * @date 2023-03-06
 */

#pragma once

#include <yaml-cpp/yaml.h>

#include "core/PwrFunc.hh"
#include "core/PwrSeqGraph.hh"

namespace ipower {
/**
 * @brief The class for dump seq graph in yaml text file for debug.
 *
 */
class PwrDumpSeqYaml : public PwrFunc {
 public:
  unsigned operator()(PwrSeqVertex* the_vertex) override;
  // unsigned operator()(PwrSeqArc* the_arc) override;
  // unsigned operator()(PwrSeqGraph* the_graph) override;

  void printText(const char* file_name);

 private:
  YAML::Node _node;
};

/**
 * @brief The class for dump seq graph in graphviz format for GUI show.
 *
 */
class PwrDumpSeqGraphViz : public PwrFunc {
 public:
  unsigned operator()(PwrSeqGraph* the_graph) override;
};
}  // namespace ipower