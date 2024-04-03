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
 * @file StaBuildRCTree.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of build rc tree.
 * @version 0.1
 * @date 2021-04-14
 */
#pragma once

#include <yaml-cpp/yaml.h>

#include <string>

#include "StaFunc.hh"

class RustSpefNet;

namespace ista {

/**
 * @brief The functor of build rc tree.
 *
 */
class StaBuildRCTree : public StaFunc {
 public:
  StaBuildRCTree() = default;
  StaBuildRCTree(std::string&& spef_file_name, DelayCalcMethod calc_method);
  ~StaBuildRCTree() override = default;

  unsigned operator()(StaGraph* the_graph) override;

  std::unique_ptr<RcNet> createRcNet(Net* net);
  DelayCalcMethod get_calc_method() { return _calc_method; }

  void printYaml(RustSpefNet& spef_net);
  void printYamlText(const char* file_name);

 private:
  std::string _spef_file_name;
  DelayCalcMethod _calc_method =
      DelayCalcMethod::kElmore;  //!< The delay calc method selected.

  YAML::Node _top_node;  //!< Dump yaml node.
};

}  // namespace ista
