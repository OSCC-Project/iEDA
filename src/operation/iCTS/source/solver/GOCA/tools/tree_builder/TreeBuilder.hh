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
 * @file TreeBuilder.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <string>
#include <vector>

#include "Inst.hh"
#include "Pin.hh"
namespace icts {
/**
 * @brief TreeBuilder for GOCA
 *       support:
 *          1. build tree:
 *                  1.1 shallow light tree
 *                  1.2 DME tree
 *          2. recover tree:
 *                  2.1 remove root pin
 *          3. place & cancel place buffer for feasible location
 */
class TreeBuilder
{
 public:
  TreeBuilder() = delete;
  ~TreeBuilder() = default;

  static std::vector<Inst*> getSubInsts(Inst* inst);
  static Inst* genBufInst(const std::string& prefix, const Point& location);
  static Inst* toBufInst(const std::string& prefix, Node* driver_node);
  static void amplifyBufferSize(Inst* inst, const size_t& level = 1);
  static void reduceBufferSize(Inst* inst, const size_t& level = 1);
  static std::vector<std::string> feasibleCell(Inst* inst, const double& skew_bound);

  static void connect(Node* parent, Node* child);
  static void disconnect(Node* parent, Node* child);
  static void directConnectTree(Pin* driver, Pin* load);
  static void shallowLightTree(Pin* driver, const std::vector<Pin*>& loads);
  static std::vector<Inst*> dmeTree(const std::string& net_name, const std::vector<Pin*>& loads,
                                    const std::optional<double>& skew_bound = std::nullopt,
                                    const std::optional<Point>& guide_loc = std::nullopt);
  static void recoverNet(Net* net);

  static void place(Inst* inst);
  static void cancelPlace(Inst* inst);

  // debug
  static void printGraphviz(Node* root, const std::string& name = "debug");
  static void writePy(Node* root, const std::string& name = "debug");
};
}  // namespace icts