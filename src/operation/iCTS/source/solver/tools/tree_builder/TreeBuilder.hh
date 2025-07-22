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
#include "bound_skew_tree/BoundSkewTree.hh"
namespace icts {
  /**
   * @brief TreeBuilder for Solver
   *       support:
   *          1. build tree:
   *                  1.1 shallow light tree
   *                  1.2 DME tree
   *          2. recover tree:
   *                  2.1 remove root pin
   *          3. place & cancel place buffer for feasible location
   */
  using SteinerTreeFunc = void (*)(const std::string&, Pin*, const std::vector<Pin*>&);
  using SkewTreeFunc
    = Inst * (*) (const std::string&, const std::vector<Pin*>&, const std::optional<double>&, const std::optional<Point>&, const TopoType&);
  class TreeBuilder
  {
  public:
    TreeBuilder() = delete;
    ~TreeBuilder() = default;

    static std::vector<Inst*> getSubInsts(Inst* inst);
    static Inst* genBufInst(const std::string& prefix, const Point& location);
    static void amplifyBufferSize(Inst* inst, const size_t& level = 1);
    static void reduceBufferSize(Inst* inst, const size_t& level = 1);
    static std::vector<std::string> feasibleCell(Inst* inst, const double& skew_bound);

    static void connect(Node* parent, Node* child);
    static void disconnect(Node* parent, Node* child);
    static void directConnectTree(Pin* driver, Pin* load);
    static void fluteTree(const std::string& net_name, Pin* driver, const std::vector<Pin*>& loads);
    static void shallowLightTree(const std::string& net_name, Pin* driver, const std::vector<Pin*>& loads);

    static Inst* boundSkewTree(const std::string& net_name, const std::vector<Pin*>& loads,
      const std::optional<double>& skew_bound = std::nullopt, const std::optional<Point>& guide_loc = std::nullopt,
      const TopoType& topo_type = TopoType::kGreedyDist);
    static Inst* noneEstBoundSkewTree(const std::string& net_name, const std::vector<Pin*>& loads,
      const std::optional<double>& skew_bound = std::nullopt,
      const std::optional<Point>& guide_loc = std::nullopt,
      const TopoType& topo_type = TopoType::kGreedyDist);
    static Inst* fluteBstSaltTree(const std::string& net_name, const std::vector<Pin*>& loads,
      const std::optional<double>& skew_bound = std::nullopt,
      const std::optional<Point>& guide_loc = std::nullopt);
    static Inst* bstSaltTree(const std::string& net_name, const std::vector<Pin*>& loads,
      const std::optional<double>& skew_bound = std::nullopt, const std::optional<Point>& guide_loc = std::nullopt,
      const TopoType& topo_type = TopoType::kGreedyDist);
    static Inst* cbsTree(const std::string& net_name, const std::vector<Pin*>& loads, const std::optional<double>& skew_bound = std::nullopt,
      const std::optional<Point>& guide_loc = std::nullopt, const TopoType& topo_type = TopoType::kGreedyDist);
    static Inst* shiftCBSTree(const std::string& net_name, const std::vector<Pin*>& loads,
      const std::optional<double>& skew_bound = std::nullopt, const std::optional<Point>& guide_loc = std::nullopt,
      const TopoType& topo_type = TopoType::kGreedyDist, const bool& shift = false,
      const std::optional<double>& max_len = std::nullopt);

    static Inst* tempTree(const std::string& net_name, const std::vector<Pin*>& loads, const std::optional<double>& skew_bound = std::nullopt,
      const std::optional<Point>& guide_loc = std::nullopt, const TopoType& topo_type = TopoType::kGreedyDist);

    static Inst* defaultTree(const std::string& net_name, const std::vector<Pin*>& loads, const std::optional<double>& skew_bound = std::nullopt,
      const std::optional<Point>& guide_loc = std::nullopt, const TopoType& topo_type = TopoType::kGreedyDist);

    static void iterativeFixSkew(Net* net, const std::optional<double>& skew_bound = std::nullopt,
      const std::optional<Point>& guide_loc = std::nullopt);

    static void convertToBinaryTree(Node* root);
    static void removeRedundant(Node* root);
    static std::string funcName(const SteinerTreeFunc& func);
    static std::string funcName(const SkewTreeFunc& func);

    static std::vector<SteinerTreeFunc> getSteinerTreeFuncs();
    static std::vector<SkewTreeFunc> getSkewTreeFuncs();

    static void localPlace(Pin* driver_pin, const std::vector<Pin*>& load_pins);
    static void localPlace(std::vector<Pin*>& pins);
    static void localPlace(std::vector<Point>& variable_locs, const std::vector<Point>& fixed_locs);

    static void updateId(Node* root);

    // debug
    static void printGraphviz(Node* root, const std::string& name = "debug");
    static void writePy(Node* root, const std::string& name = "debug");
    static void writeInstInfo(Node* root, const std::string& name = "debug");

  private:
    // function interface to name

    static const std::unordered_map<SteinerTreeFunc, std::string> kSteinterTreeName;

    static const std::unordered_map<SkewTreeFunc, std::string> kSkewTreeName;
  };

}  // namespace icts