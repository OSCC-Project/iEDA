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
 * @file Evaluator.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <iostream>
#include <vector>

#include "CtsConfig.hh"
#include "CtsDesign.hh"
#include "EvalNet.hh"
#include "GDSPloter.hh"

namespace icts {

struct CellStatsProperty
{
  int total_num;
  double total_area;
  double total_cap;
};

struct TreeNode
{
  std::string name;
  int depth;
  TreeNode* parent;
  std::vector<TreeNode*> children;
};

struct PathInfo
{
  std::string root_name;
  int min_depth;
  int max_depth;
};

class Evaluator
{
 public:
  Evaluator() = default;
  Evaluator(const Evaluator&) = default;
  ~Evaluator() = default;

  void init();
  void evaluate();

  void statistics(const std::string& save_dir);

  void plotPath(const string& inst, const string& file = "debug.gds") const;
  void plotNet(const string& net_name, const string& file = "debug.gds") const;

  void calcInfo();
  std::map<std::string, int> get_cell_dist() const { return _cell_dist_map; }
  std::map<std::string, CellStatsProperty> get_cell_stats() const { return _cell_stats_map; }
  std::vector<PathInfo> get_path_infos() const { return _path_infos; }
  double get_max_net_len() const { return _max_net_len; }
  double get_total_wire_len() const { return _total_wire_len; }

 private:
  void calcWL();
  void calcCellDist();
  void calcCellStats();
  void calcNetLevel();
  void calcPathBufStats();

  void printLog();
  void transferData();
  void initLevel() const;
  void recursiveSetLevel(CtsNet* net) const;
  void pathLevelLog() const;

  bool _have_calc = false;

  double _top_wire_len = 0.0;
  double _trunk_wire_len = 0.0;
  double _leaf_wire_len = 0.0;
  double _total_wire_len = 0.0;
  double _max_net_len = 0.0;
  double _hpwl_top_wire_len = 0.0;
  double _hpwl_trunk_wire_len = 0.0;
  double _hpwl_leaf_wire_len = 0.0;
  double _hpwl_total_wire_len = 0.0;
  double _hpwl_max_net_len = 0.0;
  std::map<std::string, int> _cell_dist_map;
  std::map<std::string, CellStatsProperty> _cell_stats_map;
  std::map<int, int> _net_level_map;
  std::vector<PathInfo> _path_infos;

  std::vector<EvalNet> _eval_nets;
  const int _default_size = 100;
};

}  // namespace icts