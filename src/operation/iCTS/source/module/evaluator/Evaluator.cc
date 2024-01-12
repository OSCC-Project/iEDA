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
 * @file Evaluator.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "Evaluator.hh"

#include <fstream>

#include "CTSAPI.hh"
#include "Net.hh"
#include "report/CtsReport.hh"
namespace icts {

void Evaluator::init()
{
  CTSAPIInst.setPropagateClock();
  printLog();
  initLevel();
  transferData();
}

void Evaluator::printLog()
{
  LOG_INFO << "\033[1;31m";
  LOG_INFO << R"(                  _             _              )";
  LOG_INFO << R"(                 | |           | |             )";
  LOG_INFO << R"(   _____   ____ _| |_   _  __ _| |_ ___  _ __  )";
  LOG_INFO << R"(  / _ \ \ / / _` | | | | |/ _` | __/ _ \| '__| )";
  LOG_INFO << R"( |  __/\ V / (_| | | |_| | (_| | || (_) | |    )";
  LOG_INFO << R"(  \___| \_/ \__,_|_|\__,_|\__,_|\__\___/|_|    )";
  LOG_INFO << "\033[0m";
  LOG_INFO << "Enter evaluator!";
}

void Evaluator::transferData()
{
  _eval_nets.clear();
  auto* design = CTSAPIInst.get_design();
  auto& clk_nets = design->get_nets();
  for (auto* clk_net : clk_nets) {
    _eval_nets.emplace_back(EvalNet(clk_net));
  }
}

void Evaluator::initLevel() const
{
  auto* design = CTSAPIInst.get_design();
  auto& clk_nets = design->get_nets();
  for (auto clk_net : clk_nets) {
    if (!clk_net->is_newly()) {
      continue;
    }
    recursiveSetLevel(clk_net);
  }
}

void Evaluator::recursiveSetLevel(CtsNet* net) const
{
  if (net == nullptr) {
    return;
  }
  auto* driver = net->get_driver_inst();
  if (driver->get_level() > 0) {
    return;
  }

  auto* design = CTSAPIInst.get_design();
  auto loads = net->get_load_insts();
  int max_level = 0;
  for (auto load : loads) {
    if (load->get_type() == CtsInstanceType::kSink) {
      load->set_level(1);
      max_level = std::max(1, max_level);
      continue;
    }
    auto sub_net_name = load->get_name().substr(0, load->get_name().length() - 4);
    auto* sub_net = design->findNet(sub_net_name);
    recursiveSetLevel(sub_net);

    max_level = std::max(load->get_level(), max_level);
  }

  driver->set_level(max_level + 1);
}

std::pair<size_t, size_t> Evaluator::getPathLevel() const
{
  struct TreeNode
  {
    std::string name;
    int depth;
    TreeNode* parent;
    std::vector<TreeNode*> children;
  };

  std::unordered_map<std::string, TreeNode*> name_to_node;
  auto gen_node = [&](CtsInstance* inst) {
    if (name_to_node.count(inst->get_name()) == 0) {
      name_to_node[inst->get_name()] = new TreeNode{inst->get_name(), 0, {}};
    }
    return name_to_node[inst->get_name()];
  };

  auto* design = CTSAPIInst.get_design();
  auto& clk_nets = design->get_nets();
  for (auto clk_net : clk_nets) {
    auto* driver = clk_net->get_driver_inst();
    auto* driver_node = gen_node(driver);
    auto loads = clk_net->get_load_insts();
    for (auto load : loads) {
      auto* load_node = gen_node(load);
      driver_node->children.emplace_back(load_node);
      load_node->parent = driver_node;
    }
  }
  // find root
  TreeNode* root = nullptr;
  for (auto [_, node] : name_to_node) {
    if (node->parent == nullptr) {
      root = node;
      break;
    }
  }
  // set depth
  root->depth = 0;
  std::function<void(TreeNode*)> set_depth = [&](TreeNode* node) {
    for (auto child : node->children) {
      child->depth = node->depth + 1;
      set_depth(child);
    }
  };
  set_depth(root);
  // find min and max depth of leaf
  int min_depth = std::numeric_limits<int>::max();
  int max_depth = 0;
  std::function<void(TreeNode*)> find_depth = [&](TreeNode* node) {
    if (node->children.empty()) {
      min_depth = std::min(min_depth, node->depth);
      max_depth = std::max(max_depth, node->depth);
    } else {
      for (auto child : node->children) {
        find_depth(child);
      }
    }
  };
  find_depth(root);
  return {min_depth, max_depth};
}

void Evaluator::evaluate()
{
  CTSAPIInst.refresh();
  for (auto eval_net : _eval_nets) {
    CTSAPIInst.buildRCTree(eval_net);
  }
  CTSAPIInst.reportTiming();
}

void Evaluator::statistics(const std::string& save_dir) const
{
  auto* config = CTSAPIInst.get_config();
  auto dir = (save_dir == "" ? config->get_sta_workspace() : save_dir) + "/statistics";
  // wirelength statistics(type: total, top, trunk, leaf, total certer dist,
  // max)
  auto wl_rpt = CtsReportTable::createReportTable("Wire length stats", CtsReportType::kWireLength);
  auto hpwl_wl_rpt = CtsReportTable::createReportTable("HPWL Wire length stats", CtsReportType::kHpWireLength);
  std::map<std::string, int> cell_count_map;
  double top_wire_len = 0.0;
  double trunk_wire_len = 0.0;
  double leaf_wire_len = 0.0;
  double total_wire_len = 0.0;
  double max_net_len = 0.0;
  double hpwl_top_wire_len = 0.0;
  double hpwl_trunk_wire_len = 0.0;
  double hpwl_leaf_wire_len = 0.0;
  double hpwl_total_wire_len = 0.0;
  double hpwl_max_net_len = 0.0;
  for (const auto& eval_net : _eval_nets) {
    auto* design = CTSAPIInst.get_design();
    // wire length
    auto* net = design->findSolverNet(eval_net.get_name());
    if (!net) {
      continue;
    }
    auto* driver_pin = net->get_driver_pin();
    double net_len = driver_pin->get_sub_len();

    double hpwl_net_len = eval_net.getHPWL();
    auto type = eval_net.netType();
    switch (type) {
      case NetType::kTop:
        top_wire_len += net_len;
        hpwl_top_wire_len += hpwl_net_len;
        break;
      case NetType::kTrunk:
        trunk_wire_len += net_len;
        hpwl_trunk_wire_len += hpwl_net_len;
        break;
      case NetType::kLeaf:
        leaf_wire_len += net_len;
        hpwl_leaf_wire_len += hpwl_net_len;
        break;
      default:
        break;
    }
    total_wire_len += net_len;
    hpwl_total_wire_len += hpwl_net_len;
    max_net_len = std::max(max_net_len, net_len);
    hpwl_max_net_len = std::max(hpwl_max_net_len, hpwl_net_len);
    if (eval_net.is_newly()) {
      // cell count
      auto cell_master = eval_net.get_driver()->get_cell_master();
      if (cell_count_map.count(cell_master) == 0) {
        cell_count_map[cell_master] = 1;
      } else {
        cell_count_map[cell_master]++;
      }
    }
  }
  (*wl_rpt) << "Top" << Str::printf("%.3f", top_wire_len) << TABLE_ENDLINE;
  (*wl_rpt) << "Trunk" << Str::printf("%.3f", trunk_wire_len) << TABLE_ENDLINE;
  (*wl_rpt) << "Leaf" << Str::printf("%.3f", leaf_wire_len) << TABLE_ENDLINE;
  (*wl_rpt) << "Total" << Str::printf("%.3f", total_wire_len) << TABLE_ENDLINE;
  (*wl_rpt) << "Max net length" << Str::printf("%.3f", max_net_len) << TABLE_ENDLINE;

  (*hpwl_wl_rpt) << "Top" << Str::printf("%.3f", hpwl_top_wire_len) << TABLE_ENDLINE;
  (*hpwl_wl_rpt) << "Trunk" << Str::printf("%.3f", hpwl_trunk_wire_len) << TABLE_ENDLINE;
  (*hpwl_wl_rpt) << "Leaf" << Str::printf("%.3f", hpwl_leaf_wire_len) << TABLE_ENDLINE;
  (*hpwl_wl_rpt) << "Total" << Str::printf("%.3f", hpwl_total_wire_len) << TABLE_ENDLINE;
  (*hpwl_wl_rpt) << "Max net length" << Str::printf("%.3f", hpwl_max_net_len) << TABLE_ENDLINE;

  auto wl_save_path = dir + "/wire_length.rpt";
  CTSAPIInst.checkFile(dir, "wire_length");
  std::ofstream wl_save_file(wl_save_path);
  wl_save_file << "Generate the report at " << Time::getNowWallTime() << std::endl;
  wl_save_file << wl_rpt->c_str() << "\n\n";
  wl_save_file << hpwl_wl_rpt->c_str();
  // depth statistics(level, num) need to build tree
  // TBD

  // cell stats(Cell type, Count, Area, Capacitance)
  auto cell_stats_rpt = CtsReportTable::createReportTable("Cell stats", CtsReportType::kCellStatus);
  struct CellStatsProperty
  {
    int total_num;
    double total_area;
    double total_cap;
  };
  std::map<std::string, CellStatsProperty> cell_property_map;
  for (auto [cell_master, count] : cell_count_map) {
    auto cell_type = CTSAPIInst.getCellType(cell_master);
    auto cell_area = CTSAPIInst.getCellArea(cell_master);
    auto cell_cap = CTSAPIInst.getCellCap(cell_master);
    if (cell_property_map.count(cell_type) == 0) {
      cell_property_map[cell_type] = {count, cell_area * count, cell_cap * count};
    } else {
      cell_property_map[cell_type].total_num += count;
      cell_property_map[cell_type].total_area += cell_area * count;
      cell_property_map[cell_type].total_cap += cell_cap * count;
    }
  }
  for (auto [type, cell_property] : cell_property_map) {
    (*cell_stats_rpt) << type << cell_property.total_num << cell_property.total_area << cell_property.total_cap << TABLE_ENDLINE;
  }
  auto cell_stats_save_path = dir + "/cell_stats.rpt";
  CTSAPIInst.checkFile(dir, "cell_stats");
  std::ofstream cell_stats_save_file(cell_stats_save_path);
  cell_stats_save_file << "Generate the report at " << Time::getNowWallTime() << std::endl;
  cell_stats_save_file << cell_stats_rpt->c_str();

  // lib cell distribution(Name, Type, Inst Count, Inst Area)
  auto lib_cell_dist_rpt = CtsReportTable::createReportTable("Library cell distribution", CtsReportType::kLibCellDist);
  for (auto [cell_master, count] : cell_count_map) {
    (*lib_cell_dist_rpt) << cell_master << CTSAPIInst.getCellType(cell_master) << count << count * CTSAPIInst.getCellArea(cell_master)
                         << TABLE_ENDLINE;
  }
  auto lib_cell_dist_save_path = dir + "/lib_cell_dist.rpt";
  CTSAPIInst.checkFile(dir, "lib_cell_dist");
  std::ofstream lib_cell_dist_save_file(lib_cell_dist_save_path);
  lib_cell_dist_save_file << "Generate the report at " << Time::getNowWallTime() << std::endl;
  lib_cell_dist_save_file << lib_cell_dist_rpt->c_str();

  // net level distribution(Level, Num)
  auto net_level_rpt = CtsReportTable::createReportTable("Net level distribution", CtsReportType::kNetLevel);
  std::map<int, int> net_level_map;
  int all_num = 0;
  for (auto eval_net : _eval_nets) {
    if (!eval_net.is_newly()) {
      continue;
    }
    auto* driver = eval_net.get_driver();
    if (net_level_map.count(driver->get_level()) == 0) {
      net_level_map[driver->get_level()] = 1;
    } else {
      net_level_map[driver->get_level()]++;
    }
  }
  for (auto [_, num] : net_level_map) {
    all_num += num;
  }
  for (auto [level, num] : net_level_map) {
    (*net_level_rpt) << level << num << 1.0 * num / all_num << TABLE_ENDLINE;
  }
  auto net_level_save_path = dir + "/net_level.rpt";
  CTSAPIInst.checkFile(dir, "net_level");
  std::ofstream net_level_save_file(net_level_save_path);
  net_level_save_file << "Generate the report at " << Time::getNowWallTime() << std::endl;
  net_level_save_file << net_level_rpt->c_str();

  // evaluate design
  CTSAPIInst.saveToLog("\n\n############Evaluate design INFO############");

  CTSAPIInst.latencySkewLog();
  CTSAPIInst.utilizationLog();

  CTSAPIInst.saveToLog("\n\n##Buffering (net) Log##\n");
  for (auto [type, cell_property] : cell_property_map) {
    CTSAPIInst.saveToLog("Cell type: ", type, ", Count: ", cell_property.total_num, ", Area: ", cell_property.total_area,
                         ", Capacitance: ", cell_property.total_cap);
  }
  if (cell_property_map.empty()) {
    CTSAPIInst.saveToLog("[No buffer is used]");
  }
  auto [min_level, max_level] = getPathLevel();
  CTSAPIInst.saveToLog("Clock Path Min num of Buffers: ", max_level == 0 ? 0 : min_level);
  CTSAPIInst.saveToLog("Clock Path Max num of Buffers: ", max_level);
  CTSAPIInst.saveToLog("Max Clock Wirelength: ", max_net_len);
  CTSAPIInst.saveToLog("Total Clock Wirelength: ", total_wire_len);

  CTSAPIInst.slackLog();

  CTSAPIInst.saveToLog("\n############Evaluate design Done############");
}

void Evaluator::plotPath(const string& inst_name, const string& file) const
{
  auto* config = CTSAPIInst.get_config();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto path = config->get_sta_workspace() + "/" + file;
  auto ofs = std::fstream(path, std::ios::out | std::ios::trunc);

  CtsInstance* path_inst = nullptr;
  for (auto& eval_net : _eval_nets) {
    auto inst = eval_net.get_instance(inst_name);
    if (inst && (eval_net.get_driver() != inst)) {
      path_inst = inst;
      break;
    }
  }
  LOG_FATAL_IF(path_inst == nullptr) << "Cannot find instance: " << inst_name;

  GDSPloter::head(ofs);
  vector<CtsInstance*> insts;
  while (path_inst) {
    GDSPloter::insertInstance(ofs, path_inst);
    insts.emplace_back(path_inst);
    auto before_load_pin = path_inst->get_load_pin();
    if (before_load_pin) {
      auto before_net = before_load_pin->get_net();
      auto driver_pin = before_net->get_driver_pin();
      if (driver_pin->is_io() || !CTSAPIInst.isClockNet(before_net->get_net_name())) {
        break;
      }
      CTSAPIInst.saveToLog("Net: ", before_net->get_net_name());
      CTSAPIInst.saveToLog("Driver Pin: ", before_net->get_driver_pin()->get_full_name());
      for (auto load_pin : before_net->get_load_pins()) {
        if (db_wrapper->ctsToIdb(load_pin)->is_flip_flop_clk()) {
          CTSAPIInst.saveToLog("Load Clock Pin: ", load_pin->get_full_name());
        }
      }
      auto driver_inst = driver_pin->get_instance();
      GDSPloter::insertWire(ofs, driver_inst->get_location(), path_inst->get_location());
      path_inst = driver_inst;
    } else {
      break;
    }
  }
  auto core = db_wrapper->get_core_bounding_box();
  GDSPloter::insertPolygon(ofs, core, "core", _default_size);
  GDSPloter::strBegin(ofs);
  GDSPloter::topBegin(ofs);
  for (auto* inst : insts) {
    GDSPloter::refInstance(ofs, inst);
  }
  GDSPloter::refPolygon(ofs, "core");
  GDSPloter::refPolygon(ofs, "WIRE");
  GDSPloter::strEnd(ofs);

  GDSPloter::tail(ofs);
  LOG_INFO << "Path to " << inst_name << " has been written to " << path;
}

void Evaluator::plotNet(const string& net_name, const string& file) const
{
  auto* design = CTSAPIInst.get_design();
  auto& clk_nets = design->get_nets();
  CtsNet* net = nullptr;
  for (auto& clk_net : clk_nets) {
    if (clk_net->get_net_name() == net_name) {
      net = clk_net;
      break;
    }
  }
  LOG_FATAL_IF(net == nullptr) << "Net " << net_name << " not found";

  auto* config = CTSAPIInst.get_config();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto path = config->get_sta_workspace() + "/" + file;
  auto ofs = std::fstream(path, std::ios::out | std::ios::trunc);

  GDSPloter::head(ofs);
  auto insts = net->get_instances();
  for (auto* inst : insts) {
    GDSPloter::insertInstance(ofs, inst);
  }
  for (const auto& wire : net->get_signal_wires()) {
    auto first = wire.get_first().point;
    auto second = wire.get_second().point;
    GDSPloter::insertWire(ofs, first, second);
  }
  auto core = db_wrapper->get_core_bounding_box();
  GDSPloter::insertPolygon(ofs, core, "core", _default_size);
  GDSPloter::strBegin(ofs);
  GDSPloter::topBegin(ofs);
  for (auto* inst : insts) {
    GDSPloter::refInstance(ofs, inst);
  }
  GDSPloter::refPolygon(ofs, "core");
  GDSPloter::refPolygon(ofs, "WIRE");
  GDSPloter::strEnd(ofs);

  GDSPloter::tail(ofs);
  LOG_INFO << "Net: " << net_name << " has been written to " << path;
}
}  // namespace icts