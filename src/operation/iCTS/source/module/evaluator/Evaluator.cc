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
#include "time/Time.hh"
namespace icts {

void Evaluator::init()
{
  CTSAPIInst.setPropagateClock();
  printLog();
  initLevel();
  transferData();
}

void Evaluator::calcInfo()
{
  if (_have_calc) {
    return;
  }
  // wirelength distribution
  calcWL();

  // depth statistics(level, num) need to build tree
  // TBD

  // lib cell distribution(Name, Type, Inst Count, Inst Area)
  calcCellDist();

  // cell stats(Cell type, Count, Area, Capacitance)
  calcCellStats();

  // net level distribution(Level, Num)
  calcNetLevel();

  // path info
  calcPathBufStats();

  _have_calc = true;
}

void Evaluator::calcWL()
{
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
        _top_wire_len += net_len;
        _hpwl_top_wire_len += hpwl_net_len;
        break;
      case NetType::kTrunk:
        _trunk_wire_len += net_len;
        _hpwl_trunk_wire_len += hpwl_net_len;
        break;
      case NetType::kLeaf:
        _leaf_wire_len += net_len;
        _hpwl_leaf_wire_len += hpwl_net_len;
        break;
      default:
        break;
    }
    _total_wire_len += net_len;
    _hpwl_total_wire_len += hpwl_net_len;
    _max_net_len = std::max(_max_net_len, net_len);
    _hpwl_max_net_len = std::max(_hpwl_max_net_len, hpwl_net_len);
  }
}

void Evaluator::calcCellDist()
{
  for (const auto& eval_net : _eval_nets) {
    auto* design = CTSAPIInst.get_design();
    // wire length
    auto* net = design->findSolverNet(eval_net.get_name());
    if (!net) {
      continue;
    }

    if (eval_net.is_newly()) {
      // cell count
      auto cell_master = eval_net.get_driver()->get_cell_master();
      if (_cell_dist_map.count(cell_master) == 0) {
        _cell_dist_map[cell_master] = 1;
      } else {
        _cell_dist_map[cell_master]++;
      }
    }
  }
}

void Evaluator::calcCellStats()
{
  for (auto [cell_master, count] : _cell_dist_map) {
    auto cell_type = CTSAPIInst.getCellType(cell_master);
    auto cell_area = CTSAPIInst.getCellArea(cell_master);
    auto cell_cap = CTSAPIInst.getCellCap(cell_master);
    if (_cell_stats_map.count(cell_type) == 0) {
      _cell_stats_map[cell_type] = {count, cell_area * count, cell_cap * count};
    } else {
      _cell_stats_map[cell_type].total_num += count;
      _cell_stats_map[cell_type].total_area += cell_area * count;
      _cell_stats_map[cell_type].total_cap += cell_cap * count;
    }
  }
}

void Evaluator::calcNetLevel()
{
  for (auto eval_net : _eval_nets) {
    if (!eval_net.is_newly()) {
      continue;
    }
    auto* driver = eval_net.get_driver();
    if (_net_level_map.count(driver->get_level()) == 0) {
      _net_level_map[driver->get_level()] = 1;
    } else {
      _net_level_map[driver->get_level()]++;
    }
  }
}

void Evaluator::calcPathBufStats()
{
  _path_infos.clear();
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
  std::vector<TreeNode*> roots;
  for (auto [_, node] : name_to_node) {
    if (node->parent == nullptr) {
      node->depth = 0;
      roots.emplace_back(node);
    }
  }
  // set depth
  std::function<void(TreeNode*)> set_depth = [&](TreeNode* node) {
    for (auto child : node->children) {
      child->depth = node->depth + 1;
      set_depth(child);
    }
  };
  std::ranges::for_each(roots, set_depth);
  std::ranges::for_each(roots, [&](TreeNode* root) {
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
    auto path_info = PathInfo{root->name, max_depth == 0 ? 0 : min_depth, max_depth};
    _path_infos.emplace_back(path_info);
  });
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
    // Commented below code to fix max_level error in cts output
    // if (!clk_net->is_newly()) {
    //   continue;
    // }
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

void Evaluator::pathLevelLog() const
{
  CTSAPIInst.logTitle("Summary of Path Level");

  std::ranges::for_each(_path_infos, [](PathInfo info) {
    CTSAPIInst.saveToLog("Root: ", info.root_name);
    CTSAPIInst.saveToLog("\tClock Path Min num of Buffers: ", info.min_depth);
    CTSAPIInst.saveToLog("\tClock Path Max num of Buffers: ", info.max_depth);
  });
  CTSAPIInst.logEnd();
}

void Evaluator::evaluate()
{
  CTSAPIInst.refresh();
  for (auto eval_net : _eval_nets) {
    CTSAPIInst.buildRCTree(eval_net);
  }
  CTSAPIInst.reportTiming();
}

void Evaluator::statistics(const std::string& save_dir)
{
  if (!_have_calc) {
    calcInfo();
  }

  auto* config = CTSAPIInst.get_config();
  auto dir = (save_dir == "" ? config->get_work_dir() : save_dir) + "/statistics";
  // wirelength statistics(type: total, top, trunk, leaf, total certer dist,
  // max)
  auto wl_rpt = CtsReportTable::createReportTable("Wire length stats", CtsReportType::kWireLength);

  (*wl_rpt) << "Top" << Str::printf("%.3f", _top_wire_len) << TABLE_ENDLINE;
  (*wl_rpt) << "Trunk" << Str::printf("%.3f", _trunk_wire_len) << TABLE_ENDLINE;
  (*wl_rpt) << "Leaf" << Str::printf("%.3f", _leaf_wire_len) << TABLE_ENDLINE;
  (*wl_rpt) << "Total" << Str::printf("%.3f", _total_wire_len) << TABLE_ENDLINE;
  (*wl_rpt) << "Max net length" << Str::printf("%.3f", _max_net_len) << TABLE_ENDLINE;

  auto hpwl_wl_rpt = CtsReportTable::createReportTable("HPWL Wire length stats", CtsReportType::kHpWireLength);
  (*hpwl_wl_rpt) << "Top" << Str::printf("%.3f", _hpwl_top_wire_len) << TABLE_ENDLINE;
  (*hpwl_wl_rpt) << "Trunk" << Str::printf("%.3f", _hpwl_trunk_wire_len) << TABLE_ENDLINE;
  (*hpwl_wl_rpt) << "Leaf" << Str::printf("%.3f", _hpwl_leaf_wire_len) << TABLE_ENDLINE;
  (*hpwl_wl_rpt) << "Total" << Str::printf("%.3f", _hpwl_total_wire_len) << TABLE_ENDLINE;
  (*hpwl_wl_rpt) << "Max net length" << Str::printf("%.3f", _hpwl_max_net_len) << TABLE_ENDLINE;

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
  for (auto [type, cell_property] : _cell_stats_map) {
    (*cell_stats_rpt) << type << cell_property.total_num << cell_property.total_area << cell_property.total_cap << TABLE_ENDLINE;
  }
  auto cell_stats_save_path = dir + "/cell_stats.rpt";
  CTSAPIInst.checkFile(dir, "cell_stats");
  std::ofstream cell_stats_save_file(cell_stats_save_path);
  cell_stats_save_file << "Generate the report at " << Time::getNowWallTime() << std::endl;
  cell_stats_save_file << cell_stats_rpt->c_str();

  // lib cell distribution(Name, Type, Inst Count, Inst Area)
  auto lib_cell_dist_rpt = CtsReportTable::createReportTable("Library cell distribution", CtsReportType::kLibCellDist);
  for (auto [cell_master, count] : _cell_dist_map) {
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
  int all_num = 0;
  for (auto [_, num] : _net_level_map) {
    all_num += num;
  }
  for (auto [level, num] : _net_level_map) {
    (*net_level_rpt) << level << num << 1.0 * num / all_num << TABLE_ENDLINE;
  }
  auto net_level_save_path = dir + "/net_level.rpt";
  CTSAPIInst.checkFile(dir, "net_level");
  std::ofstream net_level_save_file(net_level_save_path);
  net_level_save_file << "Generate the report at " << Time::getNowWallTime() << std::endl;
  net_level_save_file << net_level_rpt->c_str();

  // evaluate design
  CTSAPIInst.latencySkewLog();
  CTSAPIInst.utilizationLog();

  CTSAPIInst.logTitle("Summary of Buffering (net)");
  CTSAPIInst.saveToLog("--Cell Stats--");
  CTSAPIInst.saveToLog(cell_stats_rpt->c_str());
  if (_cell_stats_map.empty()) {
    CTSAPIInst.saveToLog("#No buffer is used#");
  }
  CTSAPIInst.saveToLog("--Wirelength Stats--");
  CTSAPIInst.saveToLog(wl_rpt->c_str());
  CTSAPIInst.logEnd();

  pathLevelLog();

  CTSAPIInst.slackLog();
}

void Evaluator::plotPath(const string& inst_name, const string& file) const
{
  auto* config = CTSAPIInst.get_config();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto path = config->get_work_dir() + "/" + file;
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
      GDSPloter::insertWire(ofs, driver_pin->get_location(), before_load_pin->get_location());
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
  auto path = config->get_work_dir() + "/" + file;
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