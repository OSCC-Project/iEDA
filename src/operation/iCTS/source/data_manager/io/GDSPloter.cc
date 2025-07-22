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
 * @file GDSPloter.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "GDSPloter.hh"

#include <filesystem>

#include "CTSAPI.hh"
#include "CtsDBWrapper.hh"
#include "json.hpp"

namespace icts {

void GDSPloter::plotDesign(const std::string& path)
{
  auto* design = CTSAPIInst.get_design();
  auto& clk_nets = design->get_nets();
  auto* config = CTSAPIInst.get_config();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto file_path = path;
  if (file_path.empty()) {
    auto dir = std::filesystem::path(config->get_work_dir()).append("output").string();
    if (!std::filesystem::exists(dir)) {
      std::filesystem::create_directories(dir);
    }
    file_path = std::filesystem::path(dir).append("cts_design.gds").string();
  }
  auto ofs = std::fstream(file_path, std::ios::out | std::ios::trunc);

  head(ofs);

  for (auto& clk_net : clk_nets) {
    size_t wire_id = 0;
    auto net_name = clk_net->get_net_name();

    for (const auto& wire : clk_net->get_signal_wires()) {
      auto wire_name = "WIRE_" + net_name + "_" + std::to_string(wire_id++);
      auto first = wire.get_first().point;
      auto second = wire.get_second().point;
      insertWire(ofs, first, second, wire_name, clk_net->get_driver_inst()->get_level());
    }
  }

  auto* idb_design = db_wrapper->get_idb()->get_def_service()->get_design();
  auto idb_insts = idb_design->get_instance_list()->get_instance_list();
  for (auto* idb_inst : idb_insts) {
    auto* inst_box = idb_inst->get_bounding_box();
    auto name = idb_inst->get_name();
    auto* cts_inst = design->findInstance(name);
    if (cts_inst) {
      insertInstance(ofs, cts_inst);
    } else {
      insertPolygon(ofs, inst_box, name, 1, 0);
    }
  }

  auto idb_blockages = idb_design->get_blockage_list()->get_blockage_list();
  for (auto* blockage : idb_blockages) {
    auto blockage_rect_list = blockage->get_rect_list();
    if (blockage_rect_list.empty()) {
      LOG_WARNING << "rectangles of blockage are empty!";
      continue;
    }
    int i = 0;
    for (auto* blockage_rect : blockage_rect_list) {
      auto name = blockage->get_instance_name() + std::to_string(i);
      ++i;
      insertPolygon(ofs, blockage_rect, name, 1, 0);
    }
  }

  auto core = db_wrapper->get_core_bounding_box();
  insertPolygon(ofs, core, "core", 100, 0);
  strBegin(ofs);
  topBegin(ofs);
  for (auto* idb_inst : idb_insts) {
    auto name = idb_inst->get_name();
    refPolygon(ofs, name);
  }

  for (auto* blockage : idb_blockages) {
    auto blockage_rect_list = blockage->get_rect_list();
    if (blockage_rect_list.empty()) {
      LOG_WARNING << "rectangles of blockage are empty!";
      continue;
    }
    for (size_t i = 0; i < blockage_rect_list.size(); ++i) {
      auto name = blockage->get_instance_name() + std::to_string(i);
      refPolygon(ofs, name);
    }
  }
  refPolygon(ofs, "core");
  for (auto& clk_net : clk_nets) {
    auto net_name = clk_net->get_net_name();

    for (size_t i = 0; i < clk_net->get_signal_wires().size(); ++i) {
      auto wire_name = "WIRE_" + net_name + "_" + std::to_string(i);
      refPolygon(ofs, wire_name);
    }
  }
  strEnd(ofs);

  tail(ofs);
}

void GDSPloter::plotFlyLine(const std::string& path)
{
  auto* design = CTSAPIInst.get_design();
  auto& clk_nets = design->get_nets();
  auto* config = CTSAPIInst.get_config();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto file_path = path;
  if (file_path.empty()) {
    auto dir = std::filesystem::path(config->get_work_dir()).append("output").string();
    if (!std::filesystem::exists(dir)) {
      std::filesystem::create_directories(dir);
    }
    file_path = std::filesystem::path(dir).append("cts_flyline.gds").string();
  }
  auto ofs = std::fstream(file_path, std::ios::out | std::ios::trunc);

  head(ofs);

  for (auto& clk_net : clk_nets) {
    auto* driver = clk_net->get_driver_inst();
    auto* driver_pin = clk_net->get_driver_pin();
    size_t wire_id = 0;
    for (auto* load_pin : clk_net->get_load_pins()) {
      auto wire_name = "WIRE_" + clk_net->get_net_name() + "_" + std::to_string(wire_id++);
      insertWire(ofs, driver_pin->get_location(), load_pin->get_location(), wire_name, driver->get_level());
    }
  }

  auto* idb_design = db_wrapper->get_idb()->get_def_service()->get_design();
  auto idb_insts = idb_design->get_instance_list()->get_instance_list();
  for (auto* idb_inst : idb_insts) {
    auto* inst_box = idb_inst->get_bounding_box();
    auto name = idb_inst->get_name();
    auto* cts_inst = design->findInstance(name);
    if (cts_inst) {
      insertInstance(ofs, cts_inst);
    } else {
      insertPolygon(ofs, inst_box, name, 1, 0);
    }
  }

  auto idb_blockages = idb_design->get_blockage_list()->get_blockage_list();
  for (auto* blockage : idb_blockages) {
    auto blockage_rect_list = blockage->get_rect_list();
    if (blockage_rect_list.empty()) {
      LOG_WARNING << "rectangles of blockage are empty!";
      continue;
    }
    int i = 0;
    for (auto* blockage_rect : blockage_rect_list) {
      auto name = blockage->get_instance_name() + std::to_string(i);
      ++i;
      insertPolygon(ofs, blockage_rect, name, 1, 0);
    }
  }

  auto* core = db_wrapper->get_core_bounding_box();
  insertPolygon(ofs, core, "core", 100, 0);
  strBegin(ofs);
  topBegin(ofs);
  for (auto* idb_inst : idb_insts) {
    auto name = idb_inst->get_name();
    refPolygon(ofs, name);
  }

  for (auto* blockage : idb_blockages) {
    auto blockage_rect_list = blockage->get_rect_list();
    if (blockage_rect_list.empty()) {
      LOG_WARNING << "rectangles of blockage are empty!";
      continue;
    }
    for (size_t i = 0; i < blockage_rect_list.size(); ++i) {
      auto name = blockage->get_instance_name() + std::to_string(i);
      refPolygon(ofs, name);
    }
  }
  refPolygon(ofs, "core");

  for (auto& clk_net : clk_nets) {
    for (size_t i = 0; i < clk_net->get_load_insts().size(); ++i) {
      auto wire_name = "WIRE_" + clk_net->get_net_name() + "_" + std::to_string(i);
      refPolygon(ofs, wire_name);
    }
  }
  strEnd(ofs);

  tail(ofs);
}

void GDSPloter::writePyDesign(const std::string& path)
{
  auto* design = CTSAPIInst.get_design();

  auto* config = CTSAPIInst.get_config();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto file_path = path;
  if (file_path.empty()) {
    auto dir = std::filesystem::path(config->get_work_dir()).append("output").string();
    if (!std::filesystem::exists(dir)) {
      std::filesystem::create_directories(dir);
    }
    file_path = std::filesystem::path(dir).append("cts_design.py").string();
  }
  int max_level = 0;
  auto* idb_design = db_wrapper->get_idb()->get_def_service()->get_design();
  auto idb_insts = idb_design->get_instance_list()->get_instance_list();
  for (auto* idb_inst : idb_insts) {
    auto* cts_inst = design->findInstance(idb_inst->get_name());
    if (cts_inst) {
      max_level = std::max(max_level, cts_inst->get_level());
    }
  }
  // gen py file
  std::ofstream py_ofs(file_path, std::ios::out | std::ios::trunc);
  py_ofs << "import matplotlib.pyplot as plt" << std::endl;
  py_ofs << "import numpy as np" << std::endl;
  py_ofs << "from matplotlib.patches import Rectangle" << std::endl;
  py_ofs << "import scienceplots" << std::endl;
  py_ofs << "plt.style.use(['science','no-latex'])" << std::endl;
  py_ofs << "def generate_color_sequence(n): " << std::endl;
  py_ofs << "    cmap = plt.get_cmap('summer')" << std::endl;
  py_ofs << "    colors = [cmap(i) for i in np.linspace(0, 1, n)]" << std::endl;
  py_ofs << "    return colors" << std::endl;
  if (max_level > 5) {
    py_ofs << "colors = generate_color_sequence(" << max_level + 2 << ")" << std::endl;
  } else {
    py_ofs << "colors = ['#FF0000', '#00FF00', '#008080', '#ff8000', '#910000', '#800080', '#FF1493', '#008B8B', '#8A2BE2', '#32CD32']"
           << std::endl;
  }
  py_ofs << "line_width = np.linspace(0.5, " << (max_level + 2) * 0.5 << ", " << max_level + 2 << ")" << std::endl;
  py_ofs << "inst_colors = generate_color_sequence(" << max_level + 2 << ")" << std::endl;
  py_ofs << "fig = plt.figure(figsize=(8,8), dpi=300)" << std::endl;

  for (auto* idb_inst : idb_insts) {
    auto* inst_box = idb_inst->get_bounding_box();
    auto* cts_inst = design->findInstance(idb_inst->get_name());
    int level = 0;
    if (cts_inst) {
      level = cts_inst->get_level();
    }
    if (level > 1) {
      continue;
    }
    py_ofs << "plt.gca().add_patch(Rectangle((" << inst_box->get_low_x() << "," << inst_box->get_low_y() << "),"
           << inst_box->get_high_x() - inst_box->get_low_x() << "," << inst_box->get_high_y() - inst_box->get_low_y()
           << ",linewidth=0.1,edgecolor='#c0c0c0',facecolor='#c0c0c0',zorder=" << level + 1 << "))" << std::endl;
  }
  auto& clk_nets = design->get_nets();
  for (auto& clk_net : clk_nets) {
    auto* driver = clk_net->get_driver_inst();
    if (driver->get_location() == Point(-1, -1)) {
      continue;
    }
    auto level = driver->get_level();

    for (const auto& wire : clk_net->get_signal_wires()) {
      auto first = wire.get_first().point;
      auto second = wire.get_second().point;
      // line width should add with level
      py_ofs << "plt.plot([" << first.x() << "," << second.x() << "],[" << first.y() << "," << second.y() << "],color=colors[" << level
             << "],linewidth=line_width[" << level << "],zorder=" << level << ")" << std::endl;
    }
  }
  for (auto* idb_inst : idb_insts) {
    auto* inst_box = idb_inst->get_bounding_box();
    auto* cts_inst = design->findInstance(idb_inst->get_name());
    int level = 0;
    if (cts_inst) {
      level = cts_inst->get_level();
    }
    if (level <= 1) {
      continue;
    }
    py_ofs << "plt.gca().add_patch(Rectangle((" << inst_box->get_low_x() << "," << inst_box->get_low_y() << "),"
           << inst_box->get_high_x() - inst_box->get_low_x() << "," << inst_box->get_high_y() - inst_box->get_low_y()
           << ",linewidth=" << 1 + 1.0 * level / 10 << ",edgecolor='black',facecolor=inst_colors[" << level + 1 << "],zorder=" << level + 1
           << "))" << std::endl;
  }
  py_ofs << "plt.axis('square')\n";
  py_ofs << "plt.axis('off')\n";
  py_ofs << "plt.savefig('cts_design.png', dpi=300, bbox_inches='tight')" << std::endl;
}

void GDSPloter::writePyFlyLine(const std::string& path)
{
  auto* design = CTSAPIInst.get_design();

  auto* config = CTSAPIInst.get_config();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto file_path = path;
  if (file_path.empty()) {
    auto dir = std::filesystem::path(config->get_work_dir()).append("output").string();
    if (!std::filesystem::exists(dir)) {
      std::filesystem::create_directories(dir);
    }
    file_path = std::filesystem::path(dir).append("cts_flyline.py").string();
  }
  int max_level = 0;
  auto* idb_design = db_wrapper->get_idb()->get_def_service()->get_design();
  auto idb_insts = idb_design->get_instance_list()->get_instance_list();
  for (auto* idb_inst : idb_insts) {
    auto* cts_inst = design->findInstance(idb_inst->get_name());
    if (cts_inst) {
      max_level = std::max(max_level, cts_inst->get_level());
    }
  }
  // gen py file
  std::ofstream py_ofs(file_path, std::ios::out | std::ios::trunc);
  py_ofs << "import matplotlib.pyplot as plt" << std::endl;
  py_ofs << "import numpy as np" << std::endl;
  py_ofs << "from matplotlib.patches import Rectangle" << std::endl;
  py_ofs << "import scienceplots" << std::endl;
  py_ofs << "plt.style.use(['science','no-latex'])" << std::endl;
  py_ofs << "def generate_color_sequence(n): " << std::endl;
  py_ofs << "    cmap = plt.get_cmap('summer')" << std::endl;
  py_ofs << "    colors = [cmap(i) for i in np.linspace(0, 1, n)]" << std::endl;
  py_ofs << "    return colors" << std::endl;
  if (max_level > 5) {
    py_ofs << "colors = generate_color_sequence(" << max_level + 2 << ")" << std::endl;
  } else {
    py_ofs << "colors = ['#FF0000', '#00FF00', '#008080', '#ff8000', '#910000', '#800080', '#FF1493', '#008B8B', '#8A2BE2', '#32CD32']"
           << std::endl;
  }
  py_ofs << "line_width = np.linspace(0.5, " << (max_level + 2) * 0.5 << ", " << max_level + 2 << ")" << std::endl;
  py_ofs << "inst_colors = generate_color_sequence(" << max_level + 2 << ")" << std::endl;
  py_ofs << "fig = plt.figure(figsize=(8,8), dpi=300)" << std::endl;

  for (auto* idb_inst : idb_insts) {
    auto* inst_box = idb_inst->get_bounding_box();
    auto* cts_inst = design->findInstance(idb_inst->get_name());
    int level = 0;
    if (cts_inst) {
      level = cts_inst->get_level();
    }
    if (level > 1) {
      continue;
    }
    py_ofs << "plt.gca().add_patch(Rectangle((" << inst_box->get_low_x() << "," << inst_box->get_low_y() << "),"
           << inst_box->get_high_x() - inst_box->get_low_x() << "," << inst_box->get_high_y() - inst_box->get_low_y()
           << ",linewidth=0.1,edgecolor='#c0c0c0',facecolor='#c0c0c0',zorder=" << level + 1 << "))" << std::endl;
  }
  auto& clk_nets = design->get_nets();
  for (auto& clk_net : clk_nets) {
    auto* driver = clk_net->get_driver_inst();
    if (driver->get_location() == Point(-1, -1)) {
      continue;
    }
    auto level = driver->get_level();
    for (auto load_pin : clk_net->get_load_pins()) {
      py_ofs << "plt.plot([" << clk_net->get_driver_pin()->get_location().x() << "," << load_pin->get_location().x() << "],["
             << clk_net->get_driver_pin()->get_location().y() << "," << load_pin->get_location().y() << "],color=colors[" << level
             << "],linewidth=line_width[" << level << "],zorder=" << level << ")" << std::endl;
    }
  }
  for (auto* idb_inst : idb_insts) {
    auto* inst_box = idb_inst->get_bounding_box();
    auto* cts_inst = design->findInstance(idb_inst->get_name());
    int level = 0;
    if (cts_inst) {
      level = cts_inst->get_level();
    }
    if (level <= 1) {
      continue;
    }
    py_ofs << "plt.gca().add_patch(Rectangle((" << inst_box->get_low_x() << "," << inst_box->get_low_y() << "),"
           << inst_box->get_high_x() - inst_box->get_low_x() << "," << inst_box->get_high_y() - inst_box->get_low_y()
           << ",linewidth=" << 1 + 1.0 * level / 10 << ",edgecolor='black',facecolor=inst_colors[" << level + 1 << "],zorder=" << level + 1
           << "))" << std::endl;
  }
  py_ofs << "plt.axis('square')\n";
  py_ofs << "plt.axis('off')\n";
  py_ofs << "plt.savefig('cts_flyline.png', dpi=300, bbox_inches='tight')" << std::endl;
}

void GDSPloter::writeJsonDesign(const std::string& path)
{
  auto* design = CTSAPIInst.get_design();
  auto* config = CTSAPIInst.get_config();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();

  auto file_path = path;
  if (file_path.empty()) {
    auto dir = std::filesystem::path(config->get_work_dir()).append("output").string();
    if (!std::filesystem::exists(dir)) {
      std::filesystem::create_directories(dir);
    }
    file_path = std::filesystem::path(dir).append("cts_design.json").string();
  }

  // Calculate the maximum level of instances
  int max_level = 0;
  auto* idb_design = db_wrapper->get_idb()->get_def_service()->get_design();
  auto idb_insts = idb_design->get_instance_list()->get_instance_list();
  for (auto* idb_inst : idb_insts) {
    if (auto* cts_inst = design->findInstance(idb_inst->get_name())) {
      max_level = std::max(max_level, cts_inst->get_level());
    }
  }

  // Create JSON object
  nlohmann::json json_data;
  json_data["design"]["max_level"] = max_level;

  // Add instance information
  nlohmann::json instances = nlohmann::json::array();
  for (auto* idb_inst : idb_insts) {
    nlohmann::json instance;
    instance["name"] = idb_inst->get_name();
    instances.push_back(instance);
  }
  json_data["design"]["instances"] = instances;

  // Add wire information
  nlohmann::json nets = nlohmann::json::array();
  for (auto& clk_nets = design->get_nets(); auto& clk_net : clk_nets) {
    auto* driver = clk_net->get_driver_inst();
    auto* driver_pin = clk_net->get_driver_pin();
    if (driver->get_location() == Point(-1, -1)) {
      continue;
    }

    nlohmann::json net;
    net["name"] = clk_net->get_net_name();
    net["driver_level"] = driver->get_level();
    net["driver_location"]["x"] = driver_pin->get_location().x();
    net["driver_location"]["y"] = driver_pin->get_location().y();

    // Add signal wires
    nlohmann::json wires = nlohmann::json::array();
    for (const auto& signal_wires = clk_net->get_signal_wires(); const auto& wire : signal_wires) {
      auto first = wire.get_first().point;
      auto second = wire.get_second().point;

      nlohmann::json wire_obj;
      wire_obj["start"]["x"] = first.x();
      wire_obj["start"]["y"] = first.y();
      wire_obj["end"]["x"] = second.x();
      wire_obj["end"]["y"] = second.y();

      wires.push_back(wire_obj);
    }
    net["wires_layout"] = wires;

    // Add delay information
    nlohmann::json delays = nlohmann::json::array();

    // Only consider a single clock source for now.
    auto clk_port = design->get_clocks().front()->get_clock_name();
    
    for (auto& p : clk_net->get_load_pins()) {
      auto delay = CTSAPIInst.getClockAT(p->get_full_name(), clk_port);
      auto driver = clk_net->get_driver_pin();

      nlohmann::json delay_obj;
      delay_obj["from"] = driver->get_full_name();
      delay_obj["to"] = p->get_full_name();
      delay_obj["delay"] = delay;
      delays.push_back(delay_obj);
    }
    net["wires_delay"] = delays;

    nets.push_back(net);
  }
  json_data["design"]["nets"] = nets;

  // Add blockage information
  nlohmann::json blockages = nlohmann::json::array();
  for (auto idb_blockages = idb_design->get_blockage_list()->get_blockage_list(); auto* blockage : idb_blockages) {
    auto blockage_rect_list = blockage->get_rect_list();
    if (blockage_rect_list.empty()) {
      continue;
    }

    for (size_t i = 0; i < blockage_rect_list.size(); ++i) {
      auto* blockage_rect = blockage_rect_list[i];

      nlohmann::json blockage_obj;
      blockage_obj["name"] = blockage->get_instance_name() + "_" + std::to_string(i);
      blockage_obj["bounding_box"]["low_x"] = blockage_rect->get_low_x();
      blockage_obj["bounding_box"]["low_y"] = blockage_rect->get_low_y();
      blockage_obj["bounding_box"]["high_x"] = blockage_rect->get_high_x();
      blockage_obj["bounding_box"]["high_y"] = blockage_rect->get_high_y();

      blockages.push_back(blockage_obj);
    }
  }
  json_data["design"]["blockages"] = blockages;

  // Add core bounding box information
  auto* core = db_wrapper->get_core_bounding_box();
  json_data["design"]["core"]["bounding_box"]["low_x"] = core->get_low_x();
  json_data["design"]["core"]["bounding_box"]["low_y"] = core->get_low_y();
  json_data["design"]["core"]["bounding_box"]["high_x"] = core->get_high_x();
  json_data["design"]["core"]["bounding_box"]["high_y"] = core->get_high_y();

  std::ofstream json_file(file_path);
  json_file << json_data.dump(2);
  json_file.close();
}

void GDSPloter::refPolygon(std::fstream& log_ofs, const string& name)
{
  log_ofs << "SREF" << std::endl;
  log_ofs << "SNAME " << name << std::endl;
  log_ofs << "XY 0:0" << std::endl;
  log_ofs << "ENDEL" << std::endl;
}

void GDSPloter::insertInstance(std::fstream& log_ofs, CtsInstance* inst)
{
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto rect = db_wrapper->get_bounding_box(inst);
  string name = inst->get_name();
  int layer = inst->get_level();
  insertPolygon(log_ofs, rect, name, layer);
}

void GDSPloter::insertWire(std::fstream& log_ofs, const Point& begin, const Point& end, const string& name, const int& layer,
                           const int& width)
{
  log_ofs << "BGNSTR" << std::endl;
  log_ofs << "STRNAME " << name << std::endl;
  log_ofs << "PATH" << std::endl;
  log_ofs << "LAYER " + std::to_string(layer) << std::endl;
  log_ofs << "DATATYPE 0" << std::endl;
  log_ofs << "WIDTH " << std::to_string(width) << std::endl;
  log_ofs << "XY" << std::endl;
  auto begin_x = std::to_string(begin.x());
  auto begin_y = std::to_string(begin.y());
  auto end_x = std::to_string(end.x());
  auto end_y = std::to_string(end.y());
  log_ofs << begin_x << ":" << begin_y << std::endl;
  log_ofs << end_x << ":" << end_y << std::endl;
  log_ofs << "ENDEL" << std::endl;
  log_ofs << "ENDSTR" << std::endl;
}

void GDSPloter::refInstance(std::fstream& log_ofs, CtsInstance* inst)
{
  refPolygon(log_ofs, inst->get_name());
}

void GDSPloter::plotPolygons(std::fstream& log_ofs, const std::vector<IdbRect*>& polys, const string& name, int layer)
{
  head(log_ofs);
  size_t idx = 0;
  for (auto poly : polys) {
    insertPolygon(log_ofs, poly, name + std::to_string(idx++));
  }
  tail(log_ofs);
}

void GDSPloter::insertPolygon(std::fstream& log_ofs, IdbRect* poly, const string& name, int layer, const int& type)
{
  insertPolygon(log_ofs, *poly, name, layer, type);
}

void GDSPloter::insertPolygon(std::fstream& log_ofs, IdbRect& poly, const string& name, int layer, const int& type)
{
  std::vector<Point> points
      = {Point(poly.get_low_x(), poly.get_low_y()), Point(poly.get_low_x(), poly.get_high_y()), Point(poly.get_high_x(), poly.get_high_y()),
         Point(poly.get_high_x(), poly.get_low_y()), Point(poly.get_low_x(), poly.get_low_y())};
  log_ofs << "BGNSTR" << std::endl;
  log_ofs << "STRNAME " << name << std::endl;

  log_ofs << "BOUNDARY" << std::endl;
  log_ofs << "LAYER " << layer << std::endl;
  log_ofs << "DATATYPE " << type << std::endl;
  log_ofs << "XY" << std::endl;
  std::ranges::for_each(points, [&](const Point& point) { log_ofs << point << std::endl; });
  log_ofs << "ENDEL" << std::endl;

  log_ofs << "ENDSTR" << std::endl;
}

void GDSPloter::topBegin(std::fstream& log_ofs)
{
  log_ofs << "STRNAME top" << std::endl;
}

void GDSPloter::strBegin(std::fstream& log_ofs)
{
  log_ofs << "BGNSTR" << std::endl;
}

void GDSPloter::strEnd(std::fstream& log_ofs)
{
  log_ofs << "ENDSTR" << std::endl;
}

void GDSPloter::plotInstances(std::fstream& log_ofs, vector<CtsInstance*>& insts)
{
  head(log_ofs);

  for (auto* inst : insts) {
    insertInstance(log_ofs, inst);
  }

  strBegin(log_ofs);
  topBegin(log_ofs);
  for (auto* inst : insts) {
    refInstance(log_ofs, inst);
  }
  strEnd(log_ofs);

  tail(log_ofs);
}

void GDSPloter::head(std::fstream& log_ofs)
{
  log_ofs << "HEADER 600" << std::endl;
  log_ofs << "BGNLIB" << std::endl;
  log_ofs << "LIBNAME CTS_Lib" << std::endl;
  log_ofs << "UNITS 0.001 1e-9" << std::endl;
}

void GDSPloter::tail(std::fstream& log_ofs)
{
  log_ofs << "ENDLIB" << std::endl;
}

}  // namespace icts