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

#include "CTSAPI.hh"
#include "CtsDBWrapper.hh"

namespace icts {

void GDSPloter::plotDesign(const std::string& path)
{
  auto* design = CTSAPIInst.get_design();
  auto& clk_nets = design->get_nets();
  auto* config = CTSAPIInst.get_config();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto file_path = path.empty() ? config->get_sta_workspace() + "/cts_design.gds" : path;
  auto ofs = std::fstream(file_path, std::ios::out | std::ios::trunc);

  head(ofs);

  for (auto& clk_net : clk_nets) {
    auto net_name = clk_net->get_net_name();

    for (const auto& wire : clk_net->get_signal_wires()) {
      auto first = wire.get_first().point;
      auto second = wire.get_second().point;
      insertWire(ofs, first, second, clk_net->get_driver_inst()->get_level());
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
      insertPolygon(ofs, inst_box, name, 1);
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
      insertPolygon(ofs, blockage_rect, name, 1);
    }
  }

  auto core = db_wrapper->get_core_bounding_box();
  insertPolygon(ofs, core, "core", 100);
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
  refPolygon(ofs, "WIRE");
  strEnd(ofs);

  tail(ofs);
}

void GDSPloter::plotFlyLine(const std::string& path)
{
  auto* design = CTSAPIInst.get_design();
  auto& clk_nets = design->get_nets();
  auto* config = CTSAPIInst.get_config();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto file_path = path.empty() ? config->get_sta_workspace() + "/cts_flyline.gds" : path;
  auto ofs = std::fstream(file_path, std::ios::out | std::ios::trunc);

  head(ofs);

  for (auto& clk_net : clk_nets) {
    auto driver = clk_net->get_driver_inst();
    for (auto load : clk_net->get_load_insts()) {
      insertWire(ofs, driver->get_location(), load->get_location(), driver->get_level());
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
      insertPolygon(ofs, inst_box, name, 1);
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
      insertPolygon(ofs, blockage_rect, name, 1);
    }
  }

  auto* core = db_wrapper->get_core_bounding_box();
  insertPolygon(ofs, core, "core", 100);
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
  refPolygon(ofs, "WIRE");
  strEnd(ofs);

  tail(ofs);
}

void GDSPloter::writePyDesign(const std::string& path)
{
  auto* design = CTSAPIInst.get_design();

  auto* config = CTSAPIInst.get_config();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto file_path = path.empty() ? config->get_sta_workspace() + "/cts_design.py" : path;
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
  py_ofs << "def generate_color_sequence(n): " << std::endl;
  py_ofs << "    cmap = plt.get_cmap('viridis')" << std::endl;
  py_ofs << "    colors = [cmap(i) for i in np.linspace(0, 1, n)]" << std::endl;
  py_ofs << "    return colors" << std::endl;
  py_ofs << "colors = generate_color_sequence(" << max_level + 1 << ")" << std::endl;
  py_ofs << "fig = plt.figure(figsize=(8,6), dpi=300)" << std::endl;

  for (auto* idb_inst : idb_insts) {
    auto* inst_box = idb_inst->get_bounding_box();
    auto* cts_inst = design->findInstance(idb_inst->get_name());
    int level = 0;
    if (cts_inst) {
      level = cts_inst->get_level();
    }
    py_ofs << "plt.plot([" << inst_box->get_low_x() << "," << inst_box->get_low_x() << "," << inst_box->get_high_x() << ","
           << inst_box->get_high_x() << "," << inst_box->get_low_x() << "],[" << inst_box->get_low_y() << "," << inst_box->get_high_y()
           << "," << inst_box->get_high_y() << "," << inst_box->get_low_y() << "," << inst_box->get_low_y() << "],color=colors[" << level
           << "])" << std::endl;
  }
  auto& clk_nets = design->get_nets();
  for (auto& clk_net : clk_nets) {
    auto level = clk_net->get_driver_inst()->get_level();
    for (const auto& wire : clk_net->get_signal_wires()) {
      auto first = wire.get_first().point;
      auto second = wire.get_second().point;
      // line width should add with level
      py_ofs << "plt.plot([" << first.x() << "," << second.x() << "],[" << first.y() << "," << second.y() << "],color=colors[" << level
             << "],linewidth=" << level + 1 << ")" << std::endl;
    }
  }
  py_ofs << "plt.savefig('cts_design.png', dpi=300)" << std::endl;
}

void GDSPloter::writePyFlyLine(const std::string& path)
{
  auto* design = CTSAPIInst.get_design();

  auto* config = CTSAPIInst.get_config();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto file_path = path.empty() ? config->get_sta_workspace() + "/cts_flyline.py" : path;
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
  py_ofs << "def generate_color_sequence(n): " << std::endl;
  py_ofs << "    cmap = plt.get_cmap('viridis')" << std::endl;
  py_ofs << "    colors = [cmap(i) for i in np.linspace(0, 1, n)]" << std::endl;
  py_ofs << "    return colors" << std::endl;
  py_ofs << "colors = generate_color_sequence(" << max_level + 1 << ")" << std::endl;
  py_ofs << "fig = plt.figure(figsize=(8,6), dpi=300)" << std::endl;

  for (auto* idb_inst : idb_insts) {
    auto* inst_box = idb_inst->get_bounding_box();
    auto* cts_inst = design->findInstance(idb_inst->get_name());
    int level = 0;
    if (cts_inst) {
      level = cts_inst->get_level();
    }
    py_ofs << "plt.plot([" << inst_box->get_low_x() << "," << inst_box->get_low_x() << "," << inst_box->get_high_x() << ","
           << inst_box->get_high_x() << "," << inst_box->get_low_x() << "],[" << inst_box->get_low_y() << "," << inst_box->get_high_y()
           << "," << inst_box->get_high_y() << "," << inst_box->get_low_y() << "," << inst_box->get_low_y() << "],color=colors[" << level
           << "])" << std::endl;
  }
  auto& clk_nets = design->get_nets();
  for (auto& clk_net : clk_nets) {
    auto level = clk_net->get_driver_inst()->get_level();
    for (auto load : clk_net->get_load_insts()) {
      py_ofs << "plt.plot([" << clk_net->get_driver_inst()->get_location().x() << "," << load->get_location().x() << "],["
             << clk_net->get_driver_inst()->get_location().y() << "," << load->get_location().y() << "],color=colors[" << level
             << "],linewidth=" << level + 1 << ")" << std::endl;
    }
  }
  py_ofs << "plt.savefig('cts_flyline.png', dpi=300)" << std::endl;
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

void GDSPloter::insertWire(std::fstream& log_ofs, const Point& begin, const Point& end, const int& layer, const int& width)
{
  log_ofs << "BGNSTR" << std::endl;
  log_ofs << "STRNAME WIRE" << std::endl;
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

void GDSPloter::insertPolygon(std::fstream& log_ofs, IdbRect* poly, const string& name, int layer)
{
  insertPolygon(log_ofs, *poly, name, layer);
}

void GDSPloter::insertPolygon(std::fstream& log_ofs, IdbRect& poly, const string& name, int layer)
{
  std::vector<Point> points
      = {Point(poly.get_low_x(), poly.get_low_y()), Point(poly.get_low_x(), poly.get_high_y()), Point(poly.get_high_x(), poly.get_high_y()),
         Point(poly.get_high_x(), poly.get_low_y()), Point(poly.get_low_x(), poly.get_low_y())};
  log_ofs << "BGNSTR" << std::endl;
  log_ofs << "STRNAME " << name << std::endl;

  log_ofs << "BOUNDARY" << std::endl;
  log_ofs << "LAYER " << layer << std::endl;
  log_ofs << "DATATYPE 1" << std::endl;
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