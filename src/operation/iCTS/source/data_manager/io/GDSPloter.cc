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

GDSPloter::GDSPloter()
{
  auto* config = CTSAPIInst.get_config();
  _log_ofs.open(config->get_gds_file(), std::ios::out | std::ios::trunc);
}

GDSPloter::GDSPloter(const string& gds_file)
{
  _log_ofs.open(gds_file, std::ios::out | std::ios::trunc);
}

void GDSPloter::plotDesign()
{
  auto* design = CTSAPIInst.get_design();
  auto& clk_nets = design->get_nets();
  auto* config = CTSAPIInst.get_config();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto path = config->get_sta_workspace() + "/cts_design.gds";
  auto plotter = GDSPloter(path);

  plotter.head();

  for (auto& clk_net : clk_nets) {
    auto net_name = clk_net->get_net_name();

    for (const auto& wire : clk_net->get_signal_wires()) {
      auto first = wire.get_first().point;
      auto second = wire.get_second().point;
      plotter.insertWire(first, second, clk_net->get_driver_inst()->get_level());
    }
  }

  auto* idb_design = db_wrapper->get_idb()->get_def_service()->get_design();
  auto idb_insts = idb_design->get_instance_list()->get_instance_list();
  for (auto idb_inst : idb_insts) {
    auto* inst_box = idb_inst->get_bounding_box();
    auto name = idb_inst->get_name();
    plotter.insertPolygon(inst_box, name, 1);
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
      plotter.insertPolygon(blockage_rect, name, 1);
    }
  }

  auto core = db_wrapper->get_core_bounding_box();
  plotter.insertPolygon(core, "core", 100);
  plotter.strBegin();
  plotter.topBegin();
  for (auto idb_inst : idb_insts) {
    auto name = idb_inst->get_name();
    plotter.refPolygon(name);
  }

  for (auto* blockage : idb_blockages) {
    auto blockage_rect_list = blockage->get_rect_list();
    if (blockage_rect_list.empty()) {
      LOG_WARNING << "rectangles of blockage are empty!";
      continue;
    }
    for (size_t i = 0; i < blockage_rect_list.size(); ++i) {
      auto name = blockage->get_instance_name() + std::to_string(i);
      plotter.refPolygon(name);
    }
  }
  plotter.refPolygon("core");
  plotter.refPolygon("WIRE");
  plotter.strEnd();

  plotter.tail();
}

void GDSPloter::plotFlyLine()
{
  auto* design = CTSAPIInst.get_design();
  auto& clk_nets = design->get_nets();
  auto* config = CTSAPIInst.get_config();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto path = config->get_sta_workspace() + "/cts_fly_line.gds";
  auto plotter = GDSPloter(path);

  plotter.head();

  for (auto& clk_net : clk_nets) {
    auto driver = clk_net->get_driver_inst();
    for (auto load : clk_net->get_load_insts()) {
      plotter.insertWire(driver->get_location(), load->get_location(), driver->get_level());
    }
  }

  auto* idb_design = db_wrapper->get_idb()->get_def_service()->get_design();
  auto idb_insts = idb_design->get_instance_list()->get_instance_list();
  for (auto idb_inst : idb_insts) {
    auto* inst_box = idb_inst->get_bounding_box();
    auto name = idb_inst->get_name();
    plotter.insertPolygon(inst_box, name, 1);
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
      plotter.insertPolygon(blockage_rect, name, 1);
    }
  }

  auto* core = db_wrapper->get_core_bounding_box();
  plotter.insertPolygon(core, "core", 100);
  plotter.strBegin();
  plotter.topBegin();
  for (auto idb_inst : idb_insts) {
    auto name = idb_inst->get_name();
    plotter.refPolygon(name);
  }

  for (auto* blockage : idb_blockages) {
    auto blockage_rect_list = blockage->get_rect_list();
    if (blockage_rect_list.empty()) {
      LOG_WARNING << "rectangles of blockage are empty!";
      continue;
    }
    for (size_t i = 0; i < blockage_rect_list.size(); ++i) {
      auto name = blockage->get_instance_name() + std::to_string(i);
      plotter.refPolygon(name);
    }
  }
  plotter.refPolygon("core");
  plotter.refPolygon("WIRE");
  plotter.strEnd();

  plotter.tail();
}

void GDSPloter::refPolygon(const string& name)
{
  _log_ofs << "SREF" << std::endl;
  _log_ofs << "SNAME " << name << std::endl;
  _log_ofs << "XY 0:0" << std::endl;
  _log_ofs << "ENDEL" << std::endl;
}

void GDSPloter::insertInstance(CtsInstance* inst)
{
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto rect = db_wrapper->get_bounding_box(inst);
  string name = inst->get_name();
  int layer = inst->get_level();
  insertPolygon(rect, name, layer);
}

void GDSPloter::insertWire(const Point& begin, const Point& end, const int& layer, const int& width)
{
  _log_ofs << "BGNSTR" << std::endl;
  _log_ofs << "STRNAME WIRE" << std::endl;
  _log_ofs << "PATH" << std::endl;
  _log_ofs << "LAYER " + std::to_string(layer) << std::endl;
  _log_ofs << "DATATYPE 0" << std::endl;
  _log_ofs << "WIDTH " << std::to_string(width) << std::endl;
  _log_ofs << "XY" << std::endl;
  auto begin_x = std::to_string(begin.x());
  auto begin_y = std::to_string(begin.y());
  auto end_x = std::to_string(end.x());
  auto end_y = std::to_string(end.y());
  _log_ofs << begin_x << ":" << begin_y << std::endl;
  _log_ofs << end_x << ":" << end_y << std::endl;
  _log_ofs << "ENDEL" << std::endl;
  _log_ofs << "ENDSTR" << std::endl;
}

void GDSPloter::refInstance(CtsInstance* inst)
{
  refPolygon(inst->get_name());
}

void GDSPloter::plotPolygons(const std::vector<IdbRect*>& polys, const string& name, int layer)
{
  head();
  size_t idx = 0;
  for (auto poly : polys) {
    insertPolygon(poly, name + std::to_string(idx++));
  }
  tail();
}

void GDSPloter::insertPolygon(IdbRect* poly, const string& name, int layer)
{
  insertPolygon(*poly, name, layer);
}

void GDSPloter::insertPolygon(IdbRect& poly, const string& name, int layer)
{
  std::vector<Point> points
      = {Point(poly.get_low_x(), poly.get_low_y()), Point(poly.get_low_x(), poly.get_high_y()), Point(poly.get_high_x(), poly.get_high_y()),
         Point(poly.get_high_x(), poly.get_low_y()), Point(poly.get_low_x(), poly.get_low_y())};
  _log_ofs << "BGNSTR" << std::endl;
  _log_ofs << "STRNAME " << name << std::endl;

  _log_ofs << "BOUNDARY" << std::endl;
  _log_ofs << "LAYER " << layer << std::endl;
  _log_ofs << "DATATYPE 0" << std::endl;
  _log_ofs << "XY" << std::endl;
  std::ranges::for_each(points, [&](const Point& point) { _log_ofs << point << std::endl; });
  _log_ofs << "ENDEL" << std::endl;

  _log_ofs << "ENDSTR" << std::endl;
}

void GDSPloter::topBegin()
{
  _log_ofs << "STRNAME top" << std::endl;
}

void GDSPloter::strBegin()
{
  _log_ofs << "BGNSTR" << std::endl;
}

void GDSPloter::strEnd()
{
  _log_ofs << "ENDSTR" << std::endl;
}

void GDSPloter::plotInstances(vector<CtsInstance*>& insts)
{
  head();

  for (auto* inst : insts) {
    insertInstance(inst);
  }

  strBegin();
  topBegin();
  for (auto* inst : insts) {
    refInstance(inst);
  }
  strEnd();

  tail();
}

void GDSPloter::head()
{
  _log_ofs << "HEADER 600" << std::endl;
  _log_ofs << "BGNLIB" << std::endl;
  _log_ofs << "LIBNAME CTS_Lib" << std::endl;
  _log_ofs << "UNITS 0.001 1e-9" << std::endl;
}

void GDSPloter::tail()
{
  _log_ofs << "ENDLIB" << std::endl;
}

}  // namespace icts