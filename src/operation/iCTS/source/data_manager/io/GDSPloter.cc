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
#include "GDSPloter.h"

#include "CTSAPI.hpp"

using namespace std;

namespace icts {

GDSPloter::GDSPloter() {
  auto *config = CTSAPIInst.get_config();
  _log_ofs.open(config->get_gds_file(), std::ios::out | std::ios::trunc);
}

GDSPloter::GDSPloter(const string &gds_file) {
  _log_ofs.open(gds_file, std::ios::out | std::ios::trunc);
}

void GDSPloter::refPolygon(const string &name) {
  _log_ofs << "SREF" << std::endl;
  _log_ofs << "SNAME " << name << std::endl;
  _log_ofs << "XY 0:0" << std::endl;
  _log_ofs << "ENDEL" << std::endl;
}

void GDSPloter::insertInstance(CtsInstance *inst) {
  auto *db_wrapper = CTSAPIInst.get_db_wrapper();
  auto rect = db_wrapper->get_bounding_box(inst);
  string name = inst->get_name();
  int layer = inst->get_level();
  insertPolygon(rect, name, layer);
}

void GDSPloter::insertWire(const Point &begin, const Point &end,
                           const int &layer, const int &width) {
  _log_ofs << "BGNSTR" << std::endl;
  _log_ofs << "STRNAME WIRE" << std::endl;
  _log_ofs << "PATH" << endl;
  _log_ofs << "LAYER " + to_string(layer) << endl;
  _log_ofs << "DATATYPE 0" << endl;
  _log_ofs << "WIDTH " << to_string(width) << endl;
  _log_ofs << "XY" << endl;
  auto begin_x = to_string(begin.x());
  auto begin_y = to_string(begin.y());
  auto end_x = to_string(end.x());
  auto end_y = to_string(end.y());
  _log_ofs << begin_x << ":" << begin_y << endl;
  _log_ofs << end_x << ":" << end_y << endl;
  _log_ofs << "ENDEL" << endl;
  _log_ofs << "ENDSTR" << std::endl;
}

void GDSPloter::refInstance(CtsInstance *inst) { refPolygon(inst->get_name()); }

void GDSPloter::topBegin() { _log_ofs << "STRNAME top" << std::endl; }

void GDSPloter::strBegin() { _log_ofs << "BGNSTR" << std::endl; }

void GDSPloter::strEnd() { _log_ofs << "ENDSTR" << std::endl; }

void GDSPloter::plotInstances(vector<CtsInstance *> &insts) {
  head();

  for (auto *inst : insts) {
    insertInstance(inst);
  }

  strBegin();
  topBegin();
  for (auto *inst : insts) {
    refInstance(inst);
  }
  strEnd();

  tail();
}

void GDSPloter::head() {
  _log_ofs << "HEADER 600" << std::endl;
  _log_ofs << "BGNLIB" << std::endl;
  _log_ofs << "LIBNAME CTS_Lib" << std::endl;
  _log_ofs << "UNITS 0.001 1e-9" << std::endl;
}

void GDSPloter::tail() { _log_ofs << "ENDLIB" << std::endl; }

}  // namespace icts