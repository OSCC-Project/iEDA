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
 * @file GDSPloter.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "CtsConfig.hh"
#include "CtsDBWrapper.hh"
#include "CtsInstance.hh"

namespace icts {
using std::fstream;
using std::string;

class GDSPloter
{
 public:
  GDSPloter();
  explicit GDSPloter(const string& gds_file);
  GDSPloter(GDSPloter&) = default;
  ~GDSPloter() { _log_ofs.close(); }
  void plotDesign();
  void plotFlyLine();

  void plotInstances(vector<CtsInstance*>& insts);
  void insertInstance(CtsInstance* inst);

  void refPolygon(const string& name);
  void refInstance(CtsInstance* inst);

  void plotPolygons(const std::vector<IdbRect*>& polys, const string& name = "polyogn", int layer = 0);

  void insertPolygon(IdbRect* poly, const string& name = "polygon", int layer = 0);

  void insertPolygon(IdbRect& poly, const string& name = "polygon", int layer = 0);

  void insertWire(const Point& begin, const Point& end, const int& layer = 0, const int& width = 80);

  void head();
  void tail();
  void topBegin();
  void strBegin();
  void strEnd();

 private:
  fstream _log_ofs;
};

}  // namespace icts