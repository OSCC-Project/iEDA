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
  GDSPloter(){};
  ~GDSPloter(){};
  static void plotDesign(const std::string& path = "");
  static void plotFlyLine(const std::string& path = "");

  static void writePyDesign(const std::string& path = "");
  static void writePyFlyLine(const std::string& path = "");

  static void writeJsonDesign(const std::string& path = "");

  static void plotInstances(std::fstream& log_ofs, vector<CtsInstance*>& insts);
  static void insertInstance(std::fstream& log_ofs, CtsInstance* inst);

  static void refPolygon(std::fstream& log_ofs, const string& name);
  static void refInstance(std::fstream& log_ofs, CtsInstance* inst);

  static void plotPolygons(std::fstream& log_ofs, const std::vector<IdbRect*>& polys, const string& name = "polyogn", int layer = 0);

  static void insertPolygon(std::fstream& log_ofs, IdbRect* poly, const string& name = "polygon", int layer = 0, const int& type = 1);

  static void insertPolygon(std::fstream& log_ofs, IdbRect& poly, const string& name = "polygon", int layer = 0, const int& type = 1);

  static void insertWire(std::fstream& log_ofs, const Point& begin, const Point& end, const string& name = "WIRE", const int& layer = 0,
                         const int& width = 80);

  static void head(std::fstream& log_ofs);
  static void tail(std::fstream& log_ofs);
  static void topBegin(std::fstream& log_ofs);
  static void strBegin(std::fstream& log_ofs);
  static void strEnd(std::fstream& log_ofs);

 private:
};

}  // namespace icts