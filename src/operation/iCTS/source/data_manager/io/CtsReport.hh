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
 * @file CtsReport.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "CTSAPI.hh"
#include "CtsConfig.hh"
#include "CtsInstance.hh"
#include "report/ReportTable.hh"

namespace icts {
using std::fstream;
using std::make_pair;
using std::map;
using std::string;
using std::vector;

enum CtsReportType
{
  kWIRE_LENGTH = 0,
  kHP_WIRE_LENGTH = 1,
  kCELL_STATS = 2,
  kLIB_CELL_DIST = 3,
  kNET_LEVEL = 4,
  kTIMING_NODE_LOG = 5,
  kLEVEL_LOG = 6,
  kLEVEL_FANOUT = 7,
  kLEVEL_NET_LEN = 8,
  kLEVEL_CAP = 9,
  kLEVEL_SLEW = 10,
  kLEVEL_DELAY = 11,
  kLEVEL_INSERT_DELAY = 12,
  kLEVEL_SKEW = 13,
};

class CtsReportTable : public ieda::ReportTable
{
 public:
  explicit CtsReportTable(const std::string& tbl_name) : ieda::ReportTable(tbl_name.c_str()) {}
  ~CtsReportTable() override = default;

  using Base = ieda::ReportTable;

  using Base::operator<<;
  using Base::operator[];

  using Base::writeRow;
  using Base::writeRowFromIterator;

  static std::unique_ptr<CtsReportTable> createReportTable(const std::string& tbl_name, const CtsReportType& type);
};

}  // namespace icts