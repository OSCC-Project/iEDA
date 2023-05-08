#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "BinaryTree.h"
#include "CTSAPI.hpp"
#include "CtsConfig.h"
#include "CtsInstance.h"
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
  kTIMING_NODE_LOG = 4,
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