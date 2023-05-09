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
#include "py_report.h"

#include <report_manager.h>

namespace python_interface {
bool reportDbSummary(const std::string& path)
{
  return rptInst->reportDBSummary(path);
}
bool reportWireLength(const std::string& path)
{
  return rptInst->reportWL(path);
}

bool reportCong(const std::string& path)
{
  return rptInst->reportCongestion(path);
}
bool reportDanglingNet(const std::string& path)
{
  return rptInst->reportDanglingNet(path);
}

bool reportRoute(const std::string& path, const std::string& netname, bool summary)
{
  return rptInst->reportRoute(path, netname, summary);
}

bool reportPlaceDistribution(const std::vector<std::string>& prefixes)
{
  return rptInst->reportPlaceDistribution(prefixes);
}

bool reportPrefixedInst(const std::string& prefix, int level, int num_threshold){
  return rptInst->reportInstLevel(prefix, level,  num_threshold);
}

bool reportDRC(const std::string& filename){
  return rptInst->reportDRC(filename);
}
}  // namespace python_interface