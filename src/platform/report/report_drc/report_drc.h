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
#pragma once

#include <iostream>
#include <map>
#include <ostream>
#include <string>

#include "ReportTable.hh"
#include "report_basic.h"

namespace iplf {

enum class ReportDrcType
{
  kNone = 0,
  kSummary,
  kConnectivity,
  kConnectivityDetail,
  kMax,
};

class ReportDRC : public ReportBase
{
 public:
  explicit ReportDRC(const std::string& report_name) : ReportBase(report_name) {}

  // void createDrcReport()
  // {
  //   idrc::DrcAPIInst.initDRC();
  //   auto result = idrc::DrcAPIInst.getCheckResult();
  //   std::vector<std::string> header = {"DRC Rule", "Count"};
  //   auto drc_tbl = std::make_shared<ieda::ReportTable>("DRC Report", header, -1);
  //   for (auto& [item, count] : result) {
  //     *drc_tbl << item << count << TABLE_ENDLINE;
  //   }
  //   this->add_table(drc_tbl);
  // }
  std::string title() override;

  std::shared_ptr<ieda::ReportTable> createDRCTable(std::map<std::string, int>& drc_result);
  std::shared_ptr<ieda::ReportTable> createConnectivityTable(
      std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int>& connectivity_result);
  std::shared_ptr<ieda::ReportTable> createConnectivityDetailTable(
      std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int>& connectivity_result);
};

}  // namespace iplf