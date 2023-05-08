#pragma once
#include <DrcAPI.hpp>
#include <iostream>
#include <map>
#include <ostream>
#include <string>

#include "DrcAPI.hpp"
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

  void createDrcReport()
  {
    idrc::DrcAPIInst.initDRC();
    auto result = idrc::DrcAPIInst.getCheckResult();
    std::vector<std::string> header = {"DRC Rule", "Count"};
    auto drc_tbl = std::make_shared<ieda::ReportTable>("DRC Report", header, -1);
    for (auto& [item, count] : result) {
      *drc_tbl << item << count << TABLE_ENDLINE;
    }
    this->add_table(drc_tbl);
  }
  std::string title() override;

  std::shared_ptr<ieda::ReportTable> createDRCTable(std::map<std::string, int>& drc_result);
  std::shared_ptr<ieda::ReportTable> createConnectivityTable(
      std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int>& connectivity_result);
  std::shared_ptr<ieda::ReportTable> createConnectivityDetailTable(
      std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int>& connectivity_result);
};

}  // namespace iplf