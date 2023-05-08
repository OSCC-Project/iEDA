#include "report_drc.h"

#include "ReportTable.hh"
#include "idm.h"

namespace iplf {

std::string ReportDRC::title()
{
  return ReportBase::title();
}

std::shared_ptr<ieda::ReportTable> ReportDRC::createDRCTable(std::map<std::string, int>& drc_result)
{
  std::vector<std::string> header_list = {"DRC Type", "Number"};
  auto tbl = std::make_shared<ieda::ReportTable>("Drc Summary", header_list, static_cast<int>(ReportDrcType::kSummary));

  for (auto& [name, nums] : drc_result) {
    *tbl << name << nums << TABLE_ENDLINE;
  }

  return tbl;
}

std::shared_ptr<ieda::ReportTable> ReportDRC::createConnectivityTable(
    std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int>& connectivity_result)
{
  std::vector<std::string> header_list = {"Connectivity Check", "Number"};
  auto tbl = std::make_shared<ieda::ReportTable>("Connectivity Summary", header_list, static_cast<int>(ReportDrcType::kConnectivity));

  auto b_result = std::get<0>(connectivity_result);
  auto disconnect_net_list = std::get<1>(connectivity_result);
  auto one_pin_list = std::get<2>(connectivity_result);
  auto net_max = std::get<3>(connectivity_result);

  if (b_result) {
    *tbl << "Nets are all coonected!" << TABLE_SKIP << TABLE_ENDLINE;
  } else {
    *tbl << "Disconneted nets [pin number >= 2]" << ieda::Str::printf("%d / %d", disconnect_net_list.size(), net_max) << TABLE_ENDLINE;
    *tbl << "Disconneted nets [pin number < 2]" << ieda::Str::printf("%d / %d", one_pin_list.size(), net_max) << TABLE_ENDLINE;
  }

  return tbl;
}

std::shared_ptr<ieda::ReportTable> ReportDRC::createConnectivityDetailTable(
    std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int>& connectivity_result)
{
  auto b_result = std::get<0>(connectivity_result);

  std::vector<std::string>& disconnected_pin_list = std::get<1>(connectivity_result);
  std::vector<std::string>& one_pin_list = std::get<2>(connectivity_result);

  std::vector<std::string> header_list = {"Disconnected Net"};
  auto tbl
      = std::make_shared<ieda::ReportTable>("DRC - Disconnected Net", header_list, static_cast<int>(ReportDrcType::kConnectivityDetail));

  if (b_result) {
    *tbl << "Nets are all coonected!" << TABLE_SKIP << TABLE_ENDLINE;
  } else {
    for (auto& net : disconnected_pin_list) {
      *tbl << net << TABLE_ENDLINE;
    }

    *tbl << TABLE_SKIP << TABLE_ENDLINE;

    for (auto& net : one_pin_list) {
      *tbl << net << TABLE_ENDLINE;
    }
  }

  return tbl;
}

}  // namespace iplf
