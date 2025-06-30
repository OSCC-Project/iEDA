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
 * @file PwrReport.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief Report power.
 * @version 0.1
 * @date 2023-04-24
 */

#include "PwrReport.hh"

#include "api/Power.hh"
#include "core/PwrAnalysisData.hh"

namespace ipower {
/**
 * @brief create power report table
 *
 * @param tbl_name
 * @return std::unique_ptr<PwrReportTable>
 */
std::unique_ptr<PwrReportTable> PwrReportPowerSummary::createReportTable(const char* tbl_name)
{
  auto report_tbl = std::make_unique<PwrReportTable>(tbl_name);

  (*report_tbl) << TABLE_HEAD;
  /* Fill each cell with operator[] */
  (*report_tbl)[0][0] = "Power Group";
  (*report_tbl)[0][1] = "Internal Power";
  (*report_tbl)[0][2] = "Switch Power";
  (*report_tbl)[0][3] = "Leakage Power";
  (*report_tbl)[0][4] = "Total Power";
  (*report_tbl)[0][5] = "(%)";
  (*report_tbl) << TABLE_ENDLINE;

  return report_tbl;
}

/**
 * @brief report power analysis data
 *
 * @param ipower
 * @return unsigned
 */
unsigned PwrReportPowerSummary::operator()(Power* ipower)
{
  auto& type_to_group_data = ipower->get_type_to_group_data();
  double net_switching_power = 0.0;
  double cell_internal_power = 0.0;
  double cell_leakage_power = 0.0;

  // calc group summary data.
  for (auto& [group_type, group_datas] : type_to_group_data) {
    double group_internal_power = 0.0;
    double group_switch_power = 0.0;
    double group_leakage_power = 0.0;

    for (auto& group_data : group_datas) {
      group_internal_power += group_data->get_internal_power();
      group_switch_power += group_data->get_switch_power();
      group_leakage_power += group_data->get_leakage_power();

      cell_internal_power += group_data->get_internal_power();
      net_switching_power += group_data->get_switch_power();
      cell_leakage_power += group_data->get_leakage_power();
    }

    auto report_group_data
        = std::make_unique<PwrReportGroupSummaryData>(group_type, group_internal_power, group_switch_power, group_leakage_power);
    _report_summary_data.add_report_group_data(std::move(report_group_data));
  }

  _report_summary_data.set_net_switching_power(net_switching_power);
  _report_summary_data.set_cell_internal_power(cell_internal_power);
  _report_summary_data.set_cell_leakage_power(cell_leakage_power);

  double total_power = cell_internal_power + net_switching_power + cell_leakage_power;
  _report_summary_data.set_total_power(total_power);

  return 1;
}
}  // namespace ipower