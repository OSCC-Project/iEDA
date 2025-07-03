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
 * @file PwrReport.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief Report power.
 * @version 0.1
 * @date 2023-04-24
 */

#pragma once

#include <memory>

#include "core/PwrGroupData.hh"
#include "include/PwrType.hh"
#include "report/ReportTable.hh"
#include "string/Str.hh"

#define TABLE_HEAD fort::header
#define TABLE_ENDLINE fort::endr
#define TABLE_SKIP ""

namespace ipower {
class Power;

using ieda::Str;

/**
 * @brief report table class.
 *
 */
class PwrReportTable : public ieda::ReportTable {
 public:
  explicit PwrReportTable(const char* table_name)
      : ieda::ReportTable(table_name) {}
  ~PwrReportTable() override = default;

  using Base = ieda::ReportTable;

  using Base::operator<<;
  using Base::operator[];

  using Base::writeRow;
  using Base::writeRowFromIterator;
};

/**
 * @brief the class of report group summary data.
 *
 */
class PwrReportGroupSummaryData {
 public:
  PwrReportGroupSummaryData(PwrGroupData::PwrGroupType group_type,
                            double internal_power, double switch_power,
                            double leakage_power)
      : _group_type(group_type),
        _internal_power(internal_power),
        _switch_power(switch_power),
        _leakage_power(leakage_power) {
    _total_power = internal_power + switch_power + leakage_power;
  }
  ~PwrReportGroupSummaryData() = default;

  [[nodiscard]] double get_internal_power() const { return _internal_power; }
  [[nodiscard]] double get_switch_power() const { return _switch_power; }
  [[nodiscard]] double get_leakage_power() const { return _leakage_power; }
  [[nodiscard]] double get_total_power() const { return _total_power; }
  [[nodiscard]] auto& get_group_type() const { return _group_type; }

 private:
  PwrGroupData::PwrGroupType _group_type;
  double _internal_power = 0.0;
  double _switch_power = 0.0;
  double _leakage_power = 0.0;
  double _total_power = 0.0;
};

/**
 * @brief the class of report summary data.
 *
 */
class PwrReportSummaryData {
 public:
  void set_net_switching_power(double net_switching_power) {
    _net_switching_power = net_switching_power;
  }
  [[nodiscard]] double get_net_switching_power() const {
    return _net_switching_power;
  }

  void set_cell_internal_power(double cell_internal_power) {
    _cell_internal_power = cell_internal_power;
  }
  [[nodiscard]] double get_cell_internal_power() const {
    return _cell_internal_power;
  }

  void set_cell_leakage_power(double cell_leakage_power) {
    _cell_leakage_power = cell_leakage_power;
  }
  [[nodiscard]] double get_cell_leakage_power() const {
    return _cell_leakage_power;
  }

  void set_total_power(double total_power) { _total_power = total_power; }
  [[nodiscard]] double get_total_power() const { return _total_power; }

  void add_report_group_data(
      std::unique_ptr<PwrReportGroupSummaryData> report_group_data) {
    _report_group_datas.emplace_back(std::move(report_group_data));
  }
  [[nodiscard]] auto& get_report_group_data() const {
    return _report_group_datas;
  }

 private:
  std::vector<std::unique_ptr<PwrReportGroupSummaryData>> _report_group_datas;
  double _net_switching_power;
  double _cell_internal_power;
  double _cell_leakage_power;
  double _total_power;
};

/**
 * @brief The macro of foreach report group data, usage:
 * PwrReportSummaryData* report_summary_data;
 * PwrReportGroupSummaryData* report_group_data;
 * FOREACH_REPORT_GROUP_DATA(report_summary_data, report_group_data)
 * {
 *    do_something_for_report_group_data();
 * }
 */
#define FOREACH_REPORT_GROUP_DATA(report_summary_data, report_group_data) \
  if (auto& report_group_datas =                                          \
          (report_summary_data)->get_report_group_data();                 \
      !report_group_datas.empty())                                        \
    for (auto p = report_group_datas.begin();                             \
         p != report_group_datas.end() ? report_group_data = p->get(),    \
              true                     : false;                           \
         ++p)

/**
 * @brief The report power function class.
 *
 */
class PwrReportPowerSummary {
 public:
  PwrReportPowerSummary(const char* rpt_file_name,
                        PwrAnalysisMode analysis_mode)
      : _rpt_file_name(Str::copy(rpt_file_name)),
        _analysis_mode(analysis_mode){};
  virtual ~PwrReportPowerSummary() { Str::free(_rpt_file_name); }

  static std::unique_ptr<PwrReportTable> createReportTable(
      const char* tbl_name);
  unsigned operator()(Power* ipower);
  auto& get_report_summary_data() { return _report_summary_data; }

 private:
  const char* _rpt_file_name;                 //!< The report file name.
  PwrAnalysisMode _analysis_mode;             //!< The power analysis mode.
  PwrReportSummaryData _report_summary_data;  //!> the report summary data.
};

}  // namespace ipower
