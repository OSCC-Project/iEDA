/**
 * @file PwrReportInstance.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief
 * @version 0.1
 * @date 2023-11-29
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <memory>

#include "core/PwrGroupData.hh"
#include "include/PwrType.hh"
#include "report/ReportTable.hh"
#include "string/Str.hh"

namespace ipower {
class Power;
using ieda::Str;

/**
 * @brief report table class.
 *
 */
class PwrReportInstanceTable : public ieda::ReportTable {
 public:
  explicit PwrReportInstanceTable(const char* table_name)
      : ieda::ReportTable(table_name) {}
  ~PwrReportInstanceTable() override = default;

  using Base = ieda::ReportTable;

  using Base::operator<<;
  using Base::operator[];

  using Base::writeRow;
  using Base::writeRowFromIterator;
};

/**
 * @brief report instance power
 *
 */
class PwrReportInstance {
 public:
  PwrReportInstance(const char* rpt_file_name, PwrAnalysisMode analysis_mode)
      : _rpt_file_name(Str::copy(rpt_file_name)),
        _analysis_mode(analysis_mode){};
  virtual ~PwrReportInstance() { Str::free(_rpt_file_name); }
  static std::unique_ptr<PwrReportInstanceTable> createReportTable(
      const char* tbl_name);
  unsigned operator()(Power* ipower);

 private:
  const char* _rpt_file_name;      //!< The report file name.
  PwrAnalysisMode _analysis_mode;  //!< The power analysis mode.
};
}  // namespace ipower
