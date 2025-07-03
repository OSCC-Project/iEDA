/**
 * @file PwrReportInstance.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief
 * @version 0.1
 * @date 2023-11-30
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "PwrReportInstance.hh"

namespace ipower {

/**
 * @brief create power report table
 *
 * @param tbl_name
 * @return std::unique_ptr<PwrReportTable>
 */
std::unique_ptr<PwrReportInstanceTable> PwrReportInstance::createReportTable(
    const char* tbl_name) {
  auto report_tbl = std::make_unique<PwrReportInstanceTable>(tbl_name);

  (*report_tbl) << TABLE_HEAD;
  /* Fill each cell with operator[] */
  (*report_tbl)[0][0] = "Instance Name";
  (*report_tbl)[0][1] = "Nominal Voltage";
  (*report_tbl)[0][2] = "Internal Power";
  (*report_tbl)[0][3] = "Switch Power";
  (*report_tbl)[0][4] = "Leakage Power";
  (*report_tbl)[0][5] = "Total Power";
  (*report_tbl) << TABLE_ENDLINE;

  return report_tbl;
}

}  // namespace ipower