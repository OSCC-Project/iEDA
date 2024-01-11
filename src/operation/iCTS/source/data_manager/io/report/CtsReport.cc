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
 * @file CtsReport.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "CtsReport.hh"

namespace icts {

std::unique_ptr<CtsReportTable> CtsReportTable::createReportTable(const std::string& tbl_name, const CtsReportType& type)
{
  auto report_tbl = std::make_unique<CtsReportTable>(tbl_name);

  switch (type) {
    case CtsReportType::kWireLength:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Type";
      (*report_tbl)[0][1] = "Wire Length (um)";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kHpWireLength:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Type";
      (*report_tbl)[0][1] = "HP Wire Length (um)";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kCellStatus:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Cell type";
      (*report_tbl)[0][1] = "Count";
      (*report_tbl)[0][2] = "Area (um^2)";
      (*report_tbl)[0][3] = "Capacitance (pF)";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kLibCellDist:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Name";
      (*report_tbl)[0][1] = "Type";
      (*report_tbl)[0][2] = "Inst\nCount";
      (*report_tbl)[0][3] = "Inst Area\n(um^2)";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kNetLevel:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Level";
      (*report_tbl)[0][1] = "Num";
      (*report_tbl)[0][2] = "Ratio";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kLevelLog:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Level";
      (*report_tbl)[0][1] = "Inst Num";
      (*report_tbl)[0][2] = "Avg Fanout";
      (*report_tbl)[0][3] = "Max Delay (ns)";
      (*report_tbl)[0][4] = "Max Skew (ns)";
      (*report_tbl)[0][5] = "Max Insert Delay (ns)";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kLevelFanout:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Level";
      (*report_tbl)[0][1] = "Inst Num";
      (*report_tbl)[0][2] = "Min Fanout";
      (*report_tbl)[0][3] = "Max Fanout";
      (*report_tbl)[0][4] = "Avg Fanout";
      (*report_tbl)[0][5] = "Violation";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kLevelNetLen:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Level";
      (*report_tbl)[0][1] = "Inst Num";
      (*report_tbl)[0][2] = "Min Net Length (um)";
      (*report_tbl)[0][3] = "Max Net Length (um)";
      (*report_tbl)[0][4] = "Avg Net Length (um)";
      (*report_tbl)[0][5] = "Violation";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kLevelCap:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Level";
      (*report_tbl)[0][1] = "Inst Num";
      (*report_tbl)[0][2] = "Min Cap (pF)";
      (*report_tbl)[0][3] = "Max Cap (pF)";
      (*report_tbl)[0][4] = "Avg Cap (pF)";
      (*report_tbl)[0][5] = "Violation";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kLevelSlew:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Level";
      (*report_tbl)[0][1] = "Inst Num";
      (*report_tbl)[0][2] = "Min Slew (ns)";
      (*report_tbl)[0][3] = "Max Slew (ns)";
      (*report_tbl)[0][4] = "Avg Slew (ns)";
      (*report_tbl)[0][5] = "Violation";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kLevelDelay:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Level";
      (*report_tbl)[0][1] = "Inst Num";
      (*report_tbl)[0][2] = "Min Delay (ns)";
      (*report_tbl)[0][3] = "Max Delay (ns)";
      (*report_tbl)[0][4] = "Avg Delay (ns)";
      (*report_tbl)[0][5] = "Violation";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kLevelInsertDelay:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Level";
      (*report_tbl)[0][1] = "Inst Num";
      (*report_tbl)[0][2] = "Min Insert Delay (ns)";
      (*report_tbl)[0][3] = "Max Insert Delay (ns)";
      (*report_tbl)[0][4] = "Avg Insert Delay (ns)";
      (*report_tbl)[0][5] = "Violation";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kLevelSkew:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Level";
      (*report_tbl)[0][1] = "Inst Num";
      (*report_tbl)[0][2] = "Min Skew (ns)";
      (*report_tbl)[0][3] = "Max Skew (ns)";
      (*report_tbl)[0][4] = "Avg Skew (ns)";
      (*report_tbl)[0][5] = "Violation";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    default:
      break;
  }
  return report_tbl;
}
}  // namespace icts