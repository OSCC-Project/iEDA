#include "CtsReport.h"

namespace icts {

std::unique_ptr<CtsReportTable> CtsReportTable::createReportTable(const std::string& tbl_name, const CtsReportType& type)
{
  auto report_tbl = std::make_unique<CtsReportTable>(tbl_name);

  switch (type) {
    case CtsReportType::kWIRE_LENGTH:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Type";
      (*report_tbl)[0][1] = "Wire Length";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kHP_WIRE_LENGTH:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Type";
      (*report_tbl)[0][1] = "HP Wire Length";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kCELL_STATS:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Cell type";
      (*report_tbl)[0][1] = "Count";
      (*report_tbl)[0][2] = "Area";
      (*report_tbl)[0][3] = "Capacitance";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kLIB_CELL_DIST:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "Name";
      (*report_tbl)[0][1] = "Type";
      (*report_tbl)[0][2] = "Inst\nCount";
      (*report_tbl)[0][3] = "Inst Area\n(um^2)";
      (*report_tbl) << TABLE_ENDLINE;
      break;
    case CtsReportType::kTIMING_NODE_LOG:
      (*report_tbl) << TABLE_HEAD;
      (*report_tbl)[0][0] = "ID";
      (*report_tbl)[0][1] = "Name";
      (*report_tbl)[0][2] = "Snake";
      (*report_tbl)[0][3] = "Net Length";
      (*report_tbl)[0][4] = "Location";
      (*report_tbl)[0][5] = "Min Delay";
      (*report_tbl)[0][6] = "Max Delay";
      (*report_tbl)[0][7] = "Insertion Type";
      (*report_tbl)[0][8] = "Slew In";
      (*report_tbl)[0][9] = "Cap Out";
      (*report_tbl)[0][10] = "Insertion Delay";
      (*report_tbl) << TABLE_ENDLINE;
      break;

    default:
      break;
  }
  return report_tbl;
}
}  // namespace icts