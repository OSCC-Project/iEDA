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
 * @File Name: report.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-11-07
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "report_basic.h"

#include "Time.hh"
#include "flow_config.h"
#include "idm.h"

namespace iplf {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool ReportBase::add_table(std::shared_ptr<ieda::ReportTable> table)
{
  _table_list.push_back(std::move(table));
  return true;
}

std::shared_ptr<ieda::ReportTable> ReportBase::get_table(int type)
{
  auto it = std::find_if(_table_list.begin(), _table_list.end(),
                         [type](std::shared_ptr<ieda::ReportTable>& rtb) { return rtb->get_type() == type; });
  if (it != _table_list.end()) {
    return *it;
  }
  return {};
}

std::string ReportBase::seperator()
{
  return "###################################################################\n";
}

std::string ReportBase::title()
{
  auto design = dmInst->get_idb_design();
  std::string title;
  std::string name = design->get_design_name();
  std::string version = design->get_version();

  std::vector<std::string> header_list = {"iEDA", flowConfigInst->get_env_info_software_version()};
  auto tbl = std::make_shared<ieda::ReportTable>("Design Info", header_list, 0);

  *tbl << "Stage" << flowConfigInst->get_status_stage() << TABLE_ENDLINE;
  *tbl << "Runtime" << flowConfigInst->get_status_runtime_string() << TABLE_ENDLINE;
  *tbl << "Memmory" << flowConfigInst->get_status_memmory_string() << TABLE_ENDLINE;

  *tbl << TABLE_SKIP << TABLE_SKIP << TABLE_ENDLINE;

  *tbl << "Design Name" << name << TABLE_ENDLINE;
  *tbl << "DEF&LEF Version" << version << TABLE_ENDLINE;
  int dbu = design->get_units()->get_micron_dbu() <= 0 ? design->get_layout()->get_units()->get_micron_dbu()
                                                       : design->get_units()->get_micron_dbu();
  *tbl << "DBU" << dbu << TABLE_ENDLINE;

  return tbl->to_string();
}

std::string ReportBase::timestamp()
{
  // Str::printf is not re-entrant , getNowWallTime will call it, thus we should save it to a string.
  std::string time_str = ieda::Time::getNowWallTime();
  std::string str = ieda::Str::printf("Time : %s\n", time_str.c_str());

  return str;
}

std::ostream& operator<<(std::ostream& ost, ReportBase& report)
{
  ost << report.seperator() << report.timestamp() << report.seperator();
  ost << report.title() << std::endl << report.seperator();
  for (auto& tbl : report.get_table_list()) {
    ost << tbl->get_tbl_name() << std::endl;
    ost << tbl->c_str() << std::endl;
  }
  return ost;
}

}  // namespace iplf
