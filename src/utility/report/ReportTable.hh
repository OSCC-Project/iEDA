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
 * @file ReportTable.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of report table.
 * @version 0.1
 * @date 2022-08-03
 */
#pragma once
#include <vector>

#include "libfort/fort.hpp"

#define TABLE_HEAD fort::header
#define TABLE_ENDLINE fort::endr
#define TABLE_SKIP ""

namespace ieda {
class ReportTable : public fort::char_table
{
 public:
  explicit ReportTable(const char* table_name, int type = 0) : _tbl_name(table_name), _type(type) {}
  explicit ReportTable(const char* table_name, std::vector<std::string> header_list, int type = 0)
      : _tbl_name(table_name), _type(type), _header_list(header_list)
  {
    initHeader();
  }
  ~ReportTable() override = default;

  const char* get_tbl_name() { return _tbl_name.c_str(); }
  int get_type() { return _type; }

  using Base = fort::char_table;

  using Base::operator<<;
  using Base::operator[];

  template <typename T, typename... Ts>
  unsigned writeRow(const T& str, const Ts&... strings);
  template <typename InputIt>
  unsigned writeRowFromIterator(InputIt first, InputIt last);

 private:
  std::string _tbl_name;  //!< The report table name.
  int _type = 0;          //!< tablel type
  std::vector<std::string> _header_list;

  /// init header
  void initHeader();
};

}  // namespace ieda