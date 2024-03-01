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
/*
 * @Author: S.J Chen
 * @Date: 2022-02-16 20:03:35
 * @LastEditTime: 2022-12-11 20:15:52
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/wrapper/DBWrapper.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_DB_WRAPPER_H
#define IPL_DB_WRAPPER_H

#include <string>

#include "data/Design.hh"
#include "data/Layout.hh"

namespace ipl {

class DBWrapper
{
 public:
  DBWrapper() = default;
  DBWrapper(const DBWrapper&) = delete;
  DBWrapper(DBWrapper&&) = delete;
  virtual ~DBWrapper() = default;

  DBWrapper& operator=(const DBWrapper&) = delete;
  DBWrapper& operator=(DBWrapper&&) = delete;

  // Layout.
  virtual const Layout* get_layout() const = 0;

  // Design.
  virtual Design* get_design() const = 0;

  // Function.
  virtual void writeDef(std::string file_name) = 0;
  virtual void updateFromSourceDataBase() = 0;
  virtual void updateFromSourceDataBase(std::vector<std::string> inst_list) = 0;
  virtual void writeBackSourceDatabase() = 0;
  virtual void initInstancesForFragmentedRow() = 0;
  virtual void saveVerilogForDebug(std::string path) = 0;
};

}  // namespace ipl

#endif