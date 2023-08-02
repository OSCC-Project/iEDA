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
 * @Date: 2022-01-21 14:33:51
 * @LastEditTime: 2023-02-22 11:32:34
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @Description:
 * @FilePath: /irefactor/src/operation/iPL/source/DataManager.hh
 * Contact : https://github.com/sjchanson
 */
#ifndef IMP_DATA_MANAGER
#define IMP_DATA_MANAGER
#include <string>
#include <vector>
// #include "DBWrapper.hh"
// #include "Design.hh"
// #include "Layout.hh"
namespace imp {
class DBWrapper;
class Layout;
class Design;

class DataManager
{
 public:
  explicit DataManager();
  ~DataManager();
  void readFormLefDef(const std::string& json_path);
  void readFormBookshell();
  void setDbWrapper(DBWrapper* db_wrapper);
  // Layout.
  const Layout* get_layout() const;

  // Design.
  Design* get_design() const;

  // Manager.
  // Function.
  void printDataManager() const;
  void printLayoutInfo() const;
  void printInstanceInfo() const;
  void printNetInfo() const;
  void printPinInfo() const;
  void printRegionInfo() const;

  void updateFromSourceDataBase();
  // void updateFromSourceDataBase(std::vector<std::string> inst_list);
  // void updateInstancesForDebug(std::vector<Instance*> inst_list);

  float obtainUtilization();

  // void saveVerilogForDebug(std::string path);

  // void writeBackSourceDataBase() { _db_wrapper->writeBackSourceDatabase(); }
  // void writeDef(std::string file_name) { _db_wrapper->writeDef(file_name); }

  bool isInitialized() { return _db_wrapper != nullptr; }

 private:
  DBWrapper* _db_wrapper;

  DataManager(const DataManager&) = delete;
  DataManager(DataManager&&) = delete;
  DataManager& operator=(const DataManager&) = delete;
  DataManager& operator=(DataManager&&) = delete;
};
}  // namespace imp
#endif