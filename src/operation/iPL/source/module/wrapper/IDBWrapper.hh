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
 * @Date: 2022-01-21 15:24:11
 * @LastEditTime: 2022-12-10 12:49:56
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/wrapper/IDBWrapper.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_IDB_WRAPPER_H
#define IPL_IDB_WRAPPER_H

#include <map>
#include <string>
#include <vector>

#include "DBWrapper.hh"
#include "Log.hh"
#include "database/IDBWDatabase.hh"

namespace ipl {

using namespace idb;

class IDBWrapper : public DBWrapper
{
 public:
  IDBWrapper() = delete;
  explicit IDBWrapper(IdbBuilder* idb_builder);
  IDBWrapper(const IDBWrapper&) = delete;
  IDBWrapper(IDBWrapper&&) = delete;
  ~IDBWrapper() override;

  IDBWrapper& operator=(const IDBWrapper&) = delete;
  IDBWrapper& operator=(IDBWrapper&&) = delete;

  // Layout.
  const Layout* get_layout() const { return _idbw_database->_layout; }

  // Design.
  Design* get_design() const { return _idbw_database->_design; }

  // function.
  void writeDef(std::string file_name) override;
  void updateFromSourceDataBase() override;
  void updateFromSourceDataBase(std::vector<std::string> inst_list) override;
  void writeBackSourceDatabase() override;
  void initInstancesForFragmentedRow() override;

  // FOR DEBUG.
  void deleteInstsForTest();
  void saveVerilogForDebug(std::string path) override;

 private:
  IDBWDatabase* _idbw_database;

  void wrapIDBData();
  void wrapLayout(IdbLayout* idb_layout);
  void wrapRows(IdbLayout* idb_layout);
  void wrapRoutingInfo(IdbLayout* idb_layout);
  void wrapCells(IdbLayout* idb_layout);
  void wrapDesign(IdbDesign* idb_design);
  void wrapIdbInstance(IdbInstance* idb_inst);
  bool wrapPartOfInstances(std::vector<std::string> inst_list);
  void updatePLInstanceInfo(IdbInstance* idb_inst, Instance* pl_inst);
  void wrapInstances(IdbDesign* idb_design);
  void wrapIdbNet(IdbNet* idb_net);
  bool wrapPartOfNetlists(std::vector<std::string> net_list);
  void updatePLNetInfo(IdbNet* idb_net, Net* pl_net);
  void wrapNetlists(IdbDesign* idb_design);
  Pin* wrapPin(IdbPin* idb_pin);
  void wrapRegions(IdbDesign* idb_design);
  void searchForDontCareNet();

  std::string fixSlash(std::string raw_str);
  std::string fixSlash2(std::string raw_str);
  Point<int32_t> calMedianOffsetFromCellCenter(IdbPin* idb_pin);
  Point<int32_t> calAverageOffsetFromCellCenter(IdbPin* idb_pin);

  bool isCoreOverlap(IdbInstance* idb_inst);
  bool isCrossInst(IdbInstance* idb_inst);
  Rectangle<int32_t> obtainCrossRect(IdbInstance* idb_inst, Rectangle<int32_t> core_shape);
  bool checkInCore(IdbInstance* idb_inst);
  void correctInstanceOrient();
};

}  // namespace ipl

#endif