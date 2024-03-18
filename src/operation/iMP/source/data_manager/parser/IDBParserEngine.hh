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

#ifndef IMP_IDB_WRAPPER_H
#define IMP_IDB_WRAPPER_H

#include <map>
#include <string>
#include <vector>

#include "ParserEngine.hh"
namespace idb {
class IdbBuilder;
class IdbInstance;
class IdbPin;
class IdbNet;
class IdbDesign;
class IdbLayout;
}  // namespace idb

namespace imp {
class Layout;
class Net;
class Pin;

class IDBParser final : public ParserEngine
{
 public:
  explicit IDBParser(idb::IdbBuilder* idb_builder);
  IDBParser(const IDBParser&) = delete;
  IDBParser(IDBParser&&) = delete;

  virtual bool read() override;
  virtual bool write() override;

  IDBParser& operator=(const IDBParser&) = delete;
  IDBParser& operator=(IDBParser&&) = delete;
  void setIdbBuilder(idb::IdbBuilder* idb_builder);

 private:
  void initNetlist();
  void initRows();
  void initCells();
  std::shared_ptr<Layout> transform(idb::IdbLayout* idb_layout);
  std::shared_ptr<Instance> transform(idb::IdbInstance* idb_inst);
  std::shared_ptr<Net> transform(idb::IdbNet* idb_net);
  std::shared_ptr<Pin> transform(idb::IdbPin* idb_pin);
  bool read(idb::IdbInstance* idb_inst, std::shared_ptr<Instance> inst);
  bool write(std::shared_ptr<Instance> inst, idb::IdbInstance* idb_inst);

 private:
  idb::IdbBuilder* _idb_builder;
  idb::IdbDesign* _idb_design;
  idb::IdbLayout* _idb_layout;
  std::unordered_map<idb::IdbInstance*, std::shared_ptr<Instance>> _idb2inst;
  std::unordered_map<idb::IdbPin*, std::shared_ptr<Pin>> _idb2pin;
  std::unordered_map<idb::IdbNet*, std::shared_ptr<Net>> _idb2net;

  std::unordered_map<std::shared_ptr<Instance>, idb::IdbInstance*> _inst2idb;
  std::unordered_map<std::shared_ptr<Pin>, idb::IdbPin*> _pin2idb;
  std::unordered_map<std::shared_ptr<Net>, idb::IdbNet*> _net2idb;
};

}  // namespace imp

#endif