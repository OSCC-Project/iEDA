/*
 * @Author: S.J Chen
 * @Date: 2022-01-26 16:29:32
 * @LastEditTime: 2022-10-27 19:19:25
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/wrapper/database/IDBWDatabase.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_WRAPPER_IDBWDATABASE_H
#define IPL_WRAPPER_IDBWDATABASE_H

#include "builder.h"
#include "data/Design.hh"
#include "data/Instance.hh"
#include "data/Layout.hh"
#include "data/Net.hh"
#include "data/Pin.hh"

namespace ipl {

using namespace idb;

class IDBWDatabase
{
 public:
  IDBWDatabase();
  IDBWDatabase(const IDBWDatabase&) = delete;
  IDBWDatabase(IDBWDatabase&&)      = delete;
  ~IDBWDatabase();

  IDBWDatabase& operator=(const IDBWDatabase&) = delete;
  IDBWDatabase& operator=(IDBWDatabase&&)      = delete;

  // tmp for gtest.
  std::map<IdbInstance*, Instance*> get_inst_map() const { return _ipl_inst_map; }
  std::map<IdbPin*, Pin*>           get_pin_map() const { return _ipl_pin_map; }
  IdbBuilder*                       get_idb_builder() const { return _idb_builder; }

 private:
  IdbBuilder* _idb_builder;

  Layout* _layout;
  Design* _design;

  std::map<IdbInstance*, Instance*> _ipl_inst_map;
  std::map<IdbPin*, Pin*>           _ipl_pin_map;
  std::map<IdbNet*, Net*>           _ipl_net_map;

  std::map<Instance*, IdbInstance*> _idb_inst_map;
  std::map<Pin*, IdbPin*>           _idb_pin_map;
  std::map<Net*, IdbNet*>           _idb_net_map;

  friend class IDBWrapper;
};

inline IDBWDatabase::IDBWDatabase() : _idb_builder(new IdbBuilder()), _layout(new Layout()), _design(new Design())
{
}

inline IDBWDatabase::~IDBWDatabase()
{
  delete _layout;
  delete _design;
}

}  // namespace ipl

#endif