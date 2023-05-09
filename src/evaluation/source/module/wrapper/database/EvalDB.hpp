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
#ifndef SRC_EVALUATOR_SOURCE_WRAPPER_DATABASE_EVALDB_HPP_
#define SRC_EVALUATOR_SOURCE_WRAPPER_DATABASE_EVALDB_HPP_

#include "EvalDesign.hpp"
#include "EvalLayout.hpp"
#include "WLNet.hpp"
#include "WLPin.hpp"
#include "builder.h"

namespace eval {

using namespace idb;

class EvalDB
{
 public:
  EvalDB();
  EvalDB(const EvalDB&) = delete;
  EvalDB(EvalDB&&) = delete;
  ~EvalDB();

  EvalDB& operator=(const EvalDB&) = delete;
  EvalDB& operator=(EvalDB&&) = delete;

  IdbBuilder* get_idb_builder() const { return _idb_builder; }
  std::map<IdbPin*, WLPin*> get_pin_map() const { return _idb_pin_map; }
  Design* get_design() const { return _design; }
  Layout* get_layout() const { return _layout; }

 private:
  IdbBuilder* _idb_builder;
  Design* _design;
  Layout* _layout;

  std::map<IdbPin*, WLPin*> _idb_pin_map;
  std::map<IdbNet*, WLNet*> _idb_net_map;

  std::map<WLPin*, IdbPin*> _pin_idb_map;
  std::map<WLNet*, IdbNet*> _net_idb_map;

  friend class DBWrapper;
};

inline EvalDB::EvalDB() : _idb_builder(new IdbBuilder()), _design(new Design()), _layout(new Layout())
{
}

inline EvalDB::~EvalDB()
{
  delete _design;
  delete _layout;
}

}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_WRAPPER_DATABASE_EVALDB_HPP_
