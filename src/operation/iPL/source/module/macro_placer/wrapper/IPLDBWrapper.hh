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
#pragma once

#include <map>
#include <string>
#include <vector>

#include "DBWrapper.hh"
#include "IPLDBWDatabase.hh"

namespace ipl::imp {

class IPLDBWrapper : public DBWrapper
{
 public:
  IPLDBWrapper() = delete;
  explicit IPLDBWrapper(ipl::PlacerDB* ipl_db);
  ~IPLDBWrapper() override;

  // Layout
  const FPLayout* get_layout() const { return _iplw_database->_layout; }

  // Design
  FPDesign* get_design() const { return _iplw_database->_design; }

  // Function
  void writeDef(string file_name) override{};
  void writeBackSourceDataBase() override;

 private:
  IPLDBWDatabase* _iplw_database;

  void wrapIPLData();
  void wrapLayout(const ipl::Layout* ipl_layout);
  void wrapDesign(ipl::Design* ipl_design);
  void wrapInstancelist(ipl::Design* ipl_design);
  void wrapNetlist(ipl::Design* ipl_design);
  FPPin* wrapPin(ipl::Pin* ipl_pin);
};
}  // namespace ipl::imp
