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
 * @file CtsDBWrapper.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <utility>

#include "CtsConfig.hh"
#include "CtsDesign.hh"
#include "CtsInstance.hh"
#include "CtsNet.hh"
#include "CtsPin.hh"
#include "IdbDesign.h"
#include "IdbEnum.h"
#include "IdbGeometry.h"
#include "IdbInstance.h"
#include "IdbLayout.h"
#include "IdbNet.h"
#include "IdbPins.h"
#include "IdbRow.h"
#include "IdbSite.h"
#include "IdbTerm.h"
#include "builder.h"
#include "def_service.h"

namespace icts {
using namespace idb;
using std::make_pair;
using std::pair;
using std::unordered_map;

class CtsDBWrapper
{
 public:
  CtsDBWrapper(IdbBuilder* idb);
  CtsDBWrapper(const CtsDBWrapper&) = default;
  ~CtsDBWrapper() {}

  // getter
  IdbBuilder* get_idb() const { return _idb; }
  IdbRect get_bounding_box(CtsInstance* inst) const;
  IdbRect* get_core_bounding_box() const;
  int get_site_width() const;
  int get_site_height() const;
  int get_row_num() const;
  int get_site_num() const;

  Point getPinLoc(CtsPin* pin);

  std::string getPinLayer(CtsPin* pin);

  bool withinCore(const Point& loc) const;

  // setter
  void set_idb(IdbBuilder* idb) { _idb = idb; }

  // read and write file
  void writeDef();

  // read data from idb
  void read();

  // interface to synthesis
  void linkIdb(CtsInstance* inst);
  void updateCell(CtsInstance* inst);

  // the operator of idb
  IdbInstance* makeIdbInstance(CtsInstance* inst);
  IdbNet* makeIdbNet(CtsNet* net);
  bool idbConnect(CtsPin* pin, CtsNet* net);
  bool idbDisconnect(CtsPin* pin);
  void linkInstanceCood(CtsInstance* inst, IdbInstance* idb_inst);
  bool ctsConnect(CtsInstance* inst, CtsPin* pin, CtsNet* net);
  bool ctsDisconnect(CtsPin* pin);

  // create cts inst and idb inst
  CtsInstance* makeInstance(const string& name, const string& cell_name);
  // create cts net and idb net
  CtsNet* makeNet(const string& name);

  IdbInstance* ctsToIdb(CtsInstance* inst);
  IdbPin* ctsToIdb(CtsPin* pin);
  IdbNet* ctsToIdb(CtsNet* net);
  IdbCoordinate<int32_t> ctsToIdb(const Point& loc) const;

  CtsInstance* idbToCts(IdbInstance* idb_inst, bool b_virtual = false);
  CtsInstance* idbToCts(IdbInstance* idb_inst, CtsInstanceType inst_type);
  CtsPin* idbToCts(IdbPin* idb_pin);
  CtsNet* idbToCts(IdbNet* idb_net);
  Point idbToCts(IdbCoordinate<int32_t>& coord) const;

  bool isClockPin(IdbPin* idb_pin) const;
  bool isFlipFlop(IdbInstance* idb_inst) const;
  bool isValidPin(IdbPin* idb_pin) const;

 private:
  IdbRow* findRow(const Point& loc) const;

  void crossRef(CtsNet* cts_net, IdbNet* idb_net);
  void crossRef(CtsPin* cts_pin, IdbPin* idb_pin);
  void crossRef(CtsInstance* cts_inst, IdbInstance* idb_inst);

  CtsPinType idbToCts(IdbConnectType idb_pin_type, IdbConnectDirection idb_pin_direction) const;

 private:
  IdbBuilder* _idb = nullptr;
  IdbDesign* _idb_design = nullptr;
  IdbLayout* _idb_layout = nullptr;

  unordered_map<CtsInstance*, IdbInstance*> _cts2idbInst;
  unordered_map<IdbInstance*, CtsInstance*> _idb2ctsInst;

  unordered_map<CtsPin*, IdbPin*> _cts2idbPin;
  unordered_map<IdbPin*, CtsPin*> _idb2ctsPin;

  unordered_map<CtsNet*, IdbNet*> _cts2idbNet;
  unordered_map<IdbNet*, CtsNet*> _idb2ctsNet;
};

inline void CtsDBWrapper::crossRef(CtsPin* cts_pin, IdbPin* idb_pin)
{
  _cts2idbPin[cts_pin] = idb_pin;
  _idb2ctsPin[idb_pin] = cts_pin;
}

inline void CtsDBWrapper::crossRef(CtsInstance* cts_inst, IdbInstance* idb_inst)
{
  _cts2idbInst[cts_inst] = idb_inst;
  _idb2ctsInst[idb_inst] = cts_inst;
}

inline void CtsDBWrapper::crossRef(CtsNet* cts_net, IdbNet* idb_net)
{
  _cts2idbNet[cts_net] = idb_net;
  _idb2ctsNet[idb_net] = cts_net;
}

}  // namespace icts
