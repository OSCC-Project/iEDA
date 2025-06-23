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

#include "DbInterface.h"

namespace ino {
using idb::IdbBuilder;
using idb::IdbCellMaster;
using idb::IdbCellMasterList;
using idb::IdbDesign;
using idb::IdbInstance;
using idb::IdbLayout;
using idb::IdbNet;
using idb::IdbPin;
using idb::IdbPins;
using idb::IdbRow;

class FixFanout {
 public:
  FixFanout(ino::DbInterface *db_interface);
  ~FixFanout() = default;

  void fixFanout();

 private:
  void checkFanout() {}

  void fixFanout(IdbNet *net);

  IdbNet *makeNet(const char *name);

  IdbInstance *makeInstance(string master_name, string inst_name);

  void disconnectPin(IdbPin *dpin, IdbNet *dnet);

  void connect(IdbInstance *dinst, IdbPin *dpin, IdbNet *dnet);

  /* data */
  ino::DbInterface *_db_interface;
  TimingEngine     *_timing_engine;
  IdbBuilder       *_idb;
  IdbDesign        *_idb_design;
  IdbLayout        *_idb_layout;

  int _insert_instance_index = 1;
  int _make_net_index = 1;

  int _fanout_vio_num = 0;

  int _max_fanout = 30;
};

} // namespace ino
