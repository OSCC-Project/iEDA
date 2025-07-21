// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file TimingIDBAdapter.hh
 * @author longshy (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-10-11
 */

#pragma once

#include <string>

#include "IdbCellMaster.h"
#include "IdbDesign.h"
#include "IdbEnum.h"
#include "IdbGeometry.h"
#include "IdbInstance.h"
#include "IdbLayer.h"
#include "IdbLayout.h"
#include "IdbNet.h"
#include "IdbPins.h"
#include "TimingDBAdapter.hh"
#include "builder.h"
#include "def_service.h"
#include "lef_service.h"
#include "sta/Sta.hh"

namespace ista {

using idb::IdbBuilder;
using idb::IdbCellMaster;
using idb::IdbCellMasterList;
using idb::IdbConnectDirection;
using idb::IdbConnectType;
using idb::IdbCoordinate;
using idb::IdbDefService;
using idb::IdbDesign;
using idb::IdbInstance;
using idb::IdbLayer;
using idb::IdbLayerRouting;
using idb::IdbLayout;
using idb::IdbLefService;
using idb::IdbNet;
using idb::IdbSpecialNet;
using idb::IdbNetList;
using idb::IdbPin;
using idb::IdbPlacementStatus;
using idb::IdbTerm;

/**
 * @brief The adapter for iDB converted from/to ista netlist.
 *
 */
class TimingIDBAdapter : public TimingDBAdapter {
 public:
  explicit TimingIDBAdapter(Sta* ista) : TimingDBAdapter(ista) {
    _debug_csv_file.open("debug_timing_idb.csv", std::ios_base::trunc);
    _debug_csv_file << "lef_resistance,segment_length,segment_width,layer,segment_resistance"
                    << std::endl;
  }
  ~TimingIDBAdapter() override {
    _debug_csv_file.close();
  }

  void set_idb(IdbBuilder* idb) {
    _idb = idb;
    _idb_def_service = idb->get_def_service();
    _idb_lef_service = idb->get_lef_service();
    _idb_design = _idb_def_service->get_design();
  }
  IdbBuilder* get_idb() const { return _idb; }

  void set_dbu(int dbu) { _dbu = dbu; }
  int get_dbu() const { return _dbu; }

  bool isPlaced(DesignObject* pin_or_port) override;
  double dbuToMeters(int distance) const override;

  void location(DesignObject* pin_or_port,
                // Return values.
                double& x, double& y, bool& exists);

  IdbCoordinate<int32_t>* idbLocation(DesignObject* pin_or_port);

  double getResistance(int num_layer, double segment_length,
                       std::optional<double> segment_width);
  double getCapacitance(int num_layer, double segment_length,
                        std::optional<double> segment_width);
  double getAverageResistance(std::optional<double>& segment_width);
  double getAverageCapacitance(std::optional<double>& segment_width);

  double capToLength(int num_layer, double cap,
                     std::optional<double>& segment_width);

  PortDir dbToSta(IdbConnectType sig_type, IdbConnectDirection io_type) const;

  Instance* dbToSta(IdbInstance* db_inst) {
    return _db2staInst.contains(db_inst) ? _db2staInst[db_inst] : nullptr;
  }
  IdbInstance* staToDb(Instance* inst) {
    return _sta2dbInst.contains(inst) ? _sta2dbInst[inst] : nullptr;
  }

  Port* dbToStaPort(IdbPin* db_port) {
    return _db2staPort.contains(db_port) ? _db2staPort[db_port] : nullptr;
  }
  IdbPin* staToDb(Port* port) {
    return _sta2dbPort.contains(port) ? _sta2dbPort[port] : nullptr;
  }

  Net* dbToSta(IdbNet* db_net) {
    return _db2staNet.contains(db_net) ? _db2staNet[db_net] : nullptr;
  }
  IdbNet* staToDb(Net* net) {
    return _sta2dbNet.contains(net) ? _sta2dbNet[net] : nullptr;
  }

  Pin* dbToStaPin(IdbPin* db_pin) {
    return _db2staPin.contains(db_pin) ? _db2staPin[db_pin] : nullptr;
  }
  IdbPin* staToDb(Pin* pin) {
    return _sta2dbPin.contains(pin) ? _sta2dbPin[pin] : nullptr;
  }

  LibCell* dbToSta(IdbCellMaster* master);
  IdbCellMaster* staToDb(const LibCell* cell) const;

  LibPort* dbToSta(IdbTerm* idb_term);
  IdbTerm* staToDb(LibPort* port) const;

  ////////////////////////Edit functions////////////////////////////////

  virtual Instance* createInstance(LibCell* cell, const char* name);
  void deleteInstance(const char* instance_name);
  virtual void substituteCell(Instance* inst, LibCell* cell);

  Pin* attach(Instance* inst, const char* port_name, Net* net);
  Port* attach(Port* port, const char* port_name, Net* net);
  void disattachPin(Pin* pin);
  void disattachPinPort(DesignObject* pin_or_port);
  void reattachPin(Net* net, Pin* old_connect_pin,
                   std::vector<Pin*> new_connect_pins);

  Net* createNet(const char* name, Instance* parent);
  Net* createNet(const char* name, Instance* parent,
                 idb::IdbConnectType connect_type);
  Net* createNet(const char* name, std::vector<std::string>& sink_pin_list,
                 idb::IdbConnectType connect_type);
  void deleteNet(Net* sta_net);

  void crossRef(Instance* sta_inst, IdbInstance* db_inst) {
    _sta2dbInst[sta_inst] = db_inst;
    _db2staInst[db_inst] = sta_inst;
  }

  void removeCrossRef(Instance* sta_inst, IdbInstance* db_inst) {
    _sta2dbInst.erase(sta_inst);
    _db2staInst.erase(db_inst);
  }

  void crossRef(Port* sta_port, IdbPin* db_port) {
    _sta2dbPort[sta_port] = db_port;
    _db2staPort[db_port] = sta_port;
  }

  void crossRef(Net* sta_net, IdbNet* db_net) {
    _sta2dbNet[sta_net] = db_net;
    _db2staNet[db_net] = sta_net;
  }

  void removeCrossRef(Net* sta_net, IdbNet* db_net) {
    _sta2dbNet.erase(sta_net);
    _db2staNet.erase(db_net);
  }

  void crossRef(Net* sta_net, IdbSpecialNet* db_net) {
    _sta2dbSpecialNet[sta_net] = db_net;
    _db2staSpecialNet[db_net] = sta_net;
  }

  void removeCrossRef(Net* sta_net, IdbSpecialNet* db_net) {
    _sta2dbSpecialNet.erase(sta_net);
    _db2staSpecialNet.erase(db_net);
  }

  void crossRef(Pin* sta_pin, IdbPin* db_pin) {
    _sta2dbPin[sta_pin] = db_pin;
    _db2staPin[db_pin] = sta_pin;
  }
  void removeCrossRef(Pin* sta_pin, IdbPin* db_pin) {
    _sta2dbPin.erase(sta_pin);
    _db2staPin.erase(db_pin);
  }

  void configStaLinkCells();
  unsigned convertDBToTimingNetlist(bool link_all_cell = false) override;

 private:
  unsigned makeTopCell();  // to do
  std::string changeStaBusNetNameToIdb(std::string sta_net_name);

  IdbBuilder* _idb = nullptr;
  IdbDesign* _idb_design = nullptr;
  int _dbu = 2000;
  IdbDefService* _idb_def_service = nullptr;
  IdbLefService* _idb_lef_service = nullptr;

  FlatMap<IdbInstance*, Instance*> _db2staInst;
  FlatMap<Instance*, IdbInstance*> _sta2dbInst;

  FlatMap<IdbPin*, Port*> _db2staPort;  // net: get io pin
  FlatMap<Port*, IdbPin*> _sta2dbPort;

  FlatMap<IdbNet*, Net*> _db2staNet;
  FlatMap<Net*, IdbNet*> _sta2dbNet;

  FlatMap<IdbSpecialNet*, Net*> _db2staSpecialNet;
  FlatMap<Net*, IdbSpecialNet*> _sta2dbSpecialNet;

  FlatMap<IdbPin*, Pin*> _db2staPin;  // net: get instance_pin
  FlatMap<Pin*, IdbPin*> _sta2dbPin;

  std::ofstream _debug_csv_file;
};

}  // namespace ista
