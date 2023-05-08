/**
 * @file TimingIDBAdapter.hh
 * @author shy long (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-10-11
 *
 * @copyright Copyright (c) 2021
 *
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
  explicit TimingIDBAdapter(Sta* ista) : TimingDBAdapter(ista) {}
  ~TimingIDBAdapter() override = default;

  void set_idb(IdbBuilder* idb) {
    _idb = idb;
    _idb_def_service = idb->get_def_service();
    _idb_lef_service = idb->get_lef_service();
    _idb_design = _idb_def_service->get_design();
  }
  IdbBuilder* get_idb() const { return _idb; }

  bool isPlaced(DesignObject* pin_or_port) override;
  double dbuToMeters(int distance) const override;

  void location(DesignObject* pin_or_port,
                // Return values.
                double& x, double& y, bool& exists);

  IdbCoordinate<int32_t>* idbLocation(DesignObject* pin_or_port);

  double getResistance(int num_layer, double segment_length,
                       std::optional<double>& segment_width);
  double getCapacitance(int num_layer, double segment_length,
                        std::optional<double>& segment_width);

  double capToLength(int num_layer, double cap,
                     std::optional<double>& segment_width);

  PortDir dbToSta(IdbConnectType sig_type, IdbConnectDirection io_type) const;

  Instance* dbToSta(IdbInstance* db_inst) {
    return _db2staInst.hasKey(db_inst) ? _db2staInst[db_inst] : nullptr;
  }
  IdbInstance* staToDb(Instance* inst) {
    return _sta2dbInst.hasKey(inst) ? _sta2dbInst[inst] : nullptr;
  }

  Port* dbToStaPort(IdbPin* db_port) {
    return _db2staPort.hasKey(db_port) ? _db2staPort[db_port] : nullptr;
  }
  IdbPin* staToDb(Port* port) {
    return _sta2dbPort.hasKey(port) ? _sta2dbPort[port] : nullptr;
  }

  Net* dbToSta(IdbNet* db_net) {
    return _db2staNet.hasKey(db_net) ? _db2staNet[db_net] : nullptr;
  }
  IdbNet* staToDb(Net* net) {
    return _sta2dbNet.hasKey(net) ? _sta2dbNet[net] : nullptr;
  }

  Pin* dbToStaPin(IdbPin* db_pin) {
    return _db2staPin.hasKey(db_pin) ? _db2staPin[db_pin] : nullptr;
  }
  IdbPin* staToDb(Pin* pin) {
    return _sta2dbPin.hasKey(pin) ? _sta2dbPin[pin] : nullptr;
  }

  LibertyCell* dbToSta(IdbCellMaster* master);
  IdbCellMaster* staToDb(const LibertyCell* cell) const;

  LibertyPort* dbToSta(IdbTerm* idb_term);
  IdbTerm* staToDb(LibertyPort* port) const;

  ////////////////////////Edit functions////////////////////////////////

  virtual Instance* makeInstance(LibertyCell* cell, const char* name);
  void removeInstance(const char* instance_name);
  virtual void replaceCell(Instance* inst, LibertyCell* cell);

  Pin* connect(Instance* inst, const char* port_name, Net* net);
  Port* connect(Port* port, const char* port_name, Net* net);
  void disconnectPin(Pin* pin);
  void disconnectPinPort(DesignObject* pin_or_port);
  void reconnectPin(Net* net, Pin* old_connect_pin,
                    std::vector<Pin*> new_connect_pins);

  Net* makeNet(const char* name, Instance* parent);
  Net* makeNet(const char* name, Instance* parent,
               idb::IdbConnectType connect_type);
  Net* makeNet(const char* name, std::vector<std::string>& sink_pin_list,
               idb::IdbConnectType connect_type);
  void removeNet(Net* sta_net);

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

  void crossRef(Pin* sta_pin, IdbPin* db_pin) {
    _sta2dbPin[sta_pin] = db_pin;
    _db2staPin[db_pin] = sta_pin;
  }
  void removeCrossRef(Pin* sta_pin, IdbPin* db_pin) {
    _sta2dbPin.erase(sta_pin);
    _db2staPin.erase(db_pin);
  }
  unsigned convertDBToTimingNetlist() override;

 private:
  unsigned makeTopCell();  // to do
  std::string changeStaBusNetNameToIdb(std::string sta_net_name);

  IdbBuilder* _idb = nullptr;
  IdbDesign* _idb_design = nullptr;
  IdbDefService* _idb_def_service = nullptr;
  IdbLefService* _idb_lef_service = nullptr;

  HashMap<IdbInstance*, Instance*> _db2staInst;
  HashMap<Instance*, IdbInstance*> _sta2dbInst;

  HashMap<IdbPin*, Port*> _db2staPort;  // net: get io pin
  HashMap<Port*, IdbPin*> _sta2dbPort;

  HashMap<IdbNet*, Net*> _db2staNet;
  HashMap<Net*, IdbNet*> _sta2dbNet;

  HashMap<IdbPin*, Pin*> _db2staPin;  // net: get instance_pin
  HashMap<Pin*, IdbPin*> _sta2dbPin;
};

}  // namespace ista
