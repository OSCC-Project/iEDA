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
#include "FixFanout.h"
#include "builder.h"

#include "IdbEnum.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"

namespace ino {

FixFanout::FixFanout(ino::DbInterface *db_interface) : _db_interface(db_interface) {
  _timing_engine = _db_interface->get_timing_engine();
  _idb = _db_interface->get_idb();
  _max_fanout = _db_interface->get_max_fanout();
}

/**
 * 临时修复io问题,此函数有问题可联系zzs
 */
void FixFanout::fixIO() {
  auto* idb_design = _idb->get_def_service()->get_design();
  idb::IdbPins *idb_io_pin_list =
      _idb->get_def_service()->get_design()->get_io_pin_list();
  std::string buffer_name = _db_interface->get_insert_buffer();
  size_t      new_buf_idx = 0;
  size_t      new_net_idx = 0;

  for (idb::IdbPin *idb_io_pin : idb_io_pin_list->get_pin_list()) {
    if (idb_io_pin->get_net() == nullptr) {
      continue;
    }
    if (idb_io_pin->get_pin_name() == idb_io_pin->get_net()->get_net_name()) {
      idb::IdbNet               *io_net = idb_io_pin->get_net();
      std::vector<idb::IdbPin *> instance_pin_list =
          io_net->get_instance_pin_list()->get_pin_list();
      // 在io net中解开所有instance_pin
      for (idb::IdbPin *instance_pin : instance_pin_list) {
        idb_design->disconnectPinFromNet(instance_pin);
      }
      // 构建新的net
      idb::IdbNet *new_net = idb_design->createOrFindNet(idb_design->makeUniqueNetName("fixio_net_" + std::to_string(new_net_idx++)),
                                                         idb::IdbConnectType::kSignal, idb::IdbCreatePolicy::kErrorIfExists);
      if (new_net == nullptr) {
        continue;
      }
      // 将原instance pin加入新net
      for (idb::IdbPin *instance_pin : instance_pin_list) {
        idb_design->connectPinToNet(instance_pin, new_net);
      }
      // 生成buf
      idb::IdbInstance *new_buf = idb_design->createInstance(idb_design->makeUniqueInstanceName("fixio_buf_" + std::to_string(new_buf_idx++)),
                                                             buffer_name, idb::IdbInstanceType::kTiming, idb::IdbPlacementStatus::kNone,
                                                             idb::IdbOrient::kNone, 0, 0, idb::IdbCreatePolicy::kErrorIfExists);
      if (new_buf == nullptr) {
        continue;
      }
      // 插入buf
      for (idb::IdbPin *buf_pin : new_buf->get_pin_list()->get_pin_list()) {
        if (buf_pin->get_term()->get_direction() == idb::IdbConnectDirection::kInput ||
            buf_pin->get_term()->get_direction() == idb::IdbConnectDirection::kOutput) {
          if (buf_pin->get_term()->get_direction() ==
              idb_io_pin->get_term()->get_direction()) {
            idb_design->connectPinToNet(buf_pin, io_net);
          } else {
            idb_design->connectPinToNet(buf_pin, new_net);
          }
        }
      }

    } else {
      idb::IdbNet *origin_net = idb_io_pin->get_net();
      // 在origin net中解开io pin
      idb_design->disconnectPinFromNet(idb_io_pin);
      // 加入原来的io net
      idb::IdbNet *io_net = idb_design->createOrFindNet(idb_io_pin->get_pin_name(), idb::IdbConnectType::kSignal);
      if (io_net == nullptr) {
        continue;
      }
      idb_design->connectPinToNet(idb_io_pin, io_net);
      // 生成buf
      idb::IdbInstance *new_buf = idb_design->createInstance(idb_design->makeUniqueInstanceName("fixio_buf_" + std::to_string(new_buf_idx++)),
                                                             buffer_name, idb::IdbInstanceType::kTiming, idb::IdbPlacementStatus::kNone,
                                                             idb::IdbOrient::kNone, 0, 0, idb::IdbCreatePolicy::kErrorIfExists);
      if (new_buf == nullptr) {
        continue;
      }
      // 插入buf
      for (idb::IdbPin *buf_pin : new_buf->get_pin_list()->get_pin_list()) {
        if (buf_pin->get_term()->get_direction() == idb::IdbConnectDirection::kInput ||
            buf_pin->get_term()->get_direction() == idb::IdbConnectDirection::kOutput) {
          if (buf_pin->get_term()->get_direction() ==
              idb_io_pin->get_term()->get_direction()) {
            idb_design->connectPinToNet(buf_pin, io_net);
          } else {
            idb_design->connectPinToNet(buf_pin, origin_net);
          }
        }
      }
    }
  }

  LOG_INFO << "[Result: ] Insert " << new_buf_idx << " fix_io buffers.\n";
  LOG_INFO << "[Result: ] Insert " << new_net_idx << " fix_io nets.\n";
}

void FixFanout::fixFanout() {
  _db_interface->set_eval_data();
  _idb_layout = _idb->get_lef_service()->get_layout();
  _idb_design = _idb->get_def_service()->get_design();
  auto net_list = _idb_design->get_net_list()->get_net_list();

  auto      *design_nl = _timing_engine->get_netlist();
  ista::Net *sta_net;
  FOREACH_NET(design_nl, sta_net) {
    auto fanout = (int)sta_net->getFanouts();
    auto idb_adpat =
        dynamic_cast<ista::TimingIDBAdapter *>(_timing_engine->get_db_adapter());
    IdbNet *db_net = idb_adpat->staToDb(sta_net);
    if (sta_net->isClockNet()) {
      _idb_design->setNetConnectType(db_net->get_net_name(), idb::IdbConnectType::kClock);
      continue;
    }
    if (fanout > _max_fanout) {
      _fanout_vio_num++;
      fixFanout(db_net);
    }
  }

  LOG_INFO << "[Result: ] Find " << _fanout_vio_num << " Net with fanout violation.\n";
  LOG_INFO << "[Result: ] Insert " << _insert_instance_index - 1 << " Buffers.\n";

  _db_interface->report()->get_ofstream() << "[Result: ] Find " << _fanout_vio_num
                                          << " Net with fanout violation.\n"
                                             "[Result: ] Insert "
                                          << _insert_instance_index - 1 << " Buffers.\n";
  _db_interface->report()->get_ofstream().close();
  _db_interface->report()->reportTime(false);
}

void FixFanout::fixFanout(IdbNet *net) {
  int  fanout = net->get_load_pins().size();
  bool have_switch_name = false;
  while (fanout > _max_fanout) {
    auto load_pins = net->get_load_pins();
    bool connect_to_port = false; // if net connect to a port need rename for the net
    for (auto pin : load_pins) {
      if (pin->is_io_pin()) {
        connect_to_port = true;
        break;
      }
    }
    /* code */
    IdbNet *in_net, *out_net;
    in_net = net;
    string net_name = ("fanout_net_" + std::to_string(_make_net_index));
    _make_net_index++;
    out_net = makeNet(net_name.c_str());
    if (out_net == nullptr) {
      return;
    }

    string buf_name = ("fanout_buf_" + std::to_string(_insert_instance_index));
    _insert_instance_index++;

    auto         insert_buffer = _db_interface->get_insert_buffer();
    IdbInstance *insert_buf = makeInstance(insert_buffer, buf_name.c_str());
    LOG_ERROR_IF(!insert_buf) << "insert buffer uninitialized.";
    if (insert_buf == nullptr) {
      return;
    }

    // get buf input_pin and output_pin
    IdbPin  *buf_input_pin = nullptr;
    IdbPin  *buf_output_pin = nullptr;
    IdbPins *buf_pins = insert_buf->get_pin_list();
    for (auto pin : buf_pins->get_pin_list()) {
      if (pin->get_term()->get_type() == idb::IdbConnectType::kPower ||
          pin->get_term()->get_type() == idb::IdbConnectType::kGround) {
        continue;
      }
      if (pin->get_term()->get_direction() == idb::IdbConnectDirection::kInput) {
        buf_input_pin = pin;
      } else if (pin->get_term()->get_direction() == idb::IdbConnectDirection::kOutput) {
        buf_output_pin = pin;
      }
    }
    LOG_ERROR_IF(!buf_input_pin) << "'buf_input_pin' uninitialized.";
    LOG_ERROR_IF(!buf_output_pin) << "'buf_output_pin' uninitialized.";
    connect(insert_buf, buf_input_pin, in_net);
    connect(insert_buf, buf_output_pin, out_net);
    for (int i = 0; i < _max_fanout; i++) {
      IdbPin *idb_pin = load_pins[i];
      if (connect_to_port && !have_switch_name) {
        string in_net_name = in_net->get_net_name();
        string out_net_name = out_net->get_net_name();
        const string temp_net_name = "__ino_fanout_swap_" + std::to_string(_make_net_index) + "_"
                                     + std::to_string(_insert_instance_index);
        auto* idb_adpat =
            dynamic_cast<ista::TimingIDBAdapter *>(_timing_engine->get_db_adapter());
        auto* in_sta_net = idb_adpat == nullptr ? nullptr : idb_adpat->dbToSta(in_net);
        auto* out_sta_net = idb_adpat == nullptr ? nullptr : idb_adpat->dbToSta(out_net);
        if (idb_adpat != nullptr && in_sta_net != nullptr && out_sta_net != nullptr) {
          idb_adpat->swapNetNames(in_sta_net, out_sta_net);
        } else {
          _idb_design->renameNet(in_net, temp_net_name);
          _idb_design->renameNet(out_net, in_net_name);
          _idb_design->renameNet(in_net, out_net_name);
        }
        have_switch_name = true;
      }
      // 1
      disconnectPin(idb_pin, in_net);
      // 2
      connect(idb_pin->get_instance(), idb_pin, out_net);
    }
    fanout = net->get_load_pins().size();
  }
}

IdbNet *FixFanout::makeNet(const char *name) {
  string str_name = _idb_design->makeUniqueNetName(name == nullptr ? "fanout_net_" : name);
  return _idb_design->createOrFindNet(str_name, idb::IdbConnectType::kSignal, idb::IdbCreatePolicy::kErrorIfExists);
}

IdbInstance *FixFanout::makeInstance(string master_name, string inst_name) {
  return _idb_design->createInstance(_idb_design->makeUniqueInstanceName(inst_name), master_name, idb::IdbInstanceType::kTiming,
                                     idb::IdbPlacementStatus::kNone, idb::IdbOrient::kNone, 0, 0,
                                     idb::IdbCreatePolicy::kErrorIfExists);
}

void FixFanout::disconnectPin(IdbPin *dpin, IdbNet *dnet) {
  if (dpin && dnet) {
    _idb_design->disconnectPinFromNet(dpin);
  }
}

void FixFanout::connect(IdbInstance *dinst, IdbPin *dpin, IdbNet *dnet) {
  if (dinst) {
    auto  &dpin_list = dinst->get_pin_list()->get_pin_list();
    string port_name = dpin->get_pin_name();
    for (auto dpin : dpin_list) {
      if (dpin->get_pin_name() == port_name) {
        if (dpin->is_io_pin()) {
          _idb_design->connectPinToNet(dpin, dnet);
        } else {
          _idb_design->connectPinToNet(dpin, dnet);
        }
        break;
      }
    }
  } else {
    _idb_design->connectPinToNet(dpin, dnet);
  }
}

} // namespace ino
