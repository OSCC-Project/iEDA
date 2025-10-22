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
  idb::IdbNetList *idb_net_list = _idb->get_def_service()->get_design()->get_net_list();
  idb::IdbInstanceList *idb_instance_list =
      _idb->get_def_service()->get_design()->get_instance_list();
  idb::IdbCellMasterList *idb_cell_master_list =
      _idb->get_def_service()->get_layout()->get_cell_master_list();
  idb::IdbPins *idb_io_pin_list =
      _idb->get_def_service()->get_design()->get_io_pin_list();
  std::string buffer_name = _db_interface->get_insert_buffer();
  size_t      new_idx = 0;

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
        io_net->remove_pin(instance_pin);
      }
      // 构建新的net
      idb::IdbNet *new_net = new IdbNet();
      new_net->set_net_name("fixio_net_" + std::to_string(new_idx++));
      idb_net_list->add_net(new_net);
      // 将原instance pin加入新net
      for (idb::IdbPin *instance_pin : instance_pin_list) {
        new_net->add_instance_pin(instance_pin);
        instance_pin->set_net(new_net);
        instance_pin->set_net_name(new_net->get_net_name());
      }
      // 生成buf
      idb::IdbInstance *new_buf = new IdbInstance();
      new_buf->set_name("fixio_buf_" + std::to_string(new_idx++));
      new_buf->set_cell_master(idb_cell_master_list->find_cell_master(buffer_name));
      idb_instance_list->add_instance(new_buf);
      // 插入buf
      for (idb::IdbPin *buf_pin : new_buf->get_pin_list()->get_pin_list()) {
        if (buf_pin->get_term()->get_direction() == idb::IdbConnectDirection::kInput ||
            buf_pin->get_term()->get_direction() == idb::IdbConnectDirection::kOutput) {
          if (buf_pin->get_term()->get_direction() ==
              idb_io_pin->get_term()->get_direction()) {
            io_net->add_instance_pin(buf_pin);
            buf_pin->set_net(io_net);
            buf_pin->set_net_name(io_net->get_net_name());
          } else {
            new_net->add_instance_pin(buf_pin);
            buf_pin->set_net(new_net);
            buf_pin->set_net_name(new_net->get_net_name());
          }
        }
      }

    } else {
      idb::IdbNet *origin_net = idb_io_pin->get_net();
      // 在origin net中解开io pin
      origin_net->remove_pin(idb_io_pin);
      // 加入原来的io net
      idb::IdbNet *io_net = idb_net_list->find_net(idb_io_pin->get_pin_name());
      if (io_net == nullptr) {
        io_net = new IdbNet();
        io_net->set_net_name(idb_io_pin->get_pin_name());
        idb_net_list->add_net(io_net);
      }
      idb_io_pin->set_net(io_net);
      idb_io_pin->set_net_name(io_net->get_net_name());
      // 生成buf
      idb::IdbInstance *new_buf = new IdbInstance();
      new_buf->set_name("fixio_buf_" + std::to_string(new_idx++));
      new_buf->set_cell_master(idb_cell_master_list->find_cell_master(buffer_name));
      idb_instance_list->add_instance(new_buf);
      // 插入buf
      for (idb::IdbPin *buf_pin : new_buf->get_pin_list()->get_pin_list()) {
        if (buf_pin->get_term()->get_direction() == idb::IdbConnectDirection::kInput ||
            buf_pin->get_term()->get_direction() == idb::IdbConnectDirection::kOutput) {
          if (buf_pin->get_term()->get_direction() ==
              idb_io_pin->get_term()->get_direction()) {
            io_net->add_instance_pin(buf_pin);
            buf_pin->set_net(io_net);
            buf_pin->set_net_name(io_net->get_net_name());
          } else {
            origin_net->add_instance_pin(buf_pin);
            buf_pin->set_net(origin_net);
            buf_pin->set_net_name(origin_net->get_net_name());
          }
        }
      }
    }
  }
}

void FixFanout::fixFanout() {
  _db_interface->set_eval_data();
  _idb_layout = _idb->get_lef_service()->get_layout();
  _idb_design = _idb->get_def_service()->get_design();
  auto net_list = _idb_design->get_net_list()->get_net_list();

  auto      *design_nl = _timing_engine->get_netlist();
  ista::Net *sta_net;
  FOREACH_NET(design_nl, sta_net) {
    if (sta_net->isClockNet()) {
      continue;
    }
    auto fanout = (int)sta_net->getFanouts();
    if (fanout > _max_fanout) {
      _fanout_vio_num++;

      auto idb_adpat =
          dynamic_cast<ista::TimingIDBAdapter *>(_timing_engine->get_db_adapter());
      IdbNet *db_net = idb_adpat->staToDb(sta_net);
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
    out_net->set_connect_type(idb::IdbConnectType::kSignal);

    string buf_name = ("fanout_buf_" + std::to_string(_insert_instance_index));
    _insert_instance_index++;

    auto         insert_buffer = _db_interface->get_insert_buffer();
    IdbInstance *insert_buf = makeInstance(insert_buffer, buf_name.c_str());
    LOG_ERROR_IF(!insert_buf) << "insert buffer uninitialized.";

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
        in_net->set_net_name(out_net_name);
        out_net->set_net_name(in_net_name);
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
  string str_name = name;
  // IdbNetList *dbnet_list = _idb_design->get_net_list();
  IdbNet *dnet = _idb_design->get_net_list()->add_net(str_name);
  return dnet;
}

IdbInstance *FixFanout::makeInstance(string master_name, string inst_name) {
  IdbCellMasterList *master_list = _idb_layout->get_cell_master_list();
  IdbCellMaster     *master = master_list->find_cell_master(master_name);
  if (!master) {
    return nullptr;
  }
  IdbInstance *idb_inst = new IdbInstance();
  idb_inst->set_name(inst_name);
  idb_inst->set_cell_master(master);

  _idb_design->get_instance_list()->add_instance(idb_inst);
  return idb_inst;
}

void FixFanout::disconnectPin(IdbPin *dpin, IdbNet *dnet) {
  if (dpin && dnet) {
    dnet->remove_pin(dpin);
  }
}

void FixFanout::connect(IdbInstance *dinst, IdbPin *dpin, IdbNet *dnet) {
  if (dinst) {
    auto  &dpin_list = dinst->get_pin_list()->get_pin_list();
    string port_name = dpin->get_pin_name();
    for (auto dpin : dpin_list) {
      if (dpin->get_pin_name() == port_name) {
        if (dpin->is_io_pin()) {
          dnet->add_io_pin(dpin);
          dpin->set_net(dnet);
          dpin->set_net_name(dnet->get_net_name());
        } else {
          dnet->add_instance_pin(dpin);
          dpin->set_net(dnet);
          dpin->set_net_name(dnet->get_net_name());
        }
        break;
      }
    }
  } else {
    dnet->add_io_pin(dpin);
    dpin->set_net(dnet);
    dpin->set_net_name(dnet->get_net_name());
  }
}

} // namespace ino
