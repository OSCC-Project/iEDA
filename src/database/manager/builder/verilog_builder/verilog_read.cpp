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
 * @project		iDB
 * @file		verilog_read.cpp
 * @author		Yell
 * @date		17/11/2021
 * @version		0.1
* @description


        There is a verilog builder to build data structure from .v file.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// reference code
/*
void testExpandInstance(idb::NetlistReader* nr, string module_name) {
  idb::NetlistModule* refModule = nr->getModuleByName(module_name);

  vector<idb::NetlistInstance*>::iterator instanceIter;
  for (instanceIter = refModule->getInstanceList()->begin(); instanceIter !=
refModule->getInstanceList()->end();
       ++instanceIter) {
    if ((*instanceIter)->isStandCell() == false) {
      idb::NetlistInstance* expandedInstance =
(*instanceIter)->getExpandViewOfInstanceInBit();

      cout << " Before Expand: " << endl;
      (*instanceIter)->Visit();
      cout << " After Expand: " << endl;
      expandedInstance->Visit();

      //            break;
    }
  }
}

void testMakeSingleModule(idb::NetlistReader* nr, string topModuleName) {
  idb::NetlistModule* topModule = nr->makeSingleModule(topModuleName);

  string testStr = "TREE";

  if (testStr == "FLAT") {
    topModule->print();
    // delete nr;
  } else if (testStr == "TREE") {
    nr->printNetlist();
    // delete topModule;
  }
}
*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "verilog_read.h"

#include <cassert>
#include <regex>

#include "IdbDesign.h"

namespace idb {

VerilogRead::VerilogRead(IdbDefService* def_service)
{
  _def_service = def_service;
}

VerilogRead::~VerilogRead()
{
}

bool VerilogRead::createDb(std::string file, std::string top_module_name)
{
  if (!_verilog_read) {
    _verilog_read = new ista::VerilogReader();
  }
  _verilog_read->read(file.c_str());

  _top_module = _verilog_read->flattenModule(top_module_name.c_str());
  if (_top_module == nullptr) {
    return false;
  }

  IdbDesign* idb_design = _def_service->get_design();
  idb_design->set_design_name(_top_module->get_module_name());

  // string testStr = "FLAT";

  // if (testStr == "FLAT") {
  //   _top_module->print();
  // } else if (testStr == "TREE") {
  //   _verilog_read->printNetlist();
  // }

  build_pins();
  build_nets();
  build_components();

  return true;
}

/**
 * @brief convert netlist port_direction to idb port_direction.
 *
 * @param port_direction
 * @return IdbConnectDirection
 */
IdbConnectDirection VerilogRead::netlistToIdb(ista::VerilogDcl::DclType port_direction) const
{
  if (port_direction == ista::VerilogDcl::DclType::kInput) {
    return IdbConnectDirection::kInput;
  } else if (port_direction == ista::VerilogDcl::DclType::kOutput) {
    return IdbConnectDirection::kOutput;
  } else if (port_direction == ista::VerilogDcl::DclType::kInout) {
    return IdbConnectDirection::kInOut;
  } else {
    std::cout << "not support.";
    return IdbConnectDirection::kNone;
  }
}

/**
 * @brief build pins.
 *
 * @return int32_t
 */
int32_t VerilogRead::build_pins()
{
  IdbDesign* idb_design = _def_service->get_design();

  auto& top_module_stmts = _top_module->get_module_stmts();

  IdbPins* idb_io_pin_list = idb_design->get_io_pin_list();
  if (!idb_io_pin_list) {
    idb_io_pin_list = new IdbPins();
    idb_design->set_io_pin_list(idb_io_pin_list);
  }

  // create pin.
  auto dcl_process = [idb_io_pin_list, this](ista::VerilogDcl::DclType dcl_type, const char* dcl_name) -> IdbPin* {
    if (dcl_type == ista::VerilogDcl::DclType::kInput || dcl_type == ista::VerilogDcl::DclType::kOutput
        || dcl_type == ista::VerilogDcl::DclType::kInout) {
      IdbPin* idb_io_pin = new IdbPin();
      idb_io_pin->set_pin_name(dcl_name);
      idb_io_pin->set_term();
      idb_io_pin->get_term()->set_direction(netlistToIdb(dcl_type));
      idb_io_pin->get_term()->set_type(IdbConnectType::kSignal);
      idb_io_pin->set_as_io();

      idb_io_pin_list->add_pin_list(idb_io_pin);
      return idb_io_pin;
    }

    return nullptr;
  };

  // process declare statement.
  auto process_dcl_stmt = [&dcl_process, idb_design](auto* dcl_stmt) {
    auto dcl_type = dcl_stmt->get_dcl_type();
    const auto* dcl_name = dcl_stmt->get_dcl_name();
    auto dcl_range = dcl_stmt->get_range();

    if (!dcl_range) {
      dcl_process(dcl_type, dcl_name);
    } else {
      auto bus_range = *(dcl_range);
      for (int index = bus_range.second; index <= bus_range.first; index++) {
        // for port or wire bus, we split to one bye one port.
        const char* one_name = Str::printf("%s[%d]", dcl_name, index);
        auto io_pin = dcl_process(dcl_type, one_name);
        if (io_pin) {
          if (index == bus_range.second) {
            IdbBus io_pin_bus(dcl_name, bus_range.first, bus_range.second);
            io_pin_bus.set_type(IdbBus::kBusType::kBusIo);
            io_pin_bus.addPin(io_pin);

            idb_design->get_bus_list()->addBusObject(std::move(io_pin_bus));

          } else {
            std::string bus_name = dcl_name;
            auto found_pin_bus = idb_design->get_bus_list()->findBus(bus_name);
            assert(found_pin_bus);
            (*found_pin_bus).get().addPin(io_pin);
          }
        }
      }
    }
  };

  int num = 0;

  for (auto& stmt : top_module_stmts) {
    if (stmt->isVerilogDclsStmt()) {
      auto* dcls_stmt = dynamic_cast<ista::VerilogDcls*>(stmt.get());
      auto& dcls = dcls_stmt->get_verilog_dcls();
      for (auto& dcl_stmt : dcls) {
        process_dcl_stmt(dcl_stmt.get());
        num++;
        if (num % 1000 == 0) {
          std::cout << "Processed " << num << " pins..." << std::endl;
        }
      }
    } else if (stmt->isVerilogDclStmt()) {
      auto* dcl_stmt = dynamic_cast<VerilogDcl*>(stmt.get());
      process_dcl_stmt(dcl_stmt);
      num++;
      if (num % 1000 == 0) {
        std::cout << "Processed " << num << " pins..." << std::endl;
      }
    }
  }
  return kVerilogSuccess;
}

/**
 * @brief build nets.
 *
 * @return int32_t
 */
int32_t VerilogRead::build_nets()
{
  IdbDesign* idb_design = _def_service->get_design();

  auto& top_module_stmts = _top_module->get_module_stmts();
  IdbPins* idb_io_pin_list = idb_design->get_io_pin_list();

  IdbNetList* idb_net_list = idb_design->get_net_list();
  if (!idb_net_list) {
    idb_net_list = new IdbNetList;
    idb_design->set_net_list(idb_net_list);
  }

  auto add_wire_net = [idb_net_list, idb_io_pin_list](std::string net_name) -> IdbNet* {
    IdbNet* idb_net = new IdbNet();
    idb_net->set_net_name(net_name);
    idb_net->set_connect_type(IdbConnectType::kSignal);
    auto* io_pin = idb_io_pin_list->find_pin(net_name);
    if (io_pin) {
      idb_net->set_io_pin(io_pin);
      io_pin->set_net(idb_net);
      io_pin->set_net_name(idb_net->get_net_name());
    }

    idb_net_list->add_net(idb_net);
    return idb_net;
  };

  auto replace_str = [](const string& str, const string& replace_str, const string& new_str) {
    std::regex re(replace_str);
    return std::regex_replace(str, re, new_str);
  };

  auto process_dcl_stmt = [&add_wire_net, &replace_str, idb_design](auto* dcl_stmt) {
    auto dcl_type = dcl_stmt->get_dcl_type();
    const auto* dcl_name = dcl_stmt->get_dcl_name();
    if (dcl_type == ista::VerilogDcl::DclType::kWire) {
      std::string net_name = dcl_name;
      if (std::string::npos != net_name.find('\\')) {
        net_name = replace_str(net_name, R"(\\)", "");
      }

      auto dcl_range = dcl_stmt->get_range();

      if (!dcl_range) {
        auto* idb_net = add_wire_net(net_name);

        if (!Str::contain(dcl_name, "\\[")) {
          auto [bus_name, bus_index] = Str::matchBusName(net_name.c_str());
          if (bus_index) {
            if (auto found_pin_bus = idb_design->get_bus_list()->findBus(bus_name); !found_pin_bus) {
              IdbBus io_pin_bus(bus_name, bus_index.value(), 0);
              io_pin_bus.set_type(IdbBus::kBusType::kBusNet);
              io_pin_bus.addNet(idb_net);

              idb_design->get_bus_list()->addBusObject(std::move(io_pin_bus));
            } else {
              (*found_pin_bus).get().updateRange(bus_index.value());
              (*found_pin_bus).get().addNet(idb_net);
            }
          }
        }
      } else {
        auto bus_range = *(dcl_range);
        for (int index = bus_range.second; index <= bus_range.first; index++) {
          // for port or wire bus, we split to one bye one port.
          const char* one_name = Str::printf("%s[%d]", dcl_name, index);
          auto idb_net = add_wire_net(one_name);
          if (index == bus_range.second) {
            IdbBus io_pin_bus(dcl_name, bus_range.first, bus_range.second);
            io_pin_bus.set_type(IdbBus::kBusType::kBusNet);
            io_pin_bus.addNet(idb_net);

            idb_design->get_bus_list()->addBusObject(std::move(io_pin_bus));

          } else {
            std::string bus_name = dcl_name;
            auto found_pin_bus = idb_design->get_bus_list()->findBus(bus_name);
            assert(found_pin_bus);
            (*found_pin_bus).get().addNet(idb_net);
          }
        }
      }
    }
  };

  int num = 0;

  for (auto& stmt : top_module_stmts) {
    if (stmt->isVerilogDclsStmt()) {
      auto* dcls_stmt = dynamic_cast<ista::VerilogDcls*>(stmt.get());
      auto& dcls = dcls_stmt->get_verilog_dcls();
      for (auto& dcl_stmt : dcls) {
        process_dcl_stmt(dcl_stmt.get());
        num++;
        if (num % 1000 == 0) {
          std::cout << "Processed " << num << " nets..." << std::endl;
        }
      }
    } else if (stmt->isVerilogDclStmt()) {
      auto* dcl_stmt = dynamic_cast<ista::VerilogDcl*>(stmt.get());
      process_dcl_stmt(dcl_stmt);
      num++;
      if (num % 1000 == 0) {
        std::cout << "Processed " << num << " nets..." << std::endl;
      }
    }
  }

  return kVerilogSuccess;
}

/**
 * @brief build components.
 *
 * @return int32_t
 */
int32_t VerilogRead::build_components()
{
  IdbDesign* idb_design = _def_service->get_design();
  IdbLayout* idb_layout = _def_service->get_layout();
  IdbCellMasterList* idb_master_list = idb_layout->get_cell_master_list();
  IdbPins* idb_io_pin_list = idb_design->get_io_pin_list();

  auto& top_module_stmts = _top_module->get_module_stmts();

  IdbInstanceList* idb_instance_list = idb_design->get_instance_list();
  if (!idb_instance_list) {
    idb_instance_list = new IdbInstanceList;
    idb_design->set_instance_list(idb_instance_list);
  }

  auto* idb_net_list = idb_design->get_net_list();

  auto add_pin = [idb_net_list, idb_io_pin_list, idb_design](const std::string& raw_name, auto* idb_pin) {
    auto replace_str = [](const string& str, const string& old_str, const string& new_str) {
      std::regex re(old_str);
      return std::regex_replace(str, re, new_str);
    };

    std::string net_name = raw_name;

    // strip \\\ char.
    if (std::string::npos != raw_name.find('\\')) {
      net_name = replace_str(raw_name, R"(\\)", "");
    }

    auto* idb_net = idb_net_list->find_net(net_name);
    if (!idb_net) {
      // judge whether bus net name.
      auto net_bus = idb_design->get_bus_list()->findBus(net_name);
      if (!net_bus) {
        // not bus net name, create common idb net.
        idb_net = new IdbNet();
        idb_net->set_net_name(net_name);
        idb_net->set_connect_type(IdbConnectType::kSignal);
        idb_net_list->add_net(idb_net);

        // judge whether contain bus index name, if bus name contain \\[, should
        // not treat as bus.
        if (!Str::contain(net_name.c_str(), "\\[")) {
          // is bus index net.
          auto [bus_name, bus_index] = Str::matchBusName(net_name.c_str());
          if (bus_index) {
            if (auto found_net_bus = idb_design->get_bus_list()->findBus(bus_name); !found_net_bus) {
              // not found net bus, create it.
              IdbBus net_bus(bus_name, bus_index.value(), bus_index.value());
              net_bus.set_type(IdbBus::kBusType::kBusNet);
              net_bus.addNet(idb_net);

              idb_design->get_bus_list()->addBusObject(std::move(net_bus));
            } else {
              // exist net bus, update range.
              (*found_net_bus).get().updateRange(bus_index.value());
              (*found_net_bus).get().addNet(idb_net);
            }
          }
        }
      } else {
        // existed bus net, get one net of bus.
        std::string pin_name = idb_pin->get_pin_name();
        auto [pin_bus_name, pin_bus_index] = Str::matchBusName(pin_name.c_str());
        assert(pin_bus_index);
        idb_net = (*net_bus).get().getNet(pin_bus_index.value());
      }
    }

    auto* io_pin = idb_io_pin_list->find_pin(net_name);
    if (io_pin && !idb_net->get_io_pin()) {
      idb_net->set_io_pin(io_pin);
      io_pin->set_net(idb_net);
      io_pin->set_net_name(idb_net->get_net_name());
    }

    idb_net->add_instance_pin(idb_pin);
    idb_pin->set_net(idb_net);
    idb_pin->set_net_name(net_name);
    idb_net->get_instance_list()->add_instance(idb_pin->get_instance());
  };

  /*lambda function flatten concate net, which maybe nested.*/
  std::function<void(VerilogNetConcatExpr*, std::vector<VerilogNetExpr*>&)> flatten_concat_net_expr
      = [&flatten_concat_net_expr](VerilogNetConcatExpr* net_concat_expr, std::vector<VerilogNetExpr*>& net_concat_vec) {
          auto& verilog_id_concat = net_concat_expr->get_verilog_id_concat();
          for (auto& verilog_id_net_expr : verilog_id_concat) {
            if (verilog_id_net_expr->isConcatExpr()) {
              flatten_concat_net_expr(dynamic_cast<VerilogNetConcatExpr*>(verilog_id_net_expr.get()), net_concat_vec);
            } else {
              net_concat_vec.push_back(verilog_id_net_expr.get());
            }
          }
        };

  // create or found bus, add idb pin to bus.
  auto create_or_found_bus = [idb_design](std::string bus_name, IdbPin* idb_pin, std::optional<int> max_bus_bit, bool is_create) {
    if (is_create) {
      IdbBus pin_bus(bus_name, max_bus_bit.value(), 0);
      pin_bus.set_type(IdbBus::kBusType::kBusInstancePin);
      pin_bus.addPin(idb_pin);

      idb_design->get_bus_list()->addBusObject(std::move(pin_bus));
    } else {
      auto found_pin_bus = idb_design->get_bus_list()->findBus(bus_name);
      assert(found_pin_bus);
      (*found_pin_bus).get().addPin(idb_pin);
    }
  };
  int num = 0;
  for (auto& stmt : top_module_stmts) {
    if (stmt->isModuleInstStmt()) {
      auto* inst_stmt = dynamic_cast<ista::VerilogInst*>(stmt.get());
      const char* inst_name = inst_stmt->get_inst_name();
      IdbInstance* idb_instance = new IdbInstance();
      idb_instance->set_name(inst_name);
      std::string cell_master_name = inst_stmt->get_cell_name();

      auto* cell_master = idb_master_list->find_cell_master(cell_master_name);
      if (cell_master == nullptr) {
        std::cout << "Error : can not find cell master = " << cell_master_name << std::endl;
        continue;
      }
      idb_instance->set_cell_master(cell_master);

      // build instance pin connected net.
      auto& port_connections = inst_stmt->get_port_connections();
      for (auto& port_connection : port_connections) {
        auto* cell_port_id = port_connection->get_port_id();
        auto* net_expr = port_connection->get_net_expr();  // get net name

        if (!net_expr) {
          continue;
        }

        if (net_expr->isIDExpr() || net_expr->isConstant()) {
          // condition for common ID and constant.
          std::string pin_name = cell_port_id->getName();
          auto* idb_pin = idb_instance->get_pin(pin_name);
          if (!idb_pin) {
            // should be pin bus, get bus size first.
            int max_bus_bit = 1;
            std::vector<IdbPin*> bus_pins;
            for (int i = 0;; ++i) {
              std::string bus_pin_name = pin_name + "[" + std::to_string(i) + "]";
              auto* idb_bus_pin = idb_instance->get_pin(bus_pin_name);

              // assert bus pin exist.
              if (i == 0) {
                assert(idb_bus_pin);
              }

              if (idb_bus_pin) {
                // the net should be net bus too, select bus index net.
                const char* net_name = net_expr->get_verilog_id()->getBaseName();
                std::optional<int> net_bus_base_index;
                if (net_expr->get_verilog_id()->isBusSliceID()) {
                  net_bus_base_index = dynamic_cast<VerilogSliceID*>(net_expr->get_verilog_id())->get_range_base();
                }
                std::string bus_net_name = std::string(net_name) + "[" + std::to_string(i + net_bus_base_index.value_or(0)) + "]";
                add_pin(bus_net_name, idb_bus_pin);
                bus_pins.emplace_back(idb_bus_pin);
              } else {
                // found the max bus width, break loop.
                max_bus_bit = i;
                break;
              }
            }

            std::string bus_name = Str::printf("%s/%s", inst_name, pin_name.c_str());
            for (int i = 0; auto* idb_bus_pin : bus_pins) {
              if (i == 0) {
                create_or_found_bus(bus_name, idb_bus_pin, max_bus_bit - 1, true);
              } else {
                create_or_found_bus(bus_name, idb_bus_pin, std::nullopt, false);
              }
              ++i;
            }

          } else {
            // exist idb pin, add to net.
            if (!net_expr->isConstant()) {
              const char* net_name = net_expr->get_verilog_id()->getName();
              add_pin(net_name, idb_pin);
            }
          }

        } else {
          // condition for net concat.
          auto* net_concat_expr = dynamic_cast<ista::VerilogNetConcatExpr*>(net_expr);
          std::vector<VerilogNetExpr*> verilog_id_concat_vec;
          flatten_concat_net_expr(net_concat_expr, verilog_id_concat_vec);

          auto* cell_port_name = cell_port_id->getName();
          std::string bus_name = Str::printf("%s/%s", inst_name, cell_port_name);

          // found pin bus size.
          std::vector<IdbPin*> bus_pin_vec;
          for (int i = 0;; ++i) {
            std::string pin_name = Str::printf("%s[%d]", cell_port_name, i);
            auto* idb_pin = idb_instance->get_pin(pin_name);
            if (!idb_pin) {
              break;
            }
            bus_pin_vec.emplace_back(idb_pin);
          }

          for (int i = bus_pin_vec.size() - 1; auto* verilog_id_net_expr : verilog_id_concat_vec) {
            assert(i >= 0);
            auto* idb_pin = bus_pin_vec[i];
            if (i == static_cast<int>(bus_pin_vec.size() - 1)) {
              // first idb pin create instance pin bus.
              create_or_found_bus(bus_name, idb_pin, i, true);
            }

            if (verilog_id_net_expr->isConstant()) {
              --i;
              // the next pin add to pin bus.
              if (i >= 0) {
                idb_pin = bus_pin_vec[i];
                create_or_found_bus(bus_name, idb_pin, std::nullopt, false);
              }
              continue;
            }

            const char* net_name = verilog_id_net_expr->get_verilog_id()->getBaseName();
            auto net_bus = idb_design->get_bus_list()->findBus(net_name);

            if (net_bus) {
              // for net bus, we need span the bus.
              int bus_left = (*net_bus).get().get_left();
              int bus_right = (*net_bus).get().get_right();

              auto* net_expr_verilog_id = verilog_id_net_expr->get_verilog_id();
              if (net_expr_verilog_id->isBusIndexID()) {
                bus_left = dynamic_cast<VerilogIndexID*>(net_expr_verilog_id)->get_index();
                bus_right = bus_left;
              } else if (net_expr_verilog_id->isBusSliceID()) {
                bus_left = dynamic_cast<VerilogSliceID*>(net_expr_verilog_id)->get_range_max();
                bus_right = dynamic_cast<VerilogSliceID*>(net_expr_verilog_id)->get_range_base();
              }

              for (int j = bus_left; j >= bus_right; --j) {
                const char* bus_one_net_name = Str::printf("%s[%d]", net_name, j);
                auto* idb_net = idb_net_list->find_net(bus_one_net_name);
                assert(idb_net);
                add_pin(bus_one_net_name, idb_pin);
                --i;
                // the next pin add to pin bus.
                if (i >= 0) {
                  idb_pin = bus_pin_vec[i];
                  create_or_found_bus(bus_name, idb_pin, std::nullopt, false);
                }
              }
            } else {
              add_pin(net_name, idb_pin);
              --i;
              // the next pin add to pin bus.
              if (i >= 0) {
                idb_pin = bus_pin_vec[i];
                create_or_found_bus(bus_name, idb_pin, std::nullopt, false);
              }
            }
          }
        }
      }
      num++;
      if (num % 1000 == 0) {
        std::cout << "Processed " << num << " components..." << std::endl;
      }
      idb_instance_list->add_instance(idb_instance);
    }
  }

  return kVerilogSuccess;
}

}  // namespace idb
