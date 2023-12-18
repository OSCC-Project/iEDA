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
 * @file VerilogReader.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-07-20
 */

#include "VerilogReader.hh"

#include <memory>

#include "log/Log.hh"
#include "string/Str.hh"

ista::VerilogReader* gVerilogReader = nullptr;

namespace ista {

VerilogID::VerilogID(const char* id) : _id(id)
{
}

VerilogIndexID::VerilogIndexID(const char* id, int index) : VerilogID(id), _index(index)
{
}

VerilogSliceID::VerilogSliceID(const char* id, int range_from, int range_to) : VerilogID(id), _range_from(range_from), _range_to(range_to)
{
}

VerilogNetIDExpr::VerilogNetIDExpr(VerilogID* verilog_id, unsigned line_no) : VerilogNetExpr(line_no), _verilog_id(verilog_id)
{
  LOG_FATAL_IF(!verilog_id);
}

VerilogNetIDExpr::VerilogNetIDExpr(const VerilogNetIDExpr& orig)
    : VerilogNetExpr(orig), _verilog_id(std::unique_ptr<VerilogID>(orig._verilog_id.get()->copy()))
{
}

VerilogNetIDExpr& VerilogNetIDExpr::operator=(const VerilogNetIDExpr& orig)
{
  if (this != &orig) {
    VerilogNetExpr::operator=(orig);
    _verilog_id = std::unique_ptr<VerilogID>(orig._verilog_id.get()->copy());
  }

  return *this;
}

VerilogNetConcatExpr::VerilogNetConcatExpr(Vector<std::unique_ptr<VerilogNetExpr>>&& verilog_id_concat, unsigned line_no)
    : VerilogNetExpr(line_no), _verilog_id_concat(std::move(verilog_id_concat))
{
}

VerilogNetConcatExpr::VerilogNetConcatExpr(const VerilogNetConcatExpr& orig) : VerilogNetExpr(orig)
{
  for (auto& net_expr : orig._verilog_id_concat) {
    _verilog_id_concat.emplace_back(net_expr.get()->copy());
  }
}

VerilogNetConcatExpr& VerilogNetConcatExpr::operator=(const VerilogNetConcatExpr& orig)
{
  if (this != &orig) {
    VerilogNetExpr::operator=(orig);
    for (auto& net_expr : orig._verilog_id_concat) {
      _verilog_id_concat.emplace_back(net_expr.get()->copy());
    }
  }

  return *this;
}

VerilogConstantExpr::VerilogConstantExpr(const char* constant, unsigned line_no)
    : VerilogNetExpr(line_no), _verilog_id(std::make_unique<VerilogID>(constant))
{
}

VerilogConstantExpr::VerilogConstantExpr(const VerilogConstantExpr& orig)
    : VerilogNetExpr(orig), _verilog_id(std::unique_ptr<VerilogID>(orig._verilog_id.get()->copy()))
{
}

VerilogConstantExpr& VerilogConstantExpr::operator=(const VerilogConstantExpr& orig)
{
  if (this != &orig) {
    VerilogNetExpr::operator=(orig);
    _verilog_id = std::unique_ptr<VerilogID>(orig._verilog_id.get()->copy());
  }

  return *this;
}

VerilogStmt::VerilogStmt(int line) : _line(line)
{
}

VerilogPortRefPortConnect::VerilogPortRefPortConnect(VerilogID* port_id, VerilogNetExpr* net_expr) : _port_id(port_id), _net_expr(net_expr)
{
}

VerilogPortRefPortConnect::VerilogPortRefPortConnect(const VerilogPortRefPortConnect& orig)
    : _port_id(std::unique_ptr<VerilogID>(orig._port_id.get()->copy())), _net_expr()
{
  if (orig._net_expr) {
    _net_expr = std::unique_ptr<VerilogNetExpr>(orig._net_expr.get()->copy());
  }
}

VerilogPortRefPortConnect& VerilogPortRefPortConnect::operator=(const VerilogPortRefPortConnect& orig)
{
  if (this != &orig) {
    _port_id = std::unique_ptr<VerilogID>(orig._port_id.get()->copy());
    _net_expr = std::unique_ptr<VerilogNetExpr>(orig._net_expr.get()->copy());
  }

  return *this;
}

VerilogInst::VerilogInst(const char* liberty_cell_name, const char* inst_name,
                         std::vector<std::unique_ptr<VerilogPortRefPortConnect>>&& port_connection, int line)
    : VerilogStmt(line), _inst_name(inst_name), _cell_name(liberty_cell_name), _port_connections(std::move(port_connection))
{
  Str::free(liberty_cell_name);
  Str::free(inst_name);
}

VerilogInst::VerilogInst(const VerilogInst& orig) : VerilogStmt(orig), _inst_name(orig._inst_name), _cell_name(orig._cell_name)
{
  for (auto& port_conn : orig._port_connections) {
    _port_connections.emplace_back(dynamic_cast<VerilogPortRefPortConnect*>(port_conn.get()->copy()));
  }
}

VerilogInst& VerilogInst::operator=(const VerilogInst& orig)
{
  if (this != &orig) {
    _inst_name = orig._inst_name;
    _cell_name = orig._cell_name;

    for (auto& port_conn : orig._port_connections) {
      _port_connections.emplace_back(dynamic_cast<VerilogPortRefPortConnect*>(port_conn.get()->copy()));
    }
  }

  return *this;
}

/**
 * @brief Get the port connect net from the inst stmt of the parent module.
 *
 * @param parent_module The inst stmt belong to.
 * @param port_id The port name.
 * @param port_bus_wide_range if the port is bus.
 * @return std::unique_ptr<VerilogNetExpr>
 */
std::unique_ptr<VerilogNetExpr> VerilogInst::getPortConnectNet(VerilogModule* parent_module, VerilogModule* inst_module, VerilogID* port_id,
                                                               std::optional<std::pair<int, int>> port_bus_wide_range)
{
  // The function for process the concate connection, port_concat_connect_net is inst connect net, index is inst module port index.
  auto get_concat_connect_net
      = [parent_module, &port_bus_wide_range](VerilogNetConcatExpr* port_concat_connect_net, int port_index) -> VerilogNetExpr* {
    auto& concat_expr_nets = port_concat_connect_net->get_verilog_id_concat();
    LOG_FATAL_IF(!port_bus_wide_range);

    int bus_range_min = std::min(port_bus_wide_range->first, port_bus_wide_range->second);
    // bus_range_max is the bus max beyond range.
    int bus_range_max = std::max(port_bus_wide_range->first, port_bus_wide_range->second) + 1;

    std::optional<int> net_index;
    VerilogNetExpr* connect_net_expr = nullptr;
    for (auto& expr_net : concat_expr_nets) {
      if (expr_net->get_verilog_id()->isBusIndexID()) {
        --bus_range_max;
      } else if (expr_net->get_verilog_id()->isBusSliceID()) {
        auto* slice_id = dynamic_cast<VerilogSliceID*>(expr_net->get_verilog_id());
        int from = slice_id->get_range_from();
        int to = slice_id->get_range_to();
        for (int j = from; ((from > to) ? j >= to : j <= to); from > to ? --j : ++j) {
          --bus_range_max;
          if (bus_range_max == port_index) {
            net_index = j;
            break;
          }
        }
      } else {
        auto* stmt = parent_module->findDclStmt(expr_net->get_verilog_id()->getBaseName());
        LOG_INFO_IF_EVERY_N(!stmt, 100) << "not found dcl stmt " << expr_net->get_verilog_id()->getBaseName();
        if (stmt) {
          if (stmt->isVerilogDclStmt()) {
            auto* dcl_stmt = dynamic_cast<VerilogDcl*>(stmt);
            auto range = dcl_stmt->get_range();
            if (range) {
              for (int j = range->first; j >= range->second; range->first > range->second ? --j : ++j) {
                --bus_range_max;
                if (bus_range_max == port_index) {
                  net_index = j;
                  break;
                }
              }
            } else {
              --bus_range_max;
            }
          } else if (stmt->isVerilogDclsStmt()) {
            auto* dcls = dynamic_cast<VerilogDcls*>(stmt);
            for (auto& dcl : dcls->get_verilog_dcls()) {
              if (Str::equal(dcl->get_dcl_name(), expr_net->get_verilog_id()->getBaseName())) {
                auto range = dcl->get_range();
                if (range) {
                  for (int j = range->first; ((range->first > range->second) ? j >= range->second : j <= range->second);
                       range->first > range->second ? --j : ++j) {
                    --bus_range_max;
                    if (bus_range_max == port_index) {
                      net_index = j;
                      break;
                    }
                  }
                } else {
                  --bus_range_max;
                }
                break;
              }
            }
          }
        } else {
          --bus_range_max;
        }
      }

      if (bus_range_max == port_index) {
        connect_net_expr = expr_net.get();
        break;
      } else if (bus_range_max < bus_range_min) {
        // should not beyond bus range min.
        break;
      }
    }

    LOG_FATAL_IF(!connect_net_expr) << "not found connect net.";

    if (net_index) {
      auto* connect_net_id = connect_net_expr->get_verilog_id();
      auto* index_verilog_id = new VerilogIndexID(connect_net_id->getBaseName(), *net_index);
      return new VerilogNetIDExpr(index_verilog_id, 0);
    } else {
      return connect_net_expr->copy();
    }
  };

  auto get_dcl_range = [](auto* dcl_stmt, const char* dcl_name) -> std::optional<std::pair<int, int>> {
    std::optional<std::pair<int, int>> range;
    if (dcl_stmt->isVerilogDclStmt()) {
      auto* verilog_dcl_stmt = (VerilogDcl*) (dcl_stmt);
      range = verilog_dcl_stmt->get_range();

    } else if (dcl_stmt->isVerilogDclsStmt()) {
      auto* verilog_dcls_stmt = (VerilogDcls*) (dcl_stmt);
      for (auto& verilog_dcl : verilog_dcls_stmt->get_verilog_dcls()) {
        if (Str::equal(verilog_dcl->get_dcl_name(), dcl_name)) {
          range = verilog_dcl->get_range();
          break;
        }
      }
    }
    return range;
  };

  // process the inst connection below.
  std::unique_ptr<VerilogNetExpr> port_connect_net;
  for (auto& port_connection : _port_connections) {
    if (Str::equal(port_connection->get_port_id()->getName(), port_id->getBaseName())) {
      if (port_connection->get_net_expr()) {
        port_connect_net = std::unique_ptr<VerilogNetExpr>(port_connection->get_net_expr()->copy());

        if (port_connect_net->isIDExpr()) {
          // is not concat expr.
          if (port_id->isBusIndexID() || port_id->isBusSliceID()) {
            // find port range first
            auto* port_dcl_stmt = inst_module->findDclStmt(port_id->getBaseName());
            auto port_range = get_dcl_range(port_dcl_stmt, port_id->getBaseName());
            assert(port_range);

            if (port_id->isBusIndexID()) {
              if (port_connect_net->get_verilog_id()->isBusSliceID()) {
                auto* port_connect_net_slice_id = dynamic_cast<VerilogSliceID*>(port_connect_net->get_verilog_id());

                int index_gap = std::abs(dynamic_cast<VerilogIndexID*>(port_id)->get_index() - port_range->first);
                if (port_connect_net_slice_id->get_range_from() > port_connect_net_slice_id->get_range_to()) {
                  index_gap = -index_gap;
                }
                auto new_port_connect_port_id = std::make_unique<VerilogIndexID>(port_connect_net_slice_id->getBaseName(),
                                                                                 port_connect_net_slice_id->get_range_from() + index_gap);
                port_connect_net->set_verilog_id(std::move(new_port_connect_port_id));
              } else if (!port_connect_net->get_verilog_id()->isBusIndexID()) {
                const char* port_connect_name = port_connect_net->get_verilog_id()->getBaseName();

                auto* port_connect_net_dcl_stmt = parent_module->findDclStmt(port_connect_name);
                auto port_connect_net_range = get_dcl_range(port_connect_net_dcl_stmt, port_connect_name);

                int port_index = dynamic_cast<VerilogIndexID*>(port_id)->get_index();
                int index_gap = std::abs(port_index - port_range->first);
                if (port_connect_net_range->first > port_connect_net_range->second) {
                  index_gap = -index_gap;
                }

                auto new_port_connect_port_id = port_connect_net_range ? std::make_unique<VerilogIndexID>(
                                                    port_connect_name, port_connect_net_range->first + index_gap)
                                                                       : std::make_unique<VerilogID>(port_connect_name);
                port_connect_net->set_verilog_id(std::move(new_port_connect_port_id));
              }
            } else if (port_id->isBusSliceID()) {
              if (port_connect_net->get_verilog_id()->isBusSliceID()) {
                auto* port_connect_net_slice_id = dynamic_cast<VerilogSliceID*>(port_connect_net->get_verilog_id());
                LOG_FATAL_IF(!port_connect_net_slice_id) << "port connect is not bus.";
                port_connect_net_slice_id->set_range_from(dynamic_cast<VerilogSliceID*>(port_id)->get_range_from());
                port_connect_net_slice_id->set_range_to(dynamic_cast<VerilogSliceID*>(port_id)->get_range_to());
              } else if (!port_connect_net->get_verilog_id()->isBusIndexID()) {
                auto new_port_connect_port_id = std::make_unique<VerilogSliceID>(port_connect_net->get_verilog_id()->getBaseName(),
                                                                                 dynamic_cast<VerilogSliceID*>(port_id)->get_range_from(),
                                                                                 dynamic_cast<VerilogSliceID*>(port_id)->get_range_to());
                port_connect_net->set_verilog_id(std::move(new_port_connect_port_id));
              }
            }

          } else {
            if (port_bus_wide_range) {
              // assert(0);
            }
          }
        } else if (port_connect_net->isConcatExpr()) {
          // should be concat expr.
          auto* port_concat_connect_net = dynamic_cast<VerilogNetConcatExpr*>(port_connect_net.get());
          LOG_FATAL_IF(!port_concat_connect_net);
          if (port_id->isBusIndexID()) {
            int index = dynamic_cast<VerilogIndexID*>(port_id)->get_index();
            auto* index_port_connect = get_concat_connect_net(port_concat_connect_net, index);
            LOG_FATAL_IF(!index_port_connect->isIDExpr() && !index_port_connect->isConstant()) << "should be id expr.";
            port_connect_net = std::unique_ptr<VerilogNetExpr>(index_port_connect);
          } else if (port_id->isBusSliceID()) {
            int from = dynamic_cast<VerilogSliceID*>(port_id)->get_range_from();
            int to = dynamic_cast<VerilogSliceID*>(port_id)->get_range_to();

            Vector<std::unique_ptr<VerilogNetExpr>> slice_concat;
            for (int index = from; ((from > to) ? index >= to : index <= to); ((from > to) ? --index : ++index)) {
              auto* index_port_connect = get_concat_connect_net(port_concat_connect_net, index);
              slice_concat.emplace_back(index_port_connect);
            }

            auto slice_concat_connect_net = std::make_unique<VerilogNetConcatExpr>(std::move(slice_concat), 0);
            port_connect_net = std::move(slice_concat_connect_net);
          }
        } else if (port_connect_net->isConstant()) {
          LOG_INFO << "port " << port_connection->get_port_id()->getName() << " connnect net is constant";
        } else {
          LOG_FATAL << "not support.";
        }
      }
      break;
    }
  }
  return port_connect_net;
}

VerilogDcl::VerilogDcl(DclType dcl_type, const char* dcl_name, int line) : VerilogStmt(line), _dcl_type(dcl_type), _dcl_name(dcl_name)
{
  Str::free(dcl_name);
}

VerilogDcls::VerilogDcls(std::vector<std::unique_ptr<VerilogDcl>>&& verilog_dcls, int line)
    : VerilogStmt(line), _verilog_dcls(std::move(verilog_dcls))
{
}

VerilogDcls::VerilogDcls(const VerilogDcls& orig) : VerilogStmt(orig)
{
  for (auto& verilog_dcl : orig._verilog_dcls) {
    _verilog_dcls.emplace_back(dynamic_cast<VerilogDcl*>(verilog_dcl.get()->copy()));
  }
}

VerilogDcls& VerilogDcls::operator=(const VerilogDcls& orig)
{
  if (this != &orig) {
    VerilogStmt::operator=(orig);
    for (auto& verilog_dcl : orig._verilog_dcls) {
      _verilog_dcls.emplace_back(dynamic_cast<VerilogDcl*>(verilog_dcl.get()->copy()));
    }
  }

  return *this;
}

VerilogModule::VerilogModule(const char* module_name, int line) : VerilogStmt(line), _module_name(module_name)
{
  Str::free(module_name);
}

VerilogAssign::VerilogAssign(VerilogNetExpr* left_net_expr, VerilogNetExpr* right_net_expr, int line)
    : VerilogStmt(line), _left_net_expr(left_net_expr), _right_net_expr(right_net_expr)
{
}
/**
 * @brief Flatten the hierarchical module.
 *
 * @param parent_module
 * @param inst_stmt the inst stmt belong to the parent module.
 * @param verilog_reader
 */
void VerilogModule::flattenModule(VerilogModule* parent_module, VerilogInst* inst_stmt, VerilogReader* verilog_reader)
{
  std::vector<VerilogStmt*> to_be_erased_stmts;
  bool have_sub_module;
  // flatten all sub module.
  do {
    have_sub_module = false;
    FOR_EACH_VERILOG_STMT(this, stmt)
    {
      if (stmt->isModuleInstStmt()) {
        auto* module_inst_stmt = dynamic_cast<VerilogInst*>(stmt.get());
        auto* sub_module = verilog_reader->findModule(module_inst_stmt->get_cell_name());
        if (sub_module) {
          have_sub_module = true;
          LOG_INFO << "flatten module " << module_inst_stmt->get_cell_name() << " inst " << module_inst_stmt->get_inst_name();

          sub_module->flattenModule(this, module_inst_stmt, verilog_reader);
          eraseStmt(module_inst_stmt);
          break;
        }
      }
    }

  } while (have_sub_module);

  if (parent_module) {
    // lambda function, process dcl stmt.
    auto process_dcl = [inst_stmt, parent_module, this](auto* stmt) {
      auto* dcl_stmt = dynamic_cast<VerilogDcl*>(stmt);
      const char* dcl_name = dcl_stmt->get_dcl_name();
      auto dcl_type = dcl_stmt->get_dcl_type();
      if ((dcl_type == VerilogDcl::DclType::kWire) && !isPort(dcl_name)) {
        std::string new_dcl_name = std::string(inst_stmt->get_inst_name()) + "/" + dcl_name;

        auto* new_dcl_stmt = dynamic_cast<VerilogDcl*>(dcl_stmt->copy());
        new_dcl_stmt->set_dcl_name(std::move(new_dcl_name));
        parent_module->addStmt(std::unique_ptr<VerilogStmt>(new_dcl_stmt));
      }
    };

    // lambda function, process inst stmt port connect.
    auto process_port_connect
        = [this, parent_module, inst_stmt](VerilogNetExpr* net_expr) -> std::optional<std::unique_ptr<VerilogNetExpr>> {
      LOG_FATAL_IF(!net_expr->isIDExpr()) << "net is not id expr.";
      auto find_dcl_stmt = [this](const char* net_base_name) -> std::optional<std::pair<int, int>> {
        auto* dcl_stmt = findDclStmt(net_base_name, true);
        std::optional<std::pair<int, int>> range;
        if (dcl_stmt) {
          if (dcl_stmt->isVerilogDclStmt()) {
            auto* verilog_dcl_stmt = (VerilogDcl*) (dcl_stmt);
            range = verilog_dcl_stmt->get_range();
          } else if (dcl_stmt->isVerilogDclsStmt()) {
            auto* verilog_dcls_stmt = (VerilogDcls*) (dcl_stmt);
            for (auto& verilog_dcl : verilog_dcls_stmt->get_verilog_dcls()) {
              if (Str::equal(verilog_dcl->get_dcl_name(), net_base_name)) {
                range = verilog_dcl->get_range();
                break;
              }
            }
          }
        }

        return range;
      };

      auto* net_expr_id = net_expr->get_verilog_id();
      const char* net_base_name = net_expr_id->getBaseName();
      std::optional<std::pair<int, int>> range;

      if (!isPort(net_base_name)) {
        // for common name, should check whether bus, get range first.
        if (!Str::contain(net_base_name, "/") && !net_expr_id->isBusIndexID() && !net_expr_id->isBusSliceID()) {
          range = find_dcl_stmt(net_base_name);
        }

        // not port, change net name to inst name / net_name.
        if (!range) {
          std::string new_net_base_name = std::string(inst_stmt->get_inst_name()) + "/" + net_base_name;
          net_expr_id->setBaseName(std::move(new_net_base_name));
          return std::nullopt;
        } else {
          Vector<std::unique_ptr<VerilogNetExpr>> verilog_id_concat;
          bool is_first_greate = range->first > range->second;
          for (int index = range->first; is_first_greate ? index >= range->second : index <= range->second;
               is_first_greate ? --index : ++index) {
            const char* new_net_name = Str::printf("%s/%s", inst_stmt->get_inst_name(), net_base_name);
            auto* index_id = new VerilogIndexID(new_net_name, index);
            auto new_index_net_id = std::make_unique<VerilogNetIDExpr>(index_id, 0);
            verilog_id_concat.emplace_back(std::move(new_index_net_id));
          }
          auto new_concat_net_id = std::make_unique<VerilogNetConcatExpr>(std::move(verilog_id_concat), 0);
          return new_concat_net_id;
        }

      } else {
        // is port, check the port whether port or port bus, then get
        // the port or port bus connect parent net.
        range = find_dcl_stmt(net_base_name);
        // get port connected parent module net.
        auto port_connect_net = inst_stmt->getPortConnectNet(parent_module, this, net_expr_id, range);
        return port_connect_net;
      }
    };

    std::function<void(std::unique_ptr<VerilogNetExpr>&)> process_concat_net_expr
        = [&process_port_connect, &process_concat_net_expr](std::unique_ptr<VerilogNetExpr>& one_net_expr) {
            if (one_net_expr->isIDExpr()) {
              auto port_connect_net = process_port_connect(one_net_expr.get());
              if (port_connect_net) {
                one_net_expr = std::move(*port_connect_net);
              }
            } else {
              if (one_net_expr->isConcatExpr()) {
                auto one_net_expr_concat = dynamic_cast<VerilogNetConcatExpr*>(one_net_expr.get());
                for (auto& one_net_expr_concat_net : one_net_expr_concat->get_verilog_id_concat()) {
                  process_concat_net_expr(one_net_expr_concat_net);
                }
              }
            }
          };

    FOR_EACH_VERILOG_STMT(this, stmt)
    {
      // for verilog dcl stmt, change the dcl name to inst name / dcl_name, then
      // add stmt to parent.
      if (stmt->isVerilogDclStmt()) {
        process_dcl(stmt.get());
      } else if (stmt->isVerilogDclsStmt()) {
        auto* dcls_stmt = dynamic_cast<VerilogDcls*>(stmt.get());
        for (auto& dcl_stmt : dcls_stmt->get_verilog_dcls()) {
          process_dcl(dcl_stmt.get());
        }
      } else if (stmt->isModuleInstStmt()) {
        // for verilog module instant stmt, first copy the module inst stmt,
        // then change the inst stmt connect net to net name / parent net
        // name(for port), next change the inst name to parent inst name /
        // current inst name.
        auto* module_inst_stmt = dynamic_cast<VerilogInst*>(stmt.get());
        auto* new_module_inst_stmt = dynamic_cast<VerilogInst*>(module_inst_stmt->copy());

        FOREACH_VERILOG_PORT_CONNECT(new_module_inst_stmt, port_connect)
        {
          auto* net_expr = port_connect->get_net_expr();
          if (net_expr) {
            if (net_expr->isIDExpr()) {
              auto port_connect_net = process_port_connect(net_expr);
              if (port_connect_net) {
                // is port connect net, set new net.
                port_connect->set_net_expr(std::move(*port_connect_net));
              }

            } else if (net_expr->isConcatExpr()) {
              auto* concat_connect_net = dynamic_cast<VerilogNetConcatExpr*>(net_expr);
              LOG_FATAL_IF(!concat_connect_net);

              for (auto& one_net_expr : concat_connect_net->get_verilog_id_concat()) {
                process_concat_net_expr(one_net_expr);
              }
            }
          }
        }

        const char* the_stmt_inst_name = module_inst_stmt->get_inst_name();
        std::string new_inst_name = std::string(inst_stmt->get_inst_name()) + "/" + the_stmt_inst_name;
        new_module_inst_stmt->set_inst_name(std::move(new_inst_name));
        parent_module->addStmt(std::unique_ptr<VerilogStmt>(new_module_inst_stmt));
      }
    }
  }
}

/**
 * @brief Read the verilog file.
 *
 * @param filename .v file.
 * @return true
 * @return false
 */
bool VerilogReader::read(const char* filename)
{
  _file_name = filename;
  gVerilogReader = this;

  bool success = false;

#ifdef ZLIB_FOUND

  auto* verilog_in = gzopen(filename, "rb");
  if (verilog_in) {
    parseBegin(verilog_in);
    success = (parse() == 0);
    LOG_FATAL_IF(!success) << "Read verilog file failed.";
    parseEnd(verilog_in);
  } else {
    LOG_FATAL << "The verilog file " << filename << " is not exist.";
  }
#else

  auto close_file = [](gzFile* file_handle) { gzclose(file_handle); };
  std::unique_ptr<gzFile, decltype(close_file)> file_handle(gzopen(filename, "rb"), close_file);
  if (file_handle) {
    gzFile* verilog_in = file_handle.get();
    parseBegin(verilog_in);
    success = (parse() == 0);
    LOG_FATAL_IF(!success) << "Read verilog file failed.";
    parseEnd(verilog_in);
  } else {
    LOG_FATAL << "The verilog file " << filename << " is not exist.";
  }

#endif
  return success;
}

/**
 * @brief find the verilog module accord name.
 *
 * @param module_name
 * @return auto&
 */
VerilogModule* VerilogReader::findModule(const char* module_name)
{
  if (_str2Module.contains(module_name)) {
    return _str2Module[module_name];
  }
  return nullptr;
}

/**
 * @brief The builder of declaration.
 *
 * @param dcl_type
 * @param dcl_name
 * @param dcl_args
 */
VerilogDcls* VerilogReader::makeDcl(VerilogDcl::DclType dcl_type, std::vector<const char*>&& dcl_args, int line)
{
  std::vector<std::unique_ptr<VerilogDcl>> declarations;
  for (const auto* dcl_name : dcl_args) {
    auto* verilg_dcl = new VerilogDcl(dcl_type, dcl_name, line);
    declarations.emplace_back(verilg_dcl);
  }
  return new VerilogDcls(std::move(declarations), line);
}

/**
 * @brief The builder of declaration.
 *
 * @param dcl_type
 * @param dcl_args
 * @param line
 * @return VerilogDcls*
 */
VerilogDcls* VerilogReader::makeDcl(VerilogDcl::DclType dcl_type, std::vector<const char*>&& dcl_args, int line, std::pair<int, int> range)
{
  std::vector<std::unique_ptr<VerilogDcl>> declarations;
  for (const auto* dcl_name : dcl_args) {
    auto* verilg_dcl = new VerilogDcl(dcl_type, dcl_name, line);
    verilg_dcl->set_range(range);
    declarations.emplace_back(verilg_dcl);
  }
  return new VerilogDcls(std::move(declarations), line);
}

/**
 * @brief The builder of port connect.
 *
 * @param port_name
 * @param net_name
 * @return VerilogPortRefPortConnect*
 */
VerilogPortRefPortConnect* VerilogReader::makePortConnect(VerilogID* port_id, VerilogNetExpr* net_expr)
{
  auto* port_ref_connect = new VerilogPortRefPortConnect(port_id, net_expr);
  return port_ref_connect;
}

/**
 * @brief The builder of verilog ID.
 *
 * @param id
 * @return VerilogID*
 */
VerilogID* VerilogReader::makeVerilogID(const char* id)
{
  auto* verilog_id = new VerilogID(id);
  return verilog_id;
}

/**
 * @brief The builder of verilog ID, such as rst_cnt[17].
 *
 * @param id
 * @param index
 * @return VerilogID*
 */
VerilogID* VerilogReader::makeVerilogID(const char* id, int index)
{
  auto* verilog_id = new VerilogIndexID(id, index);
  return verilog_id;
}

/**
 * @brief The builder of verilog ID, such as rst_cnt[17:0].
 *
 * @param id
 * @param range_from
 * @param range_to
 * @return VerilogID*
 */
VerilogID* VerilogReader::makeVerilogID(const char* id, int range_from, int range_to)
{
  auto* verilog_id = new VerilogSliceID(id, range_from, range_to);
  return verilog_id;
}

/**
 * @brief The builder of verilog net expression.
 *
 * @param verilog_id
 * @return VerilogNetExpr*
 */
VerilogNetExpr* VerilogReader::makeVerilogNetExpr(VerilogID* verilog_id, int line)
{
  auto* verilog_net_expr = new VerilogNetIDExpr(verilog_id, line);
  return verilog_net_expr;
}

/**
 * @brief The builder of verilog net concat expression.
 *
 * @param verilog_id_concat
 * @return VerilogNetExpr*
 */
VerilogNetExpr* VerilogReader::makeVerilogNetExpr(Vector<std::unique_ptr<VerilogNetExpr>>&& verilog_id_concat, int line)
{
  auto* verilog_net_expr = new VerilogNetConcatExpr(std::move(verilog_id_concat), line);
  return verilog_net_expr;
}

/**
 * @brief The builder of verilog constant expression.
 *
 * @param constant
 * @param line
 * @return VerilogNetExpr*
 */
VerilogNetExpr* VerilogReader::makeVerilogNetExpr(const char* constant, int line)
{
  auto* verilog_net_expr = new VerilogConstantExpr(constant, line);
  return verilog_net_expr;
}

/**
 * @brief The builder of liberty cell instance.
 *
 * @param liberty_cell_name
 * @param inst_name
 * @param port_connection
 * @param line
 * @return VerilogInst*
 */
VerilogInst* VerilogReader::makeModuleInst(const char* liberty_cell_name, const char* inst_name,
                                           std::vector<std::unique_ptr<VerilogPortRefPortConnect>>&& port_connection, int line)
{
  auto* module_inst = new VerilogInst(liberty_cell_name, inst_name, std::move(port_connection), line);
  return module_inst;
}

/**
 * @brief The builder of continuous assign statement.
 *
 * @param left_net_expr
 * @param right_net_expr
 * @param line
 * @return VerilogAssign*
 */
VerilogAssign* VerilogReader::makeModuleAssign(VerilogNetExpr* left_net_expr, VerilogNetExpr* right_net_expr, int line)
{
  auto* module_assign = new VerilogAssign(left_net_expr, right_net_expr, line);
  return module_assign;
}

/**
 * @brief The builder of verilog module.
 *
 * @param module_name
 * @param module_stmts
 * @return VerilogModule*
 */
VerilogModule* VerilogReader::makeModule(const char* module_name, std::vector<std::unique_ptr<VerilogStmt>>&& module_stmts, int line)
{
  auto verilog_module = std::make_unique<VerilogModule>(module_name, line);
  verilog_module->set_module_stmts(std::move(module_stmts));
  auto* ret_module = verilog_module.get();
  _verilog_modules.emplace_back(std::move(verilog_module));
  _str2Module[ret_module->get_module_name()] = ret_module;
  return ret_module;
}

/**
 * @brief The builder of verilog module include port lists.
 *
 * @param module_name
 * @param port_list
 * @param module_stmts
 * @param line
 * @return VerilogModule*
 */
VerilogModule* VerilogReader::makeModule(const char* module_name, std::vector<std::unique_ptr<VerilogID>>&& port_list,
                                         std::vector<std::unique_ptr<VerilogStmt>>&& module_stmts, int line)
{
  auto* verilog_module = makeModule(module_name, std::move(module_stmts), line);
  verilog_module->set_port_list(std::move(port_list));
  return verilog_module;
}

/**
 * @brief Flatten the hierarchical module to flatten module.
 *
 * @param module_name
 * @return VerilogModule*
 */
VerilogModule* VerilogReader::flattenModule(const char* module_name)
{
  LOG_INFO << "flatten module " << module_name << " start";
  auto* the_module = findModule(module_name);
  LOG_FATAL_IF(!the_module) << module_name << " is not found.";

  bool have_sub_module;
  do {
    have_sub_module = false;
    FOR_EACH_VERILOG_STMT(the_module, stmt)
    {
      if (stmt->isModuleInstStmt()) {
        auto* module_inst_stmt = dynamic_cast<VerilogInst*>(stmt.get());
        // iterator the sub module, reach to the deepest layer,
        // then iterate copy the inst to its parent module.
        auto* sub_module = findModule(module_inst_stmt->get_cell_name());
        if (sub_module) {
          have_sub_module = true;
          LOG_INFO << "flatten module " << module_inst_stmt->get_cell_name() << " inst " << module_inst_stmt->get_inst_name();

          sub_module->flattenModule(the_module, module_inst_stmt, this);
          the_module->eraseStmt(module_inst_stmt);
          break;
        }
      }
    }

  } while (have_sub_module);

  LOG_INFO << "flatten module " << module_name << " end";

  return the_module;
}

}  // namespace ista
