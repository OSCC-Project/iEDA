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
 * @file Sta.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of the top Sta class.
 * @version 0.1
 * @date 2020-11-27
 */

#include "Sta.hh"

#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <ranges>
#include <tuple>
#include <utility>

#include "Config.hh"
#include "StaAnalyze.hh"
#include "StaApplySdc.hh"
#include "StaBuildClockTree.hh"
#include "StaBuildGraph.hh"
#include "StaBuildPropTag.hh"
#include "StaBuildRCTree.hh"
#include "StaCheck.hh"
#include "StaClockPropagation.hh"
#include "StaClockSlewDelayPropagation.hh"
#include "StaConstPropagation.hh"
#include "StaCrossTalkPropagation.hh"
#include "StaDataPropagation.hh"
#include "StaDataSlewDelayPropagation.hh"
#include "StaDelayPropagation.hh"
#include "StaDump.hh"
#include "StaFindStartOrEnd.hh"
#include "StaGraph.hh"
#include "StaLevelization.hh"
#include "StaPathData.hh"
#include "StaReport.hh"
#include "StaSlewPropagation.hh"
#include "ThreadPool/ThreadPool.h"
#include "include/Version.hh"
#include "json/json.hpp"
#include "liberty/Lib.hh"
#include "log/Log.hh"
#include "netlist/NetlistWriter.hh"
#include "netlist/Pin.hh"
#include "sdc-cmd/Cmd.hh"
#include "sdc/SdcConstrain.hh"
#include "tcl/ScriptEngine.hh"
#include "time/Time.hh"
#include "usage/usage.hh"

#if CUDA_PROPAGATION
#include "propagation-cuda/lib_arc.cuh"
#endif

namespace ista {

static bool IsFileExists(const char *name) {
  std::ifstream f(name);
  bool is_exit = f.good();
  if (!is_exit) {
    LOG_FATAL << "File:" << name << " is not exist.";
  }
  f.close();
  return is_exit;
}

Sta *Sta::_sta = nullptr;

Sta::Sta()
    : _num_threads(32),
      _constrains(nullptr),
      _analysis_mode(AnalysisMode::kMaxMin),
      _graph(&_netlist),
      _clock_groups(sta_clock_cmp) {
  char config[] = "iSTA";
  char *argv[] = {config, nullptr};
  // We need to initialize the log system here, because Sta() may be called in
  // pybind, which does not have a main function to initialize the log system.
  Log::init(argv);

  _report_tbl_summary = StaReportPathSummary::createReportTable("sta");
  _report_tbl_TNS = StaReportClockTNS::createReportTable("TNS");
}

Sta::~Sta() = default;

/**
 * @brief Get the top sta instance, if not, create one.
 *
 * @return Sta*
 */
Sta *Sta::getOrCreateSta() {
  static std::mutex mt;
  if (_sta == nullptr) {
    std::lock_guard<std::mutex> lock(mt);
    if (_sta == nullptr) {
      _sta = new Sta();
    }
  }
  return _sta;
}

/**
 * @brief Destroy the sta.
 *
 */
void Sta::destroySta() {
  delete _sta;
  _sta = nullptr;
}

/**
 * @brief set sta report path.
 *
 * @param design_work_space
 */
void Sta::set_design_work_space(const char *design_work_space) {
  _design_work_space = design_work_space;
}

/**
 * @brief Get the constrains, if not, create one.
 *
 * @return sdcConstrain* The constrains.
 */
SdcConstrain *Sta::getConstrain() {
  if (!_constrains) {
    _constrains = std::make_unique<SdcConstrain>();
  }

  return _constrains.get();
}

/**
 * @brief the read design verilog netlist file.
 *
 * @param verilog_file
 * @return unsigned
 */
unsigned Sta::readDesignWithRustParser(const char *verilog_file) {
  LOG_INFO << "read design " << verilog_file << " start ";
  if (!IsFileExists(verilog_file)) {
    return 0;
  }

  readVerilogWithRustParser(verilog_file);
  auto &top_module_name = get_top_module_name();
  linkDesignWithRustParser(top_module_name.c_str());

  LOG_INFO << "read design " << verilog_file << " end ";
  return 1;
}

/**
 * @brief read the sdc file.
 *
 * @param sdc_file
 * @return unsigned
 */
unsigned Sta::readSdc(const char *sdc_file) {
  LOG_INFO << "read sdc " << sdc_file << " start ";
  if (!IsFileExists(sdc_file)) {
    return 0;
  }

  Sta::initSdcCmd();

  _constrains.reset();
  getConstrain();

  auto *script_engine = ScriptEngine::getOrCreateInstance();
  unsigned result =
      script_engine->evalString(Str::printf("source %s", sdc_file));

  LOG_FATAL_IF(result == 1)
      << ScriptEngine::getOrCreateInstance()->evalString(R"(puts $errorInfo)");

  LOG_INFO << "read sdc " << sdc_file << " end ";

  return 1;
}

/**
 * @brief read spef file.
 *
 * @param spef_file
 * @return unsigned
 */
unsigned Sta::readSpef(const char *spef_file) {
  LOG_INFO << "read spef " << spef_file << " start ";
  if (!IsFileExists(spef_file)) {
    return 0;
  }
  StaGraph &the_graph = get_graph();

  StaBuildRCTree func(spef_file, DelayCalcMethod::kElmore);
  func(&the_graph);

  LOG_INFO << "read spef " << spef_file << " end ";

  return 1;
}

/**
 * @brief read one aocv file.
 *
 * @param aocv_file
 * @return unsigned
 */
unsigned Sta::readAocv(const char *aocv_file) {
  if (!IsFileExists(aocv_file)) {
    return 0;
  }
  AocvReader aocv_reader(aocv_file);
  auto load_aocv = aocv_reader.readAocvLibrary();
  addAocv(std::move(load_aocv));
  return 1;
}

/**
 * @brief read aocv files.
 *
 * @param aocv_files
 * @return unsigned
 */
unsigned Sta::readAocv(std::vector<std::string> &aocv_files) {
  bool is_exit = true;
  for (auto &aocv_file : aocv_files) {
    if (!IsFileExists(aocv_file.c_str())) {
      is_exit = false;
    }
  }
  if (!is_exit) {
    return 0;
  }
  LOG_INFO << "load aocv start";

#if 0
  for (auto &aocv_file : aocv_files) {
    readAocv(aocv_file.c_str());
  }

#else

  {
    ThreadPool pool(get_num_threads());

    for (auto &aocv_file : aocv_files) {
      pool.enqueue([this, aocv_file]() { readAocv(aocv_file.c_str()); });
    }
  }

#endif

  LOG_INFO << "load aocv end";

  return 1;
}

/**
 * @brief read one lib file.
 *
 * @param lib_file
 * @return unsigned
 */
unsigned Sta::readLiberty(const char *lib_file) {
  LOG_INFO << "read liberty " << lib_file << " start ";

  if (!IsFileExists(lib_file)) {
    return 0;
  }

  Lib lib;
  auto load_lib = lib.loadLibertyWithRustParser(lib_file);
  addLibReaders(std::move(load_lib));

  LOG_INFO << "read liberty " << lib_file << " end ";

  return 1;
}

/**
 * @brief read liberty files.
 *
 * @param lib_files
 * @return unsigned
 */
unsigned Sta::readLiberty(std::vector<std::string> &lib_files) {
  LOG_INFO << "load lib start";

#if 0
  for (auto &lib_file : lib_files) {
    readLiberty(lib_file.c_str());
  }

#else

  {
    ThreadPool pool(get_num_threads());

    for (auto &lib_file : lib_files) {
      pool.enqueue([this, lib_file]() { readLiberty(lib_file.c_str()); });
    }
  }

#endif

  LOG_INFO << "load lib end";

  return 1;
}

/**
 * @brief Link liberty according the builded cells to construct the lib data, if
 * build cell is empty, link all.
 *
 * @return unsigned
 */
unsigned Sta::linkLibertys() {
  // if linked library, not repeat link.
  if (!_libs.empty()) {
    return 1;
  }

  auto link_lib = [this](auto &lib_rust_reader) {
    // master should load all lib cell.
    lib_rust_reader.set_build_cells(get_link_cells());
    lib_rust_reader.linkLib();
    auto lib = lib_rust_reader.get_library_builder()->takeLib();

    auto *lib_builder = lib_rust_reader.get_library_builder();
    delete lib_builder;

    addLib(std::move(lib));
  };

#if 0
  for (auto &lib_rust_reader : _lib_readers) {
    link_lib(lib_rust_reader);
  }

#else
  {
    ThreadPool pool(get_num_threads());

    for (auto &lib_rust_reader : _lib_readers) {
      pool.enqueue(
          [link_lib, &lib_rust_reader]() { link_lib(lib_rust_reader); });
    }
  }

#endif

  return 1;
}

/**
 * @brief Read the verilog file.
 *
 * @param verilog_file
 */
unsigned Sta::readVerilogWithRustParser(const char *verilog_file) {
  LOG_INFO << "read verilog file " << verilog_file << " start";
  if (!IsFileExists(verilog_file)) {
    return 0;
  }

  bool is_ok = _rust_verilog_reader.readVerilog(verilog_file);
  _rust_verilog_file_ptr = _rust_verilog_reader.get_verilog_file_ptr();
  LOG_WARNING_IF(!is_ok) << "read verilog file " << verilog_file << " failed.";
  LOG_INFO << "read verilog end";
  return is_ok;
}

/**
 * @brief collect linked cell to speed up liberty load.
 *
 */
void Sta::collectLinkedCell() {
  auto top_module_stmts = _rust_top_module->module_stmts;
  void *stmt;
  FOREACH_VEC_ELEM(&top_module_stmts, void, stmt) {
    if (rust_is_module_inst_stmt(stmt)) {
      RustVerilogInst *verilog_inst = rust_convert_verilog_inst(stmt);
      const char *liberty_cell_name = verilog_inst->cell_name;
      _link_cells.insert(std::string(liberty_cell_name));
    }
  }
}

/**
 * @brief Link the design file to design netlist use rust parser.
 *
 * @param top_cell_name
 */
void Sta::linkDesignWithRustParser(const char *top_cell_name) {
  LOG_INFO << "link design " << top_cell_name << " start";

  _rust_verilog_reader.flattenModule(top_cell_name);
  auto &rust_verilog_modules = _rust_verilog_reader.get_verilog_modules();
  _rust_verilog_modules = std::move(rust_verilog_modules);

  _rust_top_module = _rust_verilog_reader.get_top_module();
  LOG_FATAL_IF(!_rust_top_module) << "top module not found.";
  set_design_name(_rust_top_module->module_name);

  // collect linked cell for lib load.
  collectLinkedCell();
  // then link libs.
  linkLibertys();

  auto top_module_stmts = _rust_top_module->module_stmts;
  Netlist &design_netlist = _netlist;
  design_netlist.set_name(_rust_top_module->module_name);

  /*The verilog decalre statement process lookup table.*/
  std::map<DclType, std::function<DesignObject *(DclType, const char *)>>
      dcl_process = {
          {DclType::KInput,
           [&design_netlist](DclType dcl_type, const char *dcl_name) {
             Port in_port(dcl_name, PortDir::kIn);
             auto &ret_val = design_netlist.addPort(std::move(in_port));
             return &ret_val;
           }},
          {DclType::KOutput,
           [&design_netlist](DclType dcl_type, const char *dcl_name) {
             Port out_port(dcl_name, PortDir::kOut);
             auto &ret_val = design_netlist.addPort(std::move(out_port));
             return &ret_val;
           }},
          {DclType::KInout,
           [&design_netlist](DclType dcl_type, const char *dcl_name) {
             Port out_port(dcl_name, PortDir::kInOut);
             auto &ret_val = design_netlist.addPort(std::move(out_port));
             return &ret_val;
           }},
          {DclType::KWire,
           [&design_netlist](DclType dcl_type, const char *dcl_name) {
             Net net(dcl_name);
             auto &ret_val = design_netlist.addNet(std::move(net));
             return &ret_val;
           }}};

  /*process the verilog declare statement.*/
  auto process_dcl_stmt = [&dcl_process,
                           &design_netlist](auto *rust_verilog_dcl) {
    auto dcl_type = rust_verilog_dcl->dcl_type;
    const auto *raw_dcl_name = rust_verilog_dcl->dcl_name;
    auto dcl_range = rust_verilog_dcl->range;

    // for dcl ports and wire trimmed \ in name.
    std::string dcl_name = Str::trimmed(raw_dcl_name);
    dcl_name = Str::replace(dcl_name, R"(\\)", "");

    if (!dcl_range.has_value) {
      if (dcl_process.contains(dcl_type)) {
        dcl_process[dcl_type](dcl_type, dcl_name.c_str());
      } else {
        LOG_INFO << "not support the declaration " << dcl_name;
      }
    } else {
      auto bus_range = std::make_pair(dcl_range.start, dcl_range.end);
      for (int index = bus_range.second; index <= bus_range.first; index++) {
        // for port or wire bus, we split to one bye one port.
        const char *one_name = Str::printf("%s[%d]", dcl_name.c_str(), index);

        if (dcl_process.contains(dcl_type)) {
          auto *design_obj = dcl_process[dcl_type](dcl_type, one_name);
          if (design_obj->isPort()) {
            auto *port = dynamic_cast<Port *>(design_obj);
            if (index == bus_range.second) {
              unsigned bus_size = bus_range.first + 1;
              PortBus port_bus(dcl_name.c_str(), bus_range.first,
                               bus_range.second, bus_size,
                               port->get_port_dir());
              port_bus.addPort(index, port);
              auto &ret_val = design_netlist.addPortBus(std::move(port_bus));
              port->set_port_bus(&ret_val);
            } else {
              auto *found_port_bus =
                  design_netlist.findPortBus(dcl_name.c_str());
              found_port_bus->addPort(index, port);
              port->set_port_bus(found_port_bus);
            }
          }

        } else {
          LOG_INFO << "not support the declaration " << one_name;
        }
      }
    }
  };

  void *stmt;
  FOREACH_VEC_ELEM(&top_module_stmts, void, stmt) {
    if (rust_is_verilog_dcls_stmt(stmt)) {
      RustVerilogDcls *verilog_dcls_struct = rust_convert_verilog_dcls(stmt);
      auto verilog_dcls = verilog_dcls_struct->verilog_dcls;
      void *verilog_dcl = nullptr;
      FOREACH_VEC_ELEM(&verilog_dcls, void, verilog_dcl) {
        process_dcl_stmt(rust_convert_verilog_dcl(verilog_dcl));
      }
    } else if (rust_is_module_inst_stmt(stmt)) {
      RustVerilogInst *verilog_inst = rust_convert_verilog_inst(stmt);
      std::string inst_name = verilog_inst->inst_name;
      inst_name = Str::trimmed(inst_name.c_str());
      inst_name = Str::replace(inst_name, " ", "");
      inst_name = Str::replace(inst_name, R"(\\)", "");

      const char *liberty_cell_name = verilog_inst->cell_name;
      auto port_connections = verilog_inst->port_connections;

      auto *inst_cell = findLibertyCell(liberty_cell_name);

      if (!inst_cell) {
        LOG_INFO << "liberty cell " << liberty_cell_name << " is not exist.";
        continue;
      }

      Instance inst(inst_name.c_str(), inst_cell);

      /*lambda function create net for connect instance pin*/
      auto create_net_connection = [verilog_inst, inst_cell, &inst,
                                    &design_netlist](auto *cell_port_id,
                                                     auto *net_expr,
                                                     std::optional<int> index,
                                                     PinBus *pin_bus) {
        const char *cell_port_name;
        if (rust_is_id(cell_port_id)) {
          cell_port_name = rust_convert_verilog_id(cell_port_id)->id;
        } else if (rust_is_bus_index_id(cell_port_id)) {
          cell_port_name = rust_convert_verilog_index_id(cell_port_id)->id;
        } else {
          cell_port_name = rust_convert_verilog_slice_id(cell_port_id)->id;
        }

        auto *library_port_or_port_bus =
            inst_cell->get_cell_port_or_port_bus(cell_port_name);
        LOG_INFO_IF(!library_port_or_port_bus)
            << cell_port_name << " port is not found.";
        if (!library_port_or_port_bus) {
          return;
        }

        auto add_pin_to_net = [&design_netlist](Pin *inst_pin,
                                                std::string &net_name) {
          if (net_name.empty()) {
            return;
          }

          Net *the_net = design_netlist.findNet(net_name.c_str());
          if (the_net) {
            the_net->addPinPort(inst_pin);
          } else {
            // DLOG_INFO << "create net " << net_name;
            auto &created_net = design_netlist.addNet(Net(net_name.c_str()));

            created_net.addPinPort(inst_pin);
            the_net = &created_net;
          }
          // The same name port is default connect to net.
          if (auto *design_port = design_netlist.findPort(net_name.c_str());
              design_port && !the_net->isNetPinPort(design_port)) {
            the_net->addPinPort(design_port);
          }
        };

        auto add_pin_to_inst = [&inst, &add_pin_to_net, pin_bus](
                                   auto *pin_name, auto *library_port,
                                   std::optional<int> pin_index) -> Pin * {
          auto *inst_pin = inst.addPin(pin_name, library_port);
          if (pin_bus) {
            pin_bus->addPin(pin_index.value(), inst_pin);
          }

          return inst_pin;
        };

        LibPort *library_port = nullptr;
        std::string pin_name;
        std::string net_name;

        if (net_expr) {
          if (rust_is_id_expr(net_expr)) {
            auto *net_id = const_cast<void *>(
                rust_convert_verilog_net_id_expr(net_expr)->verilog_id);
            LOG_FATAL_IF(!net_id) << "The port connection " << cell_port_name
                                  << " net id is not exist "
                                  << "at line " << verilog_inst->line_no;

            if (rust_is_id(net_id)) {
              net_name = rust_convert_verilog_id(net_id)->id;
            } else if (rust_is_bus_index_id(net_id)) {
              net_name = rust_convert_verilog_index_id(net_id)->id;
            } else if (rust_is_bus_slice_id(net_id)) {
              net_name = rust_convert_verilog_slice_id(net_id)->id;
            }
            // fix net name contain backslash
            net_name = Str::trimBackslash(net_name);
            net_name = Str::trimmed(net_name.c_str());
          } else if (rust_is_constant(net_expr)) {
            LOG_INFO_FIRST_N(5) << "for the constant net need TODO.";
          }
        }

        if (!library_port_or_port_bus->isLibertyPortBus()) {
          library_port = dynamic_cast<LibPort *>(library_port_or_port_bus);
          pin_name = cell_port_name;
          auto *inst_pin =
              add_pin_to_inst(pin_name.c_str(), library_port, std::nullopt);

          add_pin_to_net(inst_pin, net_name);

        } else {
          auto *library_port_bus =
              dynamic_cast<LibPortBus *>(library_port_or_port_bus);
          if (index) {
            library_port = (*library_port_bus)[index.value()];
            pin_name = Str::printf("%s[%d]", cell_port_name, index.value());
            auto *inst_pin =
                add_pin_to_inst(pin_name.c_str(), library_port, index.value());

            add_pin_to_net(inst_pin, net_name);

          } else {
            for (size_t i = 0; i < library_port_bus->getBusSize(); ++i) {
              library_port = (*library_port_bus)[i];
              pin_name = Str::printf("%s[%d]", cell_port_name, i);
              auto *inst_pin =
                  add_pin_to_inst(pin_name.c_str(), library_port, i);

              std::string net_index_name;
              if (rust_is_bus_slice_id(const_cast<void *>(
                      rust_convert_verilog_net_id_expr(net_expr)
                          ->verilog_id)) ||
                  rust_is_bus_slice_id(const_cast<void *>(
                      rust_convert_verilog_constant_expr(net_expr)
                          ->verilog_id))) {
                void *verilog_id = nullptr;
                if (rust_is_id_expr(net_expr)) {
                  verilog_id = const_cast<void *>(
                      rust_convert_verilog_net_id_expr(net_expr)->verilog_id);
                } else if (rust_is_constant(net_expr)) {
                  verilog_id = const_cast<void *>(
                      rust_convert_verilog_constant_expr(net_expr)->verilog_id);
                }
                auto *net_slice_id = rust_convert_verilog_slice_id(verilog_id);

                net_index_name = rust_get_index_name(
                    net_slice_id, i + net_slice_id->range_base);
              } else {
                net_index_name = Str::printf("%s[%d]", net_name.c_str(), i);
              }

              add_pin_to_net(inst_pin, net_index_name);
            }
          }
        }
      };

      /*lambda function flatten concate net, which maybe nested.*/
      std::function<void(RustVerilogNetConcatExpr *, std::vector<void *> &)>
          flatten_concat_net_expr =
              [&flatten_concat_net_expr](
                  RustVerilogNetConcatExpr *net_concat_expr,
                  std::vector<void *> &net_concat_vec) {
                auto verilog_id_concat = net_concat_expr->verilog_id_concat;

                void *verilog_id;
                FOREACH_VEC_ELEM(&verilog_id_concat, void, verilog_id) {
                  if (rust_is_concat_expr(verilog_id)) {
                    flatten_concat_net_expr(
                        rust_convert_verilog_net_concat_expr(verilog_id),
                        net_concat_vec);
                  } else {
                    net_concat_vec.push_back(verilog_id);
                  }
                }
              };

      // create net
      void *port_connection;
      FOREACH_VEC_ELEM(&port_connections, void, port_connection) {
        LOG_FATAL_IF(!port_connection)
            << "The inst " << inst_name << " at line " << verilog_inst->line_no
            << " port connection is null";
        RustVerilogPortRefPortConnect *rust_port_connection =
            rust_convert_verilog_port_ref_port_connect(port_connection);
        // *const c_void
        void *cell_port_id = const_cast<void *>(rust_port_connection->port_id);
        // *mut c_void
        void *net_expr = rust_port_connection->net_expr;

        // create pin bus
        const char *cell_port_name;
        if (rust_is_id(cell_port_id)) {
          cell_port_name = rust_convert_verilog_id(cell_port_id)->id;
        } else if (rust_is_bus_index_id(cell_port_id)) {
          cell_port_name = rust_convert_verilog_index_id(cell_port_id)->id;
        } else {
          cell_port_name = rust_convert_verilog_slice_id(cell_port_id)->id;
        }

        auto *library_port_bus =
            inst_cell->get_cell_port_or_port_bus(cell_port_name);
        std::unique_ptr<PinBus> pin_bus;
        if (library_port_bus->isLibertyPortBus()) {
          auto bus_size =
              dynamic_cast<LibPortBus *>(library_port_bus)->getBusSize();
          pin_bus = std::make_unique<PinBus>(cell_port_name, bus_size - 1, 0,
                                             bus_size);
        }

        if (!net_expr || rust_is_id_expr(net_expr) ||
            rust_is_constant(net_expr)) {
          create_net_connection(cell_port_id, net_expr, std::nullopt,
                                pin_bus.get());
        } else {
          LOG_FATAL_IF(!pin_bus) << "pin bus is null.";
          auto *net_concat_expr =
              rust_convert_verilog_net_concat_expr(net_expr);

          std::vector<void *> verilog_id_concat_vec;
          flatten_concat_net_expr(net_concat_expr, verilog_id_concat_vec);

          for (int i = (verilog_id_concat_vec.size() - 1);
               auto *verilog_id_net_expr : verilog_id_concat_vec) {
            create_net_connection(cell_port_id, verilog_id_net_expr, i--,
                                  pin_bus.get());
          }
        }

        if (pin_bus) {
          inst.addPinBus(std::move(pin_bus));
        }
      }

      design_netlist.addInstance(std::move(inst));
    }
  }

  // build assign stmt
  // record the merge nets.
  std::map<std::string, Net *> remove_to_merge_nets;
  auto process_assign_one_to_one_net = [&design_netlist, &remove_to_merge_nets](
                                           std::string left_net_name,
                                           std::string right_net_name) {
    left_net_name = Str::trimmed(left_net_name.c_str());
    right_net_name = Str::trimmed(right_net_name.c_str());

    left_net_name = Str::replace(left_net_name, R"(\\)", "");
    right_net_name = Str::replace(right_net_name, R"(\\)", "");

    // for debug
    // if (Str::contain(left_net_name.c_str(), "io_master_araddr[0]")) {
    //   LOG_INFO << "debug";
    // }

    Net *the_left_net = design_netlist.findNet(left_net_name.c_str());
    if (!the_left_net && remove_to_merge_nets.contains(left_net_name)) {
      the_left_net = remove_to_merge_nets[left_net_name];
    }

    Net *the_right_net = design_netlist.findNet(right_net_name.c_str());
    if (!the_right_net && remove_to_merge_nets.contains(right_net_name)) {
      the_right_net = remove_to_merge_nets[right_net_name];
    }

    auto *the_left_port = design_netlist.findPort(left_net_name.c_str());
    auto *the_right_port = design_netlist.findPort(right_net_name.c_str());

    if (the_left_net && the_right_net && !the_left_port && !the_right_port) {
      LOG_INFO << "merge " << left_net_name << " = " << right_net_name << "\n";

      auto left_pin_ports = the_left_net->get_pin_ports();

      // merge left to right net.
      for (auto *left_pin_port : left_pin_ports) {
        the_left_net->removePinPort(left_pin_port);
        the_right_net->addPinPort(left_pin_port);
      }

      remove_to_merge_nets[left_net_name] = the_right_net;

    } else if (the_left_net && !the_left_port) {
      // assign net = input_port;
      if (the_right_port) {
        the_left_net->addPinPort(the_right_port);
      } else {
        LOG_ERROR << "the right port is not exist.";
      }

    } else if (the_right_net && !the_right_port) {
      // assign output_port = net;
      if (the_left_port) {
        the_right_net->addPinPort(the_left_port);
      } else {
        LOG_ERROR << "the left port is not exist.";
      }

    } else if (!the_right_net && !the_left_net && the_right_port) {
      // assign output_port = input_port;

      auto &created_net = design_netlist.addNet(Net(right_net_name.c_str()));
      LOG_FATAL_IF(!the_left_port) << "the left port is not exist.";
      created_net.addPinPort(the_left_port);
      LOG_FATAL_IF(!the_right_port) << "the right port is not exist.";
      created_net.addPinPort(the_right_port);

    } else if (!the_right_net && !the_left_net && !the_right_port) {
      // assign output_port = 1'b0(1'b1);

      auto &created_net = design_netlist.addNet(Net(left_net_name.c_str()));
      LOG_FATAL_IF(!the_left_port) << "the left port is not exist.";
      created_net.addPinPort(the_left_port);

    } else if (the_left_net && the_right_net && the_left_port &&
               the_right_port) {
      // assign output_port = output_port
      LOG_FATAL_IF(!the_right_port) << "the right port is not exist.";
      the_left_net->addPinPort(the_right_port);
    } else {
      LOG_FATAL << "assign " << left_net_name << " = " << right_net_name
                << " is not processed.";
    }

    // remove ununsed nets.
    if (the_left_net && the_left_net->get_pin_ports().size() == 0) {
      // update the remove to merge nets before remove net.
      for (auto it = remove_to_merge_nets.begin();
           it != remove_to_merge_nets.end();) {
        auto &merge_net = it->second;
        if (merge_net == the_left_net) {
          if (the_right_net && the_right_net->get_pin_ports().size() > 0) {
            it->second = the_right_net;
            ++it;
          } else {
            it = remove_to_merge_nets.erase(it);
          }
        } else {
          ++it;
        }
      }

      design_netlist.removeNet(the_left_net);
      the_left_net = nullptr;
    }

    if (the_right_net && the_right_net->get_pin_ports().size() == 0) {
      // update the remove to merge nets before remove net.
      for (auto it = remove_to_merge_nets.begin();
           it != remove_to_merge_nets.end();) {
        auto &merge_net = it->second;
        if (merge_net == the_right_net) {
          if (the_left_net) {
            merge_net = the_left_net;
            ++it;
          } else {
            it = remove_to_merge_nets.erase(it);
          }
        } else {
          ++it;
        }
      }

      design_netlist.removeNet(the_right_net);
    }

    LOG_INFO << "assign " << left_net_name << " = " << right_net_name << "\n";
  };

  FOREACH_VEC_ELEM(&top_module_stmts, void, stmt) {
    if (rust_is_module_assign_stmt(stmt)) {
      RustVerilogAssign *verilog_assign = rust_convert_verilog_assign(stmt);
      auto *left_net_expr = const_cast<void *>(verilog_assign->left_net_expr);
      auto *right_net_expr = const_cast<void *>(verilog_assign->right_net_expr);
      std::string left_net_name;
      std::string right_net_name;
      if (rust_is_id_expr(left_net_expr) && rust_is_id_expr(right_net_expr)) {
        // get left_net_name.
        auto *left_net_id = const_cast<void *>(
            rust_convert_verilog_net_id_expr(left_net_expr)->verilog_id);
        if (rust_is_id(left_net_id)) {
          left_net_name = rust_convert_verilog_id(left_net_id)->id;
        } else if (rust_is_bus_index_id(left_net_id)) {
          left_net_name = rust_convert_verilog_index_id(left_net_id)->id;
        } else {
          left_net_name = rust_convert_verilog_slice_id(left_net_id)->id;
        }
        // get right_net_name.
        auto *right_net_id = const_cast<void *>(
            rust_convert_verilog_net_id_expr(right_net_expr)->verilog_id);
        if (rust_is_id(right_net_id)) {
          right_net_name = rust_convert_verilog_id(right_net_id)->id;
        } else if (rust_is_bus_index_id(right_net_id)) {
          right_net_name = rust_convert_verilog_index_id(right_net_id)->id;
        } else {
          right_net_name = rust_convert_verilog_slice_id(right_net_id)->id;
        }

        process_assign_one_to_one_net(left_net_name, right_net_name);

      } else if ((rust_is_id_expr(left_net_expr) &&
                  rust_is_concat_expr(right_net_expr)) ||
                 (rust_is_concat_expr(left_net_expr) &&
                  rust_is_id_expr(right_net_expr))) {
        auto process_id_concat_assign = [&process_assign_one_to_one_net](
                                            auto *id_net_expr,
                                            auto *concat_net_expr,
                                            bool is_first_left) {
          std::string id_net_name;
          std::string concat_net_name;

          // assume left the not concatenation, right is concatenation. such as
          // "assign io_out_arsize = { _41_, _41_, io_in_size };"
          auto *id_net_expr_id = const_cast<void *>(
              rust_convert_verilog_net_id_expr(id_net_expr)->verilog_id);
          unsigned base_id_index = 0;
          if (rust_is_id(id_net_expr_id)) {
            id_net_name = rust_convert_verilog_id(id_net_expr_id)->id;
          } else if (rust_is_bus_slice_id(id_net_expr_id)) {
            auto slice_net_id = rust_convert_verilog_slice_id(id_net_expr_id);
            id_net_name = slice_net_id->base_id;
            base_id_index = slice_net_id->range_base;
          } else {
            std::cout << "left net id should be id or bus slice id";
            assert(false);
          }

          auto verilog_id_concat =
              rust_convert_verilog_net_concat_expr(concat_net_expr)
                  ->verilog_id_concat;

          void *one_net_expr;
          FOREACH_VEC_ELEM(&verilog_id_concat, void, one_net_expr) {
            assert(rust_is_id_expr(one_net_expr));
            auto *one_net_id =
                (void *)(rust_convert_verilog_net_id_expr(one_net_expr)
                             ->verilog_id);
            if (rust_is_id(one_net_id)) {
              std::string one_id_net_name =
                  id_net_name + "[" + std::to_string(base_id_index) + "]";
              std::string one_concat_net_name =
                  rust_convert_verilog_id(one_net_id)->id;
              if (is_first_left) {
                process_assign_one_to_one_net(one_id_net_name,
                                              one_concat_net_name);
              } else {
                process_assign_one_to_one_net(one_concat_net_name,
                                              one_id_net_name);
              }

            } else if (rust_is_bus_index_id(one_net_id)) {
              std::string one_id_net_name =
                  id_net_name + "[" + std::to_string(base_id_index) + "]";
              std::string one_concat_net_name =
                  rust_convert_verilog_index_id(one_net_id)->id;
              if (is_first_left) {
                process_assign_one_to_one_net(one_id_net_name,
                                              one_concat_net_name);
              } else {
                process_assign_one_to_one_net(one_concat_net_name,
                                              one_id_net_name);
              }
            } else {
              auto right_slice_id = rust_convert_verilog_slice_id(one_net_id);
              std::string right_net_base_name = right_slice_id->base_id;
              auto right_base_index = right_slice_id->range_base;
              while (right_base_index <= right_slice_id->range_max) {
                std::string one_id_net_name =
                    id_net_name + "[" + std::to_string(base_id_index) + "]";
                std::string one_concat_net_name =
                    right_net_base_name + "[" +
                    std::to_string(right_base_index) + "]";

                if (is_first_left) {
                  process_assign_one_to_one_net(one_id_net_name,
                                                one_concat_net_name);
                } else {
                  process_assign_one_to_one_net(one_concat_net_name,
                                                one_id_net_name);
                }

                ++base_id_index;
                ++right_base_index;
              }
            }

            ++base_id_index;
          }
        };

        if (rust_is_id_expr(left_net_expr) &&
            rust_is_concat_expr(right_net_expr)) {
          process_id_concat_assign(left_net_expr, right_net_expr, true);
        } else {
          process_id_concat_assign(right_net_expr, left_net_expr, false);
        }

      } else if (rust_is_concat_expr(left_net_expr) &&
                 rust_is_concat_expr(right_net_expr)) {
        std::function<std::vector<std::string>(RustVec &)>
            get_concat_net_names =
                [&get_concat_net_names](
                    RustVec &verilog_id_concat) -> std::vector<std::string> {
          std::vector<std::string> concat_net_names;
          void *one_net_expr;
          FOREACH_VEC_ELEM(&verilog_id_concat, void, one_net_expr) {
            if (rust_is_id_expr(one_net_expr)) {
              auto *one_net_id =
                  (void *)(rust_convert_verilog_net_id_expr(one_net_expr)
                               ->verilog_id);
              if (rust_is_id(one_net_id)) {
                std::string one_concat_net_name =
                    rust_convert_verilog_id(one_net_id)->id;
                concat_net_names.emplace_back(std::move(one_concat_net_name));
              } else if (rust_is_bus_index_id(one_net_id)) {
                std::string one_concat_net_name =
                    rust_convert_verilog_index_id(one_net_id)->id;
                concat_net_names.emplace_back(std::move(one_concat_net_name));
              } else {
                auto right_slice_id = rust_convert_verilog_slice_id(one_net_id);
                std::string right_net_base_name = right_slice_id->base_id;
                auto right_base_index = right_slice_id->range_base;
                while (right_base_index <= right_slice_id->range_max) {
                  std::string one_concat_net_name =
                      right_net_base_name + "[" +
                      std::to_string(right_base_index) + "]";
                  concat_net_names.emplace_back(std::move(one_concat_net_name));
                  ++right_base_index;
                }
              }
            } else if (rust_is_concat_expr(one_net_expr)) {
              auto one_net_concat_expr =
                  rust_convert_verilog_net_concat_expr(one_net_expr);
              auto one_net_verilog_id_concat =
                  one_net_concat_expr->verilog_id_concat;
              auto one_concat_net_names =
                  get_concat_net_names(one_net_verilog_id_concat);
              concat_net_names.insert(concat_net_names.end(),
                                      one_concat_net_names.begin(),
                                      one_concat_net_names.end());

            } else {
              assert(false);
            }
          }

          return concat_net_names;
        };

        auto left_concat_net_expr =
            rust_convert_verilog_net_concat_expr(left_net_expr);
        auto left_concat_net_names =
            get_concat_net_names(left_concat_net_expr->verilog_id_concat);

        auto right_concat_net_expr =
            rust_convert_verilog_net_concat_expr(right_net_expr);
        auto right_concat_net_names =
            get_concat_net_names(right_concat_net_expr->verilog_id_concat);

        assert(left_concat_net_names.size() == right_concat_net_names.size());

        for (size_t i = 0; i < left_concat_net_names.size(); i++) {
          // process assign net = net, which is concat net.
          std::string left_concat_net_name = left_concat_net_names[i];
          std::string right_concat_net_name = right_concat_net_names[i];

          if (left_concat_net_name == right_concat_net_name) {
            // skip same net name.
            continue;
          }

          process_assign_one_to_one_net(left_concat_net_name,
                                        right_concat_net_name);
        }

      } else {
        LOG_FATAL
            << "assign declaration's lhs/rhs is not VerilogNetIDExpr class.";
      }
    }
  }

  rust_free_verilog_file(_rust_verilog_file_ptr);
  LOG_INFO << "link design " << top_cell_name << " end";

  LOG_INFO << "design " << top_cell_name
           << " inst num: " << design_netlist.getInstanceNum();
  LOG_INFO << "design " << top_cell_name
           << " net num: " << design_netlist.getNetNum();
  LOG_INFO << "design " << top_cell_name
           << " port num: " << design_netlist.getPortNum();
}

/**
 * @brief get the design used libs.
 *
 * @return std::set<LibLibrary *>
 */
std::set<LibLibrary *> Sta::getUsedLibs() {
  std::set<LibLibrary *> used_libs;
  Instance *inst;
  FOREACH_INSTANCE(&_netlist, inst) {
    auto *used_lib = inst->get_inst_cell()->get_owner_lib();
    used_libs.insert(used_lib);
  }

  return used_libs;
}

/**
 * @brief reset constraint.
 */
void Sta::resetConstraint() { _constrains.reset(); }

/**
 * @brief Find the liberty cell from the lib.
 *
 * @param cell_name
 * @return LibCell*
 */
LibCell *Sta::findLibertyCell(const char *cell_name) {
  LibCell *found_cell = nullptr;
  for (auto &lib : _libs) {
    if (found_cell = lib->findCell(cell_name); found_cell) {
      break;
    }
  }
  return found_cell;
}

/**
 * @brief Find the data aocv object spec from the aocv.
 *
 * @param cell_name
 * @return std::optional<AocvObjectSpecSet *>
 */
std::optional<AocvObjectSpecSet *> Sta::findDataAocvObjectSpecSet(
    const char *object_name) {
  std::optional<AocvObjectSpecSet *> aocv_objectspec_set = std::nullopt;
  for (auto &aocv : _aocvs) {
    if (aocv_objectspec_set = aocv->findDataAocvObjectSpecSet(object_name);
        aocv_objectspec_set) {
      break;
    }
  }
  return aocv_objectspec_set;
}

/**
 * @brief Find the clock aocv object spec from the aocv.
 *
 * @param cell_name
 * @return std::optional<AocvObjectSpecSet *>
 */
std::optional<AocvObjectSpecSet *> Sta::findClockAocvObjectSpecSet(
    const char *object_name) {
  std::optional<AocvObjectSpecSet *> aocv_objectspec_set = std::nullopt;
  for (auto &aocv : _aocvs) {
    if (aocv_objectspec_set = aocv->findClockAocvObjectSpecSet(object_name);
        aocv_objectspec_set) {
      break;
    }
  }
  return aocv_objectspec_set;
}

/**
 * @brief Make the function equivalently liberty cell map.
 *
 * @param equiv_libs
 */
void Sta::makeClassifiedCells(std::vector<LibLibrary *> &equiv_libs) {
  if (_classified_cells) {
    _classified_cells.reset();
  }

  _classified_cells = std::make_unique<LibClassifyCell>();
  _classified_cells->classifyLibCell(equiv_libs);
}

/**
 * @brief Get the equivalently liberty cell.
 *
 * @param cell
 * @return Vector<LibCell *>*
 */
Vector<LibCell *> *Sta::classifyCells(LibCell *cell) {
  if (_classified_cells)
    return _classified_cells->getClassOfCell(cell);
  else
    return nullptr;
}

/**
 * @brief Init all supported ista cmd.
 *
 */
void Sta::initSdcCmd() {
  registerTclCmd(CmdCreateClock, "create_clock");
  registerTclCmd(CmdCreateGeneratedClock, "create_generated_clock");
  registerTclCmd(CmdSetInputTransition, "set_input_transition");
  // set_clock_transition share the set_input_transition process.
  registerTclCmd(CmdSetInputTransition, "set_clock_transition");
  registerTclCmd(CmdSetDrivingCell, "set_driving_cell");
  registerTclCmd(CmdSetLoad, "set_load");
  registerTclCmd(CmdSetInputDelay, "set_input_delay");
  registerTclCmd(CmdSetOutputDelay, "set_output_delay");
  registerTclCmd(CmdSetMaxFanout, "set_max_fanout");
  registerTclCmd(CmdSetMaxTransition, "set_max_transition");
  registerTclCmd(CmdSetMaxCapacitance, "set_max_capacitance");
  registerTclCmd(CmdCurrentDesign, "current_design");
  registerTclCmd(CmdGetClocks, "get_clocks");
  registerTclCmd(CmdGetPins, "get_pins");
  registerTclCmd(CmdGetPorts, "get_ports");
  registerTclCmd(CmdGetCells, "get_cells");
  registerTclCmd(CmdGetLibs, "get_libs");
  registerTclCmd(CmdAllClocks, "all_clocks");
  registerTclCmd(CmdAllInputs, "all_inputs");
  registerTclCmd(CmdAllOutputs, "all_outputs");
  registerTclCmd(CmdSetPropagatedClock, "set_propagated_clock");
  registerTclCmd(CmdSetClockGroups, "set_clock_groups");
  registerTclCmd(CmdSetMulticyclePath, "set_multicycle_path");
  registerTclCmd(CmdSetFalsePath, "set_false_path");
  registerTclCmd(CmdSetMaxDelay, "set_max_delay");
  registerTclCmd(CmdSetMinDelay, "set_min_delay");
  registerTclCmd(CmdSetTimingDerate, "set_timing_derate");
  registerTclCmd(CmdSetClockUncertainty, "set_clock_uncertainty");
  registerTclCmd(CmdSetUnits, "set_units");
  registerTclCmd(CmdGroupPath, "group_path");
  registerTclCmd(CmdSetOperatingConditions, "set_operating_conditions");
  registerTclCmd(CmdSetWireLoadMode, "set_wire_load_mode");
  registerTclCmd(CmdSetDisableTiming, "set_disable_timing");
  registerTclCmd(CmdSetCaseAnalysis, "set_case_analysis");
}

/**
 * @brief Get the sta clock accord the clock name.
 *
 * @param clock_name
 * @return StaClock*
 */
StaClock *Sta::findClock(const char *clock_name) {
  for (auto &clock : _clocks) {
    if (Str::equal(clock->get_clock_name(), clock_name)) {
      return clock.get();
    }
  }
  return nullptr;
}

/**
 * @brief get the fastest clock.
 *
 * @return StaClock*
 */
StaClock *Sta::getFastestClock() {
  StaClock *fastest_clock = nullptr;
  std::ranges::for_each(_clocks, [&fastest_clock](auto &clock) {
    if (!fastest_clock) {
      fastest_clock = clock.get();
    } else {
      if (fastest_clock->get_period() > clock->get_period()) {
        fastest_clock = clock.get();
      }
    }
  });
  return fastest_clock;
}

/**
 * @brief set all clock latency, such as all to zero.
 *
 * @param latency
 */
void Sta::setIdealClockNetworkLatency(double latency) {
  for (auto &clock : _clocks) {
    clock->set_ideal_clock_network_latency(NS_TO_PS(latency));
  }
}

/**
 * @brief set some clock latency.
 *
 * @param clock_name
 * @param latency
 */
void Sta::setIdealClockNetworkLatency(const char *clock_name, double latency) {
  auto *sta_clock = findClock(clock_name);
  if (sta_clock) {
    sta_clock->set_ideal_clock_network_latency(latency);
  }
}

/**
 * @brief Add io delay constrain.
 *
 * @param port_vertex
 * @param io_delay_constrain
 */
void Sta::addIODelayConstrain(StaVertex *port_vertex,
                              SdcSetIODelay *io_delay_constrain) {
  _io_delays.insert(port_vertex, io_delay_constrain);
}

/**
 * @brief Get the io delay constrain of port vertex.
 *
 * @param port_vertex
 * @return std::list<SdcSetIODelay*>
 */
std::list<SdcSetIODelay *> Sta::getIODelayConstrain(StaVertex *port_vertex) {
  return _io_delays.values(port_vertex);
}

/**
 * @brief Find the vertex.
 *
 * @param pin_name
 * @return StaVertex*
 */
StaVertex *Sta::findVertex(const char *pin_name) {
  auto objs = _netlist.findObj(pin_name, false, false);
  if (!objs.empty()) {
    auto *obj = objs.front();
    auto the_vertex = _graph.findVertex(obj);
    return the_vertex ? *the_vertex : nullptr;
  }
  return nullptr;
}

/**
 * @brief Get the vertex slew limit.
 *
 * @param vertex
 * @param mode
 * @param trans_type
 * @return double
 */
std::optional<double> Sta::getVertexSlewLimit(StaVertex *the_vertex,
                                              AnalysisMode mode,
                                              TransType trans_type) {
  std::optional<double> limit;
  std::optional<double> slew_limit;
  if (auto *obj = the_vertex->get_design_obj(); obj->isPin()) {
    auto *port = dynamic_cast<Pin *>(obj)->get_cell_port();
    slew_limit = port->get_port_slew_limit(mode);
    if (!slew_limit) {
      slew_limit =
          port->get_ower_cell()->get_owner_lib()->get_default_max_transition();
    }
  }

  auto global_slew_limit =
      (trans_type == TransType::kRise) ? getMaxRiseSlew() : getMaxFallSlew();

  auto vertex_slew_limit = (trans_type == TransType::kRise)
                               ? the_vertex->getMaxRiseSlew()
                               : the_vertex->getMaxFallSlew();

  if (vertex_slew_limit) {
    limit = vertex_slew_limit;
  } else if (global_slew_limit) {
    limit = global_slew_limit;
  } else if (slew_limit) {
    limit = slew_limit;
  }

  return limit;
}

/**
 * @brief Get the vertex slew slack.
 *
 * @param the_vertex
 * @param mode
 * @param trans_type
 * @return std::optional<double>
 */
std::optional<double> Sta::getVertexSlewSlack(StaVertex *the_vertex,
                                              AnalysisMode mode,
                                              TransType trans_type) {
  std::optional<double> slack;

  auto slew = the_vertex->getSlewNs(mode, trans_type);
  auto limit = getVertexSlewLimit(the_vertex, mode, trans_type);

  if (limit && slew) {
    slack = *limit - *slew;
  }

  return slack;
}

/**
 * @brief Get the vertex cap limit.
 *
 * @param the_vertex
 * @param mode
 * @param trans_type
 * @return std::optional<double>
 */
std::optional<double> Sta::getVertexCapacitanceLimit(StaVertex *the_vertex,
                                                     AnalysisMode mode,
                                                     TransType trans_type) {
  std::optional<double> limit;
  std::optional<double> cap_limit;
  if (auto *obj = the_vertex->get_design_obj(); obj->isPin()) {
    auto *port = dynamic_cast<Pin *>(obj)->get_cell_port();
    cap_limit = port->get_port_cap_limit(mode);
  }

  auto global_cap_limit =
      (trans_type == TransType::kRise) ? getMaxRiseCap() : getMaxFallCap();

  auto vertex_cap_limit = (trans_type == TransType::kRise)
                              ? the_vertex->getMaxRiseCap()
                              : the_vertex->getMaxFallCap();
  if (vertex_cap_limit) {
    limit = vertex_cap_limit;
  } else if (global_cap_limit) {
    limit = global_cap_limit;
  } else if (cap_limit) {
    limit = cap_limit;
  }

  return limit;
}

/**
 * @brief Get the driver vertex capacitance.
 *
 * @param the_vertex
 * @param mode
 * @param trans_type
 * @return double
 */
double Sta::getVertexCapacitance(StaVertex *the_vertex, AnalysisMode mode,
                                 TransType trans_type) {
  double capacitance = 0.0;

  auto get_cap = [mode, trans_type](auto *obj) {
    double capacitance = 0.0;
    if (obj->isPin()) {
      auto *port = dynamic_cast<Pin *>(obj)->get_cell_port();
      auto port_cap = port->get_port_cap(mode, trans_type);

      if (!port_cap) {
        capacitance = port->get_port_cap();
      } else {
        capacitance = *port_cap;
      }
    } else if (obj->isPort()) {
      double port_cap = dynamic_cast<Port *>(obj)->cap(mode, trans_type);
      capacitance = port_cap;
    }
    return capacitance;
  };
  auto *obj = the_vertex->get_design_obj();
  if (obj->isPin() && obj->isOutput()) {
    auto *pin = dynamic_cast<Pin *>(obj);
    if (auto *net = pin->get_net(); net) {
      auto loads = net->getLoads();
      for (auto *load_pin : loads) {
        capacitance += get_cap(load_pin);
      }
    }

  } else {
    capacitance = get_cap(obj);
  }
  return capacitance;
}

/**
 * @brief GEt the vertex cap slack.
 *
 * @param the_vertex
 * @param mode
 * @param trans_type
 * @return std::optional<double>
 */
std::optional<double> Sta::getVertexCapacitanceSlack(StaVertex *the_vertex,
                                                     AnalysisMode mode,
                                                     TransType trans_type) {
  std::optional<double> slack;
  // only consider driver cap.
  double capacitance = getVertexCapacitance(the_vertex, mode, trans_type);
  auto limit = getVertexCapacitanceLimit(the_vertex, mode, trans_type);

  if (limit) {
    slack = *limit - capacitance;
  }

  return slack;
}

/**
 * @brief Get the driver fanout limit.
 *
 * @param the_vertex
 * @param mode
 * @param trans_type
 * @return std::optional<double>
 */
std::optional<double> Sta::getDriverVertexFanoutLimit(StaVertex *the_vertex,
                                                      AnalysisMode mode) {
  std::optional<double> limit;
  std::optional<double> fanout_limit;
  if (auto *obj = the_vertex->get_design_obj();
      obj->isPin() && obj->isOutput()) {
    auto *port = dynamic_cast<Pin *>(obj)->get_cell_port();

    fanout_limit =
        port->get_ower_cell()->get_owner_lib()->get_default_max_fanout();
  }

  auto global_fanout_limit = getMaxFanout();
  auto vertex_fanout_limit = the_vertex->getMaxFanout();

  if (vertex_fanout_limit) {
    limit = vertex_fanout_limit;
  } else if (global_fanout_limit) {
    limit = global_fanout_limit;
  } else if (fanout_limit) {
    limit = fanout_limit;
  }

  return limit;
}

/**
 * @brief Get the driver vertex fanout limit.
 *
 * @param the_vertex
 * @param mode
 * @return std::optional<double>
 */
std::optional<double> Sta::getDriverVertexFanoutSlack(StaVertex *the_vertex,
                                                      AnalysisMode mode) {
  std::optional<double> slack;
  auto *obj = the_vertex->get_design_obj();
  if ((obj->isPin() && obj->isOutput()) || (obj->isPort() && obj->isInput())) {
    auto fanout = the_vertex->get_src_arcs().size();
    auto limit = getDriverVertexFanoutLimit(the_vertex, mode);

    if (limit) {
      slack = *limit - static_cast<double>(fanout);
    }
  }
  return slack;
}

/**
 * @brief build the graph data.
 *
 * @return unsigned
 */
unsigned Sta::buildGraph() {
  StaGraph &the_graph = get_graph();
  Vector<std::function<unsigned(StaGraph *)>> funcs = {StaBuildGraph()};
  for (auto &func : funcs) {
    the_graph.exec(func);
  }

  return 1;
}

#if CUDA_PROPAGATION
/**
 * @brief build the gpu liberty arc.
 *
 * @return unsigned
 */
unsigned Sta::buildLibArcsGPU() {
  StaGraph *the_graph = &get_graph();

  // collect all used libs.
  auto collect_lib_arc = [the_graph](auto &lib_arc_to_sta_arc) {
    std::set<LibArc *> all_used_lib_arcs;

    StaArc *the_arc;
    FOREACH_ARC(the_graph, the_arc) {
      if (the_arc->isInstArc()) {
        if (the_arc->isDelayArc() || the_arc->isCheckArc()) {
          auto *the_inst_arc = dynamic_cast<StaInstArc *>(the_arc);
          auto *the_lib_arc = the_inst_arc->get_lib_arc();
          all_used_lib_arcs.insert(the_lib_arc);

          lib_arc_to_sta_arc[the_lib_arc].emplace_back(the_inst_arc);
        }
      }
    }

    return all_used_lib_arcs;
  };

  std::map<LibArc *, std::vector<StaInstArc *>> lib_arc_to_sta_arc;
  auto all_used_lib_arcs = collect_lib_arc(lib_arc_to_sta_arc);

  std::vector<ista::Lib_Arc_GPU> lib_arcs_gpu;
  for (auto *the_lib_arc : all_used_lib_arcs) {
    auto *table_model = the_lib_arc->get_table_model();
    LibTableModel *delay_or_check_table_model;
    unsigned num_table;

    if (table_model->isDelayModel()) {
      delay_or_check_table_model =
          dynamic_cast<LibDelayTableModel *>(table_model);
      num_table = dynamic_cast<LibDelayTableModel *>(table_model)->kTableNum;
    } else {
      delay_or_check_table_model =
          dynamic_cast<LibCheckTableModel *>(table_model);
      num_table = dynamic_cast<LibCheckTableModel *>(table_model)->kTableNum;
    }

    Lib_Arc_GPU lib_gpu_arc;

    lib_gpu_arc._line_no = delay_or_check_table_model->get_line_no();
    lib_gpu_arc._num_table = num_table;
    auto lib_cap_unit =
        the_lib_arc->get_owner_cell()->get_owner_lib()->get_cap_unit();
    lib_gpu_arc._cap_unit =
        ((lib_cap_unit == CapacitiveUnit::kFF) ? Lib_Cap_unit::kFF
                                               : Lib_Cap_unit::kPF);
    auto lib_time_unit =
        the_lib_arc->get_owner_cell()->get_owner_lib()->get_time_unit();
    lib_gpu_arc._time_unit =
        ((lib_time_unit == TimeUnit::kNS)   ? Lib_Time_unit::kNS
         : (lib_time_unit == TimeUnit::kPS) ? Lib_Time_unit::kPS
                                            : Lib_Time_unit::kFS);

    lib_gpu_arc._table = new Lib_Table_GPU[lib_gpu_arc._num_table];

    for (size_t index = 0; index < num_table; index++) {
      auto *table = delay_or_check_table_model->getTable(index);

      Lib_Table_GPU gpu_table;

      if (!table) {
        lib_gpu_arc._table[index] = gpu_table;
        continue;
      }

      if (table->getAxesSize() == 0) {
        // (TODO totaosimin), need to process no axes table.
        lib_gpu_arc._table[index] = gpu_table;
        continue;
      }

      // set the x axis.
      auto &x_axis = table->getAxis(0);
      auto &x_axis_values = x_axis.get_axis_values();
      gpu_table._num_x = static_cast<unsigned>(x_axis_values.size());
      gpu_table._x = new float[gpu_table._num_x];
      for (unsigned i = 0; i < x_axis_values.size(); ++i) {
        gpu_table._x[i] = x_axis_values[i]->getFloatValue();
      }

      auto axes_size = table->get_axes().size();
      LOG_FATAL_IF(axes_size > 2);

      // set the y axis.
      if (axes_size > 1) {
        auto &y_axis = table->getAxis(1);
        auto &y_axis_values = y_axis.get_axis_values();
        gpu_table._num_y = static_cast<unsigned>(y_axis_values.size());
        gpu_table._y = new float[gpu_table._num_y];
        for (unsigned i = 0; i < y_axis_values.size(); ++i) {
          gpu_table._y[i] = y_axis_values[i]->getFloatValue();
        }
      }

      auto *table_template = table->get_table_template();
      if (axes_size == 1) {
        if (*(table_template->get_template_variable1()) ==
                LibLutTableTemplate::Variable::INPUT_NET_TRANSITION ||
            *(table_template->get_template_variable1()) ==
                LibLutTableTemplate::Variable::RELATED_PIN_TRANSITION ||
            *(table_template->get_template_variable1()) ==
                LibLutTableTemplate::Variable::INPUT_TRANSITION_TIME) {
          gpu_table._type = 0;  //(x axis denotes slew.)
        } else {
          gpu_table._type = 1;  //(x axis denotes constrain_slew_or_load.)
        }
      } else {
        if (*(table_template->get_template_variable1()) ==
                LibLutTableTemplate::Variable::INPUT_NET_TRANSITION ||
            *(table_template->get_template_variable1()) ==
                LibLutTableTemplate::Variable::RELATED_PIN_TRANSITION ||
            *(table_template->get_template_variable1()) ==
                LibLutTableTemplate::Variable::INPUT_TRANSITION_TIME) {
          gpu_table._type = 2;  // (x axis denotes slew, y axis denotes
                                // constrain_slew_or_load.)
        } else {
          gpu_table._type = 3;  //(x axis denotes constrain_slew_or_load, y axis
                                // denotes slew.)
        }
      }

      // set the values.
      auto &table_values = table->get_table_values();
      gpu_table._num_values = static_cast<unsigned>(table_values.size());
      gpu_table._values = new float[gpu_table._num_values];
      for (unsigned i = 0; i < table_values.size(); ++i) {
        gpu_table._values[i] = table_values[i]->getFloatValue();
      }

      // printLibTableGPU(gpu_table);
      // set the gpu table to the arc.(cpu index is the same as gpu index)
      lib_gpu_arc._table[index] = gpu_table;
    }

    auto &lib_gpu_arc_data = lib_arcs_gpu.emplace_back(std::move(lib_gpu_arc));
    for (auto *the_inst_arc : lib_arc_to_sta_arc[the_lib_arc]) {
      the_inst_arc->set_lib_gpu_arc(&lib_gpu_arc_data);
      the_inst_arc->set_lib_arc_id(lib_arcs_gpu.size() - 1);
    }
  }

  set_lib_gpu_arcs(std::move(lib_arcs_gpu));

  return 1;
}
#endif
/**
 * @brief Insert the seq path data.
 *
 */
unsigned Sta::insertPathData(StaClock *capture_clock, StaVertex *end_vertex,
                             StaSeqPathData *seq_data) {
  auto p = _clock_groups.find(capture_clock);
  if (p == _clock_groups.end()) {
    auto seq_path_group = std::make_unique<StaSeqPathGroup>(capture_clock);
    seq_path_group->insertPathData(end_vertex, seq_data);
    _clock_groups[capture_clock] = std::move(seq_path_group);
  } else {
    p->second->insertPathData(end_vertex, seq_data);
  }

  return 1;
}

unsigned Sta::insertPathData(StaVertex *end_vertex,
                             StaClockGatePathData *clock_gate_data) {
  if (!_clock_gate_group) {
    const char *clock_group = "**clock_gating_default**";
    _clock_gate_group = std::make_unique<StaClockGatePathGroup>(clock_group);
  }
  _clock_gate_group->insertPathData(end_vertex, clock_gate_data);
  return 1;
}

/**
 * @brief set the report froms tos build tag.
 *
 * @param prop_froms
 */
void Sta::setReportSpec(std::vector<std::string> &&prop_froms,
                        std::vector<std::string> &&prop_tos) {
  StaReportSpec report_spec;
  report_spec.set_prop_froms(std::move(prop_froms));
  report_spec.set_prop_tos(std::move(prop_tos));
  set_report_spec(std::move(report_spec));
}

/**
 * @brief set the report froms through to build tag.
 *
 * @param prop_froms
 */
void Sta::setReportSpec(std::vector<std::string> &&prop_froms,
                        std::vector<StaReportSpec::ReportList> &&prop_throughs,
                        std::vector<std::string> &&prop_tos) {
  StaReportSpec report_spec;
  report_spec.set_prop_froms(std::move(prop_froms));
  report_spec.set_prop_throughs(std::move(prop_throughs));
  report_spec.set_prop_tos(std::move(prop_tos));

  set_report_spec(std::move(report_spec));
}

/**
 * @brief Report path in text file.
 *
 * @param rpt_file_name The report text file name.
 * @return unsigned 1 if success, 0 else fail.
 */
unsigned Sta::reportPath(const char *rpt_file_name, bool is_derate,
                         bool only_wire_path) {
  auto report_path =
      [this](StaReportPathSummary &report_path_func) -> unsigned {
    unsigned is_ok = 1;

    auto path_group = get_path_group();  // specify path group.

    for (auto &&[capture_clock, seq_path_group] : _clock_groups) {
      if (!path_group ||
          path_group.value() == capture_clock->get_clock_name()) {
        is_ok = report_path_func(seq_path_group.get());
        if (!is_ok) {
          break;
        }
      }
    }

    if (_clock_gate_group) {
      is_ok = report_path_func(_clock_gate_group.get());
    }

    return is_ok;
  };

  auto report_path_of_mode = [&report_path, this, rpt_file_name, only_wire_path,
                              is_derate](AnalysisMode mode) -> unsigned {
    unsigned is_ok = 1;
    if ((get_analysis_mode() == mode) ||
        (get_analysis_mode() == AnalysisMode::kMaxMin)) {
      unsigned n_worst = get_n_worst_path_per_clock();

      StaReportPathSummary report_path_summary(rpt_file_name, mode, n_worst);
      report_path_summary.set_significant_digits(get_significant_digits());
      report_path_summary.enableJsonReport(isJsonReportEnabled());

      StaReportPathDetail report_path_detail(rpt_file_name, mode, n_worst,
                                             is_derate);
      report_path_detail.set_significant_digits(get_significant_digits());

      StaReportClockTNS report_path_TNS(rpt_file_name, mode, 1);
      report_path_TNS.set_significant_digits(get_significant_digits());
      report_path_TNS.enableJsonReport(isJsonReportEnabled());

      std::vector<StaReportPathSummary *> report_funcs{
          &report_path_summary, &report_path_detail, &report_path_TNS};

      // StaReportPathDump report_path_dump(rpt_file_name, mode, n_worst);
      StaReportPathYaml report_path_dump(rpt_file_name, mode, n_worst);

      if (c_print_delay_yaml) {
        report_funcs.emplace_back(&report_path_dump);
      }

      StaReportWirePathYaml report_wire_dump(rpt_file_name, mode, n_worst);
      if (c_print_wire_yaml) {
        report_funcs.emplace_back(&report_wire_dump);
      }

      StaReportWirePathJson report_wire_dump_json(rpt_file_name, mode, n_worst);
      if (c_print_wire_json) {
        report_funcs.emplace_back(&report_wire_dump_json);
      }

      StaReportPathDetailJson report_path_detail_json(rpt_file_name, mode,
                                                      n_worst, is_derate);

      if (isJsonReportEnabled()) {
        report_funcs.emplace_back(&report_path_detail_json);
      }

      for (auto *report_fun : report_funcs) {
        if (only_wire_path) {
          if (dynamic_cast<StaReportWirePathJson *>(report_fun)) {
            is_ok = report_path(*report_fun);
          }

        } else {
          is_ok = report_path(*report_fun);
        }
      }
    }

    return is_ok;
  };

  unsigned is_ok = report_path_of_mode(AnalysisMode::kMax);
  is_ok &= report_path_of_mode(AnalysisMode::kMin);

  if (!is_ok) {
    return is_ok;
  }

  LOG_INFO << "\n" << _report_tbl_summary->c_str();
  LOG_INFO << "\n" << _report_tbl_TNS->c_str();

  auto close_file = [](std::FILE *fp) { std::fclose(fp); };

  std::unique_ptr<std::FILE, decltype(close_file)> f(
      std::fopen(rpt_file_name, "w"), close_file);

  std::fprintf(f.get(), "Generate the report at %s, GitVersion: %s.\n",
               Time::getNowWallTime(), GIT_VERSION);
  std::fprintf(f.get(), "%s", _report_tbl_summary->c_str());  // WNS
  // report_TNS;
  std::fprintf(f.get(), "%s", _report_tbl_TNS->c_str());

  for (auto &report_tbl_detail : _report_tbl_details) {
    std::fprintf(f.get(), "%s", report_tbl_detail->c_str());
  }

  if (isJsonReportEnabled()) {
    nlohmann::json dump_json;
    dump_json["summary"] = _summary_json_report;
    dump_json["slack"] = _slack_json_report;
    dump_json["detail"] = _detail_json_report;

    auto *report_path = Str::printf("%s.json", rpt_file_name);

    std::ofstream out_file(report_path);
    if (out_file.is_open()) {
      out_file << dump_json.dump(4);  // 4 spaces indent
      LOG_INFO << "JSON report written to: " << report_path;
      out_file.close();
    } else {
      LOG_ERROR << "Failed to open JSON report file: " << report_path;
    }
  }

  return 1;
}

/**
 * @brief report trans slack.
 *
 * @param rpt_file_name
 * @return unsigned
 */
unsigned Sta::reportTrans(const char *rpt_file_name) {
  unsigned n_worst = get_n_worst_path_per_clock();
  StaReportTrans report_trans(rpt_file_name, AnalysisMode::kMax, n_worst);
  report_trans(this);

  return 1;
}

/**
 * @brief report cap slack.
 *
 * @param rpt_file_name
 * @return unsigned
 */
unsigned Sta::reportCap(const char *rpt_file_name, bool is_clock_cap) {
  unsigned n_worst = get_n_worst_path_per_clock();
  StaReportCap report_cap(rpt_file_name, AnalysisMode::kMax, n_worst,
                          is_clock_cap);
  report_cap(this);

  return 1;
}

/**
 * @brief report fanout slack.
 *
 * @param rpt_file_name
 * @return unsigned
 */
unsigned Sta::reportFanout(const char *rpt_file_name) {
  unsigned n_worst = get_n_worst_path_per_clock();
  StaReportFanout report_fanout(rpt_file_name, AnalysisMode::kMax, n_worst);
  report_fanout(this);

  return 1;
}

/**
 * @brief report the path skew.
 *
 * @param rpt_file_name
 * @param analysis_mode
 * @return unsigned
 */
unsigned Sta::reportSkew(const char *rpt_file_name,
                         AnalysisMode analysis_mode) {
  unsigned n_worst = get_n_worst_path_per_clock();

  StaReportSkewSummary report_skew_summary(rpt_file_name, analysis_mode,
                                           n_worst);
  report_skew_summary(this);

  StaReportSkewDetail report_skew_detail(rpt_file_name, analysis_mode, n_worst);
  report_skew_detail(this);

  auto close_file = [](std::FILE *fp) { std::fclose(fp); };

  std::unique_ptr<std::FILE, decltype(close_file)> f(
      std::fopen(rpt_file_name, "w"), close_file);

  std::fprintf(f.get(), "Generate the report at %s, GitVersion: %s.\n",
               Time::getNowWallTime(), GIT_VERSION);

  for (auto &report_tbl_skew : report_skew_summary.get_report_path_skews()) {
    std::fprintf(f.get(), "Clock: %s\n", report_tbl_skew->get_tbl_name());
    std::fprintf(f.get(), "%s", report_tbl_skew->c_str());
  }

  for (auto &report_tbl_skew : report_skew_detail.get_report_path_skews()) {
    std::fprintf(f.get(), "%s", report_tbl_skew->c_str());
  }

  return 1;
}

/**
 * @brief report one net in yaml format
 *
 * @param rpt_file_name
 * @param net
 * @return unsigned
 */
unsigned Sta::reportNet(const char *rpt_file_name, Net *the_net) {
  YAML::Node node;

  auto *rc_net = getRcNet(the_net);

  node["net_name"] = the_net->get_name();
  node["fanout"] = the_net->getLoads().size();
  node["driver"] = the_net->getDriver()->getFullName();

  auto driver_vertex = get_graph().findVertex(the_net->getDriver());

  unsigned index = 0;
  DesignObject *pin_port;
  FOREACH_NET_PIN(the_net, pin_port) {
    if (pin_port == the_net->getDriver()) {
      continue;
    }

    auto load_vertex = get_graph().findVertex(pin_port);

    YAML::Node load_node;
    std::string load_node_name = Str::printf("node_%d", index++);
    node[load_node_name] = load_node;

    load_node["load"] = pin_port->getFullName();

    FOREACH_MODE_TRANS(mode, trans) {
      YAML::Node one_node;
      std::string mode_str = (mode == AnalysisMode::kMax) ? "max" : "min";
      std::string trans_str = (trans == TransType::kRise) ? "rise" : "fall";
      std::string node_name = mode_str + "_" + trans_str;
      load_node[node_name] = one_node;

      one_node["total_capacitance_pf"] = (*driver_vertex)->getLoad(mode, trans);

      auto vertex_slew = (*driver_vertex)->getSlewNs(mode, trans);
      one_node["input_slew_ns"] = vertex_slew ? *vertex_slew : 0.0;

      auto vertex_resistance = (*load_vertex)->getResistance(mode, trans);
      one_node["load_resistance"] = vertex_resistance;

      one_node["load_delay_ns"] =
          rc_net->delayNs(*pin_port, RcNet::DelayMethod::kElmore).value_or(0.0);
    }
  }

  std::ofstream file(rpt_file_name, std::ios::trunc);
  file << node << std::endl;
  file.close();

  return 1;
}

/**
 * @brief report all net information.
 *
 * @param rpt_file_name
 * @return unsigned
 */
unsigned Sta::reportNet() {
  std::string design_work_space = get_design_work_space();
  std::string now_time = Time::getNowWallTime();
  std::string tmp = Str::replace(now_time, ":", "_");
  std::string path_dir = design_work_space + "/net_" + tmp;
  std::filesystem::create_directories(path_dir);

  auto *nl = get_netlist();
  Net *net;
  FOREACH_NET(nl, net) {
    auto *net_name = net->get_name();
    std::string rpt_file_name =
        Str::printf("%s/net_%s.yaml", path_dir.c_str(), net_name);
    reportNet(rpt_file_name.c_str(), net);
  }
  return 1;
}

/**
 * @brief report the specify path.
 *
 * @param rpt_file_name
 * @param analysis_mode
 * @param from_pin
 * @param through_pin
 * @param to_pin
 * @return unsigned
 */
unsigned Sta::reportFromThroughTo(const char *rpt_file_name,
                                  AnalysisMode analysis_mode,
                                  const char *from_pin, const char *through_pin,
                                  const char *to_pin) {
  StaReportSpecifyPath report_specify_path(rpt_file_name, analysis_mode,
                                           from_pin, through_pin, to_pin);
  report_specify_path(this);

  auto close_file = [](std::FILE *fp) { std::fclose(fp); };

  std::unique_ptr<std::FILE, decltype(close_file)> f(
      std::fopen(rpt_file_name, "w"), close_file);

  std::fprintf(f.get(), "Generate the report at %s, GitVersion: %s.\n",
               Time::getNowWallTime(), GIT_VERSION);

  for (auto &report_tbl_detail : _report_tbl_details) {
    std::fprintf(f.get(), "%s", report_tbl_detail->c_str());
  }

  return 1;
}

/**
 * @brief Get the end vertex seq data.
 *
 * @param delay_data
 * @return StaSeqPathData*
 */
std::vector<StaSeqPathData *> Sta::getSeqData(StaVertex *vertex,
                                              StaData *delay_data) {
  std::vector<StaSeqPathData *> seq_data_vec;
  for (const auto &[clk, seq_path_group] : _clock_groups) {
    StaPathEnd *path_end = seq_path_group->findPathEndData(vertex);
    if (path_end) {
      StaPathData *path_data =
          path_end->findPathData(dynamic_cast<StaPathDelayData *>(delay_data));
      if (path_data) {
        auto *seq_data = dynamic_cast<StaSeqPathData *>(path_data);
        seq_data_vec.emplace_back(seq_data);
      }
    }
  }

  return seq_data_vec;
}

/**
 * @brief Get WNS.
 *
 * @param clock_name
 * @param mode
 * @param trans_type
 * @return double
 */
double Sta::getWNS(const char *clock_name, AnalysisMode mode) {
  double WNS = 0;

  for (const auto &[clk, seq_path_group] : _clock_groups) {
    if (Str::equal(clk->get_clock_name(), clock_name)) {
      auto cmp = [](StaPathData *left, StaPathData *right) -> bool {
        int left_slack = left->getSlack();
        int right_slack = right->getSlack();
        return left_slack > right_slack;
      };

      std::priority_queue<StaPathData *, std::vector<StaPathData *>,
                          decltype(cmp)>
          seq_data_queue(cmp);

      StaPathEnd *path_end;
      StaPathData *path_data;
      FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end)
      FOREACH_PATH_END_DATA(path_end, mode, path_data) {
        seq_data_queue.push(path_data);
      }
      auto *worst_seq_data = seq_data_queue.top();
      WNS = FS_TO_NS(worst_seq_data->getSlack());
      break;
    }
  }
  return WNS;
}

/**
 * @brief Get TNS.
 *
 * @param clock_name
 * @param mode
 * @param trans_type
 * @return double TNS.
 */
double Sta::getTNS(const char *clock_name, AnalysisMode mode) {
  auto cmp = [](StaPathData *left, StaPathData *right) -> bool {
    int left_slack = left->getSlack();
    int right_slack = right->getSlack();
    return left_slack > right_slack;
  };

  std::priority_queue<StaPathData *, std::vector<StaPathData *>, decltype(cmp)>
      seq_data_queue(cmp);

  for (const auto &[clk, seq_path_group] : _clock_groups) {
    if (Str::equal(clk->get_clock_name(), clock_name)) {
      StaPathEnd *path_end;
      StaPathData *path_data;
      FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end)
      FOREACH_PATH_END_DATA(path_end, mode, path_data) {
        seq_data_queue.push(path_data);
      }
    }
  }

  double TNS = 0;
  while (!seq_data_queue.empty()) {
    auto *seq_path_data = dynamic_cast<StaSeqPathData *>(seq_data_queue.top());

    if (seq_path_data->getSlack() < 0) {
      TNS += FS_TO_NS(seq_path_data->getSlack());

    } else {
      break;
    }
    seq_data_queue.pop();
  }

  return TNS;
}

/**
 * @brief get local skew.
 *
 * @param clock_name
 * @param mode
 * @param trans_type
 * @return double local_skew.
 */
double Sta::getLocalSkew(const char *clock_name, AnalysisMode mode,
                         TransType trans_type) {
  auto cmp = [](StaPathData *left, StaPathData *right) -> bool {
    int left_skew = left->getSkew();
    int right_skew = right->getSkew();
    return left_skew > right_skew;
  };

  std::priority_queue<StaPathData *, std::vector<StaPathData *>, decltype(cmp)>
      seq_data_queue(cmp);

  for (const auto &[clk, seq_path_group] : _clock_groups) {
    if (Str::equal(clk->get_clock_name(), clock_name)) {
      StaPathEnd *path_end;
      StaPathData *path_data;
      FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end)
      FOREACH_PATH_END_DATA(path_end, mode, path_data) {
        seq_data_queue.push(path_data);
      }
    }
  }

  double local_skew = 0;
  while (!seq_data_queue.empty()) {
    auto *seq_path_data = dynamic_cast<StaSeqPathData *>(seq_data_queue.top());
    local_skew += FS_TO_NS(seq_path_data->getSkew());

    seq_data_queue.pop();
  }

  return local_skew;
}

/**
 * @brief get global skew.
 *
 * @param mode
 * @param trans_type
 * @return double global skew.
 */
double Sta::getGlobalSkew(AnalysisMode mode, TransType trans_type) {
  auto cmp = [](StaPathData *left, StaPathData *right) -> bool {
    int left_skew = left->getSkew();
    int right_skew = right->getSkew();
    return left_skew > right_skew;
  };

  std::priority_queue<StaPathData *, std::vector<StaPathData *>, decltype(cmp)>
      seq_data_queue(cmp);

  for (const auto &[clk, seq_path_group] : _clock_groups) {
    StaPathEnd *path_end;
    StaPathData *path_data;
    FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end)
    FOREACH_PATH_END_DATA(path_end, mode, path_data) {
      seq_data_queue.push(path_data);
    }
  }

  double global_skew = 0;
  while (!seq_data_queue.empty()) {
    auto *seq_path_data = dynamic_cast<StaSeqPathData *>(seq_data_queue.top());
    global_skew += FS_TO_NS(seq_path_data->getSkew());

    seq_data_queue.pop();
  }

  return global_skew;
}

/**
 * @brief get max skew of the flip-flop's output pin.
 *
 * @param mode
 * @param trans_type
 * @return std::map<StaVertex *, int>
 */
std::map<StaVertex *, int> Sta::getFFMaxSkew(AnalysisMode mode,
                                             TransType trans_type) {
  std::multimap<StaVertex *, StaSeqPathData *> vertex2pathdata;
  std::map<StaVertex *, int> vertex2maxskew;

  for (const auto &[clk, seq_path_group] : _clock_groups) {
    StaPathEnd *path_end;
    StaPathData *path_data;
    FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end)
    FOREACH_PATH_END_DATA(path_end, mode, path_data) {
      auto *seq_path_data = dynamic_cast<StaSeqPathData *>(path_data);
      auto path_delay_data = seq_path_data->getPathDelayData();
      StaPathDelayData *prior_path_delay_data = path_delay_data.top();
      StaVertex *driver_vertex = prior_path_delay_data->get_own_vertex();
      vertex2pathdata.insert(std::pair<StaVertex *, StaSeqPathData *>(
          driver_vertex, seq_path_data));
    }
  }

  std::multimap<StaVertex *, StaSeqPathData *>::iterator iter;
  for (iter = vertex2pathdata.begin(); iter != vertex2pathdata.end();) {
    auto cur_vertex = iter->first;
    int max_skew = 0;

    int num = vertex2pathdata.count(cur_vertex);

    for (int i = 0; i < num; i++) {
      int skew = iter->second->getSkew();
      if (max_skew < skew) {
        max_skew = skew;
      }
      iter++;
    }

    vertex2maxskew[cur_vertex] = max_skew;
  }

  return vertex2maxskew;
}

/**
 * @brief get total skew of the flip-flop's output pin.
 *
 * @param mode
 * @param trans_type
 * @return std::map<StaVertex *, int>
 */
std::map<StaVertex *, int> Sta::getFFTotalSkew(AnalysisMode mode,
                                               TransType trans_type) {
  std::multimap<StaVertex *, StaSeqPathData *> vertex2pathdata;
  std::map<StaVertex *, int> vertex2totalskew;

  for (const auto &[clk, seq_path_group] : _clock_groups) {
    StaPathEnd *path_end;
    StaPathData *path_data;
    FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end)
    FOREACH_PATH_END_DATA(path_end, mode, path_data) {
      auto *seq_path_data = dynamic_cast<StaSeqPathData *>(path_data);
      auto path_delay_data = seq_path_data->getPathDelayData();
      StaPathDelayData *prior_path_delay_data = path_delay_data.top();
      StaVertex *driver_vertex = prior_path_delay_data->get_own_vertex();
      vertex2pathdata.insert(std::pair<StaVertex *, StaSeqPathData *>(
          driver_vertex, seq_path_data));
    }
  }

  std::multimap<StaVertex *, StaSeqPathData *>::iterator iter;
  for (iter = vertex2pathdata.begin(); iter != vertex2pathdata.end();) {
    auto cur_vertex = iter->first;
    int total_skew = 0;

    int num = vertex2pathdata.count(cur_vertex);

    for (int i = 0; i < num; i++) {
      int skew = iter->second->getSkew();
      total_skew += skew;
      iter++;
    }

    vertex2totalskew[cur_vertex] = total_skew;
  }

  return vertex2totalskew;
}

/**
 * @brief get the related sinks of the skew.
 *
 * @param mode
 * @param trans_type
 * @return std::multimap<std::string,std::string>
 */
std::multimap<std::string, std::string> Sta::getSkewRelatedSink(
    AnalysisMode mode, TransType trans_type) {
  std::multimap<std::string, std::string> driver2ends;

  for (const auto &[clk, seq_path_group] : _clock_groups) {
    StaPathEnd *path_end;
    StaPathData *path_data;
    FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end)
    FOREACH_PATH_END_DATA(path_end, mode, path_data) {
      auto *seq_path_data = dynamic_cast<StaSeqPathData *>(path_data);
      auto path_delay_data = seq_path_data->getPathDelayData();
      StaPathDelayData *prior_path_delay_data = path_delay_data.top();
      StaVertex *driver_vertex = prior_path_delay_data->get_own_vertex();
      StaVertex *end_vertex = path_end->get_end_vertex();
      driver2ends.insert(std::pair<std::string, std::string>(
          driver_vertex->getName(), end_vertex->getName()));
    }
  }

  return driver2ends;
}

/**
 * @brief Get the end vertex path data.
 *
 * @param vertex
 * @param mode
 * @param trans_type
 * @return StaSeqPathData*
 */
StaSeqPathData *Sta::getWorstSeqData(std::optional<StaVertex *> vertex,
                                     AnalysisMode mode, TransType trans_type) {
  auto cmp = [](StaPathData *left, StaPathData *right) -> bool {
    int left_slack = left->getSlack();
    int right_slack = right->getSlack();
    return left_slack > right_slack;
  };

  std::priority_queue<StaPathData *, std::vector<StaPathData *>, decltype(cmp)>
      seq_data_queue(cmp);

  for (const auto &[clk, seq_path_group] : _clock_groups) {
    StaPathEnd *path_end;
    StaPathData *path_data;
    FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end) {
      if (!vertex || path_end->get_end_vertex() == *vertex) {
        FOREACH_PATH_END_DATA(path_end, mode, path_data) {
          seq_data_queue.push(path_data);
        }
      }
    }
  }

  StaSeqPathData *seq_path_data = nullptr;
  while (!seq_data_queue.empty()) {
    seq_path_data = dynamic_cast<StaSeqPathData *>(seq_data_queue.top());

    if ((seq_path_data->get_delay_data()->get_trans_type() == trans_type) ||
        (trans_type == TransType::kRiseFall)) {
      break;
    }
    seq_data_queue.pop();
  }

  return seq_path_data;
}

/**
 * @brief Get the worst slack path.
 *
 * @param mode
 * @param trans_type
 * @return StaSeqPathData*
 */
StaSeqPathData *Sta::getWorstSeqData(AnalysisMode mode, TransType trans_type) {
  return getWorstSeqData(std::nullopt, mode, trans_type);
}

/**
 * @brief obtain the start_end_slack tuples of the top n voilated timing
 * path(slack_n<slack_(n-1)<...<slack_1).
 *
 * @param top_n
 * @param mode
 * @param trans_type
 * @return std::vector<std::tuple<std::string, std::string, double>>
 */
std::vector<std::tuple<std::string, std::string, double>>
Sta::getStartEndSlackPairsOfTopNPaths(int top_n, AnalysisMode mode,
                                      TransType trans_type) {
  auto cmp = [](StaPathData *left, StaPathData *right) -> bool {
    int left_slack = left->getSlack();
    int right_slack = right->getSlack();
    return left_slack > right_slack;
  };

  std::priority_queue<StaPathData *, std::vector<StaPathData *>, decltype(cmp)>
      seq_data_queue(cmp);

  for (const auto &[clk, seq_path_group] : _clock_groups) {
    StaPathEnd *path_end;
    StaPathData *path_data;
    FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end) {
      FOREACH_PATH_END_DATA(path_end, mode, path_data) {
        seq_data_queue.push(path_data);
      }
    }
  }

  StaSeqPathData *seq_path_data = nullptr;
  std::vector<std::tuple<std::string, std::string, double>> start_end_slacks;
  while (!seq_data_queue.empty() && top_n > 0) {
    seq_path_data = dynamic_cast<StaSeqPathData *>(seq_data_queue.top());

    if (seq_path_data->get_delay_data()->get_trans_type() == trans_type) {
      auto start_pin_name =
          seq_path_data->getPathDelayData().top()->get_own_vertex()->getName();
      auto end_pin_name =
          seq_path_data->get_delay_data()->get_own_vertex()->getName();
      double slack = seq_path_data->getSlackNs();
      if (slack >= 0) {
        break;
      }
      start_end_slacks.push_back(
          std::make_tuple(start_pin_name, end_pin_name, slack));

      --top_n;
    }
    seq_data_queue.pop();
  }

  return start_end_slacks;
}

/**
 * @brief obtain the start_end_slack tuples of the top percentage voilated
 * timing path(slack_n<slack_(n-1)<...<slack_1).
 *
 * @param top_percentage
 * @param mode
 * @param trans_type
 * @return std::vector<std::tuple<std::string, std::string, double>>
 */
std::vector<std::tuple<std::string, std::string, double>>
Sta::getStartEndSlackPairsOfTopNPercentPaths(double top_percentage,
                                             AnalysisMode mode,
                                             TransType trans_type) {
  auto cmp = [](StaPathData *left, StaPathData *right) -> bool {
    int left_slack = left->getSlack();
    int right_slack = right->getSlack();
    return left_slack > right_slack;
  };

  std::priority_queue<StaPathData *, std::vector<StaPathData *>, decltype(cmp)>
      seq_data_queue(cmp);

  for (const auto &[clk, seq_path_group] : _clock_groups) {
    StaPathEnd *path_end;
    StaPathData *path_data;
    FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end) {
      FOREACH_PATH_END_DATA(path_end, mode, path_data) {
        seq_data_queue.push(path_data);
      }
    }
  }

  StaSeqPathData *seq_path_data = nullptr;
  std::vector<std::tuple<std::string, std::string, double>> start_end_slacks;
  int top_n = seq_data_queue.size() * top_percentage;
  while (!seq_data_queue.empty() && top_n > 0) {
    seq_path_data = dynamic_cast<StaSeqPathData *>(seq_data_queue.top());

    if (seq_path_data->get_delay_data()->get_trans_type() == trans_type) {
      auto start_pin_name =
          seq_path_data->getPathDelayData().top()->get_own_vertex()->getName();
      auto end_pin_name =
          seq_path_data->get_delay_data()->get_own_vertex()->getName();
      double slack = seq_path_data->getSlackNs();
      if (slack >= 0) {
        break;
      }
      start_end_slacks.push_back(
          std::make_tuple(start_pin_name, end_pin_name, slack));
      --top_n;
    }
    seq_data_queue.pop();
  }

  return start_end_slacks;
}

/**
 * @brief Get the violated StaSeqPathDatas of the two specified sinks that form
 * the timing path.
 *
 * @param vertex1 (one sink)
 * @param vertex2 (one sink)
 * @param mode
 * @return the violated StaSeqPathDatas.
 */
std::priority_queue<StaSeqPathData *, std::vector<StaSeqPathData *>,
                    decltype(seq_data_cmp)>
Sta::getViolatedSeqPathsBetweenTwoSinks(StaVertex *vertex1, StaVertex *vertex2,
                                        AnalysisMode mode) {
  // auto cmp = [](StaSeqPathData *left, StaSeqPathData *right) -> bool {
  //   int left_slack = left->getSlack();
  //   int right_slack = right->getSlack();
  //   return left_slack > right_slack;
  // };

  std::priority_queue<StaSeqPathData *, std::vector<StaSeqPathData *>,
                      decltype(seq_data_cmp)>
      seq_data_queue(seq_data_cmp);

  for (const auto &[clk, seq_path_group] : _clock_groups) {
    StaPathEnd *path_end;
    StaPathData *path_data;
    FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end) {
      if (!vertex1 || path_end->get_end_vertex() == vertex1) {
        FOREACH_PATH_END_DATA(path_end, mode, path_data) {
          auto *seq_path_data = dynamic_cast<StaSeqPathData *>(path_data);

          if (!vertex2 ||
              seq_path_data->getPathDelayData().top()->get_own_vertex() ==
                  vertex2) {
            if (seq_path_data->getSlack() < 0) {
              seq_data_queue.push(seq_path_data);
            }
          }
        }
      }
    }
  }

  return seq_data_queue;
}

/**
 * @brief Get the slack of the two specified sinks that form the timing path.
 *
 * @param vertex1 (one sink)
 * @param vertex2 (one sink)
 * @param mode
 * @return double worst slack
 */
std::optional<double> Sta::getWorstSlackBetweenTwoSinks(StaVertex *vertex1,
                                                        StaVertex *vertex2,
                                                        AnalysisMode mode) {
  auto cmp = [](StaPathData *left, StaPathData *right) -> bool {
    int left_slack = left->getSlack();
    int right_slack = right->getSlack();
    return left_slack > right_slack;
  };

  std::priority_queue<StaPathData *, std::vector<StaPathData *>, decltype(cmp)>
      seq_data_queue(cmp);

  for (const auto &[clk, seq_path_group] : _clock_groups) {
    StaPathEnd *path_end;
    StaPathData *path_data;
    FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end) {
      if (!vertex1 || path_end->get_end_vertex() == vertex1) {
        FOREACH_PATH_END_DATA(path_end, mode, path_data) {
          auto *seq_path_data = dynamic_cast<StaSeqPathData *>(path_data);

          if (!vertex2 ||
              seq_path_data->getPathDelayData().top()->get_own_vertex() ==
                  vertex2) {
            seq_data_queue.push(path_data);
          }
        }
      }
    }
  }

  StaSeqPathData *seq_path_data = nullptr;
  std::optional<double> worst_slack;
  while (!seq_data_queue.empty()) {
    seq_path_data = dynamic_cast<StaSeqPathData *>(seq_data_queue.top());
    worst_slack = seq_path_data->getSlackNs();
    break;
  }

  return worst_slack;
}

/**
 * @brief Get the slack of all two sinks that form the timing path.
 *
 * @param mode
 * @return std::map<std::pair<StaVertex *, StaVertex *>, double>
 */
std::map<std::pair<StaVertex *, StaVertex *>, double>
Sta::getWorstSlackBetweenTwoSinks(AnalysisMode mode) {
  std::map<std::pair<StaVertex *, StaVertex *>, double> twosinks2worstslack;
  for (const auto &[clk, seq_path_group] : _clock_groups) {
    StaPathEnd *path_end;
    StaPathData *path_data;
    FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end) {
      FOREACH_PATH_END_DATA(path_end, mode, path_data) {
        auto *seq_path_data = dynamic_cast<StaSeqPathData *>(path_data);
        StaVertex *start_vertex =
            seq_path_data->getPathDelayData().top()->get_own_vertex();

        if (!start_vertex->is_clock()) {
          continue;
        }

        auto *end_vertex = seq_path_data->get_delay_data()->get_own_vertex();
        if (end_vertex->is_port()) {
          continue;
        }

        auto *end_clock_vertex =
            seq_path_data->get_capture_clock_data()->get_own_vertex();

        if (start_vertex == end_clock_vertex) {
          continue;
        }

        // make launch and capture clock vertex pair.
        std::pair<StaVertex *, StaVertex *> sink_pair =
            std::make_pair(start_vertex, end_clock_vertex);

        double slack = seq_path_data->getSlackNs();

        // store worst slack.
        if (!twosinks2worstslack.contains(sink_pair) ||
            (slack < twosinks2worstslack[sink_pair])) {
          twosinks2worstslack[sink_pair] = slack;
        }
      }
    }
  }

  return twosinks2worstslack;
}

/**
 * @brief Get the worst slack of the end vertex.
 *
 * @param end_vertex
 * @return int
 */
int Sta::getWorstSlack(StaVertex *end_vertex, AnalysisMode mode,
                       TransType trans_type) {
  auto *the_worst_seq_path_data = getWorstSeqData(end_vertex, mode, trans_type);
  return the_worst_seq_path_data->getSlack();
}

/**
 * @brief write verilog.
 *
 * @param verilog_file_name
 * @param sort
 * @param include_pwr_gnd_pins
 * @param remove_cells
 * @param netlist
 */
void Sta::writeVerilog(const char *verilog_file_name,
                       std::set<std::string> &exclude_cell_names) {
  NetlistWriter writer(verilog_file_name, exclude_cell_names, _netlist);
  writer.writeModule();
}

/**
 * @brief reset graph data.
 *
 * @return unsigned
 */
unsigned Sta::resetGraphData() {
  StaGraph &the_graph = get_graph();
  the_graph.initGraph();
  the_graph.resetVertexData();
  the_graph.resetArcData();
  return 1;
}

/**
 * @brief reset path data.
 *
 * @return unsigned
 */
unsigned Sta::resetPathData() {
  reset_clock_groups();
  resetReportTbl();
  return 1;
}

/**
 * @brief update the timing data.
 *
 * @return unsigned
 */
unsigned Sta::updateTiming() {
  ieda::Stats stats;

  LOG_INFO << "update timing start";

  resetSdcConstrain();
  resetGraphData();
  resetPathData();

#if CUDA_PROPAGATION
  resetGPUData();
#endif

  StaGraph &the_graph = get_graph();
  if (_propagation_method == PropagationMethod::kDFS) {
    // DFS flow
    Vector<std::function<unsigned(StaGraph *)>> funcs = {
        StaApplySdc(StaApplySdc::PropType::kApplySdcPreProp),
        StaConstPropagation(),
        StaClockPropagation(StaClockPropagation::PropType::kIdealClockProp),
        StaCombLoopCheck(), StaSlewPropagation(), StaDelayPropagation(),
        StaClockPropagation(StaClockPropagation::PropType::kNormalClockProp),
        StaApplySdc(StaApplySdc::PropType::kApplySdcPostNormalClockProp),
        StaClockPropagation(
            StaClockPropagation::PropType::kUpdateGeneratedClockProp),
        StaApplySdc(StaApplySdc::PropType::kApplySdcPostClockProp),
        StaLevelization(), StaBuildPropTag(StaPropagationTag::TagType::kProp),
        StaDataPropagation(StaDataPropagation::PropType::kFwdProp),
        // StaCrossTalkPropagation(),
        StaDataPropagation(StaDataPropagation::PropType::kIncrFwdProp),
        StaAnalyze(), StaApplySdc(StaApplySdc::PropType::kApplySdcPostProp),
        StaDataPropagation(StaDataPropagation::PropType::kBwdProp)};

    for (auto &func : funcs) {
      the_graph.exec(func);
    }
  } else {
    // BFS flow
    Vector<std::function<unsigned(StaGraph *)>> funcs = {
        StaApplySdc(StaApplySdc::PropType::kApplySdcPreProp),
        StaConstPropagation(),
        StaClockPropagation(StaClockPropagation::PropType::kIdealClockProp),
        StaCombLoopCheck(), StaClockSlewDelayPropagation(), StaLevelization(),
        StaClockPropagation(StaClockPropagation::PropType::kNormalClockProp),
        StaApplySdc(StaApplySdc::PropType::kApplySdcPostNormalClockProp),
        StaClockPropagation(
            StaClockPropagation::PropType::kUpdateGeneratedClockProp),
        StaApplySdc(StaApplySdc::PropType::kApplySdcPostClockProp),
        StaBuildPropTag(StaPropagationTag::TagType::kProp),
#if !INTEGRATION_FWD
        StaDataSlewDelayPropagation(),
#endif
        StaDataPropagation(StaDataPropagation::PropType::kFwdProp),
        // StaCrossTalkPropagation(),

        StaDataPropagation(StaDataPropagation::PropType::kIncrFwdProp),
        StaAnalyze(), StaApplySdc(StaApplySdc::PropType::kApplySdcPostProp),
        StaDataPropagation(StaDataPropagation::PropType::kBwdProp)};

    for (auto &func : funcs) {
      the_graph.exec(func);
    }
  }

  LOG_INFO << "update timing end";

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "update timing memory usage " << memory_delta << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "update timing time elapsed " << time_delta << "s";
  return 1;
}

/**
 * @brief update the clock timing data for finding the start pins or the end
 * pins.
 *
 * @return unsigned
 */
unsigned Sta::updateClockTiming() {
  LOG_INFO << "update timing start";

  resetSdcConstrain();
  resetGraphData();
  resetPathData();

  StaGraph &the_graph = get_graph();

  Vector<std::function<unsigned(StaGraph *)>> funcs = {
      StaApplySdc(StaApplySdc::PropType::kApplySdcPreProp),
      StaConstPropagation(),
      StaClockPropagation(StaClockPropagation::PropType::kIdealClockProp),
      StaCombLoopCheck(),
      StaSlewPropagation(),
      StaDelayPropagation(),
      StaClockPropagation(StaClockPropagation::PropType::kNormalClockProp),
      StaApplySdc(StaApplySdc::PropType::kApplySdcPostNormalClockProp),
      StaClockPropagation(
          StaClockPropagation::PropType::kUpdateGeneratedClockProp),
      StaApplySdc(StaApplySdc::PropType::kApplySdcPostClockProp)};

  for (auto &func : funcs) {
    the_graph.exec(func);
  }

  LOG_INFO << "update timing end";
  return 1;
}

/**
 * @brief Find the start/end pins accordingt to the given end/start pin of the
 * timing path.
 *
 * @param the_vertex
 * @param is_find_end
 * @return std::set<std::string>
 */
std::set<std::string> Sta::findStartOrEnd(StaVertex *the_vertex,
                                          bool is_find_end) {
  std::set<std::string> pin_names;

  if (is_find_end) {
    StaFindEnd find_end;
    if (the_vertex->is_start() && the_vertex->is_clock()) {
      the_vertex->exec(find_end);
    } else {
      LOG_FATAL << "Not the correct start pin of the timing path";
    }

    auto &end_vertexes = the_vertex->get_fanout_end_vertexes();
    for (auto &end_vertex : end_vertexes) {
      std::string end_pin_name = end_vertex->getName();
      pin_names.insert(end_pin_name);
    }
  } else {
    StaFindStart find_start;
    if (the_vertex->is_end()) {
      the_vertex->exec(find_start);
    } else {
      LOG_FATAL << "Not the correct end pin of the timing path";
    }

    auto &start_vertexes = the_vertex->get_fanin_start_vertexes();
    for (auto &start_vertex : start_vertexes) {
      std::string start_pin_name = start_vertex->getName();
      pin_names.insert(start_pin_name);
    }
  }

  return pin_names;
}

/**
 * @brief generate the timing report.
 *
 * @return unsigned
 */
unsigned Sta::reportTiming(std::set<std::string> &&exclude_cell_names /*= {}*/,
                           bool is_derate /*=false*/,
                           bool is_clock_cap /*=false*/,
                           bool is_copy /*=true*/) {
  const char *design_work_space = get_design_work_space();
  std::string now_time = Time::getNowWallTime();
  std::string tmp = Str::replace(now_time, ":", "_");
  std::string copy_design_work_space =
      Str::printf("%s_sta_%s", design_work_space, tmp.c_str());

  LOG_INFO << "start write sta report.";
  LOG_INFO << "output sta report path: " << design_work_space;

  if (design_work_space == nullptr || design_work_space[0] == '\0') {
    LOG_ERROR << "The design work space is not set.";
    return 0;
  }

  if (std::filesystem::exists(design_work_space) && is_copy) {
    std::filesystem::create_directories(copy_design_work_space);
  }
  std::filesystem::create_directories(design_work_space);

  // std::string specify_path_file_name =
  //     Str::printf("%s/%s.spec", design_work_space,
  //     get_design_name().c_str());
  // reportFromThroughTo(specify_path_file_name.c_str(), AnalysisMode::kMax,
  //                     nullptr, nullptr,
  //                     "u0_soc_top/u0_nic400_bus/u_cd_c0/u_ib_slave_5_ib_s/"
  //                     "u_aw_slave_port_chan_slice/u_rev_regd_slice/_1021_:D");

  resetReportTbl();

  auto copy_file = [this, &tmp, &copy_design_work_space](
                       const std::string &file_name,
                       const std::string &file_type) {
    std::string copy_file_name =
        Str::printf("%s/%s_%s%s", copy_design_work_space.c_str(),
                    get_design_name().c_str(), tmp.c_str(), file_type.c_str());
    if (std::filesystem::exists(file_name)) {
      std::filesystem::copy_file(
          file_name, copy_file_name,
          std::filesystem::copy_options::overwrite_existing);
    }
  };

  std::string rpt_file_name =
      Str::printf("%s/%s.rpt", design_work_space, get_design_name().c_str());
  if (is_copy) {
    copy_file(rpt_file_name, ".rpt");
  }
  reportPath(rpt_file_name.c_str(), is_derate);

  std::string trans_rpt_file_name =
      Str::printf("%s/%s.trans", design_work_space, get_design_name().c_str());
  if (is_copy) {
    copy_file(trans_rpt_file_name, ".trans");
  }
  reportTrans(trans_rpt_file_name.c_str());

  std::string cap_rpt_file_name =
      Str::printf("%s/%s.cap", design_work_space, get_design_name().c_str());
  if (is_copy) {
    copy_file(cap_rpt_file_name, ".cap");
  }
  reportCap(cap_rpt_file_name.c_str(), is_clock_cap);

  std::string fanout_rpt_file_name =
      Str::printf("%s/%s.fanout", design_work_space, get_design_name().c_str());
  if (is_copy) {
    copy_file(fanout_rpt_file_name, ".fanout");
  }
  reportFanout(fanout_rpt_file_name.c_str());

  std::string setup_skew_rpt_file_name = Str::printf(
      "%s/%s_setup.skew", design_work_space, get_design_name().c_str());
  if (is_copy) {
    copy_file(setup_skew_rpt_file_name, "_setup.skew");
  }
  reportSkew(setup_skew_rpt_file_name.c_str(), AnalysisMode::kMax);

  std::string hold_skew_rpt_file_name = Str::printf(
      "%s/%s_hold.skew", design_work_space, get_design_name().c_str());
  if (is_copy) {
    copy_file(hold_skew_rpt_file_name, "_hold.skew");
  }
  reportSkew(hold_skew_rpt_file_name.c_str(), AnalysisMode::kMin);

  std::string verilog_file_name =
      Str::printf("%s/%s.v", design_work_space, get_design_name().c_str());
  if (is_copy) {
    copy_file(verilog_file_name, ".v");
  }

  if (c_print_net_yaml) {
    reportNet();
  }

  writeVerilog(verilog_file_name.c_str(), exclude_cell_names);

  reportUsedLibs();

  // for test dump timing data in memory.
  // reportTimingData(10);

  // for test dump json data.
  // reportWirePaths();

#if CUDA_PROPAGATION
  // printFlattenData();
#endif

  // dumpGraphData("/home/taosimin/ysyx_test25/2025-04-05/graph.yaml");

  LOG_INFO << "The timing engine run success.";

  // restart the timer.
  Time::start();

  return 1;
}

/**
 * @brief report timing data in memory for online analysis.
 *
 * @param n_worst_path_per_clock
 * @return unsigned
 */
std::vector<StaPathWireTimingData> Sta::reportTimingData(
    unsigned n_worst_path_per_clock) {
  LOG_INFO << "get wire timing start";
  std::vector<StaPathWireTimingData> path_timing_data;

  set_n_worst_path_per_clock(n_worst_path_per_clock);

  for (auto analysi_mode : {AnalysisMode::kMax, AnalysisMode::kMin}) {
    StaReportPathTimingData report_path_timing_data_func(
        nullptr, analysi_mode, n_worst_path_per_clock);
    for (auto &[capture_clock, seq_path_group] : _clock_groups) {
      auto group_timing_data =
          report_path_timing_data_func.getPathGroupTimingData(
              seq_path_group.get());
      path_timing_data.insert(path_timing_data.end(), group_timing_data.begin(),
                              group_timing_data.end());
    }
  }

  LOG_INFO << "the wire timing data size: " << path_timing_data.size();
  LOG_INFO << "get wire timing end";

  return path_timing_data;
}

/**
 * @brief report used libs.
 *
 * @return unsigned
 */
unsigned Sta::reportUsedLibs() {
  auto used_libs = getUsedLibs();
  for (auto *used_lib : used_libs) {
    std::string lib_name = used_lib->get_file_name();
    LOG_INFO << "used lib: " << lib_name;
  }
  return 1;
}

/**
 * @brief report wire paths.
 *
 * @return unsigned
 */
unsigned Sta::reportWirePaths() {
  LOG_INFO << "report wire paths start";
  const char *design_work_space = get_design_work_space();

  std::string path_dir = std::string(design_work_space) + "/wire_paths";

  if (std::filesystem::exists(path_dir)) {
    for (const auto &entry : std::filesystem::directory_iterator(path_dir)) {
      if (entry.is_regular_file()) {
        std::filesystem::remove(entry.path());
      }
    }
  }

  std::string rpt_file_name =
      Str::printf("%s/%s.rpt", design_work_space, get_design_name().c_str());
  reportPath(rpt_file_name.c_str(), false, true);

  LOG_INFO << "report wire paths end";

  return 1;
}

/**
 * @brief dump vertex data in yaml format.
 *
 * @param vertex_names
 */
void Sta::dumpVertexData(std::vector<std::string> vertex_names) {
  const char *design_work_space = get_design_work_space();

  unsigned index = 0;
  for (auto &vertex_name : vertex_names) {
    StaDumpYaml dump_data;
    auto *the_vertex = findVertex(vertex_name.c_str());
    the_vertex->exec(dump_data);
    const char *file_name =
        Str::printf("%s/vertex_%d.txt", design_work_space, index);
    dump_data.printText(file_name);
    ++index;
  }
}

/**
 * @brief dump netlist data in json and txt format.
 *
 */
void Sta::dumpNetlistData() {
  const char *design_work_space = get_design_work_space();

  Sta *ista = Sta::getOrCreateSta();
  auto &the_graph = ista->get_graph();
  Netlist *the_netlist = ista->get_netlist();
  Net *the_net;

  nlohmann::json net_info_json = nlohmann::json::array();
  std::set<std::string> cell_types;

  FOREACH_NET(the_netlist, the_net) {
    // initialize the json container
    nlohmann::json net_info_obj = nlohmann::json::object();
    nlohmann::json net_load_list = nlohmann::json::array();

    auto net_name = the_net->get_name();

    // driver info
    auto *driver_pin = the_net->getDriver();
    auto *driver_cell_type = "";
    if (driver_pin->isPort()) {
      driver_cell_type = "[port]";
    } else {
      driver_cell_type =
          driver_pin->get_own_instance()->get_inst_cell()->get_cell_name();
    }
    cell_types.emplace(driver_cell_type);

    auto driver_cell_level = (*the_graph.findVertex(driver_pin))->get_level();
    auto driver_cell_fanout = the_net->getLoads().size();

    // write net_name and driver info to json container
    net_info_obj["net_name"] = net_name;
    net_info_obj["driver_cell_fanout"] = driver_cell_fanout;
    net_info_obj["driver_cell_type"] = driver_cell_type;
    net_info_obj["driver_cell_level"] = driver_cell_level;

    // load info
    std::vector<DesignObject *> load_pins = the_net->getLoads();
    auto index = 1;
    for (auto *load_pin : load_pins) {
      // initialize the load_info json container
      nlohmann::json net_load_info = nlohmann::json::object();

      auto load_vertex = the_graph.findVertex(load_pin);
      LOG_FATAL_IF(!load_vertex);

      const char *load_cell_type = "";
      if (load_pin->isPort()) {
        load_cell_type = "[port]";
      } else {
        load_cell_type =
            load_pin->get_own_instance()->get_inst_cell()->get_cell_name();
      }
      cell_types.emplace(load_cell_type);

      // write the load info to container with index
      net_load_info["index"] = index++;
      net_load_info["load_cell_type"] = load_cell_type;

      // insert the net load/cell info the the net load list before next dealing
      // next load
      net_load_list.emplace_back(net_load_info);
    }

    // write load info to json container
    net_info_obj["net_load_list"] = net_load_list;

    // write the json container to net_info_json before looping next net
    net_info_json.emplace_back(net_info_obj);
  }

  // export to json file before exiting
  std::ofstream outputJson(std::string(design_work_space) +
                           std::string("/net_list_info.json"));

  if (!outputJson.is_open()) {
    std::cerr << "Error: Failed to open file for writing." << std::endl;
  }

  outputJson << std::setw(4) << net_info_json << std::endl;
  LOG_INFO << "net_list_info.json written to "
           << std::string(design_work_space) +
                  std::string("/net_list_info.json")
           << std::endl;
  outputJson.close();

  std::ofstream outputTxt(std::string(design_work_space) +
                          std::string("/cell_types.txt"));
  if (!outputTxt.is_open()) {
    std::cerr << "Error: Failed to open file for writing." << std::endl;
  }

  std::copy(cell_types.begin(), cell_types.end(),
            std::ostream_iterator<std::string>(outputTxt, "\n"));
  LOG_INFO << "cell_types.txt written to "
           << std::string(design_work_space) + std::string("/cell_types.txt")
           << std::endl;
  outputTxt.close();
}

/**
 * @brief dump timing graph data.
 *
 */
void Sta::dumpGraphData(const char *graph_file) {
  StaDumpYaml dump_data;
  dump_data.set_yaml_file_path(graph_file);

  auto &the_graph = get_graph();
  the_graph.exec(dump_data);
}

/**
 * @brief Build clock tree for GUI.
 *
 */
void Sta::buildClockTrees() {
  StaBuildClockTree build_clock_tree;
  for (auto &clock : _clocks) {
    build_clock_tree(clock.get());
  }

  _clock_trees = std::move(build_clock_tree.takeClockTrees());
}

/**
 * @brief get the instance worst slack.
 *
 * @param analysis_mode
 * @param the_inst
 * @return std::optional<double>
 */
std::optional<double> Sta::getInstWorstSlack(AnalysisMode analysis_mode,
                                             Instance *the_inst) {
  Pin *the_pin;
  std::optional<double> the_worst_inst_slack;
  FOREACH_INSTANCE_PIN(the_inst, the_pin) {
    auto *the_vertex = findVertex(the_pin);
    if (!the_vertex) {
      continue;
    }
    auto the_worst_pin_slack = the_vertex->getWorstSlackNs(analysis_mode);
    if (the_worst_pin_slack) {
      if (!the_worst_inst_slack ||
          (*the_worst_inst_slack > *the_worst_pin_slack)) {
        the_worst_inst_slack = *the_worst_pin_slack;
      }
    }
  }

  // LOG_FATAL_IF(the_worst_inst_slack)
  //     << "inst " << the_inst->get_name() << "the worst slack "
  //     << *the_worst_inst_slack;
  return the_worst_inst_slack;
}

/**
 * @brief get total negative slack of all instance pins.
 *
 * @param analysis_mode
 * @param the_inst
 * @return std::optional<double>
 */
std::optional<double> Sta::getInstTotalNegativeSlack(AnalysisMode analysis_mode,
                                                     Instance *the_inst) {
  Pin *the_pin;
  std::optional<double> the_total_negative_inst_slack;
  FOREACH_INSTANCE_PIN(the_inst, the_pin) {
    auto *the_vertex = findVertex(the_pin);
    if (!the_vertex) {
      continue;
    }
    auto the_total_negative_slack = the_vertex->getTNSNs(analysis_mode);
    if (the_total_negative_slack) {
      if (!the_total_negative_inst_slack) {
        the_total_negative_inst_slack = *the_total_negative_slack;
      } else {
        *the_total_negative_inst_slack += *the_total_negative_slack;
      }
    }
  }

  // LOG_FATAL_IF(the_total_negative_inst_slack)
  //     << "inst " << the_inst->get_name() << "the worst slack "
  //     << *the_total_negative_inst_slack;
  return the_total_negative_inst_slack;
}

/**
 * @brief get the instance worst transition.
 *
 * @param analysis_mode
 * @param the_inst
 * @return std::optional<double>
 */
std::optional<double> Sta::getInstTransition(AnalysisMode analysis_mode,
                                             Instance *the_inst) {
  Pin *the_pin;
  std::optional<double> the_worst_inst_slew;
  FOREACH_INSTANCE_PIN(the_inst, the_pin) {
    auto *the_vertex = findVertex(the_pin);
    if (!the_vertex) {
      continue;
    }
    auto the_worst_pin_slew = the_vertex->getWorstSlewNs(analysis_mode);
    if (the_worst_pin_slew) {
      if (!the_worst_inst_slew) {
        the_worst_inst_slew = *the_worst_pin_slew;
      } else {
        if ((analysis_mode == AnalysisMode::kMax) &&
            (*the_worst_inst_slew < *the_worst_pin_slew)) {
          the_worst_inst_slew = *the_worst_pin_slew;
        } else if ((analysis_mode == AnalysisMode::kMin) &&
                   (*the_worst_inst_slew > *the_worst_pin_slew)) {
          the_worst_inst_slew = *the_worst_pin_slew;
        }
      }
    }
  }

  return the_worst_inst_slew;
}

/**
 * @brief display timing map of inst worst slack.
 *
 * @param analysis_mode
 * @return auto
 */
std::map<Instance::Coordinate, double> Sta::displayTimingMap(
    AnalysisMode analysis_mode) {
  std::map<Instance::Coordinate, double> loc_to_inst_slack;
  Instance *the_inst;
  FOREACH_INSTANCE(&_netlist, the_inst) {
    auto the_inst_worst_slack = getInstWorstSlack(analysis_mode, the_inst);
    if (the_inst_worst_slack) {
      auto inst_coordinate = the_inst->get_coordinate();
      if (!inst_coordinate) {
        LOG_INFO << "inst " << the_inst->get_name() << " has no coordinate.";
        continue;
      }

      loc_to_inst_slack[*inst_coordinate] = *the_inst_worst_slack;
    }
  }

  return loc_to_inst_slack;
}

/**
 * @brief display timing tns map.
 *
 * @param analysis_mode
 * @return std::map<Instance::Coordinate, double>
 */
std::map<Instance::Coordinate, double> Sta::displayTimingTNSMap(
    AnalysisMode analysis_mode) {
  std::map<Instance::Coordinate, double> loc_to_inst_tns;
  Instance *the_inst;
  FOREACH_INSTANCE(&_netlist, the_inst) {
    auto the_inst_tns = getInstTotalNegativeSlack(analysis_mode, the_inst);
    if (the_inst_tns) {
      auto inst_coordinate = the_inst->get_coordinate();
      if (!inst_coordinate) {
        LOG_INFO << "inst " << the_inst->get_name() << " has no coordinate.";
        continue;
      }

      loc_to_inst_tns[*inst_coordinate] = *the_inst_tns;
    }
  }

  return loc_to_inst_tns;
}

/**
 * @brief get the inst transition map.
 *
 * @param analysis_mode
 * @return std::map<Instance::Coordinate, double>
 */
std::map<Instance::Coordinate, double> Sta::displayTransitionMap(
    AnalysisMode analysis_mode) {
  std::map<Instance::Coordinate, double> loc_to_inst_transition;
  Instance *the_inst;
  FOREACH_INSTANCE(&_netlist, the_inst) {
    auto the_inst_worst_transition = getInstTransition(analysis_mode, the_inst);
    if (the_inst_worst_transition) {
      auto inst_coordinate = the_inst->get_coordinate();
      if (!inst_coordinate) {
        LOG_INFO << "inst " << the_inst->get_name() << " has no coordinate.";
        continue;
      }

      loc_to_inst_transition[*inst_coordinate] = *the_inst_worst_transition;
    }
  }

  return loc_to_inst_transition;
}

double Sta::convertTimeUnit(const double src_value) {
  TimeUnit current_time_unit = getTimeUnit();
  if (current_time_unit == TimeUnit::kNS) {
    return src_value;
  } else if (current_time_unit == TimeUnit::kFS) {
    return FS_TO_NS(src_value);
  } else if (current_time_unit == TimeUnit::kPS) {
    return PS_TO_NS(src_value);
  }
  return -1;
}

double Sta::convertCapUnit(const double src_value) {
  CapacitiveUnit current_cap_unit = getCapUnit();
  if (current_cap_unit == CapacitiveUnit::kPF) {
    return src_value;
  } else if (current_cap_unit == CapacitiveUnit::kFF) {
    return FF_TO_PF(src_value);
  } else if (current_cap_unit == CapacitiveUnit::kF) {
    return F_TO_PF(src_value);
  }
  return -1;
}

#if CUDA_PROPAGATION
/**
 * @brief print flatten data for debug gpu data.
 *
 */
void Sta::printFlattenData() {
  LOG_INFO << "print flatten data path start";
  std::string design_work_space = get_design_work_space();

  auto &flatten_data = get_flatten_data();
  auto &index_to_at = get_index_to_at();

  auto &flatten_at_data = flatten_data._flatten_at_data;
  unsigned flatten_at_size = flatten_at_data.size();

  std::string flatten_at_data_path =
      design_work_space + "/flatten_at_data.yaml";
  std::ofstream output_file(flatten_at_data_path);
  unsigned at_data_index = 0;
  for (auto &at_data : flatten_at_data) {
    LOG_INFO_EVERY_N(100000) << "print flatten data path " << at_data_index
                             << " total " << flatten_at_size;
    auto *path_delay_data = index_to_at[at_data_index];
    auto *own_vertex = path_delay_data->get_own_vertex();
    const char *launch_clock_name = path_delay_data->get_launch_clock_data()
                                        ->get_prop_clock()
                                        ->get_clock_name();

    output_file << "GPU_AT_DATA_" << at_data_index++ << ": " << std::endl;
    output_file << "  own_vertex: " << own_vertex->getName() << std::endl;
    output_file << "  vertex level: " << own_vertex->get_level() << std::endl;
    output_file << "  launch_clock_name: " << launch_clock_name << std::endl;
    output_file << "  launch_clock_index: " << at_data._own_clock_index
                << std::endl;
    output_file << "  mode: "
                << (at_data._analysis_mode == GPU_Analysis_Mode::kMax ? "max"
                                                                      : "min")
                << std::endl;
    output_file << "  trans_type: "
                << (at_data._trans_type == GPU_Trans_Type::kRise ? "r" : "f")
                << std::endl;
    output_file << "  data_value: " << FS_TO_NS(at_data._data_value)
                << std::endl;
    output_file << "  src_vertex_id: " << at_data._src_vertex_id << std::endl;
    if (at_data._src_vertex_id != -1) {
      auto *src_vertex = getVertex(at_data._src_vertex_id);
      output_file << "  src_vertex: " << src_vertex->getName() << std::endl;
    } else {
      output_file << "  src_vertex: " << "NA" << std::endl;
    }

    output_file << "  src_data_index: " << at_data._src_data_index << std::endl;
    output_file << "  snk_data_index: " << at_data._snk_data_index << std::endl;
  }

  output_file.close();

  LOG_INFO << "print flatten data path end";
  LOG_INFO << "print flatten data path: " << flatten_at_data_path;
}

#endif

}  // namespace ista
