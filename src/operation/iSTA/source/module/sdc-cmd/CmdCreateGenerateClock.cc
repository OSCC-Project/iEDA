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
 * @file CmdCreateGeneratedClock.cc
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-09-23
 */
#include <algorithm>
#include <variant>

#include "Cmd.hh"
#include "ScriptEngine.hh"
#include "include/Type.hh"
#include "log/Log.hh"
#include "netlist/DesignObject.hh"
#include "sdc/SdcCollection.hh"

namespace ista {

/**
 * @brief add options and args for "create_generated_clock" command
 * @usage in Tcl-shell
 * % create_generated_clock
 *     [-name clock_name]
 *     [-source master_pin]
 *     [-divide_by divide_factor | -multiply_by multiply_factor |
 *      -edges edge_list ]
 *     [-duty_cycle percent]
 *     [-invert]
 *     [-edge_shift edge_shift_list]
 *     [-add]
 *     [-master_clock clock]
 *     [-comment comment_string]
 *     source_objects
 */
CmdCreateGeneratedClock::CmdCreateGeneratedClock(const char* cmd_name)
    : TclCmd(cmd_name) {
  // creat an int option
  // option initialization list:
  // "-divide_by" -- option name
  //       0      -- option (1 -- arg)
  //       0      -- default value
  // default int option is 0, which initialization below equals to new
  // TclIntOption("-divide_by", 0);
  auto* period_option = new TclIntOption("-divide_by", 0, 0);
  addOption(period_option);

  auto* multiply_by_option = new TclIntOption("-multiply_by", 0, 0);
  addOption(multiply_by_option);

  auto* edges_option = new TclIntListOption("-edges", 0, {});
  addOption(edges_option);

  auto* name_option = new TclStringOption("-name", 0, nullptr);
  addOption(name_option);

  // TODO: should be a string list
  auto* source_option = new TclStringOption("-source", 0, nullptr);
  addOption(source_option);

  auto* duty_cycle_option = new TclDoubleOption("-duty_cycle", 0, 0.0);
  addOption(duty_cycle_option);

  auto* invert_option = new TclSwitchOption("-invert");
  addOption(invert_option);

  auto* edge_shift_option = new TclDoubleListOption("-edge_shift", 0, {});
  addOption(edge_shift_option);

  auto* add_option = new TclSwitchOption("-add");
  addOption(add_option);

  auto* master_clock_option = new TclStringOption("-master_clock", 0, nullptr);
  addOption(master_clock_option);

  auto* comment_option = new TclStringOption("-comment", 0, nullptr);
  addOption(comment_option);

  // creat a string list arg
  // option(arg) initialization list:
  // "source_objects" -- option name
  //       1          -- arg
  //       {}         -- default value
  auto* pin_port_arg = new TclStringListOption("source_objects", 1, {});
  addOption(pin_port_arg);
}

/**
 * @brief The create_generate_clock cmd legally check.
 *
 * @return 1 if all options/args are legal, 0 if not
 */
unsigned CmdCreateGeneratedClock::check() {
  // rule: "-source" and "source_objects" are required
  TclOption* source_option = getOptionOrArg("-source");
  TclOption* source_obj_option = getOptionOrArg("source_objects");
  if (!(source_option->is_set_val() && source_obj_option->is_set_val())) {
    LOG_ERROR << "'-source' 'source_objects' are missing.";
    return 0;
  }

  // rule: "-edges", "-divide_by", "-multiply_by" are exclusive
  TclOption* edges_option = getOptionOrArg("-edges");
  TclOption* divide_by_option = getOptionOrArg("-divide_by");
  TclOption* multiply_by_option = getOptionOrArg("-multiply_by");
  unsigned period_val_count = divide_by_option->is_set_val() +
                              multiply_by_option->is_set_val() +
                              edges_option->is_set_val();
  if (period_val_count > 1) {
    LOG_ERROR << "'-divide_by'  '-multiply_by'  '-edges' are exclusive.";
    return 0;
  }

  TclOption* add_option = getOptionOrArg("-add");
  TclOption* master_clock_option = getOptionOrArg("-master_clock");
  TclOption* name_option = getOptionOrArg("-name");
  if (add_option->is_set_val() && master_clock_option->is_set_val()) {
    if (!(name_option->is_set_val())) {
      LOG_ERROR << "If you specify -add option, you must also use the -name "
                   "and -master_clock options. ";
      return 0;
    }
  } else if (add_option->is_set_val() ^ master_clock_option->is_set_val()) {
    LOG_ERROR << "If you specify -master_clock option, you must also use the "
                 "-add option, or the other way around";
    return 0;
  }

  // the clocks with the same source must have different names
  const char* generate_clock_name = name_option->getStringVal();
  const char* source_name = source_option->getStringVal();
  for (const auto& clock : _the_constrain->get_sdc_clocks()) {
    if ((clock.second->isGenerateClock()) &&
        // have the same source with other generate clocks?
        (dynamic_cast<const SdcGenerateCLock&>(*clock.second)
             .isSameSource(source_name)) &&
        // are the names same?
        (Str::equal(generate_clock_name, clock.second->get_clock_name()))) {
      LOG_ERROR << "the clocks with the same source must have different names";
      return 0;
    }
  }

  // generate clock may be update source later, need refactor check later.

  //  Use -add option to capture the case where multiple generated clocks must
  //  be specified on the same source
  // if (add_option->is_set_val()) {
  //   bool error_occur = true;
  //   for (const auto& clock : _the_constrain->get_sdc_clocks()) {
  //     if ((clock.second->isGenerateClock()) &&
  //         (dynamic_cast<const SdcGenerateCLock&>(*clock.second)
  //              .isSameSource(source_name))) {
  //       error_occur = false;
  //     }
  //   }

  //   if (error_occur) {
  //     LOG_ERROR << "Use -add option to capture the case where multiple "
  //                  "generated clocks must be specified on the same source";
  //     return 0;
  //   }
  // }

  // edges are odd number and not less than 3
  if (edges_option->is_set_val()) {
    unsigned edges_list_size = edges_option->getIntList().size();
    if (!((edges_list_size & 1) && (edges_list_size >= 3))) {
      LOG_ERROR << "The number of edges must be an odd number and not less "
                   "than 3. ";
      return 0;
    }
  }

  // The number of edge shifts specified must be equal to the number of edges
  // specified by the -edges option
  TclOption* edge_shift_option = getOptionOrArg("-edge_shift");
  if (edge_shift_option->is_set_val()) {
    if (edges_option->is_set_val()) {
      if (edges_option->getIntList().size() !=
          edge_shift_option->getDoubleList().size()) {
        LOG_ERROR << "The number of edge shifts specified must be equal to the "
                     "number of edges specified by the -edges option.";
        return 0;
      }
    }
  }

  // -duty_cycle requires -multiply_by option
  TclOption* duty_cycle_option = getOptionOrArg("-duty_cycle");
  if (duty_cycle_option->is_set_val()) {
    if (!multiply_by_option->is_set_val()) {
      LOG_ERROR << "-duty_cycle requires -multiply_by option.";
      return 0;
    }
  }

  return 1;
}

/**
 * @brief convert clock edge to (library unit) time
 ******************************************************************************
 * clock edge start from 1   (+ 1)
 * the time   start from 0.0 (+ period/2)
 ******************************************************************************
 * eg:
 * creat_clock with
 * period = 2.2
 * waveform = { 0.0, 1.1 }
 ******************************************************************************
 *   waveform :  |----|____|----|____|----|____|----|____|----|____|----
 * clock edge :  1    2    3    4    5    6    7    8    9    10   11
 *       time : 0.0  1.1  2.2  3.3  4.4  5.5  6.6  7.7  8.8  9.9  11.0
 ******************************************************************************
 * edge2Time(5, 2.2) returns 4.4
 *
 ******************************************************************************
 * @param edge
 * @param period
 * @return double
 */
inline double edge2Time(int edge, double period) {
  return (edge - 1) * (period / 2);
}

/**
 * @brief The create_generate_clock execute body.
 *
 * @return 1 if execution success, 0 if not
 */
unsigned CmdCreateGeneratedClock::exec() {
  _the_constrain = Sta::getOrCreateSta()->getConstrain();
  if (!check()) {
    return 0;
  }

  set_source_sdc_clock({"-source"});
  set_master_clock({"-master_clock"});
  set_generate_clock({"-name"});
  set_period_and_edges({"-divide_by", "-multiply_by", "-edges"});
  set_duty_cycle({"-duty_cycle"});
  set_edges_shift_and_invert({"-edge_shift", "-invert"});
  set_source_objects({"source_objects"});
  set_add({"-add"});
  set_comment({"-comment"});

  _the_constrain->addClock(_the_generate_clock);

  return 1;
}

// set source clock
void CmdCreateGeneratedClock::set_source_sdc_clock(
    std::vector<const char*> options) {
  TclOption* source_option = getOptionOrArg("-source");
  const char* source_name = source_option->getStringVal();

  Sta* ista = Sta::getOrCreateSta();
  Netlist* design_nl = ista->get_netlist();
  SdcConstrain* the_constrain = ista->getConstrain();
  auto clock_names = GetClockName(source_name, design_nl, the_constrain);

  if (!clock_names.empty()) {
    LOG_FATAL_IF(clock_names.size() != 1) << "beyond one clock";
    std::string clock_name = clock_names.front();
    _source_sdc_clock = _the_constrain->findClock(clock_name.c_str());
    LOG_ERROR_IF(!_source_sdc_clock)
        << "source clock " << clock_name << " not found";
  } else {
    _source_sdc_clock = nullptr;
  }
}

// master_clock or source clocks ?
void CmdCreateGeneratedClock::set_master_clock(
    std::vector<const char*> options) {
  TclOption* master_clock_option = getOptionOrArg("-master_clock");
  if (master_clock_option->is_set_val()) {
    const char* master_clock_name = master_clock_option->getStringVal();
    _source_sdc_clock = _the_constrain->findClock(master_clock_name);
    if (!_source_sdc_clock) {
      Sta* ista = Sta::getOrCreateSta();
      Netlist* design_nl = ista->get_netlist();
      auto object_list = FindObjOfSdc(master_clock_name, design_nl);

      for (auto& object : object_list) {
        std::visit(overloaded{
                       [this](SdcCommandObj* sdc_obj) {
                         _source_sdc_clock = dynamic_cast<SdcClock*>(sdc_obj);
                       },
                       [](DesignObject* design_obj) {
                         LOG_FATAL << "not support design obj.";
                       },

                   },
                   object);
      }

      LOG_FATAL_IF(!_source_sdc_clock)
          << "not found master clock " << master_clock_name;
    }
  }
}

// creat generate clock
void CmdCreateGeneratedClock::set_generate_clock(
    std::vector<const char*> options) {
  TclOption* name_option = getOptionOrArg("-name");

  const char* generate_clock_name;

  LOG_FATAL_IF(!name_option->is_set_val());
  if (name_option->is_set_val()) {
    generate_clock_name = name_option->getStringVal();
  }

  _the_generate_clock = new SdcGenerateCLock(generate_clock_name);

  if (!_source_sdc_clock) {
    // set source pins
    TclOption* source_option = getOptionOrArg("-source");
    const char* generate_source_pins = nullptr;
    std::set<DesignObject*> objs;
    if (source_option->is_set_val()) {
      generate_source_pins = source_option->getStringVal();

      if (Str::startWith(generate_source_pins,
                         TclEncodeResult::get_encode_preamble())) {
        auto* obj_collection = static_cast<SdcCollection*>(
            TclEncodeResult::decode(generate_source_pins));
        auto& obj_list = obj_collection->get_collection_objs();
        for (auto obj : obj_list) {
          std::visit(overloaded{
                         [](SdcCommandObj* sdc_obj) {
                           LOG_FATAL << "should not be sdc obj.";
                         },
                         [&objs](DesignObject* design_obj) {
                           objs.insert(design_obj);
                         },
                     },
                     obj);
        }
      } else {
        Sta* ista = Sta::getOrCreateSta();
        Netlist* design_nl = ista->get_netlist();
        auto pin_ports = design_nl->findObj(generate_source_pins, false, false);

        for (auto* design_obj : pin_ports) {
          objs.insert(design_obj);
        }
      }
    }
    _the_generate_clock->set_source_pins(std::move(objs));
    _the_generate_clock->set_is_need_update_source_clock();
  } else {
    const char* source_name = _source_sdc_clock->get_clock_name();
    _the_generate_clock->set_source_name(source_name);
  }
}

// set generate clock period
// If a generated clock is specified with a divide_factor value that is a
// power of 2 (1, 2, 4, ...), the rising edges of the master clock are used
// to determine the edges of the generated clock. If the divide_factor value
// is not a power of two, the edges are scaled from the master clock edges.
// ?
void CmdCreateGeneratedClock::set_period_and_edges(
    std::vector<const char*> options) {
  if (_source_sdc_clock) {
    auto source_edges = _source_sdc_clock->get_edges();
    auto the_generate_edges = source_edges;  // copy to generate edges.
    double master_period = _source_sdc_clock->get_period();
    double generate_clock_period = master_period;

    TclOption* edges_option = getOptionOrArg("-edges");
    if (edges_option->is_set_val()) {
      auto edges_list = edges_option->getIntList();
      // set period
      double period_start = edge2Time(*edges_list.begin(), master_period);
      double period_end = edge2Time(*edges_list.rbegin(), master_period);
      generate_clock_period = period_end - period_start;
      // set edges
      for (size_t i = 0; i < the_generate_edges.size(); ++i) {
        the_generate_edges[i] = edge2Time(edges_list[i], master_period);
      }
    }

    TclOption* divide_by_option = getOptionOrArg("-divide_by");

    if (divide_by_option->is_set_val()) {
      int divide_by_value = divide_by_option->getIntVal();
      generate_clock_period = master_period * divide_by_value;
      // set edges
      for (size_t i = 0; i < the_generate_edges.size(); ++i) {
        the_generate_edges[i] *= divide_by_value;
      }
    }

    TclOption* multiply_by_option = getOptionOrArg("-multiply_by");
    if (multiply_by_option->is_set_val()) {
      int multiply_by_value = multiply_by_option->getIntVal();
      generate_clock_period = master_period / multiply_by_value;
      // set edges
      for (size_t i = 0; i < the_generate_edges.size(); ++i) {
        the_generate_edges[i] /= multiply_by_value;
      }
    }

    _the_generate_clock->set_period(generate_clock_period);
    _the_generate_clock->set_edges(std::move(the_generate_edges));
  } else {
    TclOption* divide_by_option = getOptionOrArg("-divide_by");

    if (divide_by_option->is_set_val()) {
      int divide_by_value = divide_by_option->getIntVal();
      _the_generate_clock->set_divide_by(divide_by_value);
    }
  }
}

// TODO: duty_cycle (used to modify edge? how?)
void CmdCreateGeneratedClock::set_duty_cycle(std::vector<const char*> options) {
  TclOption* duty_cycle_option = getOptionOrArg("-duty_cycle");
  if (duty_cycle_option->is_set_val()) {
    _duty_cycle = duty_cycle_option->getDoubleVal();
  }
}

// edges
void CmdCreateGeneratedClock::set_edges_shift_and_invert(
    std::vector<const char*> options) {
  Sta* ista = Sta::getOrCreateSta();

  auto& the_generate_edges = _the_generate_clock->get_edges();
  // edges shift
  TclOption* edge_shift_option = getOptionOrArg("-edge_shift");
  if (edge_shift_option->is_set_val()) {
    auto edge_shift_list = edge_shift_option->getDoubleList();
    for (size_t i = 0; i < edge_shift_list.size(); ++i) {
      the_generate_edges[i] += ista->convertTimeUnit(edge_shift_list[i]);
    }
  }

  // edges invert
  TclOption* invert_option = getOptionOrArg("-invert");
  if (invert_option->is_set_val()) {
    if (!the_generate_edges.empty()) {
      size_t i = 0;
      double time_after_rising = the_generate_edges[1] - the_generate_edges[0];
      // invert by shift backward
      for (; i < the_generate_edges.size() - 1; ++i) {
        the_generate_edges[i] = the_generate_edges[1 + i];
      }
      // add first time (time_after_rising) to last
      the_generate_edges[i] += time_after_rising;
    } else {
      _the_generate_clock->set_is_waveform_inv();
    }
  }
}

void CmdCreateGeneratedClock::set_source_objects(
    std::vector<const char*> options) {
  Sta* ista = Sta::getOrCreateSta();
  Netlist* design_nl = ista->get_netlist();
  TclOption* pin_port_option = getOptionOrArg("source_objects");
  std::vector<std::string> pin_port_strs = pin_port_option->getStringList();
  std::set<DesignObject*> pins;
  for (auto& pin_port_name : pin_port_strs) {
    auto object_list = FindObjOfSdc(pin_port_name, design_nl);
    LOG_FATAL_IF(object_list.empty()) << "object list is empty.";

    for (auto& object : object_list) {
      std::visit(overloaded{
                     [](SdcCommandObj* sdc_obj) {
                       LOG_FATAL << "not support sdc obj.";
                     },
                     [&pins, this](DesignObject* design_obj) {
                       DLOG_INFO << "create generate clock "
                                 << _the_generate_clock->get_clock_name()
                                 << " for pin/port: " << design_obj->get_name();
                       pins.insert(design_obj);
                     },
                 },
                 object);
    }
  }
  _the_generate_clock->set_objs(std::move(pins));
}

// add (or overwrite) the_generate_clock
void CmdCreateGeneratedClock::set_add(std::vector<const char*> options) {
  TclOption* add_option = getOptionOrArg("-add");
  if (!add_option->is_set_val()) {
    // remove old clock(s?) (from the same source)
    for (const auto& clock : _the_constrain->get_sdc_clocks()) {
      if ((clock.second->isGenerateClock()) &&
          (dynamic_cast<const SdcGenerateCLock&>(*clock.second)
               .isSameSource(_the_generate_clock->get_clock_name()))) {
        // TODO: delete this clock (or all clocks from the same source) ?
        // or overwrite in the first place?
        LOG_ERROR << "todo: overwrite this clock";
      }
    }
  }
}

// comment
void CmdCreateGeneratedClock::set_comment(std::vector<const char*> options) {
  TclOption* comment_option = getOptionOrArg("-comment");
  if (comment_option->is_set_val()) {
    const char* comment_value = comment_option->getStringVal();
    _the_generate_clock->set_comment(comment_value);
  }
}

}  // namespace ista
