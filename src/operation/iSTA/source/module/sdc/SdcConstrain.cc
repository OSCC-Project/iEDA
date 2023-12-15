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
 * @file sdcConstrain.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2020-11-27
 */

#include "SdcConstrain.hh"

#include "SdcException.hh"
#include "SdcSetClockLatency.hh"
#include "SdcSetClockUncertainty.hh"
#include "SdcSetIODelay.hh"
#include "SdcTimingDRC.hh"
#include "SdcTimingDerate.hh"
#include "log/Log.hh"
#include "netlist/DesignObject.hh"
#include "tcl/ScriptEngine.hh"

namespace ista {

SdcConstrain::SdcConstrain() = default;

SdcConstrain::~SdcConstrain() = default;

/**
 * @brief Store the sdc clock in the sdcConstrain.
 *
 * @param clock_name The created clock name.
 * @param clock The create_clock information.
 */
void SdcConstrain::addClock(SdcClock* clock) {
  _sdc_clocks[clock->get_clock_name()] = std::unique_ptr<SdcClock>(clock);
}

// void SdcConstrain::addGeneratedClock(SdcGenerateCLock* clock) {
//   addClock(clock);
//   std::set<DesignObject*> iter = clock->get_source_pins();
//   _generated_source_pins.insert(iter.begin(), iter.end());
//   LOG_INFO << "-------------------current generated source pin num: "
//            << _generated_source_pins.size() << std::endl;
// }

/**
 * @brief Find sdc clock.
 *
 * @param clock_name
 * @return SdcClock*
 */
SdcClock* SdcConstrain::findClock(const char* clock_name) {
  if (auto it = _sdc_clocks.find(clock_name); it != _sdc_clocks.end()) {
    return it->second.get();
  }
  return nullptr;
}

/**
 * @brief Find the sdc clock accord obj, we need propagate the obj downstream to
 * get the clock, but now we have no time to do, so we find the clock on the
 * obj.
 *
 * @param design_obj
 * @return SdcClock*
 */
SdcClock* SdcConstrain::findClock(DesignObject* design_obj) {
  for (auto& [clock_name, sdc_clock] : _sdc_clocks) {
    auto& objs = sdc_clock->get_objs();
    if (objs.end() != objs.find(design_obj)) {
      return sdc_clock.get();
    }
  }
  return nullptr;
}

/**
 * @brief Store the IO constrain.
 *
 * @param io_constrain
 */
void SdcConstrain::addIOConstrain(SdcIOConstrain* io_constrain) {
  _sdc_io_constraints.emplace_back(io_constrain);
}

/**
 * @brief Store the timing derate.
 *
 * @param timing_derate
 */
void SdcConstrain::addTimingDerate(SdcTimingDerate* timing_derate) {
  _sdc_timing_derates.emplace_back(timing_derate);
}

/**
 * @brief Store the timing DRC.
 *
 * @param timing_drc
 */
void SdcConstrain::addTimingDRC(SdcTimingDRC* timing_drc) {
  _sdc_timing_drcs.emplace_back(timing_drc);
}

/**
 * @brief Store the timing latency.
 *
 * @param timing_latency
 */
void SdcConstrain::addTimingLatency(SdcSetClockLatency* timing_latency) {
  _sdc_clock_latencys.emplace_back(timing_latency);
}

/**
 * @brief Store the timing uncertainty.
 *
 * @param timing_uncertainty
 */
void SdcConstrain::addTimingUncertainty(
    SdcSetClockUncertainty* timing_uncertainty) {
  _sdc_clock_uncertaintys.emplace_back(timing_uncertainty);
}

/**
 * @brief Judge whether the clock1 and clock2 is async.
 *
 * @param clock_name1
 * @param clock_name2
 * @return true
 * @return false
 */
bool SdcConstrain::isInAsyncGroup(std::string& clock_name1,
                                  std::string& clock_name2) {
  auto it = std::find_if(_sdc_clock_groups.begin(), _sdc_clock_groups.end(),
                         [&clock_name1, &clock_name2](auto& clock_groups) {
                           return clock_groups->isInAsyncGroup(clock_name1,
                                                               clock_name2);
                         });
  return it != _sdc_clock_groups.end();
}

void SdcConstrain::addSdcException(SdcException* sdc_exception) {
  _sdc_exceptions.emplace_back(sdc_exception);
}

/**
 * @brief Find the matched sdc object or design object.
 *
 * @param pin_port_name
 * @param design_nl
 * @return std::vector<SdcCollectionObj>
 */
std::vector<SdcCollectionObj> FindObjOfSdc(const std::string& pin_port_name,
                                           Netlist* design_nl) {
  std::vector<SdcCollectionObj> objs;
  if (Str::startWith(pin_port_name.c_str(),
                     ieda::TclEncodeResult::get_encode_preamble())) {
    auto* obj_collection = static_cast<SdcCollection*>(
        ieda::TclEncodeResult::decode(pin_port_name.c_str()));
    auto& obj_list = obj_collection->get_collection_objs();
    objs = obj_list;
  } else {
    auto pin_ports = design_nl->findObj(pin_port_name.c_str(), false, false);

    for (auto* design_obj : pin_ports) {
      objs.emplace_back(design_obj);
    }
  }

  return objs;
}

/**
 * @brief Get the clock name for the option -clock, which may be use get_clocks.
 *
 * @param clock_str
 * @param design_nl
 * @return std::string
 */
std::vector<std::string> GetClockName(const char* clock_str, Netlist* design_nl,
                                      SdcConstrain* the_constrain) {
  std::vector<std::string> clock_names;
  std::string clock_name;

  if (Str::startWith(clock_str, ieda::TclEncodeResult::get_encode_preamble())) {
    auto object_list = FindObjOfSdc(clock_str, design_nl);
    LOG_FATAL_IF(object_list.empty()) << "clock " << clock_str << " is empty.";

    SdcCommandObj* sdc_clock = nullptr;

    for (auto& object : object_list) {
      std::visit(
          overloaded{
              [&sdc_clock](SdcCommandObj* sdc_obj) { sdc_clock = sdc_obj; },
              [the_constrain, &sdc_clock](DesignObject* design_obj) {
                sdc_clock = the_constrain->findClock(design_obj);
              },
          },
          object);

      if (sdc_clock) {
        clock_name = dynamic_cast<SdcClock*>(sdc_clock)->get_clock_name();
        clock_names.emplace_back(std::move(clock_name));
      }
    }

  } else {
    clock_name = clock_str;
    clock_names.emplace_back(std::move(clock_name));
  }
  return clock_names;
}

}  // namespace ista
