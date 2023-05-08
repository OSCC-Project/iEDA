/**
 * @file sdcConstrain.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2020-11-27
 *
 * @copyright Copyright (c) 2020 PCL EDA
 *
 */

#pragma once

#include <memory>
#include <vector>

#include "SdcClock.hh"
#include "SdcCollection.hh"
#include "string/StrMap.hh"

namespace ista {

class SdcClock;
class SdcIOConstrain;
class SdcTimingDerate;
class SdcTimingDRC;
class SdcSetClockLatency;
class SdcSetClockUncertainty;
class SdcException;
class DesignObject;

/**
 * @brief The class is used to store all sdc clocks.
 *
 */
class SdcConstrain {
 public:
  SdcConstrain();
  ~SdcConstrain();

  void addClock(SdcClock* clock);
  auto& get_sdc_clocks() { return _sdc_clocks; }
  SdcClock* findClock(const char* clock_name);
  SdcClock* findClock(DesignObject* design_obj);

  void addIOConstrain(SdcIOConstrain* io_constrain);
  auto& get_sdc_io_constraints() { return _sdc_io_constraints; }

  void addTimingDerate(SdcTimingDerate* timing_derate);
  auto& get_sdc_timing_derates() { return _sdc_timing_derates; }

  void addTimingDRC(SdcTimingDRC* timing_drc);
  auto& get_sdc_timing_drcs() { return _sdc_timing_drcs; }

  void addTimingLatency(SdcSetClockLatency* timing_latency);
  auto& get_sdc_clock_latencys() { return _sdc_clock_latencys; }

  void addTimingUncertainty(SdcSetClockUncertainty* timing_uncertainty);
  auto& get_sdc_clock_uncertaintys() { return _sdc_clock_uncertaintys; }

  void addClockGroups(std::unique_ptr<SdcClockGroups> clock_groups) {
    _sdc_clock_groups.emplace_back(std::move(clock_groups));
  }
  auto& get_sdc_clock_groups() { return _sdc_clock_groups; }

  bool isInAsyncGroup(std::string& clock_name1, std::string& clock_name2);

  void addSdcException(SdcException* sdc_exception);
  auto& get_sdc_exceptions() { return _sdc_exceptions; }

  void addSdcCollection(SdcCollection* sdc_collection) {
    _sdc_collections.emplace_back(sdc_collection);
  }
  auto& get_sdc_collections() { return _sdc_collections; }

 private:
  StrMap<std::unique_ptr<SdcClock>> _sdc_clocks;
  std::vector<std::unique_ptr<SdcIOConstrain>> _sdc_io_constraints;
  std::vector<std::unique_ptr<SdcTimingDerate>> _sdc_timing_derates;
  std::vector<std::unique_ptr<SdcTimingDRC>> _sdc_timing_drcs;
  std::vector<std::unique_ptr<SdcSetClockLatency>> _sdc_clock_latencys;
  std::vector<std::unique_ptr<SdcSetClockUncertainty>> _sdc_clock_uncertaintys;
  std::vector<std::unique_ptr<SdcClockGroups>> _sdc_clock_groups;
  std::vector<std::unique_ptr<SdcException>> _sdc_exceptions;
  std::vector<std::unique_ptr<SdcCollection>> _sdc_collections;
};

std::vector<SdcCollectionObj> FindObjOfSdc(const std::string& pin_port_name,
                                           Netlist* design_nl);
std::vector<std::string> GetClockName(const char* clock_str, Netlist* design_nl,
                                      SdcConstrain* the_constrain);
}  // namespace ista
