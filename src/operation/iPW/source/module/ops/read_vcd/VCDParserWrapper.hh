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
 * @file VCDParserWrapper.hh
 * @author shaozheqing 707005020@qq.com
 * @brief The class for vcd parser wrapper.
 * @version 0.1
 * @date 2023-01-10
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>

#include "VCDFileParser.hpp"
#include "log/Log.hh"
#include "ops/annotate_toggle_sp/AnnotateData.hh"

namespace ipower {

using TimeUnit = int64_t;

/**
 * @brief The base class for vcd counter, which cout the signal tc,t1,t0 etc.
 *
 */
class VcdCounter {
 public:
  VcdCounter(VCDScope* top_instance_scope, VCDSignal* signal, VCDFile* trace,
             AnnotateDB* annotate_db)
      : _top_instance_scope(top_instance_scope),
        _signal(signal),
        _trace(trace),
        _annotate_db(annotate_db){};
  virtual ~VcdCounter() = default;

  auto* get_signal() { return _signal; }
  auto* get_trace() { return _trace; }

  virtual std::vector<AnnotateToggle> countTcAndGlitch() = 0;
  virtual std::vector<AnnotateTime> countDuration() = 0;

  virtual void run() = 0;

 protected:
  VCDScope*
      _top_instance_scope;   //!< The signal top instance, for searched record.
  VCDSignal* _signal;        //!< The signal which the counter process.
  VCDFile* _trace;           //!< The trace vcd file.
  AnnotateDB* _annotate_db;  //!< The annotate db of store tc, t0, t1 etc.
};

/**
 * @brief The non bus signal counter.
 *
 */
class VcdScalarCounter : public VcdCounter {
 public:
  explicit VcdScalarCounter(VCDScope* top_instance_scope, VCDSignal* signal,
                            VCDFile* trace, AnnotateDB* annotate_db)
      : VcdCounter(top_instance_scope, signal, trace, annotate_db){};
  ~VcdScalarCounter() override = default;

  std::vector<AnnotateToggle> countTcAndGlitch() override;
  std::vector<AnnotateTime> countDuration() override;

  void run() override;

 protected:
  /*judge whether is transition, 0-1 or 1-0.*/
  bool isTransition(VCDTimedValue* prev_signal_value,
                    VCDTimedValue* curr_signal_value,
                    std::optional<int> bus_index) {
    // check time first, then check the bit is not VCD_X, if the bit value is
    // not the same, judge is transition.
    auto prev_time = prev_signal_value->time;
    auto curr_time = curr_signal_value->time;
    if (prev_time == curr_time) {
      return false;
    }

    auto& prev_value = prev_signal_value->value;
    auto& curr_value = curr_signal_value->value;

    auto prev_bit_value =
        bus_index ? (prev_value.get_value_vector())[bus_index.value()]
                  : prev_value.get_value_bit();
    auto curr_bit_value =
        bus_index ? (curr_value.get_value_vector())[bus_index.value()]
                  : curr_value.get_value_bit();

    return (prev_bit_value != VCD_X) && (curr_bit_value != VCD_X) &&
           (prev_bit_value != curr_bit_value);
  }
  auto getDuration(VCDTimedValue* prev_time_signal_value,
                   VCDTimedValue* curr_time_signal_value) {
    auto prev_time = prev_time_signal_value->time;
    auto cur_time = curr_time_signal_value->time;

    return cur_time - prev_time;
  }
  auto getUpdateDurationFunc(AnnotateTime& annotate_signal_duration_time,
                             VCDBit bit_value) {
    std::map<VCDBit, std::function<void(int64_t duration)>>
        update_duration_funcs = {
            {VCD_0,
             std::bind(&AnnotateTime::incrT0, &annotate_signal_duration_time,
                       std::placeholders::_1)},
            {VCD_1,
             std::bind(&AnnotateTime::incrT1, &annotate_signal_duration_time,
                       std::placeholders::_1)},
            {VCD_X,
             std::bind(&AnnotateTime::incrTX, &annotate_signal_duration_time,
                       std::placeholders::_1)},
            {VCD_Z,
             std::bind(&AnnotateTime::incrTZ, &annotate_signal_duration_time,
                       std::placeholders::_1)},
        };

    return update_duration_funcs[bit_value];
  }
};

/**
 * @brief The bus signal counter.
 *
 */
class VcdBusCounter : public VcdScalarCounter {
 public:
  explicit VcdBusCounter(VCDScope* top_instance_scope, VCDSignal* signal,
                         VCDFile* trace, AnnotateDB* annotate_db)
      : VcdScalarCounter(top_instance_scope, signal, trace, annotate_db){};
  ~VcdBusCounter() override = default;

  std::vector<AnnotateToggle> countTcAndGlitch() override;
  std::vector<AnnotateTime> countDuration() override;

  void run() override;
};

/**
 * @brief The vcd wrapper class of the vcd parser.
 *
 */
class VcdParserWrapper {
 public:
  VcdParserWrapper() = default;
  ~VcdParserWrapper() = default;

  bool readVCD(
      std::string_view vcd_path,
      std::optional<std::pair<int64_t, int64_t>> begin_end_time = std::nullopt);
  unsigned buildAnnotateDB(const std::string& top_instance_name);
  unsigned calcScopeToggleAndSp();
  void printAnnotateDB(std::ostream& out) { _annotate_db.printAnnotateDB(out); }
  auto* get_annotate_db() { return &_annotate_db; }

 private:
  std::unique_ptr<VCDFile> _trace;  //!< The parsed vcd file, may be only record
                                    //!< according begin time and end time.
  VCDScope* _top_instance_scope;    //!< The specifid top instance scope of vcd.
  std::optional<int64_t> _begin_time;  //!< simulation begin time.
  std::optional<int64_t> _end_time;    //!< simulation end time.
  AnnotateDB _annotate_db;  //!< The annotate database for store waveform data.
};
}  // namespace ipower
