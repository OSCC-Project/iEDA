/**
 * @file VCDParserWrapper.cc
 * @author shaozheqing 707005020@qq.com
 * @brief The vcd parser wrapper.
 * @version 0.1
 * @date 2023-01-10
 */
#include "VCDParserWrapper.hh"

#include <functional>
#include <ranges>

#include "ThreadPool/ThreadPool.h"

namespace ipower {

/**
 * @brief count the scalar signal toggle num and glitch num.
 *
 * @return std::vector<AnnotateToggle>
 */
std::vector<AnnotateToggle> VcdScalarCounter::countTcAndGlitch() {
  auto* the_signal = get_signal();
  auto signal_hash = the_signal->hash;

  auto* trace_vcd_file = get_trace();
  auto* signal_time_values = trace_vcd_file->get_signal_values(signal_hash);

  AnnotateToggle annotate_signal_toggle;

  // count the toggle, if current signal value is rise transition or fall
  // transition, count add one.
  VCDTimedValue* prev_time_signal_value = nullptr;
  for (auto& signal_time_value : *signal_time_values) {
    if (prev_time_signal_value) {
      if (isTransition(prev_time_signal_value, &signal_time_value,
                       std::nullopt)) {
        // is transition, incr tc.
        annotate_signal_toggle.incrTC();
      } else {
        // TODO(to shaozheqing) may be glitch, need check the glitch standard.
      }
    }
    prev_time_signal_value = &signal_time_value;
  }

  return {annotate_signal_toggle};
}

/**
 * @brief count scalar signal t0,t1,tx,tz.
 *
 * @return std::vector<AnnotateTime>
 */
std::vector<AnnotateTime> VcdScalarCounter::countDuration() {
  auto* the_signal = get_signal();
  auto signal_hash = the_signal->hash;

  auto* trace_vcd_file = get_trace();
  auto* signal_time_values = trace_vcd_file->get_signal_values(signal_hash);
  // auto simulation_start_time = trace_vcd_file->start_time; // if the
  // simualtion time is begin from signal middle time, may be need consider
  // start time.
  auto simulation_end_time = _annotate_db->getSimulationEndTime();

  AnnotateTime annotate_signal_duration_time;

  // count signal t0,t1,tx,tz duration, the signal may be not start zero time,
  // need consider the start time, such as t0, we accumulate the VCD bit0 time.
  VCDTimedValue* prev_time_signal_value = nullptr;
  for (auto& signal_time_value : *signal_time_values) {
    if (prev_time_signal_value) {
      auto duration = getDuration(prev_time_signal_value, &signal_time_value);
      auto& prev_bit_value = prev_time_signal_value->value;
      DLOG_FATAL_IF(prev_bit_value.get_type() != VCD_SCALAR)
          << "the scalar signal is not bit type";
      auto one_bit_value = prev_bit_value.get_value_bit();
      auto update_duration_func =
          getUpdateDurationFunc(annotate_signal_duration_time, one_bit_value);
      update_duration_func(duration);
    }
    prev_time_signal_value = &signal_time_value;
  }

  // for last time, the signal should steady to end.
  auto& last_time_signal_value = signal_time_values->back();
  auto last_time = last_time_signal_value.time;
  auto last_bit_value = last_time_signal_value.value.get_value_bit();
  auto last_time_duration = simulation_end_time - last_time;
  auto update_duration_func =
      getUpdateDurationFunc(annotate_signal_duration_time, last_bit_value);
  update_duration_func(last_time_duration);

  return {annotate_signal_duration_time};
}

/**
 * @brief count the scalar signal tc and time duration.
 *
 */
void VcdScalarCounter::run() {
  auto annotate_signal_toggle_vec = countTcAndGlitch();
  auto annotate_signal_duration_time_vec = countDuration();

  AnnotateRecord annotate_record(
      std::move(annotate_signal_toggle_vec.front()),
      std::move(annotate_signal_duration_time_vec.front()));

  /*collect the signal scope and find the signal record, write data to record.*/
  std::vector<std::string_view> parent_instance_names;
  auto* signal_scope = _signal->scope;
  while (signal_scope && (signal_scope != _top_instance_scope)) {
    parent_instance_names.emplace_back(signal_scope->name);
    signal_scope = signal_scope->parent;
  }

  auto* record_data =
      _annotate_db->findSignalRecord(parent_instance_names, _signal->reference);
  LOG_FATAL_IF(!record_data) << "not found record data";

  *record_data = std::move(annotate_record);
}

/**
 * @brief count the bus signal toggle num and glitch num.
 *
 * @return std::vector<AnnotateToggle>
 */
std::vector<AnnotateToggle> VcdBusCounter::countTcAndGlitch() {
  auto* the_signal = get_signal();
  auto signal_hash = the_signal->hash;

  auto* trace_vcd_file = get_trace();
  auto* signal_time_values = trace_vcd_file->get_signal_values(signal_hash);

  // count the toggle, if current signal value is rise transition or fall
  // transition, count add one.
  VCDTimedValue* prev_time_signal_value = nullptr;

  int lindex = the_signal->lindex;
  int rindex = the_signal->rindex;
  int bus_size = lindex - rindex + 1;
  std::vector<AnnotateToggle> annotate_signal_toggles(bus_size);
  // loop access to the bus signal
  for (auto i = rindex; i <= lindex; ++i) {
    int vec_i = lindex - i;  // bus signal value is high bit first.
    for (auto& signal_time_value : *signal_time_values) {
      if (prev_time_signal_value) {
        if (isTransition(prev_time_signal_value, &signal_time_value, vec_i)) {
          // is transition, incr tc.
          /*Each signal inside the bus needs to be recorded separately*/
          annotate_signal_toggles[i].incrTC();

        } else {
          // TODO(to shaozheqing) may be glitch, need check the glitch standard.
        }
      }
      prev_time_signal_value = &signal_time_value;
    }
  }
  return annotate_signal_toggles;
}

/**
 * @brief count bus signal t0,t1,tx,tz.
 *
 * @return std::vector<AnnotateTime>
 */
std::vector<AnnotateTime> VcdBusCounter::countDuration() {
  auto* the_signal = get_signal();
  auto signal_hash = the_signal->hash;

  auto* trace_vcd_file = get_trace();
  auto* signal_time_values = trace_vcd_file->get_signal_values(signal_hash);
  // auto simulation_start_time = trace_vcd_file->start_time; // if the
  // simualtion time is begin from signal middle time, may be need consider
  // start time.
  auto simulation_end_time = _annotate_db->getSimulationEndTime();

  // count signal t0,t1,tx,tz duration, the signal may be not start zero time,
  // need consider the start time, such as t0, we accumulate the VCD bit0 time.

  int lindex = the_signal->lindex;
  int rindex = the_signal->rindex;
  int bus_size = lindex - rindex + 1;
  std::vector<AnnotateTime> annotate_signal_duration_times(bus_size);
  // Store the last change time of the current bit of this bus signal
  std::vector<VCDTimedValue*> prev_time_signal_values(bus_size, nullptr);

  // loop access to the bus signal
  for (auto i = rindex; i <= lindex; ++i) {
    int vec_i = lindex - i;  // bus signal value is high bit first.
    auto& annotate_signal_duration_time = annotate_signal_duration_times[i];
    bool if_first_value = true;
    for (auto& signal_time_value : *signal_time_values) {
      if (prev_time_signal_values[i]) {
        auto& prev_vector = prev_time_signal_values[i]->value;

        DLOG_FATAL_IF(prev_vector.get_type() != VCD_VECTOR)
            << "the bus signal is not vector type";

        auto bit_value = ((signal_time_value.value).get_value_vector())[vec_i];
        auto prev_bit_value = (prev_vector.get_value_vector())[vec_i];
        // if is the last signal value of this bus signal or
        //  current value is different from prev
        if ((bit_value != prev_bit_value) ||
            (&signal_time_value == &signal_time_values->back())) {
          auto duration =
              getDuration(prev_time_signal_values[i], &signal_time_value);
          auto update_duration_func = getUpdateDurationFunc(
              annotate_signal_duration_time, prev_bit_value);
          update_duration_func(duration);
          // Each signal's last change time inside the bus needs to be recorded
          // separately
          prev_time_signal_values[i] = &signal_time_value;
        }
      }
      if (if_first_value) {
        prev_time_signal_values[i] = &signal_time_value;
        if_first_value = false;
      }
    }
  }

  // for last time, the signal should steady to end.
  auto& last_time_signal_value = signal_time_values->back();
  for (auto i = rindex; i <= lindex; ++i) {
    int vec_i = lindex - i;
    auto& annotate_signal_duration_time = annotate_signal_duration_times[i];
    auto last_time = last_time_signal_value.time;
    auto last_bit_value =
        last_time_signal_value.value.get_value_vector()[vec_i];
    auto last_time_duration = simulation_end_time - last_time;
    auto update_duration_func =
        getUpdateDurationFunc(annotate_signal_duration_time, last_bit_value);
    update_duration_func(last_time_duration);
  }

  return annotate_signal_duration_times;
}

/**
 * @brief count the bus signal tc and time duration.
 *
 */
void VcdBusCounter::run() {
  auto annotate_signal_toggle_vec = countTcAndGlitch();

  auto annotate_signal_duration_time_vec = countDuration();

  LOG_FATAL_IF(annotate_signal_toggle_vec.size() !=
               annotate_signal_duration_time_vec.size())
      << "the bus signal record is incomplete";

  for (std::size_t i = 0; i < annotate_signal_toggle_vec.size(); i++) {
    AnnotateRecord annotate_record(
        std::move(annotate_signal_toggle_vec[i]),
        std::move(annotate_signal_duration_time_vec[i]));

    /*collect the signal scope and find the signal record, write data to
     * record.*/
    // TODO, refactor to function.
    std::vector<std::string_view> parent_instance_names;
    auto* signal_scope = _signal->scope;
    while (signal_scope && (signal_scope != _top_instance_scope)) {
      parent_instance_names.emplace_back(signal_scope->name);
      signal_scope = signal_scope->parent;
    }

    auto* record_data = _annotate_db->findSignalRecord(
        parent_instance_names,
        _signal->reference + "[" + std::to_string(i) + "]");

    LOG_FATAL_IF(!record_data) << "not found record data";

    *record_data = std::move(annotate_record);
  }
}

/**
 * @brief read the vcd file.
 *
 * @param vcd_path
 * @param begin_end_time the trace begin-end time, unit is vcd time scale.
 * @return true
 * @return false
 */
bool VcdParserWrapper::readVCD(
    std::string_view vcd_path,
    // FIXME change to optional begin and endtime
    std::optional<std::pair<int64_t, int64_t>> begin_end_time) {
  // assume begin time and end time scale is the same with vcd file.
  VCDFileParser parser;
  if (begin_end_time) {
    std::tie(_begin_time, _end_time) = begin_end_time.value();
    parser.start_time = _begin_time.value();
    parser.end_time = _end_time.value();
  }

  std::string vcd_file_path(vcd_path.data(), vcd_path.size());

  auto* trace_file = parser.parse_file(vcd_file_path);
  _trace.reset(trace_file);

  return _trace ? true : false;
}

/**
 * @brief build annotate database according the vcd scope.
 *
 * @return unsigned
 */
unsigned VcdParserWrapper::buildAnnotateDB(
    const std::string& top_instance_name) {
  /*found the top instance scope.*/
  std::function<VCDScope*(VCDScope*)> traverse_scope =
      [&traverse_scope, &top_instance_name](auto* parent_scope) -> VCDScope* {
    auto children_scopes = parent_scope->children;
    for (auto* child_scope : children_scopes) {
      if (child_scope->name == top_instance_name) {
        return child_scope;
      }
    }

    for (auto* child_scope : children_scopes) {
      auto* found_scope = traverse_scope(child_scope);
      if (found_scope) {
        return found_scope;
      }
    }

    return nullptr;
  };

  auto* root_scope = _trace->root_scope;
  VCDScope* found_scope = nullptr;
  if (root_scope->name == top_instance_name) {
    found_scope = root_scope;
  } else {
    found_scope = traverse_scope(root_scope);
  }
  LOG_FATAL_IF(!found_scope) << "not found the scope" << top_instance_name;

  _top_instance_scope = found_scope;

  // TODO(to shaozheqing),config the annotate simualtion time and time scale.
  /*set simualtion time for annotate database*/
  if (_end_time) {
    // User set timescale end time
    if (_begin_time) {
      // User set timescale begin time
      _annotate_db.set_simulation_duration(_end_time.value() -
                                           _begin_time.value());
    } else {
      _annotate_db.set_simulation_duration(_end_time.value());
    }
  } else {
    // User did not set timescale end time, use the simulation end time
    // FIXME(to shaozheqing) set simulation end time
    int64_t simulation_end_time = _trace->get_timestamps()->back();
    _annotate_db.set_simulation_duration(simulation_end_time);
  }
  /*set timescale for annotate database*/
  auto time_scale = _trace->time_resolution;
  auto scale_unit = _trace->time_units;
  _annotate_db.set_timescale(time_scale, scale_unit);

  /*build annotate database according the scope*/
  std::function<void(VCDScope*, AnnotateInstance*)>
      build_scope_instance_signal =
          [this, &build_scope_instance_signal](
              VCDScope* the_scope, AnnotateInstance* parent_instance) {
            auto the_scope_instance =
                std::make_unique<AnnotateInstance>(the_scope->name);
            auto* the_scope_instance_ptr = the_scope_instance.get();
            if (!parent_instance) {
              _annotate_db.set_top_instance(std::move(the_scope_instance));
            } else {
              parent_instance->addChildInstance(std::move(the_scope_instance));
            }

            auto& scope_signals = the_scope->signals;
            for (auto* signal : scope_signals) {
              if (signal->type != VCD_VAR_WIRE) {
                continue;
              }
              if (signal->size == 1) {
                // scalar signal
                auto annotate_signal =
                    std::make_unique<AnnotateSignal>(signal->reference);
                the_scope_instance_ptr->addSignal(std::move(annotate_signal));
              } else {
                // bus signal
                int lindex = signal->lindex;
                int rindex = signal->rindex;
                for (auto i = rindex; i <= lindex; ++i) {
                  auto annotate_signal = std::make_unique<AnnotateSignal>(
                      signal->reference + "[" + std::to_string(i) + "]");
                  the_scope_instance_ptr->addSignal(std::move(annotate_signal));
                }
              }
            }

            auto children_scope = the_scope->children;
            for (auto* child_scope : children_scope) {
              build_scope_instance_signal(child_scope, the_scope_instance_ptr);
            }
          };

  build_scope_instance_signal(_top_instance_scope, nullptr);

  return 1;
}

/**
 * @brief calc the toggle and sp of the scope, which specify by the top
 * instance name.
 *
 * @param top_instance_name the top scope name.
 * @return unsigned return 1 if success, else 0.
 */
unsigned VcdParserWrapper::calcScopeToggleAndSp() {
  /*lambda function of count signal tc etc.*/
  auto count_signal = [this](auto* scope_signal) {
    if (scope_signal->size == 1) {
      // scalar signal
      VcdScalarCounter signal_counter(_top_instance_scope, scope_signal,
                                      _trace.get(), &_annotate_db);
      signal_counter.run();
    } else {
      // bus signal
      VcdBusCounter bus_signal_counter(_top_instance_scope, scope_signal,
                                       _trace.get(), &_annotate_db);
      bus_signal_counter.run();
    }
  };
  /*first, traverse the scope signal, build the counter thread.*/
  std::size_t num_thread = 48;
  ThreadPool thread_pool(num_thread);
  /*traverse the hier scope*/
  std::function<void(VCDScope*)> traverse_scope =
      [&traverse_scope, &count_signal, &thread_pool, this](auto* parent_scope) {
        // Calculate the signal of the current layer scope
        for (auto* scope_signal : parent_scope->signals) {
          if (scope_signal->type != VCD_VAR_WIRE) {
            continue;
          }
/*Select whether to use multithreading for count signal*/
#if 1
          thread_pool.enqueue(
              [this, count_signal](auto* scope_signal) {
                /*then count the signal tc,t1,t0 etc.*/
                count_signal(scope_signal);
              },
              scope_signal);
#else
          count_signal(scope_signal);
#endif
        }
        // View the next level of the scope
        auto children_scopes = parent_scope->children;
        for (auto* child_scope : children_scopes) {
          traverse_scope(child_scope);
        }
      };

  traverse_scope(_top_instance_scope);

  return 1;
}

}  // namespace ipower