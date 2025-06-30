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
 * @file PwrData.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief The implemention of power data.
 * @version 0.1
 * @date 2023-02-09
 */
#include "PwrData.hh"

#include "PwrVertex.hh"

namespace ipower {

/**
 * @brief get toggle data relative clock.
 *
 * @return std::string
 */
std::string PwrToggleData::getRelativeClockName() {
  LOG_FATAL_IF(_clock_domain.valueless_by_exception() == true)
      << "clock domain is null";

  std::string clock_name;
  std::visit(overloaded{[&clock_name](PwrClock* power_clock) {
                          clock_name = power_clock->get_clock_name();
                        },
                        [&clock_name](StaClock* the_own_clock) {
                          clock_name = the_own_clock->get_clock_name();
                        }},
             _clock_domain);
  return clock_name;
}

/**
 * @brief get toggle relative clock period.
 *
 * @return double
 */
double PwrToggleData::getRelativeClockPeriodNs() {
  LOG_FATAL_IF(_clock_domain.valueless_by_exception() == true)
      << "clock domain is null";

  double period_ns;
  std::visit(overloaded{[&period_ns](PwrClock* power_clock) {
                          period_ns = power_clock->get_clock_period_ns();
                        },
                        [&period_ns](StaClock* the_own_clock) {
                          period_ns = the_own_clock->getPeriodNs();
                        }},
             _clock_domain);
  return period_ns;
}

/**
 * @brief get toggle rate relative the vertex owned clock domain.
 *
 * @return double
 */
double PwrToggleData::getToggleRateRelativeToClock() {
  double toggle_rate = _toggle * getRelativeClockPeriodNs();
  return toggle_rate;
}

/**
 * @brief set toggle rate
 *
 * @param toggle_relative_clock
 * @return void
 */
void PwrToggleData::setToggle(double toggle_relative_clock) {
  _toggle = toggle_relative_clock / getRelativeClockPeriodNs();
}

/**
 * @brief Add data to bucket.
 *
 * @param data
 * @param track_stack_deep
 */
void PwrDataBucket::addData(PwrData* data, int track_stack_deep) {
  static const auto cmp = [](PwrData* left, PwrData* right) -> unsigned {
    double left_compare_value = left->getCompareValue();
    double right_compare_value = right->getCompareValue();
    return (left_compare_value > right_compare_value);
  };

  track_stack_deep++;
  if (_data_list.empty()) {
    insertData(data);
  } else {
    auto& top_data = _data_list.front();

    // check the data source.
    if (top_data->compareSignature(data)) {
      auto q = _data_list.begin();  // q is the previous data.
      bool is_insert = false;
      for (auto p = _data_list.begin(); p != _data_list.end(); q = p++) {
        // Put the biggest case(toggle or SP is bigger) in front.
        if (cmp(data, p->get())) {
          if (p == _data_list.begin()) {
            _data_list.emplace_front(data);
          } else {
            _data_list.emplace_after(q, data);
          }
          is_insert = true;
          ++_count;
          break;
        }
      }

      if (!is_insert) {
        if (data->get_data_source() == PwrDataSource::kAnnotate ||
            data->get_data_source() == PwrDataSource::kDataPropagation) {
          // Takes only the n_worst amount of data.
          if (_count < _n_worst) {
            _data_list.emplace_after(q, data);
            ++_count;
          }
        } else {
          _data_list.emplace_after(q, data);
          ++_count;
        }
      }

      // erase the beyond limit data.
      if (data->get_data_source() == PwrDataSource::kAnnotate ||
          data->get_data_source() == PwrDataSource::kDataPropagation) {
        if (_count > _n_worst) {
          auto begin_pos = _data_list.begin();
          auto delete_prev_pos = std::next(begin_pos, _n_worst - 1);
          // auto& delete_data = *(std::next(delete_prev_pos));
          // TODO fwd
          // delete_data->get_bwd()->set_fwd(nullptr);

          _data_list.erase_after(delete_prev_pos);
          --_count;
        }
      }
    } else {
      if (_next) {
        _next->addData(data, track_stack_deep);
      } else {
        _next = std::make_unique<PwrDataBucket>(_n_worst);
        _next->insertData(data);
      }
    }
  }
}

/**
 * @brief Get (Propagation/Default/Annotate) data.
 *
 * @return PwrData*
 */
PwrData* PwrDataBucket::frontData(PwrDataSource data_source) {
  PwrData* propagation_data = nullptr;
  if (!_data_list.empty()) {
    PwrData* front_data = _data_list.front().get();
    if (front_data->get_data_source() == data_source) {
      propagation_data = front_data;
    } else if (_next) {
      propagation_data = _next->frontData(data_source);
    }
  }
  return propagation_data;
}

/**
 * @brief Judge whether has next data.
 *
 * @return true if has next.
 * @return false
 */
bool PwrDataBucketIterator::hasNext() {
  if (_iter != _data_bucket->_data_list.end()) {
    return true;
  }

  bool is_ok = false;

  auto* data_bucket = _data_bucket->_next.get();
  if (data_bucket) {
    _data_bucket = data_bucket;
    _iter = data_bucket->_data_list.begin();
    is_ok = true;
  }

  return is_ok;
}

}  // namespace ipower
