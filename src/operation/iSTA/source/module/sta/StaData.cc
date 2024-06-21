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
 * @file StaData.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of sta data.
 * @version 0.1
 * @date 2021-02-19
 */
#include "StaData.hh"

#include <utility>

#include "StaClock.hh"
#include "StaVertex.hh"

namespace ista {

StaData::StaData(AnalysisMode delay_type, TransType trans_type,
                 StaVertex* own_vertex)
    : _delay_type(delay_type),
      _trans_type(trans_type),
      _own_vertex(own_vertex) {}

StaData::StaData(const StaData& orig)
    : _delay_type(orig._delay_type),
      _trans_type(orig._trans_type),
      _own_vertex(orig._own_vertex) {}

StaData& StaData::operator=(const StaData& rhs) {
  if (this != &rhs) {
    _delay_type = rhs._delay_type;
    _trans_type = rhs._trans_type;
    _own_vertex = rhs._own_vertex;
  }
  return *this;
}

StaData::StaData(StaData&& other) noexcept
    : _delay_type(other._delay_type),
      _trans_type(other._trans_type),
      _own_vertex(other._own_vertex),
      _fwd_set(std::move(other._fwd_set)),
      _bwd(other._bwd) {}

StaData& StaData::operator=(StaData&& rhs) noexcept {
  if (this != &rhs) {
    _delay_type = rhs._delay_type;
    _trans_type = rhs._trans_type;
    _own_vertex = rhs._own_vertex;
    _fwd_set = std::move(rhs._fwd_set);
    _bwd = rhs._bwd;
  }
  return *this;
}

/**
 * @brief Compare the two slew data signature.
 *
 * @param data
 * @return unsigned
 */
unsigned StaData::compareSignature(const StaSlewData* data) const {
  return data->compareSignature(this);
}

/**
 * @brief Compare the two clock data signature.
 *
 * @param data
 * @return unsigned
 */
unsigned StaData::compareSignature(const StaClockData* data) const {
  return data->compareSignature(this);
}

/**
 * @brief Compare the two delay data signature.
 *
 * @param data
 * @return unsigned
 */
unsigned StaData::compareSignature(const StaPathDelayData* data) const {
  return data->compareSignature(this);
}

/**
 * @brief Compare the two data signature default.
 *
 * @param data
 * @return unsigned
 */
unsigned StaData::compareSignature(const StaData* data) const {
  if (!data) {
    return 0;
  }

  unsigned is_same = 1;

  if (_delay_type != data->get_delay_type()) {
    is_same = 0;
  } else if (_trans_type != data->get_trans_type()) {
    is_same = 0;
  } else if (_bwd != data->get_bwd()) {
    is_same = 0;
  }

  return is_same;
}

/**
 * @brief Get the the path begin data to current data.
 *
 * @return std::stack<StaData*>
 */
std::stack<StaData*> StaData::getPathData() {
  std::stack<StaData*> path_stack;
  path_stack.push(this);

  auto* bwd_data = dynamic_cast<StaData*>(get_bwd());
  while (bwd_data) {
    path_stack.push(bwd_data);
    bwd_data = dynamic_cast<StaData*>(bwd_data->get_bwd());
  }

  return path_stack;
}

StaSlewData::StaSlewData(AnalysisMode delay_type, TransType trans_type,
                         StaVertex* own_vertex, int slew)
    : StaData(delay_type, trans_type, own_vertex), _slew(slew) {}

StaSlewData::~StaSlewData() = default;

StaSlewData::StaSlewData(const StaSlewData& orig)
    : StaData(orig), _slew(orig._slew) {
  if (orig._output_current_data) {
    auto* new_current_data = (*(orig._output_current_data))->copy();
    _output_current_data = std::unique_ptr<LibCurrentData>(new_current_data);
  }
}

StaSlewData& StaSlewData::operator=(const StaSlewData& rhs) {
  if (this != &rhs) {
    StaData::operator=(rhs);
    _slew = rhs._slew;

    if (rhs._output_current_data) {
      auto* new_current_data = (*(rhs._output_current_data))->copy();
      _output_current_data = std::unique_ptr<LibCurrentData>(new_current_data);
    }
  }
  return *this;
}

StaSlewData::StaSlewData(StaSlewData&& other) noexcept
    : StaData(std::move(other)),
      _slew(other._slew),
      _output_current_data(std::move(other._output_current_data)) {}

StaSlewData& StaSlewData::operator=(StaSlewData&& rhs) noexcept {
  if (this != &rhs) {
    StaData::operator=(std::move(rhs));
    _slew = rhs._slew;
    _output_current_data = std::move(rhs._output_current_data);
  }
  return *this;
}

/**
 * @brief Compare two slew data.
 *
 * @param data
 * @return unsigned
 */
unsigned StaSlewData::compareSignature(const StaData* data) const {
  if (!data || !data->isSlewData()) {
    return 0;
  }

  unsigned is_same = 1;

  const auto* delay_data = dynamic_cast<const StaSlewData*>(data);
  if (_delay_type != delay_data->get_delay_type()) {
    is_same = 0;
  } else if (_trans_type != delay_data->get_trans_type()) {
    is_same = 0;
  }

  return is_same;
}

StaArcDelayData::StaArcDelayData(AnalysisMode delay_type, TransType trans_type,
                                 StaArc* own_arc, int delay)
    : StaData(delay_type, trans_type, nullptr),
      _arc_delay(delay),
      _own_arc(own_arc) {}

StaArcDelayData::StaArcDelayData(const StaArcDelayData& orig)
    : StaData(orig), _arc_delay(orig._arc_delay), _own_arc(orig._own_arc) {}

StaArcDelayData& StaArcDelayData::operator=(const StaArcDelayData& rhs) {
  if (this != &rhs) {
    StaData::operator=(rhs);
    _arc_delay = rhs._arc_delay;
    _own_arc = rhs._own_arc;
  }
  return *this;
}

StaArcDelayData::StaArcDelayData(StaArcDelayData&& other) noexcept
    : StaData(std::move(other)),
      _arc_delay(other._arc_delay),
      _own_arc(other._own_arc) {}

StaArcDelayData& StaArcDelayData::operator=(StaArcDelayData&& rhs) noexcept {
  if (this != &rhs) {
    StaData::operator=(std::move(rhs));
    _arc_delay = rhs._arc_delay;
    _own_arc = rhs._own_arc;
  }
  return *this;
}

StaPathDelayData::StaPathDelayData(AnalysisMode delay_type,
                                   TransType trans_type, int64_t arrive_time,
                                   StaClockData* launch_clock_data,
                                   StaVertex* own_vertex)
    : StaData(delay_type, trans_type, own_vertex),
      _arrive_time(arrive_time),
      _launch_clock_data(launch_clock_data) {}

StaPathDelayData::~StaPathDelayData() = default;

StaPathDelayData::StaPathDelayData(const StaPathDelayData& orig)
    : StaData(orig),
      _arrive_time(orig._arrive_time),
      _req_time(orig._req_time),
      _launch_clock_data(orig._launch_clock_data) {}

StaPathDelayData& StaPathDelayData::operator=(const StaPathDelayData& rhs) {
  if (this != &rhs) {
    StaData::operator=(rhs);
    _arrive_time = rhs._arrive_time;
    _launch_clock_data = rhs._launch_clock_data;
    _req_time = rhs._req_time;
  }
  return *this;
}

StaPathDelayData::StaPathDelayData(StaPathDelayData&& other) noexcept
    : StaData(std::move(other)),
      _arrive_time(other._arrive_time),
      _req_time(other._req_time),
      _launch_clock_data(other._launch_clock_data) {}

StaPathDelayData& StaPathDelayData::operator=(StaPathDelayData&& rhs) noexcept {
  if (this != &rhs) {
    StaData::operator=(std::move(rhs));
    _arrive_time = rhs._arrive_time;
    _launch_clock_data = rhs._launch_clock_data;
    _req_time = rhs._req_time;
  }
  return *this;
}

/**
 * @brief Compare the two clock data signature.
 *
 * @param data
 * @return unsigned
 */
unsigned StaPathDelayData::compareSignature(const StaData* data) const {
  if (!data || !data->isPathDelayData()) {
    return 0;
  }

  unsigned is_same = 1;

  const auto* delay_data = dynamic_cast<const StaPathDelayData*>(data);
  if (_delay_type != delay_data->get_delay_type()) {
    is_same = 0;
  } else if (_trans_type != delay_data->get_trans_type()) {
    is_same = 0;
  } else if (_launch_clock_data->get_prop_clock() !=
             delay_data->get_launch_clock_data()->get_prop_clock()) {
    is_same = 0;
  } else if (_launch_clock_data->get_clock_wave_type() !=
             delay_data->get_launch_clock_data()->get_clock_wave_type()) {
    is_same = 0;
  } /* else if (delay_data->get_bwd()->get_own_vertex() !=
             get_bwd()->get_own_vertex()) {
    is_same = 0;
  } */

  return is_same;
}

StaClockData::StaClockData(AnalysisMode delay_type, TransType trans_type,
                           int arrive_time, StaVertex* own_vertex,
                           StaClock* prop_clock)
    : StaData(delay_type, trans_type, own_vertex),
      _arrive_time(arrive_time),
      _prop_clock(prop_clock) {
  DLOG_FATAL_IF(!prop_clock);
}

StaClockData::~StaClockData() = default;

StaClockData::StaClockData(const StaClockData& orig)
    : StaData(orig),
      _arrive_time(orig._arrive_time),
      _prop_clock(orig._prop_clock),
      _clock_wave_type(orig._clock_wave_type) {}

StaClockData& StaClockData::operator=(const StaClockData& rhs) {
  if (this != &rhs) {
    StaData::operator=(rhs);
    _arrive_time = rhs._arrive_time;
    _prop_clock = rhs._prop_clock;
    _clock_wave_type = rhs._clock_wave_type;
  }
  return *this;
}

StaClockData::StaClockData(StaClockData&& other) noexcept
    : StaData(std::move(other)),
      _arrive_time(other._arrive_time),
      _prop_clock(other._prop_clock),
      _clock_wave_type(other._clock_wave_type) {}

StaClockData& StaClockData::operator=(StaClockData&& rhs) noexcept {
  if (this != &rhs) {
    StaData::operator=(std::move(rhs));
    _arrive_time = rhs._arrive_time;
    _prop_clock = rhs._prop_clock;
    _clock_wave_type = rhs._clock_wave_type;
  }
  return *this;
}
/**
 * @brief Compare the two clock data signature.
 *
 * @param data
 * @return unsigned
 */
unsigned StaClockData::compareSignature(const StaData* data) const {
  if (!data || !data->isClockData()) {
    return 0;
  }

  unsigned is_same = 1;

  const auto* clock_data = dynamic_cast<const StaClockData*>(data);
  if (_delay_type != clock_data->get_delay_type()) {
    is_same = 0;
  } else if (_trans_type != clock_data->get_trans_type()) {
    is_same = 0;
  } else if (_prop_clock != clock_data->get_prop_clock()) {
    is_same = 0;
  } else if (_bwd != clock_data->get_bwd()) {
    is_same = 0;
  }

  return is_same;
}

StaArcWaveformData::StaArcWaveformData(AnalysisMode delay_type,
                                       TransType trans_type,
                                       StaSlewData* from_slew_data,
                                       std::vector<Waveform>&& node_waveforms)
    : StaData(delay_type, trans_type, nullptr),
      _from_slew_data(from_slew_data),
      _node_waveforms(std::move(node_waveforms)) {}

/**
 * @brief compare the two waveform signature.
 *
 * @param data
 * @return unsigned
 */
unsigned StaArcWaveformData::compareSignature(const StaData* data) const {
  if (!data || !data->isWaveformData()) {
    return 0;
  }

  unsigned is_same = 1;

  const auto* waveform_data = dynamic_cast<const StaArcWaveformData*>(data);
  if (_delay_type != waveform_data->get_delay_type()) {
    is_same = 0;
  } else if (_trans_type != waveform_data->get_trans_type()) {
    is_same = 0;
  } else if (_own_arc != waveform_data->get_own_arc()) {
    is_same = 0;
  } else if (_from_slew_data != waveform_data->get_from_slew_data()) {
    is_same = 0;
  }

  return is_same;
}

StaDataBucket::StaDataBucket(unsigned n_worst) : _n_worst(n_worst) {}

StaDataBucket::StaDataBucket(StaDataBucket&& other) noexcept
    : _data_list(std::move(other._data_list)),
      _n_worst(other._n_worst),
      _next(std::move(other._next)) {}

StaDataBucket& StaDataBucket::operator=(StaDataBucket&& rhs) noexcept {
  _data_list = std::move(rhs._data_list);
  _n_worst = rhs._n_worst;
  _next = std::move(rhs._next);
  return *this;
}

/**
 * @brief Add data to bucket.
 *
 * @param data
 */
void StaDataBucket::addData(StaData* data, int track_stack_deep) {
  static const auto cmp = [](StaData* left, StaData* right) -> unsigned {
    auto delay_type = left->get_delay_type();
    int left_compare_value = left->getCompareValue();
    int right_compare_value = right->getCompareValue();

    // Judge more critical.
    return (delay_type == AnalysisMode::kMax)
               ? (left_compare_value > right_compare_value)
               : (left_compare_value < right_compare_value);
  };

  track_stack_deep++;

  // if (track_stack_deep > 4 && data->isPathDelayData()) {
  //   LOG_INFO << "Debug";
  // }

  if (_data_list.empty()) {
    insertData(data);
  } else {
    auto& top_data = _data_list.front();

    if (top_data->compareSignature(data)) {
      auto q = _data_list.begin();
      bool is_insert = false;
      // q is the previous data.
      for (auto p = _data_list.begin(); p != _data_list.end(); q = p++) {
        // whether more critical than data.
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
        _data_list.emplace_after(q, data);
        ++_count;
      }

      // erase the beyond limit data.
      if (data->isSlewData() || data->isPathDelayData()) {
        if (_count > _n_worst) {
          auto begin_pos = _data_list.begin();
          auto delete_prev_pos = std::next(begin_pos, _n_worst - 1);
          auto& delete_data = *(std::next(delete_prev_pos));

          if (delete_data->get_bwd()) {
            delete_data->get_bwd()->erase_fwd(delete_data.get());
          }

          _data_list.erase_after(delete_prev_pos);
          --_count;
        }
      }

    } else {
      if (_next) {
        _next->addData(data, track_stack_deep);
      } else {
        _next = std::make_unique<StaDataBucket>(_n_worst);
        _next->insertData(data);
      }
    }
  }
}

StaDataBucketIterator::StaDataBucketIterator(StaDataBucket& data_bucket)
    : _data_bucket(&data_bucket), _iter(data_bucket._data_list.begin()) {}

/**
 * @brief Judge whether has next data.
 *
 * @return true if has next.
 * @return false
 */
bool StaDataBucketIterator::hasNext() {
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

/**
 * @brief Get the next data.
 *
 * @return StaData* The next data.
 */
std::unique_ptr<StaData>& StaDataBucketIterator::next() { return *_iter++; }

}  // namespace ista
