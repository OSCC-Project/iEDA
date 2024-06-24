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
 * @file StaData.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of sta path delay data, include clock path delay data, data
 * path delay data.
 * @version 0.1
 * @date 2021-02-19
 */
#pragma once

#include <forward_list>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>

#include "BTreeSet.hh"
#include "Type.hh"
#include "delay/WaveformInfo.hh"
#include "log/Log.hh"

namespace ista {

class StaVertex;
class StaArc;
class StaClock;
class StaSlewData;
class StaClockData;
class StaPathDelayData;
class StaDataBucketIterator;
class LibCurrentData;

/**
 * @brief The base class of sta data.
 *
 */
class StaData {
 public:
  StaData(AnalysisMode delay_type, TransType trans_type, StaVertex* own_vertex);
  virtual ~StaData() {
    for (auto* fwd_data : _fwd_set) {
      fwd_data->set_bwd(nullptr);
    }

    if (_bwd) {
      _bwd->erase_fwd(this);
    }
  }
  StaData(const StaData& orig);
  StaData& operator=(const StaData& rhs);
  StaData(StaData&& other) noexcept;
  StaData& operator=(StaData&& rhs) noexcept;

  virtual StaData* copy() { return new StaData(*this); }

  virtual unsigned isSlewData() const { return 0; }
  virtual unsigned isClockData() const { return 0; }
  virtual unsigned isPathDelayData() const { return 0; }
  virtual unsigned isWaveformData() const { return 0; }

  virtual int64_t get_arrive_time() const {
    LOG_FATAL << "not implemented";
    return 0;
  }
  virtual void set_arrive_time(int64_t arrive_time) {
    LOG_FATAL << "not implemented";
  }

  virtual std::optional<int> get_req_time() const {
    LOG_FATAL << "not implemented";
    return 0;
  }
  virtual void set_req_time(int req_time) { LOG_FATAL << "not implemented"; }

  virtual void incrArriveTime(int delta) { LOG_FATAL << "not implemented"; }
  AnalysisMode get_delay_type() const { return _delay_type; }
  TransType get_trans_type() const { return _trans_type; }
  void set_trans_type(TransType trans_type) { _trans_type = trans_type; }
  void flipTransType() {
    _trans_type == TransType::kRise ? _trans_type = TransType::kFall
                                    : _trans_type = TransType::kRise;
  }
  unsigned isRiseTransType() { return _trans_type == TransType::kRise; }
  unsigned isFallTransType() { return _trans_type == TransType::kFall; }

  void add_fwd(StaData* fwd) {
    std::lock_guard lk(_mt);
    _fwd_set.insert(fwd);
  }
  void erase_fwd(StaData* fwd) {
    std::lock_guard lk(_mt);
    _fwd_set.erase(fwd);
  }
  auto& get_fwd_set() const { return _fwd_set; }

  void set_bwd(StaData* bwd) { _bwd = bwd; }
  StaData* get_bwd() const { return _bwd; }

  StaVertex* get_own_vertex() { return _own_vertex; }
  void set_own_vertex(StaVertex* own_vertex) { _own_vertex = own_vertex; }

  void set_derate(std::optional<float> derate) { _derate = derate; }
  std::optional<float> get_derate() const { return _derate; }

  virtual unsigned compareSignature(const StaSlewData* data) const;
  virtual unsigned compareSignature(const StaClockData* data) const;
  virtual unsigned compareSignature(const StaPathDelayData* data) const;
  virtual unsigned compareSignature(const StaData* data) const;

  virtual int64_t getCompareValue() const {
    LOG_FATAL << "not implemention.";
    return 0;
  }

  std::stack<StaData*> getPathData();

 protected:
  AnalysisMode
      _delay_type;  //!< The delay type, max is for setup, min is for hold etc.
  TransType _trans_type;         //!< The transition type, rise/fall.
  std::optional<float> _derate;  //!< The vertex derate
  StaVertex* _own_vertex;        //!< The vertex which the data belong to.
  ieda::BTreeSet<StaData*>
      _fwd_set;  //!< The propagation fwd datas, maybe more than once.
  StaData* _bwd = nullptr;  //!< The propagation bwd data, should be one.
  std::mutex _mt;
};

/**
 * @brief The slew data of the pin.
 *
 */
class StaSlewData : public StaData {
 public:
  StaSlewData(AnalysisMode delay_type, TransType trans_type,
              StaVertex* own_vertex, int slew);
  ~StaSlewData() override;
  StaSlewData(const StaSlewData& orig);
  StaSlewData& operator=(const StaSlewData& rhs);

  StaSlewData(StaSlewData&& other) noexcept;
  StaSlewData& operator=(StaSlewData&& rhs) noexcept;

  StaSlewData* copy() override { return new StaSlewData(*this); }

  unsigned isSlewData() const override { return 1; }

  int get_slew() const { return _slew; }
  void set_slew(int slew) { _slew = slew; }

  int64_t getCompareValue() const override { return _slew; }

  unsigned compareSignature(const StaData* data) const override;

  void set_output_current_data(
      std::unique_ptr<LibCurrentData> output_current_data) {
    if (output_current_data) {
      _output_current_data = std::move(output_current_data);
    }
  }
  std::optional<LibCurrentData*> get_output_current_data() {
    return _output_current_data
               ? std::optional<LibCurrentData*>(_output_current_data->get())
               : std::nullopt;
  }

 private:
  int _slew;  //!< The slew value, unit is fs.
  std::optional<std::unique_ptr<LibCurrentData>> _output_current_data =
      std::nullopt;  //!< The output current data of driving point.
};

/**
 * @brief The arc delay data of the arc.
 *
 */
class StaArcDelayData : public StaData {
 public:
  StaArcDelayData(AnalysisMode delay_type, TransType trans_type,
                  StaArc* own_arc, int delay);
  ~StaArcDelayData() override = default;
  StaArcDelayData(const StaArcDelayData& orig);
  StaArcDelayData& operator=(const StaArcDelayData& rhs);

  StaArcDelayData(StaArcDelayData&& other) noexcept;
  StaArcDelayData& operator=(StaArcDelayData&& rhs) noexcept;

  StaArcDelayData* copy() override { return new StaArcDelayData(*this); }

  int get_arc_delay() const { return (_arc_delay + _crosstalk_delay); }
  void set_arc_delay(int arc_delay) { _arc_delay = arc_delay; }
  int64_t getCompareValue() const override { return _arc_delay; }

  int get_crosstalk_delay() const { return _crosstalk_delay; }
  void set_crosstalk_delay(int crosstalk_delay) {
    _crosstalk_delay = crosstalk_delay;
  }

 private:
  int _arc_delay;            //!< The delay value, unit is fs.
  int _crosstalk_delay = 0;  //!< The crosstalk delay, unit is fs.
  StaArc* _own_arc;          //!< The arc delay belong to.
};

/**
 * @brief The waveform data of the arc.
 *
 */
class StaArcWaveformData : public StaData {
 public:
  StaArcWaveformData(AnalysisMode delay_type, TransType trans_type,
                     StaSlewData* from_slew_data,
                     std::vector<Waveform>&& node_waveforms);
  ~StaArcWaveformData() override = default;

  StaArcWaveformData(const StaArcWaveformData& orig) = default;
  StaArcWaveformData& operator=(const StaArcWaveformData& rhs) = default;

  StaArcWaveformData(StaArcWaveformData&& other) noexcept = default;
  StaArcWaveformData& operator=(StaArcWaveformData&& rhs) noexcept = default;

  StaArcWaveformData* copy() override { return new StaArcWaveformData(*this); }

  unsigned isWaveformData() const override { return 1; }

  auto* get_from_slew_data() const { return _from_slew_data; }

  void set_own_arc(StaArc* own_arc) { _own_arc = own_arc; }
  StaArc* get_own_arc() const { return _own_arc; }

  auto& getWaveform(unsigned node_id) { return _node_waveforms.at(node_id); }

  unsigned compareSignature(const StaData* data) const override;

 private:
  StaSlewData*
      _from_slew_data;  //!< The wavform propagated from driver slew data.
  std::vector<Waveform> _node_waveforms;
  StaArc* _own_arc{nullptr};  //!< The arc delay belong to.
};

/**
 * @brief The class of data path data.
 *
 */
class StaPathDelayData : public StaData {
 public:
  StaPathDelayData(AnalysisMode delay_type, TransType trans_type,
                   int64_t arrive_time, StaClockData* launch_clock_data,
                   StaVertex* own_vertex);
  ~StaPathDelayData() override;
  StaPathDelayData(const StaPathDelayData& orig);
  StaPathDelayData& operator=(const StaPathDelayData& rhs);
  StaPathDelayData(StaPathDelayData&& other) noexcept;
  StaPathDelayData& operator=(StaPathDelayData&& rhs) noexcept;

  StaPathDelayData* copy() override { return new StaPathDelayData(*this); }

  unsigned isPathDelayData() const override { return 1; }

  int64_t get_arrive_time() const override {
    return _calibrated_derate ? _arrive_time * _calibrated_derate.value()
                              : _arrive_time;
  }
  void set_arrive_time(int64_t arrive_time) override {
    _arrive_time = arrive_time;
  }
  void incrArriveTime(int delta) override { _arrive_time += delta; }

  std::optional<int> get_req_time() const override { return _req_time; }
  void set_req_time(int req_time) override { _req_time = req_time; }

  StaClockData* get_launch_clock_data() const { return _launch_clock_data; }

  unsigned compareSignature(const StaData* data) const override;
  int64_t getCompareValue() const override { return _arrive_time; }

  void set_calibrated_derate(float calibrated_derate) {
    _calibrated_derate = calibrated_derate;
  }
  auto get_calibrated_derate() const { return _calibrated_derate; }

 private:
  int64_t _arrive_time;  //!< The arrive time value, unit is fs.
  std::optional<float>
      _calibrated_derate;  //!< The AI predicted calibrated derate
  std::optional<int> _req_time =
      std::nullopt;  //!< The req time value, unit is fs.
  StaClockData* _launch_clock_data;
};

/**
 * @brief The class of clock path data.
 *
 */
class StaClockData : public StaData {
 public:
  StaClockData(AnalysisMode delay_type, TransType trans_type, int arrive_time,
               StaVertex* own_vertex, StaClock* prop_clock);
  ~StaClockData() override;
  StaClockData(const StaClockData& orig);
  StaClockData& operator=(const StaClockData& rhs);
  StaClockData(StaClockData&& other) noexcept;
  StaClockData& operator=(StaClockData&& rhs) noexcept;

  StaClockData* copy() override { return new StaClockData(*this); }

  int64_t get_arrive_time() const override { return _arrive_time; }
  void set_arrive_time(int64_t arrive_time) override {
    _arrive_time = arrive_time;
  }
  void incrArriveTime(int delta) override { _arrive_time += delta; }

  StaClock* get_prop_clock() const { return _prop_clock; }
  unsigned isClockData() const override { return 1; }
  virtual unsigned compareSignature(const StaData* data) const;

  void set_clock_wave_type(TransType clock_wave_type) {
    _clock_wave_type = clock_wave_type;
  }
  TransType get_clock_wave_type() { return _clock_wave_type; }
  int64_t getCompareValue() const override { return _arrive_time; }

 private:
  int64_t _arrive_time;             //!< The arrive time value, unit is fs.
  StaClock* _prop_clock = nullptr;  //!< The propagated clock.
  TransType _clock_wave_type = TransType::kRise;  //!< The launch clock wave.
};

/**
 * @brief The data bucket for store sta data for every vertex.
 * The bucket can pop out the beyond limit data.
 */
class StaDataBucket {
 public:
  friend StaDataBucketIterator;

  explicit StaDataBucket(unsigned n_worst = 1);
  ~StaDataBucket() = default;
  StaDataBucket(StaDataBucket&& other) noexcept;
  StaDataBucket& operator=(StaDataBucket&& rhs) noexcept;

  unsigned bucket_size() const { return _count; }
  bool empty() { return _data_list.empty(); }
  void addData(StaData* data, int track_stack_deep);
  StaData* frontData() {
    return !_data_list.empty() ? _data_list.front().get() : nullptr;
  }

  void freeData() {
    _data_list.clear();
    _count = 0;
    _next.reset(nullptr);
  }

  unsigned isFreeData() {
    return (_data_list.empty() && _count == 0 && _next == nullptr);
  }

  // insert data to the data list directly.
  void insertData(StaData* data) {
    _data_list.emplace_front(data);
    _count++;
  }

  void set_next(std::unique_ptr<StaDataBucket>&& next_bucket) {
    _next = std::move(next_bucket);
  }
  StaDataBucket* get_next() { return _next.get(); }

 private:
  std::forward_list<std::unique_ptr<StaData>>
      _data_list;  //!< The sta data list store the data
                   //!< that has the same signature.

  unsigned _n_worst;    //!< Store the top n worst data.
  unsigned _count = 0;  //!<  For the fwd list do not provide cout.

  std::unique_ptr<StaDataBucket>
      _next;  //!< The next data bucket which has different signature.

  FORBIDDEN_COPY(StaDataBucket);
};

/**
 * @brief The data bucket iterator class.
 *
 */
class StaDataBucketIterator {
 public:
  explicit StaDataBucketIterator(StaDataBucket& data_bucket);
  ~StaDataBucketIterator() = default;

  bool hasNext();
  std::unique_ptr<StaData>& next();

 private:
  StaDataBucket* _data_bucket;
  std::forward_list<std::unique_ptr<StaData>>::iterator
      _iter;  //!< The related data bucket.
};

};  // namespace ista
