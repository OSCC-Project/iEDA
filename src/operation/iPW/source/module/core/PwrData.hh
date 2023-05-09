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
 * @file PwrData.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief the class of power data.
 * @version 0.1
 * @date 2023-02-09
 */
#pragma once

#include <forward_list>
#include <memory>
#include <variant>

#include "PwrClock.hh"
#include "include/PwrType.hh"
#include "sta/StaClock.hh"

namespace ipower {

class PwrVertex;
class PwrDataBucketIterator;

/**
 * @brief The base class of power data.
 *
 */
class PwrData {
 public:
  PwrData(PwrDataSource data_source, PwrVertex* own_vertex)
      : _data_source(data_source), _own_vertex(own_vertex){};
  virtual ~PwrData() = default;

  [[nodiscard]] virtual unsigned isSPData() const { return 0; }
  [[nodiscard]] virtual unsigned isToggleData() const { return 0; }
  [[nodiscard]] virtual double getCompareValue() const { return 0; }

  [[nodiscard]] PwrDataSource get_data_source() const { return _data_source; }

  // virtual unsigned compareSignature(const PwrSPData* data) const {}
  // virtual unsigned compareSignature(const PwrToggleData* data) const;
  virtual unsigned compareSignature(const PwrData* data) const {
    if (!data || _data_source != data->get_data_source()) {
      return 0;
    }
    return 1;
  }

  void set_fwd(PwrData* fwd) { _fwd = fwd; }
  [[nodiscard]] PwrData* get_fwd() const { return _fwd; }

  void add_bwd(PwrData* bwd) { _bwds.emplace_back(bwd); }
  [[nodiscard]] auto& get_bwds() const { return _bwds; }

 protected:
  PwrDataSource _data_source;   //!< The power data source. (include default,
                                //!< annotated from vcd, propagation)
  PwrVertex* _own_vertex;       //!< The vertex which the data belong to.
  PwrData* _fwd = nullptr;      //!< The fwd propagation data.
  std::vector<PwrData*> _bwds;  //!< The bwd propagation data.
};

/**
 * @brief The class of power SP data.
 *
 */
class PwrSPData : public PwrData {
 public:
  PwrSPData(PwrDataSource data_source, PwrVertex* own_vertex, double sp)
      : PwrData(data_source, own_vertex), _sp(sp) {}
  ~PwrSPData() override = default;
  [[nodiscard]] unsigned isSPData() const override { return 1; }
  [[nodiscard]] double getCompareValue() const override { return _sp; }

  [[nodiscard]] double get_sp() const { return _sp; }

  unsigned compareSignature(const PwrSPData* data) const {
    auto is_same = 1;
    if (!data || !data->isSPData()) {
      is_same = 0;
    } else if (data->get_data_source() != _data_source) {
      is_same = 0;
    }
    return is_same;
  }

 private:
  double _sp;
};

/**
 * @brief The class of power Toggle data.
 *
 */
class PwrToggleData : public PwrData {
 public:
  PwrToggleData(PwrDataSource data_source, PwrVertex* own_vertex, double toggle)
      : PwrData(data_source, own_vertex), _toggle(toggle) {}
  PwrToggleData(PwrDataSource data_source, PwrVertex* own_vertex)
      : PwrToggleData(data_source, own_vertex, 0.0) {}
  ~PwrToggleData() override = default;

  [[nodiscard]] unsigned isToggleData() const override { return 1; }
  [[nodiscard]] double getCompareValue() const override { return _toggle; }
  [[nodiscard]] double get_toggle() const { return _toggle; }

  void set_clock_domain(PwrClock* the_fastest_clock) {
    _clock_domain = the_fastest_clock;
  }
  void set_clock_domain(StaClock* the_clock) { _clock_domain = the_clock; }

  std::string getRelativeClockName();
  double getRelativeClockPeriodNs();

  [[nodiscard]] double getToggleRateRelativeToClock();
  void setToggle(double toggle_relative_clock);

  unsigned compareSignature(const PwrToggleData* data) const {
    auto is_same = 1;
    if (!data || !data->isToggleData()) {
      is_same = 0;
    } else if (data->get_data_source() != _data_source) {
      is_same = 0;
    }
    return is_same;
  }

 private:
  double _toggle;  //!< The unit is ns.
  std::variant<PwrClock*, StaClock*>
      _clock_domain;  //!< The toggle data relative clock, whether power fastest
                      //!< clock or sta clock.
};

/**
 * @brief The data bucket for store the same type of power data。
 * The bucket can pop out the beyond limit data.
 */
class PwrDataBucket {
 public:
  friend PwrDataBucketIterator;
  explicit PwrDataBucket(unsigned n_worst = 3) : _n_worst(n_worst) {}
  ~PwrDataBucket() = default;

  [[nodiscard]] unsigned bucket_size() const { return _count; }
  bool empty() { return _data_list.empty(); }

  void insertData(PwrData* data) {
    _data_list.emplace_front(data);
    _count++;
  }
  void addData(PwrData* data, int track_stack_deep);
  PwrData* frontData() {
    return !_data_list.empty() ? _data_list.front().get() : nullptr;
  }
  PwrData* frontData(PwrDataSource data_source);

  void set_next(std::unique_ptr<PwrDataBucket>&& next_bucket) {
    _next = std::move(next_bucket);
  }
  PwrDataBucket* get_next() { return _next.get(); }

 private:
  std::forward_list<std::unique_ptr<PwrData>>
      _data_list;       //!< The power data list store the data
                        //!< that has the same signature.
  unsigned _n_worst;    //!< Store the top n worst data.
  unsigned _count = 0;  //!<  For the fwd list do not provide cout.
  std::unique_ptr<PwrDataBucket>
      _next;  //!< The next data bucket which has different signature.
};

/**
 * @brief The data bucket iterator class.
 *
 */
class PwrDataBucketIterator {
 public:
  explicit PwrDataBucketIterator(PwrDataBucket& data_bucket)
      : _data_bucket(&data_bucket), _iter(data_bucket._data_list.begin()) {}
  ~PwrDataBucketIterator() = default;

  bool hasNext();
  std::unique_ptr<PwrData>& next() { return *_iter++; };

 private:
  PwrDataBucket* _data_bucket;
  std::forward_list<std::unique_ptr<PwrData>>::iterator
      _iter;  //!< The related data bucket.
};

}  // namespace ipower