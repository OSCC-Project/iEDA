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
#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>
namespace itf
{

/**
 * @brief One-dimensional lookup table with clamped linear interpolation.
 *
 * The table stores ordered `(key, value)` pairs. Queries inside the table range
 * are linearly interpolated; queries outside the range are clamped to the
 * nearest boundary value.
 */
template<typename TKey, typename TValue>
class itf1DLUT {
 public:
  itf1DLUT() = default;

  explicit itf1DLUT(const char* key_name, const char* value_name)
  : _points(),
    _key_name(key_name ? key_name : ""),
    _value_name(value_name ? value_name : "")
  { }

  const std::vector<std::pair<TKey, TValue>>& get_points() const { return _points; }
  std::string get_key_name() const { return _key_name; }
  std::string get_value_name() const { return _value_name; }
  bool empty() const { return _points.empty(); }
  size_t size() const { return _points.size(); }

  void add_point(TKey key, TValue value) {
    _points.emplace_back(key, value);
    sort_points_();
  }

  void set_points(const std::vector<std::pair<TKey, TValue>>& points) {
    _points = points;
    sort_points_();
  }

  void set_key_name(const char* value) { _key_name = value ? value : ""; }
  void set_value_name(const char* value) { _value_name = value ? value : ""; }

  void set_names(const char* key_name, const char* value_name) {
    set_key_name(key_name);
    set_value_name(value_name);
  }

  std::optional<TValue> query(size_t index) const {
    if (index >= _points.size()) {
      return std::nullopt;
    }
    return _points[index].second;
  }

  std::optional<TValue> query_interpolation(const TKey& key) const {
    if (_points.empty()) {
      return std::nullopt;
    }
    if (_points.size() == 1 || key <= _points.front().first) {
      return _points.front().second;
    }
    if (key >= _points.back().first) {
      return _points.back().second;
    }

    auto it_high = std::lower_bound(
        _points.begin(), _points.end(), key,
        [](const std::pair<TKey, TValue>& point, const TKey& query_key) {
          return point.first < query_key;
        });

    if (it_high == _points.end()) {
      return _points.back().second;
    }
    if (it_high->first == key) {
      return it_high->second;
    }

    auto it_low = std::prev(it_high);
    const auto& [key_low, value_low] = *it_low;
    const auto& [key_high, value_high] = *it_high;
    if (key_high == key_low) {
      return value_high;
    }

    const double ratio = static_cast<double>(key - key_low)
                       / static_cast<double>(key_high - key_low);
    return static_cast<TValue>(
        std::lerp(static_cast<double>(value_low), static_cast<double>(value_high), ratio));
  }

  void clear() {
    _points.clear();
    _key_name.clear();
    _value_name.clear();
  }

  bool operator==(const itf1DLUT& rhs) const {
    if (this == &rhs) {
      return true;
    }
    return _points == rhs._points
        && _key_name == rhs._key_name
        && _value_name == rhs._value_name;
  }

 private:
  void sort_points_() {
    std::sort(_points.begin(), _points.end(),
      [](const std::pair<TKey, TValue>& lhs, const std::pair<TKey, TValue>& rhs) {
        return lhs.first < rhs.first;
      });
  }

  std::vector<std::pair<TKey, TValue>> _points;
  std::string _key_name;
  std::string _value_name;
};

} // namespace itf
