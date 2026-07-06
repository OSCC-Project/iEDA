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
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
namespace itf
{

// look-up table
// query value by row column
template<typename T1, typename T2, typename T3>
class itf2DLUT {
 public: 
  // constructor
  itf2DLUT() = default;
  explicit itf2DLUT(const char* row_name, const char* col_name, const char* value_name)
  : _rows(),
    _cols(),
    _values(),
    _row_name(row_name),
    _col_name(col_name),
    _value_name(value_name)
  { }
  itf2DLUT(const itf2DLUT& other)
  {
    *this = other;
  }

  // getter
  const std::vector<T1>& get_rows()   const   { return _rows;       }
  const std::vector<T2>& get_cols()   const   { return _cols;       }
  const std::vector<T3>& get_values() const   { return _values;     }
  std::string get_row_name()          const   { return _row_name;   }
  std::string get_col_name()          const   { return _col_name;   }
  std::string get_value_name()        const   { return _value_name; }

  // setter
  void add_row_data(T1 e) { _rows.push_back(e);   }
  void add_col_data(T2 e) { _cols.push_back(e);   }
  void add_value(T3 e)    { _values.push_back(e); }
  void set_row_name(const char* s)     { _row_name = s;    }
  void set_col_name(const char* s)     { _col_name = s;    }
  void set_value_name(const char* s)   { _value_name = s;  }

  // operator
  itf2DLUT& operator=(const itf2DLUT& rhs) {
    if (this == &rhs) return *this;

    _rows = rhs._rows;
    _cols = rhs._cols;
    _values = rhs._values;
    _row_name = rhs._row_name;
    _col_name = rhs._col_name;
    _value_name = rhs._value_name;

    return *this;
  }

  bool operator==(const itf2DLUT& rhs) const {
    if (this == &rhs) return true;

    return _row_name == rhs._row_name
      && _col_name == rhs._col_name
      && _value_name == rhs._value_name
      && _rows == rhs._rows
      && _cols == rhs._cols
      && _values == rhs._values
    ;
  }

  // function
  
  void clear() {
    _rows.clear();
    _cols.clear();
    _values.clear();
    _row_name.clear();
    _col_name.clear();
    _value_name.clear();
  }

  // @param r_idx row index
  // @param c_idx col index
  std::optional<T3> query(int r_idx, int c_idx) const {
    int v_idx = r_idx * _cols.size() + c_idx;
    if ((0 <= r_idx) && (r_idx < (int)_rows.size()  )
     && (0 <= c_idx) && (c_idx < (int)_cols.size()  ) 
     && (0 <= v_idx) && (v_idx < (int)_values.size()) )
    {
      return _values.at(v_idx);
    } else {
      return std::nullopt;
    }
  }

  // bilinear interpolation for points inside of the range of data points,
  // keep boundary value for points outside of the range.
  // In other words, no extrapolate beyond the table.
  // @param r data in _raws 
  // @param c data in _cols
  std::optional<T3> query_interpolation(const T1& r, const T2& c) const {
    if (_rows.empty() || _cols.empty()) return std::nullopt;
    
    const auto [r_low, r_high] = bounding_indices_(_rows, r);
    const auto [c_low, c_high] = bounding_indices_(_cols, c);

    auto v_rl_cl = query(r_low, c_low);
    if (!v_rl_cl) return std::nullopt;

    if (r_low == r_high && c_low == c_high) {
      return *v_rl_cl;
    }

    if (r_low == r_high) {
      auto v_rl_ch = query(r_low, c_high);
      if (!v_rl_ch) return std::nullopt;

      const double col_ratio = interpolation_ratio_(_cols, c_low, c_high, c);
      return interpolate_value_(*v_rl_cl, *v_rl_ch, col_ratio);
    }

    if (c_low == c_high) {
      auto v_rh_cl = query(r_high, c_low);
      if (!v_rh_cl) return std::nullopt;

      const double row_ratio = interpolation_ratio_(_rows, r_low, r_high, r);
      return interpolate_value_(*v_rl_cl, *v_rh_cl, row_ratio);
    }

    auto v_rl_ch = query(r_low, c_high);
    auto v_rh_cl = query(r_high, c_low);
    auto v_rh_ch = query(r_high, c_high);

    if (v_rl_ch && v_rh_cl && v_rh_ch) {
      const double col_ratio = interpolation_ratio_(_cols, c_low, c_high, c);
      auto v_rl_cmid = interpolate_value_(*v_rl_cl, *v_rl_ch, col_ratio);
      auto v_rh_cmid = interpolate_value_(*v_rh_cl, *v_rh_ch, col_ratio);

      const double row_ratio = interpolation_ratio_(_rows, r_low, r_high, r);
      return interpolate_value_(v_rl_cmid, v_rh_cmid, row_ratio);
    }
    
    return std::nullopt;
  }

  // @param list_name data_list name. match data_list in an order of rows, cols and values.
  template<typename E>
  void add_data(const char* list_name, E e) {
    if (_row_name.compare(list_name) == 0) {
      add_row_data(T1(e));
    } else if (_col_name.compare(list_name) == 0) {
      add_col_data(T2(e));
    } else if (_value_name.compare(list_name) == 0) {
      add_value(T3(e));
    } else {
      std::cout << "fail to find data list named " << list_name << std::endl;
    }
  }

  template<typename E>
  void set_data_list(const char* list_name, const std::vector<E>& src) {
    if (_row_name.compare(list_name) == 0) {
      set_list<T1, E>(&_rows, src);
    } else if (_col_name.compare(list_name) == 0) {
      set_list<T2, E>(&_cols, src);
    } else if (_value_name.compare(list_name) == 0) {
      set_list<T3, E>(&_values, src);
    } else {
      std::cout << "fail to find data list named " << list_name << std::endl;
    }
  }

  // @param r rows_name
  // @param c cols_name
  // @param v values_name
  void set_names(const char* r, const char* c, const char* v) {
    set_row_name(r);
    set_col_name(c);
    set_value_name(v);
  }

 protected:
  // members
  std::vector<T1> _rows;
  std::vector<T2> _cols;
  std::vector<T3> _values;
  std::string _row_name;
  std::string _col_name;
  std::string _value_name;
 
 private:
  // function
  template<typename T>
  struct is_pair_ : std::false_type {};

  template<typename TFirst, typename TSecond>
  struct is_pair_<std::pair<TFirst, TSecond>> : std::true_type {};

  template<typename TValue>
  static TValue interpolate_value_(const TValue& low, const TValue& high, double ratio) {
    if constexpr (is_pair_<TValue>::value) {
      return TValue{
        interpolate_value_(low.first, high.first, ratio),
        interpolate_value_(low.second, high.second, ratio)
      };
    } else {
      return static_cast<TValue>(
          std::lerp(static_cast<double>(low), static_cast<double>(high), ratio));
    }
  }

  template<typename TValue>
  static std::pair<size_t, size_t> bounding_indices_(
      const std::vector<TValue>& values,
      const TValue& value) {
    if (value <= values.front()) {
      return {0, 0};
    }
    if (value >= values.back()) {
      const size_t last = values.size() - 1;
      return {last, last};
    }

    auto high_it = std::lower_bound(values.begin(), values.end(), value);
    const size_t high = std::distance(values.begin(), high_it);
    if (*high_it == value) {
      return {high, high};
    }
    return {high - 1, high};
  }

  template<typename TValue>
  static double interpolation_ratio_(
      const std::vector<TValue>& values,
      size_t low,
      size_t high,
      const TValue& value) {
    if (low == high) {
      return 0.0;
    }
    return static_cast<double>(value - values[low])
         / static_cast<double>(values[high] - values[low]);
  }

  template<typename E1, typename E2>
  void set_list(void* dst, const std::vector<E2>& src) {
    if (typeid(E1) == typeid(E2)) {
      std::vector<E2>* dst_ptr = (std::vector<E2>*)dst;
      dst_ptr->clear();
      for (auto& e : src) {
        dst_ptr->push_back(e);
      }
    } else {
      std::cout << "data type mismatch" << std::endl;
    }
  }
};

// 2D look-up table with title
template<typename T1, typename T2, typename T3>
class itfTitleLut : public itf2DLUT<T1, T2, T3> {
 public:
  // constructor
  itfTitleLut() : itf2DLUT<T1, T2, T3>(), _title() { }

  itfTitleLut(const char* title, const char* row_name, const char* col_name, const char* value_name)
  : itf2DLUT<T1, T2, T3>(row_name, col_name, value_name),
    _title(title)
  { }

  itfTitleLut(const itfTitleLut& other)
  : itf2DLUT<T1, T2, T3>(other),
    _title(other._title)
  { }

  itfTitleLut(const char* title, const itf2DLUT<T1, T2, T3> lut)
  : itf2DLUT<T1, T2, T3>(lut),
    _title(title)
  { }

  // getter
  std::string get_title() const { return _title; }

  // setter
  void set_title(const char* title) { _title = title; }
  void set_lut(const itf2DLUT<T1, T2, T3>& lut) {
    static_cast<itf2DLUT<T1, T2, T3>&>(*this) = lut;
  }

  // operator
  itfTitleLut& operator=(const itfTitleLut& rhs) {
    if (this == &rhs) return *this;

    static_cast<itf2DLUT<T1, T2, T3>&>(*this) = rhs;
    _title = rhs._title;

    return *this;
  }

  bool operator==(const itfTitleLut& rhs) const {
    if (this == &rhs) return true;

    return _title == rhs._title
      && static_cast<const itf2DLUT<T1, T2, T3>&>(*this) == rhs
    ;
  }

  // function
  void clear() {
    itf2DLUT<T1, T2, T3>::clear();
    _title.clear();
  }

 protected:
  std::string _title;
};

} // namespace itf
