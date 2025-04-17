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

#include "DRCHeader.hpp"
#include "Logger.hpp"

namespace idrc {

template <typename T>
class GridMap
{
 public:
  GridMap() = default;
  GridMap(int32_t x_size, int32_t y_size) { init(x_size, y_size); }
  GridMap(int32_t x_size, int32_t y_size, T value) { init(x_size, y_size, value); }
  GridMap(const GridMap& other) { copy(other); }
  GridMap(GridMap&& other) { move(std::forward<GridMap>(other)); }
  ~GridMap() { free(); }
  GridMap& operator=(const GridMap& other)
  {
    copy(other);
    return (*this);
  }
  GridMap& operator=(GridMap&& other)
  {
    move(std::forward<GridMap>(other));
    return (*this);
  }

  template <typename U>
  class Proxy
  {
   public:
    Proxy(int32_t y_size, U* data_array) : _y_size(y_size), _data_array(data_array) {}

    U& operator[](const size_t i) { return operator[](static_cast<int32_t>(i)); }

    U& operator[](const int32_t i) { return const_cast<U&>(static_cast<const Proxy&>(*this)[i]); }

    const U& operator[](const int32_t i) const
    {
      if (i < 0 || _y_size <= i) {
        DRCLOG.error(Loc::current(), "The grid map index y ", i, " is out of bounds!");
      }
      return _data_array[i];
    }

    U& front() { return operator[](0); }

    U& back() { return operator[](_y_size - 1); }

   private:
    int32_t _y_size = 0;
    U* _data_array = nullptr;
  };

  Proxy<T> operator[](const size_t i) const { return operator[](static_cast<int32_t>(i)); }

  Proxy<T> operator[](const int32_t i) const
  {
    if (i < 0 || _x_size <= i) {
      DRCLOG.error(Loc::current(), "The grid map index x ", i, " is out of bounds!");
    }
    return Proxy<T>(_y_size, _data_map[i]);
  }

  Proxy<T> front() { return operator[](0); }

  Proxy<T> back() { return operator[](_x_size - 1); }

  // getter
  int32_t get_x_size() const { return _x_size; }
  int32_t get_y_size() const { return _y_size; }
  // function
  inline void init(size_t x_size, size_t y_size);
  inline void init(int32_t x_size, int32_t y_size);
  inline void init(size_t x_size, size_t y_size, T value);
  inline void init(int32_t x_size, int32_t y_size, T value);
  inline void free();
  inline bool empty() const;
  inline bool isInside(int32_t x, int32_t y) const;

 private:
  int32_t _x_size = 0;
  int32_t _y_size = 0;
  T** _data_map = nullptr;
  // function
  inline void copy(const GridMap& other);
  inline void move(GridMap&& other);
  inline void initDataMap();
  inline void copyDataMap(T** other_data_map);
  inline void freeDataMap();
  inline void assignDataMap(T value);
};

// public

template <typename T>
inline void GridMap<T>::init(size_t x_size, size_t y_size)
{
  init(static_cast<int32_t>(x_size), static_cast<int32_t>(y_size));
}

template <typename T>
inline void GridMap<T>::init(int32_t x_size, int32_t y_size)
{
  if constexpr (std::is_same<T, int32_t>::value || std::is_same<T, double>::value) {
    init(x_size, y_size, 0);
  } else {
    init(x_size, y_size, T());
  }
}

template <typename T>
inline void GridMap<T>::init(size_t x_size, size_t y_size, T value)
{
  init(static_cast<int32_t>(x_size), static_cast<int32_t>(y_size), value);
}

template <typename T>
inline void GridMap<T>::init(int32_t x_size, int32_t y_size, T value)
{
  freeDataMap();
  _x_size = x_size;
  _y_size = y_size;
  initDataMap();
  assignDataMap(value);
}

template <typename T>
void GridMap<T>::free()
{
  _x_size = 0;
  _y_size = 0;
  freeDataMap();
}

template <typename T>
inline bool GridMap<T>::empty() const
{
  return _x_size == 0 || _y_size == 0;
}

template <typename T>
inline bool GridMap<T>::isInside(int32_t x, int32_t y) const
{
  return 0 <= x && x < _x_size && 0 <= y && y < _y_size;
}

// private

template <typename T>
inline void GridMap<T>::copy(const GridMap& other)
{
  freeDataMap();
  _x_size = other._x_size;
  _y_size = other._y_size;
  initDataMap();
  copyDataMap(other._data_map);
}

template <typename T>
inline void GridMap<T>::move(GridMap&& other)
{
  freeDataMap();
  _x_size = std::move(other._x_size);
  _y_size = std::move(other._y_size);
  _data_map = other._data_map;
  other._data_map = nullptr;
}

template <typename T>
inline void GridMap<T>::initDataMap()
{
  if (_x_size < 0 || _y_size < 0) {
    DRCLOG.error(Loc::current(), "The map size setting error!");
  }
  if (_x_size == 0 || _y_size == 0) {
    _data_map = nullptr;
    return;
  }
  _data_map = new T*[_x_size];
  _data_map[0] = new T[_x_size * _y_size];
  for (int32_t i = 1; i < _x_size; i++) {
    _data_map[i] = _data_map[i - 1] + _y_size;
  }
}

template <typename T>
inline void GridMap<T>::copyDataMap(T** other_data_map)
{
  for (int32_t i = 0; i < _x_size; i++) {
    for (int32_t j = 0; j < _y_size; j++) {
      _data_map[i][j] = other_data_map[i][j];
    }
  }
}

template <typename T>
inline void GridMap<T>::freeDataMap()
{
  if (_data_map) {
    delete[] _data_map[0];
  }
  delete[] _data_map;
  _data_map = nullptr;
}

template <typename T>
inline void GridMap<T>::assignDataMap(T value)
{
  for (int32_t i = 0; i < _x_size; i++) {
    for (int32_t j = 0; j < _y_size; j++) {
      _data_map[i][j] = value;
    }
  }
}

}  // namespace idrc
