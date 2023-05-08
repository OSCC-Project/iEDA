#pragma once

#include "Logger.hpp"
#include "RTU.hpp"

namespace irt {

template <typename T>
class GridMap
{
 public:
  GridMap() = default;
  GridMap(irt_int x_size, irt_int y_size) { init(x_size, y_size); }
  GridMap(irt_int x_size, irt_int y_size, T value) { init(x_size, y_size, value); }
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
    Proxy(irt_int y_size, U* data_array) : _y_size(y_size), _data_array(data_array) {}

    U& operator[](const size_t i) { return operator[](static_cast<irt_int>(i)); }

    U& operator[](const irt_int i) { return const_cast<U&>(static_cast<const Proxy&>(*this)[i]); }

    const U& operator[](const irt_int i) const
    {
      if (i < 0 || _y_size <= i) {
        LOG_INST.error(Loc::current(), "The grid map index y ", i, " is out of bounds!");
      }
      return _data_array[i];
    }

   private:
    irt_int _y_size = 0;
    U* _data_array = nullptr;
  };

  Proxy<T> operator[](const size_t i) const { return operator[](static_cast<irt_int>(i)); }

  Proxy<T> operator[](const irt_int i) const
  {
    if (i < 0 && _x_size <= i) {
      LOG_INST.error(Loc::current(), "The grid map index x ", i, " is out of bounds!");
    }
    return Proxy<T>(_y_size, _data_map[i]);
  }
  // getter
  irt_int get_x_size() const { return _x_size; }
  irt_int get_y_size() const { return _y_size; }
  // function
  inline void init(size_t x_size, size_t y_size);
  inline void init(irt_int x_size, irt_int y_size);
  inline void init(size_t x_size, size_t y_size, T value);
  inline void init(irt_int x_size, irt_int y_size, T value);
  inline void free();
  inline bool empty() const;
  inline bool isInside(irt_int x, irt_int y) const;

 private:
  irt_int _x_size = 0;
  irt_int _y_size = 0;
  T** _data_map = nullptr;
  // function
  inline void copy(const GridMap& other);
  inline void move(GridMap&& other);
  inline void initDataMap();
  inline void copyDataMap(T** other_data_map);
  inline void freeDataMap();
  inline void deassignDataMap(T value);
};

// public

template <typename T>
inline void GridMap<T>::init(size_t x_size, size_t y_size)
{
  init(static_cast<irt_int>(x_size), static_cast<irt_int>(y_size));
}

template <typename T>
inline void GridMap<T>::init(irt_int x_size, irt_int y_size)
{
  if constexpr (std::is_same<T, irt_int>::value || std::is_same<T, double>::value) {
    init(x_size, y_size, 0);
  } else {
    init(x_size, y_size, T());
  }
}

template <typename T>
inline void GridMap<T>::init(size_t x_size, size_t y_size, T value)
{
  init(static_cast<irt_int>(x_size), static_cast<irt_int>(y_size), value);
}

template <typename T>
inline void GridMap<T>::init(irt_int x_size, irt_int y_size, T value)
{
  freeDataMap();
  _x_size = x_size;
  _y_size = y_size;
  initDataMap();
  deassignDataMap(value);
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
inline bool GridMap<T>::isInside(irt_int x, irt_int y) const
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
    LOG_INST.error(Loc::current(), "The map size setting error!");
  }
  if (_x_size == 0 || _y_size == 0) {
    _data_map = nullptr;
    return;
  }
  _data_map = new T*[_x_size];
  _data_map[0] = new T[_x_size * _y_size];
  for (irt_int i = 1; i < _x_size; i++) {
    _data_map[i] = _data_map[i - 1] + _y_size;
  }
}

template <typename T>
inline void GridMap<T>::copyDataMap(T** other_data_map)
{
  for (irt_int i = 0; i < _x_size; i++) {
    for (irt_int j = 0; j < _y_size; j++) {
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
inline void GridMap<T>::deassignDataMap(T value)
{
  for (irt_int i = 0; i < _x_size; i++) {
    for (irt_int j = 0; j < _y_size; j++) {
      _data_map[i][j] = value;
    }
  }
}

}  // namespace irt
