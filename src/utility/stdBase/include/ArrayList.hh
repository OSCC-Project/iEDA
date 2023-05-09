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
 * @file ArrayList.h
 * @author Lh
 * @brief The ArrayList container for the eda project.
 * @version 0.1
 * @date 2020-10-20
 */
#pragma once

// #include "List.h"
namespace ieda {
template <class T, int init = 10>
class ArrayList
{
  int _size;
  int _capacity;
  T** _params;
  void grow();
  ArrayList(const ArrayList&);

 public:
  ArrayList() : _size(0), _capacity(init), _params(new T*[init]) {}
  virtual ~ArrayList();
  bool add(T* e);
  int contains(T* e);
  bool isEmpty();
  bool remove(T* e);
  int size();
  int capacity();
  void add(int index, T* e);
  T* get(int index);
  T* remove(int index);
  class Iterator;
  friend class Iterator;
  class Iterator
  {
    ArrayList& al;
    int index;

   public:
    Iterator(ArrayList& list) : al(list), index(0) {}
    bool hasNext()
    {
      if (index < al._size) {
        return true;
      }
      return false;
    }
    T* next()
    {
      if (hasNext()) {
        return al._params[index++];
      }
      return 0;
    }
  };
};

template <class T, int init>
ArrayList<T, init>::~ArrayList()
{
  delete[] _params;
}
/**
 * @brief When size and capacity are equal,
 * the capacity will be expanded by 1.5 times
 *
 * @tparam T
 * @tparam init
 */
template <class T, int init>
void ArrayList<T, init>::grow()
{
  if (_size == _capacity) {
    _capacity *= 1.5;
    T** newparams = new T*[_capacity];
    for (int i = 0; i < _size; i++)
      newparams[i] = _params[i];
    delete[] _params;
    _params = newparams;
  }
}
/**
 * @brief Appends value to the ArrayList.
 *
 * @tparam T
 * @tparam init
 * @param e
 * @return true
 * @return false
 */
template <class T, int init>
bool ArrayList<T, init>::add(T* e)
{
  grow();
  _params[_size++] = e;
  return true;
}
/**
 * @brief Returns true if the ArrayList contains an occurrence of the value e;
 *  otherwise returns false.
 *
 * @tparam T
 * @tparam init
 * @param e
 * @return int
 */
template <class T, int init>
int ArrayList<T, init>::contains(T* e)
{
  for (int i = 0; i < _size; i++) {
    if (get(i) == e) {
      return i;
    }
  }
  return -1;
}
/**
 * @brief Returns the capacity of the ArrayList.
 *
 * @tparam T
 * @tparam init
 * @return int
 */
template <class T, int init>
int ArrayList<T, init>::capacity()
{
  return _capacity;
}
/**
 * @brief Returns the size of the ArrayList.
 *
 * @tparam T
 * @tparam init
 * @return int
 */
template <class T, int init>
int ArrayList<T, init>::size()
{
  return _size;
}
template <class T, int init>
bool ArrayList<T, init>::remove(T* e)
{
  int index = contains(e);
  if (index != -1) {
    remove(index);
    return true;
  }
  return false;
}
template <class T, int init>
bool ArrayList<T, init>::isEmpty()
{
  if (_size == 0) {
    return true;
  } else {
    return false;
  }
}
/**
 * @brief Inserts the pointer e at the index positon index.
 *
 * @tparam T
 * @tparam init
 * @param index
 * @param e
 */
template <class T, int init>
void ArrayList<T, init>::add(int index, T* e)
{
  if (index > _size)
    return;
  _size = _size + 1;
  grow();
  for (int i = _size - 1; i > index; i--) {
    _params[i] = _params[i - 1];
  }
  _params[index] = e;
}
/**
 * @brief Returns the item at index position index in the ArrayList.
 *
 * @tparam T
 * @tparam init
 * @param index
 * @return T*
 */
template <class T, int init>
T* ArrayList<T, init>::get(int index)
{
  if (index < 0 || index >= _size) {
    return 0;
  }
  return _params[index];
}
/**
 * @brief Removes the element at index position i,and returns the element.
 *
 * @tparam T
 * @tparam init
 * @param index
 * @return T*
 */
template <class T, int init>
T* ArrayList<T, init>::remove(int index)
{
  if (index < 0 || index >= _size) {
    return 0;
  }
  T* result = get(index);
  if (index < (_size - 1)) {
    for (int i = index; i < _size; i++) {
      _params[i] = _params[i + 1];
    }
  }
  _params[_size - 1] = nullptr;
  _size = _size - 1;
  return result;
}
}  // namespace ieda
