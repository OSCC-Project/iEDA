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
 * @file Vector.h
 * @author Lh
 * @brief The Vector container for the eda project.
 * @version 0.1
 * @date 2020-10-20
 */
#pragma once
#include <algorithm>
#include <memory>

#include "absl/container/inlined_vector.h"

namespace ieda {
template <typename T, size_t N = 64, typename A = std::allocator<T>>
class Vector : public absl::InlinedVector<T, N, A>
{
 public:
  using Base = typename Vector::InlinedVector;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;
  using pointer = typename Base::pointer;
  using const_pointer = typename Base::const_pointer;
  using reference = typename Base::reference;
  using const_reference = typename Base::const_reference;
  using reverse_iterator = typename Base::reverse_iterator;
  using const_reverse_iterator = typename Base::const_reverse_iterator;
  using size_type = typename Base::size_type;

  using Base::assign;
  using Base::at;
  using Base::back;
  using Base::Base;
  using Base::begin;
  using Base::capacity;
  using Base::cbegin;
  using Base::cend;
  using Base::clear;
  using Base::crbegin;
  using Base::crend;
  using Base::data;
  using Base::emplace;
  using Base::emplace_back;
  using Base::empty;
  using Base::end;
  using Base::erase;
  using Base::front;
  using Base::insert;
  using Base::max_size;
  using Base::pop_back;
  using Base::push_back;
  using Base::rbegin;
  using Base::rend;
  using Base::reserve;
  using Base::resize;         // Resizes the vector to contain `n` elements.
  using Base::shrink_to_fit;  // Reduces memory usage by freeing unused memory
  using Base::size;
  using Base::swap;  // Swaps the contents of the Vector with the other
                     // Vector,for exampla:vector1.swap(vector2)

  /**
   * @brief Returns true if the vector contains an occurrence of value;
   *  otherwise returns false.
   *
   * @param value
   * @return true
   * @return false
   */
  bool contains(const T& value)
  {
    const_iterator b = this->begin();
    const_iterator e = this->end();
    return std::find(b, e, value) != e;
  }
  /**
   * @brief Returns the number of occurrences of value in the vector.
   *
   * @param value
   * @return int
   */
  int count(const T& value)
  {
    const_iterator b = this->begin();
    const_iterator e = this->end();
    return static_cast<int>((std::count(b, e, value)));
  }
  /**
   * @brief Returns the index positon of the begin occurrence of the value valu
   * in the vector,searching forward from index position start.
   *
   * @param value
   * @param start
   * @return int
   */
  int indexOf(const T& value, size_t start)
  {
    size_t size = this->size();
    if (start < 0)
      start = std::max((size_t) 0, start + size);
    if (start < size) {
      const_iterator n = this->begin() + start - 1;
      const_iterator e = this->end();
      while (++n != e) {
        if (*n == value)
          return n - this->begin();
      }
    }
    return -1;
  }
  /**
   * @brief Returns the index position of the end occurrence of value in the
   * vector, searching backward from index position start.
   *
   * @param value
   * @param start
   * @return int
   */
  int endIndexOf(const T& value, int start)
  {
    size_t size = this->size();
    if (start < 0)
      start += size;
    else if (start >= size)
      start = size - 1;
    if (start > 0) {
      const_iterator b = this->begin();
      const_iterator n = this->begin() + start + 1;
      while (n != b) {
        if (*--n == value)
          return n - b;
      }
    }
    return -1;
  }
  /**
   * @brief Returns a vector whose elements are copied from this vector,starting
   * at position pos, len elements are copied.
   *
   * @param pos
   * @param len
   * @return Vector<T>&
   */
  Vector<T, N, A>& mid(size_t pos, size_t len)
  {
    Vector<T, N, A>* midVector = new Vector<T, N, A>();
    const_iterator from = this->begin() + pos;
    const_iterator to = this->begin() + pos + len;
    for (from; from != to; ++from) {
      midVector->push_back(*from);
    }

    return *midVector;
  }

  /**
   * @brief Appends value to the vector.
   *
   * @param value
   * @return Vector<T>&
   */
  Vector<T, N, A>& operator+=(const T& value)
  {
    push_back(value);
    return *this;
  }
  /**
   * @brief Returns a vector that contains all the items in vector val1
   * followed by all the items in the vector val2.
   *
   * @param val1
   * @param val2
   * @return Vector<T>&
   */
  friend Vector<T, N, A>& operator+(const Vector<T, N, A>& val1, const Vector<T, N, A>& val2)
  {
    Vector<T>* ret_val = new Vector<T>;

    for (Vector<T, N, A>::const_iterator it1 = val1.begin(); it1 != val1.end(); it1++) {
      ret_val->push_back(*it1);
    }
    for (Vector<T, N, A>::const_iterator it2 = val2.begin(); it2 != val2.end(); it2++) {
      ret_val->push_back(*it2);
    }
    return *ret_val;
  }
  /**
   * @brief Returns the vector val1 followed by all the items in the vector
   * val2.
   *
   * @param val1
   * @param val2
   */
  // friend void operator+(EfficientList<T>& val1, EfficientList<T>& val2) {
  //   for (EfficientList<T>::iterator it1 = val2.begin(); it1 != val2.end();
  //        it1++) {
  //     val1.push_back(*it1);
  //   }
  // }
  const_reference operator[](size_t i) const { return data()[i]; }
  reference operator[](size_t i) { return data()[i]; }

  friend bool operator<(const Vector<T, N, A>& a, const Vector<T, N, A>& b)
  {
    auto a_data = a.data();
    auto b_data = b.data();
    return std::lexicographical_compare(a_data, a_data + a.size(), b_data, b_data + b.size());
  }

  friend bool operator>(const Vector<T, N, A>& a, const Vector<T, N, A>& b) { return b < a; }
  friend bool operator<=(const Vector<T, N, A>& a, const Vector<T, N, A>& b) { return !(b < a); }
  friend bool operator>=(const Vector<T, N, A>& a, const Vector<T, N, A>& b) { return !(a < b); }
};
}  // namespace ieda
