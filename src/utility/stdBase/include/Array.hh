/**
 * @file Array.h
 * @author Lh
 * @brief The Array container for the eda project.
 * @version 0.1
 * @date 2020-10-20
 */
#pragma once
#include <algorithm>

#include "absl/container/fixed_array.h"

namespace ieda {
template <typename T, size_t N>
class Array : public absl::FixedArray<T, N>
{
 public:
  using Base = typename Array::FixedArray;
  using pointer = typename Base::pointer;
  using const_pointer = typename Base::const_pointer;
  using reference = typename Base::reference;
  using const_reference = typename Base::const_reference;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;
  using const_reverse_iterator = typename Base::const_reverse_iterator;

  using Base::at;
  using Base::back;
  using Base::Base;
  using Base::begin;
  using Base::cbegin;
  using Base::cend;
  using Base::crbegin;
  using Base::crend;
  using Base::data;
  using Base::empty;
  using Base::end;
  using Base::fill;
  using Base::front;
  using Base::max_size;
  using Base::rbegin;
  using Base::rend;
  using Base::size;

  void swap(Array& arr) { std::swap_ranges(begin(), end(), arr.begin()); }

  reference operator[](size_t i) { return data()[i]; }

  friend bool operator==(const Array& lhs, const Array& rhs) { return absl::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end()); }

  friend bool operator!=(const Array& lhs, const Array& rhs) { return !(lhs == rhs); }

  friend bool operator<(const Array& lhs, const Array& rhs)
  {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
  }

  friend bool operator>(const Array& lhs, const Array& rhs) { return rhs < lhs; }

  friend bool operator<=(const Array& lhs, const Array& rhs) { return !(rhs < lhs); }

  friend bool operator>=(const Array& lhs, const Array& rhs) { return !(lhs < rhs); }
};

}  // namespace ieda