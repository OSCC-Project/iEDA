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

#ifdef USE_CPP_STD
#include <vector>
#endif

namespace ieda {

template <typename T, size_t N = 64, typename A = std::allocator<T>>
#ifndef USE_CPP_STD
class Vector : public absl::InlinedVector<T, N, A>
#else
class Vector : public std::vector<T, A>
#endif
{
 public:
#ifndef USE_CPP_STD
  using Base = typename Vector::InlinedVector;
#else
  using Base = typename Vector::vector;
#endif

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
};

}  // namespace ieda
