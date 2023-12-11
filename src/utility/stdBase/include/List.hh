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
 * @file List.h
 * @author Lh
 * @brief
 * @version 0.1
 * @date 2020-10-20
 */
#pragma once
#include <algorithm>
#include <list>

#include "absl/algorithm/algorithm.h"
namespace ieda {
template <typename T>
class List : public std::list<T>
{
 public:
  using Base = typename List::list;
  using Base::Base;
  using reference = typename Base::reference;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;
  using pointer = typename Base::pointer;
  using const_pointer = typename Base::const_pointer;
  using reverse_iterator = typename Base::reverse_iterator;
  using const_reverse_iterator = typename Base::const_reverse_iterator;
  using Base::assign;
  using Base::back;   // Returns a reference to the last item in the list.
  using Base::begin;  // Returns an STL-style iterator pointing to the first
                      // item in the list
  using Base::cbegin;
  using Base::cend;
  using Base::clear;  // Removes all items from the list
  using Base::crbegin;
  using Base::crend;
  using Base::emplace;
  using Base::empty;   // returns true if the list is empty
  using Base::end;     // Returns an STL-style iterator pointing to the imaginary
                       // item after the last item in the list
  using Base::erase;   // Removes the item associated with the iterator pos from
                       // the list, and returns an iterator to the next item in
                       // the list
  using Base::front;   // Returns a reference to the first item in the list
  using Base::insert;  // Inserts value at index position i in the list
  using Base::max_size;
  using Base::merge;  // Merges  elements of the two lists in ascending order
  using Base::pop_back;
  using Base::pop_front;
  using Base::push_back;
  using Base::push_front;
  using Base::rbegin;
  using Base::remove;
  using Base::remove_if;
  using Base::rend;
  using Base::resize;
  using Base::reverse;
  using Base::size;
  using Base::sort;
  using Base::splice;  // Splices the other list at the pos(iterator pointing)
                       // position
  using Base::swap;    // Swaps list other with this list.
  using Base::unique;  // Removes adjacent duplicate elements
 
};
}  // namespace ieda