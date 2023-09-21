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
 * @file Set.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The set container for the eda project.
 * @version 0.1
 * @date 2020-10-09
 */

#pragma once

#include <functional>
#include <utility>

#include "absl/container/btree_set.h"

namespace ieda {

/**
 * @brief A set container made up of unique keys based on tree structure.
 *
 * The set is a wrapper of btree set from google abseil containers.The btree set
 * contains ordered containers generally adhering to the STL container API
 * contract, but implemented using B-trees rather than binary trees, generally
 * more efficient.
 */
template <class KEY, class CMP = std::less<KEY>>
class BTreeSet : public absl::btree_set<KEY, CMP>
{
 public:
  using Base = typename BTreeSet::btree_set;
  using key_type = typename Base::key_type;
  using size_type = typename Base::value_type;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;
  using reverse_iterator = typename Base::reverse_iterator;
  using const_reverse_iterator = typename Base::const_reverse_iterator;

  /*constructor*/
  using Base::Base;
  /*destrcutor*/
  ~BTreeSet() = default;
  using Base::operator=;

  /*iterators*/
  using Base::begin;
  using Base::cbegin;
  using Base::cend;
  using Base::crbegin;
  using Base::crend;
  using Base::end;
  using Base::rbegin;
  using Base::rend;

  /*capacity*/
  using Base::empty;
  using Base::max_size;
  using Base::size;

  /*modifiers*/
  using Base::clear;
  using Base::emplace;
  using Base::emplace_hint;
  using Base::erase;
  using Base::extract;
  using Base::insert;
  using Base::merge;
  using Base::swap;

  /*lookup*/
  using Base::contains;
  using Base::count;
  using Base::equal_range;
  using Base::find;
  using Base::lower_bound;
  using Base::upper_bound;

  /*observer*/
  using Base::get_allocator;
  using Base::key_comp;
  using Base::value_comp;
};

/**
 * @brief A ordered set of multiple elements with equivalent keys.
 *
 * The Multiset is a wrapper of btree multiset from google abseil containers.
 * The btree set implemented using B-trees is more efficent than binary tree.
 */
template <class KEY, class CMP = std::less<KEY>>
class Multiset : public absl::btree_multiset<KEY, CMP>
{
 public:
  using Base = typename Multiset::btree_multiset;
  using key_type = typename Base::key_type;
  using size_type = typename Base::value_type;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;
  using reverse_iterator = typename Base::reverse_iterator;
  using const_reverse_iterator = typename Base::const_reverse_iterator;

  /*constructor*/
  using Base::Base;
  /*destructor*/
  ~Multiset() = default;
  using Base::operator=;

  /*iterators*/
  using Base::begin;
  using Base::cbegin;
  using Base::cend;
  using Base::crbegin;
  using Base::crend;
  using Base::end;
  using Base::rbegin;
  using Base::rend;

  /*capacity*/
  using Base::empty;
  using Base::max_size;
  using Base::size;

  /*modifiers*/
  using Base::clear;
  using Base::emplace;
  using Base::emplace_hint;
  using Base::erase;
  using Base::extract;
  using Base::insert;
  using Base::merge;
  using Base::swap;

  /*lookup*/
  using Base::contains;
  using Base::count;
  using Base::equal_range;
  using Base::find;
  using Base::lower_bound;
  using Base::upper_bound;

  /*observers*/
  using Base::get_allocator;
  using Base::key_comp;
  using Base::value_comp;
};

}  // namespace ieda
