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
 * @file FlatSet.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The hash set container for the eda project.
 * @version 0.1
 * @date 2020-10-09
 */

#pragma once

#include <unordered_set>
#include <utility>

#include "absl/container/flat_hash_set.h"

namespace ieda {

/**
 * @brief A hash set made up of unique key.
 *
 * The set is a wrapper of flat hash set from google abseil containers.The hash
 * tables are knowns as "Swiss tables".
 */
template <class KEY, class HASH = typename absl::flat_hash_set<KEY>::hasher, class EQ = typename absl::flat_hash_set<KEY>::key_equal>
class FlatSet : public absl::flat_hash_set<KEY, HASH, EQ>
{
 public:
  using Base = typename FlatSet::flat_hash_set;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;
  using value_type = typename Base::value_type;
  using hash = typename Base::hasher;
  using eq = typename Base::key_equal;

  /*constructor*/
  using Base::Base;
  /*destructor*/
  ~FlatSet() = default;
  using Base::operator=;

  /*iterator*/
  using Base::begin;
  using Base::cbegin;
  using Base::cend;
  using Base::end;

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

  /*bucket interface*/
  using Base::bucket_count;

  /*hash policy*/
  using Base::load_factor;
  using Base::max_load_factor;
  using Base::rehash;
  using Base::reserve;

  /*observers*/
  using Base::get_allocator;
  using Base::hash_function;
  using Base::key_eq;
};

/**
 * @brief A hash set of multiple elements with equivalent keys.
 *
 * The FlatMultiset is a wrapper of std unordered_multiset.So we can
 * add more convenient interface for development.
 *
 * @tparam KEY Type of key objects.
 */
template <typename KEY>
class FlatMultiset : public std::unordered_multiset<KEY>
{
 public:
  using Base = typename FlatMultiset::unordered_multiset;

  /*constructor*/
  using Base::Base;

  /*destrcutor*/
  ~FlatMultiset() = default;
  using Base::operator=;

  /*iterator*/
  using Base::begin;
  using Base::cbegin;
  using Base::cend;
  using Base::end;

  /*capacity*/
  using Base::empty;
  using Base::max_size;
  using Base::size;

  /*modifier*/
  using Base::clear;
  using Base::emplace;
  using Base::emplace_hint;
  using Base::erase;

  // using Base::extract; /* not supported yet */
  using Base::insert;

  // using Base::merge; /* not supported yet */
  using Base::swap;

  /*lookup*/
  using Base::count;
  using Base::equal_range;
  using Base::find;

  /*bucket interface*/
  using Base::bucket_count;

  /*hash policy*/
  using Base::load_factor;
  using Base::max_load_factor;
  using Base::rehash;
  using Base::reserve;

  using Base::hash_function;
  using Base::key_eq;
};

}  // namespace ieda
