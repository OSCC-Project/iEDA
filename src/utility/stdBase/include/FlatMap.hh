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
 * @file FlatMap.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The hash map container for the eda project.
 * @version 0.1
 * @date 2020-10-09
 */

#pragma once

#include <list>
#include <unordered_map>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"

namespace ieda {

/**
 * @brief A hash map made up of unique key.
 *
 * The hash map is a wrapper of flat hash map from google abseil containers.The
 * hash tables are knows as "Swiss tables" and are designed to be replacements
 * for std::unordered_map.They provide several advantages over the stl
 * containers: 1) Provides C++11 support for C++17 mechanisms such as
 * try_emplace(); 2) Support heterogeneous lookup; 3) Allow optimizations for
 * emplace({key, value}) to avoid allocating a pair in most common cases; 4)
 * Support a heterogeneous std::initializer_list to avoid extra copies for
 * construction and insertion; 5) Guarantees an O(1) erase method by returning
 * void instead of an iterator.
 *
 * Guarantees
 * 1) Keys and values are stored inline; 2) Iterators, references, and pointers
 * to elements are invalidated on rehash; 3) Move operation do not invalidate
 * iterators or pointers.
 *
 * Memory Usage
 * The container uses O((sizeof(std::pair<const K, V>) + 1) * bucket_count()),
 * the max load factor is 87.5%, after which the table doubles in size(making
 * load factor go down by 2x).
 *
 * Heterogeneous Lookup
 * Inserting into or looking up an element within an associative container
 * requires a key.In general, containers require the keys to be of a specific
 * type, which can lead to inefficiencies at call sites that need to convert
 * between near-equivalent types(such as std::string and absl::string_view).To
 * avoid this unnecessary work, the Swiss tables provide heterogeneous lookup
 * for conversions to string types, and for conversions to smart pointer
 * types(std::unique_ptr, std::shared_ptr), through the absl::Hash hashing
 * framwork.
 *
 * @tparam KEY Type of  key.
 * @tparam VALUE Type of value.
 */
template <class KEY, class VALUE, class HASH = typename absl::flat_hash_map<KEY, VALUE>::hasher,
          class EQ = typename absl::flat_hash_map<KEY, VALUE>::key_equal>
class FlatMap : public absl::flat_hash_map<KEY, VALUE, HASH, EQ>
{
 public:
  using Base = typename FlatMap::flat_hash_map;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;
  using value_type = typename Base::value_type;
  using hash = typename Base::hasher;
  using eq = typename Base::key_equal;

  /*constructor*/
  using Base::Base;

  /*destrcutor*/
  ~FlatMap() = default;
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
  using Base::extract;
  using Base::insert;
  using Base::insert_or_assign;
  using Base::merge;
  using Base::swap;
  using Base::try_emplace;

  /*lookup*/
  using Base::at;
  using Base::operator[];
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

  using Base::hash_function;
  using Base::key_eq;
};

/**
 * @brief A hash map of multiple elements with equivalent keys.
 *
 * The FlatMultimap is a wrapper of std unordered_multimap.So we can add
 * more convenient interface for development.
 *
 * @tparam KEY Type of key objects.
 * @tparam VALUE Type of value objects.
 */
template <class KEY, class VALUE>
class FlatMultimap : public std::unordered_multimap<KEY, VALUE>
{
 public:
  using Base = typename FlatMultimap::unordered_multimap;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;
  using value_type = typename Base::value_type;

  /*constructor*/
  using Base::Base;

  /*destrcutor*/
  ~FlatMultimap() = default;
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
  using Base::extract;
  using Base::insert;
  using Base::merge;
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
