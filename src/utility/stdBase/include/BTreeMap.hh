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
 * @file Map.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The map container for the eda project.
 * @version 0.1
 * @date 2020-10-09
 */

#pragma once

#include <functional>
#include <list>
#include <utility>

#include "absl/container/btree_map.h"

namespace ieda {

/**
 * @brief A map made up of unique key based on tree structure.
 *
 * The map is a wrapper of btree map from google abseil containers.The btree map
 * contains ordered containers generally adhering to the STL container API
 * contract, but implemented using (generally more efficient) B-trees rather
 * than binary trees(as used in std::map et al). The ordered containers are
 * designed to be more efficient replacements for std::map and std::set in most
 * cases.Specifically, they provide several advantages over the ordered std::
 * containers: 1) Provide lower memory overhead in most cases than their STL
 * equivalents. 2) Are generally more cache friendly(and hence faster) than
 * their STL equivalents. 3) Provide C++11 support for C++17 mechanisms such as
 * try_emplace(). 4) Support heterogeneous lookup.
 *
 * @tparam KEY
 * @tparam VALUE
 * @tparam CMP
 */
template <class KEY, class VALUE, class CMP = std::less<KEY>>
class BTreeMap : public absl::btree_map<KEY, VALUE, CMP>
{
 public:
  using Base = typename BTreeMap::btree_map;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;
  using reverse_iterator = typename Base::reverse_iterator;
  using const_reverse_iterator = typename Base::const_reverse_iterator;
  using size_type = typename Base::size_type;
  using value_type = typename Base::value_type;

  /*constructor and destructor*/
  using Base::Base;
  /*destrcutor*/
  ~BTreeMap() = default;
  using Base::operator=;

  /*accessor*/
  using Base::at;
  using Base::operator[];

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
  using Base::contains;
  using Base::count;
  using Base::equal_range;
  using Base::find;
  using Base::lower_bound;
  using Base::upper_bound;

  /*observers*/
  using Base::key_comp;
  using Base::value_comp;
};

/**
 * @brief A ordered map of multiple elements with equivalent keys.
 *
 * The Multimap is a wrapper of btree multimap from google abseil containers.
 * The btree map implemented using B-trees is more efficent than binary tree.
 */
template <typename KEY, typename VALUE, typename CMP = std::less<KEY>>
class Multimap : public absl::btree_multimap<KEY, VALUE, CMP>
{
 public:
  using Base = typename Multimap::btree_multimap;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;
  using reverse_iterator = typename Base::reverse_iterator;
  using const_reverse_iterator = typename Base::const_reverse_iterator;
  using value_type = typename Base::value_type;

  /*constructor*/
  using Base::Base;
  /*destrcutor*/
  ~Multimap() = default;

  using Base::operator=;

  /*iterator*/
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

  /**
   * @brief Insert the (key, value) to the map container.
   *
   * @param key
   * @param value
   */
  void insert(const KEY& key, const VALUE& value) { insert(value_type(key, value)); }

  /**
   * @brief Get the mapped values equavilent to the key.
   *
   * @param key
   * @return std::list<VALUE>
   */
  std::list<VALUE> values(const KEY& key)
  {
    auto ret_values = equal_range(key);
    std::list<VALUE> ret_list;
    for (auto i = ret_values.first; i != ret_values.second; ++i) {
      ret_list.push_back(i->second);
    }

    return ret_list;
  }
};

}  // namespace ieda
