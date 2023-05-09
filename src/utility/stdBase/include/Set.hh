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
class Set : public absl::btree_set<KEY, CMP>
{
 public:
  using Base = typename Set::btree_set;
  using key_type = typename Base::key_type;
  using size_type = typename Base::value_type;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;
  using reverse_iterator = typename Base::reverse_iterator;
  using const_reverse_iterator = typename Base::const_reverse_iterator;

  /*constructor*/
  using Base::Base;
  /*destrcutor*/
  ~Set() = default;
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

  /**
   * @brief Removes all items from this set that are contained in the other set.
   *
   * Returns a reference to this set.
   */
  Set<KEY, CMP>& subtract(const Set<KEY, CMP>& other)
  {
    for (const auto& e : other) {
      erase(e);
    }

    return *this;
  }
  /**
   * @brief Insert all items from the other set.
   *
   * @param other
   * @return HSet<KEY>& This set after unite the other.
   */
  Set<KEY, CMP>& unite(const Set<KEY, CMP>& other)
  {
    for (const KEY& e : other) {
      insert(e);
    }
    return *this;
  }

  /**
   * @brief Calculate the intersect between this and other.
   *
   * @param other
   * @return HSet<KEY>& This set after intersect the other.
   */
  Set<KEY, CMP>& intersect(const Set<KEY, CMP>& other)
  {
    Set<KEY, CMP> copy1;
    Set<KEY, CMP> copy2;
    if (size() <= other.size()) {
      copy1 = *this;
      copy2 = other;
    } else {
      copy1 = other;
      copy2 = *this;
      *this = copy1;
    }
    for (const auto& e : copy1) {
      if (!copy2.contains(e)) {
        erase(e);
      }
    }
    return *this;
  }

  /**
   * @brief Insert a value to the set.
   *
   * @param value
   * @return HSet<KEY>& This set after insert value.
   */
  inline Set<KEY, CMP>& operator<<(const KEY& value)
  {
    insert(value);
    return *this;
  }

  /**
   * @brief Calculate the union set between this set and the other set.
   *
   * @param other
   * @return HSet<KEY>& This set after unite the other.
   */
  inline Set<KEY, CMP>& operator|=(const Set<KEY, CMP>& other)
  {
    unite(other);
    return *this;
  }

  inline Set<KEY, CMP>& operator|=(Set<KEY, CMP>&& other)
  {
    unite(other);
    return *this;
  }
  inline Set<KEY, CMP>& operator|=(const KEY& value)
  {
    insert(value);
    return *this;
  }

  /**
   * @brief Calculate the intersect set between this set and the other set.
   *
   * @param other
   * @return HSet<KEY>& This set after intersect the other set.
   */
  inline Set<KEY, CMP>& operator&=(const Set<KEY, CMP>& other)
  {
    intersect(other);
    return *this;
  }
  inline Set<KEY, CMP>& operator&=(const KEY& value)
  {
    Set<KEY, CMP> result;
    if (contains(value)) {
      result.insert(value);
    }
    return (*this = result);
  }

  /**
   * @brief Merge the other set to the set.
   *
   * @param other
   * @return HSet<KEY>& This set after merge the other.
   */
  inline Set<KEY, CMP>& operator+=(Set<KEY, CMP>& other)
  {
    merge(other);
    return *this;
  }
  inline Set<KEY, CMP>& operator+=(const KEY& value)
  {
    insert(value);
    return *this;
  }

  /**
   * @brief Subtract the other set from the set.
   *
   * @param other
   * @return HSet<KEY>& This set after subtract the other.
   */
  inline Set<KEY, CMP>& operator-=(const Set<KEY, CMP>& other)
  {
    subtract(other);
    return *this;
  }
  inline Set<KEY, CMP>& operator-=(const KEY& value)
  {
    erase(value);
    return *this;
  }
  inline Set<KEY, CMP> operator|(const Set<KEY, CMP>& other) const
  {
    Set<KEY, CMP> result = *this;
    result |= other;
    return result;
  }
  inline Set<KEY, CMP> operator&(const Set<KEY, CMP>& other) const
  {
    Set<KEY, CMP> result = *this;
    result &= other;
    return result;
  }
  inline Set<KEY, CMP> operator+(Set<KEY, CMP>& other) const
  {
    Set<KEY, CMP> result = *this;
    result += other;
    return result;
  }
  inline Set<KEY, CMP> operator-(const Set<KEY, CMP>& other) const
  {
    Set<KEY, CMP> result = *this;
    result -= other;
    return result;
  }

  /**
   * @brief Find out if key is in the set.
   *
   * @param key
   * @return true when has key.
   * @return false when do not has key.
   */
  bool hasKey(const KEY key) const
  {
    auto find_iter = this->find(key);
    return find_iter != this->end();
  }

  /**
   * @brief Judge whether the two set is equal.
   *
   * @param set1
   * @param set2
   * @return true when the two set is equal.
   * @return false
   */
  static bool equal(const Set<KEY, CMP>* set1, const Set<KEY, CMP>* set2);

  /**
   * @brief Judge the set2 is the subset of the set.
   *
   * @param set2
   * @return true if set2 is a subset of this set.
   * @return false
   */
  bool isSubset(const Set<KEY, CMP>* set2);
  void insertSet(const Set<KEY, CMP>* set2);

  static bool intersects(Set<KEY, CMP>* set1, Set<KEY, CMP>* set2);

  /**
   * @brief Java style container itererator.
   *
   *  for example:
   *  Set::Iterator<Key*> iter(set);
   *  while (iter.hasNext()) {
   *    iter.next();
   *  }
   */
  class Iterator
  {
   public:
    Iterator() = default;
    ~Iterator() = default;

    explicit Iterator(Set<KEY, CMP>* container) : _container(container)
    {
      if (_container != nullptr) {
        _iter = _container->begin();
      }
    }

    void init(Set<KEY, CMP>* container)
    {
      _container = container;
      if (_container != nullptr) {
        _iter = _container->begin();
      }
    }

    bool hasNext() { return _container != nullptr && _iter != _container->end(); }
    Iterator& next()
    {
      ++_iter;
      return *this;
    }
    Set<KEY, CMP>* container() { return _container; }
    inline const KEY& value() const { return *_iter; }
    inline const KEY& operator*() const { return _iter->value(); }
    inline const KEY& operator->() const { return &_iter->value(); }
    inline bool operator==(const Iterator& o) const { return _iter == o._iter; }
    inline bool operator!=(const Iterator& o) const { return _iter != o._iter; }

   private:
    Set<KEY, CMP>* _container = nullptr;
    typename Set<KEY, CMP>::iterator _iter;
  };

  class ConstIterator
  {
   public:
    ConstIterator() = default;
    ~ConstIterator() = default;

    explicit ConstIterator(const Set<KEY, CMP>* container) : _container(container)
    {
      if (_container != nullptr)
        _iter = _container->begin();
    }

    void init(const Set<KEY, CMP>* container)
    {
      _container = container;
      if (_container != nullptr) {
        _iter = _container->begin();
      }
    }

    bool hasNext() { return _container != nullptr && _iter != _container->end(); }
    ConstIterator& next()
    {
      ++_iter;
      return *this;
    }
    const Set<KEY, CMP>* container() { return _container; }
    inline const KEY& value() const { return *_iter; }
    inline const KEY& operator*() const { return *_iter; }
    inline const KEY* operator->() const { return &(_iter); }
    inline bool operator==(const ConstIterator& o) const { return _iter == o._iter; }
    inline bool operator!=(const ConstIterator& o) const { return _iter != o._iter; }

   private:
    const Set<KEY, CMP>* _container = nullptr;
    typename Set<KEY, CMP>::const_iterator _iter;
  };
};

template <class KEY, class CMP>
bool Set<KEY, CMP>::equal(const Set<KEY, CMP>* set1, const Set<KEY, CMP>* set2)
{
  if ((set1 == nullptr || set1->empty()) && (set2 == nullptr || set2->empty())) {
    return true;
  } else if (set1 && set2) {
    if (set1->size() == set2->size()) {
      typename Set<KEY, CMP>::ConstIterator iter1(set1);
      typename Set<KEY, CMP>::ConstIterator iter2(set2);
      while (iter1.hasNext() && iter2.hasNext()) {
        auto& value1 = iter1.value();
        auto& value2 = iter2.value();
        if (value1 != value2) {
          return false;
        }

        iter1.next();
        iter2.next();
      }
      return true;
    } else {
      return false;
    }

  } else {
    return false;
  }
}

template <class KEY, class CMP>
bool Set<KEY, CMP>::isSubset(const Set<KEY, CMP>* set2)
{
  if (this->empty() && set2->empty()) {
    return true;
  } else {
    typename Set<KEY, CMP>::ConstIterator iter2(set2);
    while (iter2.hasNext()) {
      const KEY key2 = iter2.value();
      if (!hasKey(key2)) {
        return false;
      }

      iter2.next();
    }
    return true;
  }
}

/**
 * @brief Judge whether the two set has at least one common item.
 *
 * @tparam KEY
 * @tparam CMP
 * @param set1
 * @param set2
 * @return true if this set has at least one item in common with other.
 * @return false
 */
template <class KEY, class CMP>
bool Set<KEY, CMP>::intersects(Set<KEY, CMP>* set1, Set<KEY, CMP>* set2)
{
  if (set1 && !set1->empty() && set2 && !set2->empty()) {
    const Set<KEY, CMP>* small = set1;
    const Set<KEY, CMP>* big = set2;
    if (small->size() > big->size()) {
      small = set2;
      big = set1;
    }
    auto iter1 = big->begin();
    auto last1 = big->end();
    auto iter2 = small->begin();
    auto last2 = small->end();
    if (static_cast<float>(small->size() + big->size()) < (small->size() * log(static_cast<float>(big->size())))) {
      while (iter1 != last1 && iter2 != last2) {
        if (*iter1 < *iter2) {
          ++iter1;
        } else if (*iter2 < *iter1) {
          ++iter2;
        } else {
          return true;
        }
      }
    } else {
      for (/* empty */; iter2 != last2; ++iter2) {
        const KEY key2 = *iter2;
        if (big->find(key2) != last1) {
          return true;
        }
      }
    }
  }
  return false;
}

template <class KEY, class CMP>
void Set<KEY, CMP>::insertSet(const Set<KEY, CMP>* set2)
{
  if (set2) {
    this->insert(set2->begin(), set2->end());
  }
}

template <typename KEY, typename CMP>
inline bool operator==(const Set<KEY, CMP>& lhs, const Set<KEY, CMP>& rhs)
{
  const typename Set<KEY, CMP>::Base& set1_base = lhs;
  const typename Set<KEY, CMP>::Base& set2_base = rhs;
  return set1_base == set2_base;
}

template <typename KEY, typename CMP>
inline bool operator!=(const Set<KEY, CMP>& lhs, const Set<KEY, CMP>& rhs)
{
  const typename Set<KEY, CMP>::Base& set1_base = lhs;
  const typename Set<KEY, CMP>::Base& set2_base = rhs;
  return set1_base != set2_base;
}

template <typename KEY, typename CMP>
inline bool operator<(const Set<KEY, CMP>& lhs, const Set<KEY, CMP>& rhs)
{
  const typename Set<KEY, CMP>::Base& set1_base = lhs;
  const typename Set<KEY, CMP>::Base& set2_base = rhs;
  return set1_base < set2_base;
}

template <typename KEY, typename CMP>
inline bool operator<=(const Set<KEY, CMP>& lhs, const Set<KEY, CMP>& rhs)
{
  const typename Set<KEY, CMP>::Base& set1_base = lhs;
  const typename Set<KEY, CMP>::Base& set2_base = rhs;
  return set1_base <= set2_base;
}

template <typename KEY, typename CMP>
inline bool operator>=(const Set<KEY, CMP>& lhs, const Set<KEY, CMP>& rhs)
{
  const typename Set<KEY, CMP>::Base& set1_base = lhs;
  const typename Set<KEY, CMP>::Base& set2_base = rhs;
  return set1_base >= set2_base;
}

template <typename KEY, typename CMP>
inline bool operator>(const Set<KEY, CMP>& lhs, const Set<KEY, CMP>& rhs)
{
  const typename Set<KEY, CMP>::Base& set1_base = lhs;
  const typename Set<KEY, CMP>::Base& set2_base = rhs;
  return set1_base > set2_base;
}

template <typename KEY, typename CMP>
void swap(Set<KEY, CMP>& x, Set<KEY, CMP>& y)
{
  return x.swap(y);
}

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

template <typename KEY, typename CMP>
inline bool operator==(const Multiset<KEY, CMP>& lhs, const Multiset<KEY, CMP>& rhs)
{
  const typename Multiset<KEY, CMP>::Base& set1_base = lhs;
  const typename Multiset<KEY, CMP>::Base& set2_base = rhs;
  return set1_base == set2_base;
}

template <typename KEY, typename CMP>
inline bool operator!=(const Multiset<KEY, CMP>& lhs, const Multiset<KEY, CMP>& rhs)
{
  const typename Multiset<KEY, CMP>::Base& set1_base = lhs;
  const typename Multiset<KEY, CMP>::Base& set2_base = rhs;
  return set1_base != set2_base;
}

template <typename KEY, typename CMP>
inline bool operator<(const Multiset<KEY, CMP>& lhs, const Multiset<KEY, CMP>& rhs)
{
  const typename Multiset<KEY, CMP>::Base& set1_base = lhs;
  const typename Multiset<KEY, CMP>::Base& set2_base = rhs;
  return set1_base < set2_base;
}

template <typename KEY, typename CMP>
inline bool operator<=(const Multiset<KEY, CMP>& lhs, const Multiset<KEY, CMP>& rhs)
{
  const typename Multiset<KEY, CMP>::Base& set1_base = lhs;
  const typename Multiset<KEY, CMP>::Base& set2_base = rhs;
  return set1_base <= set2_base;
}

template <typename KEY, typename CMP>
inline bool operator>=(const Multiset<KEY, CMP>& lhs, const Multiset<KEY, CMP>& rhs)
{
  const typename Multiset<KEY, CMP>::Base& set1_base = lhs;
  const typename Multiset<KEY, CMP>::Base& set2_base = rhs;
  return set1_base >= set2_base;
}

template <typename KEY, typename CMP>
inline bool operator>(const Multiset<KEY, CMP>& lhs, const Multiset<KEY, CMP>& rhs)
{
  const typename Multiset<KEY, CMP>::Base& set1_base = lhs;
  const typename Multiset<KEY, CMP>::Base& set2_base = rhs;
  return set1_base > set2_base;
}

template <typename KEY, typename CMP>
void swap(Multiset<KEY, CMP>& x, Multiset<KEY, CMP>& y)
{
  return x.swap(y);
}

}  // namespace ieda
