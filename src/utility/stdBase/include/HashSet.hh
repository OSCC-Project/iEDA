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
 * @file HashSet.h
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
class HashSet : public absl::flat_hash_set<KEY, HASH, EQ>
{
 public:
  using Base = typename HashSet::flat_hash_set;
  using iterator = typename Base::iterator;
  using const_iterator = typename Base::const_iterator;
  using value_type = typename Base::value_type;
  using hash = typename Base::hasher;
  using eq = typename Base::key_equal;

  /*constructor*/
  using Base::Base;
  /*destructor*/
  ~HashSet() = default;
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

  /**
   * @brief Removes all items from this set that are contained in the other set.
   *
   * Returns a reference to this set.
   */
  HashSet<KEY, HASH, EQ>& subtract(const HashSet<KEY, HASH, EQ>& other)
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
   * @return HashSet<KEY, HASH, EQ>& This set after unite the other.
   */
  HashSet<KEY, HASH, EQ>& unite(const HashSet<KEY, HASH, EQ>& other)
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
   * @return HashSet<KEY, HASH, EQ>& This set after intersect the other.
   */
  HashSet<KEY, HASH, EQ>& intersect(const HashSet<KEY, HASH, EQ>& other)
  {
    HashSet<KEY, HASH, EQ> copy1;
    HashSet<KEY, HASH, EQ> copy2;
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
   * @return HashSet<KEY, HASH, EQ>& This set after insert value.
   */
  inline HashSet<KEY, HASH, EQ>& operator<<(const KEY& value)
  {
    insert(value);
    return *this;
  }

  /**
   * @brief Calculate the union set between this set and the other set.
   *
   * @param other
   * @return HashSet<KEY, HASH, EQ>& This set after unite the other.
   */
  inline HashSet<KEY, HASH, EQ>& operator|=(const HashSet<KEY, HASH, EQ>& other)
  {
    unite(other);
    return *this;
  }

  inline HashSet<KEY, HASH, EQ>& operator|=(HashSet<KEY, HASH, EQ>&& other)
  {
    unite(other);
    return *this;
  }
  inline HashSet<KEY, HASH, EQ>& operator|=(const KEY& value)
  {
    insert(value);
    return *this;
  }

  /**
   * @brief Calculate the intersect set between this set and the other set.
   *
   * @param other
   * @return HashSet<KEY, HASH, EQ>& This set after intersect the other set.
   */
  inline HashSet<KEY, HASH, EQ>& operator&=(const HashSet<KEY, HASH, EQ>& other)
  {
    intersect(other);
    return *this;
  }
  inline HashSet<KEY, HASH, EQ>& operator&=(const KEY& value)
  {
    HashSet<KEY, HASH, EQ> result;
    if (contains(value)) {
      result.insert(value);
    }
    return (*this = result);
  }

  /**
   * @brief Merge the other set to the set.
   *
   * @param other
   * @return HashSet<KEY, HASH, EQ>& This set after merge the other.
   */
  inline HashSet<KEY, HASH, EQ>& operator+=(HashSet<KEY, HASH, EQ>& other)
  {
    merge(other);
    return *this;
  }
  inline HashSet<KEY, HASH, EQ>& operator+=(const KEY& value)
  {
    insert(value);
    return *this;
  }

  /**
   * @brief Subtract the other set from the set.
   *
   * @param other
   * @return HashSet<KEY, HASH, EQ>& This set after subtract the other.
   */
  inline HashSet<KEY, HASH, EQ>& operator-=(const HashSet<KEY, HASH, EQ>& other)
  {
    subtract(other);
    return *this;
  }
  inline HashSet<KEY, HASH, EQ>& operator-=(const KEY& value)
  {
    erase(value);
    return *this;
  }
  inline HashSet<KEY, HASH, EQ> operator|(const HashSet<KEY, HASH, EQ>& other) const
  {
    HashSet<KEY, HASH, EQ> result = *this;
    result |= other;
    return result;
  }
  inline HashSet<KEY, HASH, EQ> operator&(const HashSet<KEY, HASH, EQ>& other) const
  {
    HashSet<KEY, HASH, EQ> result = *this;
    result &= other;
    return result;
  }
  inline HashSet<KEY, HASH, EQ> operator+(HashSet<KEY, HASH, EQ>& other) const
  {
    HashSet<KEY, HASH, EQ> result = *this;
    result += other;
    return result;
  }
  inline HashSet<KEY, HASH, EQ> operator-(const HashSet<KEY, HASH, EQ>& other) const
  {
    HashSet<KEY, HASH, EQ> result = *this;
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
  static bool equal(const HashSet<KEY, HASH, EQ>* set1, const HashSet<KEY, HASH, EQ>* set2);

  /**
   * @brief Judge the set2 is the subset of the set.
   *
   * @param set2
   * @return true if set2 is a subset of this set.
   * @return false
   */
  bool isSubset(const HashSet<KEY, HASH, EQ>* set2);

  void insertSet(const HashSet<KEY, HASH, EQ>* set2);
  static bool intersects(HashSet<KEY, HASH, EQ>* set1, HashSet<KEY, HASH, EQ>* set2);

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

    explicit Iterator(HashSet<KEY, HASH, EQ>* container)
    {
      if (container != nullptr) {
        _container = container;
        _iter = container->begin();
      }
    }

    void init(HashSet<KEY, HASH, EQ>* container)
    {
      if (container != nullptr) {
        _container = container;
        _iter = container->begin();
      }
    }

    bool hasNext() { return _container != nullptr && _iter != _container->end(); }
    Iterator& next()
    {
      ++_iter;
      return *this;
    }
    HashSet<KEY, HASH, EQ>* container() { return _container; }
    inline const KEY& value() const { return *_iter; }
    inline const KEY& operator*() const { return _iter->value(); }
    inline const KEY& operator->() const { return &_iter->value(); }
    inline bool operator==(const Iterator& o) const { return _iter == o._iter; }
    inline bool operator!=(const Iterator& o) const { return _iter != o._iter; }

   private:
    HashSet<KEY, HASH, EQ>* _container = nullptr;
    typename HashSet<KEY, HASH, EQ>::iterator _iter;
  };

  class ConstIterator
  {
   public:
    ConstIterator() = default;
    ~ConstIterator() = default;

    explicit ConstIterator(const HashSet<KEY, HASH, EQ>* container) : _container(container)
    {
      if (_container != nullptr) {
        _iter = _container->begin();
      }
    }

    void init(const HashSet<KEY, HASH, EQ>* container)
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
    const HashSet<KEY, HASH, EQ>* container() { return _container; }
    inline const KEY& value() const { return *_iter; }
    inline const KEY& operator*() const { return *_iter; }
    inline const KEY* operator->() const { return &(_iter); }
    inline bool operator==(const ConstIterator& o) const { return _iter == o._iter; }
    inline bool operator!=(const ConstIterator& o) const { return _iter != o._iter; }

   private:
    const HashSet<KEY, HASH, EQ>* _container = nullptr;
    typename HashSet<KEY, HASH, EQ>::const_iterator _iter;
  };
};

template <class KEY, class HASH, class EQ>
bool HashSet<KEY, HASH, EQ>::equal(const HashSet<KEY, HASH, EQ>* set1, const HashSet<KEY, HASH, EQ>* set2)
{
  if ((set1 == nullptr || set1->empty()) && (set2 == nullptr || set2->empty())) {
    return true;
  } else if (set1 && set2) {
    if (set1->size() == set2->size()) {
      typename HashSet<KEY, HASH, EQ>::ConstIterator iter1(set1);
      typename HashSet<KEY, HASH, EQ>::ConstIterator iter2(set2);
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

template <class KEY, class HASH, class EQ>
bool HashSet<KEY, HASH, EQ>::isSubset(const HashSet<KEY, HASH, EQ>* set2)
{
  if (this->empty() && set2->empty()) {
    return true;
  } else {
    typename HashSet<KEY, HASH, EQ>::ConstIterator iter2(set2);
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
 * @brief Judge whether exist the intersect set between set1 and set2.
 *
 * @tparam KEY
 * @param set1
 * @param set2
 * @return true
 * @return false
 */
template <class KEY, class HASH, class EQ>
bool HashSet<KEY, HASH, EQ>::intersects(HashSet<KEY, HASH, EQ>* set1, HashSet<KEY, HASH, EQ>* set2)
{
  if (set1 && !set1->empty() && set2 && !set2->empty()) {
    const HashSet<KEY, HASH, EQ>* small = set1;
    const HashSet<KEY, HASH, EQ>* big = set2;
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

/**
 * @brief Insert the set2 to this set.
 *
 * @tparam KEY
 * @param set2
 */
template <class KEY, class HASH, class EQ>
void HashSet<KEY, HASH, EQ>::insertSet(const HashSet<KEY, HASH, EQ>* set2)
{
  if (set2) {
    this->insert(set2->begin(), set2->end());
  }
}

template <class KEY, class HASH, class EQ>
inline void swap(HashSet<KEY, HASH, EQ>& x, HashSet<KEY, HASH, EQ>& y) noexcept(noexcept(x.swap(y)))
{
  x.swap(y);
}

template <class KEY, class HASH, class EQ>
inline bool operator==(const HashSet<KEY, HASH, EQ>& x, const HashSet<KEY, HASH, EQ>& y)
{
  return static_cast<const typename HashSet<KEY, HASH, EQ>::Base&>(x) == y;
}

template <class KEY, class HASH, class EQ>
inline bool operator!=(const HashSet<KEY, HASH, EQ>& x, const HashSet<KEY, HASH, EQ>& y)
{
  return !(x == y);
}

/**
 * @brief A hash set of multiple elements with equivalent keys.
 *
 * The HashMultiset is a wrapper of std unordered_multiset.So we can
 * add more convenient interface for development.
 *
 * @tparam KEY Type of key objects.
 */
template <typename KEY>
class HashMultiset : public std::unordered_multiset<KEY>
{
 public:
  using Base = typename HashMultiset::unordered_multiset;

  /*constructor*/
  using Base::Base;

  /*destrcutor*/
  ~HashMultiset() = default;
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

template <typename KEY>
inline void swap(HashMultiset<KEY>& x, HashMultiset<KEY>& y) noexcept(noexcept(x.swap(y)))
{
  x.swap(y);
}

template <typename KEY>
inline bool operator==(const HashMultiset<KEY>& x, const HashMultiset<KEY>& y)
{
  return static_cast<const typename HashMultiset<KEY>::Base&>(x) == y;
}

template <typename KEY>
inline bool operator!=(const HashMultiset<KEY>& x, const HashMultiset<KEY>& y)
{
  return !(x == y);
}

}  // namespace ieda
