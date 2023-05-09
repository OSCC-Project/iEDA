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
 * @file MemoryPool.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2020-11-07
 */

#pragma once

#include "boost/pool/object_pool.hpp"
#include "boost/pool/pool_alloc.hpp"
#include "boost/pool/poolfwd.hpp"
#include "boost/pool/singleton_pool.hpp"

namespace ieda {

/**
 * @brief A memory pool for store object.It is a wrapper of boost::object_pool.
 * The memory pool can reduce the memory allocate and deallocate time and
 * improve the memory usage.
 *
 * @tparam T object type.
 */
template <typename T>
class ObjectPool : public boost::object_pool<T>
{
 public:
  using Base = typename ObjectPool::object_pool;
  using size_type = typename Base::size_type;
  using difference_type = typename Base::difference_type;
  using element_type = T;

  /*constructor and destructor*/
  using Base::Base;
  ~ObjectPool() = default;

  /*construct the object*/
  using Base::construct;

  /*destory the object*/
  using Base::destroy;
};

/**
 * @brief A singleton memory pool.It can be used as global meomory pool.The Tag
 * can be used as distinguish different singleton pool.The signleton memory is
 * used to malloc pod data.
 *
 * @tparam TAG the tag for distinguish singleton pool.
 * @tparam ES the element size.
 */
template <typename TAG, unsigned ES>
class SingletonPool : public boost::singleton_pool<TAG, ES>
{
 public:
  using Base = typename SingletonPool::singleton_pool;
  using Tag = TAG;
  using size_type = typename Base::size_type;
  using difference_type = typename Base::difference_type;

  using Base::Base;
  ~SingletonPool() = default;

  using Base::free;
  using Base::malloc;
  using Base::ordered_free;
  using Base::ordered_malloc;

  /**
   * @brief Frees every memory block.
   * This function invalidates any pointers previously returned by allocation
   * functions of t.Returns true if at least one memory block was freed.
   */
  using Base::purge_memory;
  /**
   * @brief Frees every memory block that doesn't have any allocated chunks.
   * Returns true if at least one memory block was freed.   *
   */
  using Base::release_memory;
};

template <typename T>
using PoolAllocator = boost::pool_allocator<T>;  //!< pool allocator is used for allocate memory
                                                 //!< for vector, that can be used for fast and
                                                 //!< efficient memory allocation in conjunction
                                                 //!< with the C++ Standard Library containers.

template <typename T>
using FastPoolAllocator = boost::fast_pool_allocator<T>;  // pool_allocator is a more general-purpose solution, geared towards
                                                          // efficiently servicing requests for any number of contiguous
                                                          // chunks.fast_pool_allocator is also a general-purpose solution but is
                                                          // geared towards efficiently servicing requests for one chunk at a
                                                          // time; it will work for contiguous chunks, but not as well as
                                                          // pool_allocator.

}  // namespace ieda
