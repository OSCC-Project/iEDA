/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*               This file is part of the program and library                */
/*    PaPILO --- Parallel Presolve for Integer and Linear Optimization       */
/*                                                                           */
/* Copyright (C) 2020-2022 Konrad-Zuse-Zentrum                               */
/*                     fuer Informationstechnik Berlin                       */
/*                                                                           */
/* This program is free software: you can redistribute it and/or modify      */
/* it under the terms of the GNU Lesser General Public License as published  */
/* by the Free Software Foundation, either version 3 of the License, or      */
/* (at your option) any later version.                                       */
/*                                                                           */
/* This program is distributed in the hope that it will be useful,           */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU Lesser General Public License for more details.                       */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with this program.  If not, see <https://www.gnu.org/licenses/>.    */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _PAPILO_MISC_HASH_HPP_
#define _PAPILO_MISC_HASH_HPP_

#include "papilo/Config.hpp"

#ifndef PAPILO_USE_STANDARD_HASHMAP
#include "papilo/external/ska/bytell_hash_map.hpp"
#else
#include <unordered_map>
#include <unordered_set>
#endif

#include <cstdint>
#include <type_traits>

namespace papilo
{

template <typename T, int TWidth = sizeof( T )>
struct HashHelpers;

template <typename T>
struct HashHelpers<T, 4>
{
   static uint32_t
   fibonacci_muliplier()
   {
      return uint32_t( 0x9e3779b9 );
   }

   static uint32_t
   rotate_left( uint32_t x, int n )
   {
      return ( x << n ) | ( x >> ( 32 - n ) );
   }
};

template <typename T>
struct HashHelpers<T, 8>
{
   static uint64_t
   fibonacci_muliplier()
   {
      return uint64_t( 0x9e3779b97f4a7c15 );
   }

   static uint64_t
   rotate_left( uint64_t x, int n )
   {
      return ( x << n ) | ( x >> ( 64 - n ) );
   }
};

template <typename T, typename U = typename std::make_unsigned<T>::type>
struct Hasher;

// only add specialization for unsigned result types
template <typename T>
struct Hasher<T, T>
{
   T state;

   Hasher( T init = 0 ) : state( init ) {}

   template <typename U,
             typename std::enable_if<std::is_integral<U>::value, int>::type = 0>
   void
   addValue( U val )
   {
      state = ( HashHelpers<T>::rotate_left( state, 5 ) ^ T( val ) ) *
              HashHelpers<T>::fibonacci_muliplier();
   }

   T
   getHash() const
   {
      return state;
   }
};

#ifndef PAPILO_USE_STANDARD_HASHMAP

template <typename K, typename V, typename H = std::hash<K>,
          typename E = std::equal_to<K>>
using HashMap =
    ska::bytell_hash_map<K, V, H, E, Allocator<std::pair<const K, V>>>;

template <typename T, typename H = std::hash<T>, typename E = std::equal_to<T>>
using HashSet = ska::bytell_hash_set<T, H, E, Allocator<T>>;

#else

template <typename K, typename V, typename H = std::hash<K>,
          typename E = std::equal_to<K>>
using HashMap =
    std::unordered_map<K, V, H, E, Allocator<std::pair<const K, V>>>;

template <typename T, typename H = std::hash<T>, typename E = std::equal_to<T>>
using HashSet = std::unordered_set<T, H, E, Allocator<T>>;

#endif

} // namespace papilo

#endif
