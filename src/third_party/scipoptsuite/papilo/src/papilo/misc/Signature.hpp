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

#ifndef _PAPILO_MISC_SIGNATURE_HPP_
#define _PAPILO_MISC_SIGNATURE_HPP_

#include "papilo/misc/Hash.hpp"
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace papilo
{

template <typename T>
class Signature
{
 public:
   Signature() : state( 0 ) {}

   template <typename U>
   void
   add( U elem )
   {
      state |=
          1 << ( ( uint32_t( elem ) *
                   HashHelpers<uint32_t>::fibonacci_muliplier() ) >>
                 ( 32 - static_cast<int>( std::log2( 8 * sizeof( T ) ) ) ) );
   }

   // template <typename U>
   // void
   // add( U elem )
   //{
   //   state |= 1 << ( uint32_t( elem ) % ( sizeof( T ) * 8 ) );
   //}

   bool
   isSubset( Signature other )
   {
      return ( state & ~other.state ) == 0;
   }

   bool
   isSuperset( Signature other )
   {
      return ( other.state & ~state ) == 0;
   }

   bool
   isEqual( Signature other )
   {
      return state == other.state;
   }

 private:
   T state;
};

using Signature32 = Signature<uint32_t>;
using Signature64 = Signature<uint64_t>;

} // namespace papilo

#endif
