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

#ifndef _PAPILO_MISC_FLAGS_HPP_
#define _PAPILO_MISC_FLAGS_HPP_

#include <cstdint>
#include <type_traits>

namespace papilo
{

template <typename BaseType>
class Flags
{
 public:
   Flags( BaseType t ) : state( static_cast<UnderlyingType>( t ) ) {}

   Flags() : state( 0 ) {}

   template <typename... Args>
   void
   set( Args... flags )
   {
      state |= joinFlags( flags... );
   }

   void
   clear()
   {
      state = 0;
   }

   template <typename... Args>
   void
   unset( Args... flags )
   {
      state &= ~joinFlags( flags... );
   }

   template <typename... Args>
   bool
   test( Args... flags ) const
   {
      return state & joinFlags( flags... );
   }

   template <typename... Args>
   bool
   equal( Args... flags ) const
   {
      return state == joinFlags( flags... );
   }

   template <typename Archive>
   void
   serialize( Archive& ar, const unsigned int version )
   {
      ar& state;
   }

 private:
   using UnderlyingType = typename std::underlying_type<BaseType>::type;

   static UnderlyingType
   joinFlags( BaseType f1 )
   {
      return static_cast<UnderlyingType>( f1 );
   }

   template <typename... Args>
   static UnderlyingType
   joinFlags( BaseType f1, Args... other )
   {
      return static_cast<UnderlyingType>( f1 ) | joinFlags( other... );
   }

   UnderlyingType state;
};

} // namespace papilo

#endif
