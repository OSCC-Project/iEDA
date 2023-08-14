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

#ifndef _PAPILO_MISC_COMPRESS_VECTOR_HPP_
#define _PAPILO_MISC_COMPRESS_VECTOR_HPP_

#include "papilo/misc/Vec.hpp"
#include <cassert>

namespace papilo
{

/// helper function to compress a vector-like container using the given mapping
template <typename VEC>
void
compress_vector( const Vec<int>& mapping, VEC& vec )
{
   assert( vec.size() == mapping.size() );

   int newSize = 0;
   for( int i = 0; i != static_cast<int>( vec.size() ); ++i )
   {
      assert( mapping[i] <= i );

      if( mapping[i] != -1 )
      {
         vec[mapping[i]] = vec[i];
         newSize++;
      }
   }
   vec.resize( newSize );
}

/// helper function to compress a vector-like container of indicies using the
/// given mapping
template <typename VEC>
void
compress_index_vector( const Vec<int>& mapping, VEC& vec )
{
   int offset = 0;
   for( std::size_t i = 0; i < vec.size(); ++i )
   {
      int newindex = mapping[vec[i]];
      if( newindex != -1 )
         vec[i - offset] = newindex;
      else
         ++offset;
   }

   vec.resize( vec.size() - offset );
}

} // namespace papilo

#endif