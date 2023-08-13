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

#ifndef _PAPILO_MISC_ARRAY_HPP_
#define _PAPILO_MISC_ARRAY_HPP_

#include "papilo/misc/Alloc.hpp"
#include <cstdint>
#include <memory>

namespace papilo
{

template <typename T>
struct ArrayDeleter
{
   std::size_t size;

   ArrayDeleter( std::size_t _size ) : size( _size ) {}

   void
   operator()( T* p )
   {
      Allocator<T>().deallocate( p, size );
   }
};

template <typename T>
class Array
{
 public:
   Array( std::size_t n )
       : ptr( Allocator<T>().allocate( n ), ArrayDeleter<T>( n ) )
   {
   }

   std::size_t
   getSize() const
   {
      return ptr.get_deleter().size;
   }

   T&
   operator[]( int i )
   {
      return ptr[i];
   }

   const T&
   operator[]( int i ) const
   {
      return ptr[i];
   }

 private:
   std::unique_ptr<T[], ArrayDeleter<T>> ptr;
};

} // namespace papilo

#endif
