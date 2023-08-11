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

#include "papilo/core/MatrixBuffer.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/misc/fmt.hpp"

using namespace papilo;

template <bool StorageOrder>
bool
checkHeapProperty( MatrixBuffer<double>& M )
{
   using Node = GetNodeProperty<StorageOrder>;

   if( StorageOrder )
      fmt::print( "check row major heap property\n" );
   else
      fmt::print( "check col major heap property\n" );

   for( int i = 1; i != (int) M.entries.size(); ++i )
   {
      int left = Node::left( M.entries[i] );
      int right = Node::right( M.entries[i] );

      if( ( left != 0 && Node::priority( M.entries[i] ) <
                             Node::priority( M.entries[left] ) ) ||
          ( right != 0 && Node::priority( M.entries[i] ) <
                              Node::priority( M.entries[right] ) ) )
      {
         fmt::print( "heap property violated: node {} (prio {}), left {} (prio "
                     "{}), right {} (prio {})\n",
                     i, Node::priority( M.entries[i] ), left,
                     Node::priority( M.entries[left] ), right,
                     Node::priority( M.entries[right] ) );
      }
   }

   return true;
}

template <bool StorageOrder>
bool
checkBstProperty( MatrixBuffer<double>& M )
{
   using Node = GetNodeProperty<StorageOrder>;

   for( int i = 1; i != (int) M.entries.size(); ++i )
   {
      int left = Node::left( M.entries[i] );
      int right = Node::right( M.entries[i] );

      if( left != 0 && Node::lesser( M.entries[i], M.entries[left] ) )
         return false;
      if( right != 0 && Node::lesser( M.entries[right], M.entries[i] ) )
         return false;
   }

   return true;
}

TEST_CASE( "matrix-buffer", "[core]" )
{
   MatrixBuffer<double> M;

   // add entries of  the following matrix:
   // 1  2  0  0  0  0  0  0  0
   // 0  3  4  5  6  7  0  0  0
   // 0  8  0  0  0  0  0  0  0
   // 0  0  0  0  0  0  0  0  0
   // 9  10 11 0  0  0  0  12 13

   Vec<double> colmajor{1.0,  9.0, 2.0, 3.0, 8.0,  10.0, 4.0,
                        11.0, 5.0, 6.0, 7.0, 12.0, 13.0};

   M.addEntry( 4, 8, 13.0 );
   REQUIRE( checkBstProperty<true>( M ) );
   REQUIRE( checkBstProperty<false>( M ) );
   M.addEntry( 0, 0, 1.0 );
   REQUIRE( checkBstProperty<true>( M ) );
   REQUIRE( checkBstProperty<false>( M ) );
   M.addEntry( 1, 2, 4.0 );
   REQUIRE( checkBstProperty<true>( M ) );
   REQUIRE( checkBstProperty<false>( M ) );
   M.addEntry( 2, 1, 8.0 );
   REQUIRE( checkBstProperty<true>( M ) );
   REQUIRE( checkBstProperty<false>( M ) );
   M.addEntry( 4, 7, 12.0 );
   REQUIRE( checkBstProperty<true>( M ) );
   REQUIRE( checkBstProperty<false>( M ) );
   M.addEntry( 4, 0, 9.0 );
   REQUIRE( checkBstProperty<true>( M ) );
   REQUIRE( checkBstProperty<false>( M ) );
   M.addEntry( 4, 2, 11.0 );
   REQUIRE( checkBstProperty<true>( M ) );
   REQUIRE( checkBstProperty<false>( M ) );
   M.addEntry( 1, 5, 7.0 );
   REQUIRE( checkBstProperty<true>( M ) );
   REQUIRE( checkBstProperty<false>( M ) );
   M.addEntry( 0, 1, 2.0 );
   REQUIRE( checkBstProperty<true>( M ) );
   REQUIRE( checkBstProperty<false>( M ) );
   M.addEntry( 4, 1, 10.0 );
   REQUIRE( checkBstProperty<true>( M ) );
   REQUIRE( checkBstProperty<false>( M ) );
   M.addEntry( 1, 4, 6.0 );
   REQUIRE( checkBstProperty<true>( M ) );
   REQUIRE( checkBstProperty<false>( M ) );
   M.addEntry( 1, 1, 3.0 );
   REQUIRE( checkBstProperty<true>( M ) );
   REQUIRE( checkBstProperty<false>( M ) );
   M.addEntry( 1, 3, 5.0 );

   SmallVec<int, 32> stack;

   const MatrixEntry<double>* it = M.begin<true>( stack );
   REQUIRE( checkHeapProperty<true>( M ) );
   REQUIRE( checkHeapProperty<false>( M ) );
   REQUIRE( checkBstProperty<true>( M ) );
   REQUIRE( checkBstProperty<false>( M ) );

   int i = 1;

   while( it != M.end() )
   {
      REQUIRE( it->val == double( i ) );
      fmt::print( "{} at depth: {}\n", it->val, stack.size() - 1 );

      it = M.next<true>( stack );
      ++i;
   }

   REQUIRE( stack.size() == 1 );

   it = M.begin<false>( stack );
   i = 0;

   while( it != M.end() )
   {
      REQUIRE( it->val == colmajor[i] );
      fmt::print( "{} at depth: {}\n", it->val, stack.size() - 1 );

      ++i;
      it = M.next<false>( stack );
   }

   REQUIRE( stack.size() == 1 );
}
