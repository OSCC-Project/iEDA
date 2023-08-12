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

#ifndef _PAPILO_CORE_REDUCTIONS_HPP_
#define _PAPILO_CORE_REDUCTIONS_HPP_

#include "papilo/misc/Vec.hpp"
#include <cassert>

namespace papilo
{

struct ColReduction
{
   enum
   {
      NONE = -1,
      OBJECTIVE = -2,
      LOWER_BOUND = -3,
      UPPER_BOUND = -4,
      FIXED = -5,
      LOCKED = -6,
      SUBSTITUTE = -8,
      BOUNDS_LOCKED = -9,
      REPLACE = -10,
      SUBSTITUTE_OBJ = -11,
      PARALLEL = -12,
      IMPL_INT = -13,
      FIXED_INFINITY = -14,
   };
};

struct RowReduction
{
   enum
   {
      NONE = -1,
      RHS = -2,
      LHS = -3,
      REDUNDANT = -4,
      LOCKED = -5,
      RHS_INF = -7,
      LHS_INF = -8,
      SPARSIFY = -9,
      RHS_LESS_RESTRICTIVE = -10,
      LHS_LESS_RESTRICTIVE = -11,
      REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE = -12,
      SAVE_ROW = -13
   };
};

template <typename REAL>
struct Reduction
{
   /// value stored in reduction. Meaning depends on the operation
   REAL newval;

   /// index of row or negative for column specific operations
   int row;

   /// index of column or negative for row specific operations
   int col;

   Reduction( REAL _newval, int _row, int _col )
       : newval( _newval ), row( _row ), col( _col )
   {
   }
};

template <typename REAL>
class Reductions
{
 public:
   void
   startTransaction()
   {
      assert( transactions.empty() || transactions.back().end >= 0 );

      const int start = static_cast<int>( reductions.size() );
      transactions.emplace_back( start, -1 );
   }

   void
   endTransaction()
   {
      assert( !transactions.empty() && transactions.back().end == -1 );

      const int end = static_cast<int>( reductions.size() );
      assert( end != transactions.back().start );
      transactions.back().end = end;
   }

   void
   add_reduction( int row, int col, REAL newval )
   {
      reductions.emplace_back( newval, row, col );
   }

   void
   changeMatrixEntry( int row, int col, REAL newval )
   {
      assert( row >= 0 && col >= 0 );
      reductions.emplace_back( newval, row, col );
   }

   void
   changeRowLHS( int row, REAL newval )
   {
      reductions.emplace_back( newval, row, RowReduction::LHS );
   }

   void
   change_row_lhs_parallel( int row, REAL newval )
   {
      reductions.emplace_back( newval, row, RowReduction::LHS_LESS_RESTRICTIVE );
   }

   void
   changeRowRHS( int row, REAL newval )
   {
      reductions.emplace_back( newval, row, RowReduction::RHS );
   }

   void
   change_row_rhs_parallel( int row, REAL newval )
   {
      reductions.emplace_back( newval, row, RowReduction::RHS_LESS_RESTRICTIVE );
   }

   void
   bound_change_caused_by_row( int remaining_row, int deleted_row )
   {
      reductions.emplace_back(
          remaining_row, deleted_row,
          RowReduction::REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE );
   }

   void
   changeRowLHSInf( int row )
   {
      reductions.emplace_back( 0.0, row, RowReduction::LHS_INF );
   }

   void
   changeRowRHSInf( int row )
   {
      reductions.emplace_back( 0.0, row, RowReduction::RHS_INF );
   }

   void
   markRowRedundant( int row )
   {
      reductions.emplace_back( REAL{ 0.0 }, row, RowReduction::REDUNDANT );
   }

   /// lock row, i.e. modifications that come before this transaction are
   /// conflicting but not modifications that come after this transaction
   void
   lockRow( int row )
   {
      // locks are only valid inside a transaction
      assert( !transactions.empty() && transactions.back().end == -1 );
      // locks must come first within a transaction
      assert( transactions.back().start + transactions.back().nlocks ==
              static_cast<int>( reductions.size() ) );

      reductions.emplace_back( 0.0, row, RowReduction::LOCKED );
      ++transactions.back().nlocks;
   }

   void
   changeColLB( int col, REAL new_val, int forcing_row = -1 )
   {
      if(forcing_row > -1)
         reductions.emplace_back(0, forcing_row, RowReduction::SAVE_ROW);
      reductions.emplace_back( new_val, ColReduction::LOWER_BOUND, col );
   }

   void
   changeColUB( int col, REAL new_val, int forcing_row = -1 )
   {
      if( forcing_row > -1)
         reductions.emplace_back(0, forcing_row, RowReduction::SAVE_ROW);
      reductions.emplace_back( new_val, ColReduction::UPPER_BOUND, col );
   }

   void
   fixCol( int col, REAL val, int forcing_row = -1 )
   {
      if(forcing_row > -1)
         reductions.emplace_back(0, forcing_row, RowReduction::SAVE_ROW);
      reductions.emplace_back( val, ColReduction::FIXED, col );
   }

   void
   fixColPositiveInfinity( int col, int columnLength, const int* rowIndices )
   {
      for( int i = 0; i < columnLength; i++ )
         markRowRedundant( rowIndices[i] );

      reductions.emplace_back( 1, ColReduction::FIXED_INFINITY, col );
   }

   void
   fixColNegativeInfinity( int col, int columnLength, const int* rowIndices )
   {
      for( int i = 0; i < columnLength; i++ )
         markRowRedundant( rowIndices[i] );

      reductions.emplace_back( -1, ColReduction::FIXED_INFINITY, col );
   }

   /// lock column, i.e. modifications that come before this transaction are
   /// conflicting but not modifications that come after this transaction
   void
   lockCol( int col )
   {
      assert( !transactions.empty() && transactions.back().end == -1 );
      assert( transactions.back().start + transactions.back().nlocks ==
              static_cast<int>( reductions.size() ) );

      reductions.emplace_back( 0.0, ColReduction::LOCKED, col );
      ++transactions.back().nlocks;
   }

   /// lock column lower and upper bounds
   void
   lockColBounds( int col )
   {
      assert( !transactions.empty() && transactions.back().end == -1 );
      assert( transactions.back().start + transactions.back().nlocks ==
              static_cast<int>( reductions.size() ) );

      reductions.emplace_back( 0.0, ColReduction::BOUNDS_LOCKED, col );
      ++transactions.back().nlocks;
   }

   /// signal that a column in free and can be substituted in the matrix
   void
   aggregateFreeCol( int col, int equalityRow )
   {
      assert( col >= 0 && equalityRow >= 0 );
      reductions.emplace_back( static_cast<REAL>( equalityRow ),
                               ColReduction::SUBSTITUTE, col );
   }

   /// signal that a column in free and can be substituted in the matrix
   void
   substituteColInObjective( int col, int equalityRow )
   {
      assert( col >= 0 && equalityRow >= 0 );
      reductions.emplace_back( static_cast<REAL>( equalityRow ),
                               ColReduction::SUBSTITUTE_OBJ, col );
   }

   // replace col1 = factor * col2 + offset
   void
   replaceCol( int col1, int col2, REAL factor, REAL offset )
   {
      assert( col1 >= 0 && col2 >= 0 );

      startTransaction();
      reductions.emplace_back( factor, ColReduction::REPLACE, col1 );
      reductions.emplace_back( offset, ColReduction::NONE, col2 );
      endTransaction();
   }

   /// parallel columns col1 and col2 must satisfies all conditions so
   /// that they can be substituted by a new variable y = col2 + factor * col1
   /// where factor is computed by using the ratio between the two
   /// columns coefficients
   void
   mark_parallel_cols( int col1, int col2 )
   {
      assert( col1 >= 0 && col2 >= 0 );
      reductions.emplace_back( static_cast<REAL>( col2 ),
                               ColReduction::PARALLEL, col1 );
   }

   void
   impliedInteger( int col )
   {
      assert( col >= 0 );
      reductions.emplace_back( 0, ColReduction::IMPL_INT, col );
   }

   void
   sparsify( int eq, int numrows, const std::pair<int, REAL>* sparsifiedrows )
   {
      reductions.emplace_back( static_cast<REAL>( numrows ), eq,
                               RowReduction::SPARSIFY );
      for( int i = 0; i != numrows; ++i )
         reductions.emplace_back( sparsifiedrows[i].second,
                                  sparsifiedrows[i].first, RowReduction::NONE );
   }

   unsigned int
   size()
   {
      return reductions.size();
   }

   void
   clear()
   {
      reductions.clear();
      transactions.clear();
   }

   const Vec<Reduction<REAL>>&
   getReductions() const
   {
      return reductions;
   }

   struct Transaction
   {
      int start;
      int end;
      int nlocks;
      int naddcoeffs;

      Transaction( int start_, int end_ )
          : start( start_ ), end( end_ ), nlocks( 0 ), naddcoeffs( 0 )
      {
      }
   };

   const Vec<Transaction>&
   getTransactions() const
   {
      return transactions;
   }

 private:
   Vec<Reduction<REAL>> reductions;
   Vec<Transaction> transactions;

 public:
   Reduction<REAL>&
   getReduction( int i )
   {
      return reductions[i];
   }
};

template <typename REAL>
class TransactionGuard
{
 public:
   TransactionGuard( Reductions<REAL>& _reductions ) : reductions( _reductions )
   {
      _reductions.startTransaction();
   }

   ~TransactionGuard() { reductions.endTransaction(); }

 private:
   Reductions<REAL>& reductions;
};

} // namespace papilo

#endif
