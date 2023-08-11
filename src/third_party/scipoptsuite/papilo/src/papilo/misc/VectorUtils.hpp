
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

#ifndef _PAPILO_MISC_VECTOR_UTILS_HPP_
#define _PAPILO_MISC_VECTOR_UTILS_HPP_

#include "papilo/core/Problem.hpp"
#include "papilo/core/RowFlags.hpp"
#include "papilo/core/Solution.hpp"
#include "papilo/core/SparseStorage.hpp"
#include "papilo/io/Message.hpp"
#include "papilo/misc/Num.hpp"
#include "papilo/misc/Vec.hpp"
#include "papilo/misc/fmt.hpp"
#include <cassert>

namespace papilo
{

template <typename REAL>
bool
compareVectors( const Vec<REAL>& first, const Vec<REAL>& second,
                const Num<REAL>& num )
{
   bool result = std::equal( first.begin(), first.end(), second.begin(),
                             [&num]( const REAL& left, const REAL& right ) {
                                return num.isEq( left, right );
                             } );
   return result;
}

template <typename REAL>
bool
compareVariableDomains( const VariableDomains<REAL>& first,
                        const VariableDomains<REAL>& second,
                        const Num<REAL>& num )
{
   bool result = std::equal( first.begin(), first.end(), second.begin(),
                             [&num]( const REAL& left, const REAL& right ) {
                                return num.isEq( left, right );
                             } );
   return result;
}

template <typename REAL>
bool
compareColBounds( const Vec<REAL>& first_values, const Vec<REAL>& second_values,
                  const Vec<ColFlags>& first_flags,
                  const Vec<ColFlags>& second_flags, const Num<REAL>& num )
{
   int size = (int) first_values.size();
   if( size != (int) first_flags.size() )
      return false;
   if( size != (int) second_values.size() )
      return false;
   if( size != (int) second_flags.size() )
      return false;

   for( int i = 0; i < size; i++ )
   {
      if( ( first_flags[i].test( ColFlag::kLbInf ) !=
            second_flags[i].test( ColFlag::kLbInf ) ) ||
          ( first_flags[i].test( ColFlag::kUbInf ) !=
            second_flags[i].test( ColFlag::kUbInf ) ) )
         return false;

      if( !num.isEq( first_values[i], second_values[i] ) )
         return false;
   }

   return true;
}

template <typename REAL>
bool
compareRowBounds( const Vec<REAL>& first_values, const Vec<REAL>& second_values,
                  const Vec<RowFlags>& first_flags,
                  const Vec<RowFlags>& second_flags, const Num<REAL>& num )
{
   int size = (int) first_values.size();
   if( size != (int) first_flags.size() )
      return false;
   if( size != (int) second_values.size() )
      return false;
   if( size != (int) second_flags.size() )
      return false;

   for( int i = 0; i < size; i++ )
   {
      if( ( first_flags[i].test( RowFlag::kLhsInf ) !=
            second_flags[i].test( RowFlag::kLhsInf ) ) ||
          ( first_flags[i].test( RowFlag::kRhsInf ) !=
            second_flags[i].test( RowFlag::kRhsInf ) ) )
         return false;

      if( !num.isEq( first_values[i], second_values[i] ) )
         return false;
   }

   return true;
}

// compare index range
template <typename IndexRange>
bool
compareIndexRanges( const Vec<IndexRange>& first,
                    const Vec<IndexRange>& second )
{
   bool result = std::equal(
       first.begin(), first.end(), second.begin(),
       []( const IndexRange& left, const IndexRange& right ) {
          return ( left.start == right.start && left.end == right.end );
       } );

   return result;
}

// compare matrix values
template <typename REAL>
bool
compareMatrices( const SparseStorage<REAL>& first,
                 const SparseStorage<REAL>& second, const Num<REAL>& num )
{
   if( first.getNRows() != second.getNRows() )
      return false;
   if( first.getNCols() != second.getNCols() )
      return false;
   if( first.getNnz() != second.getNnz() )
      return false;

   // ranges
   if( !compareIndexRanges( first.getRowRangesVec(),
                            second.getRowRangesVec() ) )
      return false;

   // columns, values
   if( !std::equal( first.getColumnsVec().begin(), first.getColumnsVec().end(),
                    second.getColumnsVec().begin() ) )
      return false;
   if( !compareVectors( first.getValuesVec(), second.getValuesVec(), num ) )
      return false;

   return true;
}

template <typename REAL>
bool
compareMatrixToTranspose( const SparseStorage<REAL>& first,
                          const SparseStorage<REAL>& second,
                          const Num<REAL>& num )
{
   const SparseStorage<REAL> transpose = first.getTranspose();
   bool result = compareMatrices( second, transpose, num );

   return result;
}

template <typename REAL>
bool
compareMatrixToTranspose( const ConstraintMatrix<REAL>& constarint_matrix,
                          const Num<REAL>& num )
{
   const SparseStorage<REAL>& matrix = constarint_matrix.getConstraintMatrix();
   const SparseStorage<REAL>& transpose =
       constarint_matrix.getMatrixTranspose();
   return compareMatrixToTranspose( matrix, transpose, num );
}

template <typename REAL>
bool
compareProblems( const Problem<REAL>& first, const Problem<REAL>& second,
                 const Num<REAL>& num )
{
   // ncols, nrows
   int nRows1 = first.getNRows();
   int nRows2 = second.getNRows();
   const int nCols1 = first.getNCols();
   const int nCols2 = second.getNCols();

   if( nRows1 != nRows2 )
      return false;
   if( nCols1 != nCols2 )
      return false;

   // objective
   const Vec<REAL>& objective1 = first.getObjective().coefficients;
   const Vec<REAL>& objective2 = second.getObjective().coefficients;
   bool result = compareVectors( objective1, objective2, num );
   if( !result )
      return false;

   // lhs, rhs
   result = compareColBounds( first.getLowerBounds(), second.getLowerBounds(),
                              first.getColFlags(), second.getColFlags(), num );
   if( !result )
      return false;
   result = compareColBounds( first.getUpperBounds(), second.getUpperBounds(),
                              first.getColFlags(), second.getColFlags(), num );
   if( !result )
      return false;

   result = compareRowBounds( first.getConstraintMatrix().getLeftHandSides(),
                              second.getConstraintMatrix().getLeftHandSides(),
                              first.getConstraintMatrix().getRowFlags(),
                              second.getConstraintMatrix().getRowFlags(), num );
   if( !result )
      return false;

   // matrix rowwise
   result = compareMatrices( first.getConstraintMatrix().getConstraintMatrix(),
                             second.getConstraintMatrix().getConstraintMatrix(),
                             num );
   if( !result )
      return false;

   // matrix colwise (transpose)
   result = compareMatrices( first.getConstraintMatrix().getMatrixTranspose(),
                             second.getConstraintMatrix().getMatrixTranspose(),
                             num );
   if( !result )
      return false;

   // matrix to its own transpose
   result = compareMatrixToTranspose(
       first.getConstraintMatrix().getConstraintMatrix(),
       second.getConstraintMatrix().getMatrixTranspose(), num );

   // objective offset
   if( first.getObjective().offset != second.getObjective().offset )
      return false;

   return true;
}

} // namespace papilo

#endif
