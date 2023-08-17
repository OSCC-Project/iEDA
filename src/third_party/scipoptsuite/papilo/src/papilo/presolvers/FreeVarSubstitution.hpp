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

#ifndef _PAPILO_PRESOLVERS_FREEVAR_HPP_
#define _PAPILO_PRESOLVERS_FREEVAR_HPP_

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"
#include "papilo/misc/Num.hpp"
#include "papilo/misc/fmt.hpp"
#include "papilo/external/pdqsort/pdqsort.h"
#include <boost/dynamic_bitset.hpp>

namespace papilo
{

template <typename REAL>
class Substitution : public PresolveMethod<REAL>
{
   Vec<int> ntried;

 public:
   Substitution() : PresolveMethod<REAL>()
   {
      this->setName( "substitution" );
      this->setTiming( PresolverTiming::kExhaustive );
   }

   void
   compress( const Vec<int>& rowmap, const Vec<int>& colmap ) override
   {
      assert( rowmap.size() == ntried.size() );
      compress_vector( rowmap, ntried );
      Message::debug( this,
                      "compress was called, compressed ntried vector from "
                      "size {} to size {}\n",
                      rowmap.size(), ntried.size() );
   }

   bool
   initialize( const Problem<REAL>& problem,
               const PresolveOptions& presolveOptions ) override
   {
      ntried.clear();
      ntried.resize( problem.getNRows(), 0 );

      Message::debug( this, "initialized ntried vector to size {}\n",
                      ntried.size() );

      return true;
   }

   PresolveStatus
   execute( const Problem<REAL>& problem,
            const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
            Reductions<REAL>& reductions, const Timer& timer ) override;
   bool
   is_divisible( const Num<REAL>& num, int length, const REAL* row_values,
                 REAL min_abs_int_value ) const;
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class Substitution<double>;
extern template class Substitution<Quad>;
extern template class Substitution<Rational>;
#endif

template <typename REAL>
PresolveStatus
Substitution<REAL>::execute( const Problem<REAL>& problem,
                             const ProblemUpdate<REAL>& problemUpdate,
                             const Num<REAL>& num,
                             Reductions<REAL>& reductions, const Timer& timer )
{
   // go over the rows and get the equalities, extract the columns that
   // verify the conditions add them to a hash map, loop over the hash map
   // and compute the implied bounds and finally look for implied free
   // variables and add reductions
   const auto& domains = problem.getVariableDomains();
   const auto& lower_bounds = domains.lower_bounds;
   const auto& upper_bounds = domains.upper_bounds;
   const auto& cflags = domains.flags;

   const auto& activities = problem.getRowActivities();

   const auto& constMatrix = problem.getConstraintMatrix();
   const auto& lhs_values = constMatrix.getLeftHandSides();
   const auto& rhs_values = constMatrix.getRightHandSides();
   const auto& rflags = constMatrix.getRowFlags();
   const auto& nrows = constMatrix.getNRows();
   const auto& ncols = constMatrix.getNCols();
   const auto& rowperm = problemUpdate.getRandomRowPerm();

   PresolveStatus result = PresolveStatus::kUnchanged;

   using Equality = std::tuple<SparseVectorView<REAL>, int>;
   Vec<Equality> equalities;
   equalities.reserve( nrows );
   for( int i = 0; i < nrows; ++i )
   {
      if( rflags[i].test( RowFlag::kRedundant ) ||
          !rflags[i].test( RowFlag::kEquation ) ||
          constMatrix.getRowSizes()[i] <= 1 )
         continue;

      assert( !rflags[i].test( RowFlag::kLhsInf, RowFlag::kRhsInf ) &&
              lhs_values[i] == rhs_values[i] );

      equalities.emplace_back( constMatrix.getRowCoefficients( i ), i );
   }

   pdqsort( equalities.begin(), equalities.end(),
            [this, &rowperm]( const Equality& a, const Equality& b ) {
               return std::make_tuple( ntried[std::get<1>( a )],
                                       std::get<0>( a ).getLength(),
                                       rowperm[std::get<1>( a )] ) <
                      std::make_tuple( ntried[std::get<1>( b )],
                                       std::get<0>( b ).getLength(),
                                       rowperm[std::get<1>( b )] );
            } );

   boost::dynamic_bitset<> colUnusable( ncols );
   boost::dynamic_bitset<> touchedRows( nrows );
   Vec<int> column_candidates;
   column_candidates.reserve( ncols );

   for( auto equality : equalities )
   {
      int row = std::get<1>( equality );
      const int length = std::get<0>( equality ).getLength();
      const int* rowindices = std::get<0>( equality ).getIndices();
      const REAL* rowvalues = std::get<0>( equality ).getValues();
      REAL maxabsvalue = std::get<0>( equality ).getMaxAbsValue();
      REAL minabsintvalue = maxabsvalue;
      bool containsContinuous = false;
      bool containsNonBinInt = false;
      for( int i = 0; i != length; ++i )
      {
         if( !cflags[rowindices[i]].test( ColFlag::kIntegral ) )
         {
            containsContinuous = true;
            break;
         }

         if( !problemUpdate.getPresolveOptions().substitutebinarieswithints &&
             !domains.isBinary( rowindices[i] ) )
            containsNonBinInt = true;

         if( abs( rowvalues[i] ) < minabsintvalue )
            minabsintvalue = abs( rowvalues[i] );
      }

      // If the row contains no continuous variables it might be suitable for
      // substituting an integer variable. We can only substitute those
      // integer variables whose absolute coefficient value in the row is
      // equal to the smallest absolute coefficient in the row. Additional we
      // need to ensure that all coefficients are integral if divided by the
      // smallest absolute coefficient.
      if( !containsContinuous &&
          !is_divisible( num, length, rowvalues, minabsintvalue ) )
         continue;

      column_candidates.clear();

      for( int i = 0; i < length; ++i )
      {
         int col = rowindices[i];
         // if the column has been already used for a substitution or known
         // to be not implied free we can skip it
         if( colUnusable[col] )
            continue;

         // check if integrality condition is guaranteed for this column
         if( cflags[col].test( ColFlag::kIntegral ) )
         {
            // if the row contains continuous variables we cannot use it for
            // substituting an integral column
            if( containsContinuous )
               continue;

            // TODO: why not saving the index of the value also? this would
            // reduce the if statement to just comparing indices

            // the divisibility of the coefficients has been checked
            // above and we know that we can use it only if its coefficient has
            // the smallest magnitude of the coefficients within the row
            if( !num.isEq( abs( rowvalues[i] ), minabsintvalue ) )
               continue;

            // do not substitute a binary variable with integer variables if the
            // option is set
            if( !problemUpdate.getPresolveOptions()
                     .substitutebinarieswithints &&
                containsNonBinInt && domains.isBinary( rowindices[i] ) )
               continue;
         }

         column_candidates.push_back( i );
      }

      pdqsort( column_candidates.begin(), column_candidates.end(),
               [&]( int i1, int i2 ) {
                  int col1 = rowindices[i1];
                  int col2 = rowindices[i2];
                  return problemUpdate.isColBetterForSubstitution( col1, col2 );
               } );

      for( int i : column_candidates )
      {
         int col = rowindices[i];
         auto colvec = constMatrix.getColumnCoefficients( col );

         // check the numerics conditions
         if( ( abs( rowvalues[i] ) <
                   REAL(problemUpdate.getPresolveOptions().markowitz_tolerance) *
                       maxabsvalue &&
               abs( rowvalues[i] ) <
                   REAL(problemUpdate.getPresolveOptions().markowitz_tolerance) *
                       colvec.getMaxAbsValue() ) )
            continue;

         int lbrowlock = -1;
         int ubrowlock = -1;

         // mark the column to not be used in later substitutions, because
         // it's either not free or used for this substitution
         colUnusable[col] = true;

         bool upperboundImplied =
             row_implies_UB( num, lhs_values[row], rhs_values[row], rflags[row],
                             activities[row], rowvalues[i], lower_bounds[col],
                             upper_bounds[col], cflags[col] );
         bool lowerboundImplied =
             row_implies_LB( num, lhs_values[row], rhs_values[row], rflags[row],
                             activities[row], rowvalues[i], lower_bounds[col],
                             upper_bounds[col], cflags[col] );

         const REAL* colvalues = colvec.getValues();
         const int* colindices = colvec.getIndices();
         const int collength = colvec.getLength();

         auto checkIfImpliedFree = [&]( bool checkTouchedRows ) {
            int j = 0;

            while( ( !upperboundImplied || !lowerboundImplied ) &&
                   j != collength )
            {
               int colrow = colindices[j];
               REAL colval = colvalues[j];
               ++j;

               if( colrow == row )
                  continue;

               bool isTouched = touchedRows.test( colrow );

               if( ( !checkTouchedRows && isTouched ) ||
                   ( checkTouchedRows && !isTouched ) )
                  continue;

               if( !upperboundImplied )
               {
                  upperboundImplied = row_implies_UB(
                      num, lhs_values[colrow], rhs_values[colrow],
                      rflags[colrow], activities[colrow], colval,
                      lower_bounds[col], upper_bounds[col], cflags[col] );

                  if( upperboundImplied && colrow != row &&
                      colrow != lbrowlock )
                     ubrowlock = colrow;
               }

               if( !lowerboundImplied )
               {
                  lowerboundImplied = row_implies_LB(
                      num, lhs_values[colrow], rhs_values[colrow],
                      rflags[colrow], activities[colrow], colval,
                      lower_bounds[col], upper_bounds[col], cflags[col] );

                  if( lowerboundImplied && colrow != row &&
                      colrow != ubrowlock )
                     lbrowlock = colrow;
               }
            }
         };

         checkIfImpliedFree( false );
         checkIfImpliedFree( true );

         if( upperboundImplied && lowerboundImplied )
         {
            // reductions
            result = PresolveStatus::kReduced;
            ++ntried[row];

            TransactionGuard<REAL> guard{ reductions };

            reductions.lockRow( row );
            if( lbrowlock != -1 )
               reductions.lockRow( lbrowlock );
            if( ubrowlock != -1 )
               reductions.lockRow( ubrowlock );

            reductions.lockColBounds( col );

            reductions.aggregateFreeCol( col, row );
            const int* indices = colvec.getIndices();
            const int len = colvec.getLength();

            for( int j = 0; j != len; ++j )
               touchedRows.set( indices[j] );

            break;
         }
      }
   }

   return result;
}

template <typename REAL>
bool
Substitution<REAL>::is_divisible( const Num<REAL>& num, const int length,
                                  const REAL* row_values,
                                  REAL min_abs_int_value ) const
{
   for( int i = 0; i != length; ++i )
   {
      if( !num.isIntegral( row_values[i] / min_abs_int_value ) )
      {
         return false;
      }
   }
   return true;
}

} // namespace papilo

#endif
