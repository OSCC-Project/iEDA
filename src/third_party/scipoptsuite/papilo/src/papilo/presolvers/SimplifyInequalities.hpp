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

#ifndef _PAPILO_PRESOLVERS_GCD_REDUCTIONS_HPP_
#define _PAPILO_PRESOLVERS_GCD_REDUCTIONS_HPP_

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"
#include "papilo/external/pdqsort/pdqsort.h"
#include <boost/integer/common_factor.hpp>

namespace papilo
{
/**
 * Simplify Inequalities removes the "unneccessary" variables in a constraint:
 * Example:
 * 15x1 +15x2 +7x3 +3x4 +y1 <= 26
 * <=> 15x1 +15x2 <= 26  # delete variables
 * <=> x1 +x2 <=1  # divide by greatestCommonDivisor and round right side down
 *
 * if this is not possible, then the GCD is calculated, so this is used to
 * reduce rhs/lhs -> (floor[rhs/gcd]* gcd )
 * Example
 * 15x1 +15x2 +10x3 +5x4 <= 18 -> 18 can be reduced to 15
 * @tparam REAL
 */
template <typename REAL>
class SimplifyInequalities : public PresolveMethod<REAL>
{

 public:
   SimplifyInequalities() : PresolveMethod<REAL>()
   {
      this->setName( "simplifyineq" );
      this->setTiming( PresolverTiming::kMedium );
      this->setType( PresolverType::kIntegralCols );
   }

   PresolveStatus
   execute( const Problem<REAL>& problem,
            const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
            Reductions<REAL>& reductions, const Timer& timer ) override;

 private:
   REAL
   computeGreatestCommonDivisor( REAL val1, REAL val2, const Num<REAL>& num );

   void
   simplify( const REAL* values, const int* colinds, int rowLength,
             const RowActivity<REAL>& activity, const RowFlags& rflag,
             const Vec<ColFlags>& cflags, const REAL& rhs, const REAL& lhs,
             const Vec<REAL>& lbs, const Vec<REAL>& ubs, Vec<int>& colOrder,
             Vec<int>& coeffDelete, REAL& gcd, bool& change,
             const Num<REAL>& num );

   bool
   isUnbounded( int row, const Vec<RowFlags>& rowFlags ) const;

   bool
   isRedundant( int row, const Vec<RowFlags>& rflags ) const;

   bool
   isInfiniteActivity( const Vec<RowActivity<REAL>>& activities,
                       int row ) const;

   PresolveStatus
   perform_simplify_ineq_task(
       const Num<REAL>& num, const ConstraintMatrix<REAL>& consMatrix,
       const Vec<RowActivity<REAL>>& activities, const Vec<RowFlags>& rflags,
       const Vec<ColFlags>& cflags, const Vec<REAL>& lhs, const Vec<REAL>& rhs,
       const Vec<REAL>& lbs, const Vec<REAL>& ubs, int row,
       Reductions<REAL>& reductions, Vec<int>& coefficientsThatCanBeDeleted,
       Vec<int>& colOrder );
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class SimplifyInequalities<double>;
extern template class SimplifyInequalities<Quad>;
extern template class SimplifyInequalities<Rational>;
#endif

/***
 * to calculate the Greatest Common Divisor heuristics are used according to
 * "Presolve Reductions in Mixed Integer Programming" from T. Achterberg et. al.
 *
 * - Euclidian algorithm for integral values (numerical issues for flaoting
 * point)
 *
 * 1. Divide all coefficients by a_min = min{|a_ij| j in supp(A_i)}. If this
 * leads to integer values for all coefficients return
 *  d= a_min * gcd(a_i1/a_min,..., a_in /a_min)
 *
 * 2. Use a_min = 1/600 (multiply by 600), because it is a multiple of many
 * small integer values that arise as denominators in real-world problems
 *
 * @tparam REAL
 * @param val1
 * @param val2
 * @param num
 * @return gcd (with heuristics for floating points)
 */
template <typename REAL>
REAL
SimplifyInequalities<REAL>::computeGreatestCommonDivisor( REAL val1, REAL val2,
                                                          const Num<REAL>& num )
{
   auto is_int64_castable = [&num]( REAL val )
   {
      return num.isIntegral( val ) && static_cast<int64_t>( val ) == val;
   };

   if( num.isZero( val1 ) || num.isZero( val2 ) )
      return 0;

   // gcd for integer values
   if( is_int64_castable( val1 ) && is_int64_castable( val2 ) )
   {
#ifndef BOOST_VERSION_NUMBER_PATCH
      return boost::gcd( static_cast<int64_t>( val1 ),
                         static_cast<int64_t>( val2 ) );
#elif BOOST_VERSION_NUMBER_PATCH( BOOST_VERSION ) / 100 < 78
      return boost::gcd( static_cast<int64_t>( val1 ),
                         static_cast<int64_t>( val2 ) );
#else
      return boost::integer::gcd( static_cast<int64_t>( val1 ),
                                  static_cast<int64_t>( val2 ) );
#endif
   }

   // heuristic for fractional values
   // if max(abs(val1), abs(val2)) divided by d:=min(abs(val1), abs(val2)) is
   // integral, return d
   if( abs( val2 ) < abs( val1 ) )
   {
      if( is_int64_castable( val1 / val2 ) )
         return abs( val2 );
   }
   else
   {
      if( is_int64_castable( val2 / val1 ) )
         return abs( val1 );
   }

   double multiplier = 600;
   if( is_int64_castable( multiplier * val1 ) &&
       is_int64_castable( multiplier * val2 ) )
#ifndef BOOST_VERSION_NUMBER_PATCH
      return boost::gcd( static_cast<int64_t>( val1 * multiplier ),
                         static_cast<int64_t>( val2 * multiplier ) ) /
             REAL{ multiplier };
#elif BOOST_VERSION_NUMBER_PATCH( BOOST_VERSION ) / 100 < 78
      return boost::gcd( static_cast<int64_t>( val1 * multiplier ),
                         static_cast<int64_t>( val2 * multiplier ) ) /
             REAL{ multiplier };
#else
      return boost::integer::gcd( static_cast<int64_t>( val1 * multiplier ),
                                  static_cast<int64_t>( val2 * multiplier ) ) /
             REAL{ multiplier };
#endif

   // applied heuristics didn't find an greatest common divisor
   return 0;
}

template <typename REAL>
void
SimplifyInequalities<REAL>::simplify(
    const REAL* values, const int* colinds, int rowLength,
    const RowActivity<REAL>& activity, const RowFlags& rflag,
    const Vec<ColFlags>& cflags, const REAL& rhs, const REAL& lhs,
    const Vec<REAL>& lbs, const Vec<REAL>& ubs, Vec<int>& colOrder,
    Vec<int>& coeffDelete, REAL& gcd, bool& change, const Num<REAL>& num )
{
   auto maxActivity = activity.max;
   auto minActivity = activity.min;

   // sort the list 'colOrder' for integer/continuous and then for absolute
   // coefficient

   for( int i = 0; i < rowLength; ++i )
      colOrder.push_back( i );
   Vec<int>::iterator start_cont;
   start_cont =
       partition( colOrder.begin(), colOrder.end(),
                  [&colinds, &cflags]( int const& a )
                  { return cflags[colinds[a]].test( ColFlag::kIntegral ); } );
   pdqsort( colOrder.begin(), start_cont,
            [&values]( int const& a, int const& b )
            { return abs( values[a] ) > abs( values[b] ); } );

   // check if continuous variables or variables with small absolut value
   // always fit into the constraint
   REAL resmaxact = maxActivity;
   REAL resminact = minActivity;
   assert( num.isGE( resmaxact, resminact ) );

   // start value important for first variable
   gcd = values[colOrder[0]];
   assert( gcd != 0 );
   REAL siderest;
   bool redundant = false;
   // i is index of last non-redundant variable
   int i = 0;

   // iterate over ordered non-zero entries
   for( ; i != rowLength; ++i )
   {
      int column_index = colOrder[i];

      // break if variable not integral
      if( !cflags[colinds[column_index]].test( ColFlag::kIntegral ) )
         break;

      // update gcd
      gcd = computeGreatestCommonDivisor( gcd, values[column_index], num );
      if( num.isLE( gcd, 1 ) )
         break;

      assert( !cflags[colinds[column_index]].test( ColFlag::kLbInf,
                                                   ColFlag::kUbInf ) );

      // update residual activities
      // attention: the calculation inaccuracy can be greater than epsilon
      if( values[column_index] > 0 )
      {
         resmaxact -= values[column_index] * ubs[colinds[column_index]];
         resminact -= values[column_index] * lbs[colinds[column_index]];
      }
      else
      {
         resmaxact -= values[column_index] * lbs[colinds[column_index]];
         resminact -= values[column_index] * ubs[colinds[column_index]];
      }

      // calculate siderest
      if( !rflag.test( RowFlag::kRhsInf ) )
      {
         siderest = rhs - num.epsFloor( rhs / gcd ) * gcd;
      }
      else
      {
         siderest = lhs - num.epsFloor( lhs / gcd ) * gcd;
         if( num.isZero( siderest ) )
            siderest = gcd;
      }

      // check if the ordered variables on the right of i are redundant
      if( ( !rflag.test( RowFlag::kRhsInf ) && resmaxact <= siderest &&
            num.isFeasLT( siderest - gcd, resminact ) ) ||
          ( !rflag.test( RowFlag::kLhsInf ) && resminact >= siderest - gcd &&
            num.isFeasGT( siderest, resmaxact ) ) )
      {
         redundant = true;
         break;
      }
   }

   if( redundant )
   {
      change = true;
      // safe indices of redundant variables
      for( int w = i + 1; w < rowLength; ++w )
      {
         coeffDelete.push_back( colOrder[w] );
      }
   }
}

template <typename REAL>
PresolveStatus
SimplifyInequalities<REAL>::execute( const Problem<REAL>& problem,
                                     const ProblemUpdate<REAL>& problemUpdate,
                                     const Num<REAL>& num,
                                     Reductions<REAL>& reductions, const Timer& timer )
{
   const auto& consMatrix = problem.getConstraintMatrix();
   const Vec<RowActivity<REAL>>& activities = problem.getRowActivities();
   const Vec<RowFlags>& rflags = consMatrix.getRowFlags();
   const Vec<ColFlags>& cflags = problem.getColFlags();
   const Vec<REAL>& lhs = consMatrix.getLeftHandSides();
   const Vec<REAL>& rhs = consMatrix.getRightHandSides();
   const int nrows = consMatrix.getNRows();
   const Vec<REAL>& lbs = problem.getLowerBounds();
   const Vec<REAL>& ubs = problem.getUpperBounds();

   PresolveStatus result = PresolveStatus::kUnchanged;

#ifndef PAPILO_TBB
   assert( problemUpdate.getPresolveOptions().runs_sequential() );
#endif

   if( problemUpdate.getPresolveOptions().runs_sequential() ||
       !problemUpdate.getPresolveOptions().simplify_inequalities_parallel )
   {
      // allocate only once
      Vec<int> colOrder;
      Vec<int> coefficientsThatCanBeDeleted;
      for( int row = 0; row < nrows; row++ )
      {
         if( perform_simplify_ineq_task(
                 num, consMatrix, activities, rflags, cflags, lhs, rhs, lbs,
                 ubs, row, reductions, coefficientsThatCanBeDeleted,
                 colOrder ) == PresolveStatus::kReduced )
            result = PresolveStatus::kReduced;
      }
   }
#ifdef PAPILO_TBB
   else
   {
      Vec<Reductions<REAL>> stored_reductions( nrows );
      // iterate over all constraints and try to simplify them
      tbb::parallel_for(
          tbb::blocked_range<int>( 0, nrows ),
          [&]( const tbb::blocked_range<int>& r )
          {
             // allocate only once per thread
             Vec<int> colOrder;
             Vec<int> coefficientsThatCanBeDeleted;
             for( int row = r.begin(); row < r.end(); ++row )
             {
                PresolveStatus status = perform_simplify_ineq_task(
                    num, consMatrix, activities, rflags, cflags, lhs, rhs, lbs,
                    ubs, row, stored_reductions[row],
                    coefficientsThatCanBeDeleted, colOrder );
                if( status == PresolveStatus::kReduced )
                   result = PresolveStatus::kReduced;
                assert( status == PresolveStatus::kReduced ||
                        status == PresolveStatus::kUnchanged );
             }
          } );

      if( result == PresolveStatus::kUnchanged )
         return PresolveStatus::kUnchanged;

      for( int i = 0; i < (int)stored_reductions.size(); ++i )
      {
         Reductions<REAL> reds = stored_reductions[i];
         if( reds.size() > 0 )
         {
            for( const auto& transaction : reds.getTransactions() )
            {
               int start = transaction.start;
               int end = transaction.end;
               TransactionGuard<REAL> guard{ reductions };
               for( int c = start; c < end; c++ )
               {
                  Reduction<REAL>& reduction = reds.getReduction( c );
                  reductions.add_reduction( reduction.row, reduction.col,
                                            reduction.newval );
               }
            }
         }
      }
   }
#endif
   return result;
}

template <typename REAL>
PresolveStatus
SimplifyInequalities<REAL>::perform_simplify_ineq_task(
    const Num<REAL>& num, const ConstraintMatrix<REAL>& consMatrix,
    const Vec<RowActivity<REAL>>& activities, const Vec<RowFlags>& rflags,
    const Vec<ColFlags>& cflags, const Vec<REAL>& lhs, const Vec<REAL>& rhs,
    const Vec<REAL>& lbs, const Vec<REAL>& ubs, int row,
    Reductions<REAL>& reductions, Vec<int>& coefficientsThatCanBeDeleted,
    Vec<int>& colOrder )
{
   PresolveStatus result = PresolveStatus::kUnchanged;

   auto rowvec = consMatrix.getRowCoefficients( row );
   int rowLength = rowvec.getLength();

   if( isRedundant( row, rflags ) || isUnbounded( row, rflags ) ||
       isInfiniteActivity( activities, row ) ||
       // ignore empty or bound-constraints
       rowLength < 2 )
      return PresolveStatus::kUnchanged;

   const int* colinds = rowvec.getIndices();

   REAL greatestCommonDivisor = 0;
   bool isSimplificationPossible = false;

   colOrder.clear();
   coefficientsThatCanBeDeleted.clear();

   simplify( rowvec.getValues(), colinds, rowLength, activities[row],
             rflags[row], cflags, rhs[row], lhs[row], lbs, ubs, colOrder,
             coefficientsThatCanBeDeleted, greatestCommonDivisor,
             isSimplificationPossible, num );

   if( isSimplificationPossible )
   {
      assert( greatestCommonDivisor >= 1 );
      bool col_can_be_deleted = !coefficientsThatCanBeDeleted.empty();
      bool rhs_needs_update = false;
      bool lhs_needs_update = false;
      REAL new_rhs = 0;
      REAL new_lhs = 0;

      if( !rflags[row].test( RowFlag::kRhsInf ) && rhs[row] != 0 )
      {
         new_rhs = num.feasFloor( rhs[row] / greatestCommonDivisor ) *
                   greatestCommonDivisor;
         rhs_needs_update = new_rhs != rhs[row];
      }
      else if( !rflags[row].test( RowFlag::kLhsInf ) && lhs[row] != 0 )
      {
         new_lhs = num.feasCeil( lhs[row] / greatestCommonDivisor ) *
                   greatestCommonDivisor;
         lhs_needs_update = new_lhs != lhs[row];
      }

      if( !rhs_needs_update && !lhs_needs_update && !col_can_be_deleted )
         return PresolveStatus::kUnchanged;

      TransactionGuard<REAL> guard{ reductions };
      reductions.lockRow( row );

      for( int col : coefficientsThatCanBeDeleted )
      {
         reductions.changeMatrixEntry( row, colinds[col], 0 );
         result = PresolveStatus::kReduced;
      }

      // round side to multiple of greatestCommonDivisor; don't divide
      // row by greatestCommonDivisor
      if( rhs_needs_update )
      {
         assert( rhs[row] != 0 );
         reductions.changeRowRHS( row, new_rhs );
         result = PresolveStatus::kReduced;
      }
      if( lhs_needs_update )
      {
         assert( lhs[row] != 0 );
         reductions.changeRowLHS( row, new_lhs );
         result = PresolveStatus::kReduced;
      }
   }
   return result;
}

template <typename REAL>
bool
SimplifyInequalities<REAL>::isInfiniteActivity(
    const Vec<RowActivity<REAL>>& activities, int row ) const
{
   return activities[row].ninfmax != 0 || activities[row].ninfmin != 0;
}

template <typename REAL>
bool
SimplifyInequalities<REAL>::isRedundant( int row,
                                         const Vec<RowFlags>& rflags ) const
{
   return rflags[row].test( RowFlag::kRedundant );
}

template <typename REAL>
bool
SimplifyInequalities<REAL>::isUnbounded( int row,
                                         const Vec<RowFlags>& rowFlags ) const
{
   return !rowFlags[row].test( RowFlag::kRhsInf, RowFlag::kLhsInf );
}

} // namespace papilo

#endif
