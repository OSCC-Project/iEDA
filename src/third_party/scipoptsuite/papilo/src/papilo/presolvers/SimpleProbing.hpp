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

#ifndef _PAPILO_PRESOLVERS_SIMPLE_PROBING_HPP_
#define _PAPILO_PRESOLVERS_SIMPLE_PROBING_HPP_

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"
#include "papilo/io/Message.hpp"

namespace papilo
{

template <typename REAL>
class SimpleProbing : public PresolveMethod<REAL>
{
 public:
   SimpleProbing() : PresolveMethod<REAL>()
   {
      this->setName( "simpleprobing" );
      this->setType( PresolverType::kIntegralCols );
      this->setTiming( PresolverTiming::kMedium );
   }

   PresolveStatus
   execute( const Problem<REAL>& problem,
            const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
            Reductions<REAL>& reductions, const Timer& timer ) override;

   void
   calculateReductionsForSimpleProbing(
       const Num<REAL>& num, Reductions<REAL>& reductions,
       const VariableDomains<REAL>& domains,
       const Vec<papilo::RowActivity<REAL>>& activities, const REAL* rowvals,
       const int* rowcols, int rowlen, int bincol, REAL binary_coeff );

   PresolveStatus
   perform_simple_probing_step(
       const Num<REAL>& num, Reductions<REAL>& reductions,
       const VariableDomains<REAL>& domains, const Vec<ColFlags>& cflags,
       const Vec<RowActivity<REAL>>& activities,
       const ConstraintMatrix<REAL>& constMatrix, const Vec<REAL>& rhs_values,
       const Vec<int>& rowsize, const Vec<RowFlags>& rflags, int i );
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class SimpleProbing<double>;
extern template class SimpleProbing<Quad>;
extern template class SimpleProbing<Rational>;
#endif

template <typename REAL>
PresolveStatus
SimpleProbing<REAL>::execute( const Problem<REAL>& problem,
                              const ProblemUpdate<REAL>& problemUpdate,
                              const Num<REAL>& num,
                              Reductions<REAL>& reductions, const Timer& timer )
{
   assert( problem.getNumIntegralCols() != 0 );

   PresolveStatus status = PresolveStatus::kUnchanged;
   const auto& domains = problem.getVariableDomains();
   const auto& cflags = domains.flags;
   const auto& activities = problem.getRowActivities();

   const auto& rowsize = problem.getRowSizes();

   const auto& constMatrix = problem.getConstraintMatrix();
   const auto& rhs_values = constMatrix.getRightHandSides();
   const auto& rflags = constMatrix.getRowFlags();

   int nrows = problem.getNRows();

#ifndef PAPILO_TBB
   assert( problemUpdate.getPresolveOptions().runs_sequential() );
#endif

   if( problemUpdate.getPresolveOptions().runs_sequential() ||
       !problemUpdate.getPresolveOptions().simple_probing_parallel )
   {
      for( int i = 0; i < nrows; ++i )
      {
         if( perform_simple_probing_step(
                 num, reductions, domains, cflags, activities, constMatrix,
                 rhs_values, rowsize, rflags, i ) == PresolveStatus::kReduced )
            status = PresolveStatus::kReduced;
      }
   }
#ifdef PAPILO_TBB
   else
   {
      Vec<Reductions<REAL>> stored_reductions( nrows );
      tbb::parallel_for( tbb::blocked_range<int>( 0, nrows ),
                         [&]( const tbb::blocked_range<int>& r ) {
                            for( int j = r.begin(); j < r.end(); ++j )
                            {
                               PresolveStatus s = perform_simple_probing_step(
                                   num, stored_reductions[j], domains, cflags,
                                   activities, constMatrix, rhs_values, rowsize,
                                   rflags, j );
                               assert( s == PresolveStatus::kUnchanged ||
                                       s == PresolveStatus::kReduced );
                               if( s == PresolveStatus::kReduced )
                                  status = s;
                            }
                         } );

      if( status == PresolveStatus::kUnchanged )
         return PresolveStatus::kUnchanged;

      for( int i = 0; i < (int) stored_reductions.size(); ++i )
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
                  reductions.add_reduction( reduction.row,
                                            reduction.col, reduction.newval );
               }
            }
         }
      }
   }

#endif
   return status;
}

template <typename REAL>
PresolveStatus
SimpleProbing<REAL>::perform_simple_probing_step(
    const Num<REAL>& num, Reductions<REAL>& reductions,
    const VariableDomains<REAL>& domains, const Vec<ColFlags>& cflags,
    const Vec<RowActivity<REAL>>& activities,
    const ConstraintMatrix<REAL>& constMatrix, const Vec<REAL>& rhs_values,
    const Vec<int>& rowsize, const Vec<RowFlags>& rflags, int i )
{
   PresolveStatus status = PresolveStatus::kUnchanged;
   if( !rflags[i].test( RowFlag::kEquation ) || rowsize[i] <= 2 ||
       activities[i].ninfmin != 0 || activities[i].ninfmax != 0 ||
       !num.isEq( activities[i].min + activities[i].max, 2 * rhs_values[i] ) )
      return PresolveStatus::kUnchanged;

   assert( rflags[i].test( RowFlag::kEquation ) );
   assert( activities[i].ninfmin == 0 && activities[i].ninfmax == 0 );
   assert(
       num.isEq( activities[i].min + activities[i].max, 2 * rhs_values[i] ) );

   auto rowvec = constMatrix.getRowCoefficients( i );
   const REAL* rowvals = rowvec.getValues();
   const int* rowcols = rowvec.getIndices();
   const int rowlen = rowvec.getLength();

   REAL bincoef = activities[i].max - rhs_values[i];

   for( int k = 0; k != rowlen; ++k )
   {
      int col = rowcols[k];
      assert( !cflags[col].test( ColFlag::kUnbounded ) );
      if( !cflags[col].test( ColFlag::kIntegral ) ||
          domains.lower_bounds[col] != 0 || domains.upper_bounds[col] != 1 ||
          !num.isEq( abs( rowvals[k] ), bincoef ) )
         continue;

      assert( num.isEq( abs( bincoef ), activities[i].max - rhs_values[i] ) );
      assert( domains.lower_bounds[col] == 0 );
      assert( domains.upper_bounds[col] == 1 );
      assert( cflags[col].test( ColFlag::kIntegral ) );

      Message::debug( this,
                      "probing on simple equation detected {} substitutions\n",
                      rowlen - 1 );
      calculateReductionsForSimpleProbing( num, reductions, domains, activities,
                                           rowvals, rowcols, rowlen, col,
                                           rowvals[k] );
      status = PresolveStatus::kReduced;
   }
   return status;
}
template <typename REAL>
void
SimpleProbing<REAL>::calculateReductionsForSimpleProbing(
    const Num<REAL>& num, Reductions<REAL>& reductions,
    const VariableDomains<REAL>& domains,
    const Vec<papilo::RowActivity<REAL>>& activities, const REAL* rowvals,
    const int* rowcols, int rowlen, int bincol, REAL binary_coeff )
{
   for( int k = 0; k != rowlen; ++k )
   {
      int col = rowcols[k];
      if( col == bincol )
         continue;

      REAL factor;
      REAL offset;
      if( ( rowvals[k] > 0 && binary_coeff > 0 ) ||
          ( rowvals[k] < 0 && binary_coeff < 0 ) )
      {
         factor = domains.lower_bounds[col] - domains.upper_bounds[col];
         offset = domains.upper_bounds[col];
      }
      else
      {
         factor = domains.upper_bounds[col] - domains.lower_bounds[col];
         offset = domains.lower_bounds[col];
      }

      reductions.replaceCol( col, bincol, factor, offset );
   }
}

} // namespace papilo

#endif
