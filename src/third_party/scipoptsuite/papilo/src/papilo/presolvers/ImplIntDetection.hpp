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

#ifndef _PAPILO_PRESOLVERS_IMPL_INT_DETECTION_HPP_
#define _PAPILO_PRESOLVERS_IMPL_INT_DETECTION_HPP_

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"
#include "papilo/misc/Num.hpp"
#include "papilo/misc/fmt.hpp"

namespace papilo
{

template <typename REAL>
class ImplIntDetection : public PresolveMethod<REAL>
{
 public:
   ImplIntDetection() : PresolveMethod<REAL>()
   {
      this->setName( "implint" );
      this->setTiming( PresolverTiming::kExhaustive );
      this->setType( PresolverType::kMixedCols );
   }

   PresolveStatus
   execute( const Problem<REAL>& problem,
            const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
            Reductions<REAL>& reductions, const Timer& timer) override;

 private:
   PresolveStatus
   perform_implied_integer_task(
       const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
       Reductions<REAL>& reductions, const Vec<ColFlags>& cflags,
       const ConstraintMatrix<REAL>& consmatrix, const Vec<REAL>& lhs_values,
       const Vec<REAL>& rhs_values, const Vec<REAL>& lower_bounds,
       const Vec<REAL>& upper_bounds, const Vec<RowFlags>& rflags,
       int col ) const;
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class ImplIntDetection<double>;
extern template class ImplIntDetection<Quad>;
extern template class ImplIntDetection<Rational>;
#endif

template <typename REAL>
PresolveStatus
ImplIntDetection<REAL>::execute( const Problem<REAL>& problem,
                                 const ProblemUpdate<REAL>& problemUpdate,
                                 const Num<REAL>& num,
                                 Reductions<REAL>& reductions, const Timer& timer )
{
   const auto& domains = problem.getVariableDomains();
   const auto& lower_bounds = domains.lower_bounds;
   const auto& upper_bounds = domains.upper_bounds;
   const auto& cflags = domains.flags;

   const auto& consmatrix = problem.getConstraintMatrix();
   const auto& lhs_values = consmatrix.getLeftHandSides();
   const auto& rhs_values = consmatrix.getRightHandSides();
   const auto& rflags = consmatrix.getRowFlags();
   const auto& ncols = consmatrix.getNCols();

   PresolveStatus result = PresolveStatus::kUnchanged;

#ifndef PAPILO_TBB
   assert( problemUpdate.getPresolveOptions().runs_sequential() );
#endif

   if( problemUpdate.getPresolveOptions().runs_sequential() ||
      !problemUpdate.getPresolveOptions().implied_integer_parallel )
   {
      for( int col = 0; col < ncols; ++col )
      {
         if( perform_implied_integer_task(
                 problemUpdate, num, reductions, cflags, consmatrix, lhs_values,
                 rhs_values, lower_bounds, upper_bounds, rflags,
                 col ) == PresolveStatus::kReduced )
            result = PresolveStatus::kReduced;
      }
   }
#ifdef PAPILO_TBB
   else
   {
      Vec<Reductions<REAL>> stored_reductions( ncols );
      tbb::parallel_for( tbb::blocked_range<int>( 0, ncols ),
                         [&]( const tbb::blocked_range<int>& r ) {
                            for( int col = r.begin(); col < r.end(); ++col )
                            {
                               if( perform_implied_integer_task(
                                       problemUpdate, num,
                                       stored_reductions[col], cflags,
                                       consmatrix, lhs_values, rhs_values,
                                       lower_bounds, upper_bounds, rflags,
                                       col ) == PresolveStatus::kReduced )
                                  result = PresolveStatus::kReduced;
                            }
                         } );

      if( result == PresolveStatus::kUnchanged )
         return PresolveStatus::kUnchanged;

      for( int i = 0; i < (int) stored_reductions.size(); ++i )
      {
         Reductions<REAL> reds = stored_reductions[i];
         if( reds.size() > 0 )
         {
            for( const auto& reduction : reds.getReductions() )
            {
               reductions.add_reduction( reduction.row, reduction.col,
                                         reduction.newval );
            }
         }
      }
   }
#endif
   return result;
}

template <typename REAL>
PresolveStatus
ImplIntDetection<REAL>::perform_implied_integer_task(
    const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
    Reductions<REAL>& reductions, const Vec<ColFlags>& cflags,
    const ConstraintMatrix<REAL>& consmatrix, const Vec<REAL>& lhs_values,
    const Vec<REAL>& rhs_values, const Vec<REAL>& lower_bounds,
    const Vec<REAL>& upper_bounds, const Vec<RowFlags>& rflags, int col ) const
{
   PresolveStatus result = PresolveStatus::kUnchanged;
   if( cflags[col].test( ColFlag::kIntegral, ColFlag::kImplInt,
                         ColFlag::kInactive ) )
      return PresolveStatus::kUnchanged;

   bool testinequalities = problemUpdate.getPresolveOptions().dualreds == 2;
   bool impliedint = false;

   auto colvec = consmatrix.getColumnCoefficients( col );
   int collen = colvec.getLength();
   const int* colrows = colvec.getIndices();
   const REAL* colvals = colvec.getValues();

   for( int i = 0; i != collen; ++i )
   {
      int row = colrows[i];

      if( rflags[row].test( RowFlag::kRedundant ) ||
          !rflags[row].test( RowFlag::kEquation ) )
         continue;

      testinequalities = false;
      REAL scale = 1 / colvals[i];
      if( !num.isIntegral( scale * rhs_values[row] ) )
         continue;

      auto rowvec = consmatrix.getRowCoefficients( row );
      int rowlen = rowvec.getLength();
      const int* rowcols = rowvec.getIndices();
      const REAL* rowvals = rowvec.getValues();

      impliedint = true;

      for( int j = 0; j != rowlen; ++j )
      {
         int rowcol = rowcols[j];

         if( rowcol == col )
            continue;

         if( !cflags[rowcol].test( ColFlag::kIntegral, ColFlag::kImplInt ) ||
             !num.isIntegral( scale * rowvals[j] ) )
         {
            impliedint = false;
            break;
         }
      }

      if( impliedint )
         break;
   }

   if( impliedint )
   {
      // add reduction
      reductions.impliedInteger( col );
      return PresolveStatus::kReduced;
   }

   if( !testinequalities )
      return PresolveStatus::kUnchanged;

   if( !cflags[col].test( ColFlag::kLbInf ) &&
       !num.isIntegral( lower_bounds[col] ) )
      return PresolveStatus::kUnchanged;

   if( !cflags[col].test( ColFlag::kUbInf ) &&
       !num.isIntegral( upper_bounds[col] ) )
      return PresolveStatus::kUnchanged;

   impliedint = true;

   for( int i = 0; i != collen; ++i )
   {
      int row = colrows[i];

      if( rflags[row].test( RowFlag::kRedundant ) )
         continue;

      REAL scale = 1 / colvals[i];

      if( !rflags[row].test( RowFlag::kRhsInf ) &&
          !num.isIntegral( scale * rhs_values[row] ) )
      {
         impliedint = false;
         break;
      }

      if( !rflags[row].test( RowFlag::kLhsInf ) &&
          !num.isIntegral( scale * lhs_values[row] ) )
      {
         impliedint = false;
         break;
      }

      auto rowvec = consmatrix.getRowCoefficients( row );
      int rowlen = rowvec.getLength();
      const int* rowcols = rowvec.getIndices();
      const REAL* rowvals = rowvec.getValues();

      for( int j = 0; j != rowlen; ++j )
      {
         int rowcol = rowcols[j];

         if( rowcol == col )
            continue;

         if( !cflags[rowcol].test( ColFlag::kIntegral, ColFlag::kImplInt ) ||
             !num.isIntegral( scale * rowvals[j] ) )
         {
            impliedint = false;
            break;
         }
      }

      if( !impliedint )
         break;
   }

   if( impliedint )
   {
      // add reduction
      reductions.impliedInteger( col );
      result = PresolveStatus::kReduced;
   }
   return result;
}

} // namespace papilo

#endif
