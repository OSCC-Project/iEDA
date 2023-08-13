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

#ifndef _PAPILO_PRESOLVERS_SINGLETON_COLS_HPP_
#define _PAPILO_PRESOLVERS_SINGLETON_COLS_HPP_

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"
#include "papilo/misc/Hash.hpp"
#include "papilo/misc/Num.hpp"
#include "papilo/misc/fmt.hpp"

namespace papilo
{

template <typename REAL>
class SingletonCols : public PresolveMethod<REAL>
{
 public:
   SingletonCols() : PresolveMethod<REAL>()
   {
      this->setName( "colsingleton" );
      this->setTiming( PresolverTiming::kFast );
   }

   virtual PresolveStatus
   execute( const Problem<REAL>& problem,
            const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
            Reductions<REAL>& reductions, const Timer& timer) override;
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class SingletonCols<double>;
extern template class SingletonCols<Quad>;
extern template class SingletonCols<Rational>;
#endif

template <typename REAL>
PresolveStatus
SingletonCols<REAL>::execute( const Problem<REAL>& problem,
                              const ProblemUpdate<REAL>& problemUpdate,
                              const Num<REAL>& num,
                              Reductions<REAL>& reductions, const Timer& timer )
{
   const auto& domains = problem.getVariableDomains();
   const auto& lower_bounds = domains.lower_bounds;
   const auto& upper_bounds = domains.upper_bounds;
   const auto& cflags = domains.flags;

   const auto& activities = problem.getRowActivities();
   const auto& singletonCols = problemUpdate.getSingletonCols();

   const auto& constMatrix = problem.getConstraintMatrix();
   const auto& lhs_values = constMatrix.getLeftHandSides();
   const auto& rhs_values = constMatrix.getRightHandSides();
   const auto& rflags = constMatrix.getRowFlags();
   const auto& rowSizes = constMatrix.getRowSizes();
   const auto& obj = problem.getObjective().coefficients;

   PresolveStatus result = PresolveStatus::kUnchanged;

   auto handleEquation = [&]( int col, bool lowerboundImplied, bool ubimplied,
                              const REAL& val, int row, bool impliedeq,
                              const REAL& side ) {
      if( !impliedeq && rowSizes[row] <= 1 )
         return;

      result = PresolveStatus::kReduced;

      TransactionGuard<REAL> tg{ reductions };
      reductions.lockColBounds( col );
      reductions.lockRow( row );

      // if the equation is only implied, first change its side before doing
      // the substitution
      if( impliedeq )
      {
         if( rflags[row].test( RowFlag::kLhsInf ) )
         {
            assert( !rflags[row].test( RowFlag::kRhsInf ) );
            reductions.changeRowLHS( row, side );
         }
         else
         {
            assert( rflags[row].test( RowFlag::kRhsInf ) );
            reductions.changeRowRHS( row, side );
         }

         if( rowSizes[row] <= 1 )
            return;
      }

      // substitute the variable in the objective
      reductions.substituteColInObjective( col, row );

      // now check if the equation is redundant or needs to be modified
      if( lowerboundImplied && ubimplied )
      {
         // implied free -> just remove the equation completely
         reductions.markRowRedundant( row );
      }
      else
      {
         assert( lowerboundImplied || !cflags[col].test( ColFlag::kLbInf ) );
         assert( ubimplied || !cflags[col].test( ColFlag::kUbInf ) );

         // implied free only for one bound -> modify equation to be an
         // inequality and remove the columns coefficient
         reductions.changeMatrixEntry( row, col, 0 );

         if( num.isLT(val, 0) )
         {
            if( num.isLE(upper_bounds[col], 0) && !cflags[col].test( ColFlag::kUbInf ) )
            {

               if( lowerboundImplied )
                  reductions.changeRowLHSInf( row );
               else if( !num.isZero(lower_bounds[col]) )
                  reductions.changeRowLHS( row,
                                           side - lower_bounds[col] * val );
               if( ubimplied )
                  reductions.changeRowRHSInf( row );
               else if( !num.isZero(upper_bounds[col]) )
                  reductions.changeRowRHS( row,
                                           side - upper_bounds[col] * val );
            }
            else
            {
               if( ubimplied )
                  reductions.changeRowRHSInf( row );
               else if( !num.isZero(upper_bounds[col]) )
                  reductions.changeRowRHS( row,
                                           side - upper_bounds[col] * val );
               if( lowerboundImplied )
                  reductions.changeRowLHSInf( row );
               else if( !num.isZero(lower_bounds[col]) )
                  reductions.changeRowLHS( row,
                                           side - lower_bounds[col] * val );
            }
         }
         else
         {
            if( num.isLE(upper_bounds[col], 0) && !cflags[col].test( ColFlag::kUbInf ) )
            {
               if( lowerboundImplied )
                  reductions.changeRowRHSInf( row );
               else if( !num.isZero(lower_bounds[col]))
                  reductions.changeRowRHS( row,
                                           side - lower_bounds[col] * val );

               if( ubimplied )
                  reductions.changeRowLHSInf( row );
               else if( !num.isZero(upper_bounds[col]) )
                  reductions.changeRowLHS( row,
                                           side - upper_bounds[col] * val );
            }
            else
            {
               if( ubimplied )
                  reductions.changeRowLHSInf( row );
               else if( !num.isZero(upper_bounds[col])  )
                  reductions.changeRowLHS( row,
                                           side - upper_bounds[col] * val );

               if( lowerboundImplied )
                  reductions.changeRowRHSInf( row );

               else if( !num.isZero(lower_bounds[col]) )
                  reductions.changeRowRHS( row,
                                           side - lower_bounds[col] * val );
            }
         }
      }
   };

   int firstNewSingleton = problemUpdate.getFirstNewSingletonCol();
   for( std::size_t i = firstNewSingleton; i < singletonCols.size(); ++i )
   {
      int col = singletonCols[i];

      assert( constMatrix.getColSizes()[col] == 1 );
      assert( constMatrix.getColumnCoefficients( col ).getLength() == 1 );

      int row = constMatrix.getColumnCoefficients( col ).getIndices()[0];

      const REAL& val = constMatrix.getColumnCoefficients( col ).getValues()[0];

      assert( !constMatrix.isRowRedundant( row ) );

      if( rflags[row].test( RowFlag::kEquation ) )
      {
         assert( !rflags[row].test( RowFlag::kLhsInf, RowFlag::kRhsInf ) );
         assert( lhs_values[row] == rhs_values[row] );

         // singleton rows are already check in trivial presolve
         if( rowSizes[row] <= 1 )
            continue;

         // Found singleton column within an equation:
         // Check if it is implied free on one bound. In that case the
         // variable is substituted and the constraint stays as an inequality
         // constraint. Otherwise it is equaivalent to implied free variable
         // substitution.

         bool lowerboundImplied =
             row_implies_LB( num, lhs_values[row], rhs_values[row], rflags[row],
                             activities[row], val, lower_bounds[col],
                             upper_bounds[col], cflags[col] );

         if( !lowerboundImplied &&
             !problemUpdate.getPresolveOptions().removeslackvars )
            continue;

         bool ubimplied =
             row_implies_UB( num, lhs_values[row], rhs_values[row], rflags[row],
                             activities[row], val, lower_bounds[col],
                             upper_bounds[col], cflags[col] );

         if( !ubimplied &&
             ( !problemUpdate.getPresolveOptions().removeslackvars ||
               ( !lowerboundImplied && !num.isZero(obj[col]) ) ) )
            continue;

         if( cflags[col].test( ColFlag::kIntegral ) )
         {
            bool unsuitableForSubstitution = false;

            auto rowvec = constMatrix.getRowCoefficients( row );
            const int* rowinds = rowvec.getIndices();
            const REAL* rowvals = rowvec.getValues();
            for( int k = 0; k < rowvec.getLength(); ++k )
            {
               if( rowinds[k] == col )
                  continue;

               if( !cflags[rowinds[k]].test( ColFlag::kIntegral ) ||
                   !num.isIntegral( rowvals[k] / val ) )
               {
                  unsuitableForSubstitution = true;
                  break;
               }
            }

            if( unsuitableForSubstitution )
               continue;
         }

         handleEquation( col, lowerboundImplied, ubimplied, val, row, false,
                         rhs_values[row] );

         continue;
      }

      switch( problemUpdate.getPresolveOptions().dualreds )
      {
      case 0:
         // no dual reductions allowed
         continue;
      case 1:
         // only weak dual reductions allowed
         if( num.isZero(obj[col]) )
            continue;
      }

      int nuplocks = 0;
      int ndownlocks = 0;

      count_locks( val, rflags[row], ndownlocks, nuplocks );

      // ranged row (-inf < lhs < rhs < inf) -> not a singleton.
      // TODO: check if ranged row can be converted to an equation with dual
      // infer technique below
      if( nuplocks != 0 && ndownlocks != 0 )
         continue;

      if( ndownlocks == 0 && num.isGE(obj[col], 0) )
      {
         // dual fix to lower bound
         if( cflags[col].test( ColFlag::kLbInf ) )
         {
            if( !num.isZero(obj[col]) )
               return PresolveStatus::kUnbndOrInfeas;

            continue;
         }

         TransactionGuard<REAL> tg{ reductions };
         reductions.lockCol( col );
         reductions.fixCol( col, lower_bounds[col] );
         result = PresolveStatus::kReduced;

         continue;
      }

      if( nuplocks == 0 && num.isLE(obj[col], 0) )
      {
         // dual fix to upper bound
         if( cflags[col].test( ColFlag::kUbInf ) )
         {
            if( !num.isZero(obj[col]) )
               return PresolveStatus::kUnbndOrInfeas;

            continue;
         }

         TransactionGuard<REAL> tg{ reductions };
         reductions.lockCol( col );
         reductions.fixCol( col, upper_bounds[col] );
         result = PresolveStatus::kReduced;

         continue;
      }

      assert( ( num.isGT(obj[col] , 0) && ndownlocks == 1 && nuplocks == 0 ) ||
              ( num.isLT(obj[col] , 0) && ndownlocks == 0 && nuplocks == 1 ) );

      // for continuous columns we have a singleton row in the dual which
      // directly implies a bound for the dualvariable of the primal row.
      // If the dual bound implies the primal row to be an equation we can
      // substitute the singleton variable as above in the equation case
      if( !cflags[col].test( ColFlag::kIntegral ) )
      {
         bool duallbinf = true;
         bool dualubinf = true;

         assert( !num.isZero(val) );

         REAL duallb = obj[col] / val;
         REAL dualub = duallb;

         bool lowerboundImplied =
             row_implies_LB( num, lhs_values[row], rhs_values[row], rflags[row],
                             activities[row], val, lower_bounds[col],
                             upper_bounds[col], cflags[col] );
         bool ubimplied =
             row_implies_UB( num, lhs_values[row], rhs_values[row], rflags[row],
                             activities[row], val, lower_bounds[col],
                             upper_bounds[col], cflags[col] );

         if( lowerboundImplied && ubimplied )
         {
            duallbinf = false;
            dualubinf = false;
         }
         else if( lowerboundImplied )
         {
            if( num.isGT(val, 0) )
               duallbinf = false;
            else
               dualubinf = false;
         }
         else if( ubimplied )
         {
            if( num.isGT(val, 0) )
               dualubinf = false;
            else
               duallbinf = false;
         }

         if( !duallbinf && num.isGT( duallb, 0 ) )
         {
            bool removevar = true;

            assert( !rflags[row].test( RowFlag::kLhsInf ) );
            assert( rflags[row].test( RowFlag::kRhsInf ) ||
                    rhs_values[row] != lhs_values[row] );

            // check again if row implies more bounds with new right hand
            // side
            if( !lowerboundImplied )
            {
               lowerboundImplied = row_implies_LB(
                   num, lhs_values[row], lhs_values[row], RowFlag::kEquation,
                   activities[row], val, lower_bounds[col], upper_bounds[col],
                   cflags[col] );

               if( !lowerboundImplied &&
                   !problemUpdate.getPresolveOptions().removeslackvars )
                  removevar = false;
            }

            if( removevar && !ubimplied )
            {
               ubimplied = row_implies_UB(
                   num, lhs_values[row], lhs_values[row], RowFlag::kEquation,
                   activities[row], val, lower_bounds[col], upper_bounds[col],
                   cflags[col] );

               if( !ubimplied &&
                   ( !problemUpdate.getPresolveOptions().removeslackvars ||
                     ( !lowerboundImplied && !num.isZero(obj[col]) ) ) )
                  removevar = false;
            }

            if( removevar )
            {
               handleEquation( col, lowerboundImplied, ubimplied, val, row,
                               true, lhs_values[row] );
            }
            else
            {
               // if the variable should not be removed, then just apply the
               // dual reduction and change the constraint into an equation
               result = PresolveStatus::kReduced;

               TransactionGuard<REAL> tg{ reductions };
               reductions.lockColBounds( col );
               reductions.lockRow( row );
               reductions.changeRowRHS( row, lhs_values[row] );
            }
         }
         else if( !dualubinf && num.isLT( dualub, 0 ) )
         {
            bool removevar = true;

            assert( !rflags[row].test( RowFlag::kRhsInf ) );
            assert( rflags[row].test( RowFlag::kLhsInf ) ||
                    rhs_values[row] != lhs_values[row] );

            // check again if row implies more bounds with new left hand side
            if( !lowerboundImplied )
            {
               lowerboundImplied = row_implies_LB(
                   num, rhs_values[row], rhs_values[row], RowFlag::kEquation,
                   activities[row], val, lower_bounds[col], upper_bounds[col],
                   cflags[col] );

               if( !lowerboundImplied &&
                   !problemUpdate.getPresolveOptions().removeslackvars )
                  removevar = false;
            }

            if( removevar && !ubimplied )
            {
               ubimplied = row_implies_UB(
                   num, rhs_values[row], rhs_values[row], RowFlag::kEquation,
                   activities[row], val, lower_bounds[col], upper_bounds[col],
                   cflags[col] );

               if( !ubimplied &&
                   ( !problemUpdate.getPresolveOptions().removeslackvars ||
                     ( !lowerboundImplied && !num.isZero(obj[col]) ) ) )
                  removevar = false;
            }

            if( removevar )
            {
               handleEquation( col, lowerboundImplied, ubimplied, val, row,
                               true, rhs_values[row] );
            }
            else
            {
               // if the variable should not be removed, then just apply the
               // dual reduction and change the constraint into an equation
               result = PresolveStatus::kReduced;

               TransactionGuard<REAL> tg{ reductions };
               reductions.lockColBounds( col );
               reductions.lockRow( row );
               reductions.changeRowLHS( row, rhs_values[row] );
            }
         }
      }
   }

   return result;
}

} // namespace papilo

#endif
