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

#ifndef _PAPILO_PRESOLVERS_SINGLETON_STUFFING_HPP_
#define _PAPILO_PRESOLVERS_SINGLETON_STUFFING_HPP_

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"
#include "papilo/misc/Num.hpp"
#include "papilo/misc/fmt.hpp"

namespace papilo
{

template <typename REAL>
class SingletonStuffing : public PresolveMethod<REAL>
{
 public:
   SingletonStuffing() : PresolveMethod<REAL>()
   {
      this->setName( "stuffing" );
      this->setTiming( PresolverTiming::kMedium );
   }

   virtual PresolveStatus
   execute( const Problem<REAL>& problem,
            const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
            Reductions<REAL>& reductions, const Timer& timer ) override;
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class SingletonStuffing<double>;
extern template class SingletonStuffing<Quad>;
extern template class SingletonStuffing<Rational>;
#endif

template <typename REAL>
PresolveStatus
SingletonStuffing<REAL>::execute( const Problem<REAL>& problem,
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
   const auto& nrows = constMatrix.getNRows();
   const auto& colsize = constMatrix.getColSizes();
   const auto& rowsize = constMatrix.getRowSizes();
   const auto& obj = problem.getObjective().coefficients;

   PresolveStatus result = PresolveStatus::kUnchanged;
   Vec<int> rowsWithPenaltySingletons;
   Vec<uint8_t> penaltyVarCount( nrows );

   auto handleEquation = [&]( int col, bool lbimplied, bool ubimplied,
                              const REAL& val, int row, bool impliedeq,
                              const REAL& side ) {
      if( !impliedeq && rowsize[row] <= 1 )
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

         if( rowsize[row] <= 1 )
            return;
      }

      // substitute the variable in the objective
      reductions.substituteColInObjective( col, row );

      // now check if the equation is redundant or needs to be modified
      if( lbimplied && ubimplied )
      {
         // implied free -> just remove the equation completely
         reductions.markRowRedundant( row );
      }
      else
      {
         assert( lbimplied || !cflags[col].test( ColFlag::kLbInf ) );
         assert( ubimplied || !cflags[col].test( ColFlag::kUbInf ) );

         // implied free only for one bound -> modify equation to be an
         // inequality and remove the columns coefficient
         reductions.changeMatrixEntry( row, col, 0 );

         if( val < 0 )
         {
            // set the Bound first to infinity to avoid infeasiblty during bounds changes
            if( lbimplied )
               reductions.changeRowLHSInf( row );
            if( ubimplied )
               reductions.changeRowRHSInf( row );

            if( !lbimplied  && lower_bounds[col] != 0 )
               reductions.changeRowLHS( row, side - lower_bounds[col] * val );
            if(!ubimplied  &&  upper_bounds[col] != 0 )
               reductions.changeRowRHS( row, side - upper_bounds[col] * val );

         }
         else
         {
            // set the Bound first to infinity to avoid infeasiblty during bounds changes
            if( lbimplied )
               reductions.changeRowRHSInf( row );
            if( ubimplied )
               reductions.changeRowLHSInf( row );

            if( !lbimplied  && lower_bounds[col] != 0 )
               reductions.changeRowRHS( row, side - lower_bounds[col] * val );
            if( !ubimplied  && upper_bounds[col] != 0 )
               reductions.changeRowLHS( row, side - upper_bounds[col] * val );
         }
      }
   };

   for( int col : singletonCols )
   {
      assert( colsize[col] == 1 );
      assert( constMatrix.getColumnCoefficients( col ).getLength() == 1 );


      int row = constMatrix.getColumnCoefficients( col ).getIndices()[0];
      const REAL& val = constMatrix.getColumnCoefficients( col ).getValues()[0];

      assert( !constMatrix.isRowRedundant( row ) );

      REAL lhs = lhs_values[row];
      REAL rhs = rhs_values[row];

      if( rflags[row].test( RowFlag::kEquation ) )
      {
         assert( !rflags[row].test( RowFlag::kLhsInf, RowFlag::kRhsInf ) );

         // Found singleton column within an equation:
         // Check if it is implied free on one bound. In that case the
         // variable is substituted and the constraint stays as an inequality
         // constraint. Otherwise it is equaivalent to implied free variable
         // substitution.

         if( rowsize[row] <= 1 )
            continue;

         bool lbimplied =
             row_implies_LB( num, lhs, rhs, rflags[row],
                             activities[row], val, lower_bounds[col],
                             upper_bounds[col], cflags[col] );

         if( !lbimplied && !problemUpdate.getPresolveOptions().removeslackvars )
            continue;

         bool ubimplied =
             row_implies_UB( num, lhs, rhs, rflags[row],
                             activities[row], val, lower_bounds[col],
                             upper_bounds[col], cflags[col] );

         if( !ubimplied &&
             ( !problemUpdate.getPresolveOptions().removeslackvars ||
               ( !lbimplied && obj[col] != 0 ) ) )
            continue;

         if( cflags[col].test( ColFlag::kIntegral ) )
         {
            bool unsuitableForSubstitution = false;

            auto rowvec = constMatrix.getRowCoefficients( row );
            const int* rowinds = rowvec.getIndices();
            const REAL* rowvals = rowvec.getValues();
            for( int i = 0; i != rowvec.getLength(); ++i )
            {
               if( rowinds[i] == col )
                  continue;

               if( !cflags[rowinds[i]].test( ColFlag::kIntegral ) )
               {
                  unsuitableForSubstitution = true;
                  break;
               }

               if( !num.isIntegral( rowvals[i] / val ) )
               {
                  unsuitableForSubstitution = true;
                  break;
               }
            }

            if( unsuitableForSubstitution )
               continue;
         }

         handleEquation( col, lbimplied, ubimplied, val, row, false, rhs );

         continue;
      }

      switch( problemUpdate.getPresolveOptions().dualreds )
      {
      case 0:
         // no dual reductions allowed
         continue;
      case 1:
         // only weak dual reductions allowed
         if( obj[col] == 0 )
            continue;
      }

      int nuplocks = 0;
      int ndownlocks = 0;

      count_locks( val, rflags[row], ndownlocks, nuplocks );

      // ranged row -> not a singleton. @TODO: check if ranged row can be
      // converted to an equation with dual infer technique below
      if( nuplocks != 0 && ndownlocks != 0 )
         continue;

      if( ndownlocks == 0 && obj[col] >= 0 )
      {
         // dual fix to lower bound
         if( cflags[col].test( ColFlag::kLbInf ) )
         {
            if( obj[col] != 0 )
               return PresolveStatus::kUnbndOrInfeas;

            continue;
         }

         TransactionGuard<REAL> tg{ reductions };
         reductions.lockCol( col );
         reductions.fixCol( col, lower_bounds[col] );
         result = PresolveStatus::kReduced;

         continue;
      }

      if( nuplocks == 0 && obj[col] <= 0 )
      {
         // dual fix to upper bound
         if( cflags[col].test( ColFlag::kUbInf ) )
         {
            if( obj[col] != 0 )
               return PresolveStatus::kUnbndOrInfeas;

            continue;
         }

         TransactionGuard<REAL> tg{ reductions };
         reductions.lockCol( col );
         reductions.fixCol( col, upper_bounds[col] );
         result = PresolveStatus::kReduced;

         continue;
      }

      assert( ( obj[col] > 0 && ndownlocks == 1 && nuplocks == 0 ) ||
              ( obj[col] < 0 && ndownlocks == 0 && nuplocks == 1 ) );

      // We are in the penalty case, check if the row can be converted to an
      // euqation due to complementary slackness:
      // This column is a singleton row in the dual space and therefore
      // directly gives a bound on the dual variable associated with this
      // row. If the bound implies that this dual variable cannot be zero,
      // the row must hold with equality and this singleton can be handled as
      // in the equation case above. Do not remember row for stuffing if
      // successful

      // remember that row contains at least one penalty singleton and should
      // be considered for the singleton stuffing step

      // for continuous columns we have a singleton row in the dual which
      // directly implies a bound for the dualvariable of the primal row.
      // If the dual bound implies the primal row to be an equation we can
      // substitute the singleton variable as above in the equation case
      if( !cflags[col].test( ColFlag::kIntegral ) )
      {
         bool duallbinf = true;
         bool dualubinf = true;

         assert( val != 0 );

         REAL duallb = obj[col] / val;
         REAL dualub = duallb;

         bool lbimplied =
             row_implies_LB( num, lhs, rhs, rflags[row],
                             activities[row], val, lower_bounds[col],
                             upper_bounds[col], cflags[col] );
         bool ubimplied =
             row_implies_UB( num, lhs, rhs, rflags[row],
                             activities[row], val, lower_bounds[col],
                             upper_bounds[col], cflags[col] );

         if( lbimplied && ubimplied )
         {
            duallbinf = false;
            dualubinf = false;
         }
         else if( lbimplied )
         {
            if( val > 0 )
               duallbinf = false;
            else
               dualubinf = false;
         }
         else if( ubimplied )
         {
            if( val > 0 )
               dualubinf = false;
            else
               duallbinf = false;
         }

         if( !duallbinf && num.isFeasGT( duallb, 0 ) )
         {
            bool removevar = true;

            assert( !rflags[row].test( RowFlag::kLhsInf ) );
            assert( rflags[row].test( RowFlag::kRhsInf ) || ! num.isEq(rhs, lhs) );

            // check again if row implies more bounds with new right hand
            // side
            if( !lbimplied )
            {
               lbimplied = row_implies_LB(
                   num, lhs, lhs, RowFlag::kEquation,
                   activities[row], val, lower_bounds[col], upper_bounds[col],
                   cflags[col] );

               if( !lbimplied &&
                   !problemUpdate.getPresolveOptions().removeslackvars )
                  removevar = false;
            }

            if( removevar && !ubimplied )
            {
               ubimplied = row_implies_UB(
                   num, lhs, lhs, RowFlag::kEquation,
                   activities[row], val, lower_bounds[col], upper_bounds[col],
                   cflags[col] );

               if( !ubimplied &&
                   ( !problemUpdate.getPresolveOptions().removeslackvars ||
                     ( !lbimplied && obj[col] != 0 ) ) )
                  removevar = false;
            }

            if( removevar )
            {
               handleEquation( col, lbimplied, ubimplied, val, row, true, lhs );
            }
            else
            {
               // if the variable should not be removed, then just apply the
               // dual reduction and change the constraint into an equation
               result = PresolveStatus::kReduced;

               TransactionGuard<REAL> tg{ reductions };
               reductions.lockColBounds( col );
               reductions.lockRow( row );
               reductions.changeRowRHS( row, lhs );
            }
         }
         else if( !dualubinf && num.isFeasLT( dualub, 0 ) )
         {
            bool removevar = true;

            assert( !rflags[row].test( RowFlag::kRhsInf ) );
            assert( rflags[row].test( RowFlag::kLhsInf ) || rhs != lhs );

            // check again if row implies more bounds with new left hand side
            if( !lbimplied )
            {
               lbimplied = row_implies_LB(
                   num, rhs, rhs, RowFlag::kEquation,
                   activities[row], val, lower_bounds[col], upper_bounds[col],
                   cflags[col] );

               if( !lbimplied &&
                   !problemUpdate.getPresolveOptions().removeslackvars )
                  removevar = false;
            }

            if( removevar && !ubimplied )
            {
               ubimplied = row_implies_UB(
                   num, rhs, rhs, RowFlag::kEquation,
                   activities[row], val, lower_bounds[col], upper_bounds[col],
                   cflags[col] );

               if( !ubimplied &&
                   ( !problemUpdate.getPresolveOptions().removeslackvars ||
                     ( !lbimplied && obj[col] != 0 ) ) )
                  removevar = false;
            }

            if( removevar )
            {
               handleEquation( col, lbimplied, ubimplied, val, row, true, rhs );
            }
            else
            {
               // if the variable should not be removed, then just apply the
               // dual reduction and change the constraint into an equation
               result = PresolveStatus::kReduced;

               TransactionGuard<REAL> tg{ reductions };
               reductions.lockColBounds( col );
               reductions.lockRow( row );
               reductions.changeRowLHS( row, rhs );
            }
         }
         else
         {
            switch( penaltyVarCount[row] )
            {
            case 0:
               penaltyVarCount[row] = 1;
               break;
            case 1:
               penaltyVarCount[row] = 2;
               rowsWithPenaltySingletons.push_back( row );
            }
         }
      }
   }

   if( problemUpdate.getPresolveOptions().dualreds < 2 )
      return result;

   Vec<std::pair<int, REAL>> penaltyvars;

   for( int row : rowsWithPenaltySingletons )
   {
      assert( rflags[row].test( RowFlag::kLhsInf ) ||
              rflags[row].test( RowFlag::kRhsInf ) );
      assert( !rflags[row].test( RowFlag::kLhsInf ) ||
              !rflags[row].test( RowFlag::kRhsInf ) );

      int scale;
      REAL rhs;

      if( rflags[row].test( RowFlag::kRhsInf ) )
      {
         rhs = -lhs_values[row];
         scale = -1;
      }
      else
      {
         rhs = rhs_values[row];
         scale = 1;
      }

      auto rowvec = constMatrix.getRowCoefficients( row );
      const int* rowinds = rowvec.getIndices();
      const REAL* rowvals = rowvec.getValues();
      const int len = rowvec.getLength();

      // TODO do singleton stuffing
      bool suitable = true;
      for( int i = 0; i != len; ++i )
      {
         int col = rowinds[i];
         REAL coeff = scale * rowvals[i];

         // if the column is a singleton and not integral it could be a
         // penalty variable
         if( colsize[col] == 1 && !cflags[col].test( ColFlag::kIntegral ) )
         {
            // for penalty variables adjust the right hand side as if it
            // where complemented whith the cheapest bound
            if( coeff > 0 && obj[col] < 0 &&
                !cflags[col].test( ColFlag::kUbUseless ) )
            {
               rhs -= coeff * upper_bounds[col];
               penaltyvars.emplace_back( col, std::move( coeff ) );
               continue;
            }

            if( coeff < 0 && obj[col] > 0 &&
                !cflags[col].test( ColFlag::kLbUseless ) )
            {
               rhs -= coeff * lower_bounds[col];
               penaltyvars.emplace_back( col, std::move( coeff ) );
               continue;
            }

            // handle the dual fix case here again to adjust the right hand
            // side with the stronger values
            if( coeff > 0 && obj[col] >= 0 &&
                !cflags[col].test( ColFlag::kLbInf ) )
            {
               rhs -= coeff * lower_bounds[col];
               continue;
            }

            if( coeff < 0 && obj[col] <= 0 &&
                !cflags[col].test( ColFlag::kUbInf ) )
            {
               rhs -= coeff * upper_bounds[col];
               continue;
            }
         }

         // for non-singletons or singletons that where not handled in the
         // case above treat them like every other variable i.e. subtract
         // them with their largest contribution from the right hand side
         if( coeff > 0 )
         {
            if( cflags[col].test( ColFlag::kUbUseless ) )
            {
               suitable = false;
               break;
            }

            rhs -= coeff * upper_bounds[col];
         }
         else
         {
            if( cflags[col].test( ColFlag::kLbUseless ) )
            {
               suitable = false;
               break;
            }

            rhs -= coeff * lower_bounds[col];
         }
      }

      if( !suitable )
      {
         penaltyvars.clear();
         continue;
      }

      pdqsort( penaltyvars.begin(), penaltyvars.end(),
               [&]( const std::pair<int, REAL>& c1,
                    const std::pair<int, REAL>& c2 ) {
                  return ( obj[c1.first] / c1.second ) >
                         ( obj[c2.first] / c2.second );
               } );

      std::size_t k = 0;
      while( k < penaltyvars.size() && num.isFeasLT( rhs, 0 ) )
      {
         int col = penaltyvars[k].first;
         const REAL& coeff = penaltyvars[k].second;

         // for non-singletons or singletons that where not handled in the
         // case above treat them like every other variable i.e. subtract
         // them with their largest contribution from the right hand side
         if( coeff > 0 )
         {
            if( cflags[col].test( ColFlag::kLbUseless ) )
            {
               suitable = false;
               break;
            }

            rhs -= coeff * ( lower_bounds[col] - upper_bounds[col] );
         }
         else
         {
            if( cflags[col].test( ColFlag::kUbUseless ) )
            {
               suitable = false;
               break;
            }

            rhs -= coeff * ( upper_bounds[col] - lower_bounds[col] );
         }

         ++k;
      }

      if( suitable && k != penaltyvars.size() )
      {
         assert( num.isFeasGE( rhs, 0 ) );

         while( k < penaltyvars.size() )
         {
            int col = penaltyvars[k].first;
            const REAL& coeff = penaltyvars[k].second;

            if( coeff < 0 )
               reductions.fixCol( col, lower_bounds[col] );
            else
               reductions.fixCol( col, upper_bounds[col] );

            ++k;
         }
      }

      penaltyvars.clear();
   }

   return result;
}

} // namespace papilo

#endif
