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

#ifndef _PAPILO_PRESOLVERS_DUAL_FIX_HPP_
#define _PAPILO_PRESOLVERS_DUAL_FIX_HPP_

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"

namespace papilo
{

/// dual-fixing presolve method which looks at the coefficients of the objective
/// and the column entries and performs a dual fixing if possible
/// If fixing is not possible, it tries to strengthen the bounds.
template <typename REAL>
class DualFix : public PresolveMethod<REAL>
{
 public:
   DualFix() : PresolveMethod<REAL>()
   {
      this->setName( "dualfix" );
      this->setTiming( PresolverTiming::kMedium );
   }

   bool is_fix_to_infinity_allowed = true;

   bool
   initialize( const Problem<REAL>& problem,
               const PresolveOptions& presolveOptions ) override
   {
      if( presolveOptions.dualreds == 0 )
         this->setEnabled( false );
      return false;
   }

   void
   addPresolverParams( ParameterSet& paramSet ) override
   {
      paramSet.addParameter( "dualfix.is_fix_to_infinity_allowed",
                             "should variables be set to infinity if their objective value is 0?",
                             is_fix_to_infinity_allowed );
   }

   void
   set_fix_to_infinity_allowed( bool val){
       is_fix_to_infinity_allowed = val;
   };

   PresolveStatus
   execute( const Problem<REAL>& problem,
            const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
            Reductions<REAL>& reductions, const Timer& timer ) override;

 private:
   PresolveStatus
   perform_dual_fix_step( const Num<REAL>& num, Reductions<REAL>& reductions,
                          const ConstraintMatrix<REAL>& consMatrix,
                          const Vec<RowActivity<REAL>>& activities,
                          const Vec<ColFlags>& cflags,
                          const Vec<REAL>& objective, const Vec<REAL>& lbs,
                          const Vec<REAL>& ubs, const Vec<RowFlags>& rflags,
                          const Vec<REAL>& lhs, const Vec<REAL>& rhs, int& i,
                          bool no_strong_reductions,
                          bool skip_variable_tightening,
                          REAL bound_tightening_offset ) const;
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class DualFix<double>;
extern template class DualFix<Quad>;
extern template class DualFix<Rational>;
#endif

template <typename REAL>
PresolveStatus
DualFix<REAL>::execute( const Problem<REAL>& problem,
                        const ProblemUpdate<REAL>& problemUpdate,
                        const Num<REAL>& num, Reductions<REAL>& reductions, const Timer& timer )
{
   const auto& consMatrix = problem.getConstraintMatrix();
   const Vec<RowActivity<REAL>>& activities = problem.getRowActivities();
   const Vec<ColFlags>& cflags = problem.getColFlags();
   const Vec<REAL>& objective = problem.getObjective().coefficients;
   const Vec<REAL>& lbs = problem.getLowerBounds();
   const Vec<REAL>& ubs = problem.getUpperBounds();
   const int ncols = consMatrix.getNCols();
   const Vec<RowFlags>& rflags = consMatrix.getRowFlags();
   const Vec<REAL>& lhs = consMatrix.getLeftHandSides();
   const Vec<REAL>& rhs = consMatrix.getRightHandSides();

   PresolveStatus result = PresolveStatus::kUnchanged;
   bool noStrongReductions = problemUpdate.getPresolveOptions().dualreds < 2;

   // calculating the basis for variable tightening (not fixings) may lead in
   // the postsolving step to a solution that is not in a vertex. In this case a
   // crossover would be required is too expensive performance wise
   // Exception: for infinity bounds set a finite bound that is worse that the best possible bound
   const bool skip_variable_tightening =
       problem.getNumIntegralCols() == 0 &&
       problemUpdate.getPresolveOptions().calculate_basis_for_dual;

   const REAL bound_tightening_offset =
       REAL(problemUpdate.getPresolveOptions().get_variable_bound_tightening_offset());

#ifndef PAPILO_TBB
   assert( problemUpdate.getPresolveOptions().runs_sequential() );
#endif

   if( problemUpdate.getPresolveOptions().runs_sequential() ||
       !problemUpdate.getPresolveOptions().dual_fix_parallel )
   {
      for( int col = 0; col < ncols; ++col )
      {
         PresolveStatus local_status = perform_dual_fix_step(
             num, reductions, consMatrix, activities, cflags, objective, lbs,
             ubs, rflags, lhs, rhs, col, noStrongReductions, skip_variable_tightening, bound_tightening_offset );
         assert( local_status == PresolveStatus::kUnchanged ||
                 local_status == PresolveStatus::kReduced ||
                 local_status == PresolveStatus::kUnbndOrInfeas ||
                 local_status == PresolveStatus::kUnbounded );
         if( local_status == PresolveStatus::kUnbounded ||
             local_status == PresolveStatus::kUnbndOrInfeas)
            return local_status;
         else if( local_status == PresolveStatus::kReduced )
            result = PresolveStatus::kReduced;
      }
   }
#ifdef PAPILO_TBB
   else
   {
      Vec<Reductions<REAL>> stored_reductions( ncols );
      bool unbounded = false;
      tbb::parallel_for(
          tbb::blocked_range<int>( 0, ncols ),
          [&]( const tbb::blocked_range<int>& r ) {
             for( int col = r.begin(); col < r.end(); ++col )
             {
                PresolveStatus local_status = perform_dual_fix_step(
                    num, stored_reductions[col], consMatrix, activities, cflags,
                    objective, lbs, ubs, rflags, lhs, rhs, col,
                    noStrongReductions, skip_variable_tightening, bound_tightening_offset );
                assert( local_status == PresolveStatus::kUnchanged ||
                        local_status == PresolveStatus::kReduced ||
                        local_status == PresolveStatus::kUnbounded );
                if( local_status == PresolveStatus::kUnbounded )
                   unbounded = true;
                else if( local_status == PresolveStatus::kReduced )
                   result = PresolveStatus::kReduced;
             }
          } );
      if( unbounded )
         return PresolveStatus::kUnbounded;
      else if( result == PresolveStatus::kUnchanged )
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
DualFix<REAL>::perform_dual_fix_step(
    const Num<REAL>& num, Reductions<REAL>& reductions,
    const ConstraintMatrix<REAL>& consMatrix,
    const Vec<RowActivity<REAL>>& activities, const Vec<ColFlags>& cflags,
    const Vec<REAL>& objective, const Vec<REAL>& lbs, const Vec<REAL>& ubs,
    const Vec<RowFlags>& rflags, const Vec<REAL>& lhs, const Vec<REAL>& rhs,
    int& i, bool no_strong_reductions, bool skip_variable_tightening,
    REAL bound_tightening_offset ) const
{
   // skip inactive columns
   if( cflags[i].test( ColFlag::kInactive ) )
      return PresolveStatus::kUnchanged;

   // if strong dual reductions are not allowed, we cannot dual fix variables
   // with zero objective
   if( no_strong_reductions && objective[i] == 0 )
      return PresolveStatus::kUnchanged;

   PresolveStatus result = PresolveStatus::kUnchanged;

   auto colvec = consMatrix.getColumnCoefficients( i );
   int collen = colvec.getLength();
   const REAL* values = colvec.getValues();
   const int* rowinds = colvec.getIndices();
   int nuplocks = 0;
   int ndownlocks = 0;

   // count "lock" for objective function
   if( objective[i] < 0 )
      ++ndownlocks;
   else if( objective[i] > 0 )
      ++nuplocks;

   for( int j = 0; j != collen; ++j )
   {
      count_locks( values[j], rflags[rowinds[j]], ndownlocks, nuplocks );

      if( nuplocks != 0 && ndownlocks != 0 )
         break;
   }

   // fix column to lower or upper bound
   if( ndownlocks == 0 )
   {
      assert( cflags[i].test( ColFlag::kUnbounded ) || ubs[i] != lbs[i] );

      // use a transaction and lock the column to protect it from changes
      // of other presolvers
      if( !cflags[i].test( ColFlag::kLbInf ) )
      {
         TransactionGuard<REAL> guard{ reductions };

         reductions.lockCol( i );
         reductions.fixCol( i, lbs[i] );

         return PresolveStatus::kReduced;
      }
      else if( objective[i] != 0 )
      {
         return PresolveStatus::kUnbndOrInfeas;
      }
      else if( is_fix_to_infinity_allowed )
      {
         TransactionGuard<REAL> guard{ reductions };
         reductions.lockCol( i );
         reductions.fixColNegativeInfinity( i, collen, rowinds );
         return PresolveStatus::kReduced;
      }
   }
   else if( nuplocks == 0 )
   {
      assert( cflags[i].test( ColFlag::kUnbounded ) || ubs[i] != lbs[i] );

      if( !cflags[i].test( ColFlag::kUbInf ) )
      {
         TransactionGuard<REAL> guard{ reductions };

         reductions.lockCol( i );
         reductions.fixCol( i, ubs[i] );

         return PresolveStatus::kReduced;
      }
      else if( objective[i] != 0 )
      {
         return PresolveStatus::kUnbndOrInfeas;
      }
      else if( is_fix_to_infinity_allowed )
      {
         TransactionGuard<REAL> guard{ reductions };
         reductions.lockCol( i );
         reductions.fixColPositiveInfinity( i, collen, rowinds );
         return PresolveStatus::kReduced;
      }
   }
   // apply dual substitution
   else
   {
      // Function checks if considered row allows dual bound strengthening
      // and calculates tightest bound for this row.
      auto check_row = []( int ninf, REAL activity, const REAL& side,
                           const REAL& coeff, const REAL& boundval,
                           bool boundinf, bool& skip_variable, REAL& cand_bound ) {
         switch( ninf )
         {
         case 0:
            assert( !boundinf );
            // calculate residual activity
            activity -= boundval * coeff;
            break;
         case 1:
            if( boundinf )
               break;
            skip_variable = true;
            return;
         default:
            // If one of the other variables with non-zero entry is
            // unbounded, dual bound strengthening is not possible for this
            // column; skip column.
            skip_variable = true;
            return;
         }

         // calculate candidate for new bound
         cand_bound = ( side - activity ) / coeff;
      };
      // If c_i >= 0, we might derive a tighter upper bound.
      // We consider only rows of
      // M := { (a_ji < 0 and rhs != inf) or (a_ji > 0 and lhs != inf)}.
      // If all constraints in M get redundant for x_i = new_UB, the upper
      // bound can be set to new_UB.
      if( objective[i] >= 0 )
      {
         // in case of skip_variable_tightening set bounds to an infinite value
         // if possible, but increase the bound slightly
         if( skip_variable_tightening  && ! cflags[i].test( ColFlag::kUbInf ) )
            return PresolveStatus::kUnchanged;
         bool skip_performing = false;
         bool new_ub_init = false;
         REAL new_ub = REAL(0);
         int best_row = -1;

         // go through all rows with non-zero entry
         for( int j = 0; j < collen; ++j )
         {
            int row = rowinds[j];
            // candidate for new upper bound
            REAL cand_bound;

            if( consMatrix.isRowRedundant( row ) )
               continue;

            // if row is in M, calculate candidate for new upper bound
            if( values[j] < 0.0 )
            {
               if( !rflags[row].test( RowFlag::kRhsInf ) )
               {
                  check_row( activities[row].ninfmax, activities[row].max,
                             rhs[row], values[j], lbs[i],
                             cflags[i].test( ColFlag::kLbInf ), skip_performing,
                             cand_bound );
               }
               else
                  // row is not in M
                  continue;
            }
            else
            {
               if( !rflags[row].test( RowFlag::kLhsInf ) )
               {
                  check_row( activities[row].ninfmin, activities[row].min,
                             lhs[row], values[j], lbs[i],
                             cflags[i].test( ColFlag::kLbInf ), skip_performing,
                             cand_bound );
               }
               else
                  // row is not in M
                  continue;
            }

            if( skip_performing )
               break;

            // Only if variable is greater than or equal to new_UB, all rows
            // in M are redundant.
            // I. e. we round up for integer variables.
            if( cflags[i].test( ColFlag::kIntegral ) )
               cand_bound = num.epsCeil( cand_bound );

            if( !new_ub_init || cand_bound > new_ub )
            {
               new_ub = cand_bound;
               new_ub_init = true;
               best_row = row;

               // check if bound is already equal or worse than current bound
               // and abort in that case
               if( ( !cflags[i].test( ColFlag::kUbInf ) &&
                     num.isGE( new_ub, ubs[i] ) ) ||
                   new_ub >= num.getHugeVal() )
               {
                  skip_performing = true;
                  break;
               }
            }
         }

         // set new upper bound
         if( !skip_performing && new_ub_init && !num.isHugeVal( new_ub ) )
         {
            assert( cflags[i].test( ColFlag::kUbInf ) || new_ub < ubs[i] );

            // cannot detect infeasibility with this method, so at most
            // tighten the bound to the lower bound
            if( !cflags[i].test( ColFlag::kLbInf ) )
               new_ub = num.max( lbs[i], new_ub );

            // A transaction is only needed to group several reductions that
            // belong together

            TransactionGuard<REAL> guard{ reductions };

            if( skip_variable_tightening )
               new_ub += bound_tightening_offset;

            assert(best_row >= 0);
            reductions.lockCol( i );
            reductions.lockColBounds( i );
            reductions.changeColUB( i, new_ub, best_row );
            Message::debug( this, "tightened upper bound of col {} to {}\n", i,
                            double( new_ub ) );

            return PresolveStatus::kReduced;

            // If new upper bound is set, we continue with the next column.
            // Although, If c=0, we can try to derive an additional lower
            // bound it will conflict with the locks of this reduction and
            // hence will never be applied.
         }
      }

      // If c_i <= 0, we might derive a tighter lower bound.
      // We consider only rows of
      // M := { (a_ji > 0 and rhs != inf) or (a_ji < 0 and lhs != inf)}.
      // If all constraints in M get redundant for x_i = new_LB, the lower
      // bound can be set to new_LB.
      if( objective[i] <= 0 )
      {
         // in case of skip_variable_tightening set bounds to an infinite value
         // if possible, but increase the bound slightly
         if( skip_variable_tightening  && ! cflags[i].test( ColFlag::kLbInf ) )
            return PresolveStatus::kUnchanged;
         bool skip_ = false;
         bool new_lb_init = false;
         REAL new_lb = REAL(0);
         int best_row = -1;

         // go through all rows with non-zero entry
         for( int j = 0; j != collen; ++j )
         {
            int row = rowinds[j];
            // candidate for new lower bound
            REAL cand_bound;

            if( consMatrix.isRowRedundant( row ) )
               continue;

            // if row is in M, calculate candidate for new lower bound
            if( values[j] > 0.0 )
            {
               if( !rflags[row].test( RowFlag::kRhsInf ) )
               {
                  check_row( activities[row].ninfmax, activities[row].max,
                             rhs[row], values[j], ubs[i],
                             cflags[i].test( ColFlag::kUbInf ), skip_,
                             cand_bound );
               }
               else
                  // row is not in M
                  continue;
            }
            else
            {
               if( !rflags[row].test( RowFlag::kLhsInf ) )
               {
                  check_row( activities[row].ninfmin, activities[row].min,
                             lhs[row], values[j], ubs[i],
                             cflags[i].test( ColFlag::kUbInf ), skip_,
                             cand_bound );
               }
               else
                  // row is not in M
                  continue;
            }

            if( skip_ )
               break;

            // Only if variable is less than or equal to new_LB, all rows in
            // M are redundant. I. e. we round down for integer variables.
            if( cflags[i].test( ColFlag::kIntegral ) )
               cand_bound = num.epsFloor( cand_bound );

            if( !new_lb_init || cand_bound < new_lb )
            {
               new_lb = cand_bound;
               new_lb_init = true;
               best_row = row;

               // check if bound is already equal or worse than current bound
               // and abort in that case
               if( ( !cflags[i].test( ColFlag::kLbInf ) &&
                     num.isLE( new_lb, lbs[i] ) ) ||
                   new_lb <= -num.getHugeVal() )
               {
                  skip_ = true;
                  break;
               }
            }
         }

         // set new lower bound
         if( !skip_ && new_lb_init && !num.isHugeVal( new_lb ) )
         {
            assert( cflags[i].test( ColFlag::kLbInf ) || new_lb > lbs[i] );

            // cannot detect infeasibility with this method, so at most
            // tighten the bound to the upper bound
            if( !cflags[i].test( ColFlag::kUbInf ) )
               new_lb = num.min( ubs[i], new_lb );

            if( skip_variable_tightening )
               new_lb -= bound_tightening_offset;

            // A transaction is only needed to group several reductions that
            // belong together

            TransactionGuard<REAL> guard{ reductions };

            reductions.lockCol( i );
            reductions.lockColBounds( i );
            reductions.changeColLB( i, new_lb, best_row );

            Message::debug( this, "tightened lower bound of col {} to {}\n", i,
                            double( new_lb ) );

            return PresolveStatus::kReduced;
         }
      }
   }
   return result;
}

} // namespace papilo

#endif
