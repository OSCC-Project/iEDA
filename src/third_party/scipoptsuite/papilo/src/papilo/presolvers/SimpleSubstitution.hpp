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

#ifndef _PAPILO_PRESOLVERS_SIMPLE_FREEVAR_HPP_
#define _PAPILO_PRESOLVERS_SIMPLE_FREEVAR_HPP_

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"
#include "papilo/misc/Num.hpp"
#include "papilo/misc/fmt.hpp"
#if BOOST_VERSION >= 107000
   #include <boost/integer/extended_euclidean.hpp>
#else
   // use a copy to enable also older boost versions
   #include "papilo/misc/extended_euclidean.hpp"
#endif


// TODO: before this presolver starts feasibility needs to be checked
// TODO: -> maybe do the simple check before? that means inf <= b <= sup
// TODO: 4x + 2y = 4; x,y in {0,1} is substitute false
namespace papilo
{

template <typename REAL>
class SimpleSubstitution : public PresolveMethod<REAL>
{
 public:
   SimpleSubstitution() : PresolveMethod<REAL>()
   {
      this->setName( "doubletoneq" );
      this->setTiming( PresolverTiming::kMedium );
   }

   PresolveStatus
   execute( const Problem<REAL>& problem,
            const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
            Reductions<REAL>& reductions, const Timer& timer ) override;



 private:
   bool
   isConstraintsFeasibleWithGivenBounds(
       const Num<REAL>& num, const Vec<REAL>& lower_bounds,
       const Vec<REAL>& upper_bounds, const REAL* vals, REAL rhs, int subst,
       int stay, const boost::integer::euclidean_result_t<int64_t>& res ) const;

   PresolveStatus
   perform_simple_subsitution_step(
       const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
       Reductions<REAL>& reductions, const VariableDomains<REAL>& domains,
       const Vec<ColFlags>& cflags, const ConstraintMatrix<REAL>& constMatrix,
       const Vec<REAL>& lhs_values, const Vec<REAL>& rhs_values,
       const Vec<REAL>& lower_bounds, const Vec<REAL>& upper_bounds,
       const Vec<RowFlags>& rflags, const Vec<int>& rowperm, int k );
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class SimpleSubstitution<double>;
extern template class SimpleSubstitution<Quad>;
extern template class SimpleSubstitution<Rational>;
#endif

template <typename REAL>
PresolveStatus
SimpleSubstitution<REAL>::execute( const Problem<REAL>& problem,
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

   const auto& constMatrix = problem.getConstraintMatrix();
   const auto& lhs_values = constMatrix.getLeftHandSides();
   const auto& rhs_values = constMatrix.getRightHandSides();
   const auto& rflags = constMatrix.getRowFlags();
   const auto& nrows = constMatrix.getNRows();
   const auto& rowperm = problemUpdate.getRandomRowPerm();

   PresolveStatus result = PresolveStatus::kUnchanged;

#ifndef PAPILO_TBB
   assert( problemUpdate.getPresolveOptions().runs_sequential() );
#endif

   if( problemUpdate.getPresolveOptions().runs_sequential() ||
       !problemUpdate.getPresolveOptions().simple_substitution_parallel )
   {
      for( int k = 0; k < nrows; ++k )
      {
         PresolveStatus s = perform_simple_subsitution_step(
             problemUpdate, num, reductions, domains, cflags, constMatrix,
             lhs_values, rhs_values, lower_bounds, upper_bounds, rflags,
             rowperm, k );
         if( s == PresolveStatus::kReduced )
            result = PresolveStatus::kReduced;
         else if( s == PresolveStatus::kInfeasible )
            return PresolveStatus::kInfeasible;
      }
   }
#ifdef PAPILO_TBB
   else
   {
      PresolveStatus infeasible = PresolveStatus::kUnchanged;
      Vec<Reductions<REAL>> stored_reductions( nrows );
      tbb::parallel_for(
          tbb::blocked_range<int>( 0, nrows ),
          [&]( const tbb::blocked_range<int>& r ) {
             for( int j = r.begin(); j < r.end(); ++j )
             {
                PresolveStatus s = perform_simple_subsitution_step(
                    problemUpdate, num, stored_reductions[j], domains, cflags,
                    constMatrix, lhs_values, rhs_values, lower_bounds,
                    upper_bounds, rflags, rowperm, j );
                assert( s == PresolveStatus::kUnchanged ||
                        s == PresolveStatus::kReduced ||
                        s == PresolveStatus::kInfeasible );
                if( s == PresolveStatus::kReduced )
                   result = PresolveStatus::kReduced;
                else if( s == PresolveStatus::kInfeasible )
                   infeasible = PresolveStatus::kInfeasible;
             }
          } );
      if( infeasible == PresolveStatus::kInfeasible )
         return PresolveStatus::kInfeasible;
      if( result == PresolveStatus::kUnchanged )
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

   return result;
}
template <typename REAL>
PresolveStatus
SimpleSubstitution<REAL>::perform_simple_subsitution_step(
    const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
    Reductions<REAL>& reductions, const VariableDomains<REAL>& domains,
    const Vec<ColFlags>& cflags, const ConstraintMatrix<REAL>& constMatrix,
    const Vec<REAL>& lhs_values, const Vec<REAL>& rhs_values,
    const Vec<REAL>& lower_bounds, const Vec<REAL>& upper_bounds,
    const Vec<RowFlags>& rflags, const Vec<int>& rowperm, int k )
{
   PresolveStatus result;
   int i = rowperm[k];
   // check that equality flag is correct or row is redundant
   assert( rflags[i].test( RowFlag::kRedundant ) ||
           ( !rflags[i].test( RowFlag::kEquation ) &&
             ( lhs_values[i] != rhs_values[i] ||
               rflags[i].test( RowFlag::kLhsInf, RowFlag::kRhsInf ) ) ) ||
           ( rflags[i].test( RowFlag::kEquation ) &&
             !rflags[i].test( RowFlag::kLhsInf, RowFlag::kRhsInf ) &&
             lhs_values[i] == rhs_values[i] ) );

   if( rflags[i].test( RowFlag::kRedundant ) ||
       !rflags[i].test( RowFlag::kEquation ) ||
       constMatrix.getRowSizes()[i] != 2 )
      return PresolveStatus::kUnchanged;

   auto rowvec = constMatrix.getRowCoefficients( i );
   assert( rowvec.getLength() == 2 );
   const REAL* vals = rowvec.getValues();
   const int* inds = rowvec.getIndices();
   REAL rhs = rhs_values[i];

   int subst;
   int stay;

   if( cflags[inds[0]].test( ColFlag::kIntegral ) !=
       cflags[inds[1]].test( ColFlag::kIntegral ) )
   {
      if( cflags[inds[0]].test( ColFlag::kIntegral ) )
         subst = 1;
      else
         subst = 0;

      stay = 1 - subst;
   }
   else if( cflags[inds[0]].test( ColFlag::kIntegral ) )
   {
      assert( cflags[inds[1]].test( ColFlag::kIntegral ) );
      if( abs( vals[0] ) < abs( vals[1] ) ||
          ( abs( vals[0] ) == abs( vals[1] ) &&
            problemUpdate.isColBetterForSubstitution( inds[0], inds[1] ) ) )
         subst = 0;
      else
         subst = 1;

      stay = 1 - subst;

      if( !num.isIntegral( vals[stay] / vals[subst] ) )
      {
         if( !num.isIntegral( vals[stay] ) || !num.isIntegral( vals[subst] ) )
            return PresolveStatus::kUnchanged;
         auto res = boost::integer::extended_euclidean(
            static_cast<int64_t>( abs( vals[stay] ) ),
            static_cast<int64_t>( abs( vals[subst] ) ) );
         if( !num.isIntegral( rhs / res.gcd ) )
            return PresolveStatus::kInfeasible;
         // TODO: ensure isConstraintsFeasibleWithGivenBounds() works for negative sign
         else if( !isConstraintsFeasibleWithGivenBounds(
               num, lower_bounds, upper_bounds, vals, rhs, subst, stay, res ) )
            return PresolveStatus::kInfeasible;
         else
            return PresolveStatus::kUnchanged;
      }
      // problem is infeasible if gcd (i.e. vals[subst]) is not divisor of
      // rhs
      if( !num.isFeasIntegral( rhs / vals[subst] ) )
         return PresolveStatus::kInfeasible;
   }
   else
   {
      REAL absval0 = abs( vals[0] );
      REAL absval1 = abs( vals[1] );
      if( absval0 * problemUpdate.getPresolveOptions().markowitz_tolerance >
          absval1 )
         subst = 0;
      else if( absval1 *
                   problemUpdate.getPresolveOptions().markowitz_tolerance >
               absval0 )
         subst = 1;
      else if( problemUpdate.isColBetterForSubstitution( inds[0], inds[1] ) )
         subst = 0;
      else
         subst = 1;

      stay = 1 - subst;
   }

   result = PresolveStatus::kReduced;

   TransactionGuard<REAL> guard{ reductions };

   reductions.lockRow( i );

   reductions.lockColBounds( inds[subst] );

   REAL s = vals[subst] * vals[stay];
   if( !cflags[inds[subst]].test( ColFlag::kLbInf ) )
   {
      REAL staybound =
          ( rhs - vals[subst] * domains.lower_bounds[inds[subst]] ) /
          vals[stay];
      if( s < 0 && ( cflags[inds[stay]].test( ColFlag::kLbInf ) ||
                     num.isGT( staybound, domains.lower_bounds[inds[stay]] ) ) )
         reductions.changeColLB( inds[stay], staybound );

      if( s > 0 && ( cflags[inds[stay]].test( ColFlag::kUbInf ) ||
                     num.isLT( staybound, domains.upper_bounds[inds[stay]] ) ) )
         reductions.changeColUB( inds[stay], staybound );
   }

   if( !cflags[inds[subst]].test( ColFlag::kUbInf ) )
   {
      REAL staybound =
          ( rhs - vals[subst] * domains.upper_bounds[inds[subst]] ) /
          vals[stay];
      if( s > 0 && ( cflags[inds[stay]].test( ColFlag::kLbInf ) ||
                     staybound > domains.lower_bounds[inds[stay]] ) )
         reductions.changeColLB( inds[stay], staybound );

      if( s < 0 && ( cflags[inds[stay]].test( ColFlag::kUbInf ) ||
                     staybound < domains.upper_bounds[inds[stay]] ) )
         reductions.changeColUB( inds[stay], staybound );
   }

   reductions.aggregateFreeCol( inds[subst], i );
   return result;
}

/**
 * check if the aggregated variable y of the equation a2x1 + a2x2 = b is within its bounds.
 * 1. generate a solution for s a1 + t a2 = gcd(a1,a2)
 * 2. substitute variable x1 = -a2 y + s and x2 = a1 y + t
 * 3. check bounds of y
 *
 * see chapter 10.1.1 Constraint Integer Programming of Tobias Achterberg
 *
 */
template <typename REAL>
bool
SimpleSubstitution<REAL>::isConstraintsFeasibleWithGivenBounds(
    const Num<REAL>& num, const Vec<REAL>& lower_bounds,
    const Vec<REAL>& upper_bounds, const REAL* vals, REAL rhs, int subst,
    int stay, const boost::integer::euclidean_result_t<int64_t>& res ) const
{
   int res_x = vals[stay] < 0 ? res.x * -1 : res.x;
   int res_y = vals[subst] < 0 ? res.y * -1 : res.y;

   REAL initial_solution_for_x = res_x * rhs;
   REAL initial_solution_for_y = res_y * rhs;
   REAL factor = (int)(initial_solution_for_y * res.gcd / vals[stay]);

   REAL s = initial_solution_for_x + factor / res.gcd * vals[subst];
   REAL t = initial_solution_for_y - factor / res.gcd * vals[stay];

   REAL ub_sol_y = ( t - lower_bounds[subst] ) / vals[stay];
   REAL lb_sol_y = ( t - upper_bounds[subst] ) / vals[stay];
   if( vals[stay] < 0 )
      std::swap( ub_sol_y, lb_sol_y );
   REAL ub_sol_x = ( upper_bounds[stay] - s ) / vals[subst];
   REAL lb_sol_x = ( lower_bounds[stay] - s ) / vals[subst];
   if( vals[subst] < 0 )
      std::swap( ub_sol_x, lb_sol_x );

   return num.isFeasLE( num.epsCeil( lb_sol_y ), num.epsFloor( ub_sol_y ) ) &&
          num.isFeasLE( num.epsCeil( lb_sol_x ), num.epsFloor( ub_sol_x ) );
}

} // namespace papilo

#endif
