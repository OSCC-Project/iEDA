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

#ifndef _PAPILO_PRESOLVERS_FIX_CONTINUOUS_HPP_
#define _PAPILO_PRESOLVERS_FIX_CONTINUOUS_HPP_

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"

namespace papilo
{

/// presolver to fix continuous variables whose bounds are very close
template <typename REAL>
class FixContinuous : public PresolveMethod<REAL>
{
 public:
   FixContinuous() : PresolveMethod<REAL>()
   {
      this->setName( "fixcontinuous" );
      this->setTiming( PresolverTiming::kMedium );
      this->setType( PresolverType::kContinuousCols );
   }

   virtual PresolveStatus
   execute( const Problem<REAL>& problem,
            const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
            Reductions<REAL>& reductions, const Timer& timer ) override;
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class FixContinuous<double>;
extern template class FixContinuous<Quad>;
extern template class FixContinuous<Rational>;
#endif

template <typename REAL>
PresolveStatus
FixContinuous<REAL>::execute( const Problem<REAL>& problem,
                              const ProblemUpdate<REAL>& problemUpdate,
                              const Num<REAL>& num,
                              Reductions<REAL>& reductions, const Timer& timer )
{
   assert( problem.getNumContinuousCols() != 0 );

   const auto& consMatrix = problem.getConstraintMatrix();
   const auto& domains = problem.getVariableDomains();
   const auto& cflags = domains.flags;
   const auto& objective = problem.getObjective();
   const auto& lbs = problem.getLowerBounds();
   const auto& ubs = problem.getUpperBounds();

   const int ncols = consMatrix.getNCols();

   PresolveStatus result = PresolveStatus::kUnchanged;
   if( num.getFeasTol() == REAL{ 0 } )
      return result;

   for( int i = 0; i < ncols; ++i )
   {
      // dont fix removed or empty columns, integral columns, columns with
      // infinity bounds, columns that are already fixed, and columns whose
      // bounds are more than epsilon apart
      if( cflags[i].test( ColFlag::kUnbounded, ColFlag::kIntegral,
                          ColFlag::kInactive ) ||
          ( ubs[i] - lbs[i] ) > num.getFeasTol() )
         continue;

      assert( consMatrix.getColSizes()[i] >= 0 );
      assert( lbs[i] != ubs[i] );

      auto colvec = consMatrix.getColumnCoefficients( i );
      REAL maxabsval = num.max( colvec.getMaxAbsValue(), 1 );
      maxabsval = num.max( abs( objective.coefficients[i] ), maxabsval );

      // if the change in activity due to fixing this column is at most
      // epsilon in every row we can fix it
      if( ( ubs[i] - lbs[i] ) * maxabsval <= num.getFeasTol() )
      {
         REAL fixval;

         // if one bound is an integral value, use that one
         if( floor( ubs[i] ) == lbs[i] )
            fixval = lbs[i];
         else if( ceil( lbs[i] ) == ubs[i] )
            fixval = ubs[i];
         else // otherwise take the midpoint
            fixval = REAL{ 0.5 } * ( ubs[i] + lbs[i] );

         TransactionGuard<REAL> tg{ reductions };
         reductions.lockColBounds( i );
         reductions.fixCol( i, fixval );
         result = PresolveStatus::kReduced;
      }
   }

   return result;
}

} // namespace papilo

#endif
