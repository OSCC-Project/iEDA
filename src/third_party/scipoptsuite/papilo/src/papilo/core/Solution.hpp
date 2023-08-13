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

#ifndef _PAPILO_CORE_SOLUTION_HPP_
#define _PAPILO_CORE_SOLUTION_HPP_

#include "papilo/misc/Vec.hpp"

namespace papilo
{

enum class SolutionType
{
   kPrimal,
   kPrimalDual
};

enum class VarBasisStatus : int
{
   ON_UPPER = 0,
   ON_LOWER = 1,
   FIXED = 2,
   ZERO = 3,
   BASIC = 4,
   UNDEFINED = 5
};

template <typename REAL>
class Solution
{
 public:
   SolutionType type;
   Vec<REAL> primal;
   Vec<REAL> dual;
   Vec<REAL> reducedCosts;
   Vec<REAL> slack;
   bool basisAvailabe;
   Vec<VarBasisStatus> varBasisStatus;
   Vec<VarBasisStatus> rowBasisStatus;

   // Default type primal only.
   Solution() : type( SolutionType::kPrimal ), basisAvailabe( false ) {}

   explicit Solution( SolutionType type_ ) : type( type_ ), basisAvailabe( false ) {}

   Solution( SolutionType type_, Vec<REAL> values )
       : type( type_ ), primal( std::move( values ) ), basisAvailabe( false )
   {
   }

   explicit Solution( Vec<REAL> values )
       : type( SolutionType::kPrimal ), primal( std::move( values ) ),
         basisAvailabe( false )
   {
   }

   Solution( Vec<REAL> primal_values, Vec<REAL> dual_values,
             Vec<REAL> reduced_values, Vec<REAL> slack_values,
             bool basisAvailabe_value,
             Vec<VarBasisStatus> var_basis_status,
             Vec<VarBasisStatus> row_basis_status
)
       : type( SolutionType::kPrimalDual ),
         primal( std::move( primal_values ) ),
         dual( std::move( dual_values ),
         reducedCosts( std::move( reduced_values ) ),
         slack( std::move( slack_values ) ) ),
         basisAvailabe( basisAvailabe_value ),
         varBasisStatus( std::move( var_basis_status )),
         rowBasisStatus( std::move( row_basis_status ))

   {
   }
};

} // namespace papilo

#endif
