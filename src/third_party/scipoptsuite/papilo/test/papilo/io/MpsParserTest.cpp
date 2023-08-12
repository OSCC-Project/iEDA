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

#include <memory>
#include "papilo/io/MpsParser.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"

using namespace papilo;

Problem<double>
setupProblemForCoefficientStrengthening();

TEST_CASE( "mps-parser-loading-simple-problem", "[io]" )
{
   boost::optional<Problem<double>> optional = MpsParser<double>::loadProblem(
       "./resources/dual_fix_neg_inf.mps" );
   REQUIRE( optional.is_initialized() == true );
   Problem<double> problem = optional.get();
   Vec<int> expected_row_sizes{2,2,3};
   Vec<int> expected_col_sizes{3,2,2};
   REQUIRE(problem.getConstraintMatrix().getRowSizes() == expected_row_sizes);
   REQUIRE(problem.getConstraintMatrix().getColSizes() == expected_col_sizes);
}
