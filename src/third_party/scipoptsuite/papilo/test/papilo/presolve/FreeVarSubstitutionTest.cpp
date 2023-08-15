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

#include "papilo/presolvers/FreeVarSubstitution.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"
#include "papilo/core/Reductions.hpp"

#include <tuple>

using namespace papilo;

Problem<double>
setupProblemForFreeVariableSubstitution();


TEST_CASE( "happy-path-test-free-variable-detection", "[presolve]" )
{
   Problem<double> problem = setupProblemForFreeVariableSubstitution();

   double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   problem.recomputeAllActivities();

   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                                presolveOptions, num, msg );

   Substitution<double> presolvingMethod{};
   Reductions<double> reductions{};

   presolvingMethod.initialize( problem, presolveOptions );

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t);

   REQUIRE( presolveStatus == PresolveStatus::kReduced );

   REQUIRE( reductions.size() == 3 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 2 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).col == 3 );
   REQUIRE( reductions.getReduction( 1 ).row ==
            ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == 3 );
   REQUIRE( reductions.getReduction( 2 ).row ==
            ColReduction::SUBSTITUTE );
   REQUIRE( reductions.getReduction( 2 ).newval == 2 );
}

Problem<double>
setupProblemForFreeVariableSubstitution()
{
   // min x + y + z + v + w
   // 2x + y >= 1
   // x + 2z <= 2
   // x + v + w = 1
   // |x| <= 3; y <= 1 ; z >= 0
   Num<double> num{};
   double inf = 10000000;
   Vec<double> coefficients{ 1.0, 1.0, 1.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 3.0, 1.0, inf, inf, inf };
   Vec<double> lowerBounds{ -3.0, -inf, 0.0, -inf, -inf };
   Vec<uint8_t> upperBoundsInfinity{ 0, 0, 1, 1, 1 };
   Vec<uint8_t> lowerBoundsInfinity{ 0, 1, 0, 1, 1 };
   Vec<uint8_t> isIntegral{ 0, 0, 0, 0, 0 };
   Vec<double> rhs{ inf, 2.0, 1.0 };
   Vec<double> lhs{ 1.0, -inf, rhs[2] };
   Vec<std::string> rowNames{ "A1", "A2", "A3" };
   Vec<std::string> columnNames{ "x", "y", "z", "v", "w" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 2.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 1, 0, 1.0 },
       std::tuple<int, int, double>{ 1, 2, 1.0 },
       std::tuple<int, int, double>{ 2, 0, 1.0 },
       std::tuple<int, int, double>{ 2, 3, 1.0 },
       std::tuple<int, int, double>{ 2, 4, -1.0 },
   };

   ProblemBuilder<double> pb;
   pb.reserve( entries.size(), rowNames.size(), columnNames.size() );
   pb.setNumRows( rowNames.size() );
   pb.setNumCols( columnNames.size() );
   pb.setColUbAll( upperBounds );
   pb.setColLbAll( lowerBounds );
   pb.setObjAll( coefficients );
   pb.setColLbInfAll( lowerBoundsInfinity );
   pb.setColUbInfAll( upperBoundsInfinity );
   pb.setObjOffset( 0.0 );
   pb.setColIntegralAll( isIntegral );
   pb.setRowRhsAll( rhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix with free variables (3,4)" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 2, num, lhs[2] );
   return problem;
}
