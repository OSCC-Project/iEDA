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

#include "papilo/external/catch/catch.hpp"
#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"
#include "papilo/presolvers/FixContinuous.hpp"

using namespace papilo;

Problem<double>
setupProblemForTest( double upperBoundForVar3 );

TEST_CASE( "happy-path-presolve-fix-continuous", "[presolve]" )
{
      double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemForTest( num.getFeasTol() / 4 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   FixContinuous<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 2 );
   REQUIRE( reductions.getReduction( 0 ).col == 2 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );
   REQUIRE( reductions.getReduction( 1 ).row == ColReduction::FIXED );
   REQUIRE( reductions.getReduction( 1 ).col == 2 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );
}

TEST_CASE( "happy-path-no-presolve-fix-continuous", "[presolve]" )
{
      double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemForTest( num.getFeasTol() );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   FixContinuous<double> presolvingMethod{};
   Reductions<double> reductions{};
   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );
   REQUIRE( presolveStatus == PresolveStatus::kUnchanged );
   REQUIRE( reductions.size() == 0 );
}

Problem<double>
setupProblemForTest( double upperBoundForVar3 )
{
   Vec<double> coefficients{ 1.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 10.0, 10.0, upperBoundForVar3 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 1, 1, 0 };

   Vec<double> rhs{ 1.0, 2.0 };
   Vec<std::string> rowNames{ "A1", "A2" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 2.0 },
       std::tuple<int, int, double>{ 0, 2, 3.0 },
       std::tuple<int, int, double>{ 1, 1, 2.0 },
   };

   ProblemBuilder<double> pb;
   pb.reserve( 4, 2, 3 );
   pb.setNumRows( 2 );
   pb.setNumCols( 3 );
   pb.setColUbAll( upperBounds );
   pb.setColLbAll( lowerBounds );
   pb.setObjAll( coefficients );
   pb.setObjOffset( 0.0 );
   pb.setColIntegralAll( isIntegral );
   pb.setRowRhsAll( rhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix with potential continuous fix" );
   Problem<double> problem = pb.build();
   return problem;
}
