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

#include "papilo/presolvers/ConstraintPropagation.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"

using namespace papilo;

Problem<double>
setupProblemWithConstraintPropagation( bool integer_values );

TEST_CASE( "constraint-propagation-happy-path", "[presolve]" )
{
      double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemWithConstraintPropagation( true );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   presolveOptions.dualreds = 0;
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   ConstraintPropagation<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();
   problemUpdate.trivialPresolve();

#ifndef PAPILO_TBB
   presolveOptions.threads = 1;
#endif

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 8 );

   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::SAVE_ROW );
   REQUIRE( reductions.getReduction( 0 ).row == 0 );

   REQUIRE( reductions.getReduction( 1 ).col == 0 );
   REQUIRE( reductions.getReduction( 1 ).row == ColReduction::UPPER_BOUND );
   REQUIRE( reductions.getReduction( 1 ).newval == 1 );

   REQUIRE( reductions.getReduction( 2 ).col == RowReduction::SAVE_ROW );
   REQUIRE( reductions.getReduction( 2 ).row == 0 );

   REQUIRE( reductions.getReduction( 3 ).row == ColReduction::UPPER_BOUND );
   REQUIRE( reductions.getReduction( 3 ).col == 1 );
   REQUIRE( reductions.getReduction( 3 ).newval == 1 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::SAVE_ROW );
   REQUIRE( reductions.getReduction( 4 ).row == 1 );

   REQUIRE( reductions.getReduction( 5 ).col == 1 );
   REQUIRE( reductions.getReduction( 5 ).row == ColReduction::FIXED );
   REQUIRE( reductions.getReduction( 5 ).newval == 0 );

   REQUIRE( reductions.getReduction( 6 ).col == RowReduction::SAVE_ROW );
   REQUIRE( reductions.getReduction( 6 ).row == 1 );

   REQUIRE( reductions.getReduction( 7 ).col == 2 );
   REQUIRE( reductions.getReduction( 7 ).newval == 0.1 );
   REQUIRE( reductions.getReduction( 7 ).row == ColReduction::UPPER_BOUND );
}

TEST_CASE( "constraint-propagation-no-tightening-for-lp", "[presolve]" )
{
   double time = 0.0;
   Timer t{ time };
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemWithConstraintPropagation( false );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   presolveOptions.dualreds = 0;
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num , presolveOptions);
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   ConstraintPropagation<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();
   problemUpdate.trivialPresolve();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t);

   REQUIRE( presolveStatus == PresolveStatus::kUnchanged );
}

Problem<double>
setupProblemWithConstraintPropagation( bool integer_values )
{
   Vec<double> coefficients{ 1.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 10.0, 10.0, 10.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 0, integer_values, 0 };

   Vec<double> rhs{ 1.0, 2.0 };
   Vec<std::string> rowNames{ "A1", "A2" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 1, 1, 20.0 },
       std::tuple<int, int, double>{ 1, 2, 20.0 },
   };

   ProblemBuilder<double> pb;
   pb.reserve( (int)entries.size(), (int)rowNames.size(),
               (int)columnNames.size() );
   pb.setNumRows( (int)rowNames.size() );
   pb.setNumCols( (int)columnNames.size() );
   pb.setColUbAll( upperBounds );
   pb.setColLbAll( lowerBounds );
   pb.setObjAll( coefficients );
   pb.setObjOffset( 0.0 );
   pb.setColIntegralAll( isIntegral );
   pb.setRowRhsAll( rhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix for testing constraint propagation" );
   Problem<double> problem = pb.build();
   return problem;
}
