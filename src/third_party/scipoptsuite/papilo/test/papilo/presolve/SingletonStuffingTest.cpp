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

#include "papilo/presolvers/SingletonStuffing.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"
#include "papilo/core/RowFlags.hpp"

using namespace papilo;

Problem<double>
setupProblemWithSingletonStuffingColumn();

void
forceCalculationOfSingletonStuffingRows( Problem<double>& problem,
                                 ProblemUpdate<double>& problemUpdate )
{
   problem.recomputeLocks();
   problemUpdate.trivialColumnPresolve();
   problem.recomputeAllActivities();
}

TEST_CASE( "singleton-stuffing-make-sure-to-first-set-bounds-to-infinity", "[presolve]" )
{
   Num<double> num{};
   Message msg{};
   double time = 0.0;
   Timer t{ time };
   Problem<double> problem = setupProblemWithSingletonStuffingColumn();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve = PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num , msg);
   presolveOptions.dualreds = 0;
   forceCalculationOfSingletonStuffingRows( problem, problemUpdate );
   SingletonStuffing<double> presolvingMethod{};
   Reductions<double> reductions{};
   presolveOptions.dualreds = 2;


   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 7 );
   REQUIRE( reductions.getReduction( 0 ).col == 3 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).row == 1 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == RowReduction::RHS );
   REQUIRE( reductions.getReduction( 2 ).row == 1 );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );

   REQUIRE( reductions.getReduction( 3 ).col == 3 );
   REQUIRE( reductions.getReduction( 3 ).row == ColReduction::SUBSTITUTE_OBJ );
   REQUIRE( reductions.getReduction( 3 ).newval == 1 );

   REQUIRE( reductions.getReduction( 4 ).col == 3 );
   REQUIRE( reductions.getReduction( 4 ).row == 1 );
   REQUIRE( reductions.getReduction( 4 ).newval == 0 );

   REQUIRE( reductions.getReduction( 5 ).col == RowReduction::RHS_INF );
   REQUIRE( reductions.getReduction( 5 ).row == 1 );
   REQUIRE( reductions.getReduction( 5 ).newval == 0 );

   REQUIRE( reductions.getReduction( 6 ).col == RowReduction::LHS );
   REQUIRE( reductions.getReduction( 6 ).row == 1 );
   REQUIRE( reductions.getReduction( 6 ).newval == 1.98 );
}

Problem<double>
setupProblemWithSingletonStuffingColumn()
{
   Vec<double> coefficients{ 0.0, 0.0, 0.0, -9.0699679999999994 };
   Vec<double> upperBounds{ 0.0, 0.0, 0.0, 0.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0, 1.98 };
   Vec<uint8_t> upper_bound_infinity{ 1, 1, 1, 1 };

   Vec<uint8_t> isIntegral{ 0, 0, 0, 0 };

   Vec<uint8_t> isLefthandsideInfinity{ 0, 0 };
   Vec<uint8_t> isRighthandsideInfinity{ 1, 1 };
   Vec<double> rhs{ 1.0, 0 };
   Vec<double> lhs{ 1.0, -1.6239999999999999 };
   Vec<std::string> rowNames{ "A1", "row with SingletonRow" };
   Vec<std::string> columnNames{ "c1", "c2", "c3", "c4" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 0, 2, 2.0 },
       std::tuple<int, int, double>{ 1, 0, 1.0 },
       std::tuple<int, int, double>{ 1, 1, 1.0 },
       std::tuple<int, int, double>{ 1, 2, 1.0 },
       std::tuple<int, int, double>{ 1, 3, -1.0 },
   };

   ProblemBuilder<double> pb;
   pb.reserve( (int) entries.size(), (int) rowNames.size(), (int) columnNames.size() );
   pb.setNumRows( (int) rowNames.size() );
   pb.setNumCols( (int) columnNames.size() );
   pb.setColUbAll( upperBounds );
   pb.setColUbInfAll( upper_bound_infinity );
   pb.setColLbAll( lowerBounds );
   pb.setObjAll( coefficients );
   pb.setObjOffset( 0.0 );
   pb.setColIntegralAll( isIntegral );
   pb.setRowRhsAll( rhs );
   pb.setRowLhsInfAll( isLefthandsideInfinity );
   pb.setRowRhsInfAll( isRighthandsideInfinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "singleton column" );
   Problem<double> problem = pb.build();
   return problem;
}
