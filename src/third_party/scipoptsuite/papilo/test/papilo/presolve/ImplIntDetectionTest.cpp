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
#include "papilo/presolvers/ImplIntDetection.hpp"

using namespace papilo;

Problem<double>
setupProblemImplIntDetection();


TEST_CASE( "happy-path-implied-integer-detection", "[presolve]" )
{
      double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemImplIntDetection();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   ImplIntDetection<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();
   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t);
   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 1 );
   REQUIRE( reductions.getReduction( 0 ).col == 0 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::IMPL_INT );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );
}

Problem<double>
setupProblemImplIntDetection()
{
   Num<double> num{};
   Vec<double> coefficients{ 1.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 10.0, 10.0, 10.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 0, 1, 0 };

   Vec<double> rhs{ 1.0, 2.0 };
   Vec<std::string> rowNames{ "A1", "A2" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 1, 1, 20.0 },
       std::tuple<int, int, double>{ 1, 2, 30.0 },
   };

   ProblemBuilder<double> pb;
   pb.reserve( entries.size(), rowNames.size(), columnNames.size() );
   pb.setNumRows( rowNames.size() );
   pb.setNumCols( columnNames.size() );
   pb.setColUbAll( upperBounds );
   pb.setColLbAll( lowerBounds );
   pb.setObjAll( coefficients );
   pb.setObjOffset( 0.0 );
   pb.setColIntegralAll( isIntegral );
   pb.setRowRhsAll( rhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "impl int detection" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0,num, 1 );
   return problem;
}
