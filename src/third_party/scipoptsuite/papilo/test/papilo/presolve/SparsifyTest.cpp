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

#include "papilo/presolvers/Sparsify.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"

using namespace papilo;

Problem<double>
setupProblemWithSparsify();

Problem<double>
setupProblemWithSparsify2Equalities();

Problem<double>
setupProblemForSparsifyWithOneMiss( uint8_t binary );

Problem<double>
setupProblemForSparsifyWithOneMiss_2( uint8_t binary );

Problem<double>
setupProblemForSparsifyWithContinuousVariableMissTwo();

TEST_CASE( "happy-path-sparsify", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem = setupProblemWithSparsify();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   Sparsify<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );
   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 3 );
   REQUIRE( reductions.getReduction( 0 ).row == 0 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 0 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::SPARSIFY );
   REQUIRE( reductions.getReduction( 1 ).newval == 1 );

   REQUIRE( reductions.getReduction( 2 ).row == 1 );
   REQUIRE( reductions.getReduction( 2 ).col == RowReduction::NONE );
   REQUIRE( reductions.getReduction( 2 ).newval == -1 );
}

TEST_CASE( "happy-path-sparsify-two-equalities", "[presolve]" )
{
   double time = 0.0;
   Timer t{ time };
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemWithSparsify2Equalities();
   const PresolveOptions presolveOptions {};
   Statistics statistics{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   Sparsify<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );
   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 6 );
   REQUIRE( reductions.getReduction( 0 ).row == 0 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 0 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::SPARSIFY );
   REQUIRE( reductions.getReduction( 1 ).newval == 1 );

   REQUIRE( reductions.getReduction( 2 ).row == 1 );
   REQUIRE( reductions.getReduction( 2 ).col == RowReduction::NONE );
   REQUIRE( reductions.getReduction( 2 ).newval == -3 );

   REQUIRE( reductions.getReduction( 0 ).row == 0 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 0 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::SPARSIFY );
   REQUIRE( reductions.getReduction( 1 ).newval == 1 );

   REQUIRE( reductions.getReduction( 2 ).row == 1 );
   REQUIRE( reductions.getReduction( 2 ).col == RowReduction::NONE );
   REQUIRE( reductions.getReduction( 2 ).newval == -3 );
}

TEST_CASE( "failed-path-sparsify-if-misses-one-for-integer", "[presolve]" )
{
   Num<double> num{};
   Message msg{};
   double time = 0.0;
   Timer t{ time };

   Problem<double> problem = setupProblemForSparsifyWithOneMiss( 1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   Sparsify<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );
   REQUIRE( presolveStatus == PresolveStatus::kUnchanged );
}


//TODO: test disabled
TEST_CASE( "happy-path-sparsify-if-misses-one-for-continuous", "[presolve]" )
{
   Num<double> num{};
   Message msg{};
   double time = 0.0;
   Timer t{ time };

   Problem<double> problem = setupProblemForSparsifyWithOneMiss( 0 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   Sparsify<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

//   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );
//   REQUIRE( presolveStatus == PresolveStatus::kReduced );
}

TEST_CASE( "happy-path-sparsify-if-misses-one-for-continuous_2", "[presolve]" )
{
   Num<double> num{};
   Message msg{};
   double time = 0.0;
   Timer t{ time };

   Problem<double> problem = setupProblemForSparsifyWithOneMiss_2( 0 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   Sparsify<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );
   // TODO:
   REQUIRE( presolveStatus == PresolveStatus::kReduced );
}

TEST_CASE( "failed-path-sparsify-if-misses-two-for-continuous", "[presolve]" )
{
   Num<double> num{};
   Message msg{};
   double time = 0.0;
   Timer t{ time };

   Problem<double> problem =
       setupProblemForSparsifyWithContinuousVariableMissTwo();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg);
   Sparsify<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );
   REQUIRE( presolveStatus == PresolveStatus::kUnchanged );
}

Problem<double>
setupProblemWithSparsify()
{
   Num<double> num{};
   Vec<double> coefficients{ 3.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 3.0, 3.0, 3.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };

   Vec<double> rhs{ 4.0, 2.0 };
   Vec<std::string> rowNames{ "r1", "r2" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 1, 0, 1.0 },
       std::tuple<int, int, double>{ 1, 1, 1.0 },
       std::tuple<int, int, double>{ 1, 2, 1.0 },
   };

   ProblemBuilder<double> pb;
   pb.reserve( entries.size(), rowNames.size(), columnNames.size() );
   pb.setNumRows( rowNames.size() );
   pb.setNumCols( columnNames.size() );
   pb.setColUbAll( upperBounds );
   pb.setColLbAll( lowerBounds );
   pb.setObjAll( coefficients );
   pb.setObjOffset( 0.0 );
   pb.setRowRhsAll( rhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix for testing sparsify" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0,num, rhs[0] );
   problem.getConstraintMatrix().modifyLeftHandSide( 1, num, rhs[1] );
   return problem;
}

Problem<double>
setupProblemWithSparsify2Equalities()
{
   Num<double> num{};
   Vec<double> coefficients{ 3.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 3.0, 3.0, 3.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };

   Vec<double> rhs{ 4.0, 2.0 };
   Vec<std::string> rowNames{ "r1", "r2" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 1, 0, 3.0 },
       std::tuple<int, int, double>{ 1, 1, 3.0 },
   };

   ProblemBuilder<double> pb;
   pb.reserve( entries.size(), rowNames.size(), columnNames.size() );
   pb.setNumRows( rowNames.size() );
   pb.setNumCols( columnNames.size() );
   pb.setColUbAll( upperBounds );
   pb.setColLbAll( lowerBounds );
   pb.setObjAll( coefficients );
   pb.setObjOffset( 0.0 );
   pb.setRowRhsAll( rhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix for testing sparsify" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0, num, rhs[0] );
   problem.getConstraintMatrix().modifyLeftHandSide( 1, num, rhs[1] );
   return problem;
}

Problem<double>
setupProblemForSparsifyWithOneMiss_2( uint8_t binary )
{
   Num<double> num{};
   Vec<double> coefficients{ 3.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 3.0, 3.0, 3.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0 };
   Vec<uint8_t> integral{ binary, binary, binary };

   Vec<double> rhs{ 4.0, 2.0 };
   Vec<std::string> rowNames{ "r1", "r2" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 1, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 0, 2, 1.0 },
       std::tuple<int, int, double>{ 1, 2, 1.0 },
       std::tuple<int, int, double>{ 1, 1, 1.0 },
   };

   ProblemBuilder<double> pb;
   pb.reserve( entries.size(), rowNames.size(), columnNames.size() );
   pb.setNumRows( rowNames.size() );
   pb.setNumCols( columnNames.size() );
   pb.setColUbAll( upperBounds );
   pb.setColLbAll( lowerBounds );
   pb.setColIntegralAll( integral );
   pb.setObjAll( coefficients );
   pb.setObjOffset( 0.0 );
   pb.setRowRhsAll( rhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "Sparsify test: only 2 columns" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0, num, rhs[0] );
   return problem;
}

Problem<double>
setupProblemForSparsifyWithOneMiss( uint8_t binary )
{
   Num<double> num{};
   Vec<double> coefficients{ 3.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 3.0, 3.0, 3.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0 };
   Vec<uint8_t> integral{ binary, binary, binary };

   Vec<double> rhs{ 4.0, 2.0 };
   Vec<std::string> rowNames{ "r1", "r2" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 0, 2, 1.0 },
       std::tuple<int, int, double>{ 1, 2, 1.0 },
       std::tuple<int, int, double>{ 1, 1, 1.0 },
   };

   ProblemBuilder<double> pb;
   pb.reserve( entries.size(), rowNames.size(), columnNames.size() );
   pb.setNumRows( rowNames.size() );
   pb.setNumCols( columnNames.size() );
   pb.setColUbAll( upperBounds );
   pb.setColLbAll( lowerBounds );
   pb.setColIntegralAll( integral );
   pb.setObjAll( coefficients );
   pb.setObjOffset( 0.0 );
   pb.setRowRhsAll( rhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "Sparsify test: only 2 columns" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0, num, rhs[0] );
   return problem;
}

Problem<double>
setupProblemForSparsifyWithContinuousVariableMissTwo()
{
   Num<double> num{};
   Vec<double> coefficients{ 3.0, 1.0, 1.0 , 1.0};
   Vec<double> upperBounds{ 3.0, 3.0, 3.0, 4.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0, 0 };

   Vec<double> rhs{ 4.0, 2.0 };
   Vec<std::string> rowNames{ "r1", "r2" };
   Vec<std::string> columnNames{ "c1", "c2", "c3", "c4" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 0, 2, 1.0 },
       std::tuple<int, int, double>{ 0, 3, 1.0 },
       std::tuple<int, int, double>{ 1, 0, 1.0 },
       std::tuple<int, int, double>{ 1, 1, 1.0 },
   };

   ProblemBuilder<double> pb;
   pb.reserve( entries.size(), rowNames.size(), columnNames.size() );
   pb.setNumRows( rowNames.size() );
   pb.setNumCols( columnNames.size() );
   pb.setColUbAll( upperBounds );
   pb.setColLbAll( lowerBounds );
   pb.setObjAll( coefficients );
   pb.setObjOffset( 0.0 );
   pb.setRowRhsAll( rhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "Sparsify test: only 2 columns" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0, num, rhs[0] );
   return problem;
}
