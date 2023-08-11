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

#include "papilo/presolvers/ParallelColDetection.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"

using namespace papilo;

Problem<double>
setupProblemWithParallelColumns( bool first_col_int, bool second_col_int,
                                 double factor, double ub_first_col,
                                 double ub_second_col, double lb_first_col,
                                 double lb_second_col, bool objectiveZero );

Problem<double>
setupProblemWithMultipleParallelColumns();

TEST_CASE( "parallel_col_detection_2_integer_columns", "[presolve]" )
{
   double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemWithParallelColumns(
       true, true, 3.0, 10.0, 10.0, 0.0, 0.0, false );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelColDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 3 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).col == 1 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).col == 0 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).row == ColReduction::PARALLEL );
   REQUIRE( reductions.getReduction( 2 ).col == 1 );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
}

TEST_CASE( "parallel_col_detection_objective_zero", "[presolve]" )
{
   double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemWithParallelColumns(
       true, true, 0.5, 10.0, 10.0, 0.0, 0.0, true );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelColDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 3 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).col == 0 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).col == 1 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).row == ColReduction::PARALLEL );
   REQUIRE( reductions.getReduction( 2 ).col == 0 );
   REQUIRE( reductions.getReduction( 2 ).newval == 1 );
}


TEST_CASE( "parallel_col_detection_2_continuous_columns", "[presolve]" )
{
   double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemWithParallelColumns(
       false, false, 2.0, 10.0, 10.0, 0.0, 0.0, false );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelColDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 3 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).col == 1 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).col == 0 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).row == ColReduction::PARALLEL );
   REQUIRE( reductions.getReduction( 2 ).col == 1 );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
}

TEST_CASE( "parallel_col_detection_int_cont_merge_possible", "[presolve]" )
{
   double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemWithParallelColumns(
       true, false, 2.0, 10.0, 10.0, 0.0, 0.0, false );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelColDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 4 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).col == 1 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).col == 0 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 2 ).col == 1 );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );

   REQUIRE( reductions.getReduction( 3 ).row == ColReduction::PARALLEL );
   REQUIRE( reductions.getReduction( 3 ).col == 1 );
   REQUIRE( reductions.getReduction( 3 ).newval == 0 );
}

TEST_CASE( "parallel_col_detection_cont_int_merge_possible", "[presolve]" )
{
   double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemWithParallelColumns(
       false, true, 0.5, 10.0, 10.0, 0.0, 0.0, false );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelColDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 4 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).col == 0 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).col == 1 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 2 ).col == 0 );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );

   REQUIRE( reductions.getReduction( 3 ).row == ColReduction::PARALLEL );
   REQUIRE( reductions.getReduction( 3 ).col == 0 );
   REQUIRE( reductions.getReduction( 3 ).newval == 1 );
}

TEST_CASE( "parallel_col_detection_cont_int_merge_failed", "[presolve]" )
{
   double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemWithParallelColumns(
       false, true, 1.0, 0.9, 10.0, 0.0, 0.0, false );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelColDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kUnchanged );
}

TEST_CASE( "parallel_col_detection_int_cont_merge_failed", "[presolve]" )
{
   double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemWithParallelColumns(
       true, false, 1.0, 10.0, 0.9, 0.0, 0.0, false );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelColDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kUnchanged );
}

TEST_CASE( "parallel_col_detection_int_merge_failed_hole", "[presolve]" )
{
   double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemWithParallelColumns(
       true, true, 1.33333, 5.0, 7, 3.0, 5.0, false );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelColDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kUnchanged );
}

TEST_CASE( "parallel_col_detection_obj_not_parallel", "[presolve]" )
{
   double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Vec<double> obj = { 3, 2 };
   Problem<double> problem = setupProblemWithParallelColumns(
       true, true, 1.0, 10.0, 10.0, 0.0, 0.0, false );
   problem.setObjective( obj, 0 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelColDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kUnchanged );
}

TEST_CASE( "parallel_col_detection_multiple_parallel_columns", "[presolve]" )
{
   double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemWithMultipleParallelColumns();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelColDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 13 );
   REQUIRE( reductions.getTransactions().size() == 2 );
   REQUIRE( reductions.getTransactions()[0].start == 0 );
   REQUIRE( reductions.getTransactions()[0].end == 10 );
   REQUIRE( reductions.getTransactions()[1].start == 10 );
   REQUIRE( reductions.getTransactions()[1].end == 13 );
   int smallest_cont_column = 4;
   int remaining_integer_col = 1;
   Vec<int> locked_rows = { smallest_cont_column, 6, 2, 0,
                            remaining_integer_col };
   for( int i = 0; i < (int) locked_rows.size(); i++ )
   {

      REQUIRE( reductions.getReduction( i ).row == ColReduction::LOCKED );
      REQUIRE( reductions.getReduction( i ).col == locked_rows[i] );
      REQUIRE( reductions.getReduction( i ).newval == 0 );
   }
   REQUIRE( reductions.getReduction( 5 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 5 ).col == smallest_cont_column );
   REQUIRE( reductions.getReduction( 5 ).newval == 0 );

   for( int i = 1; i < (int) locked_rows.size() - 1; i++ )
   {
      REQUIRE( reductions.getReduction( 5 + i ).row == ColReduction::PARALLEL );
      REQUIRE( reductions.getReduction( 5 + i ).col == locked_rows[i] );
      REQUIRE( reductions.getReduction( 5 + i ).newval ==
               smallest_cont_column );
   }

   REQUIRE( reductions.getReduction( 10 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 10 ).col == 5 );
   REQUIRE( reductions.getReduction( 10 ).newval == 0 );

   REQUIRE( reductions.getReduction( 11 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 11 ).col == remaining_integer_col );
   REQUIRE( reductions.getReduction( 11 ).newval == 0 );

   REQUIRE( reductions.getReduction( 12 ).row == ColReduction::PARALLEL );
   REQUIRE( reductions.getReduction( 12 ).col == 5 );
   REQUIRE( reductions.getReduction( 12 ).newval == remaining_integer_col );
}

Problem<double>
setupProblemWithParallelColumns( bool first_col_int, bool second_col_int,
                                 double factor, double ub_first_col,
                                 double ub_second_col, double lb_first_col,
                                 double lb_second_col, bool objectiveZero )
{
   Vec<double> coefficients{ ! objectiveZero ? 1.0 : 0.0,
                             ! objectiveZero ? 1.0 * factor : 0.0 };
   Vec<double> lowerBounds{ lb_first_col, lb_second_col };
   Vec<double> upperBounds{ ub_first_col, ub_second_col };
   Vec<uint8_t> isIntegral{ first_col_int, second_col_int };

   Vec<double> rhs{ 1.0, 2.0 };
   Vec<std::string> rowNames{
       "A1",
       "A2",
   };
   Vec<std::string> columnNames{ "c1", "c2" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 * factor },
       std::tuple<int, int, double>{ 1, 0, 2.0 },
       std::tuple<int, int, double>{ 1, 1, 2.0 * factor },
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
   pb.setProblemName( "matrix with parallel columns (1 and 2)" );
   Problem<double> problem = pb.build();
   return problem;
}

Problem<double>
setupProblemWithMultipleParallelColumns()
{
   Vec<double> coefficients{ 3.0, 2.0, 3.0, 3.0, 1.0, 4.0, 2.0 };
   Vec<double> lowerBounds{ 0, 1, 2, 3, 4, 5, 6 };
   Vec<double> upperBounds{ 2, 3, 4, 5, 6, 7, 8 };
   Vec<uint8_t> isIntegral{ 0, 1, 0, 1, 0, 1, 0 };

   Vec<double> rhs{ 1.0, 2.0 };
   Vec<std::string> rowNames{
       "A1",
       "A2",
   };
   Vec<std::string> columnNames{ "c1", "i2", "c3", "i4", "c5", "i6", "c7" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 3.0 },
       std::tuple<int, int, double>{ 1, 0, 6.0 },

       std::tuple<int, int, double>{ 0, 1, 2.0 },
       std::tuple<int, int, double>{ 1, 1, 4.0 },

       std::tuple<int, int, double>{ 0, 2, 3.0 },
       std::tuple<int, int, double>{ 1, 2, 6.0 },

       std::tuple<int, int, double>{ 0, 3, 3.0 },
       std::tuple<int, int, double>{ 1, 3, 6.0 },

       std::tuple<int, int, double>{ 0, 4, 1.0 },
       std::tuple<int, int, double>{ 1, 4, 2.0 },

       std::tuple<int, int, double>{ 0, 5, 4.0 },
       std::tuple<int, int, double>{ 1, 5, 8.0 },

       std::tuple<int, int, double>{ 0, 6, 2.0 },
       std::tuple<int, int, double>{ 1, 6, 4.0 },
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
   pb.setProblemName( "matrix with multiple parallel columns" );
   Problem<double> problem = pb.build();
   return problem;
}
