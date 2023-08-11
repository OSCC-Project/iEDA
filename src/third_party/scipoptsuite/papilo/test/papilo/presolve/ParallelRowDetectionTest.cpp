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

#include "papilo/presolvers/ParallelRowDetection.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"

using namespace papilo;

Problem<double>
setupProblemWithNoParallelRows();

Problem<double>
setupParallelRowWithMultipleParallelRows();

Problem<double>
setupParallelRowWithMultipleParallelInequalities( double coeff );

Problem<double>
setupProblemParallelRowWithTwoInequalities( double rhs1, double rhs2,
                                            double lhs1, double lhs2,
                                            double factor1, double factor2 );

Problem<double>
setupParallelRowWithTwoParallelEquations( double rhs_first_row,
                                          double rhs_third_row, double factor1,
                                          double factor3 );

Problem<double>
setupProblemParallelRowWithMixed( bool firstRowEquation, double lhsIneq,
                                  double rhsIneq, double factorIneq,
                                  double factorEquation );

Problem<double>
setupProblemParallelRowWithInfinity( bool firstRowRhsInfinity, double lhs,
                                     double rhs, double factorlhs,
                                     double factorrhs );

TEST_CASE( "parallel-row-unchanged", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem = setupProblemWithNoParallelRows();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kUnchanged );
}

TEST_CASE( "parallel-row-two-equations-infeasible-second-row-dominant",
           "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupParallelRowWithTwoParallelEquations( 2.0, 3.0, 1, 3 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kInfeasible );
}

TEST_CASE( "parallel-row-two-equations-infeasible-first-row-dominant",
           "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupParallelRowWithTwoParallelEquations( 2.0, 3.0, 3, 1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kInfeasible );
}

TEST_CASE( "parallel-row-two-equations-feasible-second-row-dominant",
           "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupParallelRowWithTwoParallelEquations( 1.0, 3.0, 1, 3 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 3 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 2 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 0 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == RowReduction::REDUNDANT );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == 0 );
}

TEST_CASE( "parallel-row-two-equations-feasible-first-row-dominant",
           "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupParallelRowWithTwoParallelEquations( 3.0, 1.0, 3, 1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 3 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 0 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 2 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == RowReduction::REDUNDANT );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == 2 );
}

TEST_CASE( "parallel-row-two-inequalities-redundant-row-second-row-dominant",
           "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithTwoInequalities( 1.0, 3.0, -1.0, -3.0, 1, 3 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 3 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 2 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 0 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == RowReduction::REDUNDANT );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == 0 );
}

TEST_CASE( "parallel-row-two-inequalities-redundant-row-first-row-dominant",
           "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithTwoInequalities( 1.0, 3.0, -1.0, -3.0, 3, 1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 3 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 0 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 2 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == RowReduction::REDUNDANT );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == 2 );
}

TEST_CASE(
    "parallel-row-two-inequalities-tighten-lower-bound-second-row-dominant",
    "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithTwoInequalities( 1.0, 3.0, 0.0, -3.0, 1, 3 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 5 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 2 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 0 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col ==
            RowReduction::REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE );
   REQUIRE( reductions.getReduction( 2 ).newval == 2 );
   REQUIRE( reductions.getReduction( 2 ).row == 0 );

   REQUIRE( reductions.getReduction( 3 ).col ==
            RowReduction::LHS_LESS_RESTRICTIVE );
   REQUIRE( reductions.getReduction( 3 ).newval == 0 );
   REQUIRE( reductions.getReduction( 3 ).row == 2 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::REDUNDANT );
   REQUIRE( reductions.getReduction( 4 ).newval == 0 );
   REQUIRE( reductions.getReduction( 4 ).row == 0 );
}

TEST_CASE(
    "parallel-row-two-inequalities-tighten-lower-bound-first-row-dominant",
    "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithTwoInequalities( 1.0, 3.0, -6.0, -1.0, 3, 1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 5 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 0 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 2 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col ==
            RowReduction::REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == 2 );

   REQUIRE( reductions.getReduction( 3 ).col ==
            RowReduction::LHS_LESS_RESTRICTIVE );
   REQUIRE( reductions.getReduction( 3 ).newval == -3 );
   REQUIRE( reductions.getReduction( 3 ).row == 0 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::REDUNDANT );
   REQUIRE( reductions.getReduction( 4 ).newval == 0 );
   REQUIRE( reductions.getReduction( 4 ).row == 2 );
}

TEST_CASE(
    "parallel-row-two-inequalities-tighten-upper-bound-second-row-dominant",
    "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithTwoInequalities( 1.0, 5.0, 0.0, 0.0, 1, 3 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 5 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 2 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 0 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col ==
            RowReduction::REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE );
   REQUIRE( reductions.getReduction( 2 ).newval == 2 );
   REQUIRE( reductions.getReduction( 2 ).row == 0 );

   REQUIRE( reductions.getReduction( 3 ).col ==
            RowReduction::RHS_LESS_RESTRICTIVE );
   REQUIRE( reductions.getReduction( 3 ).newval == 3 );
   REQUIRE( reductions.getReduction( 3 ).row == 2 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::REDUNDANT );
   REQUIRE( reductions.getReduction( 4 ).newval == 0 );
   REQUIRE( reductions.getReduction( 4 ).row == 0 );
}

TEST_CASE(
    "parallel-row-two-inequalities-tighten-upper-bound-first-row-dominant",
    "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithTwoInequalities( 5.0, 1.0, 0.0, 0.0, 3, 1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 5 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 0 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 2 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col ==
            RowReduction::REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == 2 );

   REQUIRE( reductions.getReduction( 3 ).col ==
            RowReduction::RHS_LESS_RESTRICTIVE );
   REQUIRE( reductions.getReduction( 3 ).newval == 3 );
   REQUIRE( reductions.getReduction( 3 ).row == 0 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::REDUNDANT );
   REQUIRE( reductions.getReduction( 4 ).newval == 0 );
   REQUIRE( reductions.getReduction( 4 ).row == 2 );
}

TEST_CASE( "parallel-row-two-inequalities-infeasible-first-row-dominant",
           "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithTwoInequalities( 10.0, 2.0, 5.0, 0.0, 2, 1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kInfeasible );
}

TEST_CASE( "parallel-row-two-inequalities-infeasible-second-row-dominant",
           "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithTwoInequalities( 7.0, 2.0, 5.0, 0.0, 1, 2 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kInfeasible );
}

TEST_CASE( "parallel-row-two-inequalities-tighten-upper-bound-first-row-neg"
           "factor-dominant",
           "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithTwoInequalities( 5.0, 0.0, 0.0, -1.0, 3, -1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 5 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 0 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 2 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col ==
            RowReduction::REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == 2 );

   REQUIRE( reductions.getReduction( 3 ).col ==
            RowReduction::RHS_LESS_RESTRICTIVE );
   REQUIRE( reductions.getReduction( 3 ).newval == 3 );
   REQUIRE( reductions.getReduction( 3 ).row == 0 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::REDUNDANT );
   REQUIRE( reductions.getReduction( 4 ).newval == 0 );
   REQUIRE( reductions.getReduction( 4 ).row == 2 );
}

TEST_CASE( "parallel-row-overwrite-inf-first-row-rhs-inf", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithInfinity( true, 2, 2, 1, 1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 5 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 0 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 2 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col ==
            RowReduction::REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == 2 );

   REQUIRE( reductions.getReduction( 3 ).col ==
            RowReduction::RHS_LESS_RESTRICTIVE );
   REQUIRE( reductions.getReduction( 3 ).newval == 2 );
   REQUIRE( reductions.getReduction( 3 ).row == 0 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::REDUNDANT );
   REQUIRE( reductions.getReduction( 4 ).newval == 0 );
   REQUIRE( reductions.getReduction( 4 ).row == 2 );
}

TEST_CASE( "parallel-row-overwrite-inf-first-row-lhs-inf", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithInfinity( false, 2, 2, 1, 1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 5 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 0 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 2 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col ==
            RowReduction::REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == 2 );

   REQUIRE( reductions.getReduction( 3 ).col ==
            RowReduction::LHS_LESS_RESTRICTIVE );
   REQUIRE( reductions.getReduction( 3 ).newval == 2 );
   REQUIRE( reductions.getReduction( 3 ).row == 0 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::REDUNDANT );
   REQUIRE( reductions.getReduction( 4 ).newval == 0 );
   REQUIRE( reductions.getReduction( 4 ).row == 2 );
}

TEST_CASE( "parallel-row-overwrite-inf-first-row-lhs-inf-neg-factor",
           "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithInfinity( false, -1, 2, -1, 1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 5 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 0 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 2 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col ==
            RowReduction::REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == 2 );

   REQUIRE( reductions.getReduction( 3 ).col ==
            RowReduction::RHS_LESS_RESTRICTIVE );
   REQUIRE( reductions.getReduction( 3 ).newval == 1 );
   REQUIRE( reductions.getReduction( 3 ).row == 0 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::REDUNDANT );
   REQUIRE( reductions.getReduction( 4 ).newval == 0 );
   REQUIRE( reductions.getReduction( 4 ).row == 2 );
}

TEST_CASE( "parallel-row-mixed-infeasible-first-row-equation", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithMixed( true, 2.2, 5.0, 2, 1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kInfeasible );
}

TEST_CASE( "parallel-row-mixed-second-row-equation", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithMixed( false, 0.0, 5.0, 2, 1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 3 );
   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 2 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 0 );
   REQUIRE( reductions.getReduction( 1 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == RowReduction::REDUNDANT );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == 0 );
}

TEST_CASE( "parallel-row-mixed-infeasible-second-row-equation", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupProblemParallelRowWithMixed( false, 2.2, 5.0, 2, 1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kInfeasible );
}

TEST_CASE( "parallel-row-best-bound-is-used-for-rhs", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupParallelRowWithMultipleParallelInequalities( 1.0 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 7 );
   int locked_rows[] = { 0, 1, 2 };
   for( int i = 0; i < 3; i++ )
   {
      REQUIRE( reductions.getReduction( i ).col == RowReduction::LOCKED );
      REQUIRE( reductions.getReduction( i ).row == locked_rows[i] );
      REQUIRE( reductions.getReduction( i ).newval == 0 );
   }

   REQUIRE( reductions.getReduction( 3 ).col ==
            RowReduction::REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE );

   REQUIRE( reductions.getReduction( 4 ).col ==
            RowReduction::LHS_LESS_RESTRICTIVE );
   REQUIRE( reductions.getReduction( 4 ).row == 0 );
   REQUIRE( reductions.getReduction( 4 ).newval == -1 );

   for( int i = 0; i < 2; i++ )
   {
      REQUIRE( reductions.getReduction( 5 + i ).col ==
               RowReduction::REDUNDANT );
      REQUIRE( reductions.getReduction( 5 + i ).row == locked_rows[i + 1] );
      REQUIRE( reductions.getReduction( 5 + i ).newval == 0 );
   }
}

TEST_CASE( "parallel-row-best-bound-is-used-for-rhs-coeff-not-1", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem =
       setupParallelRowWithMultipleParallelInequalities( 0.1 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 7 );
   int locked_rows[] = { 0, 1, 2 };
   for( int i = 0; i < 3; i++ )
   {
      REQUIRE( reductions.getReduction( i ).col == RowReduction::LOCKED );
      REQUIRE( reductions.getReduction( i ).row == locked_rows[i] );
      REQUIRE( reductions.getReduction( i ).newval == 0 );
   }

   REQUIRE( reductions.getReduction( 3 ).col ==
            RowReduction::REASON_FOR_LESS_RESTRICTIVE_BOUND_CHANGE );

   REQUIRE( reductions.getReduction( 4 ).col ==
            RowReduction::LHS_LESS_RESTRICTIVE );
   REQUIRE( reductions.getReduction( 4 ).row == 0 );
   REQUIRE( reductions.getReduction( 4 ).newval == -1 );

   for( int i = 0; i < 2; i++ )
   {
      REQUIRE( reductions.getReduction( 5 + i ).col ==
               RowReduction::REDUNDANT );
      REQUIRE( reductions.getReduction( 5 + i ).row == locked_rows[i + 1] );
      REQUIRE( reductions.getReduction( 5 + i ).newval == 0 );
   }
}

TEST_CASE( "parallel-row-multiple-parallel-rows", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem = setupParallelRowWithMultipleParallelRows();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.checkChangedActivities();
   ParallelRowDetection<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 7 );
   int locked_rows[] = { 1, 2, 0, 3 };
   for( int i = 0; i < 4; i++ )
   {
      REQUIRE( reductions.getReduction( i ).col == RowReduction::LOCKED );
      REQUIRE( reductions.getReduction( i ).row == locked_rows[i] );
      REQUIRE( reductions.getReduction( i ).newval == 0 );
   }

   for( int i = 0; i < 3; i++ )
   {
      REQUIRE( reductions.getReduction( 4 + i ).col ==
               RowReduction::REDUNDANT );
      REQUIRE( reductions.getReduction( 4 + i ).row == locked_rows[i + 1] );
      REQUIRE( reductions.getReduction( 4 + i ).newval == 0 );
   }
}

Problem<double>
setupProblemWithNoParallelRows()
{
   Vec<double> coefficients{ 1.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 10.0, 10.0, 10.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 1, 1, 1 };

   Vec<double> rhs{ 1.0, 2.0, 3.0 };
   Vec<std::string> rowNames{ "A1", "A2", "A3" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 0, 2, 2.0 },

       std::tuple<int, int, double>{ 1, 1, 2.0 },
       std::tuple<int, int, double>{ 1, 1, 2.0 },

       std::tuple<int, int, double>{ 2, 0, 3.0 },
       std::tuple<int, int, double>{ 2, 1, 3.0 },
       std::tuple<int, int, double>{ 2, 2, 5.0 },
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
   pb.setProblemName( "matrix with no parallel rows" );
   Problem<double> problem = pb.build();
   return problem;
}

Problem<double>
setupParallelRowWithTwoParallelEquations( double rhs_first_row,
                                          double rhs_third_row, double factor1,
                                          double factor3 )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Vec<double> coefficients{ 1.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 10.0, 10.0, 10.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 1, 1, 1 };

   Vec<double> rhs{ rhs_first_row, 2.0, rhs_third_row };
   Vec<std::string> rowNames{ "A1", "A2", "A3" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 * factor1 },
       std::tuple<int, int, double>{ 0, 1, 1.0 * factor1 },
       std::tuple<int, int, double>{ 0, 2, 2.0 * factor1 },

       std::tuple<int, int, double>{ 1, 1, 2.0 },
       std::tuple<int, int, double>{ 1, 1, 2.0 },

       std::tuple<int, int, double>{ 2, 0, 1.0 * factor3 },
       std::tuple<int, int, double>{ 2, 1, 1.0 * factor3 },
       std::tuple<int, int, double>{ 2, 2, 2.0 * factor3 },
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
   pb.setProblemName( "matrix with parallel equations (0 and 2)" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0, num, rhs[0] );
   problem.getConstraintMatrix().modifyLeftHandSide( 2, num, rhs[2] );
   return problem;
}

Problem<double>
setupParallelRowWithMultipleParallelRows()
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Vec<double> coefficients{ 1.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 10.0, 10.0, 10.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 1, 1, 1 };

   Vec<double> rhs{ 1, 7.0, 1, 4 };
   Vec<std::string> rowNames{ "A1", "A2", "A3", "A4" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 0, 2, 2.0 },

       std::tuple<int, int, double>{ 1, 0, 7.0 },
       std::tuple<int, int, double>{ 1, 1, 7.0 },
       std::tuple<int, int, double>{ 1, 2, 14.0 },

       std::tuple<int, int, double>{ 2, 0, 1.0 },
       std::tuple<int, int, double>{ 2, 1, 1.0 },
       std::tuple<int, int, double>{ 2, 2, 2.0 },

       std::tuple<int, int, double>{ 3, 0, 2.0 },
       std::tuple<int, int, double>{ 3, 1, 2.0 },
       std::tuple<int, int, double>{ 3, 2, 4.0 },
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
   pb.setProblemName( "matrix with parallel equations (0 and 2)" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0, num, rhs[0] );
   problem.getConstraintMatrix().modifyLeftHandSide( 1, num, rhs[1] );
   return problem;
}

Problem<double>
setupParallelRowWithMultipleParallelInequalities( double coeff )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Vec<double> coefficients{ 1.0, 1.0 };
   Vec<double> upperBounds{ 10.0, 10.0 };
   Vec<double> lowerBounds{ 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 0, 0 };

   Vec<double> rhs{ 0, 1, 2 };
   Vec<uint8_t> lhs_inf{ 1, 1, 1 };
   Vec<std::string> rowNames{ "A1", "A2", "A3" };
   Vec<std::string> columnNames{ "c1", "c2" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, -1.0 },

       std::tuple<int, int, double>{ 1, 0, -1.0 },
       std::tuple<int, int, double>{ 1, 1, 1.0 },

       std::tuple<int, int, double>{ 2, 0, -coeff },
       std::tuple<int, int, double>{ 2, 1, coeff },

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
   pb.setRowLhsInfAll( lhs_inf );
   pb.setRowRhsAll( rhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "parallel rows" );
   Problem<double> problem = pb.build();
   return problem;
}

Problem<double>
setupProblemParallelRowWithTwoInequalities( double rhs1, double rhs2,
                                            double lhs1, double lhs2,
                                            double factor1, double factor2 )
{
   Vec<double> coefficients{ 1.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 10.0, 10.0, 10.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 1, 1, 1 };

   Vec<double> rhs{ rhs1, 2.0, rhs2 };
   Vec<double> lhs{ lhs1, 2.0, lhs2 };
   Vec<std::string> rowNames{ "A1", "A2", "A3" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 * factor1 },
       std::tuple<int, int, double>{ 0, 1, 1.0 * factor1 },
       std::tuple<int, int, double>{ 0, 2, 2.0 * factor1 },

       std::tuple<int, int, double>{ 1, 1, 2.0 },
       std::tuple<int, int, double>{ 1, 1, 2.0 },

       std::tuple<int, int, double>{ 2, 0, 1.0 * factor2 },
       std::tuple<int, int, double>{ 2, 1, 1.0 * factor2 },
       std::tuple<int, int, double>{ 2, 2, 2.0 * factor2 },
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
   pb.setRowLhsAll( lhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix with parallel inequalities (0 and 2)" );
   Problem<double> problem = pb.build();
   return problem;
}

Problem<double>
setupProblemParallelRowWithMixed( bool firstRowEquation, double lhsIneq,
                                  double rhsIneq, double factorIneq,
                                  double factorEquation )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Vec<double> coefficients{ 1.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 10.0, 10.0, 10.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };

   Vec<uint8_t> isIntegral{ 1, 1, 1 };
   double rhs1 = firstRowEquation ? factorEquation : rhsIneq;
   double lhs1 = firstRowEquation ? factorEquation : lhsIneq;
   double rhs2 = ( !firstRowEquation ) ? factorEquation : rhsIneq;
   double lhs2 = ( !firstRowEquation ) ? factorEquation : lhsIneq;
   double factor1 = firstRowEquation ? factorEquation : factorIneq;
   double factor2 = ( !firstRowEquation ) ? factorEquation : factorIneq;
   Vec<double> rhs{ rhs1, 2.0, rhs2 };
   Vec<double> lhs{ lhs1, 2.0, lhs2 };
   Vec<std::string> rowNames{ "A1", "A2", "A3" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 * factor1 },
       std::tuple<int, int, double>{ 0, 1, 1.0 * factor1 },
       std::tuple<int, int, double>{ 0, 2, 2.0 * factor1 },

       std::tuple<int, int, double>{ 1, 1, 2.0 },
       std::tuple<int, int, double>{ 1, 1, 2.0 },

       std::tuple<int, int, double>{ 2, 0, 1.0 * factor2 },
       std::tuple<int, int, double>{ 2, 1, 1.0 * factor2 },
       std::tuple<int, int, double>{ 2, 2, 2.0 * factor2 },
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
   pb.setRowLhsAll( lhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix with parallel inequalities (0 and 2)" );
   Problem<double> problem = pb.build();
   int i = firstRowEquation ? i = 0 : i = 2;
   problem.getConstraintMatrix().modifyLeftHandSide( i, num, rhs[i] );
   return problem;
}

Problem<double>
setupProblemParallelRowWithInfinity( bool firstRowRhsInfinity, double lhs,
                                     double rhs, double factorlhs,
                                     double factorrhs )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Vec<double> coefficients{ 1.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 10.0, 10.0, 10.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };

   Vec<uint8_t> isIntegral{ 1, 1, 1 };
   Vec<uint8_t> lhs_infinity{ !firstRowRhsInfinity, 1, firstRowRhsInfinity };
   Vec<uint8_t> rhs_infinity{ firstRowRhsInfinity, 1, !firstRowRhsInfinity };
   double factor1 = firstRowRhsInfinity ? factorlhs : factorrhs;
   double factor2 = ( !firstRowRhsInfinity ) ? factorlhs : factorrhs;
   Vec<double> rhs_all{ rhs, 2.0, rhs };
   Vec<double> lhs_all{ lhs, 2.0, lhs };
   Vec<std::string> rowNames{ "A1", "A2", "A3" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 * factor1 },
       std::tuple<int, int, double>{ 0, 1, 1.0 * factor1 },
       std::tuple<int, int, double>{ 0, 2, 2.0 * factor1 },

       std::tuple<int, int, double>{ 1, 1, 2.0 },
       std::tuple<int, int, double>{ 1, 1, 2.0 },

       std::tuple<int, int, double>{ 2, 0, 1.0 * factor2 },
       std::tuple<int, int, double>{ 2, 1, 1.0 * factor2 },
       std::tuple<int, int, double>{ 2, 2, 2.0 * factor2 },
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
   pb.setRowRhsAll( rhs_all );
   pb.setRowLhsAll( lhs_all );
   pb.setRowLhsInfAll( lhs_infinity );
   pb.setRowRhsInfAll( rhs_infinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix with parallel inequalities (0 and 2)" );
   Problem<double> problem = pb.build();
   return problem;
}
