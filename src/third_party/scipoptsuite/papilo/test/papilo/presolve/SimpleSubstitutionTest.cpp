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

#include "papilo/presolvers/SimpleSubstitution.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"

using namespace papilo;

Problem<double>
setupProblemWithSimpleSubstitution( uint8_t is_x_integer, uint8_t is_y_integer,
                                    double a_y );

Problem<double>
setupProblemWithInfeasibleBounds( double x, double y, double rhs, double coef1,
                                  double coef2, double lb1, double ub1,
                                  double lb2, double ub2 );

Problem<double>
setupProblemWithSimpleSubstitutionInfeasibleGcd();

Problem<double>
setupProblemWithSimpleSubstitutionFeasibleGcd();

PresolveStatus
check_gcd_result_with_expectation(double x, double y, double rhs, double coef1,
                                   double coef2, double lb1, double ub1, double lb2, double ub2 );

TEST_CASE( "simple-substitution-happy-path-for-2-int", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem = setupProblemWithSimpleSubstitution( 1, 1, 1.0 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   presolveOptions.dualreds = 0;
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   SimpleSubstitution<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   // Reduction => x = 2 - y/2 -> 0.5 (1 for int) <= x <= 2
   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 5 );

   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 0 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 1 ).col == 1 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == 0 );
   REQUIRE( reductions.getReduction( 2 ).row ==
            papilo::ColReduction::UPPER_BOUND );
   REQUIRE( reductions.getReduction( 2 ).newval == 2 );

   REQUIRE( reductions.getReduction( 3 ).col == 0 );
   REQUIRE( reductions.getReduction( 3 ).row ==
            papilo::ColReduction::LOWER_BOUND );
   REQUIRE( reductions.getReduction( 3 ).newval == 0.5 );

   REQUIRE( reductions.getReduction( 4 ).col == 1 );
   REQUIRE( reductions.getReduction( 4 ).row ==
            papilo::ColReduction::SUBSTITUTE );
   REQUIRE( reductions.getReduction( 4 ).newval == 0 );
}

TEST_CASE( "simple-substitution-happy-path-for-int-continuous-coeff",
           "[presolve]" )
{
   Message msg {};
   double time = 0.0;
   Timer t{ time };
   Num<double> num{};
   Problem<double> problem = setupProblemWithSimpleSubstitution( 1, 1, 2.2 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   presolveOptions.dualreds = 0;
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   SimpleSubstitution<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );
   REQUIRE( presolveStatus == PresolveStatus::kUnchanged );
}

TEST_CASE( "simple-substitution-happy-path-for-2-continuous", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem = setupProblemWithSimpleSubstitution( 0, 0, 1.0 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   presolveOptions.dualreds = 0;
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   SimpleSubstitution<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   // Reduction => x = 4 - 2y -> 0 <= x <= 4 (no further bound relaxation)
   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 3 );

   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 0 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 1 ).col == 0 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == 0 );
   REQUIRE( reductions.getReduction( 2 ).row ==
            papilo::ColReduction::SUBSTITUTE );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
}

TEST_CASE( "simple-substitution-happy-path-for-continuous-and-integer",
           "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem = setupProblemWithSimpleSubstitution( 0, 1, 1.0 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   presolveOptions.dualreds = 0;
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   SimpleSubstitution<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   // Reduction => x = 4 - 2y -> 0 <= x <= 4 (no further bound relaxation)
   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 3 );

   REQUIRE( reductions.getReduction( 0 ).col == RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).row == 0 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 1 ).col == 0 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == 0 );
   REQUIRE( reductions.getReduction( 2 ).row ==
            papilo::ColReduction::SUBSTITUTE );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
}

TEST_CASE( "simple-substitution-simple-substitution-for-2-int", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem = setupProblemWithSimpleSubstitution( 1, 1, 3.0 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   presolveOptions.dualreds = 0;
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   SimpleSubstitution<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kUnchanged );
}

TEST_CASE( "simple-substitution-2-negative-integer", "[presolve]" )
{
   // 2x - 2y = 4 with x,y in [0,3]
   REQUIRE( check_gcd_result_with_expectation(
                1.0, 1.0, 4.0, 2.0, 2.0, 0.0, 3.0, 0.0, 3.0 ) == PresolveStatus::kReduced );
}

TEST_CASE( "simple-substitution-feasible-gcd", "[presolve]" )
{
   // 3x + 8y = 37 with x in {0,7} y in {0,5} -> solution x = 7, y = 2
   REQUIRE( check_gcd_result_with_expectation(
                8.0, 3.0, 37.0, 3.0, 8.0, 0.0, 7.0, 0.0, 5.0 ) == PresolveStatus::kUnchanged );
   // -3x -8y = 37 with x in {-7,0} y in {-5,0} -> solution x = -7, y = -2
   REQUIRE( check_gcd_result_with_expectation(
                8.0, 3.0, 37.0, -3.0, -8.0, -7.0, 0.0, -5.0, 0.0 ) == PresolveStatus::kUnchanged );
   // -3x -8y = -37 with x in {0,7} y in {0,5} -> solution x = 7, y = 2
   REQUIRE( check_gcd_result_with_expectation(
                8.0, 3.0, -37.0, -3.0, -8.0, 0.0, 7.0, 0.0, 5.0 ) == PresolveStatus::kUnchanged );
   // -3x + 8y = 37 with x in {-7,0} y in {0,5} -> solution x = -7, y = 2
   REQUIRE( check_gcd_result_with_expectation(
                8.0, 3.0, 37.0, -3.0, 8.0, -7.0, 0.0, 0.0, 5.0 ) == PresolveStatus::kUnchanged );
   // 3x - 8y = 37 with x in {0,7} y in {-5,0} -> solution x = 7, y = -2
   REQUIRE( check_gcd_result_with_expectation(
                8.0, 3.0, 37.0, 3.0, -8.0, 0.0, 7.0, -5.0, 0.0 ) == PresolveStatus::kUnchanged );
}

TEST_CASE( "simple-substitution-violated-gcd", "[presolve]" )
{
   // -3x - 8y = 37 with x,y in {-5,0}
   REQUIRE( check_gcd_result_with_expectation(
                8.0, 3.0, 37.0, -3.0, 8.0, -5.0, 0.0, -5.0, 0.0 ) == PresolveStatus::kInfeasible );
   // -3x - 8y = -37 with x,y in {0,5}
   REQUIRE( check_gcd_result_with_expectation(
                8.0, 3.0, -37.0, -3.0, -8.0, 0.0, 5.0, 0.0, 5.0 ) == PresolveStatus::kInfeasible );
}


TEST_CASE( "example_10_1_in_constraint_integer_programming", "[presolve]" )
{
   // 3x + 8y = 37 with x,y in {0,5}
   REQUIRE( check_gcd_result_with_expectation(
                8.0, 3.0, 37.0, 3.0, 8.0, 0.0, 5.0, 0.0, 5.0 ) == PresolveStatus::kInfeasible );
}


TEST_CASE( "should_return_feasible_if_gcd_of_coeff_is_in_rhs", "[presolve]" )
{
   Message msg {};
   double time = 0.0;
   Timer t{ time };
   Num<double> num{};
   Problem<double> problem = setupProblemWithSimpleSubstitutionFeasibleGcd();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   presolveOptions.dualreds = 0;
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   SimpleSubstitution<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kUnchanged );
}

TEST_CASE( "should_return_infeasible_if_gcd_of_coeff_is_in_rhs", "[presolve]" )
{
   Message msg {};
   double time = 0.0;
   Timer t{ time };
   Num<double> num{};
   Problem<double> problem = setupProblemWithSimpleSubstitutionInfeasibleGcd();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   presolveOptions.dualreds = 0;
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   SimpleSubstitution<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kInfeasible );
}

PresolveStatus
check_gcd_result_with_expectation( double x, double y, double rhs, double coef1,
                                  double coef2, double lb1, double ub1,
                                  double lb2, double ub2 )
{
   Message msg {};
   double time = 0.0;
   Timer t{ time };
   Num<double> num{};
   Problem<double> problem = setupProblemWithInfeasibleBounds(
       x, y, rhs, coef1, coef2, lb1, ub1, lb2, ub2 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   presolveOptions.dualreds = 0;
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   SimpleSubstitution<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   return
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );
}

Problem<double>
setupProblemWithInfeasibleBounds( double x, double y, double rhs, double coef1,
                                  double coef2, double lb1, double ub1,
                                  double lb2, double ub2 )
{
   Num<double> num{};
   Vec<double> coefficients{ x, y };
   Vec<double> upperBounds{ ub1, ub2 };
   Vec<double> lowerBounds{ lb1, lb2 };
   Vec<uint8_t> isIntegral{ 1, 1 };

   Vec<double> rhs_values{ rhs };
   Vec<std::string> rowNames{ "A1" };
   Vec<std::string> columnNames{ "c1", "c2" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, coef1 },
       std::tuple<int, int, double>{ 0, 1, coef2 },
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
   pb.setRowRhsAll( rhs_values );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "example 10.1 in Constraint Integer Programming" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0,num, rhs );
   return problem;
}

Problem<double>
setupProblemWithSimpleSubstitution( uint8_t is_x_integer, uint8_t is_y_integer,
                                    double a_y )
{
   // 2x + y = 4
   // 0<= x,y y= 3
   Num<double> num{};
   Vec<double> coefficients{ 3.0, 1.0 };
   Vec<double> upperBounds{ 3.0, 3.0 };
   Vec<double> lowerBounds{ 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ is_x_integer, is_y_integer };

   Vec<double> rhs{ 4.0 };
   Vec<std::string> rowNames{ "A1" };
   Vec<std::string> columnNames{ "c1", "c2" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 2.0 },
       std::tuple<int, int, double>{ 0, 1, a_y },
   };

   ProblemBuilder<double> pb;
   pb.reserve( (int) entries.size(), (int) rowNames.size(), (int) columnNames.size() );
   pb.setNumRows( (int) rowNames.size() );
   pb.setNumCols( (int) columnNames.size() );
   pb.setColUbAll( upperBounds );
   pb.setColLbAll( lowerBounds );
   pb.setObjAll( coefficients );
   pb.setObjOffset( 0.0 );
   pb.setColIntegralAll( isIntegral );
   pb.setRowRhsAll( rhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix for testing simple probing" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0,num, rhs[0] );
   return problem;
}

Problem<double>
setupProblemWithSimpleSubstitutionInfeasibleGcd()
{
   // 6x + 8y = 37
   // 0<= x,y y= 5
   Num<double> num{};
   Vec<double> coefficients{ 3.0, 1.0 };
   Vec<double> upperBounds{ 5.0, 5.0 };
   Vec<double> lowerBounds{ 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 1, 1 };

   Vec<double> rhs{ 37.0 };
   Vec<std::string> rowNames{ "A1" };
   Vec<std::string> columnNames{ "c1", "c2" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 6.0 },
       std::tuple<int, int, double>{ 0, 1, 8.0 },
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
   pb.setProblemName( "gcd(x,y) is not divisor of rhs" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0,num, rhs[0] );
   return problem;
}

Problem<double>
setupProblemWithSimpleSubstitutionFeasibleGcd()
{
   // 6x + 9y = 15 with 15/6 and 9/6 no integer
   // 0<= x,y y= 5
   Num<double> num{};
   Vec<double> coefficients{ 3.0, 1.0 };
   Vec<double> upperBounds{ 5.0, 5.0 };
   Vec<double> lowerBounds{ 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 1, 1 };

   Vec<double> rhs{ 15.0 };
   Vec<std::string> rowNames{ "A1" };
   Vec<std::string> columnNames{ "c1", "c2" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 6.0 },
       std::tuple<int, int, double>{ 0, 1, 9.0 },
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
   pb.setProblemName( "gcd(x,y) is divisor of rhs" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0, num, rhs[0] );
   return problem;
}
