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

#include "papilo/presolvers/SingletonCols.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"
#include "papilo/core/RowFlags.hpp"

using namespace papilo;

Problem<double>
setupProblemWithOnlyOneEntryIn1stRowAndColumn();

Problem<double>
setupProblemWithSingletonColumn();

Problem<double>
setupProblemWithSingletonColumnInEquationWithNoImpliedBounds(
    double coefficient, double upper_bound, double lower_bound );

Problem<double>
setupProblemWithSingletonColumnInEquationWithInfinityBounds();

void
forceCalculationOfSingletonRows( Problem<double>& problem,
                                 ProblemUpdate<double>& problemUpdate )
{
   problem.recomputeLocks();
   problemUpdate.trivialColumnPresolve();
   problem.recomputeAllActivities();
}

TEST_CASE( "happy-path-singleton-column", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem = setupProblemWithSingletonColumn();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num , msg);
   forceCalculationOfSingletonRows( problem, problemUpdate );
   SingletonCols<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 5 );
   REQUIRE( reductions.getReduction( 0 ).col == 0 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == 0 );
   REQUIRE( reductions.getReduction( 1 ).col == papilo::RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == 0 );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == ColReduction::SUBSTITUTE_OBJ );

   // in matrix entry (0,0) new value 0
   REQUIRE( reductions.getReduction( 3 ).col == 0 );
   REQUIRE( reductions.getReduction( 3 ).newval == 0 );
   REQUIRE( reductions.getReduction( 3 ).row == 0 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::LHS_INF );
   REQUIRE( reductions.getReduction( 4 ).newval == 0 );
   REQUIRE( reductions.getReduction( 4 ).row == 0 );
}

TEST_CASE( "happy-path-singleton-column-equation", "[presolve]" )
{
   Num<double> num{};
   double time = 0.0;
   Timer t{ time };
   Message msg{};
   Problem<double> problem = setupProblemWithOnlyOneEntryIn1stRowAndColumn();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   forceCalculationOfSingletonRows( problem, problemUpdate );
   SingletonCols<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 5 );
   REQUIRE( reductions.getReduction( 0 ).col == 2 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).col == papilo::RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).row == 1 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == 2 );
   REQUIRE( reductions.getReduction( 2 ).row == ColReduction::SUBSTITUTE_OBJ );
   REQUIRE( reductions.getReduction( 2 ).newval == 1 );

   REQUIRE( reductions.getReduction( 3 ).col == 2 );
   REQUIRE( reductions.getReduction( 3 ).row == 1 );
   REQUIRE( reductions.getReduction( 3 ).newval == 0 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::LHS_INF );
   REQUIRE( reductions.getReduction( 4 ).row == 1 );
   REQUIRE( reductions.getReduction( 4 ).newval == 0 );
}

TEST_CASE( "happy-path-singleton-column-implied-bounds-negative-coeff-pos-bounds", "[presolve]" )
{
   Message msg{};
   double time = 0.0;
   Timer t{ time };
   const Num<double> num{};
   Problem<double> problem =
       setupProblemWithSingletonColumnInEquationWithNoImpliedBounds( -1.0, 10.0,
                                                                     3.0 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   forceCalculationOfSingletonRows( problem, problemUpdate );
   SingletonCols<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 6 );
   REQUIRE( reductions.getReduction( 0 ).col == 0 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).col == papilo::RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).row == 0 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == ColReduction::SUBSTITUTE_OBJ );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );

   REQUIRE( reductions.getReduction( 3 ).col == 0 );
   REQUIRE( reductions.getReduction( 3 ).row == 0 );
   REQUIRE( reductions.getReduction( 3 ).newval == 0 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::RHS );
   REQUIRE( reductions.getReduction( 4 ).row == 0 );
   REQUIRE( reductions.getReduction( 4 ).newval == 11 );

   REQUIRE( reductions.getReduction( 5 ).col == RowReduction::LHS );
   REQUIRE( reductions.getReduction( 5 ).row == 0 );
   REQUIRE( reductions.getReduction( 5 ).newval == 4 );
}

TEST_CASE( "happy-path-singleton-column-implied-bounds-negative-coeff-neg-bounds", "[presolve]" )
{
   Message msg{};
   double time = 0.0;
   Timer t{ time };
   const Num<double> num{};
   Problem<double> problem =
       setupProblemWithSingletonColumnInEquationWithNoImpliedBounds( -1.0, -3.0,
                                                                     -10.0 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   forceCalculationOfSingletonRows( problem, problemUpdate );
   SingletonCols<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 6 );
   REQUIRE( reductions.getReduction( 0 ).col == 0 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).col == papilo::RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).row == 0 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == ColReduction::SUBSTITUTE_OBJ );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );

   REQUIRE( reductions.getReduction( 3 ).col == 0 );
   REQUIRE( reductions.getReduction( 3 ).row == 0 );
   REQUIRE( reductions.getReduction( 3 ).newval == 0 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::LHS );
   REQUIRE( reductions.getReduction( 4 ).row == 0 );
   REQUIRE( reductions.getReduction( 4 ).newval == -9 );

   REQUIRE( reductions.getReduction( 5 ).col == RowReduction::RHS );
   REQUIRE( reductions.getReduction( 5 ).row == 0 );
   REQUIRE( reductions.getReduction( 5 ).newval == -2 );
}

TEST_CASE( "happy-path-singleton-column-implied-bounds-positive-coeff-pos-bounds", "[presolve]" )
{
   Message msg{};
   double time = 0.0;
   Timer t{ time };
   const Num<double> num{};
   Problem<double> problem =
       setupProblemWithSingletonColumnInEquationWithNoImpliedBounds( 1.0, 10.0,
                                                                     3.0 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   forceCalculationOfSingletonRows( problem, problemUpdate );
   SingletonCols<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 6 );
   REQUIRE( reductions.getReduction( 0 ).col == 0 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).col == papilo::RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).row == 0 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == ColReduction::SUBSTITUTE_OBJ );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );

   REQUIRE( reductions.getReduction( 3 ).col == 0 );
   REQUIRE( reductions.getReduction( 3 ).row == 0 );
   REQUIRE( reductions.getReduction( 3 ).newval == 0 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::LHS );
   REQUIRE( reductions.getReduction( 4 ).row == 0 );
   REQUIRE( reductions.getReduction( 4 ).newval == -9 );

   REQUIRE( reductions.getReduction( 5 ).col == RowReduction::RHS );
   REQUIRE( reductions.getReduction( 5 ).row == 0 );
   REQUIRE( reductions.getReduction( 5 ).newval == -2 );

}


TEST_CASE( "happy-path-singleton-column-implied-bounds-positive-coeff-neg-bounds", "[presolve]" )
{
   Message msg{};
   double time = 0.0;
   Timer t{ time };
   const Num<double> num{};
   Problem<double> problem =
       setupProblemWithSingletonColumnInEquationWithNoImpliedBounds( 1.0, -3.0,
                                                                     -10.0 );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   forceCalculationOfSingletonRows( problem, problemUpdate );
   SingletonCols<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 6 );
   REQUIRE( reductions.getReduction( 0 ).col == 0 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).col == papilo::RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).row == 0 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == ColReduction::SUBSTITUTE_OBJ );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );

   REQUIRE( reductions.getReduction( 3 ).col == 0 );
   REQUIRE( reductions.getReduction( 3 ).row == 0 );
   REQUIRE( reductions.getReduction( 3 ).newval == 0 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::RHS );
   REQUIRE( reductions.getReduction( 4 ).row == 0 );
   REQUIRE( reductions.getReduction( 4 ).newval == 11 );

   REQUIRE( reductions.getReduction( 5 ).col == RowReduction::LHS );
   REQUIRE( reductions.getReduction( 5 ).row == 0 );
   REQUIRE( reductions.getReduction( 5 ).newval == 4 );
}

TEST_CASE( "happy-path-singleton-column-infinity-bounds-equation", "[presolve]" )
{
   Message msg{};
   double time = 0.0;
   Timer t{ time };
   const Num<double> num{};
   Problem<double> problem =
       setupProblemWithSingletonColumnInEquationWithInfinityBounds();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   forceCalculationOfSingletonRows( problem, problemUpdate );
   SingletonCols<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 5 );
   REQUIRE( reductions.getReduction( 0 ).col == 0 );
   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).col == papilo::RowReduction::LOCKED );
   REQUIRE( reductions.getReduction( 1 ).row == 0 );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == 0 );
   REQUIRE( reductions.getReduction( 2 ).row == ColReduction::SUBSTITUTE_OBJ );
   REQUIRE( reductions.getReduction( 2 ).newval == 0 );

   REQUIRE( reductions.getReduction( 3 ).col == 0 );
   REQUIRE( reductions.getReduction( 3 ).row == 0 );
   REQUIRE( reductions.getReduction( 3 ).newval == 0 );

   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::LHS_INF );
   REQUIRE( reductions.getReduction( 4 ).row == 0 );
   REQUIRE( reductions.getReduction( 4 ).newval == 0 );

}



Problem<double>
setupProblemWithOnlyOneEntryIn1stRowAndColumn()
{
   const Num<double> num{};
   Vec<double> coefficients{ 1.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 10.0, 10.0, 10.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 0, 0, 0 };

   Vec<uint8_t> isLefthandsideInfinity{ 1, 1 };
   Vec<uint8_t> isRighthandsideInfinity{ 0, 0 };
   Vec<double> rhs{ 3.0, 10.0 };
   Vec<std::string> rowNames{ "A1", "A2" };
   Vec<std::string> columnNames{ "x", "y", "z" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 2.0 },
       std::tuple<int, int, double>{ 1, 0, 3.0 },
       std::tuple<int, int, double>{ 1, 1, 3.0 },
       std::tuple<int, int, double>{ 1, 2, 4.0 },
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
   pb.setRowLhsInfAll( isLefthandsideInfinity );
   pb.setRowRhsInfAll( isRighthandsideInfinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "singleton column & row matrix with equation" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 1, num, rhs[1] );
   return problem;
}

Problem<double>
setupProblemWithSingletonColumn()
{
   const Num<double> num{};
   Vec<double> coefficients{ 1.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 10.0, 10.0, 10.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 1, 1, 1 };

   Vec<uint8_t> isLefthandsideInfinity{ 0, 1, 1 };
   Vec<uint8_t> isRighthandsideInfinity{ 0, 0, 0 };
   Vec<double> rhs{ 1.0, 2.0, 3.0 };
   Vec<std::string> rowNames{ "A1", "A2", "A3" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 1, 1, 2.0 },
       std::tuple<int, int, double>{ 2, 2, 3.0 },
       std::tuple<int, int, double>{ 1, 2, 3.0 },
       std::tuple<int, int, double>{ 2, 1, 4.0 },
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
   pb.setRowLhsInfAll( isLefthandsideInfinity );
   pb.setRowRhsInfAll( isRighthandsideInfinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "singleton column" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0, num, 1 );
   return problem;
}

Problem<double>
setupProblemWithSingletonColumnInEquationWithNoImpliedBounds(
    double coefficient, double upper_bound, double lower_bound )
{
   const Num<double> num{};
   Vec<double> coefficients{ 0.0, 1.0, 1.0 };
   Vec<double> upperBounds{ upper_bound, 10.0, 10.0 };
   Vec<double> lowerBounds{ lower_bound, -10.0, -10.0 };
   Vec<uint8_t> isIntegral{ 0, 0, 0 };

   Vec<uint8_t> isLefthandsideInfinity{ 0, 1, 1 };
   Vec<uint8_t> isRighthandsideInfinity{ 0, 0, 0 };
   Vec<double> rhs{ 1.0, 2.0, 3.0 };
   Vec<std::string> rowNames{ "A1", "A2", "A3" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, coefficient },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 0, 2, 3.0 },
       std::tuple<int, int, double>{ 1, 1, 2.0 },
       std::tuple<int, int, double>{ 1, 2, 3.0 },
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
   pb.setRowLhsInfAll( isLefthandsideInfinity );
   pb.setRowRhsInfAll( isRighthandsideInfinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "singleton column" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0, num, 1 );
   return problem;
}

Problem<double>
setupProblemWithSingletonColumnInEquationWithInfinityBounds()
{
   const Num<double> num{};
   Vec<double> coefficients{ 0.0, 1.0, 1.0 };
   Vec<uint8_t> isIntegral{ 0, 0, 0 };
   Vec<uint8_t> upper_bound_infinity{ 1, 1, 1 };
   Vec<uint8_t> lower_bound_infinity{ 0, 0, 0 };

   Vec<double> rhs{ 1.0, 2.0, 3.0};
   Vec<std::string> rowNames{ "A1", "A2", "A3" };
   Vec<std::string> columnNames{ "c1", "c2", "c3" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 0, 2, 1.0 },
       std::tuple<int, int, double>{ 1, 1, 2.0 },
       std::tuple<int, int, double>{ 1, 2, 3.0 },
       std::tuple<int, int, double>{ 2, 1, -4.0 },
       std::tuple<int, int, double>{ 2, 2, -5.0 },
   };

   ProblemBuilder<double> pb;
   pb.reserve( entries.size(), rowNames.size(), columnNames.size() );
   pb.setNumRows( rowNames.size() );
   pb.setNumCols( columnNames.size() );
   pb.setColUbInfAll(upper_bound_infinity);
   pb.setColLbInfAll( lower_bound_infinity);
   pb.setObjAll( coefficients );
   pb.setObjOffset( 0.0 );
   pb.setColIntegralAll( isIntegral );
   pb.setRowRhsAll( rhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "singleton column" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0, num, rhs[0] );
   return problem;
}
