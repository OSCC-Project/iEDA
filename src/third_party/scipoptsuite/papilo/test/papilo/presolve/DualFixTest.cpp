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

#include "papilo/presolvers/DualFix.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"

using namespace papilo;

Problem<double>
setupMatrixForDualFixFirstColumnOnlyPositiveValues();

Problem<double>
setupMatrixForDualSubstitution( bool integer_variables );

Problem<double>
setupMatrixForDualSubstitutionEquation();

Problem<double>
setupMatrixForDualSubstitutionWithUnboundedVar();

Problem<double>
setupMatrixForDualSubstitutionIntegerRounding();

Problem<double>
setupMatrixForDualFixInfinity();

TEST_CASE( "dual-fix-trivial-column-presolve-finds-reduction", "[presolve]" )
{
   Num<double> num{};
   Message msg{};
   Problem<double> problem =
       setupMatrixForDualFixFirstColumnOnlyPositiveValues();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );

   problemUpdate.trivialPresolve();

   REQUIRE( problem.getColFlags()[0].test( ColFlag::kFixed ) );
   REQUIRE( problem.getLowerBounds()[0] == 1 );
   REQUIRE( problem.getUpperBounds()[0] == 1 );
}

TEST_CASE( "dual-fix-happy-path", "[presolve]" )
{
   double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem =
       setupMatrixForDualFixFirstColumnOnlyPositiveValues();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );

   problem.recomputeAllActivities();
   papilo::DualFix<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t);

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.getReduction( 0 ).col == 0 );
   REQUIRE( reductions.getReduction( 0 ).row == papilo::ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );
   REQUIRE( reductions.getReduction( 1 ).col == 0 );
   REQUIRE( reductions.getReduction( 1 ).row == papilo::ColReduction::FIXED );
   REQUIRE( reductions.getReduction( 1 ).newval == 1 );
}

TEST_CASE( "dual-fix-no-dual-substitution-for-lp", "[presolve]" )
{
      double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupMatrixForDualSubstitution( false );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );

   problemUpdate.trivialPresolve();
   papilo::DualFix<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t);

   REQUIRE( presolveStatus == PresolveStatus::kUnchanged );
}

TEST_CASE( "dual-fix-dual-substitution", "[presolve]" )
{
      double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupMatrixForDualSubstitution( true );
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );

   problemUpdate.trivialPresolve();
   papilo::DualFix<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t);

   REQUIRE( presolveStatus == PresolveStatus::kReduced );

   REQUIRE( reductions.getReduction( 0 ).col == 0 );
   REQUIRE( reductions.getReduction( 0 ).row == papilo::ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).col == 0 );
   REQUIRE( reductions.getReduction( 1 ).row ==
            papilo::ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == RowReduction::SAVE_ROW );
   REQUIRE( reductions.getReduction( 2 ).row == 2 );

   REQUIRE( reductions.getReduction( 3 ).col == 0 );
   REQUIRE( reductions.getReduction( 3 ).row ==
            papilo::ColReduction::UPPER_BOUND );
   REQUIRE( reductions.getReduction( 3 ).newval == 6 );
}

TEST_CASE( "dual-fix-dual-substitution-rounding", "[presolve]" )
{
      double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupMatrixForDualSubstitutionIntegerRounding();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );

   problem.recomputeAllActivities();
   papilo::DualFix<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 8 );
   REQUIRE( reductions.getReduction( 0 ).col == 0 );
   REQUIRE( reductions.getReduction( 0 ).row == papilo::ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).col == 0 );
   REQUIRE( reductions.getReduction( 1 ).row ==
            papilo::ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 6 ).col == RowReduction::SAVE_ROW );
   REQUIRE( reductions.getReduction( 6 ).row == 1 );

   REQUIRE( reductions.getReduction( 3 ).col == 0 );
   REQUIRE( reductions.getReduction( 3 ).row ==
            papilo::ColReduction::UPPER_BOUND );
   REQUIRE( reductions.getReduction( 3 ).newval == 6 );

   REQUIRE( reductions.getReduction( 4 ).col == 1 );
   REQUIRE( reductions.getReduction( 4 ).row == papilo::ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 4 ).newval == 0 );

   REQUIRE( reductions.getReduction( 5 ).col == 1 );
   REQUIRE( reductions.getReduction( 5 ).row ==
            papilo::ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 5 ).newval == 0 );

   REQUIRE( reductions.getReduction( 6 ).col == RowReduction::SAVE_ROW );
   REQUIRE( reductions.getReduction( 6 ).row == 1 );

   REQUIRE( reductions.getReduction( 7 ).col == 1 );
   REQUIRE( reductions.getReduction( 7 ).row ==
            papilo::ColReduction::LOWER_BOUND );
   REQUIRE( reductions.getReduction( 7 ).newval == 4 );
}

TEST_CASE( "dual-fix-dual-substitution-unbounded-variables", "[presolve]" )
{
      double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupMatrixForDualSubstitutionWithUnboundedVar();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );

   problem.recomputeAllActivities();
   papilo::DualFix<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kUnchanged );
}

TEST_CASE( "dual-fix-dual-substitution-equation", "[presolve]" )
{
      double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupMatrixForDualSubstitutionEquation();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );

   problem.recomputeAllActivities();
   papilo::DualFix<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 4 );

   REQUIRE( reductions.getReduction( 0 ).col == 1 );
   REQUIRE( reductions.getReduction( 0 ).row == papilo::ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   REQUIRE( reductions.getReduction( 1 ).col == 1 );
   REQUIRE( reductions.getReduction( 1 ).row ==
            papilo::ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 1 ).newval == 0 );

   REQUIRE( reductions.getReduction( 2 ).col == RowReduction::SAVE_ROW );
   REQUIRE( reductions.getReduction( 2 ).row == 0 );

   REQUIRE( reductions.getReduction( 3 ).col == 1 );
   REQUIRE( reductions.getReduction( 3 ).row ==
            papilo::ColReduction::LOWER_BOUND );
   REQUIRE( reductions.getReduction( 3 ).newval == 2 );
}

TEST_CASE( "dual-fix-infinity", "[presolve]" )
{
      double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupMatrixForDualFixInfinity();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );

   problem.recomputeAllActivities();
   papilo::DualFix<double> presolvingMethod{};
   Reductions<double> reductions{};

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() >= 5 );

   REQUIRE( reductions.getReduction( 0 ).row == papilo::ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).col == 0 );
   REQUIRE( reductions.getReduction( 0 ).newval == 0 );

   for( int i = 0; i < 3; i++ )
   {
      REQUIRE( reductions.getReduction( 1 + i ).col ==
               papilo::RowReduction::REDUNDANT );
      REQUIRE( reductions.getReduction( 1 + i ).row == i );
      REQUIRE( reductions.getReduction( 1 + i ).newval == 0 );
   }

   REQUIRE( reductions.getReduction( 4 ).col == 0 );
   REQUIRE( reductions.getReduction( 4 ).row ==
            papilo::ColReduction::FIXED_INFINITY );
   REQUIRE( reductions.getReduction( 4 ).newval == -1 );

   REQUIRE( reductions.getTransactions()[0].start == 0 );
   REQUIRE( reductions.getTransactions()[0].end == 5 );
}

Problem<double>
setupMatrixForDualFixInfinity()
{
   // min y + z
   // x +y <= 8
   // x + z <= -4
   // x - 2y - z <= 6
   // -5 <= y,z <= 5
   Vec<double> coefficients{ 0, 1.0, 1.0 };
   Vec<double> upperBounds{ 5.0, 5.0, 5.0 };
   Vec<double> lowerBounds{ 0.0, -5.0, -5.0 };
   Vec<uint8_t> isIntegral{ 0, 0, 0 };
   Vec<uint8_t> lhsInfinity{ 1, 1, 1 };
   Vec<uint8_t> rhsInfinity{ 0, 0, 0 };
   Vec<uint8_t> upperBoundsInfinity{ 0, 0, 0 };
   Vec<uint8_t> lowerBoundsInfinity{ 1, 0, 0 };

   Vec<double> rhs{ 8.0, -4.0, 6.0 };
   Vec<std::string> rowNames{ "A1", "A2", "A3" };
   Vec<std::string> columnNames{ "x", "y", "z" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 1, 0, 1.0 },
       std::tuple<int, int, double>{ 1, 2, 1.0 },
       std::tuple<int, int, double>{ 2, 0, 1.0 },
       std::tuple<int, int, double>{ 2, 1, -2.0 },
       std::tuple<int, int, double>{ 2, 2, -1.0 } };

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
   pb.setRowLhsInfAll( lhsInfinity );
   pb.setRowRhsInfAll( rhsInfinity );
   pb.setColUbInfAll( upperBoundsInfinity );
   pb.setColLbInfAll( lowerBoundsInfinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "variable x can be set to neg infinity" );
   Problem<double> problem = pb.build();
   return problem;
}

Problem<double>
setupMatrixForDualFixFirstColumnOnlyPositiveValues()
{
   // min x + y + z
   // 2x + 4 y -3 z <= 8
   // - y - z <= -4
   // 2x - 2y + z <= 6
   // 0<= x,y,z <= 4
   Vec<double> coefficients{ 1.0, 1.0, 1.0 };
   Vec<double> upperBounds{ 10.0, 10.0, 10.0 };
   Vec<double> lowerBounds{ 1.0, 1.0, 1.0 };
   Vec<uint8_t> isIntegral{ 0, 0, 0 };
   Vec<uint8_t> lhsInfinity{ 1, 1, 1 };
   Vec<uint8_t> rhsInfinity{ 0, 0, 0 };

   Vec<double> rhs{ 8.0, -4.0, 6.0 };
   Vec<std::string> rowNames{ "A1", "A2", "A3" };
   Vec<std::string> columnNames{ "x", "y", "z" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 2.0 },
       std::tuple<int, int, double>{ 0, 1, 4.0 },
       std::tuple<int, int, double>{ 0, 2, -3.0 },
       std::tuple<int, int, double>{ 1, 1, -1.0 },
       std::tuple<int, int, double>{ 1, 2, -1.0 },
       std::tuple<int, int, double>{ 2, 0, 2.0 },
       std::tuple<int, int, double>{ 2, 1, -2.0 },
       std::tuple<int, int, double>{ 2, 2, 1.0 } };

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
   pb.setRowLhsInfAll( lhsInfinity );
   pb.setRowRhsInfAll( rhsInfinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix for dual fix 1st row positive" );
   Problem<double> problem = pb.build();
   return problem;
}

Problem<double>
setupMatrixForDualSubstitution( bool integer_variables )
{
   // min y - z
   // 4 y -3 z <= 6
   // - y - z <= -3
   //- y + z <= 4
   // 0<= x,y,z <= 10
   Vec<double> coefficients{ 1.0, -1.0 };
   Vec<double> upperBounds{ 10.0, 10.0 };
   Vec<double> lowerBounds{ 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ integer_variables, integer_variables };
   Vec<uint8_t> lhsInfinity{ 1, 1, 1 };
   Vec<uint8_t> rhsInfinity{ 0, 0, 0 };

   Vec<double> rhs{ 6.0, -3.0, 4.0 };
   Vec<std::string> rowNames{ "A1", "A2", "A3" };
   Vec<std::string> columnNames{ "y", "z" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 4.0 },
       std::tuple<int, int, double>{ 0, 1, -3.0 },
       std::tuple<int, int, double>{ 1, 0, -1.0 },
       std::tuple<int, int, double>{ 1, 1, -1.0 },
       std::tuple<int, int, double>{ 2, 0, -1.0 },
       std::tuple<int, int, double>{ 2, 1, 1.0 } };

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
   pb.setRowLhsInfAll( lhsInfinity );
   pb.setRowRhsInfAll( rhsInfinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix for dual substitution" );
   Problem<double> problem = pb.build();
   return problem;
}

Problem<double>
setupMatrixForDualSubstitutionIntegerRounding()
{
   // min y - z
   // 4 y -3 z <= 6
   //- y + z <= 4.2
   // 0<= x,y,z <= 10
   Vec<double> coefficients{ 1.0, -1.0 };
   Vec<double> upperBounds{ 10.0, 10.0 };
   Vec<double> lowerBounds{ 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 1, 1 };
   Vec<uint8_t> lhsInfinity{ 1, 1 };
   Vec<uint8_t> rhsInfinity{ 0, 0 };

   Vec<double> rhs{ 6.0, 4.2 };
   Vec<std::string> rowNames{ "A1", "A2" };
   Vec<std::string> columnNames{ "y", "z" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 4.0 },
       std::tuple<int, int, double>{ 0, 1, -3.0 },
       std::tuple<int, int, double>{ 1, 0, -1.0 },
       std::tuple<int, int, double>{ 1, 1, 1.0 } };

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
   pb.setRowLhsInfAll( lhsInfinity );
   pb.setRowRhsInfAll( rhsInfinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName(
       "matrix for dual substitution (new non integer bound on integer)" );
   Problem<double> problem = pb.build();
   return problem;
}

Problem<double>
setupMatrixForDualSubstitutionWithUnboundedVar()
{
   // min y - z
   // 4 y -3 z <= 6
   // - y - z <= -3
   //- y + z <= 4
   Vec<double> coefficients{ 1.0, -1.0 };
   Vec<uint8_t> isIntegral{ 0, 0 };
   Vec<uint8_t> lhsInfinity{ 1, 1, 1 };
   Vec<uint8_t> infinity{ 1, 1 };
   Vec<uint8_t> rhsInfinity{ 0, 0, 0 };

   Vec<double> rhs{ 6.0, -3.0, 4.0 };
   Vec<std::string> rowNames{ "A1", "A2", "A3" };
   Vec<std::string> columnNames{ "y", "z" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 4.0 },
       std::tuple<int, int, double>{ 0, 1, -3.0 },
       std::tuple<int, int, double>{ 1, 0, -1.0 },
       std::tuple<int, int, double>{ 1, 1, -1.0 },
       std::tuple<int, int, double>{ 2, 0, -1.0 },
       std::tuple<int, int, double>{ 2, 1, 1.0 } };

   ProblemBuilder<double> pb;
   pb.reserve( (int)entries.size(), (int)rowNames.size(),
               (int)columnNames.size() );
   pb.setNumRows( (int)rowNames.size() );
   pb.setNumCols( (int)columnNames.size() );
   pb.setColLbInfAll( infinity );
   pb.setColUbInfAll( infinity );
   pb.setObjAll( coefficients );
   pb.setObjOffset( 0.0 );
   pb.setColIntegralAll( isIntegral );
   pb.setRowRhsAll( rhs );
   pb.setRowLhsInfAll( lhsInfinity );
   pb.setRowRhsInfAll( rhsInfinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix for dual substitution" );
   Problem<double> problem = pb.build();
   return problem;
}

Problem<double>
setupMatrixForDualSubstitutionEquation()
{
   Num<double> num{};
   Vec<double> coefficients{ 0.0, 0.0 };
   Vec<double> upperBounds{ 1.0, 3.0 };
   Vec<double> lowerBounds{ 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 1, 1 };
   Vec<uint8_t> lhsInfinity{ 0, 0 };
   Vec<uint8_t> rhsInfinity{ 0, 0 };

   Vec<double> rhs{ 3.0, 4.0 };
   Vec<std::string> rowNames{ "A1", "a" };
   Vec<std::string> columnNames{ "y", "z" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 1.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 1, 0, 2.0 },
       std::tuple<int, int, double>{ 1, 1, 1.0 },
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
   pb.setRowLhsInfAll( lhsInfinity );
   pb.setRowRhsInfAll( rhsInfinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix for dual substitution only equations" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0, num, rhs[0] );
   problem.getConstraintMatrix().modifyLeftHandSide( 1, num, rhs[1] );
   return problem;
}
