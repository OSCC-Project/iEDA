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

#include "papilo/presolvers/DominatedCols.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"

using namespace papilo;

Problem<double>
setupMatrixForDominatedCols();

Problem<double>
setupMatrixForDominatedColsParallel();

Problem<double>
setupMatrixForDominatedColsMultipleParallel();

Problem<double>
setupMatrixForMultipleDominatedCols();

TEST_CASE( "domcol-happy-path", "[presolve]" )
{
      double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupMatrixForDominatedCols();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );

   DominatedCols<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t);

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 7 );

   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).col == 0 );

   REQUIRE( reductions.getReduction( 1 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 1 ).col == 0 );

   REQUIRE( reductions.getReduction( 2 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 2 ).col == 1 );

   REQUIRE( reductions.getReduction( 3 ).row == ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 3 ).col == 1 );

   REQUIRE( reductions.getReduction( 4 ).row == 0 );
   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::LOCKED );

   REQUIRE( reductions.getReduction( 5 ).row == 0 );
   REQUIRE( reductions.getReduction( 5 ).col == RowReduction::SAVE_ROW );

   REQUIRE( reductions.getReduction( 6 ).row == ColReduction::FIXED );
   REQUIRE( reductions.getReduction( 6 ).col == 1 );
   REQUIRE( reductions.getReduction( 6 ).newval == 0 );
}

TEST_CASE( "domcol-parallel-columns", "[presolve]" )
{
      double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupMatrixForDominatedColsParallel();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );

   DominatedCols<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 7 );

   REQUIRE( reductions.getReduction( 0 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 0 ).col == 0 );

   REQUIRE( reductions.getReduction( 1 ).row ==ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 1 ).col == 0 );

   REQUIRE( reductions.getReduction( 2 ).row == ColReduction::LOCKED );
   REQUIRE( reductions.getReduction( 2 ).col == 3 );

   REQUIRE( reductions.getReduction( 3 ).row ==ColReduction::BOUNDS_LOCKED );
   REQUIRE( reductions.getReduction( 3 ).col == 3 );

   REQUIRE( reductions.getReduction( 4 ).row == 0 );
   REQUIRE( reductions.getReduction( 4 ).col == RowReduction::LOCKED );

   REQUIRE( reductions.getReduction( 5 ).row == 0 );
   REQUIRE( reductions.getReduction( 5 ).col == RowReduction::SAVE_ROW );

   REQUIRE( reductions.getReduction( 6 ).row == ColReduction::FIXED );
   REQUIRE( reductions.getReduction( 6 ).col == 3 );
   REQUIRE( reductions.getReduction( 6 ).newval == 0 );
}

TEST_CASE( "domcol-multiple-parallel-cols-generate_redundant-reductions", "[presolve]" )
{
      double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupMatrixForDominatedColsMultipleParallel();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );

   DominatedCols<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t);

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.getTransactions().size() == 3 ); // problem.getNCols()!/2
}

TEST_CASE( "domcol-multiple-columns", "[presolve]" )
{
      double time = 0.0;
   Timer t{time};
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupMatrixForMultipleDominatedCols();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );

   DominatedCols<double> presolvingMethod{};
   Reductions<double> reductions{};
   problem.recomputeAllActivities();

   PresolveStatus presolveStatus =
       presolvingMethod.execute( problem, problemUpdate, num, reductions, t );

   REQUIRE( presolveStatus == PresolveStatus::kReduced );
   REQUIRE( reductions.size() == 21 );

   Vec<int> dominating_cols = { 0, 0, 1 };
   Vec<int> dominated_cols = { 1, 2, 2 };
   for( int i = 0; i < 21; i = i + 7 )
   {
      REQUIRE( reductions.getReduction( i ).row == ColReduction::LOCKED );
      REQUIRE( reductions.getReduction( i ).col == dominating_cols[i / 5] );

      REQUIRE( reductions.getReduction( i + 1 ).row ==
               ColReduction::BOUNDS_LOCKED );
      REQUIRE( reductions.getReduction( i + 1 ).col == dominating_cols[i / 5] );

      REQUIRE( reductions.getReduction( i + 2 ).row == ColReduction::LOCKED );
      REQUIRE( reductions.getReduction( i + 2 ).col == dominated_cols[i / 5] );

      REQUIRE( reductions.getReduction( i + 3 ).row ==
               ColReduction::BOUNDS_LOCKED );
      REQUIRE( reductions.getReduction( i + 3 ).col == dominated_cols[i / 5] );

      REQUIRE( reductions.getReduction( i + 4 ).row == 0 );
      REQUIRE( reductions.getReduction( i + 4 ).col == RowReduction::LOCKED );

      REQUIRE( reductions.getReduction( i + 5 ).row == 0 );
      REQUIRE( reductions.getReduction( i + 5 ).col == RowReduction::SAVE_ROW );

      REQUIRE( reductions.getReduction( i + 6 ).row == ColReduction::FIXED );
      REQUIRE( reductions.getReduction( i + 6 ).col == dominated_cols[i / 5] );
      REQUIRE( reductions.getReduction( i + 6 ).newval == 0 );
   }
}

Problem<double>
setupMatrixForDominatedCols()
{
   // x dominates y
   // min -2x -y -2z
   // a: 2x + 3y +  z <= 6
   // b:       y + 3z <= 1
   // Optimal solution x=3, y=0, z=0

   Vec<std::string> columnNames{ "x", "y", "z" };

   Vec<double> coefficients{ -2.0, -1.0, -2.0 };
   Vec<double> upperBounds{ 4.0, 2.0, 2.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 1, 1, 1 };

   Vec<double> rhs{ 6.0, 1.0 };
   Vec<std::string> rowNames{ "a", "b" };
   Vec<uint8_t> lhsInfinity{ 1, 1 };
   Vec<uint8_t> rhsInfinity{ 0, 0 };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 2.0 },
       std::tuple<int, int, double>{ 0, 1, 3.0 },
       std::tuple<int, int, double>{ 0, 2, 1.0 },

       std::tuple<int, int, double>{ 1, 1, 1.0 },
       std::tuple<int, int, double>{ 1, 2, 3.0 },
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
   pb.setRowLhsInfAll( lhsInfinity );
   pb.setRowRhsInfAll( rhsInfinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix x dominates y" );
   Problem<double> problem = pb.build();
   return problem;
}

Problem<double>
setupMatrixForDominatedColsParallel()
{
   Vec<std::string> columnNames{ "x", "y", "z", "v", "w" };

   Vec<double> coefficients{ -1.0, 1.0, -1.0, - 1, 1 };
   Vec<double> upperBounds{ 4.0, 5.0, 2.0, 2.0, 5.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0, 0,0 };
   Vec<uint8_t> isIntegral{ 1, 0, 1, 1, 0 };

   Vec<double> rhs{ 6.0, 1.0 };
   Vec<std::string> rowNames{ "a", "b" };
   Vec<uint8_t> lhsInfinity{ 1, 1 };
   Vec<uint8_t> rhsInfinity{ 0, 0 };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 2.0 },
       std::tuple<int, int, double>{ 0, 1, 6.0 },
       std::tuple<int, int, double>{ 0, 2, 4.0 },
       std::tuple<int, int, double>{ 0, 3, 3.0 },
       std::tuple<int, int, double>{ 0, 4, 1.0 },

       std::tuple<int, int, double>{ 1, 0, 1.0 },
       std::tuple<int, int, double>{ 1, 1, -1.0 },
       std::tuple<int, int, double>{ 1, 2, -1.0 },
       std::tuple<int, int, double>{ 1, 3, 1.0 },
       std::tuple<int, int, double>{ 1, 4, 3.0 },
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
   pb.setRowLhsInfAll( lhsInfinity );
   pb.setRowRhsInfAll( rhsInfinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix parllel x and v" );
   Problem<double> problem = pb.build();
   return problem;
}

Problem<double>
setupMatrixForDominatedColsMultipleParallel()
{
   Vec<std::string> columnNames{ "x", "y", "z" };

   Vec<double> coefficients{ -1.0, -1.0, -1.0 };
   Vec<double> upperBounds{ 4.0, 2.0, 2.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 1, 1, 1 };

   Vec<double> rhs{ 6.0, 1.0 };
   Vec<std::string> rowNames{ "a", "b" };
   Vec<uint8_t> lhsInfinity{ 1, 1 };
   Vec<uint8_t> rhsInfinity{ 0, 0 };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 2.0 },
       std::tuple<int, int, double>{ 0, 1, 2.0 },
       std::tuple<int, int, double>{ 0, 2, 2.0 },

       std::tuple<int, int, double>{ 1, 0, 1.0 },
       std::tuple<int, int, double>{ 1, 1, 1.0 },
       std::tuple<int, int, double>{ 1, 2, 1.0 },
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
   pb.setRowLhsInfAll( lhsInfinity );
   pb.setRowRhsInfAll( rhsInfinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix multiple parallel cols" );
   Problem<double> problem = pb.build();
   return problem;
}


Problem<double>
setupMatrixForMultipleDominatedCols()
{
   // x dominates y
   // min -3x -2y -1z
   // a: 2x + 3y +  z <= 6
   // b:       y + 3z <= 1
   // Optimal solution x=3, y=0, z=0

   Vec<std::string> columnNames{ "x", "y", "z" };

   Vec<double> coefficients{ -3.0, -2.0, -1.0 };
   Vec<double> upperBounds{ 4.0, 2.0, 2.0 };
   Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };
   Vec<uint8_t> isIntegral{ 1, 1, 1 };

   Vec<double> rhs{ 6.0, 1.0 };
   Vec<std::string> rowNames{ "a", "b" };
   Vec<uint8_t> lhsInfinity{ 1, 1 };
   Vec<uint8_t> rhsInfinity{ 0, 0 };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 2.0 },
       std::tuple<int, int, double>{ 0, 1, 3.0 },
       std::tuple<int, int, double>{ 0, 2, 4.0 },

       std::tuple<int, int, double>{ 1, 1, 1.0 },
       std::tuple<int, int, double>{ 1, 2, 3.0 },
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
   pb.setRowLhsInfAll( lhsInfinity );
   pb.setRowRhsInfAll( rhsInfinity );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix x dominates y, z and y dominatesz" );
   Problem<double> problem = pb.build();
   return problem;
}
