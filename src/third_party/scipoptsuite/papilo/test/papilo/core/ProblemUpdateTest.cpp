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
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"
#include "papilo/core/Reductions.hpp"
#include "papilo/presolvers/ImplIntDetection.hpp"

namespace papilo
{
Problem<double>
setupProblemPresolveSingletonRow();

Problem<double>
setupProblemPresolveSingletonRowFixed();

TEST_CASE( "trivial-presolve-singleton-row", "[core]" )
{
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemPresolveSingletonRow();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.trivialPresolve();
   REQUIRE( problem.getUpperBounds()[2] == 1 );
   REQUIRE( problem.getRowFlags()[1].test( RowFlag::kRedundant ) );
}

TEST_CASE( "trivial-presolve-singleton-row-pt-2", "[core]" )
{
   Num<double> num{};
   Message msg{};
   Problem<double> problem = setupProblemPresolveSingletonRowFixed();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions );
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                        presolveOptions, num, msg );
   problemUpdate.trivialPresolve();

   REQUIRE( problem.getUpperBounds()[2] == 1 );
   REQUIRE( problem.getLowerBounds()[2] == 1 );
   REQUIRE( problem.getRowFlags()[1].test( RowFlag::kRedundant ) );
   REQUIRE( problemUpdate.getSingletonCols().size() == 2 );
}

Problem<double>
setupProblemPresolveSingletonRow()
{

   const Vec<double> coefficients{ 3.0, 1.0, 1.0 };
   Vec<std::string> rowNames{ "A1", "A2" };
   Vec<std::string> columnNames{ "x", "y", "z" };
   const Vec<double> rhs{ 3.0, 1.0 };
   const Vec<double> upperBounds{ 3.0, 7.0, 7.0 };
   const Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };
   Vec<uint8_t> integral = Vec<uint8_t>{ 1, 1, 1 };

   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 2.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 0, 2, 1.0 },
       std::tuple<int, int, double>{ 1, 2, 1.0 } };

   ProblemBuilder<double> pb;
   pb.reserve( (int) entries.size(), (int)  rowNames.size(), (int) columnNames.size() );
   pb.setNumRows( (int) rowNames.size() );
   pb.setNumCols( (int) columnNames.size() );
   pb.setColUbAll( upperBounds );
   pb.setColLbAll( lowerBounds );
   pb.setObjAll( coefficients );
   pb.setObjOffset( 0.0 );
   pb.setColIntegralAll( integral );
   pb.setRowRhsAll( rhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix for singleton row" );
   Problem<double> problem = pb.build();
   return problem;
}

Problem<double>
setupProblemPresolveSingletonRowFixed()
{
   Num<double> num{};
   const Vec<double> coefficients{ 3.0, 1.0, 1.0 };
   Vec<std::string> rowNames{ "A1", "A2" };
   Vec<std::string> columnNames{ "x", "y", "z" };
   const Vec<double> rhs{ 3.0, 1.0 };
   const Vec<double> upperBounds{ 3.0, 7.0, 7.0 };
   const Vec<double> lowerBounds{ 0.0, 0.0, 0.0 };
   Vec<uint8_t> integral = Vec<uint8_t>{ 1, 1, 1 };

   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 2.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 0, 2, 1.0 },
       std::tuple<int, int, double>{ 1, 2, 1.0 } };

   ProblemBuilder<double> pb;
   pb.reserve( (int) entries.size(), (int) rowNames.size(), (int) columnNames.size() );
   pb.setNumRows( (int) rowNames.size() );
   pb.setNumCols( (int) columnNames.size() );
   pb.setColUbAll( upperBounds );
   pb.setColLbAll( lowerBounds );
   pb.setObjAll( coefficients );
   pb.setObjOffset( 0.0 );
   pb.setColIntegralAll( integral );
   pb.setRowRhsAll( rhs );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix for singleton row fixed" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 1,num, rhs[1] );
   return problem;
}
} // namespace papilo
