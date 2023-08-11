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

#include "papilo/core/Presolve.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemBuilder.hpp"
#include "papilo/core/Reductions.hpp"
#include "papilo/presolvers/ImplIntDetection.hpp"

using namespace papilo;

static const int ELIMINATED = -1;

papilo::Vec<double>
coefficients()
{
   return papilo::Vec<double>{ 3.0, 1.0, 1.0, 1.0 };
}

papilo::Vec<double>
upperBounds()
{
   return papilo::Vec<double>{ 1.0, 1.0, 1.0, 1.0 };
}

papilo::Vec<double>
lowerBounds()
{
   return papilo::Vec<double>{ 0.0, 0.0, 0.0, 0.0 };
}

papilo::Vec<uint8_t>
isIntegral()
{
   return papilo::Vec<uint8_t>{ 1, 1, 1, 1 };
}

papilo::Vec<double>
rhs()
{
   return papilo::Vec<double>{ 2.0, 1.0 };
}

papilo::Vec<double>
lhs()
{
   return papilo::Vec<double>{ rhs()[0], rhs()[1] };
}

std::vector<int>
row_sizes()
{
   return papilo::Vec<int>{ 3, 2 };
}

papilo::Problem<double>
setupProblemWithMultiplePresolvingOptions();

std::pair<std::pair<papilo::Problem<double>, papilo::PostsolveStorage<double>>,
          std::pair<int, int>>
applyReductions( const papilo::Reductions<double>& reductions,
                 bool substitutions );

double
getEntry( papilo::Problem<double> problem, int row, int column );

int
getRowIndex( papilo::Problem<double> problem, int row, int column );

bool
isRow( papilo::Problem<double>& problem, papilo::RowFlag rowflag, int row );

TEST_CASE( "replacing-variables-is-postponed-by-flag", "[core]" )
{
   papilo::Reductions<double> reductions{};
   reductions.replaceCol( 0, 1, -1, 0 );
   reductions.replaceCol( 0, 2, -1, 0 );

   std::pair<std::pair<papilo::Problem<double>, papilo::PostsolveStorage<double>>,
             std::pair<int, int>>
       pair = applyReductions( reductions, true );
   std::pair<int, int>& result = pair.second;

   REQUIRE( result.first == 2 );
   REQUIRE( result.second == 0 );
}

//todo #fails
TEST_CASE( "happy-path-replace-variable", "[core]" )
{
   papilo::Reductions<double> reductions{};
   // substitute x = 1 - y (result of simple probing)
   reductions.replaceCol( 0, 1, -1, 0 );
   reductions.replaceCol( 0, 2, -1, 0 );

   std::pair<std::pair<papilo::Problem<double>, papilo::PostsolveStorage<double>>,
             std::pair<int, int>>
       pair = applyReductions( reductions, false );
   std::pair<int, int>& result = pair.second;

   REQUIRE( result.first == 2 );
   REQUIRE( result.second == 2 );

   papilo::Problem<double> problem = pair.first.first;

   std::vector<double> expected_objective{ 0.0, -2.0, 1.0, 1.0 };
   std::vector<int> expected_colsizes{ ELIMINATED, 1, 2, 1 };
   std::vector<int> expected_rowsizes{ 2, 2 };

   REQUIRE( problem.getObjective().coefficients == expected_objective );
   REQUIRE( problem.getConstraintMatrix().getColSizes() == expected_colsizes );
   REQUIRE( problem.getConstraintMatrix().getRightHandSides() == rhs() );
   REQUIRE( problem.getConstraintMatrix().getLeftHandSides() == lhs() );

   REQUIRE( problem.getNRows() == 2 );
   REQUIRE( !isRow( problem, papilo::RowFlag::kRedundant, 0 ) );
   REQUIRE( problem.getColFlags()[0].test( papilo::ColFlag::kSubstituted ) );

   REQUIRE( problem.getConstraintMatrix().getRowSizes() == expected_rowsizes );

   REQUIRE( getRowIndex( problem, 0, 0 ) == 1 );
   REQUIRE( getRowIndex( problem, 0, 1 ) == 2 );
   REQUIRE( getEntry( problem, 0, 0 ) == -1 );
   REQUIRE( getEntry( problem, 0, 1 ) == 1.0 );
   REQUIRE( getEntry( problem, 0, 2 ) == 1.0 );

   REQUIRE( getRowIndex( problem, 1, 0 ) == 2 );
   REQUIRE( getRowIndex( problem, 1, 1 ) == 3 );
   REQUIRE( getEntry( problem, 1, 0 ) == 1.0 );
   REQUIRE( getEntry( problem, 1, 1 ) == 1.0 );
}

TEST_CASE( "happy-path-substitute-matrix-coefficient-into-objective", "[core]" )
{
   papilo::Reductions<double> reductions{};
   {
      TransactionGuard<double> tg{ reductions };
      reductions.lockColBounds( 3 );
      reductions.lockRow( 1 );
      reductions.substituteColInObjective( 3, 1 );
      reductions.markRowRedundant( 1 );
   }

   std::pair<std::pair<papilo::Problem<double>, papilo::PostsolveStorage<double>>,
             std::pair<int, int>>
       pair = applyReductions( reductions, false );
   Problem<double> problem = pair.first.first;

   std::vector<double> expected_objective{ 3.0, 1.0, 0.0, 0.0 };
   std::vector<double> expected_upper_bounds{ 1.0, 1.0, 1.0, 0.0 };

   REQUIRE( problem.getObjective().coefficients == expected_objective );
   REQUIRE( problem.getNRows() == 2 );
   REQUIRE( problem.getUpperBounds() == expected_upper_bounds );

   REQUIRE( problem.getColFlags()[3].test( ColFlag::kSubstituted ) );

   REQUIRE( isRow( problem, RowFlag::kRedundant, 1 ) );
   REQUIRE( problem.getConstraintMatrix().getRowSizes() == row_sizes() );
   REQUIRE( getRowIndex( problem, 1, 0 ) == 2 );
   REQUIRE( getRowIndex( problem, 1, 1 ) == 3 );
   REQUIRE( getEntry( problem, 1, 0 ) == 1.0 );
   REQUIRE( getEntry( problem, 1, 1 ) == 1.0 );
}

TEST_CASE( "happy-path-aggregate-free-column", "[core]" )
{
   Reductions<double> reductions{};
   {
      // replace last column with z = 1 - w
      TransactionGuard<double> tg{ reductions };
      reductions.lockColBounds( 3 );
      reductions.lockRow( 1 );
      reductions.aggregateFreeCol( 3, 1 );
   }

   std::pair<std::pair<Problem<double>, PostsolveStorage<double>>,
             std::pair<int, int>>
       pair = applyReductions( reductions, false );
   std::pair<int, int>& result = pair.second;

   REQUIRE( result.first == 1 );
   REQUIRE( result.second == 1 );
   Problem<double> problem = pair.first.first;

   std::vector<double> expected_objective{ 3.0, 1.0, 0.0, 0.0 };
   std::vector<int> expected_colsizes{ 1, 1, 1, ELIMINATED };

   REQUIRE( problem.getObjective().coefficients == expected_objective );
   REQUIRE( problem.getUpperBounds() == upperBounds() );
   REQUIRE( problem.getLowerBounds() == lowerBounds() );
   REQUIRE( problem.getConstraintMatrix().getColSizes() == expected_colsizes );
   REQUIRE( problem.getNRows() == 2 );
   REQUIRE( isRow( problem, RowFlag::kRedundant, 1 ) );
}

TEST_CASE( "presolve-activity-is-updated-correctly-huge-values", "[core]" )
{
   double lb = 0;
   double ub = 1;
   ColFlags cflags;
   double oldcolcoef = -1e17;
   double newcolcoef = -2265603.6036036061;
   RowActivity<double> activity{};
   int rowLength = 1;
   int colindices [1] = {0};
   double rowvals [1] = {newcolcoef};
   Vec<double> lower_bounds{};
   Vec<double> upper_bounds{};
   Vec<ColFlags> flags{};
   lower_bounds.push_back(lb);
   upper_bounds.push_back(ub);
   flags.push_back(cflags);
   VariableDomains<double> domains{lower_bounds, upper_bounds, flags};
   const Num<double> num{};
   auto activityChange = []( ActivityChange actChange,
                             RowActivity<double>& activity ) {};

   update_activity_after_coeffchange( lb, ub, cflags, oldcolcoef, newcolcoef,
                                      activity, rowLength, colindices, rowvals,
                                      domains, num, activityChange );

   REQUIRE( activity.min == -2265603.6036036061);

}

Problem<double>
setupProblemWithMultiplePresolvingOptions()
{
   // 2x + y + z = 2           for simple probing
   //      z + w = 1           w = singleton column can be replaced & simple
   //      substituion
   Num<double> num{};
   Vec<std::string> rowNames{ "A1", "A2" };
   Vec<std::string> columnNames{ "c1", "c2", "c3", "c4" };
   Vec<std::tuple<int, int, double>> entries{
       std::tuple<int, int, double>{ 0, 0, 2.0 },
       std::tuple<int, int, double>{ 0, 1, 1.0 },
       std::tuple<int, int, double>{ 0, 2, 1.0 },
       std::tuple<int, int, double>{ 1, 2, 1.0 },
       std::tuple<int, int, double>{ 1, 3, 1.0 } };

   ProblemBuilder<double> pb;
   pb.reserve( entries.size(), rowNames.size(), columnNames.size() );
   pb.setNumRows( rowNames.size() );
   pb.setNumCols( columnNames.size() );
   pb.setColUbAll( upperBounds() );
   pb.setColLbAll( lowerBounds() );
   pb.setObjAll( coefficients() );
   pb.setObjOffset( 0.0 );
   pb.setColIntegralAll( isIntegral() );
   pb.setRowRhsAll( rhs() );
   pb.setRowLhsAll( lhs() );
   pb.addEntryAll( entries );
   pb.setColNameAll( columnNames );
   pb.setProblemName( "matrix for testing with multiple options" );
   Problem<double> problem = pb.build();
   problem.getConstraintMatrix().modifyLeftHandSide( 0, num, lhs()[0] );
   problem.getConstraintMatrix().modifyLeftHandSide( 1, num, lhs()[1] );
   return problem;
}

std::pair<std::pair<Problem<double>, PostsolveStorage<double>>,
          std::pair<int, int>>
applyReductions( const Reductions<double>& reductions,
                 bool substitutions )
{
   Num<double> num{};
   Message msg{};
   Problem<double> problem =
       setupProblemWithMultiplePresolvingOptions();
   Statistics statistics{};
   PresolveOptions presolveOptions{};
   PostsolveStorage<double> postsolve =
       PostsolveStorage<double>( problem, num, presolveOptions);
   ProblemUpdate<double> problemUpdate( problem, postsolve, statistics,
                                                presolveOptions, num, msg );
   problem.recomputeLocks();
   problemUpdate.trivialColumnPresolve();
   problem.recomputeAllActivities();
   problemUpdate.setPostponeSubstitutions( substitutions );
   Presolve<double> presolve{};
   presolve.addDefaultPresolvers();
   const std::pair<int, int>& result =
       presolve.applyReductions( 0, reductions, problemUpdate );
   return { { problem, postsolve }, result };
}

double
getEntry( Problem<double> problem, int row, int column )
{
   return problem.getConstraintMatrix()
       .getRowCoefficients( row )
       .getValues()[column];
}

int
getRowIndex( Problem<double> problem, int row, int column )
{
   return problem.getConstraintMatrix()
       .getRowCoefficients( row )
       .getIndices()[column];
}

bool
isRow( Problem<double>& problem, RowFlag rowflag, int row )
{
   return problem.getRowFlags()[row].test( rowflag );
}
