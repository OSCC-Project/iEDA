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

#include "papilo/misc/VectorUtils.hpp"
#include "papilo/external/catch/catch.hpp"

using namespace papilo;

TEST_CASE( "vector-comparisons", "[misc]" )
{
   Vec<double> vec_double{1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0};

   Num<double> num;

   bool vectors_same = compareVectors( vec_double, vec_double, num );
   REQUIRE( vectors_same );

   auto vec_different = vec_double;
   vec_different[0] = 7;
   bool vectors_different = compareVectors( vec_double, vec_different, num );
   REQUIRE( !vectors_different );
}

TEST_CASE( "matrix-comparisons", "[misc]" )
{
   // build the triplets for the following matrix:
   // 1  2  0  0  0  0  0  0  0
   // 0  3  4  5  6  7  0  0  0
   // 0  8  0  0  0  0  0  0  0
   // 0  0  0  0  0  0  0  0  0
   // 9  10 11 0  0  0  0  12 13
   int nrows = 5;
   int ncols = 9;
   Vec<Triplet<double>> triplets = {
       Triplet<double>{0, 0, 1.0},  Triplet<double>{0, 1, 2.0},
       Triplet<double>{1, 1, 3.0},  Triplet<double>{1, 2, 4.0},
       Triplet<double>{1, 3, 5.0},  Triplet<double>{1, 4, 6.0},
       Triplet<double>{1, 5, 7.0},  Triplet<double>{2, 1, 8.0},
       Triplet<double>{4, 0, 9.0},  Triplet<double>{4, 1, 10.0},
       Triplet<double>{4, 2, 11.0}, Triplet<double>{4, 7, 12.0},
       Triplet<double>{4, 8, 13.0}};

   Vec<int> rowsize = {2, 5, 1, -1, 5};
   Vec<int> colsize = {2, 4, 2, 1, 1, 1, -1, 1, 1};
   SparseStorage<double> matrix{triplets, nrows, ncols, true};
   SparseStorage<double> transpose = matrix.getTranspose();

   // matrix
   Num<double> num;
   bool result = compareMatrices( matrix, matrix, num );
   REQUIRE( result );

   result = compareMatrices( matrix, transpose, num );
   REQUIRE( !result );

   result = compareMatrixToTranspose( matrix, transpose, num );
   REQUIRE( result );

   transpose.getColumns()[3] = 7;
   result = compareMatrixToTranspose( matrix, transpose, num );
   REQUIRE( !result );
}

TEST_CASE( "problem-comparisons", "[misc]" )
{
   Problem<double> problem;

   Vec<double> obj_vec{0, 1, 2, 3};
   problem.setObjective( std::move( obj_vec ) );

   Vec<double> lower_bounds{0, 0, 0, 0};
   Vec<double> upper_bounds{0, 0, 0, 0};
   ColFlags flag;
   Vec<ColFlags> col_flags( 4, flag );
   problem.setVariableDomains( lower_bounds, upper_bounds, col_flags );

   // build the triplets for the following matrix:
   // 1  2  0  0
   // 0  3  4  5
   // 0  8  0  1
   int nrows = 3;
   int ncols = 4;
   Vec<Triplet<double>> triplets = {
       Triplet<double>{0, 0, 1.0}, Triplet<double>{0, 1, 2.0},
       Triplet<double>{1, 1, 3.0}, Triplet<double>{1, 2, 4.0},
       Triplet<double>{1, 3, 5.0}, Triplet<double>{2, 1, 8.0},
       Triplet<double>{2, 3, 1.0}};

   Vec<int> rowsize = {2, 3, 2};
   Vec<int> colsize = {1, 3, 1, 2};
   SparseStorage<double> matrix{triplets, nrows, ncols, true};

   Vec<double> lhs{0, 1, 2};
   Vec<double> rhs{1, 2, 3};
   RowFlags rflag;

   Vec<RowFlags> row_flags( 3, rflag );

   problem.setConstraintMatrix( matrix, lhs, rhs, row_flags );
   //  problem.setVariableNames( std::move( parser.colnames ) );
   problem.setName( "test" );
   //  problem.setConstraintNames( std::move( parser.rownames ) );

   Num<double> num;
   bool result = compareProblems( problem, problem, num );
   REQUIRE( result );

   auto different_problem = problem;
   Vec<double> obj{0, 0, 0, 3};
   different_problem.setObjective( std::move( obj ) );
   result = compareProblems( problem, different_problem, num );
   REQUIRE( !result );
}
