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

#include "papilo/core/SparseStorage.hpp"
#include "papilo/external/catch/catch.hpp"
#include "papilo/misc/compress_vector.hpp"

papilo::SparseStorage<double>
setupSparseMatrix();

TEST_CASE( "sparse storage can be created from triplets", "[core]" )
{
   papilo::SparseStorage<double> matrix = setupSparseMatrix();

   REQUIRE( matrix.getNnz() == 13 );
   REQUIRE( matrix.getNRows() == 5 );
   REQUIRE( matrix.getNCols() == 9 );

   int expectedRowRanges[5] = { 2, 5, 1, 0, 5 };

   auto rowRanges = matrix.getRowRanges();
   auto rowValues = matrix.getValues();
   auto columns = matrix.getColumns();

   for( int i = 0; i < 5; ++i )
   {
      REQUIRE( rowRanges[i].end - rowRanges[i].start == expectedRowRanges[i] );
      double* row = rowValues + rowRanges[i].start;
      int* rowColumns = columns + rowRanges[i].start;
      switch( i )
      {
      case 0:
         REQUIRE( row[0] == 1.0 );
         REQUIRE( row[1] == 2.0 );
         REQUIRE( rowColumns[0] == 0 );
         REQUIRE( rowColumns[1] == 1 );
         break;
      case 1:
         REQUIRE( row[0] == 3.0 );
         REQUIRE( row[1] == 4.0 );
         REQUIRE( row[2] == 5.0 );
         REQUIRE( row[3] == 6.0 );
         REQUIRE( row[4] == 7.0 );
         REQUIRE( rowColumns[0] == 1 );
         REQUIRE( rowColumns[1] == 2 );
         REQUIRE( rowColumns[2] == 3 );
         REQUIRE( rowColumns[3] == 4 );
         REQUIRE( rowColumns[4] == 5 );
         break;
      case 2:
         REQUIRE( row[0] == 8.0 );
         REQUIRE( rowColumns[0] == 1 );
         break;
      case 3:
         continue;
      case 4:
         REQUIRE( row[0] == 9.0 );
         REQUIRE( row[1] == 10.0 );
         REQUIRE( row[2] == 11.0 );
         REQUIRE( row[3] == 12.0 );
         REQUIRE( row[4] == 13.0 );
         REQUIRE( rowColumns[0] == 0 );
         REQUIRE( rowColumns[1] == 1 );
         REQUIRE( rowColumns[2] == 2 );
         REQUIRE( rowColumns[3] == 7 );
         REQUIRE( rowColumns[4] == 8 );
         break;
      default:
         REQUIRE_FALSE( "case default should not be reached" );
      }
   }
}

TEST_CASE( "sparse matrix can be compressed", "[core]" )
{
   papilo::SparseStorage<double> matrix = setupSparseMatrix();

   papilo::Vec<int> rowSizes = { 2, 5, 1, -1, 5 };
   papilo::Vec<int> columnSizes = { 2, 4, 2, 1, 1, 1, -1, 1, 1 };
   papilo::Vec<int> col_mapping =
       matrix.compress( rowSizes, columnSizes, false );

   papilo::Vec<int> expectedValues = { 0, 1, 2, 3, 4, 5, -1, 6, 7 };
   REQUIRE( col_mapping == expectedValues );

   papilo::compress_vector( col_mapping, columnSizes );

   auto rowRanges = matrix.getRowRanges();
   auto rowValues = matrix.getValues();
   auto columns = matrix.getColumns();

   int expectedRowRanges[4] = { 2, 5, 1, 5 };

   for( int i = 0; i < 4; ++i )
   {
      REQUIRE( rowRanges[i].end - rowRanges[i].start == expectedRowRanges[i] );
      double* row = rowValues + rowRanges[i].start;
      int* rowColumns = columns + rowRanges[i].start;
      switch( i )
      {
      case 0:
         REQUIRE( row[0] == 1.0 );
         REQUIRE( row[1] == 2.0 );
         REQUIRE( rowColumns[0] == 0 );
         REQUIRE( rowColumns[1] == 1 );
         break;
      case 1:
         REQUIRE( row[0] == 3.0 );
         REQUIRE( row[1] == 4.0 );
         REQUIRE( row[2] == 5.0 );
         REQUIRE( row[3] == 6.0 );
         REQUIRE( row[4] == 7.0 );
         REQUIRE( rowColumns[0] == 1 );
         REQUIRE( rowColumns[1] == 2 );
         REQUIRE( rowColumns[2] == 3 );
         REQUIRE( rowColumns[3] == 4 );
         REQUIRE( rowColumns[4] == 5 );
         break;
      case 2:
         REQUIRE( row[0] == 8.0 );
         REQUIRE( rowColumns[0] == 1 );
         break;
      case 3:
         REQUIRE( row[0] == 9.0 );
         REQUIRE( row[1] == 10.0 );
         REQUIRE( row[2] == 11.0 );
         REQUIRE( row[3] == 12.0 );
         REQUIRE( row[4] == 13.0 );
         REQUIRE( rowColumns[0] == 0 );
         REQUIRE( rowColumns[1] == 1 );
         REQUIRE( rowColumns[2] == 2 );
         REQUIRE( rowColumns[3] == 6 );
         REQUIRE( rowColumns[4] == 7 );
         break;
      default:
         REQUIRE_FALSE( "case default should not be reached" );
      }
   }
}

TEST_CASE(
    "sparse storage can be created from triplets and compressed on transpose",
    "[core]" )
{
   papilo::SparseStorage<double> transpose = setupSparseMatrix().getTranspose();

   REQUIRE( transpose.getNnz() == 13 );
   REQUIRE( transpose.getNRows() == 9 );
   REQUIRE( transpose.getNCols() == 5 );

   auto columnRanges = transpose.getRowRanges();
   auto columnValues = transpose.getValues();
   auto rows = transpose.getColumns();

   int expectedRowRanges[9] = { 2, 4, 2, 1, 1, 1, 0, 1, 1 };

   for( int i = 0; i < 9; ++i )
   {
      REQUIRE( columnRanges[i].end - columnRanges[i].start ==
               expectedRowRanges[i] );
      double* col = columnValues + columnRanges[i].start;
      int* columnRows = rows + columnRanges[i].start;
      switch( i )
      {
      case 0:
         REQUIRE( col[0] == 1.0 );
         REQUIRE( col[1] == 9.0 );
         REQUIRE( columnRows[0] == 0 );
         REQUIRE( columnRows[1] == 4 );
         break;
      case 1:
         REQUIRE( col[0] == 2.0 );
         REQUIRE( col[1] == 3.0 );
         REQUIRE( col[2] == 8.0 );
         REQUIRE( col[3] == 10.0 );
         REQUIRE( columnRows[0] == 0 );
         REQUIRE( columnRows[1] == 1 );
         REQUIRE( columnRows[2] == 2 );
         REQUIRE( columnRows[3] == 4 );
         break;
      case 2:
         REQUIRE( col[0] == 4.0 );
         REQUIRE( col[1] == 11.0 );
         REQUIRE( columnRows[0] == 1 );
         REQUIRE( columnRows[1] == 4 );
         break;
      case 3:
         REQUIRE( col[0] == 5.0 );
         REQUIRE( columnRows[0] == 1 );
         break;
      case 4:
         REQUIRE( col[0] == 6.0 );
         REQUIRE( columnRows[0] == 1 );
         break;
      case 5:
         REQUIRE( col[0] == 7.0 );
         REQUIRE( columnRows[0] == 1 );
         break;
      case 6:
         continue;
      case 7:
         REQUIRE( col[0] == 12.0 );
         REQUIRE( columnRows[0] == 4 );
         break;
      case 8:
         REQUIRE( col[0] == 13.0 );
         REQUIRE( columnRows[0] == 4 );
         break;
      default:
         REQUIRE_FALSE( "case default should not be reached" );
      }
   }
}

TEST_CASE( "transposed sparse matrix can be compressed", "[core]" )
{
   papilo::SparseStorage<double> transpose =setupSparseMatrix().getTranspose();

   papilo::Vec<int> columnSizes = { 2, 5, 1, -1, 5 };
   papilo::Vec<int> rowSizes = { 2, 4, 2, 1, 1, 1, -1, 1, 1 };
   papilo::Vec<int> col_mapping =
       transpose.compress( rowSizes, columnSizes, true );

   papilo::Vec<int> expectedValues = { 0, 1, 2, -1, 3 };
   REQUIRE( col_mapping == expectedValues );

   papilo::compress_vector( col_mapping, columnSizes);

   auto rowRanges = transpose.getRowRanges();
   auto columnValues = transpose.getValues();
   auto rows = transpose.getColumns();

   int expectedRowRanges[8] = { 2, 4, 2, 1, 1, 1, 1, 1 };


   for( int i = 0; i < 8; ++i )
   {
      int nonzerosEntriesInRow = rowRanges[i].end - rowRanges[i].start;
      REQUIRE( nonzerosEntriesInRow == expectedRowRanges[i] );
      double* col = columnValues + rowRanges[i].start;
      int* colRows = rows + rowRanges[i].start;
      switch( i )
      {
      case 0:
         REQUIRE( col[0] == 1.0 );
         REQUIRE( col[1] == 9.0 );
         REQUIRE( colRows[0] == 0 );
         REQUIRE( colRows[1] == 3 );
         break;
      case 1:
         REQUIRE( col[0] == 2.0 );
         REQUIRE( col[1] == 3.0 );
         REQUIRE( col[2] == 8.0 );
         REQUIRE( col[3] == 10.0 );
         REQUIRE( colRows[0] == 0 );
         REQUIRE( colRows[1] == 1 );
         REQUIRE( colRows[2] == 2 );
         REQUIRE( colRows[3] == 3 );
         break;
      case 2:
         REQUIRE( col[0] == 4.0 );
         REQUIRE( col[1] == 11.0 );
         REQUIRE( colRows[0] == 1 );
         REQUIRE( colRows[1] == 3 );
         break;
      case 3:
         REQUIRE( col[0] == 5.0 );
         REQUIRE( colRows[0] == 1 );
         break;
      case 4:
         REQUIRE( col[0] == 6.0 );
         REQUIRE( colRows[0] == 1 );
         break;
      case 5:
         REQUIRE( col[0] == 7.0 );
         REQUIRE( colRows[0] == 1 );
         break;
      case 6:
         REQUIRE( col[0] == 12.0 );
         REQUIRE( colRows[0] == 3 );
         break;
      case 7:
         REQUIRE( col[0] == 13.0 );
         REQUIRE( colRows[0] == 3 );
         break;
      default:
         REQUIRE_FALSE( "case default should not be reached" );
      }
   }
}

papilo::SparseStorage<double>
setupSparseMatrix()
{
   // build the triplets for the following matrix:
   // 1  2  0  0  0  0  0  0  0
   // 0  3  4  5  6  7  0  0  0
   // 0  8  0  0  0  0  0  0  0
   // 0  0  0  0  0  0  0  0  0
   // 9  10 11 0  0  0  0  12 13
   int numberRows = 5;
   int numberColumns = 9;

   papilo::Vec<papilo::Triplet<double>> triplets = {
       papilo::Triplet<double>{ 0, 0, 1.0 },
       papilo::Triplet<double>{ 0, 1, 2.0 },
       papilo::Triplet<double>{ 1, 1, 3.0 },
       papilo::Triplet<double>{ 1, 2, 4.0 },
       papilo::Triplet<double>{ 1, 3, 5.0 },
       papilo::Triplet<double>{ 1, 4, 6.0 },
       papilo::Triplet<double>{ 1, 5, 7.0 },
       papilo::Triplet<double>{ 2, 1, 8.0 },
       papilo::Triplet<double>{ 4, 0, 9.0 },
       papilo::Triplet<double>{ 4, 1, 10.0 },
       papilo::Triplet<double>{ 4, 2, 11.0 },
       papilo::Triplet<double>{ 4, 7, 12.0 },
       papilo::Triplet<double>{ 4, 8, 13.0 } };

   papilo::Vec<int> rowSize = { 2, 5, 1, -1, 5 };
   papilo::Vec<int> colSize = { 2, 4, 2, 1, 1, 1, -1, 1, 1 };
   papilo::SparseStorage<double> matrix{ triplets, numberRows, numberColumns,
                                         true };
   return matrix;
}
