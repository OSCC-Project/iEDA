/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*               This file is part of the program and library                */
/*    PaPILO --- Parallel Presolve for Integer and Linear Optimization       */
/*                                                                           */
/* Copyright (C) 2020-2022  Konrad-Zuse-Zentrum                              */
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

#include "papilo/core/ConstraintMatrix.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/VariableDomains.hpp"
#include "papilo/io/MpsParser.hpp"
#include "papilo/misc/Hash.hpp"
#include "papilo/misc/Vec.hpp"
#include "papilo/misc/fmt.hpp"
#ifdef PAPILO_TBB
#include "papilo/misc/tbb.hpp"
#endif
#include "papilo/external/pdqsort/pdqsort.h"
#include <algorithm>
#include <sys/stat.h>

using namespace papilo;

bool
fileExists( const std::string& name )
{
   struct stat buff;
   return ( stat( name.c_str(), &buff ) == 0 );
}

static std::pair<Vec<int>, Vec<int>>
compute_row_and_column_permutation( const Problem<double>& prob, bool verbose )
{
   const ConstraintMatrix<double>& consmatrix = prob.getConstraintMatrix();
   if( verbose )
      fmt::print( "Computing Permutation for {}\n", prob.getName() );

   std::pair<Vec<int>, Vec<int>> retval;
   Vec<uint64_t> rowhashes;
   Vec<uint64_t> colhashes;

   auto obj = [&]( int col ) {
      double tmp = prob.getObjective().coefficients[col];
      uint64_t val;
      std::memcpy( &val, &tmp, sizeof( double ) );
      return val;
   };

   auto lb = [&]( int col ) {
      double tmp = prob.getColFlags()[col].test( ColFlag::kLbInf )
                       ? std::numeric_limits<double>::lowest()
                       : prob.getLowerBounds()[col];
      uint64_t val;
      std::memcpy( &val, &tmp, sizeof( double ) );
      return val;
   };

   auto ub = [&]( int col ) {
      double tmp = prob.getColFlags()[col].test( ColFlag::kUbInf )
                       ? std::numeric_limits<double>::max()
                       : prob.getUpperBounds()[col];
      uint64_t val;
      std::memcpy( &val, &tmp, sizeof( double ) );
      return val;
   };

   auto lhs = [&]( int row ) {
      double tmp = consmatrix.getRowFlags()[row].test( RowFlag::kLhsInf )
                       ? std::numeric_limits<double>::lowest()
                       : consmatrix.getLeftHandSides()[row];
      uint64_t val;
      std::memcpy( &val, &tmp, sizeof( double ) );
      return val;
   };

   auto rhs = [&]( int row ) {
      double tmp = consmatrix.getRowFlags()[row].test( RowFlag::kRhsInf )
                       ? std::numeric_limits<double>::max()
                       : consmatrix.getRightHandSides()[row];
      uint64_t val;
      std::memcpy( &val, &tmp, sizeof( double ) );
      return val;
   };

   auto col_is_integral = [&]( int col ) {
      return static_cast<uint64_t>(
          prob.getColFlags()[col].test( ColFlag::kIntegral ) );
   };

   int ncols = prob.getNCols();
   int nrows = prob.getNRows();
   colhashes.resize( ncols + 2 );
   rowhashes.resize( nrows + 4 );

   const int LHS = ncols;
   const int RHS = ncols + 1;

   colhashes[LHS] = UINT64_MAX;
   colhashes[RHS] = UINT64_MAX - 1;

   const int OBJ = nrows;
   const int INTEGRAL = nrows + 1;
   const int LB = nrows + 2;
   const int UB = nrows + 3;

   rowhashes[OBJ] = UINT64_MAX - 2;
   rowhashes[INTEGRAL] = UINT64_MAX - 3;
   rowhashes[LB] = UINT64_MAX - 4;
   rowhashes[UB] = UINT64_MAX - 5;

   Vec<std::pair<uint64_t, int>> csrvals;
   Vec<int> csrstarts;

   csrstarts.resize( nrows + 1 );
   csrvals.reserve( consmatrix.getNnz() + 2 * nrows );

   for( int i = 0; i < nrows; ++i )
   {
      csrstarts[i] = csrvals.size();

      auto rowvec = consmatrix.getRowCoefficients( i );
      for( int k = 0; k < rowvec.getLength(); ++k )
      {
         uint64_t coef;
         std::memcpy( &coef, rowvec.getValues() + k, sizeof( double ) );
         csrvals.emplace_back( coef, rowvec.getIndices()[k] );
      }

      csrvals.emplace_back( lhs( i ), LHS );
      csrvals.emplace_back( rhs( i ), RHS );
   }

   csrstarts[nrows] = csrvals.size();

   Vec<std::pair<uint64_t, int>> cscvals;
   Vec<int> cscstarts;
   cscstarts.resize( ncols + 1 );
   cscvals.reserve( consmatrix.getNnz() + 4 * ncols );

   for( int i = 0; i < ncols; ++i )
   {
      cscstarts[i] = cscvals.size();

      auto colvec = consmatrix.getColumnCoefficients( i );
      for( int k = 0; k < colvec.getLength(); ++k )
      {
         uint64_t coef;
         std::memcpy( &coef, colvec.getValues() + k, sizeof( double ) );
         cscvals.emplace_back( coef, colvec.getIndices()[k] );
      }

      cscvals.emplace_back( obj( i ), OBJ );
      cscvals.emplace_back( col_is_integral( i ), INTEGRAL );
      cscvals.emplace_back( lb( i ), LB );
      cscvals.emplace_back( ub( i ), UB );
   }

   cscstarts[ncols] = cscvals.size();

   auto comp_rowvals = [&]( const std::pair<uint64_t, int>& a,
                            const std::pair<uint64_t, int>& b ) {
      return std::make_pair( a.first, colhashes[a.second] ) <
             std::make_pair( b.first, colhashes[b.second] );
   };

   auto comp_colvals = [&]( const std::pair<uint64_t, int>& a,
                            const std::pair<uint64_t, int>& b ) {
      return std::make_pair( a.first, rowhashes[a.second] ) <
             std::make_pair( b.first, rowhashes[b.second] );
   };

   int iters = 0;
   size_t lastncols = -1;
   HashMap<uint64_t, size_t> distinct_hashes( ncols );

   size_t lastnrows = -1;
   HashMap<uint64_t, size_t> distinct_row_hashes( nrows );

   Vec<int>& colperm = retval.second;
   colperm.resize( ncols );
   for( int i = 0; i < ncols; ++i )
      colperm[i] = i;

   Vec<int>& rowperm = retval.first;
   rowperm.resize( nrows );
   for( int i = 0; i < nrows; ++i )
      rowperm[i] = i;

   size_t nrows2 = nrows;
   size_t ncols2 = ncols;

   while( nrows2 != 0 )
   {
#ifdef PAPILO_TBB
      tbb::parallel_for(
          tbb::blocked_range<int>( 0, nrows2 ),
          [&]( const tbb::blocked_range<int>& r ) {
             for( int i = r.begin(); i != r.end(); ++i )
#else
                for( int i = 0; i < nrows2; ++i )
#endif
             {
                int row = rowperm[i];
                int start = csrstarts[row];
                int end = csrstarts[row + 1];
                pdqsort( &csrvals[start], &csrvals[end], comp_rowvals );

                Hasher<uint64_t> hasher( end - start );
                for( int k = start; k < end; ++k )
                {
                   hasher.addValue( csrvals[k].first );
                   hasher.addValue( colhashes[csrvals[k].second] );
                }

                rowhashes[row] = hasher.getHash() >> 1;
             }
#ifdef PAPILO_TBB
          } );
#endif
      distinct_row_hashes.clear();

      for( size_t i = 0; i < nrows2; ++i )
         distinct_row_hashes[rowhashes[rowperm[i]]] += 1;

      pdqsort( rowperm.begin(), rowperm.begin() + nrows2, [&]( int a, int b ) {
         return std::make_tuple( -distinct_row_hashes[rowhashes[a]],
                                 rowhashes[a], a ) <
                std::make_tuple( -distinct_row_hashes[rowhashes[b]],
                                 rowhashes[b], b );
      } );

      lastnrows = nrows2;
      nrows2 = 0;

      while( nrows2 < lastnrows )
      {
         uint64_t hashval = rowhashes[rowperm[nrows2]];
         size_t partitionsize = distinct_row_hashes[hashval];
         if( partitionsize <= 1 )
            break;

         nrows2 += partitionsize;
      }

      for( size_t i = nrows2; i < lastnrows; ++i )
      {
         rowhashes[rowperm[i]] = i;
      }

      if( nrows2 == lastnrows )
      {
         --nrows2;
         std::swap( rowperm[0], rowperm[nrows2] );
         rowhashes[rowperm[nrows2]] = nrows2;
      }

      if( ncols2 == 0 )
         break;

#ifdef PAPILO_TBB
      tbb::parallel_for(
          tbb::blocked_range<int>( 0, ncols2 ),
          [&]( const tbb::blocked_range<int>& r ) {
             for( int i = r.begin(); i != r.end(); ++i )
#else
                for( int i = 0; i < ncols; ++i )
#endif
             {
                int col = colperm[i];
                int start = cscstarts[col];
                int end = cscstarts[col + 1];
                pdqsort( &cscvals[start], &cscvals[end], comp_colvals );

                Hasher<uint64_t> hasher( end - start );
                for( int k = start; k < end; ++k )
                {
                   hasher.addValue( cscvals[k].first );
                   hasher.addValue( rowhashes[cscvals[k].second] );
                }

                colhashes[col] = hasher.getHash() >> 1;
             }
#ifdef PAPILO_TBB
          } );
#endif
      distinct_hashes.clear();

      for( size_t i = 0; i < ncols2; ++i )
         distinct_hashes[colhashes[colperm[i]]] += 1;

      pdqsort( colperm.begin(), colperm.begin() + ncols2, [&]( int a, int b ) {
         return std::make_pair( -distinct_hashes[colhashes[a]], colhashes[a] ) <
                std::make_pair( -distinct_hashes[colhashes[b]], colhashes[b] );
      } );

      lastncols = ncols2;
      ncols2 = 0;

      while( ncols2 < lastncols )
      {
         uint64_t hashval = colhashes[colperm[ncols2]];
         size_t partitionsize = distinct_hashes[hashval];
         if( partitionsize <= 1 )
            break;

         ncols2 += partitionsize;
      }

      for( size_t i = ncols2; i < lastncols; ++i )
      {
         colhashes[colperm[i]] = i;
      }

      ++iters;

      if( verbose )
         fmt::print(
             "iter {:3}: {:6} non unit col partitions and {:6} non unit row "
             "partitions\n",
             iters, ncols2, nrows2 );
   }

   return retval;
}

/// Returns True if variables in given permutation have same attributes
static bool
check_cols( const Problem<double>& prob1, const Problem<double>& prob2,
            Vec<int> perm1, const Vec<int> perm2 )
{
   assert( perm1.size() == perm2.size() );
   int ncols = perm1.size();

   const VariableDomains<double>& vd1 = prob1.getVariableDomains();
   const VariableDomains<double>& vd2 = prob2.getVariableDomains();

   const Vec<String> cnames1 = prob1.getVariableNames();
   const Vec<String> cnames2 = prob2.getVariableNames();

   auto printVarsAndIndex = [&]( int i1, int i2 ) {
      fmt::print( "Differing Variables: Problem 1: {:6} at index {:<5} vs ",
                  cnames1[i1], i1 );
      fmt::print( "Problem 2: {:6} at index {:<5}\n", cnames2[i2], i2 );
   };

   for( int i = 0; i < ncols; ++i )
   {
      int i1 = perm1[i];
      int i2 = perm2[i];

      if( vd1.flags[i1].test( ColFlag::kIntegral ) !=
          vd2.flags[i2].test( ColFlag::kIntegral ) )
      {
         // kein duplikat: eine variable ist ganzzahlig die andere nicht
         fmt::print( "Variable is integer the other not!\n" );
         printVarsAndIndex( i1, i2 );
         return false;
      }

      if( vd1.flags[i1].test( ColFlag::kUbInf ) !=
          vd2.flags[i2].test( ColFlag::kUbInf ) )
      {
         // kein duplikat: ein upper bound ist +infinity, der andere nicht
         fmt::print(
             "Variable's Upper Bound is +infty and the others is not!\n" );
         printVarsAndIndex( i1, i2 );
         return false;
      }

      if( vd1.flags[i1].test( ColFlag::kLbInf ) !=
          vd2.flags[i2].test( ColFlag::kLbInf ) )
      {
         // kein duplikat: ein lower bound ist -infinity, der andere nicht
         fmt::print(
             "Variable's Lower Bound is -infty and the others is not! " );
         printVarsAndIndex( i1, i2 );
         return false;
      }

      if( !vd1.flags[i1].test( ColFlag::kLbInf ) &&
          vd1.lower_bounds[i1] != vd2.lower_bounds[i2] )
      {
         assert( !vd2.flags[i2].test( ColFlag::kLbInf ) );
         // kein duplikat: lower bounds sind endlich aber unterschiedlich
         fmt::print( "Lower Bounds are different!\n" );
         printVarsAndIndex( i1, i2 );
         return false;
      }

      if( !vd1.flags[i1].test( ColFlag::kUbInf ) &&
          vd1.upper_bounds[i1] != vd2.upper_bounds[i2] )
      {
         assert( !vd2.flags[i2].test( ColFlag::kUbInf ) );
         // kein duplikat: upper bounds sind endlich aber unterschiedlich
         fmt::print( "Upper Bounds are different!\n" );
         printVarsAndIndex( i1, i2 );
         return false;
      }
   }
   return true;
}

/// Returns True if rows in given Permutation are same for also given variable
/// permutation
static bool
check_rows( const Problem<double>& prob1, const Problem<double>& prob2,
            Vec<int> permrow1, Vec<int> permrow2, Vec<int> permcol1,
            Vec<int> permcol2 )
{
   assert( permrow1.size() == permrow2.size() );
   assert( permcol1.size() == permcol2.size() );

   const ConstraintMatrix<double>& cm1 = prob1.getConstraintMatrix();
   const ConstraintMatrix<double>& cm2 = prob2.getConstraintMatrix();

   int nrows = permrow1.size();

   // Row flags
   const Vec<RowFlags>& rflags1 = cm1.getRowFlags();
   const Vec<RowFlags>& rflags2 = cm2.getRowFlags();
   // Get sides
   const Vec<double>& lhs1 = cm1.getLeftHandSides();
   const Vec<double>& lhs2 = cm2.getLeftHandSides();
   const Vec<double>& rhs1 = cm1.getRightHandSides();
   const Vec<double>& rhs2 = cm2.getRightHandSides();

   HashMap<int, double> coefmap;

   const Vec<String> cnames1 = prob1.getVariableNames();
   const Vec<String> cnames2 = prob2.getVariableNames();
   const Vec<String> rnames1 = prob1.getConstraintNames();
   const Vec<String> rnames2 = prob2.getConstraintNames();

   auto printConstraintsAndIndex = [&]( int i1, int i2 ) {
      fmt::print( "Differing Constraints: Problem 1: {:6} at index {:<5} vs ",
                  rnames1[i1], i1 );
      fmt::print( "Problem 2: {:6} at index {:<5}\n", rnames2[i2], i2 );
   };

   auto findcol = []( int col, Vec<int> perm ) {
      return std::distance( perm.begin(),
                            std::find( perm.begin(), perm.end(), col ) );
   };

   for( int i = 0; i < nrows; ++i )
   {
      int i1row = permrow1[i];
      int i2row = permrow2[i];

      // Check Row flags for dissimilarities
      if( rflags1[i1row].test( RowFlag::kLhsInf ) !=
          rflags2[i2row].test( RowFlag::kLhsInf ) )
      {
         fmt::print( "Row has infinite LHS in only one of both problems!\n" );
         printConstraintsAndIndex( i1row, i2row );
         return false;
      }

      if( rflags1[i1row].test( RowFlag::kRhsInf ) !=
          rflags2[i2row].test( RowFlag::kRhsInf ) )
      {
         fmt::print( "Row has infinite RHS in only one of both problems!\n" );
         printConstraintsAndIndex( i1row, i2row );
         return false;
      }

      if( rflags1[i1row].test( RowFlag::kEquation ) !=
          rflags2[i2row].test( RowFlag::kEquation ) )
      {
         fmt::print( "Row is equation in only one of both problems!\n" );
         printConstraintsAndIndex( i1row, i2row );
         return false;
      }

      if( rflags1[i1row].test( RowFlag::kIntegral ) !=
          rflags2[i2row].test( RowFlag::kIntegral ) )
      {
         fmt::print( "Row is Integral in only one of both problems!\n" );
         printConstraintsAndIndex( i1row, i2row );
         return false;
      }

      // needed? probably not
      if( rflags1[i1row].test( RowFlag::kRedundant ) !=
          rflags2[i2row].test( RowFlag::kRedundant ) )
      {
         fmt::print( "Row is redundant in only one of both problems!\n" );
         printConstraintsAndIndex( i1row, i2row );
         return false;
      }

      // Check Row LHS values
      if( rflags1[i1row].test( RowFlag::kLhsInf ) &&
          lhs1[i1row] != lhs2[i2row] )
      {
         assert( rflags2[i2row].test( RowFlag::kLhsInf ) );
         fmt::print( "Row has different LHS in both problems!\n" );
         printConstraintsAndIndex( i1row, i2row );
         return false;
      }

      // Check Row RHS values
      if( rflags1[i1row].test( RowFlag::kRhsInf ) &&
          rhs1[i1row] != rhs2[i2row] )
      {
         assert( rflags2[i2row].test( RowFlag::kRhsInf ) );
         fmt::print( "Row has different RHS in both problems!\n" );
         printConstraintsAndIndex( i1row, i2row );
         return false;
      }

      // Check Row coefficients
      const SparseVectorView<double> row1 = cm1.getRowCoefficients( i1row );
      const SparseVectorView<double> row2 = cm2.getRowCoefficients( i2row );

      // Assume: If there is different amounts of variables in constraint it is
      // not the same (not entirely true)
      const int curr_ncols = row1.getLength();
      if( curr_ncols != row2.getLength() )
      {
         fmt::print( "Row has different amounts of variables!\n" );
         printConstraintsAndIndex( i1row, i2row );
         return false;
      }

      const int* inds1 = row1.getIndices();
      const int* inds2 = row2.getIndices();
      const double* vals1 = row1.getValues();
      const double* vals2 = row2.getValues();

      for( int x = 0; x < curr_ncols; ++x )
      {
         int col = findcol( inds1[x], permcol1 );
         coefmap[col] = vals1[x];
      }

      for( int x = 0; x < curr_ncols; ++x )
      {
         int final_index2 = findcol( inds2[x], permcol2 );

         // Check if same variables are defined for row
         if( coefmap.count( final_index2 ) == 0 )
         {
            fmt::print( "Row has different variables!\n" );
            printConstraintsAndIndex( i1row, i2row );
            fmt::print( "Variable `{}` at index {} is defined for Problem 2, "
                        "but not for Problem1",
                        cnames2[i], i );
            return false;
         }

         // Check if values are same
         if( coefmap[final_index2] != vals2[x] )
         {
            fmt::print( "Row has different coefficients for variable!\n" );
            printConstraintsAndIndex( i1row, i2row );
            fmt::print( "Variables: Problem1: {} at {} vs Problem2: {} at {}\n",
                        coefmap[final_index2], permrow1[final_index2], vals2[x],
                        inds2[x] );
            return false;
         }
      }

      // clear map for next row
      coefmap.clear();
   }
   return true;
}

static bool
check_duplicates( const Problem<double>& prob1, const Problem<double>& prob2 )
{
   // Check for columns
   int ncols = prob1.getNCols();

   if( ncols != prob2.getNCols() )
   {
      // kein duplikat: unterschiedlich viele variablen
      fmt::print( "not same number of variables\n" );
      return false;
   }

   // Check for rows
   // First assume for being same you need to have same rows (even though not
   // true)
   int nrows = prob1.getNRows();

   if( nrows != prob2.getNRows() )
   {
      fmt::print( "not same number of rows: prob1:{} prob2:{}\n", nrows,
                  prob2.getNRows() );
      return false;
   }

   std::pair<Vec<int>, Vec<int>> perms1 =
       compute_row_and_column_permutation( prob1, true );
   std::pair<Vec<int>, Vec<int>> perms2 =
       compute_row_and_column_permutation( prob2, true );

   Vec<int>& perm_col1 = perms1.second;
   Vec<int>& perm_col2 = perms2.second;

   Vec<int>& perm_row1 = perms1.first;
   Vec<int>& perm_row2 = perms2.first;

   if( !check_cols( prob1, prob2, perm_col1, perm_col2 ) )
      return false;

   if( !check_rows( prob1, prob2, perm_row1, perm_row2, perm_col1, perm_col2 ) )
      return false;

   // All checks passed
   return true;
}

static uint64_t
compute_instancehash( const Problem<double>& prob )
{
   const int MAX_HASH_ITERS = 5;
   const ConstraintMatrix<double> cm = prob.getConstraintMatrix();

   int nrows = cm.getNRows();
   int ncols = cm.getNCols();
   int nnz = cm.getNnz();

   auto obj = [&]( int col ) {
      double tmp = prob.getObjective().coefficients[col];
      uint64_t val;
      std::memcpy( &val, &tmp, sizeof( double ) );
      return val;
   };

   auto lb = [&]( int col ) {
      double tmp = prob.getColFlags()[col].test( ColFlag::kLbInf )
                       ? std::numeric_limits<double>::lowest()
                       : prob.getLowerBounds()[col];
      uint64_t val;
      std::memcpy( &val, &tmp, sizeof( double ) );
      return val;
   };

   auto ub = [&]( int col ) {
      double tmp = prob.getColFlags()[col].test( ColFlag::kUbInf )
                       ? std::numeric_limits<double>::max()
                       : prob.getUpperBounds()[col];
      uint64_t val;
      std::memcpy( &val, &tmp, sizeof( double ) );
      return val;
   };

   auto col_is_integral = [&]( int col ) {
      return static_cast<uint64_t>(
          prob.getColFlags()[col].test( ColFlag::kIntegral ) );
   };

   auto lhs = [&]( int row ) {
      double tmp = cm.getRowFlags()[row].test( RowFlag::kLhsInf )
                       ? std::numeric_limits<double>::lowest()
                       : cm.getLeftHandSides()[row];
      uint64_t val;
      std::memcpy( &val, &tmp, sizeof( double ) );
      return val;
   };

   auto rhs = [&]( int row ) {
      double tmp = cm.getRowFlags()[row].test( RowFlag::kRhsInf )
                       ? std::numeric_limits<double>::max()
                       : cm.getRightHandSides()[row];
      uint64_t val;
      std::memcpy( &val, &tmp, sizeof( double ) );
      return val;
   };

   // Setup rowhashes and colhashes
   Vec<uint64_t> rowhashes;
   Vec<uint64_t> colhashes;
   colhashes.resize( ncols + 2 );
   rowhashes.resize( nrows + 4 );

   const int LHS = ncols;
   const int RHS = ncols + 1;

   colhashes[LHS] = UINT64_MAX;
   colhashes[RHS] = UINT64_MAX - 1;

   const int OBJ = nrows;
   const int INTEGRAL = nrows + 1;
   const int LB = nrows + 2;
   const int UB = nrows + 3;

   rowhashes[OBJ] = UINT64_MAX - 2;
   rowhashes[INTEGRAL] = UINT64_MAX - 3;
   rowhashes[LB] = UINT64_MAX - 4;
   rowhashes[UB] = UINT64_MAX - 5;

   // Datastructure to save coefficients columnwise
   Vec<std::pair<uint64_t, int>> csrvals;
   Vec<int> csrstarts;
   csrstarts.resize( nrows + 1 );
   csrvals.reserve( nnz + 2 * nrows );

   for( int i = 0; i < nrows; ++i )
   {
      csrstarts[i] = csrvals.size();

      auto rowvec = cm.getRowCoefficients( i );
      for( int k = 0; k < rowvec.getLength(); ++k )
      {
         uint64_t coef;
         std::memcpy( &coef, rowvec.getValues() + k, sizeof( double ) );
         csrvals.emplace_back( coef, rowvec.getIndices()[k] );
      }

      csrvals.emplace_back( lhs( i ), LHS );
      csrvals.emplace_back( rhs( i ), RHS );
   }

   csrstarts[nrows] = csrvals.size();

   // Datastructure to save coefficients rowwise
   Vec<std::pair<uint64_t, int>> cscvals;
   Vec<int> cscstarts;
   cscstarts.resize( ncols + 1 );
   cscvals.reserve( nnz + 4 * ncols );

   for( int i = 0; i < ncols; ++i )
   {
      cscstarts[i] = cscvals.size();

      auto colvec = cm.getColumnCoefficients( i );
      for( int k = 0; k < colvec.getLength(); ++k )
      {
         uint64_t coef;
         std::memcpy( &coef, colvec.getValues() + k, sizeof( double ) );
         cscvals.emplace_back( coef, colvec.getIndices()[k] );
      }

      cscvals.emplace_back( obj( i ), OBJ );
      cscvals.emplace_back( col_is_integral( i ), INTEGRAL );
      cscvals.emplace_back( lb( i ), LB );
      cscvals.emplace_back( ub( i ), UB );
   }

   cscstarts[ncols] = cscvals.size();

   auto comp_rowvals = [&]( const std::pair<uint64_t, int>& a,
                            const std::pair<uint64_t, int>& b ) {
      return std::make_pair( a.first, colhashes[a.second] ) <
             std::make_pair( b.first, colhashes[b.second] );
   };

   auto comp_colvals = [&]( const std::pair<uint64_t, int>& a,
                            const std::pair<uint64_t, int>& b ) {
      return std::make_pair( a.first, rowhashes[a.second] ) <
             std::make_pair( b.first, rowhashes[b.second] );
   };

   // Prepare permutation to not compute all hashes in each step newly
   Vec<int> colperm;
   colperm.resize( ncols );
   for( int i = 0; i < ncols; ++i )
      colperm[i] = i;

   Vec<int> rowperm;
   rowperm.resize( nrows );
   for( int i = 0; i < nrows; ++i )
      rowperm[i] = i;

   int iters = 0;

   int ncols2 = ncols;
   int lastncols = -1;
   HashMap<uint64_t, size_t> distinct_col_hashes( ncols );

   int nrows2 = nrows;
   int lastnrows = -1;
   HashMap<uint64_t, size_t> distinct_row_hashes( nrows );

   // Compute column and row hashes
   while( nrows2 != 0 && iters <= MAX_HASH_ITERS )
   {
#ifdef PAPILO_TBB
      tbb::parallel_for(
          tbb::blocked_range<int>( 0, nrows2 ),
          [&]( const tbb::blocked_range<int>& r ) {
             for( int i = r.begin(); i != r.end(); ++i )
#else
                for( int i = 0; i < nrows2; ++i )
#endif
             {
                int row = rowperm[i];
                int start = csrstarts[row];
                int end = csrstarts[row + 1];
                pdqsort( &csrvals[start], &csrvals[end], comp_rowvals );

                Hasher<uint64_t> hasher( end - start );
                for( int k = start; k < end; ++k )
                {
                   hasher.addValue( csrvals[k].first );
                   hasher.addValue( colhashes[csrvals[k].second] );
                }

                rowhashes[row] = hasher.getHash() >> 1;
             }
#ifdef PAPILO_TBB
          } );
#endif

      distinct_row_hashes.clear();

      for( int i = 0; i < nrows2; ++i )
         distinct_row_hashes[rowhashes[rowperm[i]]] += 1;

      pdqsort( rowperm.begin(), rowperm.begin() + nrows2, [&]( int a, int b ) {
         return std::make_tuple( -distinct_row_hashes[rowhashes[a]],
                                 rowhashes[a], a ) <
                std::make_tuple( -distinct_row_hashes[rowhashes[b]],
                                 rowhashes[b], b );
      } );

      lastnrows = nrows2;
      nrows2 = 0;

      while( nrows2 < lastnrows )
      {
         uint64_t hashval = rowhashes[rowperm[nrows2]];
         size_t partitionsize = distinct_row_hashes[hashval];
         if( partitionsize <= 1 )
            break;

         nrows2 += partitionsize;
      }

      for( size_t i = nrows2; i < lastnrows; ++i )
      {
         rowhashes[rowperm[i]] = i;
      }

      if( nrows2 == lastnrows )
      {
         --nrows2;
         std::swap( rowperm[0], rowperm[nrows2] );
         rowhashes[rowperm[nrows2]] = nrows2;
      }

      if( ncols2 == 0 )
         break;

#ifdef PAPILO_TBB
      tbb::parallel_for(
          tbb::blocked_range<int>( 0, ncols2 ),
          [&]( const tbb::blocked_range<int>& r ) {
             for( int i = r.begin(); i != r.end(); ++i )
#else
       for( int i = 0; i < ncols; ++i )
#endif
             {
                int col = colperm[i];
                int start = cscstarts[col];
                int end = cscstarts[col + 1];
                pdqsort( &cscvals[start], &cscvals[end], comp_colvals );

                Hasher<uint64_t> hasher( end - start );
                for( int k = start; k < end; ++k )
                {
                   hasher.addValue( cscvals[k].first );
                   hasher.addValue( rowhashes[cscvals[k].second] );
                }

                colhashes[col] = hasher.getHash() >> 1;
             }
#ifdef PAPILO_TBB
          } );
#endif
      distinct_col_hashes.clear();

      for( int i = 0; i < ncols2; ++i )
         distinct_col_hashes[colhashes[colperm[i]]] += 1;

      pdqsort( colperm.begin(), colperm.begin() + ncols2, [&]( int a, int b ) {
         return std::make_pair( -distinct_col_hashes[colhashes[a]],
                                colhashes[a] ) <
                std::make_pair( -distinct_col_hashes[colhashes[b]],
                                colhashes[b] );
      } );

      lastncols = ncols2;
      ncols2 = 0;

      while( ncols2 < lastncols )
      {
         uint64_t hashval = colhashes[colperm[ncols2]];
         size_t partitionsize = distinct_col_hashes[hashval];
         if( partitionsize <= 1 )
            break;

         ncols2 += partitionsize;
      }

      for( size_t i = ncols2; i < lastncols; ++i )
      {
         colhashes[colperm[i]] = i;
      }

      ++iters;
   }
   // Sort hashes
   pdqsort( rowhashes.begin(), rowhashes.end(),
            []( uint64_t a, uint64_t b ) { return a < b; } );
   pdqsort( colhashes.begin(), colhashes.end(),
            []( uint64_t a, uint64_t b ) { return a < b; } );

   // Put all values in the hasher
   Hasher<uint64_t> hasher( nnz );
   hasher.addValue( nrows );
   hasher.addValue( ncols );
   hasher.addValue( prob.getNumIntegralCols() );
   hasher.addValue( prob.getNumContinuousCols() );
   for( uint64_t hash : rowhashes )
      hasher.addValue( hash );
   for( uint64_t hash : colhashes )
      hasher.addValue( hash );
   return hasher.getHash();
}

int
main( int argc, char* argv[] )
{
   if( argc != 2 && argc != 3 )
   {
      fmt::print( "usage:\n" );
      fmt::print( "./check_duplicates instance1.mps instance2.mps  - check for "
                  "duplicates\n" );
      fmt::print( "./check_duplicates instance1.mps                - compute "
                  "unique hash for instance" );
      return 1;
   }
   assert( argc == 2 || argc == 3 );

   // Load and check problem 1
   if( !fileExists( argv[1] ) )
   {
      fmt::print( "Error: Can not find instance at `{}`\n", argv[1] );
      return 1;
   }
   boost::optional<Problem<double>> prob1t =
       MpsParser<double>::loadProblem( argv[1] );
   if( !prob1t )
   {
      fmt::print( "error loading problem {}\n", argv[1] );
      return 1;
   }
   Problem<double> prob1 = *prob1t;

   if( argc == 2 )
   {
      uint64_t result = compute_instancehash( prob1 );
      fmt::print( "{}\n", result );
   }
   else
   {
      // Load and check problem 2
      if( !fileExists( argv[2] ) )
      {
         fmt::print( "Error: Can not find instance at `{}`\n", argv[2] );
         return 1;
      }
      boost::optional<Problem<double>> prob2t =
          MpsParser<double>::loadProblem( argv[2] );
      if( !prob2t )
      {
         fmt::print( "error loading problem {}\n", argv[1] );
         return 1;
      }
      Problem<double> prob2 = *prob2t;
      fmt::print( "duplicates: {}\n", check_duplicates( prob1, prob2 ) );
   }

   return 0;
}
