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

#ifndef _PAPILO_PARALLEL_ROW_DETECTION_HPP_
#define _PAPILO_PARALLEL_ROW_DETECTION_HPP_

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"
#include "papilo/misc/Hash.hpp"
#ifdef PAPILO_TBB
#include "papilo/misc/tbb.hpp"
#endif
#include "papilo/external/pdqsort/pdqsort.h"

namespace papilo
{

template <typename REAL>
// Identical row reduction needs to be done before
class ParallelRowDetection : public PresolveMethod<REAL>
{
   struct SupportHashCompare
   {
      SupportHashCompare() = default;

      static size_t
      hash( const std::pair<int, const int*>& row )
      {
         Hasher<size_t> hasher( row.first );

         const int* support = row.second;

         for( int i = 0; i != row.first; ++i )
         {
            hasher.addValue( support[i] );
         }

         return hasher.getHash();
      }

      static bool
      equal( const std::pair<int, const int*>& row1,
             const std::pair<int, const int*>& row2 )
      {
         int length = row1.first;

         if( length != row2.first )
            return false;

         return memcmp( static_cast<const void*>( row1.second ),
                        static_cast<const void*>( row2.second ),
                        length * sizeof( int ) ) == 0;
      }
   };

   struct SupportHash
   {
      std::size_t
      operator()( const std::pair<int, const int*>& row ) const
      {
         return SupportHashCompare::hash( row );
      }
   };

   struct SupportEqual
   {
      bool
      operator()( const std::pair<int, const int*>& row1,
                  const std::pair<int, const int*>& row2 ) const
      {
         return SupportHashCompare::equal( row1, row2 );
      }
   };

   void
   findParallelRows( const Num<REAL>& num, const int* bucket, int bucketsize,
                     const ConstraintMatrix<REAL>& constMatrix,
                     Vec<int>& parallel_rows );

   void
   computeRowHashes( const ConstraintMatrix<REAL>& constMatrix,
                     unsigned int* rowhashes );

   void
   computeSupportId( const ConstraintMatrix<REAL>& constMatrix,
                     unsigned int* supporthashes );

   int
   determineBucketSize( int nRows, std::unique_ptr<unsigned int[]>& supportid,
                        std::unique_ptr<unsigned int[]>& coefhash,
                        std::unique_ptr<int[]>& row, int i );

 public:
   ParallelRowDetection() : PresolveMethod<REAL>()
   {
      this->setName( "parallelrows" );
      this->setTiming( PresolverTiming::kMedium );
   }

   PresolveStatus
   execute( const Problem<REAL>& problem,
            const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
            Reductions<REAL>& reductions, const Timer& timer) override;
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class ParallelRowDetection<double>;
extern template class ParallelRowDetection<Quad>;
extern template class ParallelRowDetection<Rational>;
#endif

template <typename REAL>
void
ParallelRowDetection<REAL>::findParallelRows(
    const Num<REAL>& num, const int* bucket, int bucketsize,
    const ConstraintMatrix<REAL>& constMatrix, Vec<int>& parallel_rows )
{
   // TODO if bucketsize too large do gurobi trick
   auto row1 = constMatrix.getRowCoefficients( bucket[0] );

   const int length = row1.getLength();
   const REAL* coefs1 = row1.getValues();

   if( length < 2 )
      return;

   parallel_rows.push_back( bucket[0] );

   for( int j = 1; j < bucketsize; ++j )
   {
      auto row2 = constMatrix.getRowCoefficients( bucket[j] );

      // support should already be checked
      assert( length == row2.getLength() );
      assert( std::memcmp( static_cast<const void*>( row1.getIndices() ),
                           static_cast<const void*>( row2.getIndices() ),
                           length * sizeof( int ) ) == 0 );

      const REAL* coefs2 = row2.getValues();

      bool parallel = true;

      if( num.isGE( abs( coefs1[0] ), abs( coefs2[0] ) ) )
      {
         REAL scale2 = coefs1[0] / coefs2[0];

         for( int k = 1; k < length; ++k )
         {
            if( !num.isEq( coefs1[k], scale2 * coefs2[k] ) )
            {
               parallel = false;
               break;
            }
         }

         if( parallel )
            parallel_rows.push_back( bucket[j] );
      }
      else
      {
         REAL scale1 = coefs2[0] / coefs1[0];
         for( int k = 1; k < length; ++k )
         {
            if( !num.isEq( scale1 * coefs1[k], coefs2[k] ) )
            {
               parallel = false;
               break;
            }
         }
         if( parallel )
            parallel_rows.push_back( bucket[j] );
      }
   }
   if( parallel_rows.size() == 1 )
      parallel_rows.clear();
}

template <typename REAL>
void
ParallelRowDetection<REAL>::computeRowHashes(
    const ConstraintMatrix<REAL>& constMatrix, unsigned int* rowhashes )
{
#ifdef PAPILO_TBB
   tbb::parallel_for(
       tbb::blocked_range<int>( 0, constMatrix.getNRows() ),
       [&]( const tbb::blocked_range<int>& r ) {
          for( int i = r.begin(); i != r.end(); ++i )
#else
   for( int i = 0; i != constMatrix.getNRows(); ++i )
#endif
          {
             // compute hash-value for coefficients

             auto rowcoefs = constMatrix.getRowCoefficients( i );
             const REAL* rowvals = rowcoefs.getValues();
             const int len = rowcoefs.getLength();

             Hasher<unsigned int> hasher( len );
             // only makes sense for non-singleton rows
             // (should not occur after redundant rows are
             // already deleted)
             if( len > 1 )
             {
                // compute scale such that the first coefficient is
                // positive 1/golden ratio. The choice of
                // the constant is arbitrary and is used to make cases
                // where two coefficients that are equal
                // within epsilon get different values are
                // more unlikely by choosing some irrational number
                REAL scale = REAL( 2.0 / ( 1.0 + sqrt( 5.0 ) ) ) / rowvals[0];

                // add scaled coefficients of other row
                // entries to compute the hash
                for( int j = 1; j != len; ++j )
                {
                   hasher.addValue( Num<REAL>::hashCode( rowvals[j] * scale ) );
                }
             }

             rowhashes[i] = hasher.getHash();
          }
#ifdef PAPILO_TBB
       } );
#endif
}

template <typename REAL>
void
ParallelRowDetection<REAL>::computeSupportId(
    const ConstraintMatrix<REAL>& constMatrix, unsigned int* supporthashes )
{
   using SupportMap =
       HashMap<std::pair<int, const int*>, int, SupportHash, SupportEqual>;

   SupportMap supportMap(
       static_cast<std::size_t>( constMatrix.getNRows() * 1.1 ) );

   for( int i = 0; i < constMatrix.getNRows(); ++i )
   {
      auto row = constMatrix.getRowCoefficients( i );
      int length = row.getLength();
      const int* support = row.getIndices();

      auto insResult =
          supportMap.emplace( std::make_pair( length, support ), i );

      if( insResult.second )
         supporthashes[i] = i;
      else // support already exists, use the previous support id
         supporthashes[i] = insResult.first->second;
   }
}

template <typename REAL>
PresolveStatus
ParallelRowDetection<REAL>::execute( const Problem<REAL>& problem,
                                     const ProblemUpdate<REAL>& problemUpdate,
                                     const Num<REAL>& num,
                                     Reductions<REAL>& reductions, const Timer& timer )
{
   const auto& constMatrix = problem.getConstraintMatrix();
   const auto& lhs_values = constMatrix.getLeftHandSides();
   const auto& rhs_values = constMatrix.getRightHandSides();
   const auto& rflags = constMatrix.getRowFlags();
   const int nRows = constMatrix.getNRows();
   const Vec<int>& rowperm = problemUpdate.getRandomRowPerm();

   PresolveStatus result = PresolveStatus::kUnchanged;

   // get called less and less over time regardless of success since the
   // presolver can be expensive otherwise
   this->skipRounds( this->getNCalls() );

   // lambda mark all reductions except one redundant and update rhs/lhs
   auto handleRows = [&reductions, &result, &lhs_values, &rhs_values, &rflags,
                      &num, &constMatrix]( Vec<int> parallel_rows ) {
      using std::swap;

      assert( parallel_rows.size() >= 2 );
      int remaining_row = parallel_rows[0];
      bool is_remaining_row_equality =
          rflags[remaining_row].test( RowFlag::kEquation );
      REAL coefficient =
          constMatrix.getRowCoefficients( remaining_row ).getValues()[0];
      bool rhs_infinity = rflags[remaining_row].test( RowFlag::kRhsInf );
      bool lhs_infinity = rflags[remaining_row].test( RowFlag::kLhsInf );
      REAL rhs_value = rhs_values[remaining_row];
      int row_with_best_rhs_value = remaining_row;
      REAL lhs_value = lhs_values[remaining_row];
      int row_with_best_lhs_value = remaining_row;

      //iterates over parallel_rows, stores the remaining row and updates rhs/lhs
      for( int i = 1; i < (int) parallel_rows.size(); i++ )
      {
         int parallel_row = parallel_rows[i];

         const REAL coefs2 =
             constMatrix.getRowCoefficients( parallel_row ).getValues()[0];

         REAL ratio = coefficient / coefs2;
         REAL scaled_rhs = rhs_values[parallel_row] * ratio;
         REAL scaled_lhs = lhs_values[parallel_row] * ratio;
         if( ratio < REAL{ 0.0 } )
            swap( scaled_lhs, scaled_rhs );

         // CASE 1: 2 equalities
         if( rflags[parallel_row].test( RowFlag::kEquation ) &&
             is_remaining_row_equality )
         {
            if( !num.isFeasEq( rhs_value, scaled_rhs ) )
            {
               result = PresolveStatus::kInfeasible;
               break;
            }
            if( !num.isGE( abs( coefficient ), abs( coefs2 ) ) )
            {
               remaining_row = parallel_row;
               coefficient = coefs2;
               rhs_value = rhs_values[remaining_row];
               lhs_value = lhs_values[remaining_row];
               row_with_best_rhs_value = parallel_row;
               row_with_best_lhs_value = parallel_row;
            }
         }
         // CASE 2: new equation is equality
         else if( rflags[parallel_row].test( RowFlag::kEquation ) )
         {
            if( ( ! rhs_infinity && num.isLT( rhs_value, scaled_rhs ) ) ||
                ( ! lhs_infinity && num.isGT( lhs_value, scaled_lhs ) ) )
            {
               result = PresolveStatus::kInfeasible;
               break;
            }
            remaining_row = parallel_row;
            is_remaining_row_equality = true;
            rhs_infinity = false;
            lhs_infinity = false;
            coefficient = coefs2;
            rhs_value = rhs_values[remaining_row];
            lhs_value = lhs_values[remaining_row];
            row_with_best_rhs_value = parallel_row;
            row_with_best_lhs_value = parallel_row;
         }
         // CASE 3: stored equation is equality
         else if( is_remaining_row_equality )
         {
            bool scaled_lhs_inf = rflags[parallel_row].test( RowFlag::kLhsInf );
            bool scaled_rhs_inf = rflags[parallel_row].test( RowFlag::kRhsInf );
            if( ratio < REAL{ 0.0 } )
               swap( scaled_lhs_inf, scaled_rhs_inf );
            if( ( ! scaled_rhs_inf &&
                  num.isLT( scaled_rhs, rhs_value ) ) ||
                ( ! scaled_lhs_inf &&
                  num.isGT( scaled_lhs, lhs_value ) ) )
            {
               result = PresolveStatus::kInfeasible;
               break;
            }
         }
         // CASE 4: two inequalities
         else
         {
            bool scaled_lhs_inf = rflags[parallel_row].test( RowFlag::kLhsInf );
            bool scaled_rhs_inf = rflags[parallel_row].test( RowFlag::kRhsInf );
            if( ratio < REAL{ 0.0 } )
               swap( scaled_lhs_inf, scaled_rhs_inf );
            if( ( !rflags[remaining_row].test( RowFlag::kRhsInf ) &&
                  !scaled_lhs_inf &&
                  num.isLT( rhs_value, scaled_lhs ) ) ||
                ( !rflags[remaining_row].test( RowFlag::kLhsInf ) &&
                  !scaled_rhs_inf &&
                  num.isLT( scaled_rhs, lhs_value ) ) )
            {
               result = PresolveStatus::kInfeasible;
               break;
            }
            if( ! num.isGE( abs( coefficient ), abs( coefs2 ) ) )
            {
               REAL new_ratio = coefs2 / coefficient;
               REAL new_adjusted_rhs = rhs_value * new_ratio;
               REAL new_adjusted_lhs = lhs_value * new_ratio;
               if( new_ratio < REAL{ 0.0 } )
               {
                  swap( new_adjusted_lhs, new_adjusted_rhs );
                  swap( lhs_infinity, rhs_infinity );
               }

               remaining_row = parallel_row;
               coefficient = coefs2;
               if( !rhs_infinity &&
                   ( scaled_rhs_inf ||
                     num.isLT( new_adjusted_rhs, rhs_values[parallel_row] ) ) )
               {
                  rhs_value = new_adjusted_rhs;
                  rhs_infinity = false;
               }
               else
               {
                  rhs_value = rhs_values[parallel_row];
                  rhs_infinity = rflags[parallel_row].test( RowFlag::kRhsInf );
                  row_with_best_rhs_value = parallel_row;

               }
               if( !lhs_infinity &&
                   ( scaled_lhs_inf ||
                     num.isGT( new_adjusted_lhs, lhs_values[parallel_row] ) ) )
               {
                  lhs_value = new_adjusted_lhs;
                  lhs_infinity = false;
               }
               else
               {
                  lhs_value = lhs_values[parallel_row];
                  lhs_infinity = rflags[parallel_row].test( RowFlag::kLhsInf );
                  row_with_best_lhs_value = parallel_row;
               }
            }
            else
            {
               if( !scaled_rhs_inf &&
                   ( rhs_infinity ||
                     num.isLT( scaled_rhs, rhs_value ) ) )
               {
                  rhs_value = scaled_rhs;
                  rhs_infinity = false;
                  row_with_best_rhs_value = parallel_row;
               }

               if( !scaled_lhs_inf &&
                   ( lhs_infinity ||
                     num.isGT( scaled_lhs, lhs_value ) ) )
               {
                  lhs_value = scaled_lhs;
                  lhs_infinity = false;
                  row_with_best_lhs_value = parallel_row;
               }
            }
         }
      }

      TransactionGuard<REAL> guard{ reductions };
      reductions.lockRow( remaining_row );
      for( int parallel_row : parallel_rows )
      {
         if( parallel_row != remaining_row )
            reductions.lockRow( parallel_row );
      }
      if( lhs_infinity != rflags[remaining_row].test( RowFlag::kLhsInf ) ||
          lhs_value != lhs_values[remaining_row] )
      {
         reductions.bound_change_caused_by_row( remaining_row, row_with_best_lhs_value );
         reductions.change_row_lhs_parallel( remaining_row, lhs_value );
      }
      if( rhs_infinity != rflags[remaining_row].test( RowFlag::kRhsInf ) ||
          rhs_value != rhs_values[remaining_row] )
      {
         reductions.bound_change_caused_by_row( remaining_row, row_with_best_rhs_value );
         reductions.change_row_rhs_parallel( remaining_row, rhs_value );
      }
      for( int parallel_row : parallel_rows )
      {
         if( parallel_row != remaining_row )
            reductions.markRowRedundant( parallel_row );
      }
   };

   assert( nRows > 0 );

   std::unique_ptr<unsigned int[]> supportid{ new unsigned int[nRows] };
   std::unique_ptr<unsigned int[]> coefhash{ new unsigned int[nRows] };
   std::unique_ptr<int[]> row{ new int[nRows] };

#ifdef PAPILO_TBB
   tbb::parallel_invoke(
       [nRows, &row]() {
          for( int i = 0; i < nRows; ++i )
             row[i] = i;
       },
       [&constMatrix, &coefhash, this]() {
          computeRowHashes( constMatrix, coefhash.get() );
       },
       [&constMatrix, &supportid, this]() {
          computeSupportId( constMatrix, supportid.get() );
       } );
#else
   for( int i = 0; i < nRows; ++i )
      row[i] = i;
   computeRowHashes( constMatrix, coefhash.get() );
   computeSupportId( constMatrix, supportid.get() );
#endif

   pdqsort( row.get(), row.get() + nRows, [&]( int a, int b ) {
      return supportid[a] < supportid[b] ||
             ( supportid[a] == supportid[b] && coefhash[a] < coefhash[b] ) ||
             ( supportid[a] == supportid[b] && coefhash[a] == coefhash[b] &&
               rowperm[a] < rowperm[b] );
   } );

   Vec<Vec<int>> stored_parallel_rows;

   for( int i = 0; i < nRows; )
   {
      int bucketSize =
          determineBucketSize( nRows, supportid, coefhash, row, i );

      // if more  than one row is in the bucket try to find parallel rows
      if( bucketSize > 1 )
      {
         Vec<int> parallel_rows;
         parallel_rows.reserve( bucketSize );
         findParallelRows( num, row.get() + i, bucketSize, constMatrix,
                           parallel_rows );
         if( !parallel_rows.empty() )
            stored_parallel_rows.emplace_back( parallel_rows );
      }
      i = bucketSize + i;
   }

   if( !stored_parallel_rows.empty() )
   {
      result = PresolveStatus::kReduced;

      for( const Vec<int>& parallel_rows : stored_parallel_rows )
      {

         assert( !parallel_rows.empty() );
         handleRows( parallel_rows );

         if( result == PresolveStatus::kInfeasible )
            break;
      }
   }

   return result;
}

template <typename REAL>
int
ParallelRowDetection<REAL>::determineBucketSize(
    int nRows, std::unique_ptr<unsigned int[]>& supportid,
    std::unique_ptr<unsigned int[]>& coefhash, std::unique_ptr<int[]>& row,
    int i )
{
   int j;
   for( j = i + 1; j < nRows; ++j )
   {
      if( coefhash[row[i]] != coefhash[row[j]] ||
          supportid[row[i]] != supportid[row[j]] )
      {
         break;
      }
   }
   assert( j > i );
   return j - i;
}

} // namespace papilo

#endif
