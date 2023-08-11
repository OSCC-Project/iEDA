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

#ifndef _PAPILO_PRESOLVERS_SPARSIFY_HPP_
#define _PAPILO_PRESOLVERS_SPARSIFY_HPP_

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"
#include "papilo/core/SingleRow.hpp"
#include "papilo/misc/Hash.hpp"
#ifdef PAPILO_TBB
#include "papilo/misc/tbb.hpp"
#endif

namespace papilo
{

template <typename REAL>
class Sparsify : public PresolveMethod<REAL>
{
   double maxscale = 1000;

   using HitCount = uint16_t;

   struct SparsifyData
   {
      Vec<HitCount> candrowhits;
      Vec<int> candrows;
      Vec<std::pair<int, REAL>> sparsify;
      Vec<std::tuple<int, int, int>> reductionBuffer;

      explicit SparsifyData( int nrows ) : candrowhits( nrows )
      {
         candrows.reserve( nrows );
      }
   };

 public:
   Sparsify() : PresolveMethod<REAL>()
   {
      this->setName( "sparsify" );
      this->setTiming( PresolverTiming::kExhaustive );
      this->setDelayed( true );
   }

   void
   addPresolverParams( ParameterSet& paramSet ) override
   {
      paramSet.addParameter(
          "sparsify.maxscale",
          "maximum absolute scale to use for cancelling nonzeros",
          this->maxscale, 1.0 );
   }

   PresolveStatus
   execute( const Problem<REAL>& problem,
            const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
            Reductions<REAL>& reductions, const Timer& timer ) override;
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class Sparsify<double>;
extern template class Sparsify<Quad>;
extern template class Sparsify<Rational>;
#endif

template <typename REAL>
PresolveStatus
Sparsify<REAL>::execute( const Problem<REAL>& problem,
                         const ProblemUpdate<REAL>& problemUpdate,
                         const Num<REAL>& num, Reductions<REAL>& reductions, const Timer& timer )
{
   // go over the rows and get the equalities, extract the columns that
   // verify the conditions add them to a hash map, loop over the hash map
   // and compute the implied bounds and finally look for implied free
   // variables and add reductions
   const auto& domains = problem.getVariableDomains();
   const auto& lower_bounds = domains.lower_bounds;
   const auto& upper_bounds = domains.upper_bounds;
   const auto& cflags = domains.flags;

   const ConstraintMatrix<REAL>& consmatrix = problem.getConstraintMatrix();

   const auto& rflags = consmatrix.getRowFlags();
   const auto& rowsize = consmatrix.getRowSizes();
   const auto& nrows = consmatrix.getNRows();

   auto isBinaryCol = [&]( int col ) {
      return cflags[col].test( ColFlag::kIntegral ) &&
             !cflags[col].test( ColFlag::kUnbounded ) &&
             lower_bounds[col] == 0 && upper_bounds[col] == 1;
   };

   PresolveStatus result = PresolveStatus::kUnchanged;

   // after each call skip more rounds to not call sparsify too often
   this->skipRounds( this->getNCalls() );

   Vec<int> equalities;
   equalities.reserve( nrows );

   for( int i = 0; i < nrows; ++i )
   {
      if( rflags[i].test( RowFlag::kRedundant ) ||
          !rflags[i].test( RowFlag::kEquation ) || rowsize[i] <= 1 ||
          rowsize[i] > std::numeric_limits<HitCount>::max() )
         continue;

      assert( !rflags[i].test( RowFlag::kLhsInf, RowFlag::kRhsInf ) &&
              consmatrix.getLeftHandSides()[i] ==
                  consmatrix.getRightHandSides()[i] );

      equalities.emplace_back( i );
   }

#ifdef PAPILO_TBB
   tbb::combinable<SparsifyData> sparsifyData(
       [nrows]() { return SparsifyData( nrows ); } );

   tbb::parallel_for(
       tbb::blocked_range<int>( 0, static_cast<int>( equalities.size() ) ),
       [&]( const tbb::blocked_range<int>& r ) {
          SparsifyData& localData = sparsifyData.local();
          std::size_t sparsifyStart;

          auto& candrowhits = localData.candrowhits;
          auto& candrows = localData.candrows;
          auto& sparsify = localData.sparsify;
          auto& reductionBuffer = localData.reductionBuffer;

          for( int i = r.begin(); i < r.end(); ++i )
#else
   SparsifyData s = SparsifyData(nrows);
   auto& candrowhits = s.candrowhits;
   auto& candrows = s.candrows;
   auto& sparsify = s.sparsify;
   std::size_t sparsifyStart;
   auto& reductionBuffer = s.reductionBuffer;
   for( int i = 0; i < (int) equalities.size(); ++i )
#endif
          {
             int eqrow = equalities[i];

             auto rowvec = consmatrix.getRowCoefficients( eqrow );

             int eqlen = rowvec.getLength();
             const int* eqcols = rowvec.getIndices();
             bool cancelint = true;
             int minhits = eqlen - 1;
             int nint = 0;
             Message::debug(
                 this,
                 "trying sparsification with equality row {} of length {}\n",
                 eqrow, eqlen );

             if( problem.getNumIntegralCols() != 0 )
             {
                int ncont = 0;
                int nbin = 0;

                for( int counter = 0; counter != eqlen; ++counter )
                {
                   int col = eqcols[counter];

                   if( !cflags[col].test( ColFlag::kIntegral ) )
                      ++ncont;
                   else if( isBinaryCol( col ) )
                      ++nbin;
                   else
                   {
                      ++nint;
                      continue;
                   }

                   auto colvec = consmatrix.getColumnCoefficients( col );
                   const int* colrows = colvec.getIndices();
                   int collen = colvec.getLength();

                   for( int j = 0; j != collen; ++j )
                   {
                      int row = colrows[j];

                      if( row == eqrow )
                         continue;

                      if( candrowhits[row] == 0 )
                      {
                         if( nbin + ncont > 2 )
                            continue;

                         candrows.push_back( row );
                      }

                      ++candrowhits[row];
                   }
                }

                if( nbin + nint == 0 )
                {
                   auto it = std::remove_if( candrows.begin(), candrows.end(),
                                             [&]( int _r ) {
                                                if( candrowhits[_r] < ncont - 1 )
                                                {
                                                   candrowhits[_r] = 0;
                                                   return true;
                                                }
                                                return false;
                                             } );

                   cancelint = false;

                   candrows.erase( it, candrows.end() );
                }
                else
                {
                   auto it = std::remove_if(
                       candrows.begin(), candrows.end(), [&]( int _r ) {
                          if( candrowhits[_r] < nbin + ncont - 1 )
                          {
                             candrowhits[_r] = 0;
                             return true;
                          }
                          if( cancelint )
                             candrowhits[_r] = nbin + ncont;
                          return false;
                       } );

                   candrows.erase( it, candrows.end() );

                   minhits = eqlen;
                }
             }

             if( problem.getNumIntegralCols() == 0 || nint != 0 )
             {
                for( int counter = 0; counter != eqlen; ++counter )
                {
                   int col = eqcols[counter];

                   if( problem.getNumIntegralCols() != 0 &&
                       ( !cflags[col].test( ColFlag::kIntegral ) ||
                         isBinaryCol( col ) ) )
                      continue;

                   auto colvec = consmatrix.getColumnCoefficients( col );
                   const int* colrows = colvec.getIndices();
                   int collen = colvec.getLength();

                   for( int j = 0; j != collen; ++j )
                   {
                      int row = colrows[j];

                      if( row == eqrow )
                         continue;

                      if( candrowhits[row] == 0 )
                      {
                         if( counter > eqlen - minhits )
                            continue;

                         candrows.push_back( row );
                      }

                      ++candrowhits[row];
                   }
                }

                auto it = std::remove_if( candrows.begin(), candrows.end(),
                                          [&]( int _r ) {
                                             if( candrowhits[_r] < minhits )
                                             {
                                                candrowhits[_r] = 0;
                                                return true;
                                             }
                                             return false;
                                          } );

                candrows.erase( it, candrows.end() );
             }

             if( !candrows.empty() )
             {
                Vec<REAL> scales( eqlen );
                const REAL* eqvals = rowvec.getValues();

                sparsifyStart = sparsify.size();
                sparsify.reserve( sparsifyStart + candrows.size() );

                for( int candrow : candrows )
                {
                   auto candrowvec = consmatrix.getRowCoefficients( candrow );
                   const int* candcols = candrowvec.getIndices();
                   const REAL* candvals = candrowvec.getValues();
                   int candlen = candrowvec.getLength();

                   if( !cancelint && candrowhits[candrow] != eqlen )
                   {
                      bool has_integral = false;
                      for( int j = 0; j != candlen; ++j )
                      {
                         if( cflags[candcols[j]].test( ColFlag::kIntegral ) )
                         {
                            has_integral = true;
                            break;
                         }
                      }

                      if( has_integral )
                         continue;
                   }

                   int h = 0;
                   int j = 0;

                   int currcancel = 0;

                   while( h != eqlen && j != candlen )
                   {
                      if( eqcols[h] == candcols[j] )
                      {
                         scales[h] = -candvals[j] / eqvals[h];

                         ++h;
                         ++j;
                      }
                      else if( eqcols[h] < candcols[j] )
                      {
                         --currcancel;
                         scales[h] = 0;
                         ++h;
                      }
                      else
                      {
                         ++j;
                      }
                   }

                   while( h != eqlen )
                   {
                      --currcancel;
                      scales[h] = 0;
                      ++h;
                   }

                   pdqsort( scales.begin(), scales.end() );

                   int bestcancel = 0;
                   REAL bestscale = 0;

                   for( int k = 0; k != eqlen - 1; ++k )
                   {
                      if( scales[k] == 0 || abs( scales[k] ) > maxscale )
                         continue;

                      int ncancel = currcancel;

                      for( int l = k + 1; l != eqlen; ++l )
                      {
                         if( num.isEq( scales[k], scales[l] ) )
                            ++ncancel;
                         else
                            break;
                      }

                      if( ncancel > bestcancel )
                      {
                         bestcancel = ncancel;
                         bestscale = scales[k];
                      }
                   }

                   if( bestcancel > 0 )
                   {
                      Message::debug(
                          this,
                          "equation row{} cancels {} nonzeros on row{} "
                          "with scale {}\n",
                          eqrow, bestcancel, candrow, bestscale );

                      sparsify.emplace_back( candrow, bestscale );
                   }
                }

                for( int candrow : candrows )
                   candrowhits[candrow] = 0;
                candrows.clear();

                if( sparsify.size() != sparsifyStart )
                   reductionBuffer.emplace_back( eqrow, int( sparsifyStart ),
                                                 int( sparsify.size() ) );
             }
          }
#ifdef PAPILO_TBB
       } );
#endif
   int nreductions = 0;
#ifdef PAPILO_TBB
   sparsifyData.combine_each( [&]( const SparsifyData& localData ) {
      nreductions += localData.reductionBuffer.size();
   } );
#else
   nreductions = s.reductionBuffer.size();
#endif

   if( nreductions != 0 )
   {
      result = PresolveStatus::kReduced;

      Vec<std::tuple<int, int, std::pair<int, REAL>*>> reductionData;

      reductionData.reserve( nreductions );

#ifdef PAPILO_TBB
      sparsifyData.combine_each( [&]( SparsifyData& localData ) {
         for( const std::tuple<int, int, int>& reductionTuple :
              localData.reductionBuffer )
         {
            int eqrow = std::get<0>( reductionTuple );
            int start = std::get<1>( reductionTuple );
            int end = std::get<2>( reductionTuple );
            reductionData.emplace_back( eqrow, end - start,
                                        &localData.sparsify[start] );
         }
      } );
#else
      for( const std::tuple<int, int, int>& reductionTuple :
           s.reductionBuffer )
      {
         int eqrow = std::get<0>( reductionTuple );
         int start = std::get<1>( reductionTuple );
         int end = std::get<2>( reductionTuple );
         reductionData.emplace_back( eqrow, end - start,
                                     &s.sparsify[start] );
      }
#endif

      const Vec<int>& rowperm = problemUpdate.getRandomRowPerm();

      pdqsort( reductionData.begin(), reductionData.end(),
               [&]( const std::tuple<int, int, std::pair<int, REAL>*>& a,
                    const std::tuple<int, int, std::pair<int, REAL>*>& b ) {
                  int eqrowA = std::get<0>( a );
                  int eqrowB = std::get<0>( b );

                  int numCandsA = std::get<1>( a );
                  int numCandsB = std::get<1>( b );

                  return std::make_tuple( rowsize[eqrowA], -numCandsA,
                                          rowperm[eqrowA] ) <
                         std::make_tuple( rowsize[eqrowB], -numCandsB,
                                          rowperm[eqrowB] );
               } );

      for( const std::tuple<int, int, std::pair<int, REAL>*>& reductionTuple :
           reductionData )
      {
         int equality_row = std::get<0>( reductionTuple );
         int numeric = std::get<1>( reductionTuple );
         const std::pair<int, REAL>* sparsify = std::get<2>( reductionTuple );

         TransactionGuard<REAL> tg{ reductions };
         reductions.lockRow( equality_row );
         reductions.sparsify( equality_row, numeric, sparsify );
      }
   }

   Message::debug( this, "sparsify finished\n" );

   return result;
}

} // namespace papilo

#endif
