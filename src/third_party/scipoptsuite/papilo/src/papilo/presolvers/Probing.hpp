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

#ifndef _PAPILO_PRESOLVERS_PROBING_HPP_
#define _PAPILO_PRESOLVERS_PROBING_HPP_

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/ProbingView.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"
#include "papilo/core/SingleRow.hpp"
#include "papilo/misc/Array.hpp"
#include "papilo/misc/Hash.hpp"
#include "papilo/misc/Vec.hpp"
#include "papilo/misc/compress_vector.hpp"
#include "papilo/misc/fmt.hpp"
#ifdef PAPILO_TBB
#include "papilo/misc/tbb.hpp"
#endif
#include <atomic>
#include <boost/functional/hash.hpp>

namespace papilo
{

const static int DEFAULT_MAX_BADGE_SIZE = -1;

template <typename REAL>
class Probing : public PresolveMethod<REAL>
{
   Vec<int> nprobed;
   int maxinitialbadgesize = 1000;
   int minbadgesize = 10;
   int max_badge_size = DEFAULT_MAX_BADGE_SIZE;
   double mincontdomred = 0.3;

 public:
   Probing() : PresolveMethod<REAL>()
   {
      this->setName( "probing" );
      this->setTiming( PresolverTiming::kExhaustive );
      this->setType( PresolverType::kIntegralCols );
   }

   void
   compress( const Vec<int>& rowmap, const Vec<int>& colmap ) override
   {
      assert( colmap.size() == nprobed.size() );
      compress_vector( colmap, nprobed );
      Message::debug( this,
                      "compress was called, compressed nprobed vector from "
                      "size {} to size {}\n",
                      colmap.size(), nprobed.size() );
   }

   bool
   initialize( const Problem<REAL>& problem,
               const PresolveOptions& presolveOptions ) override
   {
      nprobed.clear();
      nprobed.resize( problem.getNCols(), 0 );

      Message::debug( this, "initialized nprobed vector to size {}\n",
                      nprobed.size() );

      return true;
   }

   void
   addPresolverParams( ParameterSet& paramSet ) override
   {
      paramSet.addParameter( "probing.maxinitialbadgesize",
                             "maximum number of probing candidates probed in "
                             "the first badge of candidates",
                             maxinitialbadgesize, 1 );

      paramSet.addParameter( "probing.minbadgesize",
                             "minimum number of probing candidates probed in "
                             "a single badge of candidates",
                             minbadgesize, 1 );

      paramSet.addParameter( "probing.maxbadgesize",
                             "maximal number of probing candidates probed in "
                             "a single badge of candidates",
                             max_badge_size, DEFAULT_MAX_BADGE_SIZE );

      paramSet.addParameter(
          "probing.mincontdomred",
          "minimum fraction of domain that needs to be reduced for continuous "
          "variables to accept a bound change in probing",
          mincontdomred, 0.0, 1.0 );
   }

   PresolveStatus
   execute( const Problem<REAL>& problem,
            const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
            Reductions<REAL>& reductions, const Timer& timer ) override;

   bool
   isBinaryVariable( REAL upper_bound, REAL lower_bound, int column_size,
                     const Flags<ColFlag>& colFlag ) const;

   void
   set_max_badge_size( int val);

};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class Probing<double>;
extern template class Probing<Quad>;
extern template class Probing<Rational>;
#endif

template <typename REAL>
PresolveStatus
Probing<REAL>::execute( const Problem<REAL>& problem,
                        const ProblemUpdate<REAL>& problemUpdate,
                        const Num<REAL>& num, Reductions<REAL>& reductions,
                        const Timer& timer)
{
   if( problem.getNumIntegralCols() == 0 )
      return PresolveStatus::kUnchanged;

   const auto& domains = problem.getVariableDomains();
   const Vec<REAL>& lower_bounds = domains.lower_bounds;
   const Vec<REAL>& upper_bounds = domains.upper_bounds;
   const Vec<ColFlags>& cflags = domains.flags;

   const auto& consMatrix = problem.getConstraintMatrix();
   const auto& lhs = consMatrix.getLeftHandSides();
   const auto& rhs = consMatrix.getRightHandSides();
   const Vec<RowFlags>& rowFlags = consMatrix.getRowFlags();
   const auto& activities = problem.getRowActivities();
   const int ncols = problem.getNCols();
   const Vec<int>& colsize = consMatrix.getColSizes();
   const auto& colperm = problemUpdate.getRandomColPerm();

   Vec<int> probing_cands;
   probing_cands.reserve( ncols );

   for( int i = 0; i != ncols; ++i )
   {
      if( isBinaryVariable( upper_bounds[i], lower_bounds[i], colsize[i],
                            cflags[i] ) )
         probing_cands.push_back( i );
   }

   if( probing_cands.empty() )
      return PresolveStatus::kUnchanged;

   Array<std::atomic_int> probing_scores( ncols );

   for( int i = 0; i != ncols; ++i )
      probing_scores[i].store( 0, std::memory_order_relaxed );

   if( nprobed.empty() )
   {
      nprobed.resize( size_t( ncols ), 0 );

      assert( static_cast<int>( nprobed.size() ) == ncols );
      assert( std::all_of( nprobed.begin(), nprobed.end(),
                           []( int n ) { return n == 0; } ) );
   }

#ifdef PAPILO_TBB
   tbb::parallel_for(
       tbb::blocked_range<int>( 0, problem.getNRows() ),
       [&]( const tbb::blocked_range<int>& r )
       {
          Vec<std::pair<REAL, int>> binary_variables_in_row;
          for( int row = r.begin(); row != r.end(); ++row )
#else
   Vec<std::pair<REAL, int>> binary_variables_in_row;
   for( int row = 0; row != problem.getNRows(); ++row )
#endif
          {
             if( consMatrix.isRowRedundant( row ) )
                continue;

             if( ( activities[row].ninfmin != 0 ||
                   rowFlags[row].test( RowFlag::kRhsInf ) ) &&
                 ( activities[row].ninfmax != 0 ||
                   rowFlags[row].test( RowFlag::kLhsInf ) ) )
                continue;

             auto rowvec = consMatrix.getRowCoefficients( row );
             const int* colinds = rowvec.getIndices();
             const REAL* rowvals = rowvec.getValues();
             const int rowlen = rowvec.getLength();

             binary_variables_in_row.reserve( rowlen );

             for( int i = 0; i != rowlen; ++i )
             {
                if( isBinaryVariable( upper_bounds[i], lower_bounds[i],
                                      colsize[i], cflags[i] ) )
                   binary_variables_in_row.emplace_back( rowvals[i],
                                                         colinds[i] );
             }

             const int nbinvarsrow =
                 static_cast<int>( binary_variables_in_row.size() );

             if( nbinvarsrow == 0 )
                continue;

             pdqsort( binary_variables_in_row.begin(),
                      binary_variables_in_row.end(),
                      []( const std::pair<REAL, int>& a,
                          const std::pair<REAL, int>& b ) {
                         return abs( a.first ) > abs( b.first );
                      } );

             for( int i = 0; i < nbinvarsrow; ++i )
             {
                // TODO: wouldn't be simpler: calculate minimplcoef, if greater
                // equals than abs, then discard
                int col = binary_variables_in_row[i].second;
                REAL abscoef = abs( binary_variables_in_row[i].first );
                REAL minimplcoef = abscoef;

                if( activities[row].ninfmin == 0 &&
                    !rowFlags[row].test( RowFlag::kRhsInf ) )
                   minimplcoef = std::min(
                       minimplcoef,
                       REAL( rhs[row] - activities[row].min - abscoef ) );

                if( activities[row].ninfmax == 0 &&
                    !rowFlags[row].test( RowFlag::kLhsInf ) )
                   minimplcoef =
                       std::min( minimplcoef, REAL( activities[row].max -
                                                    abscoef - lhs[row] ) );

                if( num.isFeasLE( abscoef, minimplcoef ) )
                   break;

                int nimplbins = 0;
                for( int j = i + 1; j != nbinvarsrow; ++j )
                {
                   if( num.isFeasGT( abs( binary_variables_in_row[j].first ),
                                     minimplcoef ) )
                      ++nimplbins;
                   else
                      break;
                }

                if( nimplbins != 0 )
                   probing_scores[col].fetch_add( nimplbins,
                                                  std::memory_order_relaxed );
                else
                   break;
             }

             binary_variables_in_row.clear();
          }

#ifdef PAPILO_TBB
       });
#endif

   pdqsort( probing_cands.begin(), probing_cands.end(),
            [this, &probing_scores, &colsize, &colperm]( int col1, int col2 ) {
               std::pair<double, double> s1;
               std::pair<double, double> s2;
               if( nprobed[col2] == 0 && probing_scores[col2] > 0 )
                  s2.first = probing_scores[col2] /
                             static_cast<double>( colsize[col2] );
               else
                  s2.first = 0;
               if( nprobed[col1] == 0 && probing_scores[col1] > 0 )
                  s1.first = probing_scores[col1] /
                             static_cast<double>( colsize[col1] );
               else
                  s1.first = 0;

               s1.second =
                   ( probing_scores[col1].load( std::memory_order_relaxed ) /
                     static_cast<double>( 1 + nprobed[col1] * colsize[col1] ) );
               s2.second =
                   ( probing_scores[col2].load( std::memory_order_relaxed ) /
                     static_cast<double>( 1 + nprobed[col2] * colsize[col2] ) );
               return s1 > s2 || ( s1 == s2 && colperm[col1] < colperm[col2] );
            } );

   const Vec<int>& rowsize = consMatrix.getRowSizes();

   int current_badge_start = 0;

   int64_t working_limit = consMatrix.getNnz() * 2;
   int initial_badge_limit = 0.1 * working_limit;

   const int nprobingcands = static_cast<int>( probing_cands.size() );
   int badge_size = 0;
   for( int i : probing_cands )
   {
      ++badge_size;

      if( badge_size == maxinitialbadgesize )
         break;

      initial_badge_limit -= colsize[i];
      if( initial_badge_limit <= 0 )
         break;

      auto colvec = consMatrix.getColumnCoefficients( i );
      const int* rowinds = colvec.getIndices();
      for( int k = 0; k != colvec.getLength(); ++k )
      {
         initial_badge_limit -= ( rowsize[rowinds[k]] - 1 );

         if( initial_badge_limit <= 0 )
            break;
      }

      if( initial_badge_limit <= 0 )
         break;
   }

   badge_size = std::max( std::min( nprobingcands, minbadgesize ), badge_size );

   int current_badge_end = current_badge_start + badge_size;
   int n_useless = 0;
   bool abort = false;

   HashMap<std::pair<int, int>, int, boost::hash<std::pair<int, int>>>
       substitutionsPos;
   Vec<ProbingSubstitution<REAL>> substitutions;
   Vec<int> boundPos( size_t( 2 * ncols ), 0 );
   Vec<ProbingBoundChg<REAL>> boundChanges;
   boundChanges.reserve( ncols );

   std::atomic_bool infeasible{ false };

   // use tbb combinable so that each thread will copy the activities and
   // bounds at most once
#ifdef PAPILO_TBB
   tbb::combinable<ProbingView<REAL>> probing_views( [this, &problem, &num]() {
      ProbingView<REAL> probingView( problem, num );
      probingView.setMinContDomRed( mincontdomred );
      return probingView;
   } );
#else
   ProbingView<REAL> probingView( problem, num );
   probingView.setMinContDomRed( mincontdomred );
#endif

   do
   {
      Message::debug( this, "probing candidates {} to {}\n",
                      current_badge_start,
                      current_badge_end );

      auto propagate_variables = [&]( int start, int end) {
#ifdef PAPILO_TBB
         tbb::parallel_for(
             tbb::blocked_range<int>( start, end ),
             [&]( const tbb::blocked_range<int>& r )
             {
                ProbingView<REAL>& probingView = probing_views.local();

                for( int i = r.begin(); i != r.end(); ++i )
#else
         for( int i = start; i < end; i++ )
#endif
                {
                   if( PresolveMethod<REAL>::is_time_exceeded(
                           timer, problemUpdate.getPresolveOptions().tlim ) )
                      break;
                   const int col = probing_cands[i];

                   assert( cflags[col].test( ColFlag::kIntegral ) &&
                               (lower_bounds[col] == 0 ||
                           upper_bounds[col] == 1 ));

                   if( infeasible.load( std::memory_order_relaxed ) )
                      break;

                   assert( !probingView.isInfeasible() );
                   probingView.setProbingColumn( col, true );
                   probingView.propagateDomains();
                   probingView.storeImplications();
                   probingView.reset();

                   if( infeasible.load( std::memory_order_relaxed ) )
                      break;

                   assert( !probingView.isInfeasible() );
                   probingView.setProbingColumn( col, false );
                   probingView.propagateDomains();

                   bool globalInfeasible = probingView.analyzeImplications();
                   probingView.reset();

                   ++nprobed[col];

                   if( globalInfeasible )
                   {
                      infeasible.store( true, std::memory_order_relaxed );
                      break;
                   }
                }
#ifdef PAPILO_TBB
             } );
#endif
      };

      propagate_variables( current_badge_start, current_badge_end );

      if( PresolveMethod<REAL>::is_time_exceeded(
              timer, problemUpdate.getPresolveOptions().tlim ) )
         return PresolveStatus::kUnchanged;

      if( infeasible.load( std::memory_order_relaxed ) )
         return PresolveStatus::kInfeasible;

      int64_t amountofwork = 0;
      int nfixings = 0;
      int nboundchgs = 0;
      int nsubstitutions = -substitutions.size();

#ifdef PAPILO_TBB
      probing_views.combine_each( [&]( ProbingView<REAL>& probingView ) {
#endif
         const auto& probingBoundChgs = probingView.getProbingBoundChanges();
         const auto& probingSubstitutions =
             probingView.getProbingSubstitutions();

         amountofwork += probingView.getAmountOfWork();

         for( const ProbingSubstitution<REAL>& subst : probingSubstitutions )
         {
            auto insres = substitutionsPos.emplace(
                std::make_pair( subst.col1, subst.col2 ),
                substitutions.size() );

            if( insres.second )
               substitutions.push_back( subst );
         }

         for( const ProbingBoundChg<REAL>& boundChg : probingBoundChgs )
         {
            if( boundPos[2 * boundChg.col + boundChg.upper] == 0 )
            {
               // found new bound change
               boundChanges.emplace_back( boundChg );
               boundPos[2 * boundChg.col + boundChg.upper] =
                   boundChanges.size();

               // check if column is now fixed
               if( ( boundChg.upper &&
                     boundChg.bound == lower_bounds[boundChg.col] ) ||
                   ( !boundChg.upper &&
                     boundChg.bound == upper_bounds[boundChg.col] ) )
                  ++nfixings;
               else
                  ++nboundchgs;
            }
            else
            {
               // already changed that bound
               ProbingBoundChg<REAL>& otherBoundChg =
                   boundChanges[boundPos[2 * boundChg.col + boundChg.upper] -
                                1];

               if( boundChg.upper && boundChg.bound < otherBoundChg.bound )
               {
                  // new upper bound change is tighter
                  otherBoundChg.bound = boundChg.bound;

                  // check if column is now fixed
                  if( boundChg.bound == lower_bounds[boundChg.col] )
                     ++nfixings;
               }
               else if( !boundChg.upper &&
                        boundChg.bound > otherBoundChg.bound )
               {
                  // new lower bound change is tighter
                  otherBoundChg.bound = boundChg.bound;

                  // check if column is now fixed
                  if( boundChg.bound == upper_bounds[boundChg.col] )
                     ++nfixings;
               }

               // do only count fixings in this case for two reasons:
               // 1) the number of bound changes depends on the order and
               // would make probing non deterministic 2) the boundchange was
               // already counted in previous rounds and will only be added
               // once
            }
         }

         probingView.clearResults();
#ifdef PAPILO_TBB
      } );
#endif
      nsubstitutions += substitutions.size();
      current_badge_start = current_badge_end;

      if( nfixings == 0 && nboundchgs == 0 && nsubstitutions == 0 )
         n_useless += amountofwork;
      else
         n_useless = 0;

      Message::debug(
          this,
          "probing found: {} fixings, {} substitutions, {} bound changes\n",
          nfixings, nsubstitutions, nboundchgs );

      int64_t extrawork =
          ( ( 0.1 * ( nfixings + nsubstitutions ) + 0.01 * nboundchgs ) *
            consMatrix.getNnz() );

      working_limit -= amountofwork;
      working_limit += extrawork;

      badge_size = static_cast<int>(
          ceil( badge_size * static_cast<double>( working_limit + extrawork ) /
                (double) amountofwork ) );
      badge_size = std::min( nprobingcands - current_badge_start, badge_size );
      if( max_badge_size > 0 )
         badge_size = std::min( max_badge_size, badge_size );
      current_badge_end = current_badge_start + badge_size;

      abort = n_useless >= consMatrix.getNnz() * 2 || working_limit < 0 ||
              current_badge_start == current_badge_end ||
              PresolveMethod<REAL>::is_time_exceeded(timer, problemUpdate.getPresolveOptions().tlim );
   } while( !abort );

   PresolveStatus result = PresolveStatus::kUnchanged;

   if( !boundChanges.empty() )
   {
      pdqsort(
          boundChanges.begin(), boundChanges.end(),
          []( const ProbingBoundChg<REAL>& a, const ProbingBoundChg<REAL>& b ) {
             return ( a.col << 1 | a.upper ) < ( b.col << 1 | b.upper );
          } );

      for( const ProbingBoundChg<REAL>& boundChg : boundChanges )
      {
         if( boundChg.upper )
            reductions.changeColUB( boundChg.col, boundChg.bound );
         else
            reductions.changeColLB( boundChg.col, boundChg.bound );
      }

      result = PresolveStatus::kReduced;
   }

   if( !substitutions.empty() )
   {
      pdqsort( substitutions.begin(), substitutions.end(),
               []( const ProbingSubstitution<REAL>& a,
                   const ProbingSubstitution<REAL>& b ) {
                  return std::make_pair( a.col1, a.col2 ) >
                         std::make_pair( b.col1, b.col2 );
               } );

      int lastsubstcol = -1;

      for( const ProbingSubstitution<REAL>& subst : substitutions )
      {
         if( subst.col1 == lastsubstcol )
            continue;

         lastsubstcol = subst.col1;

         reductions.replaceCol( subst.col1, subst.col2, subst.col2scale,
                                subst.col2const );
      }

      result = PresolveStatus::kReduced;
   }

   return result;
}

template <typename REAL>
bool
Probing<REAL>::isBinaryVariable( REAL upper_bound, REAL lower_bound,
                                 int column_size,
                                 const Flags<ColFlag>& colFlag ) const
{
   return !colFlag.test( ColFlag::kUnbounded ) &&
          colFlag.test( ColFlag::kIntegral ) && column_size > 0 &&
          lower_bound == 0 && upper_bound == 1;
}

template <typename REAL>
void
Probing<REAL>::set_max_badge_size( int val)
{
   max_badge_size = val;
}


} // namespace papilo

#endif
