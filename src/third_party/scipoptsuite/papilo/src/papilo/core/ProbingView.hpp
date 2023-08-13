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

#ifndef _PAPILO_CORE_PROBING_VIEW_HPP_
#define _PAPILO_CORE_PROBING_VIEW_HPP_

#include "papilo/core/Problem.hpp"
#include "papilo/core/SingleRow.hpp"
#include "papilo/io/Message.hpp"
#include <memory>
#include "papilo/misc/MultiPrecision.hpp"
#include "papilo/misc/Vec.hpp"

namespace papilo
{

template <typename REAL>
struct ProbingBoundChg
{
   REAL bound;
   unsigned int col : 31;
   unsigned int upper : 1;

   ProbingBoundChg( bool upper_, int col_, REAL bound_ )
   {
      this->upper = upper_ ? 1 : 0;
      this->col = static_cast<unsigned int>( col_ );
      this->bound = bound_;
   }
};

template <typename REAL>
struct ProbingSubstitution
{
   REAL col2scale;
   REAL col2const;
   int col1;
   int col2;

   ProbingSubstitution( int col1_, REAL col2scale_, int col2_, REAL col2const_ )
       : col2scale( col2scale_ ), col2const( col2const_ ), col1( col1_ ),
         col2( col2_ )
   {
   }
};

template <typename REAL>
class ProbingView
{
 public:
   ProbingView( const Problem<REAL>& problem, const Num<REAL>& num );

   void
   setMinIntDomRed( const REAL& value )
   {
      this->minintdomred = value;
   }

   void
   setMinContDomRed( const REAL& value )
   {
      this->mincontdomred = value;
   }

   void
   reset();

   void
   setProbingColumn( int col, bool value )
   {
      // remember probing column and probed value
      probingCol = col;
      probingValue = value;

      // fix upper/lower bound of probed column
      if( value )
         changeLb( col, 1.0 );
      else
         changeUb( col, 0.0 );
   }

   void
   activityChanged( ActivityChange actchange, int rowid,
                    RowActivity<REAL>& activity );
   void
   changeLb( int col, REAL newlb );

   void
   changeUb( int col, REAL newub );

   void
   storeImplications();

   bool
   analyzeImplications();

   void
   propagateDomains();

   bool
   isInfeasible() const
   {
      return infeasible;
   }

   int
   getNumSubstitutions() const
   {
      return static_cast<int>( substitutions.size() );
   }

   Vec<ProbingBoundChg<REAL>>&
   getProbingBoundChanges()
   {
      return boundChanges;
   }

   Vec<ProbingSubstitution<REAL>>&
   getProbingSubstitutions()
   {
      return substitutions;
   }

   const Vec<ProbingBoundChg<REAL>>&
   getProbingBoundChanges() const
   {
      return boundChanges;
   }

   const Vec<ProbingSubstitution<REAL>>&
   getProbingSubstitutions() const
   {
      return substitutions;
   }

   int64_t
   getAmountOfWork() const
   {
      return amountofwork;
   }

   const Vec<REAL>&
   getProbingLowerBounds() const
   {
      return probing_lower_bounds;
   }

   const Vec<REAL>&
   getProbingUpperBounds() const
   {
      return probing_upper_bounds;
   }

   const Vec<ColFlags>&
   getProbingDomainFlags() const
   {
      return probing_domain_flags;
   }

   void
   clearResults()
   {
      amountofwork = 0;
      boundChanges.clear();
      substitutions.clear();
   }

 private:
   // reference to problem and numerics class
   const Problem<REAL>& problem;
   const Num<REAL>& num;
   REAL minintdomred;
   REAL mincontdomred;

   // datastructures used for probing
   Vec<int> changed_lbs;
   Vec<int> changed_ubs;
   Vec<int> changed_activities;
   Vec<REAL> probing_lower_bounds;
   Vec<REAL> probing_upper_bounds;
   Vec<ColFlags> probing_domain_flags;
   Vec<RowActivity<REAL>> probing_activities;

   Vec<int> prop_activities;
   Vec<int> next_prop_activities;

   bool infeasible;
   int round;
   int probingCol;
   bool probingValue;

   // datastructures for storing result of probing on one value
   Vec<ProbingBoundChg<REAL>> otherValueImplications;
   bool otherValueInfeasible;

   // results of probing and statistics
   Vec<ProbingBoundChg<REAL>> boundChanges;
   Vec<ProbingSubstitution<REAL>> substitutions;

   int64_t amountofwork;
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class ProbingView<double>;
extern template class ProbingView<Quad>;
extern template class ProbingView<Rational>;
#endif

template <typename REAL>
ProbingView<REAL>::ProbingView( const Problem<REAL>& problem_,
                                const Num<REAL>& num_ )
    : problem( problem_ ), num( num_ ),
      probing_lower_bounds( problem_.getLowerBounds() ),
      probing_upper_bounds( problem_.getUpperBounds() ),
      probing_domain_flags( problem_.getColFlags() ),
      probing_activities( problem_.getRowActivities() )
{
   round = -2;
   infeasible = false;
   amountofwork = 0;
   probingCol = -1;
   probingValue = false;
   otherValueInfeasible = false;
   minintdomred = num.getFeasTol() * 1000;
   mincontdomred = 0.3;
}

template <typename REAL>
void
ProbingView<REAL>::reset()
{
   const Vec<int>& rowsize = problem.getConstraintMatrix().getRowSizes();

   const auto& orig_lbs = problem.getLowerBounds();
   for( int i : changed_lbs )
   {
      if( i < 0 )
      {
         int c = -i - 1;
         assert( !probing_domain_flags[c].test( ColFlag::kLbUseless ) &&
                 problem.getColFlags()[c].test( ColFlag::kLbUseless ) );
         probing_domain_flags[c].set( ColFlag::kLbUseless );
#ifndef NDEBUG
         probing_lower_bounds[c] = orig_lbs[c];
#endif
      }
      else
         probing_lower_bounds[i] = orig_lbs[i];
   }
   changed_lbs.clear();

   const auto& orig_ubs = problem.getUpperBounds();
   for( int i : changed_ubs )
   {
      if( i < 0 )
      {
         int c = -i - 1;
         assert( !probing_domain_flags[c].test( ColFlag::kUbUseless ) &&
                 problem.getColFlags()[c].test( ColFlag::kUbUseless ) );
         probing_domain_flags[c].set( ColFlag::kUbUseless );
#ifndef NDEBUG
         probing_upper_bounds[c] = orig_ubs[c];
#endif
      }
      else
         probing_upper_bounds[i] = orig_ubs[i];
   }
   changed_ubs.clear();

   const auto& orig_activities = problem.getRowActivities();
   for( int i : changed_activities )
   {
      amountofwork += rowsize[i];
      probing_activities[i] = orig_activities[i];
   }
   changed_activities.clear();

   // reset should result in original domains and activities
   assert( std::equal( orig_lbs.begin(), orig_lbs.end(),
                       probing_lower_bounds.begin() ) );
   assert( std::equal( orig_ubs.begin(), orig_ubs.end(),
                       probing_upper_bounds.begin() ) );
   assert( std::equal(
       orig_activities.begin(), orig_activities.end(),
       probing_activities.begin(),
       []( const RowActivity<REAL>& a, const RowActivity<REAL>& b ) {
          return a.ninfmax == b.ninfmax && a.ninfmin == b.ninfmin &&
                 a.min == b.min && a.max == b.max &&
                 a.lastchange == b.lastchange;
       } ) );

   round = -2;
   prop_activities.clear();
   next_prop_activities.clear();
   infeasible = false;
   probingCol = -1;
}

template <typename REAL>
void
ProbingView<REAL>::activityChanged( ActivityChange actchange, int rowid,
                                    RowActivity<REAL>& activity )
{
   const auto& consMatrix = problem.getConstraintMatrix();
   const auto& lhs = consMatrix.getLeftHandSides();
   const auto& rhs = consMatrix.getRightHandSides();
   const auto& rflags = consMatrix.getRowFlags();

   // mark the lastchange fields with round values starting from -2
   // and counting backward By doing this we avoid that we need to
   // alter the lastchange field after copying the original activities
   // since they are always larger or equal to -1
   if( activity.lastchange > -2 )
      changed_activities.push_back(
          rowid ); // activity was changed for the first time

   if( activity.lastchange != round )
      next_prop_activities.push_back( rowid );

   activity.lastchange = round;

   // check if the updated activity is reliable or if it is zero relative to
   // the initial activity
   const RowActivity<REAL>& origactivity = problem.getRowActivities()[rowid];

   bool unreliable;

   if( actchange == ActivityChange::kMin )
      unreliable = ( activity.ninfmin <= 1 && activity.min != 0 &&
                     origactivity.min != 0 &&
                     num.isZero( activity.min / origactivity.min ) );
   else
      unreliable = ( activity.ninfmax <= 1 && activity.max != 0 &&
                     origactivity.max != 0 &&
                     num.isZero( activity.max / origactivity.max ) );

   if( unreliable )
   {
      auto rowvec = problem.getConstraintMatrix().getRowCoefficients( rowid );

      activity = compute_row_activity( rowvec.getValues(), rowvec.getIndices(),
                                       rowvec.getLength(), probing_lower_bounds,
                                       probing_upper_bounds,
                                       probing_domain_flags, round );
   }

   // check for infeasibility
   if( actchange == ActivityChange::kMin && activity.ninfmin == 0 &&
       !rflags[rowid].test( RowFlag::kRhsInf ) &&
       num.isFeasLT( rhs[rowid], activity.min ) &&
       num.isSafeLT( rhs[rowid], activity.min ) )
   {
      Message::debug( this,
                      "[{}:{}] probing on col {} with val {} is infeasible min "
                      "activity is {:.15}, right hand side is {:.15}, and "
                      "original max activity was {:.15}\n",
                      __FILE__, __LINE__, probingCol, probingValue,
                      double( activity.min ), double( rhs[rowid] ),
                      double( problem.getRowActivities()[rowid].min ) );
      infeasible = true;
   }

   if( actchange == ActivityChange::kMax && activity.ninfmax == 0 &&
       !rflags[rowid].test( RowFlag::kLhsInf ) &&
       num.isFeasGT( lhs[rowid], activity.max ) &&
       num.isSafeGT( lhs[rowid], activity.max ) )
   {
      Message::debug( this,
                      "[{}:{}] probing on col {} with val {} is infeasible max "
                      "activity is {:.15}, left hand side is {:.15}, and "
                      "original max activity was {:.15}\n",
                      __FILE__, __LINE__, probingCol, probingValue,
                      double( activity.max ), double( lhs[rowid] ),
                      double( problem.getRowActivities()[rowid].max ) );
      infeasible = true;
   }
}

template <typename REAL>
void
ProbingView<REAL>::changeLb( int col, REAL newlb )
{
   const auto& consMatrix = problem.getConstraintMatrix();
   auto colvec = consMatrix.getColumnCoefficients( col );
   const auto& orig_lbs = problem.getLowerBounds();

   // bound must be tighter than current domains
   bool lbinf = probing_domain_flags[col].test( ColFlag::kLbUseless );
   assert( lbinf || probing_lower_bounds[col] != newlb );

   Message::debug( this, "changing probing lower bound of col {} to {}\n", col,
                   double( newlb ) );

   if( lbinf )
   {
      // bound was not altered yet, store the negative (index + 1) to
      // indicate that the infinity flag was altered
      probing_domain_flags[col].unset( ColFlag::kLbUseless );
      changed_lbs.push_back( -col - 1 );
   }
   else if( probing_lower_bounds[col] == orig_lbs[col] &&
            !problem.getColFlags()[col].test( ColFlag::kLbUseless ) )
      // if bound was not altered yet remember it in the index vector
      changed_lbs.push_back( col );

   // change the bound in the domain vector
   REAL oldlb = probing_lower_bounds[col];
   probing_lower_bounds[col] = newlb;

   // update the probing activities by using the column view
   update_activities_after_boundchange(
       colvec.getValues(), colvec.getIndices(), colvec.getLength(),
       BoundChange::kLower, oldlb, newlb, lbinf, probing_activities,
       [this]( ActivityChange actChange, int rowid,
               RowActivity<REAL>& activity ) {
          activityChanged( actChange, rowid, activity );
       },
       true );
}

template <typename REAL>
void
ProbingView<REAL>::changeUb( int col, REAL newub )
{
   const auto& consMatrix = problem.getConstraintMatrix();
   auto colvec = consMatrix.getColumnCoefficients( col );
   const auto& orig_ubs = problem.getUpperBounds();

   // bound must be tighter than current domains
   bool ubinf = probing_domain_flags[col].test( ColFlag::kUbUseless );
   assert( ubinf || probing_upper_bounds[col] != newub );

   Message::debug( this, "changing probing upper bound of col {} to {}\n", col,
                   double( newub ) );

   if( ubinf )
   {
      // bound was not altered yet, store the negative (index + 1) to
      // indicate that the infinity flag was altered
      probing_domain_flags[col].unset( ColFlag::kUbUseless );
      changed_ubs.push_back( -col - 1 );
   }
   else if( probing_upper_bounds[col] == orig_ubs[col] &&
            !problem.getColFlags()[col].test( ColFlag::kUbUseless ) )
      // if bound was not altered yet remember it in the index vector
      changed_ubs.push_back( col );

   // change the bound in the domain vector
   REAL oldub = probing_upper_bounds[col];
   probing_upper_bounds[col] = newub;

   // update the probing activities by using the column view
   update_activities_after_boundchange(
       colvec.getValues(), colvec.getIndices(), colvec.getLength(),
       BoundChange::kUpper, oldub, newub, ubinf, probing_activities,
       [this]( ActivityChange actChange, int rowid,
               RowActivity<REAL>& activity ) {
          activityChanged( actChange, rowid, activity );
       },
       true );
}

template <typename REAL>
void
ProbingView<REAL>::storeImplications()
{
   otherValueInfeasible = isInfeasible();

   if( otherValueInfeasible )
      return;

   otherValueImplications.clear();
   otherValueImplications.reserve( changed_lbs.size() + changed_ubs.size() -
                                   1 );

   for( int c : changed_lbs )
   {
      int col = c < 0 ? -c - 1 : c;

      if( col == probingCol )
         continue;

      otherValueImplications.emplace_back(
          ProbingBoundChg<REAL>( false, col, probing_lower_bounds[col] ) );
   }

   for( int c : changed_ubs )
   {
      int col = c < 0 ? -c - 1 : c;

      if( col == probingCol )
         continue;

      otherValueImplications.emplace_back(
          ProbingBoundChg<REAL>( true, col, probing_upper_bounds[col] ) );
   }
}

template <typename REAL>
bool
ProbingView<REAL>::analyzeImplications()
{
   const auto& orig_ubs = problem.getUpperBounds();
   const auto& orig_lbs = problem.getLowerBounds();
   const Vec<ColFlags>& orig_domain_flags = problem.getColFlags();

   // check infeasibility
   if( otherValueInfeasible )
   {
      // both probing values are infeasible
      if( infeasible )
         return true;

      // only the other probing value is infeasible, so store all changed
      // bounds as bound changes note that this includes the fixing of the
      // probing column to the probed value
      boundChanges.reserve( boundChanges.size() + changed_lbs.size() +
                            changed_ubs.size() );

      for( int c : changed_lbs )
      {
         int col = c < 0 ? -c - 1 : c;

         assert( c >= 0 ||
                 ( !probing_domain_flags[col].test( ColFlag::kLbUseless ) &&
                   orig_domain_flags[col].test( ColFlag::kLbUseless ) ) );
         assert( c < 0 ||
                 ( !probing_domain_flags[col].test( ColFlag::kLbUseless ) &&
                   !orig_domain_flags[col].test( ColFlag::kLbUseless ) ) );
         assert( c < 0 || probing_lower_bounds[col] > orig_lbs[col] );

         boundChanges.emplace_back(
             ProbingBoundChg<REAL>( false, col, probing_lower_bounds[col] ) );
      }

      for( int c : changed_ubs )
      {
         int col = c < 0 ? -c - 1 : c;

         assert( c >= 0 ||
                 ( !probing_domain_flags[col].test( ColFlag::kUbUseless ) &&
                   orig_domain_flags[col].test( ColFlag::kUbUseless ) ) );
         assert( c < 0 ||
                 ( !probing_domain_flags[col].test( ColFlag::kUbUseless ) &&
                   !orig_domain_flags[col].test( ColFlag::kUbUseless ) ) );
         assert( c < 0 || probing_upper_bounds[col] < orig_ubs[col] );

         boundChanges.emplace_back(
             ProbingBoundChg<REAL>( true, col, probing_upper_bounds[col] ) );
      }

      return false;
   }

   boundChanges.reserve( boundChanges.size() + otherValueImplications.size() +
                         1 );

   if( infeasible )
   {
      if( probingValue )
      {
         // probing to 1 is infeasible, fix probing column to 0
         boundChanges.emplace_back(
             ProbingBoundChg<REAL>( true, probingCol, 0.0 ) );
      }
      else
      {
         // probing to 0 is infeasible, fix probing column to 1
         boundChanges.emplace_back(
             ProbingBoundChg<REAL>( false, probingCol, 1.0 ) );
      }
   }

   assert( !otherValueInfeasible );

   for( const ProbingBoundChg<REAL>& boundChg : otherValueImplications )
   {
      bool impliedFixing =
          ( boundChg.upper &&
            !orig_domain_flags[boundChg.col].test( ColFlag::kLbUseless ) &&
            orig_lbs[boundChg.col] == boundChg.bound ) ||
          ( !boundChg.upper &&
            !orig_domain_flags[boundChg.col].test( ColFlag::kUbUseless ) &&
            orig_ubs[boundChg.col] == boundChg.bound );

      // only this probing branch is infeasible, so add all implications of
      // the other value as bound change
      if( infeasible )
      {
         boundChanges.emplace_back( boundChg );

         continue;
      }

      bool fixed =
          ( !probing_domain_flags[boundChg.col].test( ColFlag::kUnbounded ) &&
            probing_lower_bounds[boundChg.col] ==
                probing_upper_bounds[boundChg.col] );

      if( impliedFixing && fixed &&
          !num.isFeasEq( probing_lower_bounds[boundChg.col], boundChg.bound ) )
      { // column is fixed to different values in both probing branches
         REAL zerofixval;
         REAL onefixval;

         // Determine the fixed values for probing with value 0 and 1
         if( probingValue )
         {
            zerofixval = boundChg.bound;
            onefixval = probing_lower_bounds[boundChg.col];
         }
         else
         {
            zerofixval = probing_lower_bounds[boundChg.col];
            onefixval = boundChg.bound;
         }

         int col1 = boundChg.col;
         int col2 = probingCol;
         REAL scale = onefixval - zerofixval;

         // in case both columns are binary, we keep the one whith the
         // smaller index
         if( col1 < col2 && abs( scale ) == 1 &&
             ( zerofixval == 1 || zerofixval == 0 ) &&
             probing_domain_flags[col1].test( ColFlag::kIntegral ) )
            std::swap( col1, col2 );

         // add the corresponding substitution.
         substitutions.emplace_back(
             ProbingSubstitution<REAL>( col1, scale, col2, zerofixval ) );
      }
      else if( boundChg.upper &&
               !probing_domain_flags[boundChg.col].test( ColFlag::kUbInf ) &&
               ( orig_domain_flags[boundChg.col].test( ColFlag::kUbInf ) ||
                 orig_ubs[boundChg.col] > probing_upper_bounds[boundChg.col] ) )
      {
         assert( orig_domain_flags[boundChg.col].test( ColFlag::kUbInf ) ||
                 orig_ubs[boundChg.col] > boundChg.bound );

         // upper bound is tightened in both probing branches
         boundChanges.emplace_back( ProbingBoundChg<REAL>(
             true, boundChg.col,
             num.max( boundChg.bound, probing_upper_bounds[boundChg.col] ) ) );
      }
      else if( !boundChg.upper &&
               !probing_domain_flags[boundChg.col].test( ColFlag::kLbInf ) &&
               ( orig_domain_flags[boundChg.col].test( ColFlag::kLbInf ) ||
                 orig_lbs[boundChg.col] < probing_lower_bounds[boundChg.col] ) )
      {
         assert( orig_domain_flags[boundChg.col].test( ColFlag::kLbInf ) ||
                 orig_lbs[boundChg.col] < boundChg.bound );

         // lower bound is tightened in both probing branches
         boundChanges.emplace_back( ProbingBoundChg<REAL>(
             false, boundChg.col,
             num.min( boundChg.bound, probing_lower_bounds[boundChg.col] ) ) );
      }
   }

   return false;
}

template <typename REAL>
void
ProbingView<REAL>::propagateDomains()
{
   const auto& consMatrix = problem.getConstraintMatrix();
   const auto& lhs = consMatrix.getLeftHandSides();
   const auto& rhs = consMatrix.getRightHandSides();
   const auto& rflags = consMatrix.getRowFlags();

   using std::swap;

   swap( prop_activities, next_prop_activities );
   next_prop_activities.clear();

   while( !prop_activities.empty() )
   {
      int curr_round = round--;

      Message::debug( this,
                      "starting probing propagation round {} on {} rows\n",
                      -curr_round - 2, prop_activities.size() );

      for( int candrow : prop_activities )
      {
         bool propagate = false;

         if( !rflags[candrow].test( RowFlag::kRhsInf ) &&
             probing_activities[candrow].ninfmin <= 1 )
            propagate = true;

         if( !rflags[candrow].test( RowFlag::kLhsInf ) &&
             probing_activities[candrow].ninfmax <= 1 )
            propagate = true;

         if( !propagate )
            continue;

         auto rowvec = consMatrix.getRowCoefficients( candrow );

         propagate_row(candrow,
             rowvec.getValues(), rowvec.getIndices(), rowvec.getLength(),
             probing_activities[candrow], lhs[candrow], rhs[candrow],
             rflags[candrow], probing_lower_bounds, probing_upper_bounds,
             probing_domain_flags,
             [this]( BoundChange bndChg, int colid, REAL newbound , int row ) {
                if( num.isHugeVal( newbound ) )
                   return;

                bool isint = probing_domain_flags[colid].test(
                    ColFlag::kIntegral, ColFlag::kImplInt );

                if( bndChg == BoundChange::kLower )
                {
                   if( isint )
                      newbound = num.feasCeil( newbound );

                   if( !probing_domain_flags[colid].test( ColFlag::kUbInf ) &&
                       newbound > probing_upper_bounds[colid] )
                   {
                      if( num.isFeasGT( newbound,
                                        probing_upper_bounds[colid] ) )
                      {
                         Message::debug( this,
                                         "[{}:{}] probing on col {} with "
                                         "val {} is infeasible\n",
                                         __FILE__, __LINE__, probingCol,
                                         probingValue );
                         infeasible = true;
                         return;
                      }

                      newbound = probing_upper_bounds[colid];
                   }

                   REAL delta = newbound - probing_lower_bounds[colid];
                   bool finiteDomain =
                       !probing_domain_flags[colid].test( ColFlag::kUbInf );

                   REAL mindomred = isint ? minintdomred : mincontdomred;

                   if( probing_domain_flags[colid].test(
                           ColFlag::kLbUseless ) ||
                       ( finiteDomain && delta > 0 &&
                         ( delta / ( probing_upper_bounds[colid] -
                                     probing_lower_bounds[colid] ) >=
                           mindomred ) ) )
                      changeLb( colid, newbound );
                }
                else
                {
                   assert( bndChg == BoundChange::kUpper );
                   if( isint )
                      newbound = num.feasFloor( newbound );

                   if( !probing_domain_flags[colid].test( ColFlag::kLbInf ) &&
                       newbound < probing_lower_bounds[colid] )
                   {
                      if( num.isFeasLT( newbound,
                                        probing_lower_bounds[colid] ) )
                      {
                         Message::debug( this,
                                         "[{}:{}] probing on col {} with "
                                         "val {} is infeasible\n",
                                         __FILE__, __LINE__, probingCol,
                                         probingValue );
                         infeasible = true;
                         return;
                      }

                      newbound = probing_lower_bounds[colid];
                   }

                   REAL delta = probing_upper_bounds[colid] - newbound;
                   bool finiteDomain =
                       !probing_domain_flags[colid].test( ColFlag::kLbInf );

                   REAL mindomred = isint ? minintdomred : mincontdomred;

                   if( probing_domain_flags[colid].test(
                           ColFlag::kUbUseless ) ||
                       ( finiteDomain && delta > 0 &&
                         ( delta / ( probing_upper_bounds[colid] -
                                     probing_lower_bounds[colid] ) >=
                           mindomred ) ) )
                      changeUb( colid, newbound );
                }
             } );
         if( infeasible )
            return;
      }

      swap( prop_activities, next_prop_activities );
      next_prop_activities.clear();
   }
}

} // namespace papilo

#endif
