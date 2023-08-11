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

#ifndef _PAPILO_CORE_SINGLE_ROW_HPP_
#define _PAPILO_CORE_SINGLE_ROW_HPP_

#include "papilo/core/RowFlags.hpp"
#include "papilo/core/VariableDomains.hpp"
#include "papilo/misc/Flags.hpp"
#include "papilo/misc/Num.hpp"
#include "papilo/misc/Vec.hpp"
#include <tuple>

namespace papilo
{

enum class BoundChange
{
   kLower,
   kUpper
};

enum class ActivityChange
{
   kMin,
   kMax
};

enum class RowStatus
{
   kInfeasible,
   kRedundant,
   kRedundantLhs,
   kRedundantRhs,
   kUnknown,
};

template <typename REAL>
struct RowActivity
{
   /// minimal activity of the row
   REAL min;

   /// maximal activity of the row
   REAL max;

   /// number of variables that contribute with an infinite bound to the minimal
   /// activity of this row
   int ninfmin;

   /// number of variables that contribute with an infinite bound to the maximal
   /// activity of this row
   int ninfmax;

   /// last presolving round where this activity changed
   int lastchange;

   bool
   repropagate( ActivityChange actChange, RowFlags rflags )
   {
      if( actChange == ActivityChange::kMin &&
          !rflags.test( RowFlag::kRhsInf ) && ninfmin <= 1 )
         return true;

      if( actChange == ActivityChange::kMax &&
          !rflags.test( RowFlag::kLhsInf ) && ninfmax <= 1 )
         return true;

      return false;
   }

   RowStatus
   checkStatus( const Num<REAL>& num, RowFlags rflags, const REAL& lhs,
                const REAL& rhs ) const
   {
      RowStatus status = RowStatus::kRedundant;

      if( !rflags.test( RowFlag::kLhsInf ) )
      {
         if( ninfmax == 0 && num.isFeasLT( max, lhs ) &&
             num.isSafeLT( max, lhs ) )
            return RowStatus::kInfeasible;

         if( ninfmin == 0 && num.isFeasGE( min, lhs ) )
            status = RowStatus::kRedundantLhs;
         else
            status = RowStatus::kUnknown;
      }

      if( !rflags.test( RowFlag::kRhsInf ) )
      {
         if( ninfmin == 0 && num.isFeasGT( min, rhs ) &&
             num.isSafeGT( min, rhs ) )
            return RowStatus::kInfeasible;

         if( ninfmax == 0 && num.isFeasLE( max, rhs ) )
         {
            if( status == RowStatus::kUnknown )
               status = RowStatus::kRedundantRhs;
            else
               status = RowStatus::kRedundant;
         }
         else if( status == RowStatus::kRedundant )
            status = RowStatus::kUnknown;
      }
      else if( status == RowStatus::kRedundantLhs )
         status = RowStatus::kRedundant;

      return status;
   }

   template <typename Archive>
   void
   serialize( Archive& ar, const unsigned int version )
   {
      ar& min;
      ar& max;
      ar& ninfmin;

      ar& ninfmax;
      ar& lastchange;
   }

   RowActivity() = default;
};

/// counts the locks for the given row entry
template <typename REAL>
void
count_locks( const REAL& val, RowFlags rflags, int& ndownlocks, int& nuplocks )
{
   assert( val != 0 );

   if( val < 0 )
   {
      if( !rflags.test( RowFlag::kLhsInf ) )
         ++nuplocks;

      if( !rflags.test( RowFlag::kRhsInf ) )
         ++ndownlocks;
   }
   else
   {
      if( !rflags.test( RowFlag::kLhsInf ) )
         ++ndownlocks;

      if( !rflags.test( RowFlag::kRhsInf ) )
         ++nuplocks;
   }
}

template <typename REAL>
RowActivity<REAL>
compute_row_activity( const REAL* rowvals, const int* colindices, int rowlen,
                      const Vec<REAL>& lower_bounds,
                      const Vec<REAL>& upper_bounds, const Vec<ColFlags>& flags,
                      int presolveround = -1 )
{
   RowActivity<REAL> activity;

   activity.min = 0.0;
   activity.max = 0.0;
   activity.ninfmin = 0;
   activity.ninfmax = 0;
   activity.lastchange = presolveround;

   for( int j = 0; j < rowlen; ++j )
   {
      int col = colindices[j];
      if( !flags[col].test( ColFlag::kUbUseless ) )
      {
         if( rowvals[j] < 0 )
            activity.min += rowvals[j] * upper_bounds[col];
         else
            activity.max += rowvals[j] * upper_bounds[col];
      }
      else
      {
         assert( flags[col].test( ColFlag::kUbUseless ) );
         if( rowvals[j] < 0 )
            ++activity.ninfmin;
         else
            ++activity.ninfmax;
      }

      if( !flags[col].test( ColFlag::kLbUseless ) )
      {
         if( rowvals[j] < 0 )
            activity.max += rowvals[j] * lower_bounds[col];
         else
            activity.min += rowvals[j] * lower_bounds[col];
      }
      else
      {
         assert( flags[col].test( ColFlag::kLbUseless ) );
         if( rowvals[j] < 0 )
            ++activity.ninfmax;
         else
            ++activity.ninfmin;
      }
   }

   return activity;
}

template <typename REAL>
REAL
compute_minimal_row_activity( const REAL* rowvals, const int* colindices, int rowlen,
                      const Vec<REAL>& lower_bounds,
                      const Vec<REAL>& upper_bounds, const Vec<ColFlags>& flags)
{
   REAL min = 0.0;

   for( int j = 0; j < rowlen; ++j )
   {
      int col = colindices[j];
      if( !flags[col].test( ColFlag::kUbUseless ) &&  rowvals[j] < 0 )
            min += rowvals[j] * upper_bounds[col];
      if( !flags[col].test( ColFlag::kLbUseless ) && rowvals[j] > 0 )
            min += rowvals[j] * lower_bounds[col];
   }
   return min ;
}

template <typename REAL>
REAL
compute_maximal_row_activity( const REAL* rowvals, const int* colindices, int rowlen,
                      const Vec<REAL>& lower_bounds,
                      const Vec<REAL>& upper_bounds, const Vec<ColFlags>& flags)
{
   REAL max = 0.0;

   for( int j = 0; j < rowlen; ++j )
   {
      int col = colindices[j];
      if( !flags[col].test( ColFlag::kUbUseless ) && rowvals[j] > 0 )
            max += rowvals[j] * upper_bounds[col];
      if( !flags[col].test( ColFlag::kLbUseless ) && rowvals[j] < 0 )
            max += rowvals[j] * lower_bounds[col];
   }

   return max;
}

/// update the vector of row activities after lower or upper bounds of a column
/// changed. The last argument must be callable with arguments (ActivityChange,
/// rowid, RowActivity) and is called to inform about row activities that
/// changed
template <typename REAL>
ActivityChange
update_activity_after_boundchange( const REAL& colval, BoundChange type,
                                   const REAL& oldbound, const REAL& newbound,
                                   bool oldbound_inf,
                                   RowActivity<REAL>& activity )
{
   assert( oldbound_inf ||
           ( type == BoundChange::kLower && newbound != oldbound ) ||
           ( type == BoundChange::kUpper && newbound != oldbound ) );

   if( type == BoundChange::kLower )
   {
      if( colval < REAL{ 0.0 } )
      {
         if( oldbound_inf )
         {
            assert( activity.ninfmax > 0 );
            --activity.ninfmax;

            activity.max += newbound * colval;
         }
         else
         {
            activity.max += ( newbound - oldbound ) * colval;
         }

         return ActivityChange::kMax;
      }
      else
      {
         if( oldbound_inf )
         {
            assert( activity.ninfmin > 0 );
            --activity.ninfmin;

            activity.min += newbound * colval;
         }
         else
         {
            activity.min += ( newbound - oldbound ) * colval;
         }

         return ActivityChange::kMin;
      }
   }
   else
   {
      if( colval < REAL{ 0.0 } )
      {
         if( oldbound_inf )
         {
            assert( activity.ninfmin > 0 );
            --activity.ninfmin;

            activity.min += newbound * colval;
         }
         else
         {
            activity.min += ( newbound - oldbound ) * colval;
         }

         return ActivityChange::kMin;
      }
      else
      {
         if( oldbound_inf )
         {
            assert( activity.ninfmax > 0 );
            --activity.ninfmax;

            activity.max += newbound * colval;
         }
         else
         {
            activity.max += ( newbound - oldbound ) * colval;
         }

         return ActivityChange::kMax;
      }
   }
}

/// update the vector of row activities after removing a finite lower or upper
/// bound of a column
template <typename REAL>
void
update_activities_remove_finite_bound( const int* colinds, const REAL* colvals,
                                       int collen, BoundChange type,
                                       const REAL& oldbound,
                                       Vec<RowActivity<REAL>>& activities )
{
   if( type == BoundChange::kLower )
   {
      for( int i = 0; i != collen; ++i )
      {
         const REAL& colval = colvals[i];
         RowActivity<REAL>& activity = activities[colinds[i]];

         if( colval < REAL{ 0.0 } )
         {
            activity.max -= oldbound * colval;
            ++activity.ninfmax;
         }
         else
         {
            activity.min -= oldbound * colval;
            ++activity.ninfmin;
         }
      }
   }
   else
   {
      for( int i = 0; i != collen; ++i )
      {
         const REAL& colval = colvals[i];
         RowActivity<REAL>& activity = activities[colinds[i]];

         if( colval < REAL{ 0.0 } )
         {
            activity.min -= oldbound * colval;
            ++activity.ninfmin;
         }
         else
         {
            activity.max -= oldbound * colval;
            ++activity.ninfmax;
         }
      }
   }
}

/// update the vector of row activities after lower or upper bounds of a column
/// changed. The last argument must be callable with arguments (ActivityChange,
/// rowid, RowActivity) and is called to inform about row activities that
/// changed
template <typename REAL, typename ACTIVITYCHANGE>
void
update_activities_after_boundchange( const REAL* colvals, const int* colrows,
                                     int collen, BoundChange type,
                                     REAL oldbound, REAL newbound,
                                     bool oldbound_inf,
                                     Vec<RowActivity<REAL>>& activities,
                                     ACTIVITYCHANGE&& activityChange,
                                     bool watchInfiniteActivities = false )
{
   assert( oldbound_inf ||
           ( type == BoundChange::kLower && newbound != oldbound ) ||
           ( type == BoundChange::kUpper && newbound != oldbound ) );

   for( int i = 0; i < collen; ++i )
   {
      RowActivity<REAL>& activity = activities[colrows[i]];

      ActivityChange actChange = update_activity_after_boundchange(
          colvals[i], type, oldbound, newbound, oldbound_inf, activity );

      if( actChange == ActivityChange::kMin &&
          ( activity.ninfmin == 0 || watchInfiniteActivities ) )
         activityChange( ActivityChange::kMin, colrows[i], activity );

      if( actChange == ActivityChange::kMax &&
          ( activity.ninfmax == 0 || watchInfiniteActivities ) )
         activityChange( ActivityChange::kMax, colrows[i], activity );
   }
}

/**
 * updates the row activity for a changed coefficient in the matrix.
 * In case that the difference between the old and new coefficient is large,
 * the activity is recalculated entirely to prevent numerical difficulties.
 * The last argument must be callable with arguments (ActivityChange,
 * RowActivity) and is called to inform about row activities that changed.
 * @tparam REAL
 * @tparam ACTIVITYCHANGE
 * @param collb
 * @param colub
 * @param cflags
 * @param oldcolcoef
 * @param newcolcoef
 * @param activity
 * @param rowLength
 * @param colindices
 * @param rowvals
 * @param domains
 * @param num
 * @param activityChange
 */
template <typename REAL, typename ACTIVITYCHANGE>
void
update_activity_after_coeffchange( REAL collb, REAL colub, ColFlags cflags,
                                   REAL oldcolcoef, REAL newcolcoef,
                                   RowActivity<REAL>& activity,
                                   int rowLength, const int* colindices,
                                   const REAL* rowvals,
                                   const VariableDomains<REAL>& domains,
                                   const Num<REAL> num,
                                   ACTIVITYCHANGE&& activityChange )
{
   assert( oldcolcoef != newcolcoef );

   if( oldcolcoef * newcolcoef <= 0.0 )
   { // the sign of the coefficient flipped, so the column bounds now contribute
     // to the opposite activity bound

      // remember old activity
      RowActivity<REAL> oldactivity = activity;

      if( oldcolcoef != 0.0 )
      { // if the old coefficient was not 0.0 we remove its contributions to the
        // minimum and maximum activity
         // remove old contributions of the lower bound
         if( cflags.test( ColFlag::kLbUseless ) )
         {
            if( oldcolcoef < 0.0 )
               --activity.ninfmax;
            else
               --activity.ninfmin;
         }
         else
         {
            if( oldcolcoef < 0.0 )
               activity.max -= oldcolcoef * collb;
            else
               activity.min -= oldcolcoef * collb;
         }

         // remove old contributions of the upper bound
         if( cflags.test( ColFlag::kUbUseless ) )
         {
            if( oldcolcoef < 0.0 )
               --activity.ninfmin;
            else
               --activity.ninfmax;
         }
         else
         {
            if( oldcolcoef < 0.0 )
               activity.min -= oldcolcoef * colub;
            else
               activity.max -= oldcolcoef * colub;
         }
      }

      if( newcolcoef != 0.0 )
      { // if the new coefficient is not 0.0 we add its contributions to the
        // minimum and maximum activity
         // add new contributions of the lower bound
         if( cflags.test( ColFlag::kLbUseless ) )
         {
            if( newcolcoef < 0.0 )
               ++activity.ninfmax;
            else
               ++activity.ninfmin;
         }
         else
         {
            if( newcolcoef < 0.0 )
               activity.max += newcolcoef * collb;
            else
               activity.min += newcolcoef * collb;
         }

         // addnewold contributions of the upper bound
         if( cflags.test( ColFlag::kUbUseless ) )
         {
            if( newcolcoef < 0.0 )
               ++activity.ninfmin;
            else
               ++activity.ninfmax;
         }
         else
         {
            if( newcolcoef < 0.0 )
               activity.min += newcolcoef * colub;
            else
               activity.max += newcolcoef * colub;
         }
      }

      if( ( oldactivity.ninfmin != 0 && activity.ninfmin == 0 ) ||
          ( oldactivity.ninfmin == 0 && activity.ninfmin == 0 &&
            oldactivity.min != activity.min ) )
         activityChange( ActivityChange::kMin, activity );

      if( ( oldactivity.ninfmax != 0 && activity.ninfmax == 0 ) ||
          ( oldactivity.ninfmax == 0 && activity.ninfmax == 0 &&
            oldactivity.max != activity.max ) )
         activityChange( ActivityChange::kMax, activity );
   }
   else
   { // the sign of the coefficient did not flip, so the column bounds still
     // contribute to the same activity bound
      bool isDifferenceHugeVal = num.isHugeVal( newcolcoef - oldcolcoef );
      if( !cflags.test( ColFlag::kLbUseless ) && collb != 0.0 )
      {
         if( newcolcoef < REAL{ 0.0 } )
         {
            if( isDifferenceHugeVal )
               activity.max = compute_maximal_row_activity(
                   rowvals, colindices, rowLength, domains.lower_bounds,
                   domains.upper_bounds, domains.flags );
            else
               activity.max += collb * ( newcolcoef - oldcolcoef );

            if( activity.ninfmax == 0 )
               activityChange( ActivityChange::kMax, activity );
         }
         else
         {
            if( isDifferenceHugeVal )
               activity.min = compute_minimal_row_activity(
                   rowvals, colindices, rowLength, domains.lower_bounds,
                   domains.upper_bounds, domains.flags );

            else
               activity.min += collb * ( newcolcoef - oldcolcoef );
            if( activity.ninfmin == 0 )
               activityChange( ActivityChange::kMin, activity );
         }
      }

      if( !cflags.test( ColFlag::kUbUseless ) && colub != 0.0 )
      {
         if( newcolcoef < REAL{ 0.0 } )
         {
            if( isDifferenceHugeVal )
               activity.min = compute_minimal_row_activity(
                   rowvals, colindices, rowLength, domains.lower_bounds,
                   domains.upper_bounds, domains.flags );
            else
               activity.min += colub * ( newcolcoef - oldcolcoef );
            if( activity.ninfmin == 0 )
               activityChange( ActivityChange::kMin, activity );
         }
         else
         {
            if( isDifferenceHugeVal )
               activity.max = compute_maximal_row_activity(
                   rowvals, colindices, rowLength, domains.lower_bounds,
                   domains.upper_bounds, domains.flags );
            else
               activity.max += colub * ( newcolcoef - oldcolcoef );
            if( activity.ninfmax == 0 )
               activityChange( ActivityChange::kMax, activity );
         }
      }
   }
}

/// propagate domains of variables using the given a row and its activity. The
/// last argument must be callable with arguments (BoundChange, colid, newbound, row)
/// and is called to inform about column bounds that changed.
template <typename REAL, typename BOUNDCHANGE>
void
propagate_row( int row, const REAL* rowvals, const int* colindices, int rowlen,
               const RowActivity<REAL>& activity, REAL lhs, REAL rhs,
               RowFlags rflags, const Vec<REAL>& lower_bounds,
               const Vec<REAL>& upper_bounds, const Vec<ColFlags>& domainFlags,
               BOUNDCHANGE&& boundchange )
{

   bool adj_rhs = false;
   if( activity.ninfmin == 1 && activity.ninfmax == 0 &&
       rflags.test( RowFlag::kRhsInf ) )
   {
      adj_rhs = true;
      rhs = activity.max;
   }

   if( ( !rflags.test( RowFlag::kRhsInf ) && activity.ninfmin <= 1 ) ||
       adj_rhs )
   {
      for( int j = 0; j < rowlen; ++j )
      {
         int col = colindices[j];
         REAL lb = lower_bounds[col];
         REAL ub = upper_bounds[col];
         REAL minresact = activity.min;
         REAL val = rowvals[j];

         if( val < REAL{ 0.0 } )
         {
            if( activity.ninfmin == 1 )
            {
               if( !domainFlags[col].test( ColFlag::kUbUseless ) )
                  continue;

               j = rowlen;
            }
            else
            {
               assert( !domainFlags[col].test( ColFlag::kUbUseless ) );
               minresact -= val * ub;
            }

            REAL newlb = ( rhs - minresact ) / val;
            if( domainFlags[col].test( ColFlag::kLbInf ) || newlb > lb )
               boundchange( BoundChange::kLower, col, newlb, row );
         }
         else
         {
            if( activity.ninfmin == 1 )
            {
               if( !domainFlags[col].test( ColFlag::kLbUseless ) )
                  continue;

               j = rowlen;
            }
            else
            {
               assert( !domainFlags[col].test( ColFlag::kLbUseless ) );
               minresact -= val * lb;
            }

            REAL newub = ( rhs - minresact ) / val;
            if( domainFlags[col].test( ColFlag::kUbInf ) || newub < ub )
               boundchange( BoundChange::kUpper, col, newub, row );
         }
      }
   }

   bool adj_lhs = false;
   if( activity.ninfmax == 1 && activity.ninfmin == 0 &&
       rflags.test( RowFlag::kLhsInf ) )
   {
      adj_lhs = true;
      lhs = activity.min;
   }

   if( ( !rflags.test( RowFlag::kLhsInf ) && activity.ninfmax <= 1 ) ||
       adj_lhs )
   {
      for( int j = 0; j < rowlen; ++j )
      {
         int col = colindices[j];
         REAL lb = lower_bounds[col];
         REAL ub = upper_bounds[col];
         REAL maxresact = activity.max;
         REAL val = rowvals[j];

         if( val < REAL{ 0.0 } )
         {
            if( activity.ninfmax == 1 )
            {
               if( !domainFlags[col].test( ColFlag::kLbUseless ) )
                  continue;

               j = rowlen;
            }
            else
            {
               assert( !domainFlags[col].test( ColFlag::kLbUseless ) );
               maxresact -= val * lb;
            }

            REAL newub = ( lhs - maxresact ) / val;
            if( domainFlags[col].test( ColFlag::kUbInf ) || newub < ub )
               boundchange( BoundChange::kUpper, col, newub, row );
         }
         else
         {
            if( activity.ninfmax == 1 )
            {
               if( !domainFlags[col].test( ColFlag::kUbUseless ) )
                  continue;

               j = rowlen;
            }
            else
            {
               assert( !domainFlags[col].test( ColFlag::kUbUseless ) );
               maxresact -= val * ub;
            }

            REAL newlb = ( lhs - maxresact ) / val;
            if( domainFlags[col].test( ColFlag::kLbInf ) || newlb > lb )
               boundchange( BoundChange::kLower, col, newlb, row );
         }
      }
   }
}

template <typename REAL>
bool
row_implies_LB( const Num<REAL>& num, REAL lhs, REAL rhs, RowFlags rflags,
                const RowActivity<REAL>& activity, REAL colcoef, REAL collb,
                REAL colub, ColFlags cflags )

{
   if( cflags.test( ColFlag::kLbInf ) )
      return true;

   REAL resact;
   REAL side;

   if( colcoef > 0.0 && !rflags.test( RowFlag::kLhsInf ) )
   {
      if( activity.ninfmax == 0 )
      {
         assert( !cflags.test( ColFlag::kUbUseless ) );
         resact = activity.max - colub * colcoef;
      }
      else if( activity.ninfmax == 1 && cflags.test( ColFlag::kUbUseless ) )
         resact = activity.max;
      else
         return false;

      side = lhs;
   }
   else if( colcoef < 0.0 && !rflags.test( RowFlag::kRhsInf ) )
   {
      if( activity.ninfmin == 0 )
      {
         assert( !cflags.test( ColFlag::kUbUseless ) );
         resact = activity.min - colub * colcoef;
      }
      else if( activity.ninfmin == 1 && cflags.test( ColFlag::kUbUseless ) )
         resact = activity.min;
      else
         return false;

      side = rhs;
   }
   else
      return false;

   return num.isFeasGE( ( side - resact ) / colcoef, collb );
}

template <typename REAL>
bool
row_implies_UB( const Num<REAL>& num, REAL lhs, REAL rhs, RowFlags rflags,
                const RowActivity<REAL>& activity, REAL colcoef, REAL collb,
                REAL colub, ColFlags cflags )
{
   if( cflags.test( ColFlag::kUbInf ) )
      return true;

   REAL resact;
   REAL side;

   if( colcoef > 0.0 && !rflags.test( RowFlag::kRhsInf ) )
   {
      if( activity.ninfmin == 0 )
      {
         assert( !cflags.test( ColFlag::kLbUseless ) );
         resact = activity.min - collb * colcoef;
      }
      else if( activity.ninfmin == 1 && cflags.test( ColFlag::kLbUseless ) )
         resact = activity.min;
      else
         return false;

      side = rhs;
   }
   else if( colcoef < 0.0 && !rflags.test( RowFlag::kLhsInf ) )
   {
      if( activity.ninfmax == 0 )
      {
         assert( !cflags.test( ColFlag::kLbUseless ) );
         resact = activity.max - collb * colcoef;
      }
      else if( activity.ninfmax == 1 && cflags.test( ColFlag::kLbUseless ) )
         resact = activity.max;
      else
         return false;

      side = lhs;
   }
   else
      return false;

   return num.isFeasLE( ( side - resact ) / colcoef, colub );
}

} // namespace papilo

#endif
