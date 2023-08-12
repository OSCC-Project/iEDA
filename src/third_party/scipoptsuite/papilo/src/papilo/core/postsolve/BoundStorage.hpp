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

#ifndef _PAPILO_CORE_STORED_HPP_
#define _PAPILO_CORE_STORED_HPP_

#include "papilo/misc/Num.hpp"
#include "papilo/core/Problem.hpp"

namespace papilo
{

template <typename REAL>
class BoundStorage
{
 private:
   Num<REAL> num;
   Vec<REAL> col_cost;
   Vec<REAL> row_lhs;
   Vec<REAL> row_rhs;
   Vec<int> row_infinity_lhs;
   Vec<int> row_infinity_rhs;

 public:
      Vec<int> col_infinity_lower;
      Vec<int> col_infinity_upper;
      Vec<REAL> col_lower;
      Vec<REAL> col_upper;

 public:
   BoundStorage( const Num<REAL>& n, int cols, int rows, bool is_primal_dual )
   {
      if( ! is_primal_dual )
         return;
      num = n;
      col_cost.assign( cols, 0 );
      col_lower.assign( cols, 0 );
      col_upper.assign( cols, 0 );
      row_lhs.assign( rows, 0 );
      row_rhs.assign( rows, 0 );
      col_infinity_upper.assign( cols, 1 );
      col_infinity_lower.assign( cols, 1 );
      row_infinity_rhs.assign( rows, 1 );
      row_infinity_lhs.assign( rows, 1 );
   }

   void
   set_bounds_of_variable( int col, bool lb_inf, bool ub_inf, REAL lb, REAL ub )
   {
      assert( lb_inf || ub_inf || lb <= ub );
      col_lower[col] = lb;
      col_upper[col] = ub;
      col_infinity_lower[col] = lb_inf;
      col_infinity_upper[col] = ub_inf;
   }

   void
   set_bound_of_variable( int col, bool is_lower, bool inf, REAL value )
   {
      if( is_lower )
      {
         col_lower[col] = value;
         col_infinity_lower[col] = inf;
      }
      else
      {
         col_upper[col] = value;
         col_infinity_upper[col] = inf;
      }
   }

   bool
   is_lower_and_upper_bound_infinity( int col )
   {
      return col_infinity_lower[col] && col_infinity_upper[col];
   }

   void
   set_bounds_of_row( int row, bool lhs_inf, bool rhs_inf, REAL lhs, REAL rhs )
   {
      assert( rhs_inf || lhs_inf || lhs <= rhs );
      row_lhs[row] = lhs;
      row_rhs[row] = rhs;
      row_infinity_lhs[row] = rhs_inf;
      row_infinity_rhs[row] = lhs_inf;
   }

   bool
   is_on_upper_bound( int col, REAL value )
   {
      return ! col_infinity_upper[col] && num.isEq( value, col_upper[col] );
   }

   bool
   is_on_lower_bound( int col, REAL value )
   {
      return ! col_infinity_lower[col] && num.isEq( value, col_lower[col] );
   }

   bool
   check_bounds( const Problem<REAL>& problem )
   {
      const Vec<ColFlags>& colFlags = problem.getColFlags();
      const Vec<REAL>& lowerBounds = problem.getLowerBounds();
      const Vec<REAL>& upperBounds = problem.getUpperBounds();

      for( int i = 0; i < problem.getNCols(); i++ )
      {
         if( colFlags[i].test( ColFlag::kInactive ) )
            continue;
         if( col_infinity_lower[i] != colFlags[i].test( ColFlag::kLbInf ) )
            return false;
         if( col_infinity_upper[i] != colFlags[i].test( ColFlag::kUbInf ) )
            return false;
         if( !num.isEq( col_upper[i], upperBounds[i] ) &&
             !colFlags[i].test( ColFlag::kUbInf ) )
            return false;
         if( !num.isEq( col_lower[i], lowerBounds[i] ) &&
             !colFlags[i].test( ColFlag::kLbInf ) )
            return false;
      }
      return true;
   }
};

} // namespace papilo

#endif
