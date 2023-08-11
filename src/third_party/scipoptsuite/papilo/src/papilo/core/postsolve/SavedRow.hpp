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

#ifndef _PAPILO_CORE_SAVED_ROW_HPP_
#define _PAPILO_CORE_SAVED_ROW_HPP_

#include "papilo/core/Solution.hpp"
#include "papilo/core/postsolve/BoundStorage.hpp"
#include "papilo/core/postsolve/ReductionType.hpp"
#include "papilo/misc/Num.hpp"
#include "papilo/misc/StableSum.hpp"
#include "papilo/misc/Vec.hpp"

namespace papilo
{

template <typename REAL>
class SavedRow
{
 private:
   Num<REAL> num;

   int row;

   Vec<int> col_coefficients;
   Vec<REAL> row_values;

   REAL slack;

   bool is_lhs_inf;
   REAL lhs;
   bool is_rhs_inf;
   REAL rhs;
   int length;

 public:
   SavedRow( const Num<REAL>& n, int i, const Vec<ReductionType>& types,
             const Vec<int>& start, const Vec<int>& indices,
             const Vec<REAL>& values, const Vec<REAL>& primal_solution )
   {
      int next_type = i - 1;
      int next_but_one_type = i - 2;

      int saved_row = start[next_type];
      if( types[next_type] == ReductionType::kSaveRow )
         saved_row = start[next_type];
      else if( types[next_but_one_type] == ReductionType::kSaveRow )
         saved_row = start[next_but_one_type];
      else
         assert( false );
      row = indices[saved_row];
      length = (int)values[saved_row];
      is_lhs_inf = indices[saved_row + 1] == 1;
      lhs = values[saved_row + 1];
      is_rhs_inf = indices[saved_row + 2] == 1;
      rhs = values[saved_row + 2];

      col_coefficients.resize( length );
      row_values.resize( length );

      StableSum<REAL> slack_of_row{};
      for( int j = 0; j < length; ++j )
      {
         int col_index = indices[saved_row + 3 + j];
         REAL val = values[saved_row + 3 + j];
         col_coefficients[j] = col_index;
         row_values[j] = val;
         slack_of_row.add( val * primal_solution[col_index] );
      }
      slack = slack_of_row.get();
   }

   bool
   is_on_lhs()
   {
      return !is_lhs_inf && num.isFeasEq( slack, lhs );
   }

   bool
   is_on_rhs()
   {
      return !is_rhs_inf && num.isFeasEq( slack, rhs );
   }

   int
   getRow()
   {
      return row;
   }

   int
   getLength()
   {
      return length;
   }

   REAL
   getCoeffOfCol( int col )
   {
      for( int j = 0; j < length; ++j )
      {
         if( col_coefficients[j] == col )
            return row_values[j];
      }
      assert( false );
      return 0;
   }

   int
   getCoeff( int index )
   {
      return col_coefficients[index];
   }

   REAL
   getValue( int index )
   {
      return row_values[index];
   }

   VarBasisStatus
   getVBS()
   {
      if( is_on_lhs() && is_on_rhs() )
         return VarBasisStatus::FIXED;
      else if( is_on_rhs() )
         return VarBasisStatus::ON_UPPER;
      else if( is_on_lhs() )
         return VarBasisStatus::ON_LOWER;
      else if( is_lhs_inf && is_rhs_inf && num.isZero( slack ) )
         return VarBasisStatus::ZERO;
      return VarBasisStatus::BASIC;
   }

   bool
   is_violated( const Vec<REAL>& primal,
                const BoundStorage<REAL>& stored_bounds )
   {
      for( int i = 0; i < length; i++ )
      {
         int index = col_coefficients[i];
         REAL sol = primal[index];
         bool is_lb_violated = !stored_bounds.col_infinity_lower[index] &&
                               num.isLT( sol, stored_bounds.col_lower[index] );
         bool is_ub_violated = !stored_bounds.col_infinity_upper[index] &&
                               num.isGT( sol, stored_bounds.col_upper[index] );
         if(is_lb_violated || is_ub_violated)
            return true;
      }
      bool rhs_violated = !is_rhs_inf && num.isFeasGT( slack, rhs );
      bool lhs_violated = !is_lhs_inf && num.isFeasLT( slack, lhs );
      return rhs_violated || lhs_violated;
   }
};

} // namespace papilo

#endif
