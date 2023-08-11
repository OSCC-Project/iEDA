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

#ifndef _PAPILO_CORE_PROBLEM_HPP_
#define _PAPILO_CORE_PROBLEM_HPP_

#include "papilo/core/ConstraintMatrix.hpp"
#include "papilo/core/Objective.hpp"
#include "papilo/core/SingleRow.hpp"
#include "papilo/core/VariableDomains.hpp"
#include "papilo/io/Message.hpp"
#include "papilo/misc/MultiPrecision.hpp"
#include "papilo/misc/StableSum.hpp"
#include "papilo/misc/String.hpp"
#include "papilo/misc/Vec.hpp"
#include "papilo/misc/fmt.hpp"
#ifdef PAPILO_TBB
#include "papilo/misc/tbb.hpp"
#endif

namespace papilo
{

/// struct to hold counters for up an downlocks of a column
struct Locks
{
   int up;
   int down;

   template <typename Archive>
   void
   serialize( Archive& ar, const unsigned int version )
   {
      ar& up;
      ar& down;
   }
};

/// class representing the problem consisting of the constraint matrix, the left
/// and right hand side values, the variable domains, the column bounds,
/// column integrality restrictions, and the objective function
template <typename REAL>
class Problem
{
 public:
   /// set objective function
   void
   setObjective( Vec<REAL> coefficients, REAL offset = 0.0 )
   {
      objective = Objective<REAL>{ std::move( coefficients ), offset };
   }

   /// set objective function
   void
   setObjective( Objective<REAL>&& obj )
   {
      objective = obj;
   }

   /// set (transposed) constraint matrix
   void
   setConstraintMatrix( SparseStorage<REAL> cons_matrix, Vec<REAL> lhs_values,
                        Vec<REAL> rhs_values, Vec<RowFlags> row_flags,
                        bool transposed = false )
   {
      assert( lhs_values.size() == rhs_values.size() );
      assert( lhs_values.size() == row_flags.size() );
      assert( ( transposed ? cons_matrix.getNCols()
                           : cons_matrix.getNRows() ) == row_flags.size() );

      auto cons_matrix_other = cons_matrix.getTranspose();
      if( transposed )
         constraintMatrix = ConstraintMatrix<REAL>{
             std::move( cons_matrix_other ), std::move( cons_matrix ),
             std::move( lhs_values ), std::move( rhs_values ),
             std::move( row_flags ) };
      else
         constraintMatrix = ConstraintMatrix<REAL>{
             std::move( cons_matrix ), std::move( cons_matrix_other ),
             std::move( lhs_values ), std::move( rhs_values ),
             std::move( row_flags ) };
   }

   /// set constraint matrix
   void
   setConstraintMatrix( ConstraintMatrix<REAL>&& cons_matrix )
   {
      constraintMatrix = cons_matrix;
   }

   /// set domains of variables
   void
   setVariableDomains( VariableDomains<REAL>&& domains )
   {
      variableDomains = domains;

      nintegers = 0;
      ncontinuous = 0;

      for( ColFlags cf : variableDomains.flags )
      {
         if( cf.test( ColFlag::kIntegral ) )
            ++nintegers;
         else
            ++ncontinuous;
      }
   }

   /// set domains of variables
   void
   setVariableDomains( Vec<REAL> lower_bounds, Vec<REAL> upper_bounds,
                       Vec<ColFlags> col_flags )
   {
      variableDomains = VariableDomains<REAL>{ std::move( lower_bounds ),
                                               std::move( upper_bounds ),
                                               std::move( col_flags ) };
      nintegers = 0;
      ncontinuous = 0;

      for( ColFlags cf : variableDomains.flags )
      {
         if( cf.test( ColFlag::kIntegral ) )
            ++nintegers;
         else
            ++ncontinuous;
      }
   }

   /// returns number of active integral columns
   int
   getNumIntegralCols() const
   {
      return nintegers;
   }

   /// returns number of active integral columns
   int&
   getNumIntegralCols()
   {
      return nintegers;
   }

   /// returns number of active continuous columns
   int
   getNumContinuousCols() const
   {
      return ncontinuous;
   }

   /// returns number of active continuous columns
   int&
   getNumContinuousCols()
   {
      return ncontinuous;
   }

   /// set variable names
   void
   setVariableNames( Vec<String> var_names )
   {
      variableNames = std::move( var_names );
   }

   /// set constraint names
   void
   setConstraintNames( Vec<String> cons_names )
   {
      constraintNames = std::move( cons_names );
   }

   /// set problem name
   void
   setName( String name_ )
   {
      this->name = std::move( name_ );
   }

   /// get the problem matrix
   const ConstraintMatrix<REAL>&
   getConstraintMatrix() const
   {
      return constraintMatrix;
   }

   /// get the problem matrix
   ConstraintMatrix<REAL>&
   getConstraintMatrix()
   {
      return constraintMatrix;
   }

   /// get number of columns
   int
   getNCols() const
   {
      return constraintMatrix.getNCols();
   }

   /// get number of rows
   int
   getNRows() const
   {
      return constraintMatrix.getNRows();
   }

   /// get the objective function
   const Objective<REAL>&
   getObjective() const
   {
      return objective;
   }

   /// get the objective function
   Objective<REAL>&
   getObjective()
   {
      return objective;
   }

   /// get the variable domains
   const VariableDomains<REAL>&
   getVariableDomains() const
   {
      return variableDomains;
   }

   /// get the variable domains
   VariableDomains<REAL>&
   getVariableDomains()
   {
      return variableDomains;
   }

   const Vec<ColFlags>&
   getColFlags() const
   {
      return variableDomains.flags;
   }

   Vec<ColFlags>&
   getColFlags()
   {
      return variableDomains.flags;
   }

   const Vec<RowFlags>&
   getRowFlags() const
   {
      return constraintMatrix.getRowFlags();
   }

   Vec<RowFlags>&
   getRowFlags()
   {
      return constraintMatrix.getRowFlags();
   }

   /// get the variable names
   const Vec<String>&
   getVariableNames() const
   {
      return variableNames;
   }

   /// get the constraint names
   const Vec<String>&
   getConstraintNames() const
   {
      return constraintNames;
   }

   /// get the problem name
   const String&
   getName() const
   {
      return name;
   }

   /// get the (dense) vector of variable lower bounds
   const Vec<REAL>&
   getLowerBounds() const
   {
      return variableDomains.lower_bounds;
   }

   /// get the (dense) vector of variable lower bounds
   Vec<REAL>&
   getLowerBounds()
   {
      return variableDomains.lower_bounds;
   }

   /// get the (dense) vector of variable upper bounds
   const Vec<REAL>&
   getUpperBounds() const
   {
      return variableDomains.upper_bounds;
   }

   /// get the (dense) vector of variable upper bounds
   Vec<REAL>&
   getUpperBounds()
   {
      return variableDomains.upper_bounds;
   }

   /// get the (dense) vector of column sizes
   const Vec<int>&
   getColSizes() const
   {
      return constraintMatrix.getColSizes();
   }

   /// get the (dense) vector of column sizes
   Vec<int>&
   getColSizes()
   {
      return constraintMatrix.getColSizes();
   }

   /// get the (dense) vector of row sizes
   const Vec<int>&
   getRowSizes() const
   {
      return constraintMatrix.getRowSizes();
   }

   /// get the (dense) vector of row sizes
   Vec<int>&
   getRowSizes()
   {
      return constraintMatrix.getRowSizes();
   }

   /// substitute a variable in the objective using an equality constraint
   /// given by a row index
   void
   substituteVarInObj( const Num<REAL>& num, int col, int equalityrow );

   bool
   computeSolViolations( const Num<REAL>& num, const Vec<REAL>& sol,
                         REAL& boundviolation, REAL& rowviolation,
                         REAL& intviolation ) const
   {
      if( (int) sol.size() != getNCols() )
         return false;

      boundviolation = 0;
      intviolation = 0;

      for( int i = 0; i != getNCols(); ++i )
      {
         if( !variableDomains.flags[i].test( ColFlag::kLbInf ) &&
             sol[i] < variableDomains.lower_bounds[i] )
         {
            REAL thisviol = variableDomains.lower_bounds[i] - sol[i];

            if( !num.isFeasZero( thisviol ) )
               Message::debug( this,
                               "lower bound {} of column {} with solution "
                               "value {} is violated by {}\n",
                               double( variableDomains.lower_bounds[i] ), i,
                               double( sol[i] ), double( thisviol ) );

            boundviolation = num.max( boundviolation, thisviol );
         }

         if( !variableDomains.flags[i].test( ColFlag::kUbInf ) &&
             sol[i] > variableDomains.upper_bounds[i] )
         {
            REAL thisviol = sol[i] - variableDomains.upper_bounds[i];

            if( !num.isFeasZero( thisviol ) )
               Message::debug( this,
                               "upper bound {} of column {} with solution "
                               "value {} is violated by {}\n",
                               double( variableDomains.upper_bounds[i] ), i,
                               double( sol[i] ), double( thisviol ) );

            boundviolation = num.max( boundviolation, thisviol );
         }

         if( variableDomains.flags[i].test( ColFlag::kIntegral ) )
         {
            REAL thisviol = abs( num.round( sol[i] ) - sol[i] );

            if( !num.isFeasZero( thisviol ) )
               Message::debug( this,
                               "integrality of column {} with solution value "
                               "{} is violated by {}\n",
                               i, double( sol[i] ), double( thisviol ) );

            intviolation = num.max( intviolation, thisviol );
         }
      }

      rowviolation = 0;

      const Vec<RowFlags>& rflags = getRowFlags();
      const Vec<REAL>& lhs = constraintMatrix.getLeftHandSides();
      const Vec<REAL>& rhs = constraintMatrix.getRightHandSides();

      for( int i = 0; i != getNRows(); ++i )
      {
         auto rowvec = constraintMatrix.getRowCoefficients( i );
         const REAL* vals = rowvec.getValues();
         const int* inds = rowvec.getIndices();

         StableSum<REAL> activitySum;
         for( int j = 0; j != rowvec.getLength(); ++j )
            activitySum.add( sol[inds[j]] * vals[j] );

         REAL activity = activitySum.get();

         if( !rflags[i].test( RowFlag::kRhsInf )
             && num.isFeasGT( activity, rhs[i] ) )
         {
            Message::debug( this,
                            "the activity {} of constraint {}  "
                            "{} is greater than the righthandside {}\n",
                            activity, i, rhs[i] );
            rowviolation = num.max( rowviolation, activity - rhs[i] );
         }

         if( !rflags[i].test( RowFlag::kLhsInf )
             && num.isFeasLT( activity, rhs[i] ) )
         {
            Message::debug( this,
                            "the activity {} of constraint {}  "
                            "{} is greater than the lefthandside {}\n",
                            activity, i, lhs[i] );
            rowviolation = num.max( rowviolation, lhs[i] - activity );
         }
      }

      return num.isFeasZero( boundviolation ) &&
             num.isFeasZero( intviolation ) && num.isFeasZero( rowviolation );
   }

   REAL
   computeSolObjective( const Vec<REAL>& sol ) const
   {
      assert( (int) sol.size() == getNCols() );

      StableSum<REAL> obj( objective.offset );
      for( int i = 0; i < getNCols(); ++i )
         obj.add( sol[i] * objective.coefficients[i] );

      return obj.get();
   }

   /// return const reference to vector of row activities
   const Vec<RowActivity<REAL>>&
   getRowActivities() const
   {
      return rowActivities;
   }

   /// return reference to vector of row activities
   Vec<RowActivity<REAL>>&
   getRowActivities()
   {
      return rowActivities;
   }

   /// returns a reference to the vector of locks of each column, the locks
   /// include the objective cutoff constraint
   Vec<Locks>&
   getLocks()
   {
      return locks;
   }

   std::pair<Vec<int>, Vec<int>>
   compress( bool full = false )
   {
      std::pair<Vec<int>, Vec<int>> mappings =
          constraintMatrix.compress( full );

      // update information about columns that is stored by index
#ifdef PAPILO_TBB
      tbb::parallel_invoke(
          [this, &mappings, full]() {
             compress_vector( mappings.second, objective.coefficients );
             if( full )
                objective.coefficients.shrink_to_fit();
          },
          [this, &mappings, full]() {
             variableDomains.compress( mappings.second, full );
          },
          [this, &mappings, full]() {
             // compress row activities
             // recomputeAllActivities();
             if( rowActivities.size() != 0 )
                compress_vector( mappings.first, rowActivities );

             if( full )
                rowActivities.shrink_to_fit();
          } );
#else
      compress_vector( mappings.second, objective.coefficients );
      variableDomains.compress( mappings.second, full );
      if( rowActivities.size() != 0 )
         compress_vector( mappings.first, rowActivities );
      if( full )
      {
         objective.coefficients.shrink_to_fit();
         rowActivities.shrink_to_fit();
      }
#endif

      // compress row activities
      return mappings;
   }

   /// sets the tolerance of the input format
   void
   setInputTolerance( REAL inputTolerance_ )
   {
      this->inputTolerance = std::move( inputTolerance_ );
   }

   void
   recomputeAllActivities()
   {
      rowActivities.resize( getNRows() );

      // loop through rows once, compute initial acitvities, detect trivial
      // redundancy
#ifdef PAPILO_TBB
      tbb::parallel_for(
          tbb::blocked_range<int>( 0, getNRows() ),
          [this]( const tbb::blocked_range<int>& r ) {
             for( int row = r.begin(); row < r.end(); ++row )
#else
      for( int row = 0; row < getNRows(); ++row )
#endif
             {
                auto rowvec = constraintMatrix.getRowCoefficients( row );
                rowActivities[row] = compute_row_activity(
                    rowvec.getValues(), rowvec.getIndices(), rowvec.getLength(),
                    variableDomains.lower_bounds, variableDomains.upper_bounds,
                    variableDomains.flags );
             }
#ifdef PAPILO_TBB
          } );
#endif
   }

   void
   recomputeLocks()
   {
      locks.resize( getNCols() );

      // loop through rows once, compute initial activities, detect trivial
      // redundancy
#ifdef PAPILO_TBB
      tbb::parallel_for(
          tbb::blocked_range<int>( 0, getNCols() ),
          [this]( const tbb::blocked_range<int>& c ) {
             for( int col = c.begin(); col != c.end(); ++col )
#else
      for( int col = 0; col < getNCols(); ++col )
#endif
             {
                auto colvec = constraintMatrix.getColumnCoefficients( col );

                const REAL* vals = colvec.getValues();
                const int* inds = colvec.getIndices();
                int len = colvec.getLength();
                const auto& rflags = getRowFlags();

                for( int i = 0; i != len; ++i )
                   count_locks( vals[i], rflags[inds[i]], locks[col].down,
                                locks[col].up );
             }
#ifdef PAPILO_TBB
          } );
#endif
   }

   std::pair<int, int>
   removeRedundantBounds( const Num<REAL>& num, Vec<ColFlags>& cflags,
                          Vec<RowActivity<REAL>>& activities ) const
   {
      const Vec<REAL>& lhs = constraintMatrix.getLeftHandSides();
      const Vec<REAL>& rhs = constraintMatrix.getRightHandSides();
      const Vec<int>& colsize = constraintMatrix.getColSizes();
      const Vec<RowFlags>& rflags = getRowFlags();

      const Vec<REAL>& lbs = getLowerBounds();
      const Vec<REAL>& ubs = getUpperBounds();
      int nremoved = 0;
      int nnewfreevars = 0;

      Vec<std::tuple<int, REAL, int>> colperm( getNCols() );

      for( int i = 0; i != getNCols(); ++i )
         colperm[i] = std::make_tuple(
             colsize[i],
             constraintMatrix.getColumnCoefficients( i ).getDynamism(), i );

      pdqsort( colperm.begin(), colperm.end() );

      for( const auto& tuple : colperm )
      {
         int col = std::get<2>( tuple );

         if( cflags[col].test( ColFlag::kInactive ) ||
             !cflags[col].test( ColFlag::kUnbounded ) )
            continue;

         auto colvec = constraintMatrix.getColumnCoefficients( col );
         const int* colrows = colvec.getIndices();
         const REAL* colvals = colvec.getValues();
         const int collen = colvec.getLength();

         int k = 0;

         ColFlags colf = cflags[col];

         while( ( !colf.test( ColFlag::kLbInf ) ||
                  !colf.test( ColFlag::kUbInf ) ) &&
                k != collen )
         {
            int row = colrows[k];

            if( rflags[row].test( RowFlag::kRedundant ) )
            {
               ++k;
               continue;
            }

            if( !colf.test( ColFlag::kLbInf ) &&
                row_implies_LB( num, lhs[row], rhs[row], rflags[row],
                                activities[row], colvals[k], lbs[col], ubs[col],
                                cflags[col] ) )
               colf.set( ColFlag::kLbInf );

            if( !colf.test( ColFlag::kUbInf ) &&
                row_implies_UB( num, lhs[row], rhs[row], rflags[row],
                                activities[row], colvals[k], lbs[col], ubs[col],
                                cflags[col] ) )
               colf.set( ColFlag::kUbInf );

            ++k;
         }

         if( colf.test( ColFlag::kLbInf ) && colf.test( ColFlag::kUbInf ) )
         {
            int oldnremoved = nremoved;
            if( !cflags[col].test( ColFlag::kLbInf ) )
            {
               update_activities_remove_finite_bound( colrows, colvals, collen,
                                                      BoundChange::kLower,
                                                      lbs[col], activities );
               cflags[col].set( ColFlag::kLbInf );
               ++nremoved;
            }

            if( !cflags[col].test( ColFlag::kUbInf ) )
            {
               update_activities_remove_finite_bound( colrows, colvals, collen,
                                                      BoundChange::kUpper,
                                                      ubs[col], activities );
               cflags[col].set( ColFlag::kUbInf );
               ++nremoved;
            }

            if( oldnremoved != nremoved )
               ++nnewfreevars;
         }
      }

      return std::make_pair( nremoved, nnewfreevars );
   }

   template <typename Archive>
   void
   serialize( Archive& ar, const unsigned int version )
   {
      ar& name;
      ar& inputTolerance;
      ar& objective;

      ar& constraintMatrix;
      ar& variableDomains;
      ar& ncontinuous;
      ar& nintegers;

      ar& variableNames;
      ar& constraintNames;
      ar& rowActivities;

      ar& locks;
   }

 private:
   String name;
   REAL inputTolerance{ 0 };
   Objective<REAL> objective;
   ConstraintMatrix<REAL> constraintMatrix;
   VariableDomains<REAL> variableDomains;
   int ncontinuous;
   int nintegers;

   Vec<String> variableNames;
   Vec<String> constraintNames;

   /// minimal and maximal row activities
   Vec<RowActivity<REAL>> rowActivities;

   /// up and down locks for each column
   Vec<Locks> locks;
};

template <typename REAL>
void
Problem<REAL>::substituteVarInObj( const Num<REAL>& num, int col, int row )
{
   auto& consMatrix = getConstraintMatrix();
   auto& objcoefficients = getObjective().coefficients;
   REAL freevarCoefInObj = objcoefficients[col];

   if( freevarCoefInObj == REAL{ 0 } )
      return;

   const auto equalityrow = consMatrix.getRowCoefficients( row );
   const int length = equalityrow.getLength();
   const REAL* values = equalityrow.getValues();
   const int* indices = equalityrow.getIndices();

   int consid = consMatrix.getSparseIndex( col, row );
   assert( consid >= 0 );
   assert( indices[consid] == col );
   REAL freevarCoefInCons = values[consid];

   REAL substscale = -freevarCoefInObj / freevarCoefInCons;

   objcoefficients[col] = REAL{ 0.0 };
   for( int j = 0; j < length; ++j )
   {
      if( indices[j] == col )
         continue;

      REAL newobjcoeff = objcoefficients[indices[j]] + values[j] * substscale;
      if( num.isZero( newobjcoeff ) )
         newobjcoeff = 0;

      objcoefficients[indices[j]] = newobjcoeff;
   }

   assert( consMatrix.getRowFlags()[row].test( RowFlag::kEquation ) &&
           !consMatrix.getRowFlags()[row].test( RowFlag::kRhsInf ) &&
           !consMatrix.getRowFlags()[row].test( RowFlag::kLhsInf ) &&
           consMatrix.getLeftHandSides()[row] ==
               consMatrix.getRightHandSides()[row] );
   getObjective().offset -= consMatrix.getLeftHandSides()[row] * substscale;
}

} // namespace papilo

#endif
