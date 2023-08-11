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

#ifndef _PAPILO_CORE_POSTSOLVE_LISTENER_HPP_
#define _PAPILO_CORE_POSTSOLVE_LISTENER_HPP_

#include "papilo/core/Problem.hpp"
#include "papilo/core/postsolve/PostsolveType.hpp"
#include "papilo/core/postsolve/ReductionType.hpp"
#include "papilo/misc/MultiPrecision.hpp"
#include "papilo/misc/Num.hpp"
#include "papilo/misc/PrimalDualSolValidation.hpp"
#include "papilo/misc/StableSum.hpp"
#include "papilo/misc/Vec.hpp"
#include "papilo/misc/compress_vector.hpp"
#include "papilo/misc/fmt.hpp"
#ifdef PAPILO_TBB
#include "papilo/misc/tbb.hpp"
#endif
#include <fstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/tmpdir.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

namespace papilo
{

template <typename REAL>
class SparseVectorView;

struct IndexRange;

/// type to store necessary data for post solve
template <typename REAL>
class PostsolveStorage
{
 public:
   unsigned int nColsOriginal;
   unsigned int nRowsOriginal;

   /// mapping of reduced problems column indices to column indices in the
   /// original problem
   Vec<int> origcol_mapping;

   /// mapping of reduced problems row indices to row indices in the original
   /// problem
   Vec<int> origrow_mapping;

   // set to full for development of postsolve,
   // later will not be default value
   // PostsolveType postsolveType = PostsolveType::kFull;
   PostsolveType postsolveType = PostsolveType::kPrimal;

   //types containes the Reductiontypes
   Vec<ReductionType> types;

   // indices/values can be considered as Vec<Vec<int/REAL>>
   // To reduce the number of vectors and hence speedup there is only one vector and
   // start indicates by saving the start index where the information of the Reduction type begins
   Vec<int> indices;
   Vec<REAL> values;

   // indices where the information of the i-th reduction inside start/values starts
   // information go from [start[i], start [i+1])
   Vec<int> start;

   Problem<REAL> problem;

   PresolveOptions presolveOptions;

   Num<REAL> num;

   PostsolveStorage() = default;

   PostsolveStorage( int nrows, int ncols )
   {
      origrow_mapping.reserve( nrows );
      origrow_mapping.reserve( ncols );

      for( int i = 0; i < nrows; ++i )
         origrow_mapping.push_back( i );

      for( int i = 0; i < ncols; ++i )
         origcol_mapping.push_back( i );

      nColsOriginal = ncols;
      nRowsOriginal = nrows;

      start.push_back( 0 );
   }

   PostsolveStorage( const Problem<REAL>& _problem, const Num<REAL>& _num, const PresolveOptions _options )
       : problem( _problem ), presolveOptions(_options), num( _num )
   {
      nRowsOriginal = _problem.getNRows();
      nColsOriginal = _problem.getNCols();

      origrow_mapping.reserve( nRowsOriginal );
      origrow_mapping.reserve( nColsOriginal );

      for( unsigned int i = 0; i < nRowsOriginal; ++i )
         origrow_mapping.push_back( (int) i );

      for( unsigned int i = 0; i < nColsOriginal; ++i )
         origcol_mapping.push_back( (int) i );

      start.push_back( 0 );

      // release excess storage in original problem copy
      this->problem.compress( true );
   }

   void
   storeRedundantRow( int row );

   void
   storeVarBoundChange( bool isLowerBound, int col, REAL oldBound,
                         bool was_infinity, REAL newBound );

   void
   storeRowBoundChange( bool isLhs, int row, REAL newBound, bool is_infinity, REAL oldBound, bool was_infinity );

   void
   storeRowBoundChangeForcedByRow( bool isLhs, int row, REAL newBound, bool isInfinity );

   void
   storeReasonForRowBoundChangeForcedByRow( int remained_row, int deleted_row, REAL factor);

   void
   storeReducedBoundsAndCost( const Vec<REAL>& col_lb, const Vec<REAL>& col_ub,
                               const Vec<REAL>& row_lhs, const Vec<REAL>& row_rhs,
                               const Vec<REAL>& coefficients,
                               const Vec<RowFlags>& row_flags,
                               const Vec<ColFlags>& col_flags );

   void
   storeFixedCol( int col, REAL val,
                   const SparseVectorView<REAL>& colvec,
                   const Vec<REAL>& cost );

   void
   storeCoefficientChange(int row, int col, REAL new_value);

   void
   storeDualValue( bool is_column_dual, int index, REAL value );

   void
   storeSavedRow( int row, const SparseVectorView<REAL>& coefficients,
                   REAL lhs, REAL rhs, const RowFlags& flags );

   void
   storeFixedInfCol( int col, REAL val, REAL bound,
                      const Problem<REAL>& currentProblem );

   void
   storeSubstitution( int col, int row, const Problem<REAL>& currentProblem );

   void
   storeSubstitution( int col, SparseVectorView<REAL> equalityLHS,
                       REAL equalityRHS );

   /// col1 = col2scale * col2 and are merged into a new column y = col2 +
   /// col2scale * col1 which takes over the index of col2
   void
   storeParallelCols( int col1, bool col1integral, bool col1lbinf,
                       REAL& col1lb, bool col1ubinf, REAL& col1ub,
                       int col2, bool col2integral, bool col2lbinf,
                       REAL& col2lb, bool col2ubinf, REAL& col2ub,
                       REAL& col2scale );

   void
   compress( const Vec<int>& rowmapping, const Vec<int>& colmapping,
             bool full = false )
   {
#ifdef PAPILO_TBB
      tbb::parallel_invoke(
          [this, &colmapping, full]() {
             compress_vector( colmapping, origcol_mapping );
             if( full )
                origcol_mapping.shrink_to_fit();
          },
          [this, &rowmapping, full]() {
             // update information about rows that is stored by index
             compress_vector( rowmapping, origrow_mapping );
             if( full )
                origrow_mapping.shrink_to_fit();
          } );
#else
      compress_vector( colmapping, origcol_mapping );
      compress_vector( rowmapping, origrow_mapping );
      if( full )
      {
         origrow_mapping.shrink_to_fit();
         origcol_mapping.shrink_to_fit();
      }
#endif
   }

   template <typename Archive>
   void
   serialize( Archive& ar, const unsigned int version )
   {
      ar& nColsOriginal;
      ar& nRowsOriginal;
      ar& origcol_mapping;
      ar& origrow_mapping;
      ar& postsolveType;
      ar& types;
      ar& indices;
      ar& values;
      ar& start;

      ar& problem;

      ar& num;
   }


   const Problem<REAL>&
   getOriginalProblem() const
   {
      return problem;
   }

   const Num<REAL>&
   getNum() const
   {
      return num;
   }

 private:
   void
   finishStorage()
   {
      assert( types.size() == start.size() );
      assert( values.size() == indices.size() );
      start.push_back( values.size() );
   }

   void
   push_back_row( int row, const Problem<REAL>& currentProblem );

   void
   push_back_col( int col, const Problem<REAL>& currentProblem );

};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class PostsolveStorage<double>;
extern template class PostsolveStorage<Quad>;
extern template class PostsolveStorage<Rational>;
#endif

template <typename REAL>
void
PostsolveStorage<REAL>::push_back_row( int row,
                                        const Problem<REAL>& currentProblem )
{
   const auto& coefficients =
       currentProblem.getConstraintMatrix().getRowCoefficients( row );
   REAL lhs = currentProblem.getConstraintMatrix().getLeftHandSides()[row];
   REAL rhs = currentProblem.getConstraintMatrix().getRightHandSides()[row];
   const auto& flags = currentProblem.getConstraintMatrix().getRowFlags()[row];

   const REAL* coefs = coefficients.getValues();
   const int* columns = coefficients.getIndices();
   const int length = coefficients.getLength();

   indices.push_back( origrow_mapping[row] );
   values.push_back( (double)length );

   // LB
   if( flags.test( RowFlag::kLhsInf ) )
      indices.push_back( 1 );
   else
      indices.push_back( 0 );
   values.push_back( lhs );

   // UB
   if( flags.test( RowFlag::kRhsInf ) )
      indices.push_back( 1 );
   else
      indices.push_back( 0 );
   values.push_back( rhs );

   for( int i = 0; i < length; ++i )
   {
      indices.push_back( origcol_mapping[columns[i]] );
      values.push_back( coefs[i] );
   }
}

template <typename REAL>
void
PostsolveStorage<REAL>::push_back_col( int col,
                                        const Problem<REAL>& currentProblem )
{
   const auto& coefficients =
       currentProblem.getConstraintMatrix().getColumnCoefficients( col );
   ColFlags flags = currentProblem.getColFlags()[col];
   REAL obj = currentProblem.getObjective().coefficients[col];

   const REAL* coefs = coefficients.getValues();
   const int* row_indices = coefficients.getIndices();
   const int length = coefficients.getLength();

   indices.push_back( origcol_mapping[col] );
   values.push_back( (double)length );

   indices.push_back( 0 );
   values.push_back( obj );

   // UB
   if( flags.test( ColFlag::kUbInf ) )
      indices.push_back( 1 );
   else
      indices.push_back( 0 );
   values.push_back( currentProblem.getUpperBounds()[col] );

   // LB
   if( flags.test( ColFlag::kLbInf ) )
      indices.push_back( 1 );
   else
      indices.push_back( 0 );
   values.push_back( currentProblem.getLowerBounds()[col] );

   for( int i = 0; i < length; ++i )
   {
      indices.push_back( origrow_mapping[row_indices[i]] );
      values.push_back( coefs[i] );
   }
}

template <typename REAL>
void
PostsolveStorage<REAL>::storeRedundantRow( int row )
{
   if( postsolveType == PostsolveType::kPrimal )
      return;

   types.push_back( ReductionType::kRedundantRow );
   indices.push_back(origrow_mapping[row] );
   values.push_back( 0 );

   finishStorage();
}


template <typename REAL>
void
PostsolveStorage<REAL>::storeVarBoundChange( bool isLowerBound,
                                               int col,
                                               REAL oldBound,
                                               bool was_infinity,
                                               REAL newBound )
{
   if( postsolveType == PostsolveType::kPrimal )
      return;
   types.push_back( ReductionType::kVarBoundChange );
   if( isLowerBound )
      indices.push_back( 1 );
   else
      indices.push_back( 0 );
   values.push_back( 0 );

   int c = origcol_mapping[col];
   indices.push_back( c );
   values.push_back( newBound );

   indices.push_back( was_infinity );
   values.push_back( oldBound );

   finishStorage();
}


template <typename REAL>
void
PostsolveStorage<REAL>::storeRowBoundChange( bool isLhs, int row, REAL newBound, bool is_infinity, REAL oldBound, bool was_infinity ){
   if( postsolveType == PostsolveType::kPrimal )
      return;
   types.push_back( ReductionType::kRowBoundChange );
   if( isLhs )
      indices.push_back( 1 );
   else
      indices.push_back( 0 );
   values.push_back( origrow_mapping[row] );
   indices.push_back( is_infinity );
   values.push_back( newBound );
   indices.push_back( was_infinity );
   values.push_back( oldBound );

   finishStorage();
}

template <typename REAL>
void
PostsolveStorage<REAL>::storeRowBoundChangeForcedByRow(  bool isLhs, int row, REAL newBound, bool isInfinity ){
   if( postsolveType == PostsolveType::kPrimal )
      return;
   types.push_back( ReductionType::kRowBoundChangeForcedByRow );
   if( isLhs )
      indices.push_back( 1 );
   else
      indices.push_back( 0 );
   values.push_back( origrow_mapping[row] );
   indices.push_back( isInfinity );
   values.push_back( newBound );

   finishStorage();
}

template <typename REAL>
void
PostsolveStorage<REAL>::storeReasonForRowBoundChangeForcedByRow( int remained_row, int deleted_row, REAL factor){
   if( postsolveType == PostsolveType::kPrimal )
      return;
   types.push_back( ReductionType::kReasonForRowBoundChangeForcedByRow );
   indices.push_back( origrow_mapping[remained_row] );
   values.push_back( factor );
   indices.push_back( origrow_mapping[deleted_row] );
   values.push_back( 0 );

   finishStorage();
}


template <typename REAL>
void
PostsolveStorage<REAL>::storeFixedCol( int col, REAL val,
                                 const SparseVectorView<REAL>& colvec,
                                 const Vec<REAL>& cost )
{
   types.push_back( ReductionType::kFixedCol );
   indices.push_back( origcol_mapping[col] );
   values.push_back( val );

   if( postsolveType == PostsolveType::kFull )
   {
      const int length = colvec.getLength();
      indices.push_back( length );
      values.push_back( cost[col] );

      const REAL* vals = colvec.getValues();
      const int* inds = colvec.getIndices();

      for( int j = 0; j < length; j++ )
      {
         indices.push_back( origrow_mapping[inds[j]] );
         values.push_back( vals[j] );
      }
   }

   finishStorage();
}

template <typename REAL>
void
PostsolveStorage<REAL>::storeFixedInfCol(
    int col, REAL val, REAL bound, const Problem<REAL>& currentProblem )
{
   types.push_back( ReductionType::kFixedInfCol );
   indices.push_back( origcol_mapping[col] );
   values.push_back( val );

   const auto& coefficients =
       currentProblem.getConstraintMatrix().getColumnCoefficients( col );
   const int* row_indices = coefficients.getIndices();

   indices.push_back( coefficients.getLength() );
   values.push_back( bound );

   for( int i = 0; i < coefficients.getLength(); i++ )
      push_back_row( row_indices[i], currentProblem );

   finishStorage();
}

template <typename REAL>
void
PostsolveStorage<REAL>::storeCoefficientChange( int row, int col,
                                                  REAL new_value )
{
   if( postsolveType == PostsolveType::kPrimal )
      return;
   types.push_back( ReductionType::kCoefficientChange );
   indices.push_back( origrow_mapping[row] );
   indices.push_back( origcol_mapping[col] );
   values.push_back( new_value );
   values.push_back( 0 );
   finishStorage();

}

template <typename REAL>
void
PostsolveStorage<REAL>::storeDualValue( bool is_column_dual, int index, REAL value )
{
   if( postsolveType == PostsolveType::kPrimal )
      return;
   if( is_column_dual )
      types.push_back( ReductionType::kColumnDualValue );
   else
      types.push_back( ReductionType::kRowDualValue );

   indices.push_back( index );
   values.push_back( value );
   finishStorage();
}

 template <typename REAL>
 void
PostsolveStorage<REAL>::storeSavedRow( int row,
                                  const SparseVectorView<REAL>& coefficients,
                                  REAL lhs, REAL rhs, const RowFlags& flags )
 {
    if( postsolveType == PostsolveType::kPrimal )
       return;
    const REAL* coefs = coefficients.getValues();
    const int* columns = coefficients.getIndices();
    const int length = coefficients.getLength();

    types.push_back( ReductionType::kSaveRow );
    indices.push_back( origrow_mapping[row] );
    values.push_back( (double)length );

    // LB
    if( flags.test( RowFlag::kLhsInf ) )
       indices.push_back( 1 );
    else
       indices.push_back( 0 );
    values.push_back( lhs );

    // UB
    if( flags.test( RowFlag::kRhsInf ) )
       indices.push_back( 1 );
    else
       indices.push_back( 0 );
    values.push_back( rhs );

    for( int i = 0; i < length; ++i )
    {
       indices.push_back( origcol_mapping[columns[i]] );
       values.push_back( coefs[i] );
    }

    finishStorage();
 }

template <typename REAL>
void
 PostsolveStorage<REAL>::storeSubstitution( int col, int row,
                                             const Problem<REAL>& currentProblem )
{
   types.push_back( ReductionType::kSubstitutedColWithDual );
   push_back_row( row, currentProblem );
   if( postsolveType == PostsolveType::kFull )
      push_back_col( col, currentProblem );
   else
   {
      indices.push_back( origcol_mapping[col] );
      values.push_back( 0 );
   }

   finishStorage();
}

template <typename REAL>
void
PostsolveStorage<REAL>::storeSubstitution( int col,
                                             SparseVectorView<REAL> equalityLHS,
                                             REAL equalityRHS )
{
   const REAL* coefs = equalityLHS.getValues();
   const int* columns = equalityLHS.getIndices();
   const int length = equalityLHS.getLength();
   assert( length > 1 );

   types.push_back( ReductionType::kSubstitutedCol );
   values.push_back( equalityRHS );
   indices.push_back( origcol_mapping[col] );
   for( int i = 0; i < length; ++i )
   {
      indices.push_back( origcol_mapping[columns[i]] );
      values.push_back( coefs[i] );
   }
   finishStorage();
}

/// col1 = col2scale * col2 and are merged into a new column y = col2 +
/// col2scale * col1 which takes over the index of col2
template <typename REAL>
void
PostsolveStorage<REAL>::storeParallelCols( int col1, bool col1integral,
                                     bool col1lbinf, REAL& col1lb,
                                     bool col1ubinf, REAL& col1ub,
                                     int col2, bool col2integral,
                                     bool col2lbinf, REAL& col2lb,
                                     bool col2ubinf, REAL& col2ub,
                                     REAL& col2scale )
{
   // encode the finiteness of the bounds in one integer and store it as
   // value for column 1
   int col1BoundFlags = 0;
   int col2BoundFlags = 0;

   if( col1integral )
      col1BoundFlags |= static_cast<int>( ColFlag::kIntegral );
   if( col1lbinf )
      col1BoundFlags |= static_cast<int>( ColFlag::kLbInf );
   if( col1ubinf )
      col1BoundFlags |= static_cast<int>( ColFlag::kUbInf );
   if( col2integral )
      col2BoundFlags |= static_cast<int>( ColFlag::kIntegral );
   if( col2lbinf )
      col2BoundFlags |= static_cast<int>( ColFlag::kLbInf );
   if( col2ubinf )
      col2BoundFlags |= static_cast<int>( ColFlag::kUbInf );

   // add all information
   indices.push_back( origcol_mapping[col1] );
   indices.push_back( col1BoundFlags );
   indices.push_back( origcol_mapping[col2] );
   indices.push_back( col2BoundFlags );
   indices.push_back( -1 ); // last index slot is not used
   values.push_back( col1lb );
   values.push_back( col1ub );
   values.push_back( col2lb );
   values.push_back( col2ub );
   values.push_back( col2scale );

   // add the range and the type of the reduction
   types.push_back( ReductionType::kParallelCol );

   finishStorage();
}

template <typename REAL>
void
PostsolveStorage<REAL>::storeReducedBoundsAndCost(
    const Vec<REAL>& col_lb, const Vec<REAL>& col_ub, const Vec<REAL>& row_lhs,
    const Vec<REAL>& row_rhs, const Vec<REAL>& coefficients,
    const Vec<RowFlags>& row_flags, const Vec<ColFlags>& col_flags )
{
   // store the row/col bounds and the objective of the reduced problem
   if( postsolveType == PostsolveType::kPrimal )
      return;
   types.push_back( ReductionType::kReducedBoundsCost );

   // col bound
   for( int col = 0; col < (int) col_lb.size(); col++ )
   {
      int flag_lb = 0;
      int flag_ub = 0;
      if( col_flags[col].test( ColFlag::kLbInf ) )
         flag_lb = 1;
      if( col_flags[col].test( ColFlag::kUbInf ) )
         flag_ub = 1;
      indices.push_back( flag_lb );
      values.push_back( col_lb[col] );
      indices.push_back( flag_ub );
      values.push_back( col_ub[col] );
   }

   // row bounds
   for( int row = 0; row < (int) row_lhs.size(); row++ )
   {
      int flag_lb = 0;
      int flag_ub = 0;
      if( row_flags[row].test( RowFlag::kLhsInf ) )
         flag_lb = 1;
      if( row_flags[row].test( RowFlag::kRhsInf ) )
         flag_ub = 1;
      indices.push_back( flag_lb );
      values.push_back( row_lhs[row] );
      indices.push_back( flag_ub );
      values.push_back( row_rhs[row] );
   }

   // col coefficients
   for( int col = 0; col < (int) coefficients.size(); col++ )
   {
      indices.push_back( col );
      values.push_back( coefficients[col] );
   }

   finishStorage();
}

} // namespace papilo

#endif
