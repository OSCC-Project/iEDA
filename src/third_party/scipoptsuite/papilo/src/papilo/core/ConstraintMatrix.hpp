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

#ifndef _PAPILO_CORE_CONSTRAINT_MATRIX_HPP_
#define _PAPILO_CORE_CONSTRAINT_MATRIX_HPP_

#define UNUSED(expr) do { (void)(expr); } while (0)


#include "papilo/core/MatrixBuffer.hpp"
#include "papilo/core/Objective.hpp"
#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/RowFlags.hpp"
#include "papilo/core/SingleRow.hpp"
#include "papilo/core/SparseStorage.hpp"
#include "papilo/core/VariableDomains.hpp"
#include "papilo/misc/MultiPrecision.hpp"
#include "papilo/misc/Num.hpp"
#include "papilo/misc/Vec.hpp"
#include "papilo/misc/compress_vector.hpp"
#include "papilo/misc/fmt.hpp"
#ifdef PAPILO_TBB
#include "papilo/misc/tbb.hpp"
#endif
#include <algorithm>

namespace papilo
{

/// non-owning type representing a sparse vector
template <typename REAL>
class SparseVectorView
{
 public:
   SparseVectorView() : vals( nullptr ), indices( nullptr ), len( 0 ) {}

   SparseVectorView( const REAL* _vals, const int* _inds, int _len )
       : vals( _vals ), indices( _inds ), len( _len )
   {
   }

   const REAL*
   getValues() const
   {
      return vals;
   }

   const int*
   getIndices() const
   {
      return indices;
   }

   int
   getLength() const
   {
      return len;
   }

   REAL
   getMaxAbsValue() const
   {
      REAL maxabsval = 0.0;

      for( int i = 0; i != len; ++i )
         maxabsval = std::max( REAL( abs( vals[i] ) ), maxabsval );

      return maxabsval;
   }

   std::pair<REAL, REAL>
   getMinMaxAbsValue() const
   {
      if( len != 0 )
      {
         REAL maxabsval = abs( vals[0] );
         REAL minabsval = maxabsval;

         for( int i = 1; i != len; ++i )
         {
            maxabsval = std::max( REAL( abs( vals[i] ) ), maxabsval );
            minabsval = std::min( REAL( abs( vals[i] ) ), minabsval );
         }

         return std::make_pair( minabsval, maxabsval );
      }

      return std::make_pair( 0, 0 );
   }

   REAL
   getDynamism() const
   {
      if( len != 0 )
      {
         REAL maxabsval = abs( vals[0] );
         REAL minabsval = maxabsval;

         for( int i = 1; i != len; ++i )
         {
            maxabsval = Num<REAL>::max( abs( vals[i] ), maxabsval );
            minabsval = Num<REAL>::min( abs( vals[i] ), minabsval );
         }

         return maxabsval / minabsval;
      }

      return 0;
   }

 private:
   const REAL* vals;
   const int* indices;
   int len;
};

/// type representing the constraint matrix including the left and right hand
/// sides
template <typename REAL>
class ConstraintMatrix
{
 public:
   ConstraintMatrix() = default;

   /// construct the constraint matrix from the given values
   ConstraintMatrix( SparseStorage<REAL> cons_matrix_init,
                     SparseStorage<REAL> cons_matrix_transp_init,
                     Vec<REAL> lhs_values_init, Vec<REAL> rhs_values_init,
                     Vec<RowFlags> row_flags_init )
       : cons_matrix( std::move( cons_matrix_init ) ),
         cons_matrix_transp( std::move( cons_matrix_transp_init ) ),
         lhs_values( std::move( lhs_values_init ) ),
         rhs_values( std::move( rhs_values_init ) ),
         flags( std::move( row_flags_init ) )
   {
      rowsize.reserve( cons_matrix.getNRows() );
      colsize.reserve( cons_matrix.getNCols() );

      auto rowranges = cons_matrix.getRowRanges();

      assert( (int) flags.size() == cons_matrix.getNRows() );

      for( int i = 0; i < cons_matrix.getNRows(); ++i )
         rowsize.push_back( rowranges[i].end - rowranges[i].start );

      auto colranges = cons_matrix_transp.getRowRanges();

      for( int i = 0; i < cons_matrix.getNCols(); ++i )
         colsize.push_back( colranges[i].end - colranges[i].start );
   }

   /// returns number of rows in the constraint matrix
   int
   getNRows() const
   {
      return cons_matrix.getNRows();
   }

   /// returns number of columns in the constraint matrix
   int
   getNCols() const
   {
      return cons_matrix.getNCols();
   }

   /// returns number of non-zeros in the constraint matrix
   int
   getNnz() const
   {
      assert( cons_matrix.getNnz() == cons_matrix_transp.getNnz() );
      return cons_matrix.getNnz();
   }


   std::pair<const IndexRange*, int>
   getRangeInfo() const
   {
      return std::make_pair( cons_matrix.getRowRanges(),
                             cons_matrix.getNRows() );
   }

   const int*
   getColumns() const
   {
      return cons_matrix.getColumns();
   }

   /// returns a sparse vector view on the row coefficients and their column
   /// indices
   SparseVectorView<REAL>
   getRowCoefficients( int r ) const
   {
      assert( r >= 0 && r < getNRows() );

      auto index_range = cons_matrix.getRowRanges()[r];

      return SparseVectorView<REAL>{
          cons_matrix.getValues() + index_range.start,
          cons_matrix.getColumns() + index_range.start,
          index_range.end - index_range.start };
   }

   /// returns a sparse vector view on the column coefficients and their row
   /// indices
   SparseVectorView<REAL>
   getColumnCoefficients( int c ) const
   {
      assert( c >= 0 && c < getNCols() );

      auto index_range = cons_matrix_transp.getRowRanges()[c];

      return SparseVectorView<REAL>{
          cons_matrix_transp.getValues() + index_range.start,
          cons_matrix_transp.getColumns() + index_range.start,
          index_range.end - index_range.start };
   }

   /// returns maximal change of constraint feasibility for the given change of
   /// value in the given column
   REAL
   getMaxFeasChange( int col, const REAL& val ) const
   {
      return abs( val * getColumnCoefficients( col ).getMaxAbsValue() );
   }

   /// returns dense vector with left hand side values of each row
   Vec<REAL>&
   getLeftHandSides()
   {
      return lhs_values;
   }

   /// returns dense vector with right hand side values of each row
   Vec<REAL>&
   getRightHandSides()
   {
      return rhs_values;
   }

   /// returns dense vector with left hand side values of each row
   const Vec<REAL>&
   getLeftHandSides() const
   {
      return lhs_values;
   }

   /// returns dense vector with right hand side values of each row
   const Vec<REAL>&
   getRightHandSides() const
   {
      return rhs_values;
   }

   const Vec<RowFlags>&
   getRowFlags() const
   {
      return flags;
   }

   Vec<RowFlags>&
   getRowFlags()
   {
      return flags;
   }

   /// modify the value of an element from the left hand side
   template <bool infval = false>
   void
   modifyLeftHandSide( const int index, const Num<REAL>& num,
                       const REAL& value = 0)
   {
      assert( index >= 0 );
      assert( index < getNRows() );
      if( !infval )
      {
         flags[index].unset( RowFlag::kLhsInf );

         if( num.isEq( value, rhs_values[index] ) )
            lhs_values[index] = rhs_values[index];
         else
            lhs_values[index] = value;

         if( !flags[index].test( RowFlag::kRhsInf ) &&
             lhs_values[index] == rhs_values[index]  )
            flags[index].set( RowFlag::kEquation );
         else
            flags[index].unset( RowFlag::kEquation );
      }
      else
      {
         flags[index].unset( RowFlag::kEquation );
         flags[index].set( RowFlag::kLhsInf );
      }
   }

   /// modify the value of an element from the right hand side
   template <bool infval = false>
   void
   modifyRightHandSide( const int index, const Num<REAL>& num,
                                         const REAL& value = 0)
   {
      assert( index >= 0 );
      assert( index < getNRows() );
      if( !infval )
      {
         flags[index].unset( RowFlag::kRhsInf );
         if( num.isEq( value, lhs_values[index] ) )
            rhs_values[index] = lhs_values[index];
         else
            rhs_values[index] = value;

         if( !flags[index].test( RowFlag::kLhsInf ) &&
             lhs_values[index] == rhs_values[index] )
            flags[index].set( RowFlag::kEquation );
         else
            flags[index].unset( RowFlag::kEquation );
      }
      else
      {
         flags[index].unset( RowFlag::kEquation );
         flags[index].set( RowFlag::kRhsInf );
      }
   }

   /// is given row redundant
   bool
   isRowRedundant( const int row ) const
   {
      return flags[row].test( RowFlag::kRedundant );
   }

   /// Compress the storage and the indices by removing empty rows and columns
   /// from the system. Returns a pair of vectors that store for each index used
   /// previously the new index or -1 if the corresponding row/column has been
   /// removed.
   /// The first vector of the pair stores the mapping for the rows, the second
   /// vector stores the mapping of the columns.
   std::pair<Vec<int>, Vec<int>>
   compress( bool full = false );

   void
   deleteRowsAndCols( Vec<int>& deletedRows, Vec<int>& deletedCols,
                      Vec<RowActivity<REAL>>& activities,
                      Vec<int>& singletonRows, Vec<int>& singletonCols,
                      Vec<int>& emptyCols );

   const Vec<int>&
   getRowSizes() const
   {
      return rowsize;
   }

   Vec<int>&
   getRowSizes()
   {
      return rowsize;
   }

   const Vec<int>&
   getColSizes() const
   {
      return colsize;
   }

   Vec<int>&
   getColSizes()
   {
      return colsize;
   }

   template <typename CoeffChanged>
   void
   changeCoefficients( const MatrixBuffer<REAL>& matrixBuffer,
                       Vec<int>& singletonRows, Vec<int>& singletonCols,
                       Vec<int>& emptyCols, Vec<RowActivity<REAL>>& activities,
                       CoeffChanged&& coeffChanged )
   {
      if( matrixBuffer.empty() )
         return;

      // update row major storage, and pass down the coeffChanged callback
#ifdef PAPILO_TBB
      tbb::parallel_invoke(
          [&]() {
#endif
             SmallVec<int, 32> buffer;
             const MatrixEntry<REAL>* iter =
                 matrixBuffer.template begin<true>( buffer );

             while( iter != matrixBuffer.end() )
             {
                int row = iter->row;

                int newsize = cons_matrix.changeRowInplace(
                    row,
                    [&]() {
                       return iter != matrixBuffer.end() && iter->row == row;
                    },
                    [&]() {
                       auto nextval = std::make_pair( iter->col, iter->val );
                       iter = matrixBuffer.template next<true>( buffer );
                       return nextval;
                    },
                    coeffChanged );

                if( newsize != rowsize[row] )
                {
                   switch( newsize )
                   {
                   case 0:
                      activities[row].min = 0;
                      activities[row].max = 0;
                      break;
                   case 1:
                      singletonRows.push_back( row );
                   default:
                      break;
                   }

                   rowsize[row] = newsize;
                }
             }
#ifdef PAPILO_TBB
          },
          [&]() {
#endif
             SmallVec<int, 32> buffer2;

             // update col major storage, do not pass down the coeffChanged
             // callback so that it
             /// is only called once
             const MatrixEntry<REAL>* iter2 =
                 matrixBuffer.template begin<false>( buffer2 );

             while( iter2 != matrixBuffer.end() )
             {
                int col = iter2->col;

                int newsize = cons_matrix_transp.changeRowInplace(
                    col,
                    [&]() {
                       return iter2 != matrixBuffer.end() && iter2->col == col;
                    },
                    [&]() {
                       auto nextval = std::make_pair( iter2->row, iter2->val );
                       iter2 = matrixBuffer.template next<false>( buffer2 );
                       return nextval;
                    },
                    []( int, int, REAL, REAL ) {} );

                if( newsize != colsize[col] )
                {
                   switch( newsize )
                   {
                   case 0:
                      emptyCols.push_back( col );
                      break;
                   case 1:
                      singletonCols.push_back( col );
                   default:
                      break;
                   }
                   // in case that a singleton var is aggregated and has 2
                   // appearances and then is reduced again immediately it may
                   // appear two times in the list -> causes no bug but some
                   // unneccessary overhead
                   colsize[col] = newsize;
                }
             }
#ifdef PAPILO_TBB
          } );
#endif
   }

   /// get the index of the column in the sparse array of row coefficients
   template <bool transpose = false>
   int
   getSparseIndex( int col, int row ) const
   {
      if( !transpose )
      {
         auto rowCoef = getRowCoefficients( row );
         const int* indices = rowCoef.getIndices();
         const int len = rowCoef.getLength();

         int index = std::lower_bound( indices, indices + len, col ) - indices;
         if( index != len && indices[index] == col )
            return index;
         return -1;
      }
      else
      {
         auto colCoef = getColumnCoefficients( col );
         const int* indices = colCoef.getIndices();
         const int len = colCoef.getLength();

         int index = std::lower_bound( indices, indices + len, row ) - indices;
         if( index != len && indices[index] == row )
            return index;
         return -1;
      }
   }

   /// checks in there is enough sapce in the sparse storage to performe an
   /// aggregation, and remove rows from the input relevantRows where the
   /// sparsity condition is not verified, returns true if the condition is
   /// verified on all relevantRows
   bool
   checkAggregationSparsityCondition( int col,
                                      const SparseVectorView<REAL>& equalityLHS,
                                      int maxfillin, int maxshiftperrow,
                                      Vec<int>& indbuffer );

   int
   sparsify( const Num<REAL>& num, int eqrow, const REAL& scale, int targetrow,
             Vec<int>& intbuffer, Vec<REAL>& valbuffer,
             const VariableDomains<REAL>& domains, Vec<int>& changedActivities,
             Vec<RowActivity<REAL>>& activities, Vec<int>& singletonRows,
             Vec<int>& singletonCols, Vec<int>& emptyCols, int presolveround );

   /// perform the substitution of a column using an equality, and updates
   /// the sides
   /// @param substituted_col the columns to substituted
   /// @param equalityLHS the right hand side of the equality i.e sum{
   /// a_j*x_i }, (needs to be sorted)
   /// @param equalityRHS the left hand side of the equality such that sum{
   /// a_j*x_i } = equalityRHS
   void
   aggregate( const Num<REAL>& num, int substituted_col, SparseVectorView<REAL> equalityLHS,
              REAL equalityRHS, const VariableDomains<REAL>& domains,
              Vec<int>& indbuffer, Vec<REAL>& valbuffer,
              Vec<Triplet<REAL>>& tripletbuffer, Vec<int>& changedActivities,
              Vec<RowActivity<REAL>>& activities, Vec<int>& singletonRows,
              Vec<int>& singletonCols, Vec<int>& emptyCols, int presolveround );

   const SparseStorage<REAL>&
   getMatrixTranspose() const
   {
      return cons_matrix_transp;
   }

   SparseStorage<REAL>&
   getMatrixTranspose()
   {
      return cons_matrix_transp;
   }

   const SparseStorage<REAL>&
   getConstraintMatrix() const
   {
      return cons_matrix;
   }

   SparseStorage<REAL>&
   getConstraintMatrix()
   {
      return cons_matrix;
   }

   template <typename Archive>
   void
   serialize( Archive& ar, const unsigned int version )
   {
      ar& cons_matrix;

      if( Archive::is_loading::value )
         cons_matrix_transp = cons_matrix.getTranspose();

      ar& lhs_values;
      ar& rhs_values;
      ar& flags;
      ar& rowsize;
      ar& colsize;
   }

 private:
   /// row-major compressed sparse storage (CSR) of the constraint matrix
   SparseStorage<REAL> cons_matrix;

   /// column-major compressed sparse storage (CSC) of the
   /// constraint matrix
   SparseStorage<REAL> cons_matrix_transp;

   /// left hand side values for each row in the constraint
   /// matrix
   Vec<REAL> lhs_values;

   /// right hand side values for each row in the constraint
   /// matrix
   Vec<REAL> rhs_values;

   Vec<RowFlags> flags;

   /// additional vector storing the number of non-zeros
   /// within each row of the constraint matrix
   Vec<int> rowsize;

   /// additional vector storing the number of non-zeros
   /// within each column of the constraint matrix
   Vec<int> colsize;
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class ConstraintMatrix<double>;
extern template class ConstraintMatrix<Quad>;
extern template class ConstraintMatrix<Rational>;
#endif

template <typename REAL>
std::pair<Vec<int>, Vec<int>>
ConstraintMatrix<REAL>::compress( bool full )
{
   std::pair<Vec<int>, Vec<int>> mappings;
#ifdef PAPILO_TBB
   tbb::parallel_invoke(
       [this, &mappings, full]() {
          mappings.first =
              cons_matrix_transp.compress( colsize, rowsize, full );
       },
       [this, &mappings, full]() {
          mappings.second = cons_matrix.compress( rowsize, colsize, full );
       } );
#else
   mappings.first =
       cons_matrix_transp.compress( colsize, rowsize, full );
   mappings.second = cons_matrix.compress( rowsize, colsize, full );
#endif

#ifdef PAPILO_TBB
   tbb::parallel_invoke(
       [this, &mappings, full]() {
          compress_vector( mappings.second, colsize );
          if( full )
             colsize.shrink_to_fit();
       },
       [this, &mappings, full]() {
          compress_vector( mappings.first, rowsize );
          if( full )
             rowsize.shrink_to_fit();
       },
       [this, &mappings, full]() {
          compress_vector( mappings.first, lhs_values );
          if( full )
             lhs_values.shrink_to_fit();
       },
       [this, &mappings, full]() {
          compress_vector( mappings.first, rhs_values );
          if( full )
             rhs_values.shrink_to_fit();
       },
       [this, &mappings, full]() {
          compress_vector( mappings.first, flags );
          if( full )
             flags.shrink_to_fit();
       } );
#else
   compress_vector( mappings.second, colsize );
   compress_vector( mappings.first, rowsize );
   compress_vector( mappings.first, lhs_values );
   compress_vector( mappings.first, rhs_values );
   compress_vector( mappings.first, flags );

   if( full )
   {
      colsize.shrink_to_fit();
      rowsize.shrink_to_fit();
      rhs_values.shrink_to_fit();
      lhs_values.shrink_to_fit();
      flags.shrink_to_fit();
   }

#endif

   return mappings;
}

template <typename REAL>
void
ConstraintMatrix<REAL>::deleteRowsAndCols( Vec<int>& deletedRows,
                                           Vec<int>& deletedCols,
                                           Vec<RowActivity<REAL>>& activities,
                                           Vec<int>& singletonRows,
                                           Vec<int>& singletonCols,
                                           Vec<int>& emptyCols )
{
   if( deletedRows.empty() && deletedCols.empty() )
      return;

   // CSR storage
   int* rowcols = cons_matrix.getColumns();
   IndexRange* rowranges = cons_matrix.getRowRanges();
   REAL* rowvalues = cons_matrix.getValues();

   // CSC storage
   int* colrows = cons_matrix_transp.getColumns();
   IndexRange* colranges = cons_matrix_transp.getRowRanges();
   REAL* colvalues = cons_matrix_transp.getValues();

#ifdef PAPILO_TBB
   tbb::parallel_invoke(
       [this, &deletedRows]() {
          for( int row : deletedRows )
          {
             cons_matrix.getNnz() -= rowsize[row];
             rowsize[row] = -1;
          }
       },
       [this, &deletedCols]() {
          for( int col : deletedCols )
             colsize[col] = -1;
       } );
#else
   for( int row : deletedRows )
   {
      cons_matrix.getNnz() -= rowsize[row];
      rowsize[row] = -1;
   }
   for( int col : deletedCols )
      colsize[col] = -1;
#endif

   // delete rows from row storage and update column sizes
#ifdef PAPILO_TBB
   tbb::parallel_invoke(
       [this, &deletedRows, rowranges, rowcols, &activities]() {
#endif
          for( int row : deletedRows )
          {
             assert( flags[row].test( RowFlag::kRedundant ) );

             for( int i = rowranges[row].start; i != rowranges[row].end; ++i )
             {
                int col = rowcols[i];
                if( colsize[col] == -1 )
                   continue;

                --colsize[col];
                assert( colsize[col] >= 0 );
             }

             rowranges[row].start = rowranges[row + 1].start;
             rowranges[row].end = rowranges[row + 1].start;
             lhs_values[row] = 0.0;
             rhs_values[row] = 0.0;
             activities[row].ninfmax = 0;
             activities[row].ninfmin = 0;
             activities[row].min = 0;
             activities[row].max = 0;
          }
#ifdef PAPILO_TBB
       },
       // delete cols from col storage and update rowsizes
       [this, &deletedCols, colranges, colrows] {
#endif
          for( int col : deletedCols )
          {
             for( int i = colranges[col].start; i != colranges[col].end; ++i )
             {
                int row = colrows[i];

                if( rowsize[row] == -1 )
                   continue;

                --rowsize[row];
                assert( rowsize[row] >= 0 );
             }

             colranges[col].start = colranges[col + 1].start;
             colranges[col].end = colranges[col + 1].start;
          }
#ifdef PAPILO_TBB
       } );
#endif

#ifdef PAPILO_TBB
   tbb::parallel_invoke(
       [this, colranges, &singletonCols, &emptyCols, colrows, colvalues]() {
#endif
          for( int col = 0; col != getNCols(); ++col )
          {
             // if the size did not change, skip column
             if( colsize[col] == -1 ||
                 colsize[col] == colranges[col].end - colranges[col].start )
                continue;

             // if the size is now 1, add to singleton column vector
             switch( colsize[col] )
             {
             case 0:
                emptyCols.push_back( col );
                colranges[col].start = colranges[col + 1].start;
                colranges[col].end = colranges[col + 1].start;
                break;
             case 1:
                singletonCols.push_back( col );
             }

             if( colsize[col] >= 1 )
             {
                // now move contents of column to occupy free spaces
                int j = 0;

                for( int i = colranges[col].start; i != colranges[col].end;
                     ++i )
                {
                   int row = colrows[i];

                   if( rowsize[row] == -1 )
                      ++j;
                   else if( j > 0 )
                   {
                      colvalues[i - j] = colvalues[i];
                      colrows[i - j] = colrows[i];
                   }
                }
                assert( colsize[col] >= 0 );
                assert( colsize[col] + colranges[col].start ==
                        colranges[col].end - j );

                colranges[col].end = colranges[col].start + colsize[col];
             }
          }
#ifdef PAPILO_TBB
       },
       [this, rowranges, &singletonRows, &activities, rowcols, rowvalues]() {
#endif
          for( int row = 0; row != getNRows(); ++row )
          {
             // if the size did not change, skip row
             if( rowsize[row] == -1 ||
                 rowsize[row] == rowranges[row].end - rowranges[row].start )
                continue;

             // if the size is now 1, add to singleton row vector
             switch( rowsize[row] )
             {
             case 0:
                activities[row].min = 0;
                activities[row].max = 0;
                break;
             case 1:
                singletonRows.push_back( row );
             }

             // now move contents of row to occupy free spaces
             int j = 0;

             for( int i = rowranges[row].start; i != rowranges[row].end; ++i )
             {
                int col = rowcols[i];

                if( colsize[col] == -1 )
                   ++j;
                else if( j > 0 )
                {
                   rowvalues[i - j] = rowvalues[i];
                   rowcols[i - j] = rowcols[i];
                }
             }

             assert( rowsize[row] >= 0 );
             assert( rowsize[row] + rowranges[row].start ==
                     rowranges[row].end - j );
             cons_matrix.getNnz() -= j;

             rowranges[row].end = rowranges[row].start + rowsize[row];
          }
#ifdef PAPILO_TBB
       } );
#endif

   cons_matrix_transp.getNnz() = cons_matrix.getNnz();

   deletedRows.clear();
   deletedCols.clear();
}

template <typename REAL>
bool
ConstraintMatrix<REAL>::checkAggregationSparsityCondition(
    int col, const SparseVectorView<REAL>& equalityLHS, int maxfillin,
    int maxshiftperrow, Vec<int>& indbuffer )
{
   const int* indices = equalityLHS.getIndices();
   const int len = equalityLHS.getLength();

   const auto& freecolcoef = getColumnCoefficients( col );
   const int* freecolindices = freecolcoef.getIndices();
   const int length = freecolcoef.getLength();

   auto rowranges = cons_matrix.getRowRanges();
   auto colranges = cons_matrix_transp.getRowRanges();

   int totalfillin = 0;
   bool eqinmatrix = false;
   bool shift = true;

   indbuffer.clear();
   indbuffer.reserve( std::max( length, len ) );

   for( int k = 0; k < length; ++k )
   {
      int row = freecolindices[k];
      auto currentrow = getRowCoefficients( row );
      const int* rowindices = currentrow.getIndices();
      const int currentrowlen = currentrow.getLength();

      if( rowindices == indices )
      {
         totalfillin -= len;
         eqinmatrix = true;
         indbuffer.push_back( 0 );
         continue;
      }

      int i = 0;
      int j = 0;
      int fillin = -1;

      while( i < len && j < currentrowlen )
      {
         if( indices[i] == rowindices[j] )
         {
            ++i;
            ++j;
         }
         else if( indices[i] < rowindices[j] )
         {
            ++i;
            ++fillin;
         }
         else
         {
            ++j;
         }
      }

      fillin += ( len - i );
      totalfillin += fillin;

      int sparespace = rowranges[row + 1].start - rowranges[row].end;

      if( sparespace < fillin )
         shift = true;

      indbuffer.push_back( fillin );
   }

   if( totalfillin > maxfillin )
   {
      indbuffer.clear();
      return false;
   }

   if( shift && !cons_matrix.shiftRows( freecolindices, length, maxshiftperrow,
                                        indbuffer ) )
   {
      indbuffer.clear();
      return false;
   }

   indbuffer.clear();
   shift = false;

   for( int k = 0; k < len; ++k )
   {
      int currentcolind = indices[k];
      if( currentcolind == col )
      {
         indbuffer.push_back( 0 );
         continue;
      }

      auto currentcol = getColumnCoefficients( currentcolind );
      const int* colindices = currentcol.getIndices();
      const int currentcollen = currentcol.getLength();

      int i = 0;
      int j = 0;
      int fillin = eqinmatrix ? -1 : 0;

      while( i < length && j < currentcollen )
      {
         if( freecolindices[i] == colindices[j] )
         {
            ++i;
            ++j;
         }
         else if( freecolindices[i] < colindices[j] )
         {
            ++i;
            ++fillin;
         }
         else
         {
            ++j;
         }
      }

      fillin += ( length - i );

      int sparespace =
          colranges[currentcolind + 1].start - colranges[currentcolind].end;

      if( sparespace < fillin )
         shift = true;

      indbuffer.push_back( fillin );
   }

   if( shift && !cons_matrix_transp.shiftRows( indices, len, maxshiftperrow,
                                               indbuffer ) )
   {
      indbuffer.clear();
      return false;
   }

   indbuffer.clear();
   return true;
}

template <typename REAL>
int
ConstraintMatrix<REAL>::sparsify(
    const Num<REAL>& num, int eqrow, const REAL& scale, int targetrow,
    Vec<int>& intbuffer, Vec<REAL>& valbuffer,
    const VariableDomains<REAL>& domains, Vec<int>& changedActivities,
    Vec<RowActivity<REAL>>& activities, Vec<int>& singletonRows,
    Vec<int>& singletonCols, Vec<int>& emptyCols, int presolveround )
{
   int ncancel = 0;
   int fillincol = -1;
   REAL fillinval = 0;
   IndexRange* colranges = cons_matrix_transp.getRowRanges();

   const IndexRange& eqrange = cons_matrix.getRowRanges()[eqrow];
   IndexRange& targetrange = cons_matrix.getRowRanges()[targetrow];

   int* rowcols = cons_matrix.getColumns();
   REAL* rowvals = cons_matrix.getValues();

   int j = eqrange.start;
   int k = targetrange.start;

   while( j != eqrange.end && k != targetrange.end )
   {
      if( rowcols[j] == rowcols[k] )
      {
         REAL newval = rowvals[j] * scale + rowvals[k];

         if( num.isZero( newval ) )
            ++ncancel;
         else if( num.isFeasZero( newval ) )
            return 0;

         ++j;
         ++k;
      }
      else if( rowcols[j] < rowcols[k] )
      {
         if( fillincol != -1 )
            return 0;

         fillincol = rowcols[j];
         if( colranges[fillincol + 1].start - colranges[fillincol].start ==
             colsize[fillincol] )
            return 0;

         fillinval = scale * rowvals[j];
         --ncancel;
         ++j;
      }
      else
      {
         ++k;
      }
   }

   int remainingfillin = eqrange.end - j;

   if( remainingfillin != 0 )
   {
      if( remainingfillin != 1 )
         return 0;
      if( fillincol != -1 )
         return 0;

      fillincol = rowcols[j];
      if( colranges[fillincol + 1].start - colranges[fillincol].start ==
          colsize[fillincol] )
         return 0;

      fillinval = scale * rowvals[j];
      --ncancel;
   }

   if( ncancel <= 0 )
      return 0;

   // change coefficients for column where fillin occurs (we allow at most one
   // such column, otherwise we return from this function before we reach this
   // part)
   if( fillincol != -1 )
   {

      assert( colsize[fillincol] <
              colranges[fillincol + 1].start - colranges[fillincol].start );

      colsize[fillincol] = cons_matrix_transp.changeRow(
          fillincol, 0, 1,
          [&]( int i ) {
             assert( i == 0 );
             return targetrow;
          },
          [&]( int i ) {
             assert( i == 0 );
             return fillinval;
          },
          []( const REAL& curr, const REAL& val ) {
             assert( curr == 0 );
             return val;
          },
          []( int, int, REAL, REAL ) {}, valbuffer, intbuffer );
   }

   // change coefficients in column storage for other columns
   j = eqrange.start;
   k = targetrange.start;

   while( j != eqrange.end && k != targetrange.end )
   {
      if( rowcols[j] == rowcols[k] )
      {
         int col = rowcols[k];
         assert( col != fillincol );

         REAL newval = rowvals[k] + scale * rowvals[j];

         if( num.isZero( newval ) )
         {
            --colsize[col];

            switch( colsize[col] )
            {
            case 0:
               emptyCols.push_back( col );
               break;
            case 1:
               singletonCols.push_back( col );
            }

            newval = 0;
         }

         int count = 0;
         int newsize = cons_matrix_transp.changeRowInplace(
             col, [&]() { return ( count++ ) == 0; },
             [&]() {
                assert( count == 1 );
                return std::make_pair( targetrow, newval );
             },
             []( int, int, REAL, REAL ) {} );

         UNUSED(newsize);
         assert( newsize == colsize[col] );

         ++j;
         ++k;
      }
      else if( rowcols[j] < rowcols[k] )
      {
         assert( rowcols[j] == fillincol );
         ++j;
      }
      else
      {
         assert( rowcols[k] < rowcols[j] );
         ++k;
      }
   }

   assert( j == eqrange.end ||
           ( j + 1 == eqrange.end && rowcols[j] == fillincol ) );

   // update sides if necessary
   if( rhs_values[eqrow] != 0 )
   {
      if( !flags[targetrow].test( RowFlag::kLhsInf ) )
         lhs_values[targetrow] += scale * rhs_values[eqrow];

      if( !flags[targetrow].test( RowFlag::kRhsInf ) )
         rhs_values[targetrow] += scale * rhs_values[eqrow];

      // due to numerics the row can become an equation
      if( !flags[targetrow].test( RowFlag::kLhsInf, RowFlag::kRhsInf,
                                  RowFlag::kEquation ) &&
          lhs_values[targetrow] == rhs_values[targetrow] )
         flags[targetrow].set( RowFlag::kEquation );
   }

   // finally update the row
   auto updateActivity = [&]( int row, int col, REAL oldval, REAL newval ) {
      auto activityChange = [row, presolveround, &changedActivities](
                                ActivityChange actChange,
                                RowActivity<REAL>& activity ) {
         if( activity.lastchange == presolveround )
            return;

         if( actChange == ActivityChange::kMin && activity.ninfmin > 1 )
            return;

         if( actChange == ActivityChange::kMax && activity.ninfmax > 1 )
            return;

         activity.lastchange = presolveround;
         changedActivities.push_back( row );
      };
      const SparseVectorView<REAL>& rowvec = getRowCoefficients( row );
      update_activity_after_coeffchange(
          domains.lower_bounds[col], domains.upper_bounds[col],
          domains.flags[col], oldval, newval, activities[row], rowvec.getLength(),
          rowvec.getIndices(), rowvec.getValues(), domains, num,
          activityChange );
   };

   int newsize = cons_matrix.changeRow(
       targetrow, eqrange.start, eqrange.end,
       [&]( int i ) { return rowcols[i]; },
       [&]( int i ) { return scale * rowvals[i]; },
       [&]( const REAL& a, const REAL& b ) {
          REAL val = a + b;
          if( num.isZero( val ) )
             val = 0;
          return val;
       },
       updateActivity, valbuffer, intbuffer );

   assert( rowsize[targetrow] - ncancel == newsize );
   rowsize[targetrow] = newsize;

   switch( rowsize[targetrow] )
   {
   case 0:
      activities[targetrow].min = 0;
      activities[targetrow].max = 0;
      break;
   case 1:
      singletonRows.push_back( targetrow );
   }

   assert( cons_matrix.getNnz() == cons_matrix_transp.getNnz() );

   return ncancel;
}

template <typename REAL>
void
ConstraintMatrix<REAL>::aggregate(
    const Num<REAL>& num, int substituted_col, SparseVectorView<REAL> equalityLHS,
    REAL equalityRHS, const VariableDomains<REAL>& domains, Vec<int>& indbuffer,
    Vec<REAL>& valbuffer, Vec<Triplet<REAL>>& tripletbuffer,
    Vec<int>& changedActivities, Vec<RowActivity<REAL>>& activities,
    Vec<int>& singletonRows, Vec<int>& singletonCols, Vec<int>& emptyCols,
    int presolveround )
{
   const int equalitylen = equalityLHS.getLength();
   const REAL* equalityvalues = equalityLHS.getValues();
   const int* equalityindices = equalityLHS.getIndices();

   assert( std::is_sorted( equalityindices, equalityindices + equalitylen ) );

   int freeColPos;
   for( freeColPos = 0; freeColPos < equalitylen; ++freeColPos )
   {
      if( equalityindices[freeColPos] == substituted_col )
         break;
   }
   assert( freeColPos != equalitylen );

   REAL eqbasescale = REAL{ -1 } / equalityvalues[freeColPos];

   assert( tripletbuffer.empty() );
   tripletbuffer.reserve( equalitylen * colsize[substituted_col] );

   auto updateActivity = [presolveround, &changedActivities, &domains,
                          &activities, &tripletbuffer, this, num](
                             int row, int col, REAL oldval, REAL newval ) {
      if( oldval == newval )
         return;

      auto activityChange = [row, presolveround, &changedActivities](
                                ActivityChange actChange,
                                RowActivity<REAL>& activity ) {
         if( activity.lastchange == presolveround )
            return;

         if( actChange == ActivityChange::kMin && activity.ninfmin > 1 )
            return;

         if( actChange == ActivityChange::kMax && activity.ninfmax > 1 )
            return;

         activity.lastchange = presolveround;
         changedActivities.push_back( row );
      };

      tripletbuffer.emplace_back( col, row, newval );

      const SparseVectorView<REAL>& rowVec = getRowCoefficients( row );
      update_activity_after_coeffchange(
          domains.lower_bounds[col], domains.upper_bounds[col],
          domains.flags[col], oldval, newval, activities[row],
          rowVec.getLength(), rowVec.getIndices(), rowVec.getValues(), domains,
          num, activityChange );
   };

   auto mergeVal = [&]( const REAL& oldval, const REAL& addition ) {
      REAL val = oldval + addition;
      if( num.isZero( val ) )
         return REAL{ 0 };

      return val;
   };

   const auto& freecol = getColumnCoefficients( substituted_col );
   const REAL* freecolcoef = freecol.getValues();
   const int* freecolindices = freecol.getIndices();
   const int freecollength = freecol.getLength();

   // make the changes in the matrix and update the constraints' sides
   for( int i = 0; i < freecollength; ++i )
   {
      int row = freecolindices[i];

      assert( flags[row].test( RowFlag::kRedundant ) ||
              ( !flags[row].test( RowFlag::kEquation ) &&
                ( flags[row].test( RowFlag::kLhsInf, RowFlag::kRhsInf ) ||
                  lhs_values[row] != rhs_values[row] ) ) ||
              ( flags[row].test( RowFlag::kEquation ) &&
                !flags[row].test( RowFlag::kLhsInf, RowFlag::kRhsInf ) &&
                lhs_values[row] == rhs_values[row] ) );

      // do not modify the equations content while it is still used,
      // just set the size and the sides to zero since they are copied in
      // local variables
      if( cons_matrix.getColumns() + cons_matrix.rowranges[row].start ==
          equalityindices )
      {
         for( int k = 0; k < equalitylen; ++k )
            tripletbuffer.emplace_back( equalityindices[k], row, 0 );

         flags[row].set( RowFlag::kRedundant );
         cons_matrix.rowranges[row].start =
             cons_matrix.rowranges[row + 1].start;
         cons_matrix.rowranges[row].end = cons_matrix.rowranges[row + 1].start;
         lhs_values[row] = 0.0;
         rhs_values[row] = 0.0;
         cons_matrix.getNnz() -= rowsize[row];
         rowsize[row] = -1;
         continue;
      }

      REAL eqscale = eqbasescale * freecolcoef[i];

      int newsize = cons_matrix.changeRow(
          row, int{ 0 }, equalitylen,
          [&]( int k ) { return equalityindices[k]; },
          [&]( int k ) {
             return k == freeColPos ? REAL( -freecolcoef[i] )
                                    : REAL( equalityvalues[k] * eqscale );
          },
          mergeVal, updateActivity, valbuffer, indbuffer );

      if( newsize != rowsize[row] )
      {
         switch( newsize )
         {
         case 0:
            activities[row].min = 0;
            activities[row].max = 0;
            break;
         case 1:
            singletonRows.push_back( row );
         default:
            break;
         }

         rowsize[row] = newsize;
      }

      assert(
          std::all_of( getRowCoefficients( row ).getIndices(),
                       getRowCoefficients( row ).getIndices() + rowsize[row],
                       [&]( int rowcol ) { return rowcol != substituted_col; } ) );

      // change the bounds
      if( equalityRHS != 0 )
      {
         if( !flags[row].test( RowFlag::kLhsInf ) )
            lhs_values[row] += eqscale * equalityRHS;

         if( !flags[row].test( RowFlag::kRhsInf ) )
            rhs_values[row] += eqscale * equalityRHS;

         // due to numerics the row can become an equation
         if( !flags[row].test( RowFlag::kLhsInf, RowFlag::kRhsInf,
                               RowFlag::kEquation ) &&
             lhs_values[row] == rhs_values[row] )
            flags[row].set( RowFlag::kEquation );
      }

      assert( flags[row].test( RowFlag::kRedundant ) ||
              ( !flags[row].test( RowFlag::kEquation ) &&
                ( flags[row].test( RowFlag::kLhsInf, RowFlag::kRhsInf ) ||
                  lhs_values[row] != rhs_values[row] ) ) ||
              ( flags[row].test( RowFlag::kEquation ) &&
                !flags[row].test( RowFlag::kLhsInf, RowFlag::kRhsInf ) &&
                lhs_values[row] == rhs_values[row] ) );
   }

   if( !tripletbuffer.empty() )
   {
      pdqsort( tripletbuffer.begin(), tripletbuffer.end() );

      auto handleCol = [&]( int col, int start, int end ) {
         int newsize = cons_matrix_transp.changeRow(
             col, start, end,
             [&]( int k ) { return std::get<1>( tripletbuffer[k] ); },
             [&]( int k ) { return std::get<2>( tripletbuffer[k] ); },
             []( const REAL& oldval, const REAL& newval ) { return newval; },
             []( int, int, REAL, REAL ) {}, valbuffer, indbuffer );

         if( newsize != colsize[col] )
         {
            switch( newsize )
            {
            case 0:
               emptyCols.push_back( col );
               break;
            case 1:
               singletonCols.push_back( col );
            default:
               break;
            }
            //TODO: remove column from singleton columns if necessary
            colsize[col] = newsize;
         }
      };

      int start = 0;
      int currcol = std::get<0>( tripletbuffer[0] );
      int nchgs = tripletbuffer.size();
      for( int i = 1; i != nchgs; ++i )
      {
         if( std::get<0>( tripletbuffer[i] ) != currcol )
         {
            handleCol( currcol, start, i );
            currcol = std::get<0>( tripletbuffer[i] );
            start = i;
         }
      }

      handleCol( currcol, start, nchgs );

      tripletbuffer.clear();
   }

   // set column size to zero
   cons_matrix_transp.rowranges[substituted_col].start =
       cons_matrix_transp.rowranges[substituted_col + 1].start;
   cons_matrix_transp.rowranges[substituted_col].end =
       cons_matrix_transp.rowranges[substituted_col + 1].start;
   cons_matrix_transp.getNnz() -= colsize[substituted_col];
   colsize[substituted_col] = -1;

   assert( cons_matrix_transp.getNnz() == cons_matrix.getNnz() );
}

} // namespace papilo

#endif
