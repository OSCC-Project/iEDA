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

#ifndef _PAPILO_CORE_SPARSE_STORAGE_HPP_
#define _PAPILO_CORE_SPARSE_STORAGE_HPP_

#include "papilo/misc/MultiPrecision.hpp"
#include "papilo/misc/Vec.hpp"
#include "papilo/external/pdqsort/pdqsort.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <tuple>

namespace papilo
{

/// type definition for a non-zero entry in triplet format
template <typename REAL>
using Triplet = std::tuple<int, int, REAL>;

// forward declaration of constraint matrix to declare the constraint matrix a
// friend class of SparseStorage
template <typename REAL>
class ConstraintMatrix;

struct IndexRange
{
   int start;
   int end;

   IndexRange() : start( -1 ), end( -1 ){};

   template <typename Archive>
   void
   serialize( Archive& ar, const unsigned int version )
   {
      ar& start;
      ar& end;
   }
};

/// Sparse storage class to store a matrix in modified CSR format
/// which includes a start and end for each row and allows for some free
/// space between the rows which is useful if the matrix may be altered
template <typename REAL>
class SparseStorage
{
   friend class ConstraintMatrix<REAL>;

 public:
   static constexpr double DEFAULT_SPARE_RATIO = 2.0;
   static constexpr int DEFAULT_MIN_INTER_ROW_SPACE = 4;

   SparseStorage() = default;
   SparseStorage( Vec<Triplet<REAL>> entries, int nRows_in, int nCols_in,
                  bool sorted = false, double spareRatio = DEFAULT_SPARE_RATIO,
                  int minInterRowSpace = DEFAULT_MIN_INTER_ROW_SPACE );
   SparseStorage( REAL* values_in, int* rowstart_in, int* columns_in,
                  int nRows_in, int nCols_in, int nnz_in,
                  double spareRatio = DEFAULT_SPARE_RATIO,
                  int minInterRowSpace = DEFAULT_MIN_INTER_ROW_SPACE );
   SparseStorage( int nRows_in, int nCols_in, int nnz_in, double spareRatio,
                  int minInterRowSpace );

   SparseStorage<REAL>
   getTranspose() const;

   Vec<int>
   compress( const Vec<int>& rowsize, const Vec<int>& colsize,
             bool full = false );

   int
   getNRows() const
   {
      return nRows;
   }

   bool
   shiftRows( const int* rowinds, int ninds, int maxshiftperrow,
              const Vec<int>& requiredSpareSpace );

   int
   getNCols() const
   {
      return nCols;
   }

   int
   getNnz() const
   {
      return nnz;
   }

   int&
   getNnz()
   {
      return nnz;
   }

   int
   getNAlloc() const
   {
      return nAlloc;
   }

   const REAL*
   getValues() const
   {
      return values.data();
   }

   REAL*
   getValues()
   {
      return values.data();
   }

   const Vec<REAL>&
   getValuesVec() const
   {
      return values;
   }

   const IndexRange*
   getRowRanges() const
   {
      return rowranges.data();
   }

   const Vec<IndexRange>&
   getRowRangesVec() const
   {
      return rowranges;
   }

   IndexRange*
   getRowRanges()
   {
      return rowranges.data();
   }

   const int*
   getColumns() const
   {
      return columns.data();
   }

   int*
   getColumns()
   {
      return columns.data();
   }

   const Vec<int>&
   getColumnsVec() const
   {
      return columns;
   }

   Vec<int>
   getRowStarts() const;

   // function to change existing coefficients in row. Must not be called with
   // coefficients that are currently not in the row. Changes must be given in
   // sorted order.
   template <typename HasNext, typename GetNext, typename CoeffChanged>
   int
   changeRowInplace( int row, HasNext&& hasNext, GetNext&& getNext,
                     CoeffChanged&& coeffChanged )
   {
      int i = rowranges[row].start;
      int j = 0;

      while( hasNext() )
      {
         int col;
         REAL newval;

         std::tie( col, newval ) = getNext();

         while( col != columns[i] )
         {
            if( j != 0 )
            {
               columns[i - j] = columns[i];
               values[i - j] = std::move( values[i] );
            }
            ++i;
         }

         coeffChanged( row, col, values[i], newval );

         if( newval == 0 )
         {
            ++j;
         }
         else if( j != 0 )
         {
            columns[i - j] = columns[i];
            values[i - j] = std::move( newval );
         }
         else
         {
            values[i] = std::move( newval );
         }

         ++i;
      }

      if( j != 0 )
      {
         while( i != rowranges[row].end )
         {
            columns[i - j] = columns[i];
            values[i - j] = std::move( values[i] );
            ++i;
         }

         rowranges[row].end -= j;
         nnz -= j;
      }

      return rowranges[row].end - rowranges[row].start;
   }

   /// change coefficients inside the given row. The details of how to change
   /// the row are templatized to allow for a generalized use of this function.
   /// The iteration of the changes is controlled by the Iter template type and
   /// the GetCol and GetVal callbacks to retrieve the column and the value out
   /// of that iterator type. If a matching support has been found the MergeVals
   /// callback is called with the first argument being the rows current value.
   /// The value returned by MergeVals is used as the new coefficient. This
   /// allows to use the function for setting the coefficients to values, or to
   /// add values to the coefficients. If a coefficient was changed the caller
   /// is informed about the old and the new coefficient via the CoeffChanged
   /// callback.
   /// Returns the new size of the row.
   template <typename Iter, typename GetCol, typename GetVal,
             typename MergeVals, typename CoeffChanged>
   int
   changeRow( int row, Iter it, Iter itend, GetCol&& getCol, GetVal&& getVal,
              MergeVals&& mergeVals, CoeffChanged&& coeffChanged,
              Vec<REAL>& valbuffer, Vec<int>& indbuffer )
   {
      auto rowmaxlen =
          rowranges[row].end - rowranges[row].start + ( itend - it );
      assert( valbuffer.empty() );
      assert( indbuffer.empty() );

      valbuffer.reserve( rowmaxlen );
      indbuffer.reserve( rowmaxlen );

      int i = rowranges[row].start;

      while( i != rowranges[row].end && it != itend )
      {
         int col = getCol( it );

         if( columns[i] == col )
         {
            REAL newval = mergeVals( values[i], getVal( it ) );
            coeffChanged( row, col, values[i], newval );

            if( newval != 0.0 )
            {
               indbuffer.push_back( col );
               valbuffer.push_back( std::move( newval ) );
            }
            ++i;
            ++it;
         }
         else if( columns[i] < col )
         {
            indbuffer.push_back( columns[i] );
            valbuffer.push_back( values[i] );
            ++i;
         }
         else
         {
            REAL newval = getVal( it );
            coeffChanged( row, col, 0.0, newval );

            indbuffer.push_back( col );
            valbuffer.push_back( std::move( newval ) );
            ++it;
         }
      }

      if( i != rowranges[row].end )
      {
         indbuffer.insert( indbuffer.end(), &columns[i],
                           &columns[rowranges[row].end] );
         valbuffer.insert( valbuffer.end(), &values[i],
                           &values[rowranges[row].end] );
      }
      else
      {
         while( it != itend )
         {
            int col = getCol( it );
            REAL newval = getVal( it );
            coeffChanged( row, col, 0.0, newval );

            indbuffer.push_back( col );
            valbuffer.push_back( std::move( newval ) );
            ++it;
         }
      }

      // copy over values from buffer
      assert( valbuffer.size() == indbuffer.size() );
      assert( std::is_sorted( indbuffer.begin(), indbuffer.end() ) );

      int newsize = static_cast<int>( indbuffer.size() );

      assert( newsize <= rowranges[row + 1].start - rowranges[row].start );

      nnz = nnz - rowranges[row].end + rowranges[row].start + newsize;

      // copy over values from buffer
      std::copy_n( valbuffer.data(), newsize, &values[rowranges[row].start] );
      std::memcpy( &columns[rowranges[row].start], indbuffer.data(),
                   sizeof( int ) * newsize );

      rowranges[row].end = rowranges[row].start + newsize;

      valbuffer.clear();
      indbuffer.clear();

      return newsize;
   }

   template <typename Archive>
   void
   serialize( Archive& ar, const unsigned int version )
   {
      ar& nRows;
      ar& nCols;
      ar& nnz;
      ar& nAlloc;
      ar& spareRatio;
      ar& minInterRowSpace;

      if( Archive::is_loading::value )
      {
         assert( values.empty() );
         assert( rowranges.empty() );
         assert( columns.empty() );

         rowranges.resize( nRows + 1 );
         values.resize( nAlloc );
         columns.resize( nAlloc );
      }

      for( int i = 0; i != nRows + 1; ++i )
         ar& rowranges[i];

      for( int i = 0; i != nRows; ++i )
      {
         for( int j = rowranges[i].start; j != rowranges[i].end; ++j )
         {
            ar& values[j];
            ar& columns[j];
         }
      }
   }

   int
   computeRowAlloc( int rowsize ) const
   {
      return static_cast<int>( rowsize * spareRatio ) + minInterRowSpace;
   }

 private:
   int
   computeNAlloc() const
   {
      return static_cast<int>( nnz * spareRatio ) + nRows * minInterRowSpace;
   }

   Vec<REAL> values;
   Vec<IndexRange> rowranges;
   Vec<int> columns;

   int nRows = -1;
   int nCols = -1;
   int nnz = -1;
   int nAlloc = -1;
   double spareRatio = 0.0;
   int minInterRowSpace = 0;
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class SparseStorage<double>;
extern template class SparseStorage<Quad>;
extern template class SparseStorage<Rational>;
#endif

template <typename REAL>
SparseStorage<REAL>::SparseStorage( Vec<Triplet<REAL>> entries, int nRows_in,
                                    int nCols_in, bool sorted,
                                    double spareRatio_, int minInterRowSpace_ )
    : nRows( nRows_in ), nCols( nCols_in ), spareRatio( spareRatio_ ),
      minInterRowSpace( minInterRowSpace_ )
{
   assert( spareRatio_ >= 0.0 );
   assert( !sorted || std::is_sorted( entries.begin(), entries.end() ) );

   if( !sorted )
      pdqsort( entries.begin(), entries.end() );

   nnz = entries.size();
   nAlloc = computeNAlloc();

   rowranges.resize( nRows + 1 );
   values.resize( nAlloc );
   columns.resize( nAlloc );

   rowranges[0].start = 0;

   int idx = 0;
   int current_row = 0;
   for( auto entry : entries )
   {
      int row;
      int col;
      REAL value;

      std::tie( row, col, value ) = entry;

      assert( row >= 0 && row < nRows );
      assert( col >= 0 && col < nCols );

      if( row != current_row )
      {
         assert( row > current_row );

         rowranges[current_row].end = idx;

         idx = rowranges[current_row].start +
               computeRowAlloc( rowranges[current_row].end -
                                rowranges[current_row].start );
         assert( idx > rowranges[current_row].end );

         rowranges[current_row + 1].start = idx;

         // there might be empty rows
         for( int r = current_row + 1; r < row; r++ )
         {
            rowranges[r].end = idx;
            rowranges[r + 1].start = idx;
         }

         current_row = row;
      }

      if( value != 0 )
      {
         assert( idx < nAlloc );

         values[idx] = value;
         columns[idx++] = col;
      }
      else
         --nnz;
   }

   rowranges[current_row].end = idx;

   idx = rowranges[current_row].start +
         computeRowAlloc( rowranges[current_row].end -
                          rowranges[current_row].start );
   assert( idx > rowranges[current_row].end );
   assert( idx <= nAlloc );

   rowranges[current_row + 1].start = idx;

   // there might be empty rows at the end
   for( int r = current_row + 1; r < nRows; r++ )
   {
      rowranges[r].end = idx;
      rowranges[r + 1].start = idx;
   }

   rowranges[nRows].end = idx;
}

template <typename REAL>
SparseStorage<REAL>::SparseStorage( int nRows_in, int nCols_in, int nnz_in,
                                    double spareRatio_, int minInterRowSpace_ )
    : nRows( nRows_in ), nCols( nCols_in ), nnz( nnz_in ),
      spareRatio( spareRatio_ ), minInterRowSpace( minInterRowSpace_ )
{
   nAlloc = computeNAlloc();
   assert( spareRatio_ >= 1.0 );

   rowranges.resize( nRows + 1 );
   values.resize( nAlloc );
   columns.resize( nAlloc );

   rowranges[nRows].start = nAlloc;
   rowranges[nRows].end = nAlloc;
}

template <typename REAL>
SparseStorage<REAL>::SparseStorage( REAL* values_in, int* rowstart_in,
                                    int* columns_in, int nRows_in, int nCols_in,
                                    int nnz_in, double spareRatio_in,
                                    int minInterRowSpace_in )
    : nRows( nRows_in ), nCols( nCols_in ), nnz( nnz_in ),
      spareRatio( spareRatio_in ), minInterRowSpace( minInterRowSpace_in )
{
   assert( nRows_in >= 0 && nnz_in >= 0 && spareRatio >= 1.0 );
   assert( rowstart_in );

   // compute length of new storage
   nAlloc = computeNAlloc();

   columns.resize( nAlloc );
   values.resize( nAlloc );
   rowranges.resize( nRows + 1 );

   // build storage
   int shift = 0;
   for( int r = 0; r < nRows; r++ )
   {
      rowranges[r].start = rowstart_in[r] + shift;

      for( int j = rowstart_in[r]; j < rowstart_in[r + 1]; j++ )
      {
         if( values_in[j] != REAL{ 0.0 } )
         {
            assert( j + shift >= 0 );

            values[j + shift] = values_in[j];
            columns[j + shift] = columns_in[j];
         }
         else
         {
            shift--;
         }
      }

      rowranges[r].end = rowstart_in[r + 1] + shift;
      const int rowsize = rowranges[r].end - rowranges[r].start;
      const int rowalloc = computeRowAlloc( rowsize );
      shift += rowalloc - rowsize;
   }

   assert( nRows == 0 );

   rowranges[nRows].start = rowstart_in[nRows] + shift;
   rowranges[nRows].end = rowranges[nRows].start;
}

template <typename REAL>
SparseStorage<REAL>
SparseStorage<REAL>::getTranspose() const
{
//   if( nCols <= 0 )
//      return SparseStorage<REAL>{};

   // compute nnz of each row of At (column of A)

   Vec<int> w( size_t( nCols ), 0 );

   for( int r = 0; r < nRows; r++ )
   {
      const int start = rowranges[r].start;
      const int end = rowranges[r].end;

      for( int j = start; j < end; j++ )
      {
         assert( values[j] != REAL{ 0.0 } );
         w[columns[j]]++;
      }
   }

   assert( spareRatio >= 1.0 );

   SparseStorage<REAL> transpose{ nCols, nRows, nnz, spareRatio,
                                  minInterRowSpace };

   // set row ranges of transpose
   transpose.rowranges[0].start = 0;

   for( int i = 1; i <= nCols; i++ )
   {
      const int oldstart = transpose.rowranges[i - 1].start;
      const int oldend = oldstart + w[i - 1];
      assert( oldend >= oldstart );

      transpose.rowranges[i - 1].end = oldend;
      transpose.rowranges[i].start =
          oldstart + transpose.computeRowAlloc( w[i - 1] );

      w[i - 1] = oldstart;
   }

   transpose.rowranges[nCols].start = transpose.nAlloc;
   transpose.rowranges[nCols].end = transpose.nAlloc;

   // fill values and columns arrays of transpose
   for( int r = 0; r < nRows; r++ )
   {
      const int start = rowranges[r].start;
      const int end = rowranges[r].end;

      for( int j = start; j < end; j++ )
      {
         const int idx = w[columns[j]];

         assert( idx < transpose.nAlloc );

         transpose.values[idx] = values[j];
         transpose.columns[idx] = r;

         w[columns[j]] = idx + 1;
      }
   }
   return transpose;
}

template <typename REAL>
Vec<int>
SparseStorage<REAL>::compress( const Vec<int>& rowsize, const Vec<int>& colsize,
                               bool full )
{
   if( full )
   {
      spareRatio = 1.0;
      minInterRowSpace = 0;
   }
   // now create and fill storage
   Vec<int> colsmap( static_cast<std::size_t>( nCols ) );

   if( nCols > 0 )
   {
      int colcount = 0;

      for( int i = 0; i < nCols; i++ )
      {
         if( colsize[i] >= 0 )
            colsmap[i] = colcount++;
         else
            colsmap[i] = -1;
      }

      nCols = colcount;
   }

   if( nRows > 0 )
   {
      int offset = 0;
      int rowcount = 0;
      for( int r = 0; r < nRows; r++ )
      {
         const int start = rowranges[r].start;
         const int end = rowranges[r].end;
         const int rowalloc = rowranges[r + 1].start - start;

         // empty row?
         if( rowsize[r] == -1 )
            offset += rowalloc;
         else
         {
            rowranges[rowcount].start = start;
            rowranges[rowcount].end = end;

            if( offset > 0 )
            {
               // move values and columns
               assert( start >= offset );

               std::move( &values[start], &values[end],
                          &values[start - offset] );
               std::move( &columns[start], &columns[end],
                          &columns[start - offset] );

               rowranges[rowcount].start -= offset;
               rowranges[rowcount].end -= offset;
            }

            offset = std::max(
                offset + rowalloc - computeRowAlloc( end - start ), 0 );

            ++rowcount;
         }

         assert( offset <= nAlloc );
      }

      rowranges[rowcount].start = rowranges[nRows].start - offset;
      rowranges[rowcount].end = rowranges[nRows].end - offset;

      nRows = rowcount;
      nAlloc = nAlloc - offset;
      assert( nAlloc >= 0 );

      rowranges.resize( nRows + 1 );
      values.resize( nAlloc );
      columns.resize( nAlloc );

      if( full )
      {
         rowranges.shrink_to_fit();
         values.shrink_to_fit();
         columns.shrink_to_fit();
      }

      for( int r = 0; r < nRows; r++ )
      {
         const int start = rowranges[r].start;
         const int end = rowranges[r].end;

         for( int j = start; j < end; j++ )
         {
            assert( columns[j] >= 0 );
            assert( columns[j] < static_cast<int>( colsmap.size() ) );
            columns[j] = colsmap[columns[j]];
            assert( columns[j] >= 0 );
            assert( columns[j] < nCols );
         }
      }
   }

   return colsmap;
}

template <typename REAL>
bool
SparseStorage<REAL>::shiftRows( const int* rowinds, int ninds,
                                int maxshiftperrow,
                                const Vec<int>& requiredSpareSpace )
{
   assert( ninds > 0 );
   assert( rowinds != nullptr );
   assert( (int) requiredSpareSpace.size() == ninds );
   assert( std::is_sorted( rowinds, rowinds + ninds ) );

   for( int i = 0; i != ninds; ++i )
   {
      const int row = rowinds[i];
      int missingspace = requiredSpareSpace[i] -
                         ( rowranges[row + 1].start - rowranges[row].end );
      if( missingspace > 0 )
      {
         int leftbound = i == 0 ? 0 : rowinds[i - 1] + 1;
         int rightbound = i == ninds - 1 ? nRows : rowinds[i + 1];

         int l = row;
         int r = row + 1;
         int lastshiftleft = 0;
         int lastshiftright = 0;
         int maxshift = maxshiftperrow;
         while( missingspace > 0 )
         {
            if( l > leftbound && r < rightbound )
            {
               int nspaceleft = std::min(
                   missingspace, rowranges[l].start - rowranges[l - 1].end );
               int nspaceright = std::min(
                   missingspace, rowranges[r + 1].start - rowranges[r].end );
               int nshiftleft = rowranges[l].end - rowranges[l].start;
               int nshiftright = rowranges[r].end - rowranges[r].start;

               bool goleft;
               if( nshiftleft == 0 )
                  goleft = true;
               else if( nshiftright == 0 )
                  goleft = false;
               else if( nshiftleft <= maxshift &&
                        nspaceleft / (double)nshiftleft >=
                            nspaceright / (double)nshiftright )
                  goleft = true;
               else if( nshiftright <= maxshift )
                  goleft = false;
               else
                  return false;

               // take direction that gives the most space per shifted
               // nonzero
               if( goleft )
               {
                  maxshift -= nshiftleft;
                  if( nspaceleft != 0 )
                  {
                     lastshiftleft = nspaceleft;
                     missingspace -= lastshiftleft;
                  }
                  --l;
               }
               else
               {
                  maxshift -= nshiftright;
                  if( nspaceright != 0 )
                  {
                     lastshiftright = nspaceright;
                     missingspace -= lastshiftright;
                  }
                  ++r;
               }
            }
            else if( l > leftbound &&
                     rowranges[l].end - rowranges[l].start <= maxshift )
            {
               maxshift -= rowranges[l].end - rowranges[l].start;
               lastshiftleft = std::min(
                   missingspace, rowranges[l].start - rowranges[l - 1].end );
               missingspace -= lastshiftleft;
               --l;
            }
            else if( r < rightbound &&
                     rowranges[r].end - rowranges[r].start <= maxshift )
            {
               maxshift -= rowranges[r].end - rowranges[r].start;
               lastshiftright = std::min( missingspace, rowranges[r + 1].start -
                                                            rowranges[r].end );
               missingspace -= lastshiftright;
               ++r;
            }
            else
               return false;
         }

         assert( missingspace == 0 &&
                 ( lastshiftleft > 0 || lastshiftright > 0 ) );
         if( lastshiftleft > 0 )
         {
            do
            {
               ++l;
               // skip possibly empty rows that did not increase the available
               // space
            } while( rowranges[l].start == rowranges[l - 1].end );

            REAL* valsout = &values[rowranges[l].start - lastshiftleft];
            int* colsout = &columns[rowranges[l].start - lastshiftleft];

            assert( rowranges[l - 1].end <=
                    rowranges[l].start - lastshiftleft );

            while( l <= row )
            {
               int shift = &values[rowranges[l].start] - valsout;

#ifndef NDEBUG
               Vec<REAL> tmpvals;
               Vec<int> tmpinds;
               tmpvals.insert( tmpvals.end(), &values[rowranges[l].start],
                               &values[rowranges[l].end] );
               tmpinds.insert( tmpinds.end(), &columns[rowranges[l].start],
                               &columns[rowranges[l].end] );
#endif
               if( rowranges[l].start != rowranges[l].end )
               {
                  valsout = std::move( &values[rowranges[l].start],
                                       &values[rowranges[l].end], valsout );
                  colsout = std::move( &columns[rowranges[l].start],
                                       &columns[rowranges[l].end], colsout );
               }

               rowranges[l].start -= shift;
               rowranges[l].end -= shift;
               assert( &columns[rowranges[l].end] == colsout );
               assert( &values[rowranges[l].end] == valsout );
               assert( rowranges[l - 1].end <= rowranges[l].start );
               assert( rowranges[l].end - rowranges[l].start ==
                       (int) tmpvals.size() );
               assert( std::equal( tmpvals.begin(), tmpvals.end(),
                                   &values[rowranges[l].start] ) );
               assert( std::equal( tmpinds.begin(), tmpinds.end(),
                                   &columns[rowranges[l].start] ) );
               ++l;
            }
         }

         if( lastshiftright > 0 )
         {
            do
            {
               --r;
               // skip possibly empty rows that did not increase the available
               // space
            } while( rowranges[r].end == rowranges[r + 1].start );

            REAL* valsout = &values[rowranges[r].end + lastshiftright];
            int* colsout = &columns[rowranges[r].end + lastshiftright];

            assert( rowranges[r + 1].start >=
                    rowranges[r].end + lastshiftright );

            while( r > row )
            {
               int shift = valsout - &values[rowranges[r].end];

#ifndef NDEBUG
               Vec<REAL> tmpvals;
               Vec<int> tmpinds;
               tmpvals.insert( tmpvals.end(), &values[rowranges[r].start],
                               &values[rowranges[r].end] );
               tmpinds.insert( tmpinds.end(), &columns[rowranges[r].start],
                               &columns[rowranges[r].end] );
#endif
               if( rowranges[r].start != rowranges[r].end )
               {
                  valsout =
                      std::move_backward( &values[rowranges[r].start],
                                          &values[rowranges[r].end], valsout );
                  colsout =
                      std::move_backward( &columns[rowranges[r].start],
                                          &columns[rowranges[r].end], colsout );
               }

               rowranges[r].start += shift;
               rowranges[r].end += shift;
               assert( &columns[rowranges[r].start] == colsout );
               assert( &values[rowranges[r].start] == valsout );
               assert( rowranges[r + 1].start >= rowranges[r].end );
               assert( rowranges[r].end - rowranges[r].start ==
                       (int) tmpvals.size() );
               assert( std::equal( tmpvals.begin(), tmpvals.end(),
                                   &values[rowranges[r].start] ) );
               assert( std::equal( tmpinds.begin(), tmpinds.end(),
                                   &columns[rowranges[r].start] ) );
               --r;
            }
         }

         assert( rowranges[row + 1].start - rowranges[row].end >=
                 requiredSpareSpace[i] );
      }

      // space should suffice now, either because it already did, or because it
      // was modified
      assert( requiredSpareSpace[i] <=
              ( rowranges[row + 1].start - rowranges[row].end ) );
   }

   return true;
}

template <typename REAL>
Vec<int>
SparseStorage<REAL>::getRowStarts() const
{
   int size = getNRows() + 1;
   Vec<int> colStart( size );

   unsigned int i;
   for( i = 0; i < colStart.size() - 1; ++i )
   {
      colStart[i] = rowranges[i].start;
      assert( rowranges[i].end == rowranges[i + 1].start );
   }
   colStart[i] = rowranges[i].end;

   return colStart;
}

} // namespace papilo

#endif
