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

#ifndef _PAPILO_MISC_DEPENDENT_ROWS_HPP_
#define _PAPILO_MISC_DEPENDENT_ROWS_HPP_

#include "papilo/Config.hpp"

#ifdef PAPILO_HAVE_LUSOL
extern "C"
{
#include "papilo/external/lusol/clusol.h"
}
#endif

#include "papilo/core/ConstraintMatrix.hpp"
#include "papilo/core/SparseStorage.hpp"
#include "papilo/misc/Vec.hpp"
#include <algorithm>
#include <array>
#include <boost/heap/d_ary_heap.hpp>

namespace papilo
{

template <typename REAL>
class DependentRows
{
 public:
#ifdef PAPILO_HAVE_LUSOL
   constexpr static bool Enabled = true;
#else
   constexpr static bool Enabled = false;
#endif

   DependentRows( int64_t nrows_, int64_t ncols_, int64_t maxnnz_ )
   {
      this->nrows = nrows_;
      this->ncols = ncols_ + 1;

      mat.reserve( maxnnz_ );
   }

   void
   reset( int64_t nrows_, int64_t ncols_, int64_t maxnnz )
   {
      this->nrows = nrows_;
      this->ncols = ncols_ + 1;
      mat.clear();
      mat.reserve( maxnnz );
   }

   void
   addRow( int rowIndex, SparseVectorView<REAL> rowValues, REAL side )
   {
      int len = rowValues.getLength();
      const int* inds = rowValues.getIndices();
      const REAL* vals = rowValues.getValues();

      mat.startBadge();

      for( int i = 0; i != len; ++i )
         mat.addBadgeEntry( rowIndex, inds[i], vals[i] );

      if( side != 0 )
         mat.addBadgeEntry( rowIndex, this->ncols - 1, side );

      mat.finishBadge();
   }

   struct PivotCandidate
   {
      int idx;
      int colsize;
      int rowsize;

      bool
      operator<( const PivotCandidate& other ) const
      {
         int trivial = colsize == 1 || rowsize == 1;
         int othertrivial = other.colsize == 1 || other.rowsize == 1;

         return std::make_tuple( trivial, colsize, rowsize ) >
                std::make_tuple( othertrivial, other.colsize, other.rowsize );
      }
   };

   struct LUSOL_Input
   {
      int64_t nrows;
      int64_t ncols;
      Vec<double> A;
      Vec<int64_t> indc;
      Vec<int64_t> indr;

      void
      setSize( int64_t _nrows, int64_t _ncols, int64_t nnz )
      {
         this->nrows = _nrows;
         this->ncols = _ncols;

         // preallocate enough storage for LUSOL to work
         int64_t alloc =
             std::max( { 8 * nnz, 32 * nrows, 32 * ncols, int64_t{ 10000 } } );
         A.reserve( alloc );
         indc.reserve( alloc );
         indr.reserve( alloc );
      }

      void
      addNnz( int64_t i, int64_t j, double val )
      {
         A.push_back( double( val ) );
         indc.push_back( i );
         indr.push_back( j );
      }

      void
      applyScaling()
      {
         // apply equilibrium scaling algorithm for remaining matrix
         Vec<double> rowmax( nrows );
         Vec<double> rowmin( nrows );
         Vec<double> colmax( ncols );
         Vec<double> colmin( ncols );

         for( int i = 0; i < (int) A.size(); ++i )
         {
            double absai = abs( A[i] );
            rowmax[indc[i] - 1] = std::max( rowmax[indc[i] - 1], absai );
            colmax[indr[i] - 1] = std::max( colmax[indr[i] - 1], absai );

            if( rowmin[indc[i] - 1] == 0 || absai < rowmin[indc[i] - 1] )
               rowmin[indc[i] - 1] = absai;
            if( colmin[indr[i] - 1] == 0 || absai < colmin[indr[i] - 1] )
               colmin[indr[i] - 1] = absai;
         }

         double maxrowratio = 1;
         for( int i = 0; i < (int) rowmax.size(); ++i )
            maxrowratio = std::max( maxrowratio, rowmax[i] / rowmin[i] );

         double maxcolratio = 1;
         for( int i = 0; i < (int) colmax.size(); ++i )
            maxcolratio = std::max( maxcolratio, colmax[i] / colmin[i] );

         if( maxrowratio < maxcolratio )
         {
            for( int i = 0; i < (int) A.size(); ++i )
               A[i] = ( A[i] / rowmax[indc[i] - 1] );

            for( int i = 0; i < (int) colmax.size(); ++i )
               colmax[i] = 0;

            for( int i = 0; i < (int) A.size(); ++i )
               colmax[indr[i] - 1] =
                   std::max( colmax[indr[i] - 1], abs( A[i] ) );

            for( int i = 0; i < (int) A.size(); ++i )
               A[i] = ( A[i] / colmax[indr[i] - 1] );
         }
         else
         {
            for( int i = 0; i < (int) A.size(); ++i )
               A[i] = ( A[i] / colmax[indr[i] - 1] );

            for( int i = 0; i < (int) rowmax.size(); ++i )
               rowmax[i] = 0;

            for( int i = 0; i < (int) A.size(); ++i )
               rowmax[indc[i] - 1] =
                   std::max( rowmax[indc[i] - 1], abs( A[i] ) );

            for( int i = 0; i < (int) A.size(); ++i )
               A[i] = ( A[i] / rowmax[indc[i] - 1] );
         }
      }

      void
      computeDependentColumns( Vec<int>& colmapping )
      {
#ifdef PAPILO_HAVE_LUSOL
         // is passed in : int64_t nelem;
         std::array<int64_t, 30> luparm;
         std::array<double, 30> parmlu;

         Vec<int64_t> p( nrows );     // allocate for nrows (=m)
         Vec<int64_t> lenr( nrows );  // allocate for nrows
         Vec<int64_t> locr( nrows );  // allocate for nrows
         Vec<int64_t> iqloc( nrows ); // allocate for nrows
         Vec<int64_t> ipinv( nrows ); // allocate for nrows

         Vec<int64_t> q( ncols );     // allocate for ncols (=n)
         Vec<int64_t> lenc( ncols );  // allocate for ncols
         Vec<int64_t> locc( ncols );  // allocate for ncols
         Vec<int64_t> iploc( ncols ); // allocate for ncols
         Vec<int64_t> iqinv( ncols ); // allocate for ncols

         Vec<double> w( ncols );
         int64_t inform;

         double factol = 2.5; // Stability tolerance

         luparm[0] = 6;  // File number for printed messages
         luparm[1] = -1; // Print level. >= 0 to get singularity info.
                         //              >=10 to get more LU statistics.
                         //              >=50 to get info on each pivot.
         luparm[2] = 5;  // maxcol
         luparm[5] = 1;  // Threshold Pivoting: 0 = TPP, 1 = TRP, 2 = TCP
         luparm[7] = 0;  // keepLU

         parmlu[0] = factol;  // Ltol1:  max |Lij| during Factor
         parmlu[1] = factol;  // Ltol2:  max |Lij| during Update
         parmlu[2] = 3.0e-13; // small:  drop tolerance
         parmlu[3] = 3.7e-11; // Utol1:  absolute tol for small Uii
         parmlu[4] = 3.7e-11; // Utol2:  relative tol for small Uii
         parmlu[5] = 3.0;     // Uspace:
         parmlu[6] = 0.3;     // dens1
         parmlu[7] = 0.5;     // dens2

         int64_t nelem = A.size();
         int64_t lena = A.capacity();

         clu1fac( &nrows, &ncols, &nelem, &lena, luparm.data(), parmlu.data(),
                  A.data(), indc.data(), indr.data(), p.data(), q.data(),
                  lenc.data(), lenr.data(), locc.data(), locr.data(),
                  iploc.data(), iqloc.data(), ipinv.data(), iqinv.data(),
                  w.data(), &inform );

         if( ( inform == 0 || inform == 1 ) && luparm[10] > 0 )
         {
            for( int i = 0; i < ncols; ++i )
            {
               if( w[i] > 0 )
                  colmapping[i] = -1;
            }

            colmapping.erase(
                std::remove( colmapping.begin(), colmapping.end(), -1 ),
                colmapping.end() );

            assert( colmapping.size() == luparm[10] );

            return;
         }
#endif
         colmapping.clear();
      }
   };

   int64_t
   preprocessLUFac( const Message& msg, const Num<REAL>& num,
                    LUSOL_Input& lusolInput, Vec<int>& rowmapping )
   {
      SmallVec<int, 32> stack;
      SmallVec<int, 32> stack2;

      Vec<int> rowsize( nrows );
      Vec<int> colsize( ncols );

      for( int i = 1; i != (int) mat.entries.size(); ++i )
      {
         assert( mat.entries[i].row < nrows );
         assert( mat.entries[i].col < ncols );
         assert( mat.entries[i].val != 0 );
         ++rowsize[mat.entries[i].row];
         ++colsize[mat.entries[i].col];
      }

#ifndef NDEBUG
      for( int i = 0; i != (int) colsize.size(); ++i )
      {
         int tmpcolsize = 0;
         for( auto coliter = mat.template beginStart<false>( stack, -1, i );
              coliter->col == i; coliter = mat.template next<false>( stack ) )
            ++tmpcolsize;
         assert( tmpcolsize == colsize[i] );
      }
      for( int i = 0; i != (int) rowsize.size(); ++i )
      {
         int tmprowsize = 0;
         for( auto rowiter = mat.template beginStart<true>( stack, i, -1 );
              rowiter->row == i; rowiter = mat.template next<true>( stack ) )
            ++tmprowsize;
         assert( tmprowsize == rowsize[i] );
      }
#endif

      boost::heap::d_ary_heap<PivotCandidate, boost::heap::mutable_<false>,
                              boost::heap::arity<4>>
          heap;

      heap.reserve( 2 * mat.getNnz() );

      for( int i = 1; i != (int) mat.entries.size(); ++i )
      {
         PivotCandidate p{ i, colsize[mat.entries[i].col],
                           rowsize[mat.entries[i].row] };
         heap.push( p );
      }

      int nremoved = 0;
      REAL minpivot = num.getFeasTol() * 1e4;

      while( !heap.empty() )
      {
         PivotCandidate pivot = heap.top();
         heap.pop();

         MatrixEntry<REAL>* pivotentry = &mat.entries[pivot.idx];

         // if pivot is zero we skip it
         if( pivotentry->val == 0 )
            continue;

         bool trivialpivot =
             colsize[pivotentry->col] == 1 || rowsize[pivotentry->row] == 1;

         // always accept trivial pivots, but if the pivot is non-trivial do
         // some additional checks
         if( !trivialpivot )
         {
            if( abs( pivotentry->val ) < minpivot )
               continue;

            // if the pivot was altered while in the heap and is not as good as
            // it was when added, then we add it to the heap for later
            if( pivot.colsize < colsize[pivotentry->col] ||
                pivot.rowsize < rowsize[pivotentry->row] )
            {
               pivot.colsize = colsize[pivotentry->col];
               pivot.rowsize = rowsize[pivotentry->row];
               heap.push( pivot );
               continue;
            }

            // only treat columns of at most size 2
            if( pivot.colsize > 2 )
               break;
         }

         assert( colsize[pivotentry->col] > 0 );
         assert( rowsize[pivotentry->row] > 0 );

         nremoved++;

         int col = pivotentry->col;
         int pivotrow = pivotentry->row;
         REAL pivotrowscale = -1.0 / pivotentry->val;

         if( !trivialpivot )
         {
            for( const MatrixEntry<REAL>* rowiter =
                     mat.template beginStart<true>( stack, pivotrow, -1 );
                 rowiter->row == pivotrow;
                 rowiter = mat.template next<true>( stack ) )
            {
               if( rowiter->col == col || rowiter->val == 0 )
                  continue;

               REAL pivrowval = rowiter->val * pivotrowscale;

               for( const MatrixEntry<REAL>* coliter =
                        mat.template beginStart<false>( stack2, -1, col );
                    coliter->col == col;
                    coliter = mat.template next<false>( stack2 ) )
               {
                  if( coliter->row == pivotrow || coliter->val == 0 )
                     continue;

                  const MatrixEntry<REAL>* rowentry =
                      mat.template findEntry<false>( coliter->row,
                                                     rowiter->col );

                  REAL delta = pivrowval * coliter->val;

                  if( rowentry != nullptr )
                  {
                     REAL newval = rowentry->val + delta;
                     if( rowentry->val == 0 )
                     {
                        ++colsize[rowentry->col];
                        ++rowsize[rowentry->row];
                     }

                     assert( rowiter->col != col ||
                             ( num.isZero( newval ) && rowentry->val != 0 ) );

                     if( num.isZero( newval ) )
                     {
                        const_cast<MatrixEntry<REAL>*>( rowentry )->val = 0;
                        --colsize[rowentry->col];
                        --rowsize[rowentry->row];
                     }
                     else
                     {
                        const_cast<MatrixEntry<REAL>*>( rowentry )->val =
                            newval;
                        int rowentryidx = ( rowentry - &mat.entries[0] );
                        assert( rowentryidx > 0 &&
                                rowentryidx < (int) mat.entries.size() &&
                                rowentry == &mat.entries[rowentryidx] );
                        heap.push( PivotCandidate{ rowentryidx,
                                                   colsize[rowentry->col],
                                                   rowsize[rowentry->row] } );
                     }
                  }
                  else
                  {
                     ++colsize[rowiter->col];
                     ++rowsize[coliter->row];
                     int rowentryidx = int( mat.entries.size() );
                     heap.push( PivotCandidate{ rowentryidx,
                                                colsize[rowiter->col],
                                                rowsize[coliter->row] } );
                     int tmpcoliteridx = ( coliter - &mat.entries[0] );
                     int tmprowiteridx = ( rowiter - &mat.entries[0] );
                     mat.addEntry( coliter->row, rowiter->col, delta );
                     coliter = &mat.entries[tmpcoliteridx];
                     rowiter = &mat.entries[tmprowiteridx];
                  }
               }
            }
         }

         for( const MatrixEntry<REAL>* coliter =
                  mat.template beginStart<false>( stack, -1, col );
              coliter->col == col; coliter = mat.template next<false>( stack ) )
         {
            if( coliter->val == 0 )
               continue;

            --rowsize[coliter->row];
            --colsize[coliter->col];
            const_cast<MatrixEntry<REAL>*>( coliter )->val = 0;
         }

         for( const MatrixEntry<REAL>* rowiter =
                  mat.template beginStart<true>( stack, pivotrow, -1 );
              rowiter->row == pivotrow;
              rowiter = mat.template next<true>( stack ) )
         {
            if( rowiter->val == 0 )
               continue;

            --rowsize[rowiter->row];
            --colsize[rowiter->col];
            const_cast<MatrixEntry<REAL>*>( rowiter )->val = 0;
         }

         assert( rowsize[pivotrow] == 0 );
         assert( colsize[col] == 0 );

         rowsize[pivotrow] = -1;
         colsize[col] = -1;
      }

      int64_t remainingnnz = 0;

      for( int i = 0; i < (int) rowsize.size(); ++i )
      {
         if( rowsize[i] != -1 )
         {
            rowmapping.push_back( i );
            remainingnnz += rowsize[i];
            rowsize[i] = rowmapping.size();
         }
      }

      msg.info( "preprocessed LU factor has {} nonzeros, preprocessing removed "
                "{} rows/cols\n",
                remainingnnz, nremoved );
      if( remainingnnz == 0 )
         return remainingnnz;

      nrows = rowmapping.size();
      ncols = 0;
      for( int i = 0; i < (int) colsize.size(); ++i )
      {
         if( colsize[i] != -1 )
         {
            ++ncols;
            colsize[i] = ncols;
         }
      }

      // add data to lusol transposed
      lusolInput.setSize( ncols, nrows, remainingnnz );

      for( int i = 1; i != (int) mat.entries.size(); ++i )
      {
         if( mat.entries[i].val == 0 )
            continue;

         lusolInput.addNnz( colsize[mat.entries[i].col],
                            rowsize[mat.entries[i].row],
                            double( mat.entries[i].val ) );
      }

      assert( remainingnnz == (int) lusolInput.A.size() );

      return remainingnnz;
   }

   Vec<int>
   getDependentRows( const Message& msg, const Num<REAL>& num )
   {
      Vec<int> rowmapping;
      LUSOL_Input lusolInput;

      if( !DependentRows<REAL>::Enabled )
         return rowmapping;

      int64_t nelem = preprocessLUFac( msg, num, lusolInput, rowmapping );

      // no remaining nonzeros means all remaining rows are redundant
      if( nelem > 0 )
      {
         lusolInput.applyScaling();

         msg.info( "calling LUSOL on remaining factor\n" );

         lusolInput.computeDependentColumns( rowmapping );
      }

      return rowmapping;
   }

 private:
   int64_t nrows;
   int64_t ncols;

   MatrixBuffer<REAL> mat;
};

} // namespace papilo

#endif
