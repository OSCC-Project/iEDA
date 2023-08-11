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

#ifndef _PAPILO_PRESOLVERS_DOMINATED_COLS_HPP_
#define _PAPILO_PRESOLVERS_DOMINATED_COLS_HPP_

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"
#include "papilo/core/SingleRow.hpp"
#include "papilo/misc/Hash.hpp"
#include "papilo/misc/Signature.hpp"
#ifdef PAPILO_TBB
#include "papilo/misc/tbb.hpp"
#endif

namespace papilo
{

template <typename REAL>
class DominatedCols : public PresolveMethod<REAL>
{
 public:
   DominatedCols() : PresolveMethod<REAL>()
   {
      this->setName( "domcol" );
      this->setTiming( PresolverTiming::kExhaustive );
   }

   bool
   initialize( const Problem<REAL>& problem,
               const PresolveOptions& presolveOptions ) override
   {
      if( presolveOptions.dualreds < 2 )
         this->setEnabled( false );
      return false;
   }

   /// stores implied bound information and signatures for a column
   struct ColInfo
   {
      Signature32 pos;
      Signature32 neg;
      int lbfree = 0;
      int ubfree = 0;

      Signature32
      getNegSignature( int scale ) const
      {
         assert( scale == 1 || scale == -1 );
         return scale == 1 ? neg : pos;
      }

      Signature32
      getPosSignature( int scale ) const
      {
         assert( scale == 1 || scale == -1 );
         return scale == 1 ? pos : neg;
      }

      bool
      allowsDomination( int scale, const ColInfo& other, int otherscale ) const
      {
         return getNegSignature( scale ).isSuperset(
                    other.getNegSignature( otherscale ) ) &&
                getPosSignature( scale ).isSubset(
                    other.getPosSignature( otherscale ) );
      }
   };

   struct DomcolReduction
   {
      int col1;
      int col2;
      int implrowlock;
      BoundChange boundchg;
   };

   PresolveStatus
   execute( const Problem<REAL>& problem,
            const ProblemUpdate<REAL>& problemUpdate, const Num<REAL>& num,
            Reductions<REAL>& reductions, const Timer& timer ) override;
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class DominatedCols<double>;
extern template class DominatedCols<Quad>;
extern template class DominatedCols<Rational>;
#endif

template <typename REAL>
PresolveStatus
DominatedCols<REAL>::execute( const Problem<REAL>& problem,
                              const ProblemUpdate<REAL>& problemUpdate,
                              const Num<REAL>& num,
                              Reductions<REAL>& reductions, const Timer& timer )
{
   const auto& obj = problem.getObjective().coefficients;
   const auto& consMatrix = problem.getConstraintMatrix();
   const auto& lbValues = problem.getLowerBounds();
   const auto& ubValues = problem.getUpperBounds();
   const auto& lhsValues = consMatrix.getLeftHandSides();
   const auto& rhsValues = consMatrix.getRightHandSides();
   const auto& rflags = consMatrix.getRowFlags();
   const auto& cflags = problem.getColFlags();
   const auto& activities = problem.getRowActivities();
   const auto& rowsize = consMatrix.getRowSizes();
   const int ncols = problem.getNCols();

   PresolveStatus result = PresolveStatus::kUnchanged;

   // do not call dominated column presolver too often, since it can be
   // expensive
   this->skipRounds( this->getNCalls() );

   Vec<ColInfo> colinfo( ncols );
#ifdef PAPILO_TBB
   tbb::concurrent_vector<int> unboundedcols;
#else
   Vec<int> unboundedcols;
#endif
   unboundedcols.reserve( ncols );

   // compute signatures and implied bound information of all columns in
   // parallel
#ifdef PAPILO_TBB
   tbb::parallel_for(
       tbb::blocked_range<int>( 0, ncols ),
       [&]( const tbb::blocked_range<int>& r ) {
          for( int col = r.begin(); col != r.end(); ++col )
#else
   for( int col = 0; col != ncols; ++col )
#endif
          {
             auto colvec = consMatrix.getColumnCoefficients( col );
             int collen = colvec.getLength();
             const int* colrows = colvec.getIndices();
             const REAL* colvals = colvec.getValues();

             if( cflags[col].test( ColFlag::kLbInf ) )
                colinfo[col].lbfree = -1;
             if( cflags[col].test( ColFlag::kUbInf ) )
                colinfo[col].ubfree = -1;

             for( int j = 0; j != collen; ++j )
             {
                int row = colrows[j];
                if( colinfo[col].ubfree == 0 &&
                    row_implies_UB( num, lhsValues[row], rhsValues[row],
                                    rflags[row], activities[row], colvals[j],
                                    lbValues[col], ubValues[col],
                                    cflags[col] ) )
                   colinfo[col].ubfree = j + 1;

                if( colinfo[col].lbfree == 0 &&
                    row_implies_LB( num, lhsValues[row], rhsValues[row],
                                    rflags[row], activities[row], colvals[j],
                                    lbValues[col], ubValues[col],
                                    cflags[col] ) )
                   colinfo[col].lbfree = j + 1;

                if( !rflags[row].test( RowFlag::kLhsInf, RowFlag::kRhsInf ) )
                {
                   // ranged row or equality, add to positive and negative
                   // signature
                   colinfo[col].pos.add( row );
                   colinfo[col].neg.add( row );
                }
                else if( rflags[row].test( RowFlag::kLhsInf ) )
                {
                   // <= constraint, add to positive signature for positive
                   // coefficient and negative signature otherwise
                   if( colvals[j] < 0 )
                      colinfo[col].neg.add( row );
                   else
                      colinfo[col].pos.add( row );
                }
                else
                {
                   // >= constraint, add to positive signature for negative
                   // coefficient and negative signature otherwise
                   assert( rflags[row].test( RowFlag::kRhsInf ) );
                   if( colvals[j] < 0 )
                      colinfo[col].pos.add( row );
                   else
                      colinfo[col].neg.add( row );
                }
             }

             if( colinfo[col].lbfree != 0 || colinfo[col].ubfree != 0 )
                unboundedcols.push_back( col );
          }
#ifdef PAPILO_TBB
       } );
#endif

   auto checkDominance = [&]( int col1, int col2, int scal1, int scal2 ) {
      assert( !cflags[col1].test( ColFlag::kIntegral ) ||
              cflags[col2].test( ColFlag::kIntegral ) );

      // first check if the signatures rule out domination
      if( !colinfo[col1].allowsDomination( scal1, colinfo[col2], scal2 ) )
         return false;

      auto col1vec = consMatrix.getColumnCoefficients( col1 );
      int col1len = col1vec.getLength();
      const int* col1rows = col1vec.getIndices();
      const REAL* col1vals = col1vec.getValues();

      auto col2vec = consMatrix.getColumnCoefficients( col2 );
      int col2len = col2vec.getLength();
      const int* col2rows = col2vec.getIndices();
      const REAL* col2vals = col2vec.getValues();

      int i = 0;
      int j = 0;

      while( i != col1len && j != col2len )
      {
         REAL val1;
         REAL val2;
         RowFlags rowf;

         if( col1rows[i] == col2rows[j] )
         {
            val1 = col1vals[i] * scal1;
            val2 = col2vals[j] * scal2;
            rowf = rflags[col1rows[i]];

            ++i;
            ++j;
         }
         else if( col1rows[i] < col2rows[j] )
         {
            val1 = col1vals[i] * scal1;
            val2 = 0;
            rowf = rflags[col1rows[i]];

            ++i;
         }
         else
         {
            assert( col1rows[i] > col2rows[j] );
            val1 = 0;
            val2 = col2vals[j] * scal2;
            rowf = rflags[col2rows[j]];

            ++j;
         }

         if( !rowf.test( RowFlag::kLhsInf, RowFlag::kRhsInf ) )
         {
            // ranged row or equality, values must be equal
            if( !num.isEq( val1, val2 ) )
               return false;
         }
         else if( rowf.test( RowFlag::kLhsInf ) )
         {
            // <= constraint, col1 must have a smaller or equal coefficient
            if( num.isGT( val1, val2 ) )
               return false;
         }
         else
         {
            // >= constraint, col1 must have a larger or equal coefficient
            assert( rowf.test( RowFlag::kRhsInf ) );
            if( num.isLT( val1, val2 ) )
               return false;
         }
      }

      while( i != col1len )
      {
         REAL val1 = col1vals[i] * scal1;
         RowFlags rowf = rflags[col1rows[i]];
         ++i;

         if( !rowf.test( RowFlag::kLhsInf, RowFlag::kRhsInf ) )
         {
            // ranged row or equality, values must be equal
            return false;
         }
         else if( rowf.test( RowFlag::kLhsInf ) )
         {
            // <= constraint, col1 must have a smaller or equal coefficient
            if( num.isGT( val1, 0 ) )
               return false;
         }
         else
         {
            // >= constraint, col1 must have a larger or equal coefficient
            assert( rowf.test( RowFlag::kRhsInf ) );
            if( num.isLT( val1, 0 ) )
               return false;
         }
      }

      while( j != col2len )
      {
         REAL val2 = col2vals[j] * scal2;
         RowFlags rowf = rflags[col2rows[j]];
         ++j;

         if( !rowf.test( RowFlag::kLhsInf, RowFlag::kRhsInf ) )
         {
            // ranged row or equality, values must be equal
            return false;
         }
         else if( rowf.test( RowFlag::kLhsInf ) )
         {
            // <= constraint, col1 must have a smaller or equal coefficient
            if( num.isGT( 0, val2 ) )
               return false;
         }
         else
         {
            // >= constraint, col1 must have a larger or equal coefficient
            assert( rowf.test( RowFlag::kRhsInf ) );
            if( num.isLT( 0, val2 ) )
               return false;
         }
      }

      return true;
   };

#ifdef PAPILO_TBB
   tbb::concurrent_vector<DomcolReduction> domcolreductions;
#else
   Vec<DomcolReduction> domcolreductions;
#endif

#ifdef PAPILO_TBB
   // scan unbounded columns if they dominate other columns
   tbb::parallel_for(
       tbb::blocked_range<int>( 0, (int) unboundedcols.size() ),
       [&]( const tbb::blocked_range<int>& r ) {
          for( int k = r.begin(); k != r.end(); ++k )
#else
   for( int k = 0; k < (int) unboundedcols.size(); ++k )
#endif
          {
             int i = unboundedcols[k];
             int lbfree = colinfo[i].lbfree;
             int ubfree = colinfo[i].ubfree;

             assert( lbfree != 0 || ubfree != 0 );

             auto colvec = consMatrix.getColumnCoefficients( i );
             int collen = colvec.getLength();
             const int* colrows = colvec.getIndices();
             const REAL* colvals = colvec.getValues();
             int scale;
             int implrowlock;

             // determine the scale of the dominating column depending on
             // whether the upper or lower bound is free, and remember which
             // row needs to be locked to protect the implied bound (if any)
             if( ubfree != 0 )
             {
                scale = 1;
                implrowlock = ubfree > 0 ? colrows[ubfree - 1] : -1;
             }
             else if( lbfree != 0 )
             {
                scale = -1;
                implrowlock = lbfree > 0 ? colrows[lbfree - 1] : -1;
             }
             else
                continue;

             int bestrow = -1;
             int bestrowsize = std::numeric_limits<int>::max();

             for( int j = 0; j < collen; ++j )
             {
                int row = colrows[j];
                if( ( !rflags[row].test( RowFlag::kLhsInf, RowFlag::kRhsInf ) ||
                      ( !rflags[row].test( RowFlag::kRhsInf ) &&
                        scale * colvals[j] > 0 ) ||
                      ( !rflags[row].test( RowFlag::kLhsInf ) &&
                        scale * colvals[j] < 0 ) ) &&
                    rowsize[row] < bestrowsize )
                {
                   bestrow = j;
                   bestrowsize = rowsize[row];
                }
             }

             if( bestrow == -1 || bestrowsize <= 1 )
                continue;

             auto candrowvec =
                 consMatrix.getRowCoefficients( colrows[bestrow] );
             REAL colval = colvals[bestrow] * scale;
             REAL colobj = obj[i] * scale;
             bestrow = colrows[bestrow];
             int rowlen = candrowvec.getLength();
             const int* rowcols = candrowvec.getIndices();
             const REAL* rowvals = candrowvec.getValues();

             for( int j = 0; j != rowlen; ++j )
             {
                int col = rowcols[j];
                if( col == i || ( cflags[i].test( ColFlag::kIntegral ) &&
                                  !cflags[col].test( ColFlag::kIntegral ) ) )
                   continue;

                bool fixtolb = false;
                bool fixtoub = false;

                if( !rflags[bestrow].test( RowFlag::kLhsInf,
                                           RowFlag::kRhsInf ) )
                {
                   if( !cflags[col].test( ColFlag::kLbInf ) &&
                       num.isEq( colval, rowvals[j] ) &&
                       num.isLE( colobj, obj[col] ) &&
                       checkDominance( i, col, scale, 1 ) )
                      fixtolb = true;

                   if( !cflags[col].test( ColFlag::kUbInf ) &&
                       num.isEq( colval, -rowvals[j] ) &&
                       num.isLE( colobj, -obj[col] ) &&
                       checkDominance( i, col, scale, -1 ) )
                      fixtoub = true;
                }
                else if( rflags[bestrow].test( RowFlag::kLhsInf ) )
                {
                   assert( colval > 0 &&
                           !rflags[bestrow].test( RowFlag::kRhsInf ) );
                   if( !cflags[col].test( ColFlag::kLbInf ) &&
                       num.isLE( colval, rowvals[j] ) &&
                       num.isLE( colobj, obj[col] ) &&
                       checkDominance( i, col, scale, 1 ) )
                      fixtolb = true;

                   if( !cflags[col].test( ColFlag::kUbInf ) &&
                       num.isLE( colval, -rowvals[j] ) &&
                       num.isLE( colobj, -obj[col] ) &&
                       checkDominance( i, col, scale, -1 ) )
                      fixtoub = true;
                }
                else
                {
                   assert( colval < 0 &&
                           rflags[bestrow].test( RowFlag::kRhsInf ) );
                   if( !cflags[col].test( ColFlag::kLbInf ) &&
                       num.isGE( colval, rowvals[j] ) &&
                       num.isLE( colobj, obj[col] ) &&
                       checkDominance( i, col, scale, 1 ) )
                      fixtolb = true;

                   if( !cflags[col].test( ColFlag::kUbInf ) &&
                       num.isGE( colval, -rowvals[j] ) &&
                       num.isLE( colobj, -obj[col] ) &&
                       checkDominance( i, col, scale, -1 ) )
                      fixtoub = true;
                }

                if( fixtolb || fixtoub )
                {
                   domcolreductions.push_back( DomcolReduction{
                       i, col, implrowlock,
                       fixtolb ? BoundChange::kUpper : BoundChange::kLower } );
                }
             }
          }
#ifdef PAPILO_TBB
       } );
#endif

   if( !domcolreductions.empty() )
   {
      result = PresolveStatus::kReduced;
      // sort reductions by the smallest col in the domcol reduction, so that
      // parallel ones are adjacent
      pdqsort( domcolreductions.begin(), domcolreductions.end(),
               []( const DomcolReduction& a, const DomcolReduction& b ) {
                  bool smaller_first_a = a.col1 < a.col2;
                  bool smaller_first_b = b.col1 < b.col2;
                  int smaller_row_a = smaller_first_a ? a.col1 : a.col2;
                  int smaller_row_b = smaller_first_b ? b.col1 : b.col2;
                  if( smaller_row_a != smaller_row_b )
                     return smaller_row_a < smaller_row_b;
                  return ( ( !smaller_first_a ) ? a.col1 : a.col2 ) <
                         ( ( !smaller_first_b ) ? b.col1 : b.col2 );
               } );

      for( int i = 0; i < (int) domcolreductions.size(); i++ )
      {
         // check if consecutively reductions are equal
         const DomcolReduction dr = domcolreductions[i];
         if( i < (int) domcolreductions.size() - 1 )
         {
            const DomcolReduction dr2 = domcolreductions[i + 1];
            if( dr2.col1 == dr.col2 && dr.col1 == dr2.col2 )
            {
               if( dr.implrowlock > 0 )
                  continue;
               i++;
            }
         }
         TransactionGuard<REAL> tg{ reductions };
         reductions.lockCol( dr.col1 );
         reductions.lockColBounds( dr.col1 );
         reductions.lockCol( dr.col2 );
         reductions.lockColBounds( dr.col2 );
         //TODO: check if >=0 is correct instead of >0
         if( dr.implrowlock >= 0 )
            reductions.lockRow( dr.implrowlock );

         if( dr.boundchg == BoundChange::kUpper )
            reductions.fixCol( dr.col2, lbValues[dr.col2], dr.implrowlock );
         else
            reductions.fixCol( dr.col2, ubValues[dr.col2], dr.implrowlock);
      }

   }

   return result;
}

} // namespace papilo

#endif
