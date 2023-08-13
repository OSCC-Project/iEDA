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

#ifndef _PAPILO_CORE_COMPONENTS_HPP_
#define _PAPILO_CORE_COMPONENTS_HPP_

#include "papilo/core/Problem.hpp"
#include "papilo/misc/Hash.hpp"
#include "papilo/misc/Vec.hpp"
#include "papilo/external/pdqsort/pdqsort.h"
#include <boost/pending/disjoint_sets.hpp>

namespace papilo
{

struct ComponentInfo
{
   int componentid;
   int nintegral;
   int ncontinuous;
   int nnonz;

   bool
   operator<( const ComponentInfo& other ) const
   {
      return std::make_tuple( nintegral, nnonz, ncontinuous, componentid ) <
             std::make_tuple( other.nintegral, other.nnonz, other.ncontinuous,
                              other.componentid );
   }
};

class Components
{
 private:
   Vec<int> col2comp;
   Vec<int> row2comp;
   Vec<int> compcols;
   Vec<int> comprows;
   Vec<int> compcolstart;
   Vec<int> comprowstart;
   Vec<ComponentInfo> compInfo;

 public:
   const int*
   getComponentsRows( int c ) const
   {
      return &comprows[comprowstart[c]];
   }

   int
   getComponentsNumRows( int c ) const
   {
      return comprowstart[c + 1] - comprowstart[c];
   }

   int
   getRowComponentIdx( int row ) const
   {
      return row2comp[row];
   }

   const int*
   getComponentsCols( int c ) const
   {
      return &compcols[compcolstart[c]];
   }

   int
   getComponentsNumCols( int c ) const
   {
      return compcolstart[c + 1] - compcolstart[c];
   }

   int
   getColComponentIdx( int col ) const
   {
      return col2comp[col];
   }

   const Vec<ComponentInfo>&
   getComponentInfo() const
   {
      return compInfo;
   }

   template <typename REAL>
   int
   detectComponents( const Problem<REAL>& problem )
   {
      const int ncols = problem.getNCols();
      std::unique_ptr<int[]> rank{ new int[ncols] };
      std::unique_ptr<int[]> parent{ new int[ncols] };
      boost::disjoint_sets<int*, int*> djsets( rank.get(), parent.get() );

      for( int i = 0; i != ncols; ++i )
         djsets.make_set( i );

      const ConstraintMatrix<REAL>& consMatrix = problem.getConstraintMatrix();
      const IndexRange* ranges;
      int nrows;

      std::tie( ranges, nrows ) = consMatrix.getRangeInfo();
      const int* colinds = consMatrix.getColumns();

      for( int r = 0; r != nrows; ++r )
      {
         if( ranges[r].end - ranges[r].start <= 1 )
            continue;

         int firstcol = colinds[ranges[r].start];

         for( int i = ranges[r].start + 1; i != ranges[r].end; ++i )
            djsets.link( firstcol, colinds[i] );
      }

      HashMap<int, int> componentmap;

      for( int i = 0; i != ncols; ++i )
      {
         int nextid = static_cast<int>( componentmap.size() );
         componentmap.insert( { djsets.find_set( i ), nextid } );
      }

      int numcomponents = static_cast<int>( componentmap.size() );

      if( numcomponents > 1 )
      {
         col2comp.resize( ncols );
         compcols.resize( ncols );

         for( int i = 0; i != ncols; ++i )
         {
            col2comp[i] = componentmap[djsets.find_set( i )];
            compcols[i] = i;
         }

         row2comp.resize( nrows );
         comprows.resize( nrows );
         for( int i = 0; i != nrows; ++i )
         {
            assert( problem.getConstraintMatrix()
                        .getRowCoefficients( i )
                        .getLength() > 0 );
            int col = problem.getConstraintMatrix()
                          .getRowCoefficients( i )
                          .getIndices()[0];
            row2comp[i] = col2comp[col];
            comprows[i] = i;
         }

         pdqsort( compcols.begin(), compcols.end(), [&]( int col1, int col2 ) {
            return col2comp[col1] < col2comp[col2];
         } );

         compcolstart.resize( numcomponents + 1 );
         int k = 0;

         // first component starts at 0
         compcolstart[0] = 0;

         // fill out starts for second to last components, also reuse the
         // col2comp vector to map the columns of a component to indices
         // starting at 0 without gaps
         for( int i = 1; i != numcomponents; ++i )
         {
            while( k != ncols && col2comp[compcols[k]] == i - 1 )
            {
               col2comp[compcols[k]] = k - compcolstart[i - 1];
               ++k;
            }

            compcolstart[i] = k;
         }

         while( k != ncols )
         {
            assert( col2comp[compcols[k]] == numcomponents - 1 );
            col2comp[compcols[k]] = k - compcolstart[numcomponents - 1];
            ++k;
         }

         // last component ends at ncols
         compcolstart[numcomponents] = ncols;

         pdqsort( comprows.begin(), comprows.end(), [&]( int row1, int row2 ) {
            return row2comp[row1] < row2comp[row2];
         } );

         comprowstart.resize( numcomponents + 1 );
         k = 0;
         // first component starts at 0
         comprowstart[0] = 0;

         // fill out starts for second to last components, also reuse the
         // row2comp vector to map the rows of a component to indices starting
         // at 0 without gaps
         for( int i = 1; i != numcomponents; ++i )
         {
            while( k != nrows && row2comp[comprows[k]] == i - 1 )
            {
               row2comp[comprows[k]] = k - comprowstart[i - 1];
               ++k;
            }

            comprowstart[i] = k;
         }

         while( k != nrows )
         {
            assert( row2comp[comprows[k]] == numcomponents - 1 );
            row2comp[comprows[k]] = k - comprowstart[numcomponents - 1];
            ++k;
         }
         // last component ends at nrows
         comprowstart[numcomponents] = nrows;

         // compute size informaton of components
         compInfo.resize( numcomponents );
         const auto& colsizes = problem.getColSizes();
         const auto& cflags = problem.getColFlags();

         for( int i = 0; i != numcomponents; ++i )
         {
            for( int j = compcolstart[i]; j != compcolstart[i + 1]; ++j )
            {
               if( cflags[compcols[j]].test( ColFlag::kIntegral ) )
                  ++compInfo[i].nintegral;
               else
                  ++compInfo[i].ncontinuous;

               compInfo[i].nnonz += colsizes[compcols[j]];
               compInfo[i].componentid = i;
            }
         }

         pdqsort( compInfo.begin(), compInfo.end() );
      }

      return numcomponents;
   }
};

} // namespace papilo

#endif
