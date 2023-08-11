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

#ifndef _PAPILO_IO_MPS_WRITER_
#define _PAPILO_IO_MPS_WRITER_

#include "papilo/Config.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/misc/Vec.hpp"
#include "papilo/misc/fmt.hpp"
#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <cmath>
#include <fstream>
#include <iostream>

#ifdef PAPILO_USE_BOOST_IOSTREAMS_WITH_BZIP2
#include <boost/iostreams/filter/bzip2.hpp>
#endif
#ifdef PAPILO_USE_BOOST_IOSTREAMS_WITH_ZLIB
#include <boost/iostreams/filter/gzip.hpp>
#endif

namespace papilo
{

/// Writer to write problem structures into an mps file
template <typename REAL>
struct MpsWriter
{
   static void
   writeProb( const std::string& filename, const Problem<REAL>& prob,
              const Vec<int>& row_mapping, const Vec<int>& col_mapping )
   {
      const ConstraintMatrix<REAL>& consmatrix = prob.getConstraintMatrix();
      const Vec<std::string>& consnames = prob.getConstraintNames();
      const Vec<std::string>& varnames = prob.getVariableNames();
      const Vec<REAL>& lhs = consmatrix.getLeftHandSides();
      const Vec<REAL>& rhs = consmatrix.getRightHandSides();
      const Objective<REAL>& obj = prob.getObjective();
      const Vec<ColFlags>& col_flags = prob.getColFlags();
      const Vec<RowFlags>& row_flags = prob.getRowFlags();

      std::ofstream file( filename, std::ofstream::out );
      boost::iostreams::filtering_ostream out;

#ifdef PAPILO_USE_BOOST_IOSTREAMS_WITH_ZLIB
      if( boost::algorithm::ends_with( filename, ".gz" ) )
         out.push( boost::iostreams::gzip_compressor() );
#endif

#ifdef PAPILO_USE_BOOST_IOSTREAMS_WITH_BZIP2
      if( boost::algorithm::ends_with( filename, ".bz2" ) )
         out.push( boost::iostreams::bzip2_compressor() );
#endif

      out.push( file );

      fmt::print( out, "*ROWS:         {}\n", consmatrix.getNRows() );
      fmt::print( out, "*COLUMNS:      {}\n", consmatrix.getNCols() );
      fmt::print( out, "*INTEGER:      {}\n", prob.getNumIntegralCols() );
      fmt::print( out, "*NONZERO:      {}\n*\n*\n", consmatrix.getNnz() );

      fmt::print( out, "NAME          {}\n", prob.getName() );
      fmt::print( out, "ROWS\n" );
      fmt::print( out, " N  OBJ\n" );
      bool hasRangedRow = false;
      for( int i = 0; i < consmatrix.getNRows(); ++i )
      {
         assert( !consmatrix.isRowRedundant( i ) );
         char type;

         if( row_flags[i].test( RowFlag::kLhsInf ) &&
             row_flags[i].test( RowFlag::kRhsInf ) )
            type = 'N';
         else if( row_flags[i].test( RowFlag::kRhsInf ) )
            type = 'G';
         else if( row_flags[i].test( RowFlag::kLhsInf ) )
            type = 'L';
         else
         {
            if( !row_flags[i].test( RowFlag::kEquation ) )
               hasRangedRow = true;
            type = 'E';
         }

         fmt::print( out, " {}  {}\n", type, consnames[row_mapping[i]] );
      }

      fmt::print( out, "COLUMNS\n" );

      int hasintegral = prob.getNumIntegralCols() != 0;

      for( int integral = 0; integral <= hasintegral; ++integral )
      {
         if( integral )
            fmt::print( out,
                        "    MARK0000  'MARKER'                 'INTORG'\n" );

         for( int i = 0; i < consmatrix.getNCols(); ++i )
         {
            if( col_flags[i].test( ColFlag::kInactive ) )
               continue;

            if( ( !col_flags[i].test( ColFlag::kIntegral ) && integral ) ||
                ( col_flags[i].test( ColFlag::kIntegral ) && !integral ) )
               continue;

            assert(
                !col_flags[i].test( ColFlag::kFixed, ColFlag::kSubstituted ) );

            if( obj.coefficients[i] != 0.0 )
            {
               fmt::print( out, "    {: <9} OBJ       {:.15}\n",
                           varnames[col_mapping[i]],
                           double( obj.coefficients[i] ) );
            }

            SparseVectorView<REAL> column =
                consmatrix.getColumnCoefficients( i );

            const int* rowinds = column.getIndices();
            const REAL* colvals = column.getValues();
            int len = column.getLength();

            for( int j = 0; j < len; ++j )
            {
               int r = rowinds[j];

               // discard redundant rows when writing problem
               if( consmatrix.isRowRedundant( r ) )
                  continue;

               // normal row
               fmt::print( out, "    {: <9} {: <9} {:.15}\n",
                           varnames[col_mapping[i]], consnames[row_mapping[r]],
                           double( colvals[j] ) );
            }
         }

         if( integral )
            fmt::print( out,
                        "    MARK0000  'MARKER'                 'INTEND'\n" );
      }

      const Vec<REAL>& lower_bounds = prob.getLowerBounds();
      const Vec<REAL>& upper_bounds = prob.getUpperBounds();

      fmt::print( out, "RHS\n" );

      if( obj.offset != 0 )
      {
         if( obj.offset != REAL{ 0.0 } )
            fmt::print( out, "    B         {: <9} {:.15}\n", "OBJ",
                        double( REAL( -obj.offset ) ) );
      }

      for( int i = 0; i < consmatrix.getNRows(); ++i )
      {
         // discard redundant rows when writing problem
         if( consmatrix.isRowRedundant( i ) )
            continue;

         if( row_flags[i].test( RowFlag::kLhsInf ) &&
             row_flags[i].test( RowFlag::kRhsInf ) )
            continue;

         if( row_flags[i].test( RowFlag::kLhsInf ) )
         {
            if( rhs[i] != REAL{ 0.0 } )
               fmt::print( out, "    B         {: <9} {:.15}\n",
                           consnames[row_mapping[i]], double( rhs[i] ) );
         }
         else
         {
            if( lhs[i] != REAL{ 0.0 } )
               fmt::print( out, "    B         {: <9} {:.15}\n",
                           consnames[row_mapping[i]], double( lhs[i] ) );
         }
      }

      if( hasRangedRow )
      {
         fmt::print( out, "RANGES\n" );
         for( int i = 0; i < consmatrix.getNRows(); ++i )
         {
            if( row_flags[i].test( RowFlag::kLhsInf, RowFlag::kRhsInf,
                                   RowFlag::kEquation, RowFlag::kRedundant ) )
               continue;

            double rangeval = double( REAL( rhs[i] - lhs[i] ) );

            if( rangeval != 0 )
            {
               fmt::print( out, "    B         {: <9} {:.15}\n",
                           consnames[row_mapping[i]], rangeval );
            }
         }
      }

      fmt::print( out, "BOUNDS\n" );

      for( int i = 0; i < consmatrix.getNCols(); ++i )
      {
         if( col_flags[i].test( ColFlag::kInactive ) )
            continue;

         if( !col_flags[i].test( ColFlag::kLbInf ) &&
             !col_flags[i].test( ColFlag::kUbInf ) &&
             lower_bounds[i] == upper_bounds[i] )
         {
            fmt::print( out, " FX BND       {: <9} {:.15}\n",
                        varnames[col_mapping[i]], double( lower_bounds[i] ) );
         }
         else
         {
            if( col_flags[i].test( ColFlag::kLbInf ) || lower_bounds[i] != 0.0 )
            {
               if( col_flags[i].test( ColFlag::kLbInf ) )
                  fmt::print( out, " MI BND       {}\n",
                              varnames[col_mapping[i]] );
               else
                  fmt::print( out, " LO BND       {: <9} {:.15}\n",
                              varnames[col_mapping[i]],
                              double( lower_bounds[i] ) );
            }

            if( !col_flags[i].test( ColFlag::kUbInf ) )
               fmt::print( out, " UP BND       {: <9} {:.15}\n",
                           varnames[col_mapping[i]],
                           double( upper_bounds[i] ) );
            else
               fmt::print( out, " PL BND       {: <9}\n",
                           varnames[col_mapping[i]] );
         }
      }
      fmt::print( out, "ENDATA\n" );
   }
};

} // namespace papilo

#endif
