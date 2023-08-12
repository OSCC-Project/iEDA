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

#ifndef _PAPILO_IO_SOL_WRITER_HPP_
#define _PAPILO_IO_SOL_WRITER_HPP_

#include "papilo/Config.hpp"
#include "papilo/misc/Vec.hpp"
#include "papilo/misc/fmt.hpp"
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
struct SolWriter
{
   static void
   writePrimalSol( const std::string& filename, const Vec<REAL>& sol,
                   const Vec<REAL>& objective, const REAL& solobj,
                   const Vec<std::string>& colnames )
   {
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

      fmt::print( out, "{: <50} {: <18.15}\n", "=obj=", double( solobj ) );

      for( int i = 0; i != (int) sol.size(); ++i )
      {
         if( sol[i] != 0.0 )
         {
            fmt::print( out, "{: <50} {: <18.15}   obj({:.15})\n", colnames[i],
                        double( sol[i] ), double( objective[i] ) );
         }
      }
   }

   static void
   writeDualSol( const std::string& filename, const Vec<REAL>& sol,
                 const Vec<REAL>& rhs, const Vec<REAL>& lhs,
                 const REAL& obj_value, const Vec<std::string>& row_names )
   {
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

      fmt::print( out, "{: <50} {: <18.15}\n", "=obj=", double( obj_value ) );

      for( int i = 0; i < (int) sol.size(); ++i )
      {
         if( sol[i] != 0.0 )
         {
            REAL objective = lhs[i];
            if( sol[i] < 0 )
               objective = rhs[i];
            fmt::print( out, "{: <50} {: <18.15}   obj({:.15})\n", row_names[i],
                        double( sol[i] ), double( objective ) );
         }
      }
   }

   static void
   writeReducedCostsSol( const std::string& filename, const Vec<REAL>& sol,
                         const Vec<REAL>& ub, const Vec<REAL>& lb,
                         const REAL& solobj, const Vec<std::string>& col_names )
   {
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

      fmt::print( out, "{: <50} {: <18.15}\n", "=obj=", double( solobj ) );

      for( int i = 0; i < (int) sol.size(); ++i )
      {
         if( sol[i] != 0.0 )
         {
            REAL objective = lb[i];
            if( sol[i] < 0 )
               objective = ub[i];
            fmt::print( out, "{: <50} {: <18.15}   obj({:.15})\n", col_names[i],
                        double( sol[i] ), double( objective ) );
         }
      }
   }

   static void
   writeBasis( const std::string& filename, const Vec<VarBasisStatus>& colBasis,
               const Vec<VarBasisStatus>& rowBasis, const Vec<std::string>& col_names, const Vec<std::string>& row_names )
   {
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

      int rowSize = (int) rowBasis.size();
      assert(colBasis.size() == col_names.size());
      assert( rowSize == row_names.size());


      out.push( file );
      int row = 0;
      fmt::print( out, "NAME  papilo.bas\n");

      for( int col = 0; col < (int) colBasis.size(); ++col )
      {
         if( colBasis[col] == VarBasisStatus::BASIC )
         {
            /* Find non basic row */
            for(; row < rowSize; row++)
            {
               if(rowBasis[row] != VarBasisStatus::BASIC)
                  break;
            }
            assert( rowBasis[row] == VarBasisStatus::ON_UPPER ||
                    rowBasis[row] == VarBasisStatus::ON_LOWER ||
                    rowBasis[row] == VarBasisStatus::ZERO ||
                    rowBasis[row] == VarBasisStatus::FIXED
                    );
            if( colBasis[col] == VarBasisStatus::ON_UPPER )
               fmt::print( out, "  XU {: <50} {: <50}\n", col_names[col],
                           row_names[row]);
            else
               fmt::print( out, "  XL {: <50} {: <50}\n", col_names[col],
                           row_names[row]);
            row++;
         }
         else if( colBasis[col] == VarBasisStatus::ON_UPPER )
            fmt::print( out, "  UL {: <50}\n", col_names[col]);
         else if( colBasis[col] == VarBasisStatus::ON_LOWER ||
                  colBasis[col] == VarBasisStatus::ZERO )
         {
            /* Default is all non-basic variables on lower bound (if finite) or
             * at zero (if free). nothing to do in this case.
             */
         }
         else
            assert( false );
      }
      fmt::print( out, "ENDDATA\n");

//      assert(check_if_remaining_rows_are_basic( rowBasis, rowSize, row ));
   }

   static bool
   check_if_remaining_rows_are_basic( const Vec<VarBasisStatus>& rowBasis,
                                      int rowSize, int row )
   {
      // Check that we covered all nonbasic rows - the remaining should be basic.
      for(; row < rowSize; row++)
      {
         if(rowBasis[row] != VarBasisStatus::BASIC)
            break;
      }

      return row == rowSize ;
   }
};

} // namespace papilo

#endif