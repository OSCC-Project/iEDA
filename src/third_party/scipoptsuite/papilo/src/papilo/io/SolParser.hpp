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

#ifndef _PAPILO_IO_SOL_PARSER_HPP_
#define _PAPILO_IO_SOL_PARSER_HPP_

#include "papilo/misc/Hash.hpp"
#include "papilo/misc/String.hpp"
#include "papilo/misc/Vec.hpp"
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>

namespace papilo
{

template <typename REAL>
struct SolParser
{

   static bool
   read( const std::string& filename, const Vec<int>& origcol_mapping,
         const Vec<String>& colnames, Solution<REAL>& sol)
   {
      std::ifstream file( filename, std::ifstream::in );
      boost::iostreams::filtering_istream in;

      if( !file )
         return false;

#ifdef PAPILO_USE_BOOST_IOSTREAMS_WITH_ZLIB
      if( boost::algorithm::ends_with( filename, ".gz" ) )
         in.push( boost::iostreams::gzip_decompressor() );
#endif

#ifdef PAPILO_USE_BOOST_IOSTREAMS_WITH_BZIP2
      if( boost::algorithm::ends_with( filename, ".bz2" ) )
      in.push( boost::iostreams::bzip2_decompressor() );
#endif

      in.push( file );

      HashMap<String, int> nameToCol;

      for( size_t i = 0; i != origcol_mapping.size(); ++i )
      {
         int origcol = origcol_mapping[i];
         nameToCol.emplace( colnames[origcol], i );
      }

      sol.primal.resize( origcol_mapping.size(), REAL{ 0 } );
      String strline;

      skip_header( colnames, in, strline );

      do
      {
         auto tokens = split( strline.c_str() );
         assert( !tokens.empty() );

         auto it = nameToCol.find( tokens[0] );
         if( it != nameToCol.end() )
         {
            assert( tokens.size() > 1 );
            sol.primal[it->second] = std::stod( tokens[1] );
         }
         else if(strline.empty()){}
         else
         {
            fmt::print( stderr,
                        "WARNING: skipping unknown column {} in solution\n",
                        tokens[0] );
         }
      } while( getline( in, strline ) );

      return true;
   }

 private:

   static void
   skip_header( const Vec<String>& colnames,
                boost::iostreams::filtering_istream& filteringIstream,
                String& strline )
   {
      while(getline( filteringIstream, strline ))
      {
         for(const auto & colname : colnames)
         {
            if( strline.rfind( colname ) == 0 )
               return;
         }
      }
   }

   Vec<String> static split( const char* str )
   {
      Vec<String> tokens;
      char c1 = ' ';
      char c2 = '\t';

      do
      {
         const char* begin = str;

         while( *str != c1 && *str != c2 && *str )
            str++;

         tokens.emplace_back( begin, str );

         while( ( *str == c1 || *str == c2 ) && *str )
            str++;

      } while( 0 != *str );

      return tokens;
   }
};

} // namespace papilo

#endif
