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

#ifndef _PAPILO_IO_MESSAGE_HPP_
#define _PAPILO_IO_MESSAGE_HPP_

#include "papilo/misc/ParameterSet.hpp"
#include "papilo/misc/fmt.hpp"
#include <cstdio>
#include <type_traits>
#include <utility>

namespace papilo
{

enum class VerbosityLevel : int
{
   kQuiet = 0,
   kError = 1,
   kWarning = 2,
   kInfo = 3,
   kDetailed = 4,
};

struct EnableDebugOutput
{
};

class Message
{
   int verbosity{ static_cast<int>( VerbosityLevel::kInfo ) };

   void ( *write )( int level, const char* data, size_t len,
                    void* usrdata ) = nullptr;
   void* write_usrdata = nullptr;

 public:
   void
   setOutputCallback( void ( *writecb )( int level, const char* data,
                                         size_t len, void* usrdata ),
                      void* writecb_usrdata )
   {
      write = writecb;
      write_usrdata = writecb_usrdata;
   }

   template <typename... Args>
   void
   print( VerbosityLevel level, fmt::string_view format_str,
          Args... args ) const
   {
      fmt::basic_memory_buffer<char, fmt::inline_buffer_size, Allocator<char>>
          buf;
      fmt::vformat_to(
          buf, format_str,
          { fmt::make_format_args( std::forward<Args>( args )... ) } );
      std::size_t size = buf.size();

      if( write != nullptr )
      {
         buf.push_back( '\0' );
         write( static_cast<int>( level ), buf.data(), size,
                const_cast<void*>( write_usrdata ) );
      }
      else
      {
         std::fwrite( buf.data(), 1, size, stdout );
      }
   }

   void
   addParameters( ParameterSet& paramSet )
   {
      paramSet.addParameter( "message.verbosity",
                             "verbosity to be used: 0 - quiet, 1 - errors, 2 - "
                             "warnings, 3 - normal, 4 - detailed",
                             verbosity, 0, 4 );
   }

   void
   setVerbosityLevel( VerbosityLevel value )
   {
      this->verbosity = static_cast<int>( value );
   }

   VerbosityLevel
   getVerbosityLevel() const
   {
      return static_cast<VerbosityLevel>( this->verbosity );
   }

   template <typename... Args>
   void
   detailed( Args&&... args ) const
   {
      switch( static_cast<VerbosityLevel>( verbosity ) )
      {
      case VerbosityLevel::kDetailed:
         print( VerbosityLevel::kDetailed, std::forward<Args>( args )... );
         break;
      case VerbosityLevel::kInfo:
      case VerbosityLevel::kWarning:
      case VerbosityLevel::kError:
      case VerbosityLevel::kQuiet:
         break;
      }
   }

   template <typename... Args>
   void
   info( Args&&... args ) const
   {
      switch( static_cast<VerbosityLevel>( verbosity ) )
      {
      case VerbosityLevel::kDetailed:
      case VerbosityLevel::kInfo:
         print( VerbosityLevel::kInfo, std::forward<Args>( args )... );
         break;
      case VerbosityLevel::kWarning:
      case VerbosityLevel::kError:
      case VerbosityLevel::kQuiet:
         break;
      }
   }

   template <typename... Args>
   void
   warn( Args&&... args ) const
   {
      switch( static_cast<VerbosityLevel>( verbosity ) )
      {
      case VerbosityLevel::kDetailed:
      case VerbosityLevel::kInfo:
      case VerbosityLevel::kWarning:
         print( VerbosityLevel::kWarning, std::forward<Args>( args )... );
         break;
      case VerbosityLevel::kError:
      case VerbosityLevel::kQuiet:
         break;
      }
   }

   template <typename... Args>
   void
   error( Args&&... args ) const
   {
      switch( static_cast<VerbosityLevel>( verbosity ) )
      {
      case VerbosityLevel::kDetailed:
      case VerbosityLevel::kInfo:
      case VerbosityLevel::kWarning:
      case VerbosityLevel::kError:
         print( VerbosityLevel::kError, std::forward<Args>( args )... );
         break;
      case VerbosityLevel::kQuiet:
         break;
      }
   }

   // select with SFINAE to print debug output or depending on whether the class
   // type of the pointer inherits from the marker struct EnableDebugOutput

   template <typename T, typename... Args,
             typename std::enable_if<
                 !std::is_base_of<EnableDebugOutput, T>::value, int>::type = 0>
   static void
   debug( const T*, Args&&... args )
   {
   }

   template <typename T, typename... Args,
             typename std::enable_if<
                 std::is_base_of<EnableDebugOutput, T>::value, int>::type = 0>
   static void
   debug( const T*, Args&&... args )
   {
      fmt::print( std::forward<Args>( args )... );
   }
};

} // namespace papilo

#endif
