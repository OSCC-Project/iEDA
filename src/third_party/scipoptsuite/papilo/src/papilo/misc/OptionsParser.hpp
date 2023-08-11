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

#ifndef _PAPILO_MISC_OPTIONS_PARSER_HPP_
#define _PAPILO_MISC_OPTIONS_PARSER_HPP_

#include "papilo/misc/fmt.hpp"
#include <boost/program_options.hpp>
#include <fstream>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace papilo
{

using namespace boost::program_options;

enum class Command
{
   kNone,
   kPresolve,
   kSolve,
   kPostsolve
};

struct ArithmeticType
{
   enum
   {
      kDouble = 'd',
      kQuad = 'q',
      kRational = 'r'
   };
};

struct OptionsInfo
{
   Command command = Command::kNone;
   std::string instance_file;
   std::string reduced_problem_file;
   std::string postsolve_archive_file;
   std::string reduced_solution_file;
   std::string orig_solution_file;
   std::string orig_dual_solution_file;
   std::string orig_reduced_costs_file;
   std::string orig_basis_file;
   std::string scip_settings_file;
   std::string optimal_solution_file;
   std::string soplex_settings_file;
   std::string param_settings_file;
   std::string objective_reference;
   std::vector<std::string> unparsed_options;
   double tlim = std::numeric_limits<double>::max();
   char arithmetic_type = ArithmeticType::kDouble;
   int nthreads;
   bool print_stats;
   bool print_params;
   bool is_complete;

   bool
   checkFiles()
   {
      if( existsFile(instance_file))
      {
         fmt::print( "file {} is not valid\n", instance_file );
         return false;
      }

      if( command == Command::kPostsolve && existsFile(postsolve_archive_file) )
      {
         fmt::print( "file {} is not valid\n", postsolve_archive_file );
         return false;
      }

      if( command == Command::kPostsolve && existsFile(reduced_solution_file)  )
      {
         fmt::print( "file {} is not valid\n", reduced_solution_file );
         return false;
      }

      if( existsFile( scip_settings_file ) )
      {
         fmt::print( "file {} is not valid\n", scip_settings_file );
         return false;
      }

      if( existsFile( optimal_solution_file ) )
      {
         fmt::print( "file {} is not valid\n", optimal_solution_file );
         return false;
      }
      if( existsFile( orig_reduced_costs_file ) )
      {
         fmt::print( "file {} is not valid\n", orig_reduced_costs_file );
         return false;
      }
      if( existsFile( orig_basis_file ) )
      {
         fmt::print( "file {} is not valid\n", orig_basis_file );
         return false;
      }
      if( existsFile( orig_dual_solution_file ) )
      {
         fmt::print( "file {} is not valid\n", orig_dual_solution_file );
         return false;
      }
      if( existsFile( optimal_solution_file ) )
      {
         fmt::print( "file {} is not valid\n", optimal_solution_file );
         return false;
      }


      if( existsFile( soplex_settings_file ))
      {
         fmt::print( "file {} is not valid\n", soplex_settings_file );
         return false;
      }

      if( !print_params && existsFile( param_settings_file ))
      {
         fmt::print( "file {} is not valid\n", param_settings_file );
         return false;
      }

      return true;
   }

   bool
   existsFile( std::string& filename ) const
   {
      return !filename.empty() && !std::ifstream( filename );
   }

   void
   parse( const std::string& commandString,
          const std::vector<std::string>& opts = std::vector<std::string>() )
   {
      is_complete = false;
      if( commandString == "presolve" )
         command = Command::kPresolve;
      else if( commandString == "solve" )
         command = Command::kSolve;
      else if( commandString == "postsolve" )
         command = Command::kPostsolve;
      else
      {
         fmt::print( "unknown command: {}\n", commandString );
         return;
      }

      std::string arithmetic_type_message = fmt::format(
          "'{}' for double precision, '{}' for quad precision, and '{}' "
          "for exact rational arithmetic",
          (char)ArithmeticType::kDouble, (char)ArithmeticType::kQuad,
          (char)ArithmeticType::kRational );

      options_description desc( fmt::format( "{} command", commandString ) );

      desc.add_options()( "file,f", value( &instance_file ), "instance file" );

      desc.add_options()(
          "arithmetic-type,a",
          value( &arithmetic_type )->default_value( ArithmeticType::kDouble ),
          arithmetic_type_message.c_str() );
      desc.add_options()( "postsolve-archive,v",
                          value( &postsolve_archive_file ),
                          "filename for postsolve archive" );

      if( command == Command::kPresolve )
      {
         desc.add_options()( "validate-solution,b",
                             value( &optimal_solution_file ),
                             "optimal solution for validation" );
      }

      if( command != Command::kPostsolve )
      {
         desc.add_options()( "reduced-problem,r",
                             value( &reduced_problem_file ),
                             "filename for reduced problem" );

         desc.add_options()( "parameter-settings,p",
                             value( &param_settings_file ),
                             "filename for presolve parameter settings" );

         desc.add_options()(
             "print-params",
             bool_switch( &print_params )->default_value( false ),
             "print possible parameters presolving" );

         desc.add_options()( "threads,t",
                             value( &nthreads )->default_value( 0 ) );
      }

      if( command != Command::kPresolve )
      {
         desc.add_options()( "reduced-solution,u",
                             value( &reduced_solution_file ),
                             "filename for solution of reduced problem" );
         desc.add_options()( "solution,l", value( &orig_solution_file ),
                             "filename for solution" );
         desc.add_options()( "dualsolution", value( &orig_dual_solution_file ),
                             "filename for dual solution" );
         desc.add_options()( "reducedcosts,c", value( &orig_reduced_costs_file ),
                             "filename for reduced costs" );
         desc.add_options()( "basis,w", value( &orig_basis_file ),
                             "filename for basis information" );
         desc.add_options()( "reference-objective,o",
                             value( &objective_reference ),
                             "correct objective value for validation" );
         desc.add_options()( "validate-solution,b",
                             value( &optimal_solution_file ),
                             "optimal solution for validation" );
      }

      if( command == Command::kSolve )
      {
         desc.add_options()( "scip-settings,s", value( &scip_settings_file ),
                             "SCIP settings file" );
         desc.add_options()( "soplex-settings,x",
                             value( &soplex_settings_file ),
                             "SoPlex settings file" );
         desc.add_options()(
             "tlim", value( &tlim )->default_value( tlim ),
             "time limit for solver (including presolve time)" );
         desc.add_options()(
             "print-stats", bool_switch( &print_stats )->default_value( false ),
             "print detailed solver statistics" );
      }

      if( opts.empty() )
      {
         fmt::print( "\n{}\n", desc );
         return;
      }

      variables_map vm;

      parsed_options parsed = command_line_parser( opts )
                                  .options( desc )
                                  .allow_unregistered()
                                  .run();
      store( parsed, vm );
      notify( vm );

      if( !checkFiles() )
         return;

      if( arithmetic_type != ArithmeticType::kDouble &&
          arithmetic_type != ArithmeticType::kQuad &&
          arithmetic_type != ArithmeticType::kRational )
         fmt::print( "invalid arithmetic type '{}'\nvalid options are {}\n",
                     (char)arithmetic_type, arithmetic_type_message );

      switch( command )
      {
      case Command::kSolve:
      case Command::kPresolve:
         if( instance_file.empty() )
         {
            fmt::print( "{} requires an instance file\n", commandString );
            return;
         }
         break;
      case Command::kPostsolve:
         if( postsolve_archive_file.empty() || reduced_solution_file.empty() )
         {
            fmt::print(
                "{} requires a postsolve archive and a reduced solution\n",
                commandString );
            return;
         }
         break;
      case Command::kNone:
         assert( false );
      }

      unparsed_options =
          collect_unrecognized( parsed.options, exclude_positional );

      is_complete = true;
   }
};

OptionsInfo
parseOptions( int argc, char* argv[] )
{
   OptionsInfo optionsInfo;
   using namespace boost::program_options;
   using boost::none;
   using boost::optional;
   std::string usage =
       fmt::format( "usage:\n {} [COMMAND] [ARGUMENTS]\n", argv[0] );

   // global description.
   // will capture the command and arguments as unrecognised
   options_description global{};
   global.add_options()( "help,h", "produce help message" );
   global.add_options()( "command", value<std::string>(),
                         "command: {presolve, solve, postsolve}." );
   global.add_options()( "args", value<std::vector<std::string>>(),
                         "arguments for the command" );

   positional_options_description pos;
   pos.add( "command", 1 );
   pos.add( "args", -1 );

   parsed_options parsed = command_line_parser( argc, argv )
                               .options( global )
                               .positional( pos )
                               .allow_unregistered()
                               .run();

   variables_map vm;
   store( parsed, vm );

   if( vm.count( "help" ) || vm.empty() )
   {
      fmt::print( "{}\n{}", usage, global );
      optionsInfo.parse( "presolve" );
      optionsInfo.parse( "solve" );
      optionsInfo.parse( "postsolve" );
      return optionsInfo;
   }

   // we branch on each command
   // and parse the arguments passed with the command
   if( vm.count( "command" ) )
   {
      auto command = vm["command"].as<std::string>();

      std::vector<std::string> opts =
          collect_unrecognized( parsed.options, include_positional );

      opts.erase( opts.begin() );

      optionsInfo.parse( command, opts );
   }

   return optionsInfo;
}

} // namespace papilo

#endif
