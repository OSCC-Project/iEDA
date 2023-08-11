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

#include "papilo/Config.hpp"
#include "papilo/core/ConstraintMatrix.hpp"
#include "papilo/core/Objective.hpp"
#include "papilo/core/Presolve.hpp"
#include "papilo/core/VariableDomains.hpp"
#include "papilo/io/MpsParser.hpp"
#include "papilo/io/MpsWriter.hpp"
#include "papilo/io/SolParser.hpp"
#include "papilo/misc/MultiPrecision.hpp"
#include "papilo/misc/OptionsParser.hpp"
#include "papilo/misc/VersionLogger.hpp"
#include "papilo/misc/Wrappers.hpp"
#ifdef PAPILO_TBB
#include "papilo/misc/tbb.hpp"
#endif

#include <boost/program_options.hpp>
#include <cassert>
#include <fstream>

#ifdef PAPILO_HAVE_SCIP

#include "papilo/interfaces/ScipInterface.hpp"
#include "scip/scipdefplugins.h"
static void
setupscip( SCIP* scip, void* usrdata )
{
   papilo::OptionsInfo* optInfo =
       reinterpret_cast<papilo::OptionsInfo*>( usrdata );

   SCIP_CALL_ABORT( SCIPincludeDefaultPlugins( scip ) );

   if( !optInfo->scip_settings_file.empty() )
   {
      SCIP_CALL_ABORT(
          SCIPreadParams( scip, optInfo->scip_settings_file.c_str() ) );
   }
}

template <typename REAL>
static std::unique_ptr<papilo::SolverFactory<REAL>>
get_mip_solver_factory( papilo::OptionsInfo& optionsInfo )
{
   return papilo::ScipFactory<REAL>::create( setupscip, &optionsInfo );
}

#elif defined PAPILO_HAVE_HIGHS

#include "papilo/interfaces/HighsInterface.hpp"

template <typename REAL>
static std::unique_ptr<papilo::SolverFactory<REAL>>
get_mip_solver_factory( papilo::OptionsInfo& optionsInfo )
{
   return papilo::HighsFactory<REAL>::create();
}

#elif defined PAPILO_HAVE_GUROBI

#include "papilo/interfaces/GurobiInterface.hpp"


template <typename REAL>
static std::unique_ptr<papilo::SolverFactory<REAL>>
get_mip_solver_factory( papilo::OptionsInfo& optionsInfo )
{
   return papilo::GurobiFactory<REAL>::create( );
}
#else

template <typename REAL>
static std::unique_ptr<papilo::SolverFactory<REAL>>
get_mip_solver_factory( papilo::OptionsInfo& optionsInfo )
{
   return nullptr;
}

#endif

#if defined( PAPILO_HAVE_SOPLEX )
#include "papilo/interfaces/SoplexInterface.hpp"

static void
setupsoplex( soplex::SoPlex& spx, void* usrdata )
{
   papilo::OptionsInfo* optInfo =
       reinterpret_cast<papilo::OptionsInfo*>( usrdata );

   if( !optInfo->soplex_settings_file.empty() )
      spx.loadSettingsFile( optInfo->soplex_settings_file.c_str() );
}

template <typename REAL>
static std::unique_ptr<papilo::SolverFactory<REAL>>
get_lp_solver_factory( papilo::OptionsInfo& optionsInfo )
{
   return papilo::SoplexFactory<REAL>::create( setupsoplex, &optionsInfo );
}
#elif defined PAPILO_HAVE_GLOP

#include "papilo/interfaces/GlopInterface.hpp"


template <typename REAL>
static std::unique_ptr<papilo::SolverFactory<REAL>>
get_lp_solver_factory( papilo::OptionsInfo& optionsInfo )
{
   return papilo::GlopFactory<REAL>::create( );
}

#else

template <typename REAL>
static std::unique_ptr<papilo::SolverFactory<REAL>>
get_lp_solver_factory( papilo::OptionsInfo& optionsInfo )
{
   return nullptr;
}

#endif

int
main( int argc, char* argv[] )
{
   using namespace papilo;
//#ifdef PAPILO_HAVE_GLOP
//   google::InitGoogleLogging(argv[0]);
//#endif
   print_header();

   // get the options passed by the user
   OptionsInfo optionsInfo;
   try
   {
      optionsInfo = parseOptions( argc, argv );
   }
   catch( const boost::program_options::error& ex )
   {
      std::cerr << "Error while parsing the options.\n" << '\n';
      std::cerr << ex.what() << '\n';
      return 1;
   }

   if( !optionsInfo.is_complete )
      return 0;

   // run the command passed as argument
   switch( optionsInfo.command )
   {
   case Command::kNone:
      return 1;
   case Command::kPresolve:
   case Command::kSolve:
      switch( optionsInfo.arithmetic_type )
      {
      case ArithmeticType::kDouble:
         if( presolve_and_solve<double>(
                 optionsInfo, get_lp_solver_factory<double>( optionsInfo ),
                 get_mip_solver_factory<double>( optionsInfo ) ) !=
             ResultStatus::kOk )
            return 1;
         break;
      case ArithmeticType::kQuad:
         if( presolve_and_solve<Quad>(
                 optionsInfo, get_lp_solver_factory<Quad>( optionsInfo ),
                 get_mip_solver_factory<Quad>( optionsInfo ) ) !=
             ResultStatus::kOk )
            return 1;
         break;
      case ArithmeticType::kRational:
         if( presolve_and_solve<papilo::Rational>(
                 optionsInfo, get_lp_solver_factory<papilo::Rational>( optionsInfo ),
                 get_mip_solver_factory<papilo::Rational>( optionsInfo ) ) !=
             ResultStatus::kOk )
            return 1;
      }
      break;
   case Command::kPostsolve:
      switch( optionsInfo.arithmetic_type )
      {
      case ArithmeticType::kDouble:
         postsolve<double>( optionsInfo );
         break;
      case ArithmeticType::kQuad:
         postsolve<Quad>( optionsInfo );
         break;
      case ArithmeticType::kRational:
         postsolve<papilo::Rational>( optionsInfo );
      }
   }

   return 0;
}
