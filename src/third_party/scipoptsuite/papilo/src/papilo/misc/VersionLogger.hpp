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

#ifndef _PAPILO_MISC_VERSION_LOGGER_HPP_
#define _PAPILO_MISC_VERSION_LOGGER_HPP_

#include "papilo/core/Presolve.hpp"
#include "papilo/core/postsolve/Postsolve.hpp"
#include "papilo/io/MpsParser.hpp"
#include "papilo/io/MpsWriter.hpp"
#include "papilo/io/SolParser.hpp"
#include "papilo/io/SolWriter.hpp"
#include "papilo/misc/NumericalStatistics.hpp"
#include "papilo/misc/OptionsParser.hpp"
#ifdef PAPILO_TBB
#include "papilo/misc/tbb.hpp"
#endif

#ifdef PAPILO_HAVE_SOPLEX
#include "soplex.h"
#include "soplex/spxgithash.h"
#endif

#ifdef PAPILO_HAVE_SCIP
#include "scip/scipgithash.h"
#endif


#ifdef PAPILO_HAVE_GLOP
#include "ortools/base/version.h"
#endif

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <string>
#include <utility>

namespace papilo
{

void
join( const Vec<std::string>& v, char c, std::string& s )
{

   s.clear();

   for( auto p = v.begin(); p != v.end(); ++p )
   {
      s += *p;
      if( p != v.end() - 1 )
         s += c;
   }
}

void
print_header()
{
   std::string mode = "optimized";
#ifndef NDEBUG
   mode = "debug";
#endif

   Vec<std::string> list_of_solvers;
   std::string solvers = "";
#ifdef PAPILO_HAVE_HIGHS
   list_of_solvers.push_back( "HiGHS" );
#endif
#ifdef PAPILO_HAVE_SCIP
   list_of_solvers.push_back( "SCIP" );
#endif
#ifdef PAPILO_HAVE_GUROBI
   list_of_solvers.push_back( "Gurobi" );
#endif
#ifdef PAPILO_HAVE_GLOP
   list_of_solvers.push_back( "Ortools" );
#endif
#ifdef PAPILO_HAVE_SOPLEX
   list_of_solvers.push_back( "SoPlex" );
#endif
   if( list_of_solvers.empty() )
      solvers = "none";
   else
      join( list_of_solvers, ',', solvers );
#ifdef PAPILO_GITHASH_AVAILABLE
   fmt::print( "PaPILO version {}.{}.{} [mode: {}][Solvers: {}][GitHash: {}]\n",
               PAPILO_VERSION_MAJOR, PAPILO_VERSION_MINOR, PAPILO_VERSION_PATCH,
               mode, solvers, PAPILO_GITHASH );
#else
   fmt::print( "PaPILO version {}.{}.{} [mode: {}][Solvers: {}][GitHash: ]\n",
               PAPILO_VERSION_MAJOR, PAPILO_VERSION_MINOR, PAPILO_VERSION_PATCH,
               mode, solvers );
#endif
   fmt::print( "Copyright (C) 2020-2022 Zuse Institute Berlin (ZIB)\n" );
   fmt::print( "\n" );

   fmt::print( "External libraries: \n" );

#ifdef BOOST_FOUND
   fmt::print( "  Boost    {}.{}.{} \t (https://www.boost.org/)\n",
               BOOST_VERSION_NUMBER_MINOR( BOOST_VERSION ),
               BOOST_VERSION_NUMBER_PATCH( BOOST_VERSION ) / 100,
               BOOST_VERSION_NUMBER_MAJOR( BOOST_VERSION ) );
#endif
#ifdef PAPILO_TBB
   // TODO: TBB is missing not able to retrieve version
   fmt::print( "  TBB            \t Thread building block https://github.com/oneapi-src/oneTBB developed by Intel\n");
#endif
#ifdef PAPILO_HAVE_GMP
   fmt::print( "  GMP      {}  \t GNU Multiple Precision Arithmetic Library "
               "developed by T. Granlund (gmplib.org)\n",
               GMP_VERSION );
#endif
#ifdef PAPILO_HAVE_HIGHS
   // TODO: add Highs Solver -> waiting for official release
    fmt::print( "  HiGHS   {} \t high performance software "
               "for linear optimization (https://www.maths.ed.ac.uk/hall/HiGHS/) [GitHash: {}]\n" , "pre-release",
    "TBD");
#endif
    //TODO
#ifdef PAPILO_HAVE_GLOP
    fmt::print( "  ORTOOLS  {}.{}   \t fast and portable software for combinatorial optimization developed by Google.\n" ,
                operations_research::OrToolsMajorVersion(), operations_research::OrToolsMinorVersion() );
#endif

#ifdef PAPILO_HAVE_SCIP
   fmt::print( "  SCIP     {}.{}.{} \t Mixed Integer Programming Solver "
               "developed at Zuse "
               "Institute Berlin (scip.zib.de) [GitHash: {}]\n",
               SCIP_VERSION_MAJOR, SCIP_VERSION_MINOR, SCIP_VERSION_PATCH,
               SCIPgetGitHash() );
#endif
#ifdef PAPILO_HAVE_GUROBI
   fmt::print( "  Gurobi            \t Gurobi Optimizer \n" );
#endif
#ifdef PAPILO_HAVE_SOPLEX
   fmt::print(
       "  SoPlex   {}.{}.{} \t Linear Programming Solver developed at Zuse "
       "Institute Berlin (soplex.zib.de) [GitHash: {}]\n",
       SOPLEX_VERSION / 100, ( SOPLEX_VERSION % 100 ) / 10, SOPLEX_VERSION % 10,
       soplex::getGitHash() );
#endif


   fmt::print( "\n" );
}



} // namespace papilo

#endif
