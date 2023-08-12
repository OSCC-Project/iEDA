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

#ifndef _PAPILO_INTERFACES_SOLVER_INTERFACE_HPP_
#define _PAPILO_INTERFACES_SOLVER_INTERFACE_HPP_

#include "papilo/core/Components.hpp"
#include "papilo/io/Message.hpp"
#include "papilo/misc/ParameterSet.hpp"
#include "papilo/misc/String.hpp"
#include "papilo/misc/Vec.hpp"
#include <memory>
#include <string>

namespace papilo
{

enum class SolverType : int
{
   LP,
   MIP
};

enum class SolverStatus : int
{
   kInit,
   kOptimal,
   kInfeasible,
   kUnbounded,
   kUnbndOrInfeas,
   kInterrupted,
   kError
};

template <typename REAL>
class SolverInterface
{
 protected:
   SolverStatus status;

 public:
   SolverInterface() : status( SolverStatus::kInit ) {}

   virtual void
   setUp( const Problem<REAL>& prob, const Vec<int>& row_maps,
          const Vec<int>& col_maps ) = 0;

   virtual void
   setUp( const Problem<REAL>& prob, const Vec<int>& row_maps,
          const Vec<int>& col_maps, const Components& components,
          const ComponentInfo& component ) = 0;

   virtual void
   solve() = 0;

   virtual SolverType
   getType() = 0;

   virtual String
   getName() = 0;

   virtual void
   printDetails()
   {
   }

   virtual void
   readSettings( const String& file )
   {
   }

   SolverStatus
   getStatus()
   {
      return status;
   }

   virtual void
   setNodeLimit( int num )
   {
   }

   virtual void
   setGapLimit( const REAL& gaplim )
   {
   }

   virtual void
   setSoftTimeLimit( double tlim )
   {
   }

   virtual void
   setTimeLimit( double tlim ) = 0;

   virtual void
   setVerbosity( VerbosityLevel verbosity ) = 0;

   virtual bool
   getSolution( Solution<REAL>& sol ) = 0;

   virtual bool
   getSolution( const Components& components, int component,
                Solution<REAL>& sol ) = 0;

   virtual REAL
   getDualBound() = 0;

   virtual bool
   is_dual_solution_available() = 0;

   virtual void
   addParameters( ParameterSet& paramSet )
   {
   }

   virtual ~SolverInterface() {}
};

template <typename REAL>
class SolverFactory
{
 public:
   virtual std::unique_ptr<SolverInterface<REAL>>
   newSolver( VerbosityLevel verbosity = VerbosityLevel::kQuiet ) const = 0;

   virtual void
   add_parameters( ParameterSet& parameter ) const = 0;

   virtual ~SolverFactory() {}
};

} // namespace papilo

#endif
