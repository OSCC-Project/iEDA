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

#ifndef _PAPILO_INTERFACES_GLOP_INTERFACE_HPP_
#define _PAPILO_INTERFACES_GLOP_INTERFACE_HPP_

#include "papilo/misc/Vec.hpp"
#include <cassert>
#include <stdexcept>

#include "ortools/linear_solver/linear_solver.h"
#include "papilo/core/Problem.hpp"
#include "papilo/core/Solution.hpp"
#include "papilo/interfaces/SolverInterface.hpp"

namespace papilo
{

using operations_research::MPObjective;
using operations_research::MPSolver;
using operations_research::MPConstraint;
using operations_research::MPVariable;

template <typename REAL>
class GlopInterface : public SolverInterface<REAL>
{
 private:
   int n_cols{};
   Vec<MPVariable*> variables;
   std::unique_ptr<MPSolver> solver;
   std::string solver_id = "PDLP";

   int
   doSetUp( const Problem<REAL>& problem, const Vec<int>& origRowMap,
            const Vec<int>& origColMap )
   {
      if( !std::is_same<REAL, double>::value )
      {
         fmt::print( "Please use double precision when solving with "
                     "OR-TOOLS." );
         return -1;
      }
      auto domains = problem.getVariableDomains();
      n_cols = problem.getNCols();

      //TODO: enable log output for PDLP
      solver = std::unique_ptr<MPSolver>( MPSolver::CreateSolver( solver_id ) );
      solver->EnableOutput();
      const double infinity = MPSolver::infinity();

      auto& row_flags = problem.getRowFlags();
      auto lhs = problem.getConstraintMatrix().getLeftHandSides();
      auto rhs = problem.getConstraintMatrix().getRightHandSides();
      const Vec<String>& varNames = problem.getVariableNames();
      auto coefficients = problem.getObjective().coefficients;

      variables = Vec<MPVariable*>{};
      variables.reserve(problem.getNRows());
      MPObjective* const objective = solver->MutableObjective();
      for( int i = 0; i < problem.getNCols(); i++ )
      {
         double lb = domains.flags[i].test( ColFlag::kLbInf )
                         ? -infinity
                         : domains.lower_bounds[i];
         double ub = domains.flags[i].test( ColFlag::kUbInf )
                         ? infinity
                         : domains.upper_bounds[i];
         variables.push_back(solver->MakeNumVar( lb, ub, varNames[origColMap[i]]));
         objective->SetCoefficient( variables[i], (double)coefficients[i] );
      }
      assert( solver->NumVariables() == problem.getNCols() );

      for( int i = 0; i < problem.getNRows(); i++ )
      {
         auto data = problem.getConstraintMatrix().getRowCoefficients( i );
         double l = row_flags[i].test( RowFlag::kLhsInf ) ? -infinity : lhs[i];
         double r = row_flags[i].test( RowFlag::kRhsInf ) ? infinity : rhs[i];

         MPConstraint* const con = solver->MakeRowConstraint( l, r );
         for( int j = 0; j < data.getLength(); j++ )
         {
            double coeff = data.getValues()[j];
            int index = data.getIndices()[j];
            con->SetCoefficient( variables[index], coeff );
         }
      }
      assert( solver->NumConstraints() == problem.getNRows() );

      objective->SetMinimization();

      return 0;
   }

   int
   doSetUp( const Problem<REAL>& problem, const Vec<int>& origRowMap,
            const Vec<int>& origColMap, const Components& components,
            const ComponentInfo& component )
   {
      if( !std::is_same<REAL, double>::value )
      {
         fmt::print( "Please use double precision when solving with "
                     "OR-TOOLS." );
         return -1;
      }
      int ncols = components.getComponentsNumCols( component.componentid );
      int nrows = components.getComponentsNumRows( component.componentid );
      const int* colset = components.getComponentsCols( component.componentid );
      const int* rowset = components.getComponentsRows( component.componentid );

      const Vec<String>& varNames = problem.getVariableNames();
      const Vec<String>& rowNames = problem.getConstraintNames();
      const VariableDomains<REAL>& domains = problem.getVariableDomains();
      const Vec<REAL>& obj = problem.getObjective().coefficients;
      const Vec<REAL>& rhs = problem.getConstraintMatrix().getRightHandSides();
      const Vec<REAL>& lhs = problem.getConstraintMatrix().getLeftHandSides();
      const auto consMatrix = problem.getConstraintMatrix();
      auto coefficients = problem.getObjective().coefficients;
      auto& row_flags = problem.getRowFlags();

      n_cols = problem.getNCols();

      solver = std::unique_ptr<MPSolver>(MPSolver::CreateSolver( solver_id ));
      solver->EnableOutput();
      const double infinity = MPSolver::infinity();


      variables = Vec<MPVariable*>{};
      variables.reserve(problem.getNRows());

      MPObjective* const objective = solver->MutableObjective();
      for( int j = 0; j < ncols; j++ )
      {
         int col = colset[j];
         double lb = domains.flags[col].test( ColFlag::kLbInf )
                         ? -infinity
                         : domains.lower_bounds[col];
         double ub = domains.flags[col].test( ColFlag::kUbInf )
                         ? infinity
                         : domains.upper_bounds[col];
         variables[col] = solver->MakeNumVar( lb, ub, varNames[origColMap[col]] );
         objective->SetCoefficient( variables[col], (double)coefficients[col] );
      }
      assert( solver->NumVariables() == ncols );

      for( int i = 0; i < nrows; i++ )
      {
         int row = rowset[i];
         auto data = problem.getConstraintMatrix().getRowCoefficients( row );
         double l = row_flags[row].test( RowFlag::kLhsInf ) ? -infinity : lhs[row];
         double r = row_flags[row].test( RowFlag::kRhsInf ) ? infinity : rhs[row];

         MPConstraint* const con = solver->MakeRowConstraint( l, r );
         for( int j = 0; j < data.getLength(); j++ )
         {
            double coeff = data.getValues()[j];
            int index = components.getColComponentIdx( data.getIndices()[j]);
            con->SetCoefficient( variables[index], coeff );
         }
      }
      assert( solver->NumConstraints() == nrows );
      objective->SetMinimization();
      return 0;
   }

 public:
   GlopInterface() =default;

   void
   setUp( const Problem<REAL>& prob, const Vec<int>& row_maps,
          const Vec<int>& col_maps, const Components& components,
          const ComponentInfo& component ) override
   {
      if( doSetUp( prob, row_maps, col_maps, components, component ) != 0 )
         this->status = SolverStatus::kError;
   }

   void
   setNodeLimit( int num ) override
   {
   }

   void
   setGapLimit( const REAL& gaplim ) override
   {
   }

   void
   setSoftTimeLimit( double tlim ) override
   {
   }

   void
   setTimeLimit( double tlim ) override
   {
      solver->SetTimeLimit(absl::Seconds(tlim));
   }

   void
   setVerbosity( VerbosityLevel verbosity ) override
   {
   }

   void
   setUp( const Problem<REAL>& prob, const Vec<int>& row_maps,
          const Vec<int>& col_maps ) override
   {
      if( doSetUp( prob, row_maps, col_maps ) != 0 )
         this->status = SolverStatus::kError;
   }

   void
   solve() override
   {
      MPSolver::ResultStatus result_status = solver->Solve();

      assert( this->status != SolverStatus::kError );

      switch( result_status )
      {
      case MPSolver::INFEASIBLE:
         this->status = SolverStatus::kInfeasible;
         break;
      case MPSolver::UNBOUNDED:
         this->status = SolverStatus::kUnbounded;
         break;
      case MPSolver::OPTIMAL:
         this->status = SolverStatus::kOptimal;
         break;
      case MPSolver::FEASIBLE:
      case MPSolver::NOT_SOLVED:
         this->status = SolverStatus::kInterrupted;
         break;
      case MPSolver::ABNORMAL:
      case MPSolver::MODEL_INVALID:
         this->status = SolverStatus::kError;
         break;
      }
   }

   REAL
   getDualBound() override
   {
      return 0;
   }

   bool
   is_dual_solution_available() override
   {
      return false;
   }


   bool
   getSolution( Solution<REAL>& solbuffer ) override
   {
      Vec<REAL> primal{};
      for( int i = 0; i < n_cols; i++ )
         primal.push_back( variables[i]->solution_value() );
      solbuffer = Solution<REAL>( primal );
      return true;
   }

   bool
   getSolution( const Components& components, int component,
                Solution<REAL>& solbuffer ) override
   {
      const int* colset = components.getComponentsCols( component );

      for( std::size_t i = 0; i < n_cols; ++i )
         solbuffer.primal[colset[i]] = variables[i]->solution_value();

      return true;
   }

   SolverType
   getType() override
   {
      return SolverType::MIP;
   }

   String
   getName() override
   {
      return "or-tools";
   }

   void
   printDetails() override
   {
   }

   void
   addParameters( ParameterSet& paramSet ) override
   {
      paramSet.addParameter( "ortools.solver_id", "solver id", solver_id );
   }

   ~GlopInterface() = default;
};

template <typename REAL>
class GlopFactory : public SolverFactory<REAL>
{

   GlopFactory() {
      solver = new GlopInterface<REAL>();
   }

   GlopInterface<REAL>* solver;
 public:
   virtual std::unique_ptr<SolverInterface<REAL>>
   newSolver( VerbosityLevel verbosity ) const
   {
      auto glop = std::unique_ptr<SolverInterface<REAL>>( solver );
      return std::move( glop );
   }

   virtual void
   add_parameters( ParameterSet& parameter ) const
   {
      //TODO: configure parameters of GLOP
      solver->addParameters( parameter );
   }

   static std::unique_ptr<SolverFactory<REAL>>
   create()
   {
      return std::unique_ptr<SolverFactory<REAL>>( new GlopFactory<REAL>() );
   }
};

} // namespace papilo

#endif
