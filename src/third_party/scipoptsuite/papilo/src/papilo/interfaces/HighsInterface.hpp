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

#ifndef _PAPILO_INTERFACES_HIGHS_INTERFACE_HPP_
#define _PAPILO_INTERFACES_HIGHS_INTERFACE_HPP_

#include "papilo/misc/String.hpp"
#include "papilo/misc/Vec.hpp"
#include <cassert>
#include <stdexcept>
#include <type_traits>

#include "Highs.h"
#include "papilo/core/Problem.hpp"
#include "papilo/interfaces/SolverInterface.hpp"

namespace papilo
{

template <typename REAL>
class HighsInterface : public SolverInterface<REAL>
{
 private:
   Highs solver;
   HighsOptions opts;
   static constexpr double inf = std::numeric_limits<double>::infinity();

 public:
   HighsInterface() {}

   void
   setTimeLimit( double tlim ) override
   {
      opts.time_limit = tlim;
   }

   void
   setUp( const Problem<REAL>& problem, const Vec<int>& row_maps,
          const Vec<int>& col_maps ) override
   {
      int ncols = problem.getNCols();
      int nrows = problem.getNRows();
      const Vec<String>& varNames = problem.getVariableNames();
      const Vec<String>& consNames = problem.getConstraintNames();
      const VariableDomains<REAL>& domains = problem.getVariableDomains();
      const Objective<REAL>& obj = problem.getObjective();
      const ConstraintMatrix<REAL>& consMatrix = problem.getConstraintMatrix();
      const auto& lhs_values = consMatrix.getLeftHandSides();
      const auto& rhs_values = consMatrix.getRightHandSides();
      const auto& rflags = problem.getRowFlags();

      HighsLp model;

      model.sense_ = ObjSense::kMinimize;
      model.offset_ = double( obj.offset );

      model.num_row_ = nrows;
      model.num_col_ = ncols;

      model.col_cost_.resize( ncols );
      model.col_lower_.resize( ncols );
      model.col_upper_.resize( ncols );
      model.integrality_.resize( ncols );

      model.row_lower_.resize( nrows );
      model.row_upper_.resize( nrows );

      for( int i = 0; i != nrows; ++i )
      {
         model.row_lower_[i] = rflags[i].test( RowFlag::kLhsInf )
                                  ? -inf
                                  : double( lhs_values[i] );
         model.row_upper_[i] =
             rflags[i].test( RowFlag::kRhsInf ) ? inf : double( rhs_values[i] );
      }

      model.a_index_.resize( consMatrix.getNnz() );
      model.a_value_.resize( consMatrix.getNnz() );
      model.a_start_.resize( ncols + 1 );
      model.a_start_[ncols] = consMatrix.getNnz();

      int start = 0;

      for( int i = 0; i < ncols; ++i )
      {
         assert( !domains.flags[i].test( ColFlag::kInactive ) );

         model.col_lower_[i] = domains.flags[i].test( ColFlag::kLbInf )
                                  ? -inf
                                  : double( domains.lower_bounds[i] );
         model.col_upper_[i] = domains.flags[i].test( ColFlag::kUbInf )
                                  ? inf
                                  : double( domains.upper_bounds[i] );

         model.col_cost_[i] = double( obj.coefficients[i] );

         model.integrality_[i] =
             domains.flags[i].test( ColFlag::kImplInt )
                 ? HighsVarType::kImplicitInteger
                 : ( domains.flags[i].test( ColFlag::kIntegral )
                         ? HighsVarType::kInteger
                         : HighsVarType::kContinuous );

         auto colvec = consMatrix.getColumnCoefficients( i );

         int collen = colvec.getLength();
         const int* colrows = colvec.getIndices();
         const REAL* colvals = colvec.getValues();

         model.a_start_[i] = start;

         for( int k = 0; k != collen; ++k )
         {
            model.a_value_[start + k] = double( colvals[k] );
            model.a_index_[start + k] = colrows[k];
         }

         start += collen;
      }

      solver.passModel( std::move( model ) );
   }

   void
   setUp( const Problem<REAL>& problem, const Vec<int>& row_maps,
          const Vec<int>& col_maps, const Components& components,
          const ComponentInfo& component ) override
   {
      const Vec<String>& varNames = problem.getVariableNames();
      const Vec<String>& consNames = problem.getConstraintNames();
      const VariableDomains<REAL>& domains = problem.getVariableDomains();
      const Objective<REAL>& obj = problem.getObjective();
      const auto& consMatrix = problem.getConstraintMatrix();
      const auto& lhs_values = consMatrix.getLeftHandSides();
      const auto& rhs_values = consMatrix.getRightHandSides();
      const auto& rflags = problem.getRowFlags();
      const int* rowset = components.getComponentsRows( component.componentid );
      const int* colset = components.getComponentsCols( component.componentid );
      int numrows = components.getComponentsNumRows( component.componentid );
      int numcols = components.getComponentsNumCols( component.componentid );

      HighsLp model;

      /* set the objective sense and offset */
      model.sense_ = ObjSense::kMinimize;
      model.offset_ = 0;

      model.num_row_ = numrows;
      model.num_col_ = numcols;

      model.col_cost_.resize( numcols );
      model.col_lower_.resize( numcols );
      model.col_upper_.resize( numcols );
      model.integrality_.resize( numcols );

      model.row_lower_.resize( numrows );
      model.row_upper_.resize( numrows );

      for( int i = 0; i != numrows; ++i )
      {
         int row = rowset[i];

         assert( components.getRowComponentIdx( row ) == i );

         model.row_lower_[i] = rflags[row].test( RowFlag::kLhsInf )
                                  ? -inf
                                  : double( lhs_values[row] );
         model.row_upper_[i] = rflags[row].test( RowFlag::kRhsInf )
                                  ? inf
                                  : double( rhs_values[row] );
      }

      model.a_index_.resize( component.nnonz );
      model.a_value_.resize( component.nnonz );
      model.a_start_.resize( numcols + 1 );
      model.a_start_[numcols] = component.nnonz;

      int start = 0;

      for( int i = 0; i != numcols; ++i )
      {
         int col = colset[i];

         assert( components.getColComponentIdx( col ) == i );
         assert( !domains.flags[col].test( ColFlag::kInactive ) );

         model.col_lower_[i] = domains.flags[col].test( ColFlag::kLbInf )
                                  ? -inf
                                  : double( domains.lower_bounds[col] );
         model.col_upper_[i] = domains.flags[col].test( ColFlag::kUbInf )
                                  ? inf
                                  : double( domains.upper_bounds[col] );

         model.col_cost_[i] = double( obj.coefficients[col] );

         model.integrality_[i] =
             domains.flags[col].test( ColFlag::kImplInt )
                 ? HighsVarType::kImplicitInteger
                 : ( domains.flags[col].test( ColFlag::kIntegral )
                         ? HighsVarType::kInteger
                         : HighsVarType::kContinuous );

         auto colvec = consMatrix.getColumnCoefficients( col );

         int collen = colvec.getLength();
         const int* colrows = colvec.getIndices();
         const REAL* colvals = colvec.getValues();

         model.a_start_[i] = start;

         for( int k = 0; k != collen; ++k )
         {
            model.a_value_[start + k] = double( colvals[k] );
            model.a_index_[start + k] =
                components.getRowComponentIdx( colrows[k] );
         }

         start += collen;
      }

      solver.passModel( std::move( model ) );
   }

   void
   solve() override
   {
      solver.passHighsOptions( opts );

      if( solver.run() == HighsStatus::kError )
      {
         this->status = SolverStatus::kError;
         return;
      }

      HighsModelStatus stat = solver.getModelStatus();

      switch( stat )
      {
      default:
         this->status = SolverStatus::kError;
         return;
      case HighsModelStatus::kInfeasible:
         this->status = SolverStatus::kInfeasible;
         return;
      case HighsModelStatus::kUnbounded:
         this->status = SolverStatus::kUnbounded;
         return;
      case HighsModelStatus::kTimeLimit:
      case HighsModelStatus::kIterationLimit:
         this->status = SolverStatus::kInterrupted;
         return;
      case HighsModelStatus::kOptimal:
         this->status = SolverStatus::kOptimal;
      }
   }

   void
   setVerbosity( VerbosityLevel verbosity ) override
   {
      switch( verbosity )
      {
      case VerbosityLevel::kQuiet:
         opts.output_flag = false;
         solver.setHighsOptionValue( "output_flag", false );
         break;
      case VerbosityLevel::kError:
      case VerbosityLevel::kWarning:
      case VerbosityLevel::kInfo:
      case VerbosityLevel::kDetailed:
         opts.output_flag = true;
         solver.setHighsOptionValue( "output_flag", true );
      }
   }

   REAL
   getDualBound() override
   {

      //      TODO:
      //      if( this->status == SolverStatus::kOptimal )
      //         return -inf;
      return solver.getHighsInfo().mip_dual_bound;
   }

   bool
   getSolution( Solution<REAL>& sol ) override
   {
      const HighsSolution& highsSol = solver.getSolution();
      int numcols = solver.getNumCols();
      int numrows = solver.getNumRows();

      if( highsSol.col_value.size() != numcols )
         return false;

      // get primal values
      sol.primal.resize( numcols );
      for( int i = 0; i < numcols; ++i )
         sol.primal[i] = highsSol.col_value[i];

      // return if no dual requested
      if( sol.type == SolutionType::kPrimal )
         return true;

      if( highsSol.col_dual.size() != numcols ||
          highsSol.row_dual.size() != numrows )
      {
         sol.type = SolutionType::kPrimal;
         return true;
      }

      // get reduced costs
      sol.reducedCosts.resize( numcols );
      for( int i = 0; i < numcols; ++i )
         sol.reducedCosts[i] = REAL( highsSol.col_dual[i] );

      // get row duals
      sol.dual.resize( numrows );
      for( int i = 0; i < numrows; ++i )
         sol.dual[i] = - REAL( highsSol.row_dual[i] );

      sol.basisAvailabe = false;
      sol.varBasisStatus.resize( numcols, VarBasisStatus::UNDEFINED );
      sol.rowBasisStatus.resize( numrows, VarBasisStatus::UNDEFINED );

      return true;
   }

   bool
   getSolution( const Components& components, int component,
                Solution<REAL>& sol ) override
   {
      if( this->status != SolverStatus::kOptimal )
         return false;

      int numcols = solver.getNumCols();
      int numrows = solver.getNumRows();
      const HighsSolution& highsSol = solver.getSolution();
      if( highsSol.col_value.size() != numcols )
         return false;

      assert( components.getComponentsNumCols( component ) == numcols );

      const int* compcols = components.getComponentsCols( component );
      for( int i = 0; i != numcols; ++i )
         sol.primal[compcols[i]] = REAL( highsSol.col_value[i] );

      if( sol.type == SolutionType::kPrimal )
         return true;

      if( highsSol.col_dual.size() != numcols ||
          highsSol.row_dual.size() != numrows )
      {
         sol.type = SolutionType::kPrimal;
         return true;
      }

      for( int i = 0; i != numcols; ++i )
         sol.reducedCosts[compcols[i]] = REAL( highsSol.col_dual[i] );

      const int* comprows = components.getComponentsRows( component );

      for( int i = 0; i != numrows; ++i )
         sol.dual[comprows[i]] = REAL( highsSol.row_dual[i] );

      return true;
   }

   void
   addParameters( ParameterSet& paramSet ) override
   {
   }

   bool
   is_dual_solution_available() override
   {
      return false;
   }

   SolverType
   getType() override
   {
      return SolverType::MIP;
   }

   String
   getName() override
   {
      return "HiGHS";
   }

   void
   printDetails() override
   {
   }
};

template <typename REAL>
class HighsFactory : public SolverFactory<REAL>
{
   HighsFactory() = default;

 public:
   std::unique_ptr<SolverInterface<REAL>>
   newSolver( VerbosityLevel verbosity ) const override
   {
      auto highs =
          std::unique_ptr<SolverInterface<REAL>>( new HighsInterface<REAL>() );

      highs->setVerbosity( verbosity );

      return std::move( highs );
   }

   virtual void
   add_parameters( ParameterSet& parameter ) const
   {
   }
   static std::unique_ptr<SolverFactory<REAL>>
   create()
   {
      return std::unique_ptr<SolverFactory<REAL>>( new HighsFactory<REAL>() );
   }
};

} // namespace papilo

#endif
