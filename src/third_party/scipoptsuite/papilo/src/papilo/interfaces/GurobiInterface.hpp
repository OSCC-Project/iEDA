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

#ifndef _PAPILO_INTERFACES_GUROBI_INTERFACE_HPP_
#define _PAPILO_INTERFACES_GUROBI_INTERFACE_HPP_

#include "papilo/misc/Vec.hpp"
#include <cassert>
#include <stdexcept>

#include "gurobi_c++.h"
#include "papilo/core/Problem.hpp"
#include "papilo/core/Solution.hpp"
#include "papilo/interfaces/SolverInterface.hpp"

namespace papilo
{

template <typename REAL>
class GurobiInterface : public SolverInterface<REAL>
{
 private:
   int grb_status;
   GRBEnv* env;
   Solution<REAL> sol;

   int
   doSetUp( const Problem<REAL>& problem, const Vec<int>& origRowMap,
            const Vec<int>& origColMap )
   {
      if( !std::is_same<REAL, double>::value )
      {
         fmt::print( "Please use double precision when solving with "
                     "Gurobi." );
         return -1;
      }

      const Vec<String>& varNames = problem.getVariableNames();
      const Vec<String>& rowNames = problem.getConstraintNames();
      const VariableDomains<REAL>& domains = problem.getVariableDomains();
      const Vec<REAL>& obj = problem.getObjective().coefficients;
      const Vec<REAL>& rhs = problem.getConstraintMatrix().getRightHandSides();
      const Vec<REAL>& lhs = problem.getConstraintMatrix().getLeftHandSides();
      const auto consMatrix = problem.getConstraintMatrix();

      env = new GRBEnv();
      GRBModel model = GRBModel( *env );
      model.set( GRB_StringAttr_ModelName, problem.getName() );
      model.set( GRB_IntAttr_ModelSense, GRB_MINIMIZE );

      int ncols = problem.getNCols();
      GRBVar* vars = new GRBVar[ncols];

      for( int i = 0; i < ncols; ++i )
      {
         assert( !domains.flags[i].test( ColFlag::kInactive ) );

         double lb = domains.flags[i].test( ColFlag::kLbInf )
                         ? -GRB_INFINITY
                         : domains.lower_bounds[i];
         double ub = domains.flags[i].test( ColFlag::kUbInf )
                         ? GRB_INFINITY
                         : domains.upper_bounds[i];

         char type;
         if( domains.flags[i].test( ColFlag::kIntegral ) )
         {
            if( lb == 0 && ub == 1 )
               type = GRB_BINARY;
            else
               type = GRB_INTEGER;
         }
         else if( domains.flags[i].test( ColFlag::kImplInt ) )
            type = GRB_INTEGER;
         else
            type = GRB_CONTINUOUS;

         vars[i] = model.addVar( lb, ub, double( obj[i] ), type,
                                 varNames[origColMap[i]].c_str() );
         model.update();
      }

      for( int i = 0; i < problem.getNRows(); ++i )
      {

         auto row_coeff = problem.getConstraintMatrix().getRowCoefficients( i );
         GRBLinExpr grbLinExpr = 0;
         for( int j = 0; j < row_coeff.getLength(); ++j )
            grbLinExpr += vars[row_coeff.getIndices()[j]] *
                          double( row_coeff.getValues()[j] );
         if( consMatrix.getRowFlags()[i].test( RowFlag::kEquation ) )
         {
            model.addConstr( grbLinExpr, GRB_EQUAL, double( rhs[i] ),
                             rowNames[origRowMap[i]] );
            continue;
         }
         if( !consMatrix.getRowFlags()[i].test( RowFlag::kLhsInf ) )
            model.addConstr( grbLinExpr, GRB_GREATER_EQUAL, double( lhs[i] ),
                             rowNames[origRowMap[i]] + "_lhs" );

         if( !consMatrix.getRowFlags()[i].test( RowFlag::kRhsInf ) )
            model.addConstr( grbLinExpr, GRB_LESS_EQUAL, double( rhs[i] ),
                             rowNames[origRowMap[i]] + "_rhs" );
         model.update();
      }
      model.update();
      model.write( "test.mps" );
      model.optimize();
      grb_status = model.get( GRB_IntAttr_Status );
      fmt::print( "{}\n", model.get( GRB_DoubleAttr_ObjVal ) );
      Vec<REAL> primal{};
      try
      {
         for( int i = 0; i < ncols; i++ )
            primal.push_back( REAL( vars[i].get( GRB_DoubleAttr_X ) ) );
         sol = Solution<REAL>( primal );
      }
      catch( GRBException& ex )
      {
         return 1;
      }
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
                     "Gurobi." );
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

      env = new GRBEnv();
      GRBModel model = GRBModel( *env );
      model.set( GRB_StringAttr_ModelName, problem.getName() );
      model.set( GRB_IntAttr_ModelSense, GRB_MINIMIZE );

      GRBVar* vars = new GRBVar[ncols];

      for( int i = 0; i < ncols; ++i )
      {
         int col = colset[i];

         assert( !domains.flags[col].test( ColFlag::kInactive ) );

         double lb = domains.flags[col].test( ColFlag::kLbInf )
                         ? -GRB_INFINITY
                         : domains.lower_bounds[col];
         double ub = domains.flags[col].test( ColFlag::kUbInf )
                         ? GRB_INFINITY
                         : domains.upper_bounds[col];

         char type;
         if( domains.flags[col].test( ColFlag::kIntegral ) )
         {
            if( lb == 0 && ub == 1 )
               type = GRB_BINARY;
            else
               type = GRB_INTEGER;
         }
         else if( domains.flags[col].test( ColFlag::kImplInt ) )
            type = GRB_INTEGER;
         else
            type = GRB_CONTINUOUS;

         vars[col] = model.addVar( lb, ub, double( obj[col] ), type,
                                   varNames[origColMap[col]].c_str() );
         model.update();
      }

      for( int i = 0; i < nrows; ++i )
      {
         int row = rowset[i];
         auto row_coeff =
             problem.getConstraintMatrix().getRowCoefficients( row );
         GRBLinExpr grbLinExpr = 0;
         for( int j = 0; j < row_coeff.getLength(); ++j )
            grbLinExpr += vars[row_coeff.getIndices()[j]] *
                          double( row_coeff.getValues()[j] );
         if( consMatrix.getRowFlags()[row].test( RowFlag::kEquation ) )
         {
            model.addConstr( grbLinExpr, GRB_EQUAL, double( rhs[row] ),
                             rowNames[origRowMap[row]] );
            continue;
         }
         if( !consMatrix.getRowFlags()[row].test( RowFlag::kLhsInf ) )
            model.addConstr( grbLinExpr, GRB_GREATER_EQUAL, double( lhs[row] ),
                             rowNames[origRowMap[row]] + "_lhs" );

         if( !consMatrix.getRowFlags()[row].test( RowFlag::kRhsInf ) )
            model.addConstr( grbLinExpr, GRB_LESS_EQUAL, double( rhs[row] ),
                             rowNames[origRowMap[row]] + "_rhs" );
         model.update();
      }
      model.update();
      model.write( "test.mps" );
      model.optimize();
      grb_status = model.get( GRB_IntAttr_Status );
      fmt::print( "{}\n", model.get( GRB_DoubleAttr_ObjVal ) );
      Vec<REAL> primal{};
      try
      {
         for( int i = 0; i < ncols; i++ )
            primal.push_back( REAL( vars[i].get( GRB_DoubleAttr_X ) ) );
         sol = Solution<REAL>( primal );
      }
      catch( GRBException& ex )
      {
         return 1;
      }
      return 0;
   }

 public:
   GurobiInterface() {}

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

      assert( this->status != SolverStatus::kError );

      switch( grb_status )
      {
      case GRB_INTERRUPTED:
      case GRB_ITERATION_LIMIT:
      case GRB_NODE_LIMIT:
      case GRB_TIME_LIMIT:
         this->status = SolverStatus::kInterrupted;
         break;
      case GRB_UNBOUNDED:
         this->status = SolverStatus::kUnbndOrInfeas;
         return;
      case GRB_INFEASIBLE:
         this->status = SolverStatus::kInfeasible;
         return;
      case GRB_INF_OR_UNBD:
         this->status = SolverStatus::kUnbounded;
         return;
      case GRB_OPTIMAL:
         this->status = SolverStatus::kOptimal;
      }
   }

   REAL
   getDualBound() override
   {
      // TODO:
      return 0;
   }

   bool
   getSolution( Solution<REAL>& solbuffer ) override
   {
      solbuffer = sol;
      return true;
   }

   bool
   getSolution( const Components& components, int component,
                Solution<REAL>& solbuffer ) override
   {
      const int* colset = components.getComponentsCols( component );
      assert( components.getComponentsNumCols( component ) ==
              sol.primal.size() );

      for( std::size_t i = 0; i < sol.primal.size(); ++i )
         solbuffer.primal[colset[i]] = sol.primal[i];

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
      return "gurobi";
   }

   void
   printDetails() override
   {
   }

   bool
   is_dual_solution_available() override
   {
      return false;
   }


   void
   addParameters( ParameterSet& paramSet ) override
   {
   }

   ~GurobiInterface() = default;
};

template <typename REAL>
class GurobiFactory : public SolverFactory<REAL>
{

   GurobiFactory() {}

 public:
   virtual std::unique_ptr<SolverInterface<REAL>>
   newSolver( VerbosityLevel verbosity ) const
   {
      auto gurobi =
          std::unique_ptr<SolverInterface<REAL>>( new GurobiInterface<REAL>() );

      return std::move( gurobi );
   }

   virtual void
   add_parameters( ParameterSet& parameter ) const
   {
   }
   static std::unique_ptr<SolverFactory<REAL>>
   create()
   {
      return std::unique_ptr<SolverFactory<REAL>>( new GurobiFactory<REAL>() );
   }
};

} // namespace papilo

#endif
