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

#ifndef _PAPILO_INTERFACES_SCIP_INTERFACE_HPP_
#define _PAPILO_INTERFACES_SCIP_INTERFACE_HPP_

#define UNUSED(expr) do { (void)(expr); } while (0)

#include "papilo/misc/Vec.hpp"
#include <cassert>
#include <stdexcept>

#include "papilo/core/Problem.hpp"
#include "papilo/interfaces/SolverInterface.hpp"
#include "scip/cons_linear.h"
#include "scip/scip.h"

#include "scip/struct_paramset.h"

namespace papilo
{

template <typename REAL>
class ScipInterface : public SolverInterface<REAL>
{
 private:
   SCIP* scip;
   Vec<SCIP_VAR*> vars;

   SCIP_RETCODE
   doSetUp( const Problem<REAL>& problem, const Vec<int>& origRowMap,
            const Vec<int>& origColMap )
   {
      int ncols = problem.getNCols();
      int nrows = problem.getNRows();
      const Vec<String>& varNames = problem.getVariableNames();
      const Vec<String>& consNames = problem.getConstraintNames();
      const VariableDomains<REAL>& domains = problem.getVariableDomains();
      const Objective<REAL>& obj = problem.getObjective();
      const auto& consMatrix = problem.getConstraintMatrix();
      const auto& lhs_values = consMatrix.getLeftHandSides();
      const auto& rhs_values = consMatrix.getRightHandSides();
      const auto& rflags = problem.getRowFlags();

      SCIP_CALL( SCIPcreateProbBasic( scip, problem.getName().c_str() ) );

      vars.resize( problem.getNCols() );

      for( int i = 0; i < ncols; ++i )
      {
         SCIP_VAR* var;
         assert( !domains.flags[i].test( ColFlag::kInactive ) );

         SCIP_Real lb = domains.flags[i].test( ColFlag::kLbInf )
                            ? -SCIPinfinity( scip )
                            : SCIP_Real( domains.lower_bounds[i] );
         SCIP_Real ub = domains.flags[i].test( ColFlag::kUbInf )
                            ? SCIPinfinity( scip )
                            : SCIP_Real( domains.upper_bounds[i] );
         SCIP_VARTYPE type;
         if( domains.flags[i].test( ColFlag::kIntegral ) )
         {
            if( lb == REAL{ 0 } && ub == REAL{ 1 } )
               type = SCIP_VARTYPE_BINARY;
            else
               type = SCIP_VARTYPE_INTEGER;
         }
         else if( domains.flags[i].test( ColFlag::kImplInt ) )
            type = SCIP_VARTYPE_IMPLINT;
         else
            type = SCIP_VARTYPE_CONTINUOUS;

         SCIP_CALL( SCIPcreateVarBasic(
             scip, &var, varNames[origColMap[i]].c_str(), lb, ub,
             SCIP_Real( obj.coefficients[i] ), type ) );
         SCIP_CALL( SCIPaddVar( scip, var ) );
         vars[i] = var;

         SCIP_CALL( SCIPreleaseVar( scip, &var ) );
      }

      Vec<SCIP_VAR*> consvars;
      Vec<SCIP_Real> consvals;
      consvars.resize( problem.getNCols() );
      consvals.resize( problem.getNCols() );

      for( int i = 0; i < nrows; ++i )
      {
         SCIP_CONS* cons;

         auto rowvec = consMatrix.getRowCoefficients( i );
         const REAL* vals = rowvec.getValues();
         const int* inds = rowvec.getIndices();
         SCIP_Real lhs = rflags[i].test( RowFlag::kLhsInf )
                             ? -SCIPinfinity( scip )
                             : SCIP_Real( lhs_values[i] );
         SCIP_Real rhs = rflags[i].test( RowFlag::kRhsInf )
                             ? SCIPinfinity( scip )
                             : SCIP_Real( rhs_values[i] );

         for( int k = 0; k != rowvec.getLength(); ++k )
         {
            consvars[k] = vars[inds[k]];
            consvals[k] = SCIP_Real( vals[k] );
         }

         SCIP_CALL( SCIPcreateConsBasicLinear(
             scip, &cons, consNames[origRowMap[i]].c_str(), rowvec.getLength(),
             consvars.data(), consvals.data(), lhs, rhs ) );
         SCIP_CALL( SCIPaddCons( scip, cons ) );
         SCIP_CALL( SCIPreleaseCons( scip, &cons ) );
      }

      if( obj.offset != REAL{ 0 } )
         SCIP_CALL( SCIPaddOrigObjoffset( scip, SCIP_Real( obj.offset ) ) );

      return SCIP_OKAY;
   }

   SCIP_RETCODE
   doSetUp( const Problem<REAL>& problem, const Vec<int>& origRowMap,
            const Vec<int>& origColMap, const Components& components,
            const ComponentInfo& component )
   {
      int ncols = components.getComponentsNumCols( component.componentid );
      int nrows = components.getComponentsNumRows( component.componentid );
      const int* colset = components.getComponentsCols( component.componentid );
      const int* rowset = components.getComponentsRows( component.componentid );
      const Vec<String>& varNames = problem.getVariableNames();
      const Vec<String>& consNames = problem.getConstraintNames();
      const VariableDomains<REAL>& domains = problem.getVariableDomains();
      const Objective<REAL>& obj = problem.getObjective();
      const auto& consMatrix = problem.getConstraintMatrix();
      const auto& lhs_values = consMatrix.getLeftHandSides();
      const auto& rhs_values = consMatrix.getRightHandSides();
      const auto& rflags = problem.getRowFlags();

      SCIP_CALL( SCIPcreateProbBasic( scip, problem.getName().c_str() ) );

      vars.resize( ncols );

      for( int i = 0; i < ncols; ++i )
      {
         int col = colset[i];
         SCIP_VAR* var;
         assert( !domains.flags[col].test( ColFlag::kInactive ) );

         SCIP_Real lb = domains.flags[col].test( ColFlag::kLbInf )
                            ? -SCIPinfinity( scip )
                            : SCIP_Real( domains.lower_bounds[col] );
         SCIP_Real ub = domains.flags[col].test( ColFlag::kUbInf )
                            ? SCIPinfinity( scip )
                            : SCIP_Real( domains.upper_bounds[col] );
         SCIP_VARTYPE type;
         if( domains.flags[col].test( ColFlag::kIntegral ) )
         {
            if( lb == REAL{ 0 } && ub == REAL{ 1 } )
               type = SCIP_VARTYPE_BINARY;
            else
               type = SCIP_VARTYPE_INTEGER;
         }
         else
            type = SCIP_VARTYPE_CONTINUOUS;

         SCIP_CALL( SCIPcreateVarBasic(
             scip, &var, varNames[origColMap[col]].c_str(), lb, ub,
             SCIP_Real( obj.coefficients[col] ), type ) );
         SCIP_CALL( SCIPaddVar( scip, var ) );
         vars[i] = var;

         SCIP_CALL( SCIPreleaseVar( scip, &var ) );
      }

      Vec<SCIP_VAR*> consvars;
      Vec<SCIP_Real> consvals;
      consvars.resize( ncols );
      consvals.resize( ncols );

      for( int i = 0; i < nrows; ++i )
      {
         int row = rowset[i];
         SCIP_CONS* cons;

         auto rowvec = consMatrix.getRowCoefficients( row );
         const REAL* vals = rowvec.getValues();
         const int* inds = rowvec.getIndices();
         SCIP_Real lhs = rflags[row].test( RowFlag::kLhsInf )
                             ? -SCIPinfinity( scip )
                             : SCIP_Real( lhs_values[row] );
         SCIP_Real rhs = rflags[row].test( RowFlag::kRhsInf )
                             ? SCIPinfinity( scip )
                             : SCIP_Real( rhs_values[row] );

         for( int k = 0; k != rowvec.getLength(); ++k )
         {
            consvars[k] = vars[components.getColComponentIdx( inds[k] )];
            consvals[k] = SCIP_Real( vals[k] );
         }

         SCIP_CALL( SCIPcreateConsBasicLinear(
             scip, &cons, consNames[origRowMap[row]].c_str(),
             rowvec.getLength(), consvars.data(), consvals.data(), lhs, rhs ) );
         SCIP_CALL( SCIPaddCons( scip, cons ) );
         SCIP_CALL( SCIPreleaseCons( scip, &cons ) );
      }

      if( obj.offset != REAL{ 0 } )
         SCIP_CALL( SCIPaddOrigObjoffset( scip, SCIP_Real( obj.offset ) ) );

      return SCIP_OKAY;
   }

 public:
   ScipInterface() : scip( nullptr )
   {
      if( SCIPcreate( &scip ) != SCIP_OKAY )
         throw std::runtime_error( "could not create SCIP" );
   }

   SCIP*
   getSCIP()
   {
      return scip;
   }

   void
   readSettings( const String& file ) override
   {
      if( SCIPreadParams( scip, file.c_str() ) != SCIP_OKAY )
         this->status = SolverStatus::kError;
   }

   void
   setUp( const Problem<REAL>& prob, const Vec<int>& row_maps,
          const Vec<int>& col_maps, const Components& components,
          const ComponentInfo& component ) override
   {
      if( doSetUp( prob, row_maps, col_maps, components, component ) !=
          SCIP_OKAY )
         this->status = SolverStatus::kError;
   }

   void
   setNodeLimit( int num ) override
   {
      if( SCIPsetLongintParam( scip, "limits/nodes", num ) != SCIP_OKAY )
         this->status = SolverStatus::kError;
   }

   void
   setGapLimit( const REAL& gaplim ) override
   {
      if( SCIPsetRealParam( scip, "limits/gap", SCIP_Real( gaplim ) ) !=
          SCIP_OKAY )
         this->status = SolverStatus::kError;
   }

   void
   setSoftTimeLimit( double tlim ) override
   {
      if( SCIPsetIntParam( scip, "timing/clocktype", 2 ) != SCIP_OKAY )
         this->status = SolverStatus::kError;

      if( SCIPsetRealParam( scip, "limits/softtime", SCIP_Real( tlim ) ) !=
          SCIP_OKAY )
         this->status = SolverStatus::kError;
   }

   void
   setTimeLimit( double tlim ) override
   {
      if( SCIPsetIntParam( scip, "timing/clocktype", 2 ) != SCIP_OKAY )
         this->status = SolverStatus::kError;

      if( SCIPsetRealParam( scip, "limits/time", SCIP_Real( tlim ) ) !=
          SCIP_OKAY )
         this->status = SolverStatus::kError;
   }

   void
   setVerbosity( VerbosityLevel verbosity ) override
   {
      switch( verbosity )
      {
      case VerbosityLevel::kQuiet:
         SCIP_CALL_ABORT( SCIPsetIntParam( scip, "display/verblevel", 0 ) );
         break;
      case VerbosityLevel::kError:
         SCIP_CALL_ABORT( SCIPsetIntParam( scip, "display/verblevel", 1 ) );
         break;
      case VerbosityLevel::kWarning:
         SCIP_CALL_ABORT( SCIPsetIntParam( scip, "display/verblevel", 2 ) );
         break;
      case VerbosityLevel::kInfo:
         SCIP_CALL_ABORT( SCIPsetIntParam( scip, "display/verblevel", 4 ) );
         break;
      case VerbosityLevel::kDetailed:
         SCIP_CALL_ABORT( SCIPsetIntParam( scip, "display/verblevel", 5 ) );
      }
   }

   void
   setUp( const Problem<REAL>& prob, const Vec<int>& row_maps,
          const Vec<int>& col_maps ) override
   {
      if( doSetUp( prob, row_maps, col_maps ) != SCIP_OKAY )
         this->status = SolverStatus::kError;
   }

   void
   solve() override
   {
      assert( this->status != SolverStatus::kError );

      if( SCIPsolve( scip ) != SCIP_OKAY )
      {
         this->status = SolverStatus::kError;
         return;
      }

      switch( SCIPgetStatus( scip ) )
      {
      case SCIP_STATUS_UNKNOWN:
         this->status = SolverStatus::kError;
         return;
      case SCIP_STATUS_USERINTERRUPT:
      case SCIP_STATUS_NODELIMIT:
      case SCIP_STATUS_TOTALNODELIMIT:
      case SCIP_STATUS_STALLNODELIMIT:
      case SCIP_STATUS_TIMELIMIT:
      case SCIP_STATUS_MEMLIMIT:
      case SCIP_STATUS_GAPLIMIT:
      case SCIP_STATUS_SOLLIMIT:
      case SCIP_STATUS_BESTSOLLIMIT:
      case SCIP_STATUS_RESTARTLIMIT:
#if SCIP_VERSION_MAJOR >= 6
      case SCIP_STATUS_TERMINATE:
#endif
         this->status = SolverStatus::kInterrupted;
         break;
      case SCIP_STATUS_INFORUNBD:
         this->status = SolverStatus::kUnbndOrInfeas;
         return;
      case SCIP_STATUS_INFEASIBLE:
         this->status = SolverStatus::kInfeasible;
         return;
      case SCIP_STATUS_UNBOUNDED:
         this->status = SolverStatus::kUnbounded;
         return;
      case SCIP_STATUS_OPTIMAL:
         this->status = SolverStatus::kOptimal;
      }
   }

   REAL
   getDualBound() override
   {
      return SCIPgetDualbound( scip );
   }

   bool
   getSolution( Solution<REAL>& solbuffer ) override
   {
      SCIP_SOL* sol = SCIPgetBestSol( scip );

      if( solbuffer.type != SolutionType::kPrimal )
         return false;

      solbuffer.primal.resize( vars.size() );

      if( sol != nullptr )
      {
         SCIP_SOL* finitesol;
         SCIP_Bool success;

#ifndef NDEBUG
         SCIP_Bool feasible;
         SCIP_CALL_ABORT(
             SCIPcheckSolOrig( scip, sol, &feasible, TRUE, TRUE ) );
#endif
         SCIP_CALL_ABORT(
             SCIPcreateFiniteSolCopy( scip, &finitesol, sol, &success ) );

         if( finitesol != nullptr )
         {
            for( std::size_t i = 0; i != vars.size(); ++i )
               solbuffer.primal[i] =
                   REAL( SCIPgetSolVal( scip, finitesol, vars[i] ) );

            SCIP_CALL_ABORT( SCIPfreeSol( scip, &finitesol ) );
         }
         else
         {
            for( std::size_t i = 0; i != vars.size(); ++i )
               solbuffer.primal[i] =
                   REAL( SCIPgetSolVal( scip, sol, vars[i] ) );
         }

         return true;
      }

      return false;
   }

   bool
   getSolution( const Components& components, int component,
                Solution<REAL>& solbuffer ) override
   {
      SCIP_SOL* sol = SCIPgetBestSol( scip );

      if( solbuffer.type != SolutionType::kPrimal )
         return false;

      if( sol != nullptr )
      {
         SCIP_SOL* finitesol;
         SCIP_Bool success;

         const int* colset = components.getComponentsCols( component );
         assert( components.getComponentsNumCols( component ) == vars.size() );

         SCIP_CALL_ABORT(
             SCIPcreateFiniteSolCopy( scip, &finitesol, sol, &success ) );

         if( finitesol != nullptr )
         {
            for( std::size_t i = 0; i != vars.size(); ++i )
               solbuffer.primal[colset[i]] =
                   REAL( SCIPgetSolVal( scip, finitesol, vars[i] ) );

            SCIP_CALL_ABORT( SCIPfreeSol( scip, &finitesol ) );
         }
         else
         {
            for( std::size_t i = 0; i != vars.size(); ++i )
               solbuffer.primal[colset[i]] =
                   REAL( SCIPgetSolVal( scip, sol, vars[i] ) );
         }

         return true;
      }

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
      return "SCIP";
   }

   void
   printDetails() override
   {
      SCIP_RETCODE retcode = SCIPprintStatistics( scip, stdout );
      UNUSED(retcode);
      assert( retcode == SCIP_OKAY );
   }

   bool
   is_dual_solution_available() override
   {
      return false;
   }

   void
   addParameters( ParameterSet& paramSet ) override
   {
      SCIP_PARAM** params = SCIPgetParams( scip );
      int nparams = SCIPgetNParams( scip );

      for( int i = 0; i != nparams; ++i )
      {
         switch( params[i]->paramtype )
         {
         case SCIP_PARAMTYPE_BOOL:
         {
            unsigned int* valptr;
            if( params[i]->data.boolparam.valueptr != nullptr )
               valptr = params[i]->data.boolparam.valueptr;
            else
               valptr = &params[i]->data.boolparam.curvalue;

            paramSet.addParameter( params[i]->name, params[i]->desc,
                                   *reinterpret_cast<bool*>( valptr ) );
            break;
         }
         case SCIP_PARAMTYPE_CHAR:
         {
            char* valptr;
            if( params[i]->data.charparam.valueptr != nullptr )
               valptr = params[i]->data.charparam.valueptr;
            else
               valptr = &params[i]->data.charparam.curvalue;

            char* last = params[i]->data.charparam.allowedvalues;

            if( last != nullptr )
            {
               while( *last != '\0' )
                  ++last;

               paramSet.addParameter(
                   params[i]->name, params[i]->desc, *valptr,
                   Vec<char>( params[i]->data.charparam.allowedvalues, last ) );
            }

            break;
         }
         case SCIP_PARAMTYPE_INT:
         {
            int* valptr;
            if( params[i]->data.intparam.valueptr != nullptr )
               valptr = params[i]->data.intparam.valueptr;
            else
               valptr = &params[i]->data.intparam.curvalue;

            paramSet.addParameter( params[i]->name, params[i]->desc, *valptr,
                                   params[i]->data.intparam.minvalue,
                                   params[i]->data.intparam.maxvalue );
            break;
         }
         case SCIP_PARAMTYPE_LONGINT:
         {
            std::int64_t* valptr;
            if( params[i]->data.longintparam.valueptr != nullptr )
               valptr = reinterpret_cast<std::int64_t*>(
                   params[i]->data.longintparam.valueptr );
            else
               valptr = reinterpret_cast<std::int64_t*>(
                   &params[i]->data.longintparam.curvalue );

            paramSet.addParameter(
                params[i]->name, params[i]->desc, *valptr,
                std::int64_t( params[i]->data.longintparam.minvalue ),
                std::int64_t( params[i]->data.longintparam.maxvalue ) );
            break;
         }
         case SCIP_PARAMTYPE_REAL:
         {
            double* valptr;
            if( params[i]->data.realparam.valueptr != nullptr )
               valptr = params[i]->data.realparam.valueptr;
            else
               valptr = &params[i]->data.realparam.curvalue;

            paramSet.addParameter( params[i]->name, params[i]->desc, *valptr,
                                   params[i]->data.realparam.minvalue,
                                   params[i]->data.realparam.maxvalue );
            break;
         }
         case SCIP_PARAMTYPE_STRING:
            break;
         }
      }
   }

   ~ScipInterface()
   {
      if( scip != nullptr )
      {
         SCIP_RETCODE retcode = SCIPfree( &scip );
         UNUSED(retcode);
         assert( retcode == SCIP_OKAY );
      }
   }
};

template <typename REAL>
class ScipFactory : public SolverFactory<REAL>
{
   void ( *scipsetup )( SCIP* scip, void* usrdata );
   void* scipsetup_usrdata;

   ScipFactory( void ( *scipsetup )( SCIP* scip, void* usrdata ),
                void* scipsetup_usrdata )
       : scipsetup( scipsetup ), scipsetup_usrdata( scipsetup_usrdata )
   {
   }

 public:
   virtual std::unique_ptr<SolverInterface<REAL>>
   newSolver( VerbosityLevel verbosity ) const
   {
      auto scip =
          std::unique_ptr<SolverInterface<REAL>>( new ScipInterface<REAL>() );

      if( scipsetup != nullptr )
         scipsetup( static_cast<ScipInterface<REAL>*>( scip.get() )->getSCIP(),
                    scipsetup_usrdata );

      scip->setVerbosity( verbosity );

      return std::move( scip );
   }

   virtual void
   add_parameters( ParameterSet& parameter ) const
   {
   }

   static std::unique_ptr<SolverFactory<REAL>>
   create( void ( *scipsetup )( SCIP* scip, void* usrdata ) = nullptr,
           void* scipsetup_usrdata = nullptr )
   {
      return std::unique_ptr<SolverFactory<REAL>>(
          new ScipFactory<REAL>( scipsetup, scipsetup_usrdata ) );
   }
};

} // namespace papilo

#endif
