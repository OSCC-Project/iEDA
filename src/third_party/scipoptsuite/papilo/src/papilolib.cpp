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

#include "papilolib.h"
#include "scip/pub_message.h"
#include "scip/scipdefplugins.h"
#include <cassert>
#include <cstdlib>
#include <new>

static void*
malloc_default_cb( size_t size, void* )
{
   return std::malloc( size );
}

static void
free_default_cb( void* ptr, void* )
{
   std::free( ptr );
}

static void* alloc_usrdata = nullptr;
static void* ( *malloccb )( size_t size, void* usrptr ) = malloc_default_cb;
static void ( *freecb )( void* ptr, void* usrptr ) = free_default_cb;

template <class T>
struct CallbackAllocator
{
   typedef T value_type;

   CallbackAllocator() noexcept {}

   template <class U>
   CallbackAllocator( const CallbackAllocator<U>& ) noexcept
   {
   }
   template <class U>
   bool
   operator==( const CallbackAllocator<U>& ) const noexcept
   {
      return true;
   }
   template <class U>
   bool
   operator!=( const CallbackAllocator<U>& ) const noexcept
   {
      return false;
   }

   T*
   allocate( const size_t n ) const
   {
      if( n == 0 )
      {
         return nullptr;
      }
      if( n > static_cast<size_t>( -1 ) / sizeof( T ) )
      {
         throw std::bad_array_new_length();
      }
      void* const pv = malloccb( n * sizeof( T ), alloc_usrdata );
      if( !pv )
      {
         throw std::bad_alloc();
      }
      return static_cast<T*>( pv );
   }

   void
   deallocate( T* const p, size_t ) const noexcept
   {
      freecb( p, alloc_usrdata );
   }
};

#include "papilo/misc/Alloc.hpp"
using namespace papilo;

namespace papilo
{
template <typename T>
struct AllocatorTraits<T>
{
   using type = std::allocator<T>;
};
} // namespace papilo

#include "papilo/core/ConstraintMatrix.hpp"
#include "papilo/core/Objective.hpp"
#include "papilo/core/Presolve.hpp"
#include "papilo/core/ProblemBuilder.hpp"
#include "papilo/core/VariableDomains.hpp"
#include "papilo/interfaces/ScipInterface.hpp"
#include "papilo/interfaces/SoplexInterface.hpp"
#include "papilo/misc/MultiPrecision.hpp"
#ifdef PAPILO_TBB
#include "papilo/misc/tbb.hpp"
#endif
#include "papilo/core/postsolve/Postsolve.hpp"
#ifdef PAPILO_MPS_WRITER
#include "papilo/io/MpsWriter.hpp"
#endif

#include <boost/program_options.hpp>
#include <cassert>
#include <fstream>

struct Papilo_Problem
{
   ProblemBuilder<double> problemBuilder;
   double infinity;
};

PAPILO_PROBLEM*
papilo_problem_create( double infinity, const char* name, int nnz_hint,
                       int row_hint, int col_hint )
{
   assert( nnz_hint >= 0 );

   // allocate memory and construct with placement new
   PAPILO_PROBLEM* prob = new( Allocator<PAPILO_PROBLEM>().allocate( 1 ) )
       PAPILO_PROBLEM{ ProblemBuilder<double>(), infinity };

   // allocate memory for given number of nonzeros
   prob->problemBuilder.reserve( nnz_hint, row_hint, col_hint );

   if( name == nullptr )
      prob->problemBuilder.setProblemName( "problem" );
   else
      prob->problemBuilder.setProblemName( name );

   return prob;
}

int
papilo_problem_add_cols( PAPILO_PROBLEM* problem, int num, const double* lb,
                         const double* ub, const unsigned char* integral,
                         const double* obj, const char** colnames )
{
   assert( num >= 0 );

   if( num == 0 )
      return -1;

   int ncols = problem->problemBuilder.getNumCols();

   problem->problemBuilder.setNumCols( ncols + num );

   for( int i = 0; i != num; ++i )
   {
      int col = ncols + i;
      problem->problemBuilder.setObj( col, obj[i] );

      bool isLbInf = lb[i] <= -problem->infinity;
      bool isUbInf = ub[i] >= problem->infinity;

      problem->problemBuilder.setColLbInf( col, isLbInf );
      problem->problemBuilder.setColUbInf( col, isUbInf );

      if( !isLbInf )
         problem->problemBuilder.setColLb( col, lb[i] );

      if( !isUbInf )
         problem->problemBuilder.setColUb( col, ub[i] );

      problem->problemBuilder.setColIntegral( col, integral[i] );

      if( colnames != nullptr )
         problem->problemBuilder.setColName( col, colnames[i] );
      else
         problem->problemBuilder.setColName(
             col, fmt::format( "x{}", col ).c_str() );
   }

   return ncols;
}

int
papilo_problem_add_col( PAPILO_PROBLEM* problem, double lb, double ub,
                        unsigned char integral, double obj,
                        const char* colname )
{
   int ncols = problem->problemBuilder.getNumCols();

   problem->problemBuilder.setNumCols( ncols + 1 );

   int col = ncols;
   problem->problemBuilder.setObj( col, obj );

   bool isLbInf = lb <= -problem->infinity;
   bool isUbInf = ub >= problem->infinity;

   problem->problemBuilder.setColLbInf( col, isLbInf );
   problem->problemBuilder.setColUbInf( col, isUbInf );

   if( !isLbInf )
      problem->problemBuilder.setColLb( col, lb );

   if( !isUbInf )
      problem->problemBuilder.setColUb( col, ub );

   problem->problemBuilder.setColIntegral( col, integral );

   if( colname != nullptr )
      problem->problemBuilder.setColName( col, colname );
   else
      problem->problemBuilder.setColName( col,
                                          fmt::format( "x{}", col ).c_str() );

   return ncols;
}

int
papilo_problem_get_num_cols( PAPILO_PROBLEM* problem )
{
   return problem->problemBuilder.getNumCols();
}

int
papilo_problem_get_num_rows( PAPILO_PROBLEM* problem )
{
   return problem->problemBuilder.getNumRows();
}

int
papilo_problem_get_num_nonzeros( PAPILO_PROBLEM* problem )
{
   // todo
   return 0;
}

void
papilo_problem_change_col_lb( PAPILO_PROBLEM* problem, int col, double lb )
{
   problem->problemBuilder.setColLb( col, lb );
}

void
papilo_problem_change_col_ub( PAPILO_PROBLEM* problem, int col, double ub )
{
   problem->problemBuilder.setColUb( col, ub );
}

void
papilo_problem_change_col_integral( PAPILO_PROBLEM* problem, int col,
                                    unsigned char integral )
{
   problem->problemBuilder.setColIntegral( col, integral );
}

void
papilo_problem_change_col_obj( PAPILO_PROBLEM* problem, int col, double obj )
{
   problem->problemBuilder.setObj( col, obj );
}

int
papilo_problem_add_simple_rows( PAPILO_PROBLEM* problem, int num,
                                const unsigned char* rowtypes,
                                const double* side, const char** rownames )
{
   assert( num >= 0 );

   if( num == 0 )
      return -1;

   int nrows = problem->problemBuilder.getNumRows();

   problem->problemBuilder.setNumRows( nrows + num );

   for( int i = 0; i != num; ++i )
   {
      int row = nrows + i;

      switch( rowtypes[i] )
      {
      case PAPILO_ROW_TYPE_GREATER:
         problem->problemBuilder.setRowRhsInf( row, true );
         problem->problemBuilder.setRowLhsInf( row, false );
         problem->problemBuilder.setRowLhs( row, side[i] );
         break;
      case PAPILO_ROW_TYPE_LESSER:
         problem->problemBuilder.setRowRhsInf( row, false );
         problem->problemBuilder.setRowLhsInf( row, true );
         problem->problemBuilder.setRowRhs( row, side[i] );
         break;
      case PAPILO_ROW_TYPE_EQUAL:
         problem->problemBuilder.setRowRhsInf( row, false );
         problem->problemBuilder.setRowLhsInf( row, false );
         problem->problemBuilder.setRowLhs( row, side[i] );
         problem->problemBuilder.setRowRhs( row, side[i] );
      }

      if( rownames != nullptr )
         problem->problemBuilder.setRowName( row, rownames[i] );
      else
         problem->problemBuilder.setRowName(
             row, fmt::format( "c{}", row ).c_str() );
   }

   return nrows;
}

int
papilo_problem_add_generic_rows( PAPILO_PROBLEM* problem, int num,
                                 const double* lhs, const double* rhs,
                                 const char** rownames )
{
   assert( num >= 0 );

   if( num == 0 )
      return -1;

   int nrows = problem->problemBuilder.getNumRows();

   problem->problemBuilder.setNumRows( nrows + num );

   for( int i = 0; i != num; ++i )
   {
      int row = nrows + i;

      bool isLhsInf = lhs[i] <= -problem->infinity;
      bool isRhsInf = rhs[i] >= problem->infinity;

      problem->problemBuilder.setRowRhsInf( row, isRhsInf );

      if( isLhsInf )
         problem->problemBuilder.setRowLhsInf( row, true );
      else
         problem->problemBuilder.setRowLhs( row, lhs[i] );

      if( isRhsInf )
         problem->problemBuilder.setRowRhsInf( row, true );
      else
         problem->problemBuilder.setRowRhs( row, rhs[i] );

      if( rownames != nullptr )
         problem->problemBuilder.setRowName( row, rownames[i] );
      else
         problem->problemBuilder.setRowName(
             row, fmt::format( "c{}", row ).c_str() );
   }

   return nrows;
}

int
papilo_problem_add_simple_row( PAPILO_PROBLEM* problem, unsigned char rowtype,
                               double side, const char* rowname )
{
   int nrows = problem->problemBuilder.getNumRows();

   problem->problemBuilder.setNumRows( nrows + 1 );

   int row = nrows;

   switch( rowtype )
   {
   case PAPILO_ROW_TYPE_GREATER:
      problem->problemBuilder.setRowRhsInf( row, true );
      problem->problemBuilder.setRowLhsInf( row, false );
      problem->problemBuilder.setRowLhs( row, side );
      break;
   case PAPILO_ROW_TYPE_LESSER:
      problem->problemBuilder.setRowRhsInf( row, false );
      problem->problemBuilder.setRowLhsInf( row, true );
      problem->problemBuilder.setRowRhs( row, side );
      break;
   case PAPILO_ROW_TYPE_EQUAL:
      problem->problemBuilder.setRowRhsInf( row, false );
      problem->problemBuilder.setRowLhsInf( row, false );
      problem->problemBuilder.setRowLhs( row, side );
      problem->problemBuilder.setRowRhs( row, side );
   }

   if( rowname != nullptr )
      problem->problemBuilder.setRowName( row, rowname );
   else
      problem->problemBuilder.setRowName( row,
                                          fmt::format( "c{}", row ).c_str() );

   return nrows;
}

int
papilo_problem_add_generic_row( PAPILO_PROBLEM* problem, double lhs, double rhs,
                                const char* rowname )
{
   int nrows = problem->problemBuilder.getNumRows();

   problem->problemBuilder.setNumRows( nrows + 1 );

   int row = nrows;

   bool isLhsInf = lhs <= -problem->infinity;
   bool isRhsInf = rhs >= problem->infinity;

   if( isLhsInf )
      problem->problemBuilder.setRowLhsInf( row, true );
   else
      problem->problemBuilder.setRowLhs( row, lhs );

   if( isRhsInf )
      problem->problemBuilder.setRowRhsInf( row, true );
   else
      problem->problemBuilder.setRowRhs( row, rhs );

   if( rowname != nullptr )
      problem->problemBuilder.setRowName( row, rowname );
   else
      problem->problemBuilder.setRowName( row,
                                          fmt::format( "c{}", row ).c_str() );

   return nrows;
}

void
papilo_problem_free( PAPILO_PROBLEM* prob )
{
   // call destructor
   prob->~Papilo_Problem();

   // deallocate memory
   Allocator<PAPILO_PROBLEM>().deallocate( prob, 1 );
}

void
papilo_problem_add_nonzero( PAPILO_PROBLEM* problem, int row, int col,
                            double val )
{
   problem->problemBuilder.addEntry( row, col, val );
}

void
papilo_problem_add_nonzeros_row( PAPILO_PROBLEM* problem, int row, int num,
                                 const int* cols, const double* vals )
{
   problem->problemBuilder.addRowEntries( row, num, cols, vals );
}

void
papilo_problem_add_nonzeros_col( PAPILO_PROBLEM* problem, int col, int num,
                                 const int* rows, const double* vals )
{
   problem->problemBuilder.addColEntries( col, num, rows, vals );
}

void
papilo_problem_add_nonzeros_csr( PAPILO_PROBLEM* problem, const int* rowstart,
                                 const int* cols, const double* vals )
{
   for( int row = 0; row != problem->problemBuilder.getNumRows(); ++row )
      problem->problemBuilder.addRowEntries(
          row, rowstart[row + 1] - rowstart[row], cols + rowstart[row],
          vals + rowstart[row] );
}

void
papilo_problem_add_nonzeros_csc( PAPILO_PROBLEM* problem, const int* colstart,
                                 const int* rows, const double* vals )
{
   for( int col = 0; col != problem->problemBuilder.getNumCols(); ++col )
      problem->problemBuilder.addColEntries(
          col, colstart[col + 1] - colstart[col], rows + colstart[col],
          vals + colstart[col] );
}

enum class SolverState
{
   INIT,
   PROBLEM_LOADED,
   PROBLEM_PRESOLVED,
   PROBLEM_SOLVED
};

class MessageStreambuf : public std::stringbuf
{
   const Message& msg;

 public:
   MessageStreambuf( const Message& msg ) : msg( msg ) {}

   int
   sync() override
   {
      msg.info( this->str() );
      this->str( "" );
      return 0;
   }
};

class MessageStream : public std::ostream
{
   MessageStreambuf messageStreambuf;

 public:
   MessageStream( const Message& msg )
       : messageStreambuf( msg ), std::ostream( &messageStreambuf )
   {
   }
};

struct Papilo_Solver
{
   SolverState state;
   Problem<double> problem;
   Presolve<double> presolve;
   ParameterSet paramSet;
   PresolveResult<double> presolveResult;
   std::unique_ptr<MessageStream> messageStream;
   std::unique_ptr<SolverInterface<double>> mipSolver;
   ParameterSet mipParamSet;
   std::unique_ptr<SolverInterface<double>> lpSolver;
   ParameterSet lpParamSet;
   Solution<double> solution;
   double tlimsoft;
   PAPILO_SOLVING_INFO solveinfo;
};

static void
PrintSCIPMessage( SCIP_MESSAGEHDLR* handler, FILE* file, const char* message,
                  VerbosityLevel level )
{
   if( file && file != stdout )
   {
      fputs( message, file );
      fflush( file );
   }
   else
   {
      Message* msg =
          reinterpret_cast<Message*>( SCIPmessagehdlrGetData( handler ) );
      msg->print( level, message );
   }
}

static SCIP_DECL_ERRORPRINTING( messageError )
{
   SCIP_MESSAGEHDLR* messagehdlr = static_cast<SCIP_MESSAGEHDLR*>( data );
   PrintSCIPMessage( messagehdlr, NULL, msg, VerbosityLevel::kError );
}

static SCIP_DECL_MESSAGEWARNING( messageWarning )
{
   PrintSCIPMessage( messagehdlr, file, msg, VerbosityLevel::kWarning );
}

static SCIP_DECL_MESSAGEINFO( messageInfo )
{
   PrintSCIPMessage( messagehdlr, file, msg, VerbosityLevel::kInfo );
}

static SCIP_DECL_MESSAGEDIALOG( messageDialog )
{
   PrintSCIPMessage( messagehdlr, file, msg, VerbosityLevel::kInfo );
}

static void
setupscip( SCIP* scip, void* usrdata )
{
   PAPILO_SOLVER* solver = reinterpret_cast<PAPILO_SOLVER*>( usrdata );

   SCIP_RETCODE retcode = SCIPincludeDefaultPlugins( scip );
   if( retcode != SCIP_OKAY )
   {
      // todo
      assert( false );
   }
   SCIP_MESSAGEHDLR* msghdlr;
   retcode = SCIPmessagehdlrCreate(
       &msghdlr, false, NULL, false, messageWarning, messageDialog, messageInfo,
       NULL,
       reinterpret_cast<SCIP_MESSAGEHDLRDATA*>( &solver->presolve.message() ) );

   if( retcode != SCIP_OKAY )
   {
      // todo
      assert( false );
   }

   SCIPmessageSetErrorPrinting( messageError, msghdlr );
   SCIPsetMessagehdlr( scip, msghdlr );

   // copy settings from main solver if further instances are created, e.g. for
   // components
   ScipInterface<double>* mainscip =
       static_cast<ScipInterface<double>*>( solver->mipSolver.get() );
   if( mainscip )
   {
      retcode = SCIPcopyParamSettings( mainscip->getSCIP(), scip );
      if( retcode != SCIP_OKAY )
      {
         // todo
         assert( false );
      }
   }
}

static void
setupsoplex( soplex::SoPlex& spx, void* usrdata )
{
   PAPILO_SOLVER* solver = reinterpret_cast<PAPILO_SOLVER*>( usrdata );

   spx.spxout.setStream( soplex::SPxOut::ERROR, *solver->messageStream );
   spx.spxout.setStream( soplex::SPxOut::WARNING, *solver->messageStream );
   spx.spxout.setStream( soplex::SPxOut::DEBUG, *solver->messageStream );
   spx.spxout.setStream( soplex::SPxOut::INFO1, *solver->messageStream );
   spx.spxout.setStream( soplex::SPxOut::INFO2, *solver->messageStream );
   spx.spxout.setStream( soplex::SPxOut::INFO3, *solver->messageStream );

   // copy settings from main solver if further instances are created, e.g. for
   // components
   SoplexInterface<double>* mainspx =
       static_cast<SoplexInterface<double>*>( solver->lpSolver.get() );
   if( mainspx )
   {
      spx.setSettings( mainspx->getSoPlex().settings() );
   }
}

PAPILO_SOLVER*
papilo_solver_create()
{
   PAPILO_SOLVER* solver =
       new( Allocator<PAPILO_SOLVER>().allocate( 1 ) ) PAPILO_SOLVER();
   solver->messageStream = std::unique_ptr<MessageStream>(
       new MessageStream( solver->presolve.message() ) );
   solver->presolve.addDefaultPresolvers();
   solver->presolve.setLPSolverFactory( SoplexFactory<double>::create(
       setupsoplex, reinterpret_cast<void*>( solver ) ) );
   solver->presolve.setMIPSolverFactory( ScipFactory<double>::create(
       setupscip, reinterpret_cast<void*>( solver ) ) );
   solver->paramSet = solver->presolve.getParameters();
   solver->mipSolver = solver->presolve.getMIPSolverFactory()->newSolver();
   solver->lpSolver = solver->presolve.getLPSolverFactory()->newSolver();
   solver->mipSolver->addParameters( solver->mipParamSet );
   solver->lpSolver->addParameters( solver->lpParamSet );

   solver->state = SolverState::INIT;
   solver->solveinfo.solvingtime = 0.0;

   solver->tlimsoft = std::numeric_limits<double>::max();
   solver->paramSet.addParameter(
       "presolve.tlimsoft",
       "soft time limit that is only set after a solution has been found",
       solver->tlimsoft, 0.0 );

   return solver;
}

void
papilo_solver_set_trace_callback( PAPILO_SOLVER* solver,
                                  void ( *thetracecb )( int level,
                                                        const char* data,
                                                        size_t size,
                                                        void* usrptr ),
                                  void* usrptr )
{
   solver->presolve.message().setOutputCallback( thetracecb, usrptr );
}

void
papilo_solver_free( PAPILO_SOLVER* solver )
{
   // call destructor
   solver->~Papilo_Solver();

   // deallocate
   Allocator<PAPILO_SOLVER>().deallocate( solver, 1 );
}

/// Set bool parameter with given key to the given value
PAPILO_PARAM_RESULT
papilo_solver_set_param_bool( PAPILO_SOLVER* solver, const char* key,
                              unsigned int val )
{
   try
   {
      solver->paramSet.setParameter( key, static_cast<bool>( val ) );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_param_real( PAPILO_SOLVER* solver, const char* key,
                              double val )
{
   try
   {
      solver->paramSet.setParameter( key, val );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_param_int( PAPILO_SOLVER* solver, const char* key, int val )
{
   try
   {
      solver->paramSet.setParameter( key, val );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_param_char( PAPILO_SOLVER* solver, const char* key, char val )
{
   try
   {
      solver->paramSet.setParameter( key, val );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_param_string( PAPILO_SOLVER* solver, const char* key,
                                const char* val )
{
   try
   {
      solver->paramSet.setParameter( key, val );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_mip_param_real( PAPILO_SOLVER* solver, const char* key,
                                  double val )
{
   try
   {
      solver->mipParamSet.setParameter( key, val );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_mip_param_int( PAPILO_SOLVER* solver, const char* key,
                                 int val )
{
   try
   {
      solver->mipParamSet.setParameter( key, val );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_mip_param_bool( PAPILO_SOLVER* solver, const char* key,
                                  unsigned int val )
{
   try
   {
      solver->mipParamSet.setParameter( key, static_cast<bool>( val ) );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_mip_param_int64( PAPILO_SOLVER* solver, const char* key,
                                   int64_t val )
{
   try
   {
      solver->mipParamSet.setParameter( key, val );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_mip_param_char( PAPILO_SOLVER* solver, const char* key,
                                  char val )
{
   try
   {
      solver->mipParamSet.setParameter( key, val );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_mip_param_string( PAPILO_SOLVER* solver, const char* key,
                                    const char* val )
{
   try
   {
      solver->mipParamSet.setParameter( key, val );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_lp_param_real( PAPILO_SOLVER* solver, const char* key,
                                 double val )
{
   try
   {
      solver->lpParamSet.setParameter( key, val );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_lp_param_int( PAPILO_SOLVER* solver, const char* key,
                                int val )
{
   try
   {
      solver->lpParamSet.setParameter( key, val );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_lp_param_bool( PAPILO_SOLVER* solver, const char* key,
                                 unsigned int val )
{
   try
   {
      solver->lpParamSet.setParameter( key, static_cast<bool>( val ) );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_lp_param_int64( PAPILO_SOLVER* solver, const char* key,
                                  int64_t val )
{
   try
   {
      solver->lpParamSet.setParameter( key, val );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_lp_param_char( PAPILO_SOLVER* solver, const char* key,
                                 char val )
{
   try
   {
      solver->lpParamSet.setParameter( key, val );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

PAPILO_PARAM_RESULT
papilo_solver_set_lp_param_string( PAPILO_SOLVER* solver, const char* key,
                                   const char* val )
{
   try
   {
      solver->lpParamSet.setParameter( key, val );
   }
   catch( const std::invalid_argument& )
   {
      return PAPILO_PARAM_NOT_FOUND;
   }
   catch( const std::domain_error& )
   {
      return PAPILO_PARAM_WRONG_TYPE;
   }
   catch( const std::out_of_range& )
   {
      return PAPILO_PARAM_INVALID_VALUE;
   }

   return PAPILO_PARAM_CHANGED;
}

void
papilo_solver_load_problem( PAPILO_SOLVER* solver, PAPILO_PROBLEM* problem )
{
   assert( solver->state == SolverState::INIT );

   solver->problem = problem->problemBuilder.build();

   solver->solveinfo.dualbound = -problem->infinity;
   solver->solveinfo.bestsol_obj = problem->infinity;
   solver->solveinfo.bestsol_boundviol = problem->infinity;
   solver->solveinfo.bestsol_consviol = problem->infinity;
   solver->solveinfo.bestsol_intviol = problem->infinity;
   solver->solveinfo.solve_result = PAPILO_SOLVE_RESULT_STOPPED;
   solver->solveinfo.solvingtime = 0.0;
   solver->solveinfo.presolvetime = 0.0;

   solver->state = SolverState::PROBLEM_LOADED;
}

void
papilo_solver_write_mps( PAPILO_SOLVER* solver, const char* filename )
{
   assert( solver->state == SolverState::PROBLEM_LOADED );

#ifdef PAPILO_MPS_WRITER
   Vec<int> rowmapping( solver->problem.getNRows() );

   for( int i = 0; i != solver->problem.getNRows(); ++i )
      rowmapping[i] = i;

   Vec<int> colmapping( solver->problem.getNCols() );
   for( int i = 0; i != solver->problem.getNCols(); ++i )
      colmapping[i] = i;

   MpsWriter<double>::writeProb( filename, solver->problem, rowmapping,
                                 colmapping );
#else
   solver->presolve.message().warn(
       "cannot write problem to {}: MPS writer not available\n", filename );
#endif
}

void
papilo_solver_set_num_threads( PAPILO_SOLVER* solver, int numthreads )
{
   solver->presolve.getPresolveOptions().threads = std::max( 0, numthreads );
}

PAPILO_SOLVING_INFO*
papilo_solver_start( PAPILO_SOLVER* solver )
{
   switch( solver->state )
   {
   default:
   case SolverState::INIT:
      assert( false );
      break;
   case SolverState::PROBLEM_LOADED:
      solver->presolveResult = solver->presolve.apply( solver->problem );
      solver->state = SolverState::PROBLEM_PRESOLVED;
      solver->solveinfo.presolvetime =
          solver->presolve.getStatistics().presolvetime;
      solver->presolve.message().info(
          "presolving finished after {:.3f} seconds\n\n",
          solver->solveinfo.presolvetime );
   case SolverState::PROBLEM_PRESOLVED:
      switch( solver->presolveResult.status )
      {
      case PresolveStatus::kUnchanged:
      case PresolveStatus::kReduced:
         break;
      case PresolveStatus::kInfeasible:
         solver->solveinfo.solve_result = PAPILO_SOLVE_RESULT_INFEASIBLE;
         return &solver->solveinfo;
      case PresolveStatus::kUnbndOrInfeas:
         solver->solveinfo.solve_result = PAPILO_SOLVE_RESULT_UNBND_OR_INFEAS;
         return &solver->solveinfo;
      case PresolveStatus::kUnbounded:
         solver->solveinfo.solve_result = PAPILO_SOLVE_RESULT_UNBOUNDED;
         return &solver->solveinfo;
      }
   case SolverState::PROBLEM_SOLVED:
      SolverStatus status;
      SolverInterface<double>* solverInterface = nullptr;

      if( solver->problem.getNCols() > 0 )
      {
         if( solver->presolveResult.postsolve.getOriginalProblem()
                 .getNumIntegralCols() == 0 )
            solverInterface = solver->lpSolver.get();
         else
            solverInterface = solver->mipSolver.get();

         if( solverInterface == nullptr )
         {
            solver->presolve.message().error(
                "no solver available for solving\n" );

            solver->solveinfo.solve_result = PAPILO_SOLVE_RESULT_STOPPED;
            break;
         }

         solverInterface->setVerbosity( solver->presolve.getVerbosityLevel() );

         if( solver->presolve.getPresolveOptions().tlim !=
             std::numeric_limits<double>::max() )
         {
            double tlim = solver->presolve.getPresolveOptions().tlim -
                          solver->solveinfo.presolvetime;
            if( tlim <= 0 )
            {
               solver->presolve.message().info(
                   "time limit reached in presolving\n" );

               solver->solveinfo.solve_result = PAPILO_SOLVE_RESULT_STOPPED;
               break;
            }

            solverInterface->setTimeLimit( tlim );
         }

         if( solver->tlimsoft != std::numeric_limits<double>::max() )
         {
            solverInterface->setSoftTimeLimit( std::max(
                0.0, solver->tlimsoft - solver->solveinfo.presolvetime ) );
         }

         solver->state = SolverState::PROBLEM_SOLVED;

         {
            Timer t( solver->solveinfo.solvingtime );

            solverInterface->setUp(
                solver->problem,
                solver->presolveResult.postsolve.origrow_mapping,
                solver->presolveResult.postsolve.origcol_mapping );

            solverInterface->solve();

            if( static_cast<int>( solver->presolve.getVerbosityLevel() ) >=
                static_cast<int>( VerbosityLevel::kInfo ) )
               solverInterface->printDetails();
         }
         status = solverInterface->getStatus();

         if( status == SolverStatus::kInfeasible )
            fmt::print( "\nsolving detected infeasible problem after {:.3f} seconds\n",
                        solver->solveinfo.solvingtime + solver->solveinfo.presolvetime );
         else if( status == SolverStatus::kUnbounded )
            fmt::print( "\nsolving detected unbounded problem after {:.3f} seconds\n",
                        solver->solveinfo.solvingtime + solver->solveinfo.presolvetime );
         else if( status == SolverStatus::kUnbndOrInfeas )
            fmt::print( "\nsolving detected unbounded or infeasible problem after "
                        "{:.3f} seconds\n",
                        solver->solveinfo.solvingtime + solver->solveinfo.presolvetime );
         else
            fmt::print( "\nsolving finished after {:.3f} seconds\n",
                        solver->solveinfo.solvingtime + solver->solveinfo.presolvetime );
      }
      else
      {
         status = SolverStatus::kOptimal;
         solver->presolve.message().info( "problem solved in presolving\n" );
      }

      Solution<double> solution;
      Postsolve<double> postsolve{ solver->presolve.message(),
                                   solver->presolveResult.postsolve.num };

      switch( status )
      {
      case SolverStatus::kOptimal:
         solver->solveinfo.solve_result = PAPILO_SOLVE_RESULT_STOPPED;

         if( solverInterface != nullptr &&
             !solverInterface->getSolution( solution ) )
            break;
         if( postsolve.undo( solution, solver->solution,
                             solver->presolveResult.postsolve ) ==
             PostsolveStatus::kOk )
         {
            solver->solveinfo.bestsol = solver->solution.primal.data();
            solver->solveinfo.solve_result = PAPILO_SOLVE_RESULT_OPTIMAL;

            // compute objective
            solver->solveinfo.bestsol_obj =
                solver->presolveResult.postsolve.problem.computeSolObjective(
                    solver->solution.primal );

            solver->solveinfo.dualbound = solverInterface == nullptr
                                              ? solver->solveinfo.bestsol_obj
                                              : solverInterface->getDualBound();

            // compute violations
            solver->presolveResult.postsolve.problem.computeSolViolations(
                solver->presolveResult.postsolve.num, solver->solution.primal,
                solver->solveinfo.bestsol_boundviol,
                solver->solveinfo.bestsol_consviol,
                solver->solveinfo.bestsol_intviol );
         }

         break;
      case SolverStatus::kInterrupted:
         solver->solveinfo.solve_result = PAPILO_SOLVE_RESULT_STOPPED;

         if( solverInterface != nullptr )
            solver->solveinfo.dualbound = solverInterface->getDualBound();

         if( solverInterface != nullptr &&
             !solverInterface->getSolution( solution ) )
            break;

         if( postsolve.undo( solution, solver->solution,
                             solver->presolveResult.postsolve ) ==
             PostsolveStatus::kOk )
         {
            solver->solveinfo.bestsol = solver->solution.primal.data();
            solver->solveinfo.solve_result = PAPILO_SOLVE_RESULT_FEASIBLE;

            // compute objective
            solver->solveinfo.bestsol_obj =
                solver->presolveResult.postsolve.problem.computeSolObjective(
                    solver->solution.primal );

            // compute violations
            solver->presolveResult.postsolve.problem.computeSolViolations(
                solver->presolveResult.postsolve.num, solver->solution.primal,
                solver->solveinfo.bestsol_boundviol,
                solver->solveinfo.bestsol_consviol,
                solver->solveinfo.bestsol_intviol );
         }

         break;
      case SolverStatus::kInfeasible:
         solver->solveinfo.solve_result = PAPILO_SOLVE_RESULT_INFEASIBLE;
         break;
      case SolverStatus::kUnbounded:
         solver->solveinfo.solve_result = PAPILO_SOLVE_RESULT_UNBOUNDED;
         break;
      case SolverStatus::kUnbndOrInfeas:
         solver->solveinfo.solve_result = PAPILO_SOLVE_RESULT_UNBND_OR_INFEAS;
         break;
      case SolverStatus::kInit:
         assert( false );
      case SolverStatus::kError:
         solver->presolve.message().error(
             "solver terminated with an error\n" );
         solver->solveinfo.solve_result = PAPILO_SOLVE_RESULT_STOPPED;
      }
   }

   return &solver->solveinfo;
}
