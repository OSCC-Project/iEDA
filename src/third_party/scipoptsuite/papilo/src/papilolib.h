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

#ifndef __PAPILOLIB_H__
#define __PAPILOLIB_H__

#include "papilolib_export.h"

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>
#include <stdint.h>

   /// Enum type to specify the rows with a single side value
   typedef enum Papilo_RowType
   {
      PAPILO_ROW_TYPE_GREATER = 0,
      PAPILO_ROW_TYPE_LESSER = 1,
      PAPILO_ROW_TYPE_EQUAL = 2,
   } PAPILO_ROW_TYPE;

   typedef struct Papilo_Problem PAPILO_PROBLEM;

   /// Create a new problem datastructure. Reserves space for the given number
   /// of rows, columns, and nonzeros to prevent unnecessary reallocation
   /// during problem creation. The given value of infinity is used to
   /// determine the finiteness of bounds and rowsides. The name pointer and
   /// the space hints are optional and can be NULL/0.
   PAPILOLIB_EXPORT PAPILO_PROBLEM*
   papilo_problem_create( double infinity, const char* name, int nnz_hint,
                          int row_hint, int col_hint );

   /// Free the problem datastructure
   PAPILOLIB_EXPORT void
   papilo_problem_free( PAPILO_PROBLEM* );

   /// Add columns (variables) to the problem and returns the index of the first
   /// new column or -1 if it does not exist (i.e. num == 0).
   /// If colnames is NULL generic names are used.
   PAPILOLIB_EXPORT int
   papilo_problem_add_cols( PAPILO_PROBLEM* problem, int num, const double* lb,
                            const double* ub, const unsigned char* integral,
                            const double* obj, const char** colnames );

   /// Adds a column (variable) to the problem and returns its index.
   /// If colnames is NULL a generic name is used.
   PAPILOLIB_EXPORT int
   papilo_problem_add_col( PAPILO_PROBLEM* problem, double lb, double ub,
                           unsigned char integral, double obj,
                           const char* colname );

   /// returns number of columns
   PAPILOLIB_EXPORT int
   papilo_problem_get_num_cols( PAPILO_PROBLEM* problem );

   /// returns number of rows
   PAPILOLIB_EXPORT int
   papilo_problem_get_num_rows( PAPILO_PROBLEM* problem );

   /// returns number of columns
   PAPILOLIB_EXPORT int
   papilo_problem_get_num_nonzeros( PAPILO_PROBLEM* problem );

   /// Change columns lower bound
   PAPILOLIB_EXPORT void
   papilo_problem_change_col_lb( PAPILO_PROBLEM* problem, int col, double lb );

   /// Change columns upper bound
   PAPILOLIB_EXPORT void
   papilo_problem_change_col_ub( PAPILO_PROBLEM* problem, int col, double ub );

   /// Change columns integrality restrictions
   PAPILOLIB_EXPORT void
   papilo_problem_change_col_integral( PAPILO_PROBLEM* problem, int col,
                                       unsigned char integral );

   /// Change columns objective coefficient
   PAPILOLIB_EXPORT void
   papilo_problem_change_col_obj( PAPILO_PROBLEM* problem, int col,
                                  double obj );

   /// Add simple linear constraints (no ranged rows) to the problem and returns
   /// the index of the first new row or -1 if it does not exist (i.e. num ==
   /// 0). If rownames is NULL generic names are used.
   PAPILOLIB_EXPORT int
   papilo_problem_add_simple_rows( PAPILO_PROBLEM* problem, int num,
                                   const unsigned char* rowtypes,
                                   const double* side, const char** rownames );

   /// Add linear constraints with given left and right hand sides to the
   /// problem  and returns the
   /// index of the first new row or -1 if it does not exist (i.e. num == 0).
   /// If rownames is NULL generic names are used. The rows can be a ranged row
   /// if both lhs and rhs are finite.
   PAPILOLIB_EXPORT int
   papilo_problem_add_generic_rows( PAPILO_PROBLEM* problem, int num,
                                    const double* lhs, const double* rhs,
                                    const char** rownames );

   /// Add one simple linear constraint (no ranged row) to the problem and
   /// returns its index. If rowname is NULL generic names are used.
   PAPILOLIB_EXPORT int
   papilo_problem_add_simple_row( PAPILO_PROBLEM* problem,
                                  unsigned char rowtype, double side,
                                  const char* rowname );

   /// Add one linear constraints with given left and right hand side to the
   /// problem  and returns its index. If the rowname is NULL generic names are
   /// used. The row can be a ranged row if both lhs and rhs are finite.
   PAPILOLIB_EXPORT int
   papilo_problem_add_generic_row( PAPILO_PROBLEM* problem, double lhs,
                                   double rhs, const char* rowname );

   /// Add a nonzero entry for the given column to the given constraint
   PAPILOLIB_EXPORT void
   papilo_problem_add_nonzero( PAPILO_PROBLEM* problem, int row, int col,
                               double val );

   /// Add the nonzero entries for the given columns to the given row
   PAPILOLIB_EXPORT void
   papilo_problem_add_nonzeros_row( PAPILO_PROBLEM* problem, int row, int num,
                                    const int* cols, const double* vals );

   /// Add the nonzero entries for the given column to the given rows
   PAPILOLIB_EXPORT void
   papilo_problem_add_nonzeros_col( PAPILO_PROBLEM* problem, int col, int num,
                                    const int* rows, const double* vals );

   /// Add the nonzero entries given in compressed sparse row format. The array
   /// rowstart must have size at least nrows + 1 and the arrays cols and vals
   /// must have size at least rowstart[nrows].
   PAPILOLIB_EXPORT void
   papilo_problem_add_nonzeros_csr( PAPILO_PROBLEM* problem,
                                    const int* rowstart, const int* cols,
                                    const double* vals );

   /// Add the nonzero entries given in compressed sparse column format. The
   /// array colstart must have size at least ncols + 1 and the arrays rows and
   /// vals must have size at least colstart[ncols].
   PAPILOLIB_EXPORT void
   papilo_problem_add_nonzeros_csc( PAPILO_PROBLEM* problem,
                                    const int* colstart, const int* rows,
                                    const double* vals );

   /// Solver type for presolve library
   typedef struct Papilo_Solver PAPILO_SOLVER;

   /// Create a new solver datastructure
   PAPILOLIB_EXPORT PAPILO_SOLVER*
   papilo_solver_create();

   /// Set callback for message output. If the callback is set to NULL output is
   /// printed to stdout. The level argument is set as follows:
   /// 1 - errors, 2 - warnings, 3 - info, 4 - detailed
   PAPILOLIB_EXPORT void
   papilo_solver_set_trace_callback( PAPILO_SOLVER* solver,
                                     void ( *thetracecb )( int level,
                                                           const char* data,
                                                           size_t size,
                                                           void* usrptr ),
                                     void* usrptr );

   /// Free the problem datastructure
   PAPILOLIB_EXPORT void
   papilo_solver_free( PAPILO_SOLVER* );

   /// Return values for setting solver parameters
   typedef enum Papilo_ParamResult
   {
      /// parameter was successfully changed
      PAPILO_PARAM_CHANGED = 0,

      /// parameter does not exist
      PAPILO_PARAM_NOT_FOUND = 1,

      /// parameter is of a different type
      PAPILO_PARAM_WRONG_TYPE = 2,

      /// parameter was set to an invalid value
      PAPILO_PARAM_INVALID_VALUE = 3,

   } PAPILO_PARAM_RESULT;

   /// Set bool parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_param_bool( PAPILO_SOLVER* solver, const char* key,
                                 unsigned int val );

   /// Set real parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_param_real( PAPILO_SOLVER* solver, const char* key,
                                 double val );

   /// Set integer parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_param_int( PAPILO_SOLVER* solver, const char* key,
                                int val );

   /// Set character parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_param_char( PAPILO_SOLVER* solver, const char* key,
                                 char val );

   /// Set string parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_param_string( PAPILO_SOLVER* solver, const char* key,
                                   const char* val );

   /// Set real parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_mip_param_real( PAPILO_SOLVER* solver, const char* key,
                                     double val );

   /// Set integer parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_mip_param_int( PAPILO_SOLVER* solver, const char* key,
                                    int val );

   /// Set boolean parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_mip_param_bool( PAPILO_SOLVER* solver, const char* key,
                                     unsigned int val );

   /// Set 64bit integer parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_mip_param_int64( PAPILO_SOLVER* solver, const char* key,
                                      int64_t val );

   /// Set character parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_mip_param_char( PAPILO_SOLVER* solver, const char* key,
                                     char val );

   /// Set string parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_mip_param_string( PAPILO_SOLVER* solver, const char* key,
                                       const char* val );

   /// Set real parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_lp_param_real( PAPILO_SOLVER* solver, const char* key,
                                    double val );

   /// Set integer parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_lp_param_int( PAPILO_SOLVER* solver, const char* key,
                                   int val );

   /// Set boolean parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_lp_param_bool( PAPILO_SOLVER* solver, const char* key,
                                    unsigned int val );

   /// Set 64bit integer parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_lp_param_int64( PAPILO_SOLVER* solver, const char* key,
                                     int64_t val );

   /// Set character parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_lp_param_char( PAPILO_SOLVER* solver, const char* key,
                                    char val );

   /// Set string parameter with given key to the given value
   PAPILOLIB_EXPORT PAPILO_PARAM_RESULT
   papilo_solver_set_lp_param_string( PAPILO_SOLVER* solver, const char* key,
                                      const char* val );

   /// Return values for solving
   typedef enum Papilo_SolveResult
   {
      /// problem was solved to optimality
      PAPILO_SOLVE_RESULT_OPTIMAL = 0,

      /// solving stopped early with a feasible solution due to limits or
      /// interrupts
      PAPILO_SOLVE_RESULT_FEASIBLE = 1,

      /// solving stopped early and without a solution due to limits or
      /// interrupts
      PAPILO_SOLVE_RESULT_STOPPED = 2,

      /// problem was detected to be unbounded or infeasible
      PAPILO_SOLVE_RESULT_UNBND_OR_INFEAS = 3,

      /// problem was detected to be unbounded
      PAPILO_SOLVE_RESULT_UNBOUNDED = 4,

      /// problem was detected to be infeasible
      PAPILO_SOLVE_RESULT_INFEASIBLE = 5,

   } PAPILO_SOLVE_RESULT;

   /// Load the problem into the solver, the problem datastructure will be empty
   /// afterwards and can be reused to build another problem
   PAPILOLIB_EXPORT void
   papilo_solver_load_problem( PAPILO_SOLVER* solver, PAPILO_PROBLEM* problem );

   /// Write the problem to an mps file. Must be called after the problem
   /// was loaded into the solver. If the filename ends with .gz or .bz2 the
   /// file is written compressed
   PAPILOLIB_EXPORT void
   papilo_solver_write_mps( PAPILO_SOLVER* solver, const char* filename );

   /// Set the maximum number of threads to be used when the solver is
   /// started. Can be -1 to determine the number of threads automatically
   /// based on the hardware (default behavior), or a number larger or equal
   /// to 1.
   PAPILOLIB_EXPORT
   void
   papilo_solver_set_num_threads( PAPILO_SOLVER* solver, int numthreads );

   typedef struct
   {
      PAPILO_SOLVE_RESULT solve_result;
      const double* bestsol;
      double bestsol_obj;
      double bestsol_intviol;
      double bestsol_boundviol;
      double bestsol_consviol;
      double dualbound;
      double presolvetime;
      double solvingtime;
   } PAPILO_SOLVING_INFO;

   /// Start the solving process
   PAPILOLIB_EXPORT PAPILO_SOLVING_INFO*
   papilo_solver_start( PAPILO_SOLVER* solver );

#ifdef __cplusplus
}
#endif

#endif