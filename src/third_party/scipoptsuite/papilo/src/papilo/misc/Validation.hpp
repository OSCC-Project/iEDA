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

namespace papilo
{

#include "papilo/core/postsolve/Postsolve.hpp"
#include "papilo/core/postsolve/PostsolveStorage.hpp"

template <typename REAL>
struct Validation
{
   static void
   validateProblem( const Problem<REAL>& problem,
                    const PostsolveStorage<REAL>& postsolve,
                    const std::string& optimal_solution_file,
                    const PresolveStatus status )
   {
      Solution<REAL> optimal_solution;
      bool success = parse_solution( postsolve, optimal_solution_file, optimal_solution );

      if( success &&
          ( status == PresolveStatus::kUnchanged ||
            status == PresolveStatus::kReduced ) &&
          check_if_solution_is_contained_in_problem( problem, postsolve,
                                                     optimal_solution ) &&
          can_reduced_solution_be_recalculated( problem, postsolve,
                                                optimal_solution ) )
         fmt::print( "validation: SUCCESS\n" );
      else
         fmt::print( "validation: FAILURE\n" );
   }

 private:
   static bool
   can_reduced_solution_be_recalculated(
       const Problem<REAL>& problem, const PostsolveStorage<REAL>& postsolveStorage,
       const Solution<REAL>& optimal_solution )
   {
      bool validation_succcess = true;
      Vec<REAL> vec{};
      for( int i = 0; i < (int) postsolveStorage.origcol_mapping.size(); i++ )
         vec.push_back( optimal_solution.primal[postsolveStorage.origcol_mapping[i]] );

      Solution<REAL> calculated_orig_solution{};
      Solution<REAL> reducedSolution = Solution<REAL>( vec );
      const Message msg{};
      Postsolve<REAL> postsolve{msg, postsolveStorage.getNum()};
      postsolve.undo( reducedSolution, calculated_orig_solution,
                      postsolveStorage );
      for( int i = 0; i < (int) postsolveStorage.nColsOriginal; i++ )
      {
         if( ! postsolveStorage.getNum().isFeasEq( optimal_solution.primal[i],
                 calculated_orig_solution.primal[i] ) )
         {
            fmt::print(
                "postsolve for variable {} not equal {} !={}\n",
                problem.getVariableNames()[i],
                (double) optimal_solution.primal[i],
                (double) calculated_orig_solution.primal[i] );
            validation_succcess = false;
         }
      }
      return validation_succcess;
   }

   static bool
   parse_solution( const PostsolveStorage<REAL>& postsolveStorage,
                   const std::string& optimal_solution_file,
                   Solution<REAL>& optimal_solution)
   {
      SolParser<REAL> parser;

      std::vector<int> one_to_one_mapping;
      for( int i = 0; i < (int) postsolveStorage.nColsOriginal; i++ )
         one_to_one_mapping.push_back( i );

      return parser.read(
          optimal_solution_file, one_to_one_mapping,
          postsolveStorage.getOriginalProblem().getVariableNames(), optimal_solution );
   }

   static bool
   check_if_solution_is_contained_in_problem(
       const Problem<REAL>& problem, const PostsolveStorage<REAL>& postsolve,
       const Solution<REAL> optimal_solution )
   {
      bool validation_success = true;

      for( int i = 0; i < problem.getNCols(); i++ )
      {
         REAL solution_coeff =
             optimal_solution.primal[postsolve.origcol_mapping[i]];
         if( !problem.getColFlags()[i].test( ColFlag::kUbInf ) &&
             !postsolve.getNum().isFeasLE( solution_coeff,
                                           problem.getUpperBounds()[i] ) )
         {
            fmt::print(
                "lb {} of var {} violates bounds for value {} ",
                (double) problem.getLowerBounds()[i],
                problem.getVariableNames()[postsolve.origcol_mapping[i]],
                (double) solution_coeff );
            validation_success = false;
         }
         if( !problem.getColFlags()[i].test( ColFlag::kLbInf ) &&
             !postsolve.getNum().isFeasGE( solution_coeff,
                                           problem.getLowerBounds()[i] ) )
         {
            fmt::print(
                "ub {} of var {} violates bounds for value {} ",
                (double) problem.getUpperBounds()[i],
                problem.getVariableNames()[postsolve.origcol_mapping[i]],
                (double) solution_coeff );
            validation_success = false;
         }
      }

      for( int i = 0; i < problem.getConstraintMatrix().getNRows(); i++ )
      {
         REAL row_value = calculateRowValueForSolution( problem, postsolve,
                                                        optimal_solution, i );

         if( problem.getConstraintMatrix().getRowFlags()[i].test(
                 RowFlag::kEquation ) &&
             !postsolve.getNum().isFeasEq(
                 row_value,
                 problem.getConstraintMatrix().getRightHandSides()[i] ) )
         {
            fmt::print(
                "equality in row {} is violated: {} != {}\n",
                problem.getConstraintNames()[postsolve.origrow_mapping[i]],
                (double) row_value,
                (double) problem.getConstraintMatrix().getRightHandSides()[i] );
            validation_success = false;
         }
         else
         {
            if( !problem.getConstraintMatrix().getRowFlags()[i].test(
                    RowFlag::kRhsInf ) &&
                !postsolve.getNum().isFeasLE(
                    row_value,
                    problem.getConstraintMatrix().getRightHandSides()[i] ) )
            {
               fmt::print(
                   "LE inequality in row {} is violated: {} !<= {}\n",
                   problem.getConstraintNames()[postsolve.origrow_mapping[i]],
                   (double) row_value,
                   (double) problem.getConstraintMatrix().getRightHandSides()[i] );
               validation_success = false;
            }

            if( !problem.getConstraintMatrix().getRowFlags()[i].test(
                    RowFlag::kLhsInf ) &&
                !postsolve.getNum().isFeasGE(
                    (double) row_value,
                    (double) problem.getConstraintMatrix().getLeftHandSides()[i] ) )
            {
               fmt::print(
                   "GE inequality in row {} is violated: {} !>= {}\n",
                   problem.getConstraintNames()[postsolve.origrow_mapping[i]],
                   (double) row_value,
                   (double) problem.getConstraintMatrix().getLeftHandSides()[i] );
               validation_success = false;
            }
         }
      }
      return validation_success;
   }

   static REAL
   calculateRowValueForSolution( const Problem<REAL>& problem,
                                 const PostsolveStorage<REAL>& postsolve,
                                 const Solution<REAL>& optimal_solution, int i )
   {
      const SparseVectorView<REAL>& row =
          problem.getConstraintMatrix().getRowCoefficients( i );
      REAL row_value = 0;
      for( int j = 0; j < row.getLength(); ++j )
      {
         int col_index = row.getIndices()[j];
         REAL solution_coeff =
             optimal_solution.primal[postsolve.origcol_mapping[col_index]];
         REAL col_value = row.getValues()[j];
         row_value += solution_coeff * col_value;
      }
      return row_value;
   }
};

} // namespace papilo
