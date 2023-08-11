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

#ifndef _PAPILO_CORE_PRESOLVE_OPTIONS_HPP_
#define _PAPILO_CORE_PRESOLVE_OPTIONS_HPP_

#include "papilo/misc/ParameterSet.hpp"
#include <type_traits>

namespace papilo
{

struct PresolveOptions
{
   // add the parameters with their default values
   int maxfillinpersubstitution = 10;

   double markowitz_tolerance = 0.01;

   int maxshiftperrow = 10;

   bool substitutebinarieswithints = true;

   int dualreds = 2;

   double abortfac = 8e-4;

   double lpabortfac = 1e-2;

   bool boundrelax = false;

   int componentsmaxint = 0;

   double compressfac = 0.85;

   double tlim = std::numeric_limits<double>::max();

   double minabscoeff = 1e-10;

   double feastol = 1e-6;

   double epsilon = 1e-9;

   double hugeval = 1e8;

   bool removeslackvars = true;

   int weakenlpvarbounds = 0;

   int detectlindep = 1;

   int threads = 0;

   unsigned int randomseed = 0;

   bool apply_results_immediately_if_run_sequentially = true;

   bool dual_fix_parallel = false;

   bool simple_probing_parallel = false;

   bool implied_integer_parallel = false;

   bool simple_substitution_parallel = false;

   bool constraint_propagation_parallel = true;

   bool coefficient_strengthening_parallel = true;

   bool simplify_inequalities_parallel = true;

   bool calculate_basis_for_dual = true;

   double bound_tightening_offset = 0.0001;

   bool validation_after_every_postsolving_step = false;

   void
   addParameters( ParameterSet& paramSet )
   {
      paramSet.addParameter(
          "presolve.dualreds",
          "0: disable dual reductions, 1: allow dual reductions that never cut "
          "off optimal solutions, 2: allow all dual reductions",
          dualreds, 0, 2 );
      paramSet.addParameter(
          "substitution.markowitz_tolerance",
          "markowitz tolerance value for allowing a substitution",
          markowitz_tolerance, 0.0, 1.0 );
      paramSet.addParameter(
          "presolve.abortfac",
          "abort factor of weighted number of reductions for presolving",
          abortfac, 0.0, 1.0 );
      paramSet.addParameter(
          "presolve.lpabortfac",
          "abort factor of weighted number of reductions for presolving LPs",
          lpabortfac, 0.0, 1.0 );
      paramSet.addParameter(
          "substitution.maxfillin",
          "maximum estimated fillin for variable substitutions",
          maxfillinpersubstitution, 0 );
      paramSet.addParameter( "presolve.randomseed", "random seed value",
                             randomseed );
      paramSet.addParameter( "substitution.maxshiftperrow",
                             "maximum amount of nonzeros being moved to make "
                             "space for fillin from substitutions within a row",
                             maxshiftperrow, 0 );
      paramSet.addParameter( "substitution.binarieswithints",
                             "should substitution of binary variables with "
                             "general integers be allowed",
                             substitutebinarieswithints );
      paramSet.addParameter(
          "presolve.boundrelax",
          "relax bounds of implied free variables after presolving",
          boundrelax );
      paramSet.addParameter( "presolve.removeslackvars",
                             "remove slack variables in equations",
                             removeslackvars );
      paramSet.addParameter(
          "presolve.componentsmaxint",
          "maximum number of integral variables for trying to solve "
          "disconnected components of the problem in presolving (-1: disabled)",
          componentsmaxint, -1 );
      paramSet.addParameter( "presolve.compressfac",
                             "compress the problem if fewer than compressfac "
                             "times the number of rows or columns are active",
                             compressfac, 0.0, 1.0 );
      paramSet.addParameter( "presolve.tlim", "time limit for presolve", tlim,
                             0.0 );
      paramSet.addParameter( "presolve.minabscoeff",
                             "minimum absolute coefficient value allowed in "
                             "matrix, before it is set to zero",
                             minabscoeff, 0.0, 1e-1 );
      paramSet.addParameter( "numerics.feastol", "the feasibility tolerance",
                             feastol, 0.0, 1e-1 );
      paramSet.addParameter( "numerics.epsilon",
                             "epsilon tolerance to consider two values equal",
                             epsilon, 0.0, 1e-1 );
      paramSet.addParameter( "numerics.hugeval",
                             "absolute bound value that is considered too huge "
                             "for activitity based calculations",
                             hugeval, 0.0 );
      paramSet.addParameter(
          "presolve.weakenlpvarbounds",
          "weaken bounds obtained by constraint propagation by this factor of "
          "the feasibility tolerance if the problem is an LP",
          weakenlpvarbounds );
      paramSet.addParameter( "presolve.detectlindep",
                             "detect and remove linearly dependent equations "
                             "and free columns (0: off, 1: for LPs, 2: always)",
                             detectlindep, 0, 2 );
      paramSet.addParameter( "presolve.threads",
                             "maximal number of threads to use (0: automatic)",
                             threads, 0 );
      paramSet.addParameter(
          "presolve.apply_results_immediately_if_run_sequentially",
          "# if only one thread (presolve.threads = 1) is used, apply the "
          "reductions immediately afterwards",
          apply_results_immediately_if_run_sequentially );
      paramSet.addParameter(
          "propagation.parallel",
          "#execute loop over rows in constraintpropagation in parallel",
          constraint_propagation_parallel );
      paramSet.addParameter(
          "coefftightening.parallel",
          "#execute loop over rows in coefficientstrengthening in parallel",
          coefficient_strengthening_parallel );
      paramSet.addParameter(
          "dualfix.parallel",
          "#execute loop over columns in dualfix in parallel",
          dual_fix_parallel );
      paramSet.addParameter(
          "implint.parallel",
          "#execute loop over rows in impliedint in parallel",
          implied_integer_parallel );
      paramSet.addParameter(
          "simpleprobing.parallel",
          "#execute loop over rows in simpleprobing in parallel",
          simple_probing_parallel );
      paramSet.addParameter(
          "doubletoneq.parallel",
          "#execute loop over rows in simplesubstitution in parallel",
          simple_substitution_parallel );
      paramSet.addParameter(
          "simplifyineq.parallel",
          "#execute loop over rows in simplifyineq in parallel",
          simplify_inequalities_parallel );
      paramSet.addParameter(
          "calculate_basis_for_dual",
          "#if basis for LP should be calculated presolving steps tightening the variable bounds can not be applied.",
          calculate_basis_for_dual );
      paramSet.addParameter(
          "validation_after_every_postsolving_step",
          "# should the primal/dual solution be validated during after every postsolving step? ",
          validation_after_every_postsolving_step );
      paramSet.addParameter(
          "bound_tightening_offset",
          "# defines the offset for bound tightening ",
          bound_tightening_offset );
   }

   bool
   runs_sequential() const
   {
      return threads == 1;
   }

   double
   get_variable_bound_tightening_offset() const
   {
      if( bound_tightening_offset > feastol )
         return bound_tightening_offset;
      return feastol * 10;
   };
};

} // namespace papilo

#endif
