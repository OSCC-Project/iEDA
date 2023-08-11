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

#ifndef _PAPILO_CORE_PRESOLVE_HPP_
#define _PAPILO_CORE_PRESOLVE_HPP_

#include <algorithm>
#include <cctype>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <utility>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "papilo/core/PresolveMethod.hpp"
#include "papilo/core/PresolveOptions.hpp"
#include "papilo/core/Problem.hpp"
#include "papilo/core/ProblemUpdate.hpp"
#include "papilo/core/Statistics.hpp"
#include "papilo/core/postsolve/Postsolve.hpp"
#include "papilo/core/postsolve/PostsolveStorage.hpp"
#include "papilo/interfaces/SolverInterface.hpp"
#include "papilo/io/Message.hpp"
#include "papilo/misc/DependentRows.hpp"
#include "papilo/misc/ParameterSet.hpp"
#include "papilo/misc/Timer.hpp"
#include "papilo/misc/Vec.hpp"
#ifdef PAPILO_TBB
#include "papilo/misc/tbb.hpp"
#endif
#include "papilo/presolvers/CoefficientStrengthening.hpp"
#include "papilo/presolvers/ConstraintPropagation.hpp"
#include "papilo/presolvers/DominatedCols.hpp"
#include "papilo/presolvers/DualFix.hpp"
#include "papilo/presolvers/DualInfer.hpp"
#include "papilo/presolvers/FixContinuous.hpp"
#include "papilo/presolvers/FreeVarSubstitution.hpp"
#include "papilo/presolvers/ImplIntDetection.hpp"
#include "papilo/presolvers/ParallelColDetection.hpp"
#include "papilo/presolvers/ParallelRowDetection.hpp"
#include "papilo/presolvers/Probing.hpp"
#include "papilo/presolvers/SimpleProbing.hpp"
#include "papilo/presolvers/SimpleSubstitution.hpp"
#include "papilo/presolvers/SimplifyInequalities.hpp"
#include "papilo/presolvers/SingletonCols.hpp"
#include "papilo/presolvers/SingletonStuffing.hpp"
#include "papilo/presolvers/Sparsify.hpp"

namespace papilo
{

template <typename REAL>
struct PresolveResult
{
   PostsolveStorage<REAL> postsolve;
   PresolveStatus status;
};

enum class Delegator
{
   kAbort,
   kFast,
   kMedium,
   kExhaustive,
   kExceeded
};

template <typename REAL>
class Presolve
{
 public:
   void

   addDefaultPresolvers()
   {
      using uptr = std::unique_ptr<PresolveMethod<REAL>>;

      // fast presolvers
      addPresolveMethod( uptr( new SingletonCols<REAL>() ) );
      addPresolveMethod( uptr( new CoefficientStrengthening<REAL>() ) );
      addPresolveMethod( uptr( new ConstraintPropagation<REAL>() ) );

      //medium presolvers
      addPresolveMethod( uptr( new SimpleProbing<REAL>() ) );
      addPresolveMethod( uptr( new ParallelRowDetection<REAL>() ) );
      addPresolveMethod( uptr( new ParallelColDetection<REAL>() ) );
      addPresolveMethod( uptr( new SingletonStuffing<REAL>() ) );
      addPresolveMethod( uptr( new DualFix<REAL>() ) );
      addPresolveMethod( uptr( new FixContinuous<REAL>() ) );
      addPresolveMethod( uptr( new SimplifyInequalities<REAL>() ) );
      addPresolveMethod( uptr( new SimpleSubstitution<REAL>() ) );

      //exhaustive presolvers
      addPresolveMethod( uptr( new ImplIntDetection<REAL>() ) );
      addPresolveMethod( uptr( new DominatedCols<REAL>() ) );
      addPresolveMethod( uptr( new DualInfer<REAL> ) );
      addPresolveMethod( uptr( new Probing<REAL>() ) );
      addPresolveMethod( uptr( new Substitution<REAL>() ) );
      addPresolveMethod( uptr( new Sparsify<REAL>() ) );
   }

   ParameterSet
   getParameters()
   {
      ParameterSet paramSet;
      msg.addParameters( paramSet );
      presolveOptions.addParameters( paramSet );

      for( const std::unique_ptr<PresolveMethod<REAL>>& presolver : presolvers )
         presolver->addParameters( paramSet );

      return paramSet;
   }

   /***
    * presolves the problem and applies the reductions found by the presolvers
    * immediately to it.
    * The functions returns the PresolveStatus (Reduced, Unchanged, ) and the
    * postsolve information
    *
    * @tparam REAL: computational accuracy template
    * @param problem: the problem to be presolved
    * @return: presolved problem and PresolveResult contains postsolve
    * information
    */
   PresolveResult<REAL>
   apply( Problem<REAL>& problem, bool store_dual_postsolve = true );

   /// add presolve method to presolving
   void
   addPresolveMethod( std::unique_ptr<PresolveMethod<REAL>> presolveMethod )
   {
      presolvers.emplace_back( std::move( presolveMethod ) );
   }

   void
   setLPSolverFactory( std::unique_ptr<SolverFactory<REAL>> value )
   {
      this->lpSolverFactory = std::move( value );
   }

   void
   setMIPSolverFactory( std::unique_ptr<SolverFactory<REAL>> value )
   {
      this->mipSolverFactory = std::move( value );
   }

   const std::unique_ptr<SolverFactory<REAL>>&
   getLPSolverFactory() const
   {
      return this->lpSolverFactory;
   }

   const std::unique_ptr<SolverFactory<REAL>>&
   getMIPSolverFactory() const
   {
      return this->mipSolverFactory;
   }

   void
   setPresolverOptions( const PresolveOptions& value )
   {
      this->presolveOptions = value;
   }

   const PresolveOptions&
   getPresolveOptions() const
   {
      return this->presolveOptions;
   }

   PresolveOptions&
   getPresolveOptions()
   {
      return this->presolveOptions;
   }

   /// get epsilon value for numerical comparisons
   const REAL&
   getEpsilon() const
   {
      return num.getEpsilon();
   }

   /// get feasibility tolerance value
   const REAL&
   getFeasTol() const
   {
      return num.getFeasTol();
   }

   /// set the verbosity level
   void
   setVerbosityLevel( VerbosityLevel verbosity )
   {
      msg.setVerbosityLevel( verbosity );
   }

   /// get the verbosity level
   VerbosityLevel
   getVerbosityLevel() const
   {
      return msg.getVerbosityLevel();
   }

   const Message&
   message() const
   {
      return msg;
   }

   Message&
   message()
   {
      return msg;
   }

   /// access statistics of presolving
   const Statistics&
   getStatistics() const
   {
      return stats;
   }

   std::pair<int, int>
   applyReductions( int p, const Reductions<REAL>& reductions_,
                    ProblemUpdate<REAL>& probUpdate );

 private:
   // data to perform presolving
   Vec<PresolveStatus> results;
   Vec<std::unique_ptr<PresolveMethod<REAL>>> presolvers;
   Vec<Reductions<REAL>> reductions;
   Delegator round_to_evaluate;

   Vec<std::pair<const Reduction<REAL>*, const Reduction<REAL>*>>
       postponedReductions;
   Vec<int> postponedReductionToPresolver;

   // settings for presolve behavior
   Num<REAL> num;
   Message msg;
   PresolveOptions presolveOptions;
   // statistics
   Statistics stats;

   std::unique_ptr<SolverFactory<REAL>> lpSolverFactory;
   std::unique_ptr<SolverFactory<REAL>> mipSolverFactory;

   Vec<std::pair<int, int>> presolverStats;
   bool lastRoundReduced;
   int nunsuccessful;
   bool rundelayed;

   /// evaluate result array of each presolver, return the largest result value
   PresolveStatus
   evaluateResults();

   void
   finishRound( ProblemUpdate<REAL>& probUpdate );

   void
   applyPostponed( ProblemUpdate<REAL>& probUpdate );

   Delegator
   determine_next_round( Problem<REAL>& problem,
                         ProblemUpdate<REAL>& probUpdate,
                         const Statistics& roundStats,
                         const Timer& presolvetimer, bool unchanged = false );

   PresolveStatus
   apply_all_presolver_reductions( ProblemUpdate<REAL>& probUpdate );

   void
   printRoundStats( bool unchanged, std::string rndtype );

   void
   printPresolversStats();

 private:
   void
   logStatus( const Problem<REAL>& problem,
              const PostsolveStorage<REAL>& postsolveStorage ) const;

   bool
   is_time_exceeded( const Timer& presolvetimer ) const;

   bool
   is_only_slighlty_changes( const Problem<REAL>& problem,
                             const ProblemUpdate<REAL>& probUpdate,
                             const Statistics& roundStats ) const;

   Delegator
   increase_delegator( Delegator delegator );

   std::string
   get_round_type( Delegator delegator );

   Delegator
   increase_round_if_last_run_was_not_successfull(
       const Problem<REAL>& problem, const ProblemUpdate<REAL>& probUpdate,
       const Statistics& roundStats, bool unchanged );

   Delegator
   handle_case_exceeded( Delegator& next_round );

   PresolveStatus
   evaluate_and_apply( const Timer& timer, Problem<REAL>& problem,
                       PresolveResult<REAL>& result,
                       ProblemUpdate<REAL>& probUpdate,
                       const Statistics& oldstats, bool run_sequential );

   void
   apply_reduction_of_solver( ProblemUpdate<REAL>& probUpdate,
                              size_t index_presolver );

   void
   apply_result_sequential( int index_presolver,
                             ProblemUpdate<REAL>& probUpdate,
                             bool& run_sequential );

   void
   run_presolvers( const Problem<REAL>& problem,
                   const std::pair<int, int>& presolver_2_run,
                   ProblemUpdate<REAL>& probUpdate, bool& run_sequential, const Timer& timer );

   bool
   is_status_infeasible_or_unbounded( const PresolveStatus& status ) const;

   bool
   are_only_dual_postsolve_presolvers_enabled();
};

#ifdef PAPILO_USE_EXTERN_TEMPLATES
extern template class Presolve<double>;
extern template class Presolve<Quad>;
extern template class Presolve<Rational>;
#endif

/***
 * presolves the problem and applies the reductions found by the presolvers
 * immediately to it.
 * The functions returns the PresolveStatus (Reduced, Unchanged, ) and the
 * postsolve information
 *
 * @tparam REAL: computational accuracy template
 * @param problem: the problem to be presolved
 * @param store_dual_postsolve: should dual postsolve reductions stored in the postsolve stack
 * @return: presolved problem and PresolveResult contains postsolve information
 */
template <typename REAL>
PresolveResult<REAL>
Presolve<REAL>::apply( Problem<REAL>& problem, bool store_dual_postsolve )
{
#ifdef PAPILO_TBB
   tbb::task_arena arena( presolveOptions.threads == 0
                              ? tbb::task_arena::automatic
                              : presolveOptions.threads );
#endif

#ifdef PAPILO_TBB
   return arena.execute( [this, &problem, store_dual_postsolve]() {
#endif
      stats = Statistics();
      num.setFeasTol( REAL{ presolveOptions.feastol } );
      num.setEpsilon( REAL{ presolveOptions.epsilon } );
      num.setHugeVal( REAL{ presolveOptions.hugeval } );

      Timer timer( stats.presolvetime );

      ConstraintMatrix<REAL>& constraintMatrix = problem.getConstraintMatrix();
      Vec<REAL>& rhsVals = constraintMatrix.getRightHandSides();
      Vec<RowFlags>& rflags = constraintMatrix.getRowFlags();
      const Vec<int>& rowsize = constraintMatrix.getRowSizes();

      msg.info( "\nstarting presolve of problem {}:\n", problem.getName() );
      msg.info( "  rows:     {}\n", problem.getNRows() );
      msg.info( "  columns:  {}\n", problem.getNCols() );
      msg.info( "  int. columns:  {}\n", problem.getNumIntegralCols() );
      msg.info( "  cont. columns:  {}\n", problem.getNumContinuousCols() );
      msg.info( "  nonzeros: {}\n\n", problem.getConstraintMatrix().getNnz() );

      PresolveResult<REAL> result;

      result.postsolve =
          PostsolveStorage<REAL>( problem, num, presolveOptions );

#ifndef PAPILO_TBB
      if( presolveOptions.threads != 1 )
         msg.warn( "PaPILO without TBB can only use one thread. Number of "
                   "threads is set to 1\n" );
      presolveOptions.threads = 1;
#endif

      if( store_dual_postsolve && problem.getNumIntegralCols() == 0 )
      {
         if( presolveOptions.componentsmaxint == -1 && presolveOptions.detectlindep == 0 &&
             are_only_dual_postsolve_presolvers_enabled())
            result.postsolve.postsolveType = PostsolveType::kFull;
         else
         {
            msg.error(
                "Please turn off the presolvers substitution and sparsify and "
                "componentsdetection to use dual postsolving\n" );
            return result;
         }
      }
      result.status = PresolveStatus::kUnchanged;

      std::stable_sort( presolvers.begin(), presolvers.end(),
                        []( const std::unique_ptr<PresolveMethod<REAL>>& a,
                            const std::unique_ptr<PresolveMethod<REAL>>& b ) {
                           return static_cast<int>( a->getTiming() ) <
                                  static_cast<int>( b->getTiming() );
                        } );

      std::pair<int, int> fastPresolvers;
      std::pair<int, int> mediumPresolvers;
      std::pair<int, int> exhaustivePresolvers;

      int npresolvers = static_cast<int>( presolvers.size() );

      fastPresolvers.first = fastPresolvers.second = 0;
      while( fastPresolvers.second < npresolvers &&
             presolvers[fastPresolvers.second]->getTiming() ==
                 PresolverTiming::kFast )
         ++fastPresolvers.second;

      mediumPresolvers.first = mediumPresolvers.second = fastPresolvers.second;
      while( mediumPresolvers.second < npresolvers &&
             presolvers[mediumPresolvers.second]->getTiming() ==
                 PresolverTiming::kMedium )
         ++mediumPresolvers.second;

      exhaustivePresolvers.first = exhaustivePresolvers.second =
          mediumPresolvers.second;
      while( exhaustivePresolvers.second < npresolvers &&
             presolvers[exhaustivePresolvers.second]->getTiming() ==
                 PresolverTiming::kExhaustive )
         ++exhaustivePresolvers.second;

      reductions.resize( presolvers.size() );
      results.resize( presolvers.size() );

      round_to_evaluate = Delegator::kFast;

      presolverStats.resize( presolvers.size(), std::pair<int, int>( 0, 0 ) );

      ProblemUpdate<REAL> probUpdate( problem, result.postsolve, stats,
                                      presolveOptions, num, msg );

      for( int i = 0; i != npresolvers; ++i )
      {
         if( presolvers[i]->isEnabled() )
         {
            if( presolvers[i]->initialize( problem, presolveOptions ) )
               probUpdate.observeCompress( presolvers[i].get() );
         }
      }

      result.status = probUpdate.trivialPresolve();

      if( result.status == PresolveStatus::kInfeasible ||
          result.status == PresolveStatus::kUnbndOrInfeas ||
          result.status == PresolveStatus::kUnbounded )
         return result;

      printRoundStats( false, "Trivial" );
      round_to_evaluate = Delegator::kFast;

      finishRound( probUpdate );
      ++stats.nrounds;

      nunsuccessful = 0;
      rundelayed = true;
      for( int i = 0; i < npresolvers; ++i )
      {
         if( presolvers[i]->isEnabled() && presolvers[i]->isDelayed() )
         {
            rundelayed = false;
            break;
         }
      }

      Statistics last_rounds_stats = stats;
      do
      {
         bool was_executed_sequential = false;
         // if problem is trivial abort here
         if( probUpdate.getNActiveCols() == 0 ||
             probUpdate.getNActiveRows() == 0 )
            break;

         switch( round_to_evaluate )
         {
         case Delegator::kFast:
            run_presolvers( problem, fastPresolvers, probUpdate,
                            was_executed_sequential, timer );
            break;
         case Delegator::kMedium:
            run_presolvers( problem, mediumPresolvers, probUpdate,
                            was_executed_sequential, timer );
            break;
         case Delegator::kExhaustive:
            run_presolvers( problem, exhaustivePresolvers, probUpdate,
                            was_executed_sequential, timer );
            break;
         default:
            assert( false );
         }

         result.status =
             evaluate_and_apply( timer, problem, result, probUpdate,
                                 last_rounds_stats, was_executed_sequential );
         if( is_status_infeasible_or_unbounded( result.status ) )
            return result;
         last_rounds_stats = stats;

      } while( round_to_evaluate != Delegator::kAbort );

      if( stats.ntsxapplied > 0 || stats.nboundchgs > 0 ||
          stats.ncoefchgs > 0 || stats.ndeletedcols > 0 ||
          stats.ndeletedrows > 0 || stats.nsidechgs > 0 )
      {
         result.status = probUpdate.trivialPresolve();

         if( result.status == PresolveStatus::kInfeasible ||
             result.status == PresolveStatus::kUnbndOrInfeas ||
             result.status == PresolveStatus::kUnbounded )
            return result;

         probUpdate.clearStates();
         probUpdate.check_and_compress();
      }

      printPresolversStats();

      if( DependentRows<REAL>::Enabled &&
          ( presolveOptions.detectlindep == 2 ||
            ( problem.getNumIntegralCols() == 0 &&
              presolveOptions.detectlindep == 1 ) ) )
      {
         ConstraintMatrix<REAL>& consMatrix = problem.getConstraintMatrix();
         Vec<int> equations;

         equations.reserve( problem.getNRows() );
         size_t eqnnz = 0;

         for( int i = 0; i != problem.getNRows(); ++i )
         {
            if( rflags[i].test( RowFlag::kRedundant ) ||
                !rflags[i].test( RowFlag::kEquation ) )
               continue;

            equations.push_back( i );
            eqnnz += rowsize[i] + 1;
         }

         if( !equations.empty() )
         {
            DependentRows<REAL> depRows( equations.size(), problem.getNCols(),
                                         eqnnz );

            for( size_t i = 0; i != equations.size(); ++i )
               depRows.addRow( i, consMatrix.getRowCoefficients( equations[i] ),
                               REAL( rhsVals[equations[i]] ) );

            Vec<int> dependentEqs;
            double factorTime = 0.0;
            msg.info( "found {} equations, checking for linear dependency\n",
                      equations.size() );
            {
               Timer t{ factorTime };
               dependentEqs = depRows.getDependentRows( msg, num );
            }
            msg.info( "{} equations are redundant, factorization took {} "
                      "seconds\n",
                      dependentEqs.size(), factorTime );

            if( !dependentEqs.empty() )
            {
               for( int dependentEq : dependentEqs )
               {
                  probUpdate.markRowRedundant( equations[dependentEq] );
               }
               probUpdate.flush( true );
            }
         }

         if( presolveOptions.dualreds == 2 )
         {
            Vec<int> freeCols;
            freeCols.reserve( problem.getNCols() );
            size_t freeColNnz = 0;

            const Vec<ColFlags>& cflags = problem.getColFlags();
            const Vec<int>& colsize = problem.getColSizes();
            const Vec<REAL>& obj = problem.getObjective().coefficients;

            for( int col = 0; col != problem.getNCols(); ++col )
            {
               if( cflags[col].test( ColFlag::kInactive, ColFlag::kIntegral ) ||
                   !cflags[col].test( ColFlag::kLbInf ) ||
                   !cflags[col].test( ColFlag::kUbInf ) )
                  continue;

               freeCols.push_back( col );
               freeColNnz += colsize[col] + 1;
            }

            if( !freeCols.empty() )
            {
               DependentRows<REAL> depRows( freeCols.size(), problem.getNRows(),
                                            freeColNnz );

               for( size_t i = 0; i != freeCols.size(); ++i )
                  depRows.addRow(
                      i, consMatrix.getColumnCoefficients( freeCols[i] ),
                      obj[freeCols[i]] );

               Vec<int> dependentFreeCols;
               double factorTime = 0.0;
               msg.info(
                   "found {} free columns, checking for linear dependency\n",
                   freeCols.size(), freeColNnz );

               {
                  Timer t{ factorTime };
                  dependentFreeCols = depRows.getDependentRows( msg, num );
               }

               msg.info( "{} free columns are redundant, factorization took {} "
                         "seconds\n",
                         dependentFreeCols.size(), factorTime );

               if( !dependentFreeCols.empty() )
               {
                  for( int dependentFreeCol : dependentFreeCols )
                     probUpdate.fixCol( freeCols[dependentFreeCol], 0 );

                  probUpdate.flush( true );
               }
            }
         }
      }

      // finally compress problem fully and release excess storage even if
      // problem was not reduced
      probUpdate.compress( true );

      // check whether problem was reduced
      if( stats.ntsxapplied > 0 || stats.nboundchgs > 0 ||
          stats.ncoefchgs > 0 || stats.ndeletedcols > 0 ||
          stats.ndeletedrows > 0 || stats.nsidechgs > 0 )
      {
         if( presolveOptions.boundrelax && problem.getNumIntegralCols() == 0 )
         {
            int nremoved;
            int nnewfreevars;

            std::tie( nremoved, nnewfreevars ) =
                probUpdate.removeRedundantBounds();
            if( nremoved != 0 )
               msg.info( "removed {} redundant column bounds, got {} new free "
                         "variables\n",
                         nremoved, nnewfreevars );
         }

         bool detectComponents = presolveOptions.componentsmaxint != -1;

         if( !lpSolverFactory && problem.getNumContinuousCols() != 0 )
            detectComponents = false;

         if( !mipSolverFactory && problem.getNumIntegralCols() != 0 )
            detectComponents = false;

         if( problem.getNCols() == 0 )
            detectComponents = false;

         if( detectComponents  && probUpdate.getNActiveCols() > 0 )
         {
            assert( problem.getNCols() != 0 && problem.getNRows() != 0 );
            Components components;

            int ncomponents = components.detectComponents( problem );

            if( ncomponents > 1 )
            {
               const Vec<ComponentInfo>& compInfo =
                   components.getComponentInfo();

               msg.info( "found {} disconnected components\n", ncomponents );
               msg.info(
                   "largest component has {} cols ({} int., {} cont.) and "
                   "{} nonzeros\n",
                   compInfo[ncomponents - 1].nintegral +
                       compInfo[ncomponents - 1].ncontinuous,
                   compInfo[ncomponents - 1].nintegral,
                   compInfo[ncomponents - 1].ncontinuous,
                   compInfo[ncomponents - 1].nnonz );

               Solution<REAL> solution;
               solution.primal.resize( problem.getNCols() );
               Vec<uint8_t> componentSolved( ncomponents );

               if( result.postsolve.postsolveType == PostsolveType::kFull )
               {
                  solution.type = SolutionType::kPrimalDual;
                  solution.reducedCosts.resize( problem.getNCols() );
                  solution.dual.resize( problem.getNRows() );
                  solution.varBasisStatus.resize( problem.getNCols() );
               }

#ifdef PAPILO_TBB
               tbb::parallel_for(
                   tbb::blocked_range<int>( 0, ncomponents - 1 ),
                   [this, &components, &solution, &problem, &result, &compInfo,
                    &componentSolved,
                    &timer]( const tbb::blocked_range<int>& r ) {
                      for( int i = r.begin(); i != r.end(); ++i )
#else
               for( int i = 0; i < ncomponents - 1; ++i )

#endif
                      {
                         if( compInfo[i].nintegral == 0 )
                         {
                            std::unique_ptr<SolverInterface<REAL>> solver =
                                lpSolverFactory->newSolver(
                                    VerbosityLevel::kQuiet );

                            solver->setUp( problem,
                                           result.postsolve.origrow_mapping,
                                           result.postsolve.origcol_mapping,
                                           components, compInfo[i] );

                            if( presolveOptions.tlim !=
                                std::numeric_limits<double>::max() )
                            {
                               double tlim =
                                   presolveOptions.tlim - timer.getTime();
                               if( tlim <= 0 )
                                  break;
                               solver->setTimeLimit( tlim );
                            }

                            solver->solve();

                            SolverStatus status = solver->getStatus();

                            if( status == SolverStatus::kOptimal )
                            {
                               if( solver->getSolution( components,
                                                        compInfo[i].componentid,
                                                        solution ) )
                                  componentSolved[compInfo[i].componentid] =
                                      true;
                            }
                         }
                         else if( compInfo[i].nintegral <=
                                  presolveOptions.componentsmaxint )
                         {
                            std::unique_ptr<SolverInterface<REAL>> solver =
                                mipSolverFactory->newSolver(
                                    VerbosityLevel::kQuiet );

                            solver->setGapLimit( 0 );
                            solver->setNodeLimit(
                                problem.getConstraintMatrix().getNnz() /
                                std::max( compInfo[i].nnonz, 1 ) );

                            solver->setUp( problem,
                                           result.postsolve.origrow_mapping,
                                           result.postsolve.origcol_mapping,
                                           components, compInfo[i] );

                            if( presolveOptions.tlim !=
                                std::numeric_limits<double>::max() )
                            {
                               double tlim =
                                   presolveOptions.tlim - timer.getTime();
                               if( tlim <= 0 )
                                  break;
                               solver->setTimeLimit( tlim );
                            }

                            solver->solve();

                            SolverStatus status = solver->getStatus();

                            if( status == SolverStatus::kOptimal )
                            {
                               if( solver->getSolution( components,
                                                        compInfo[i].componentid,
                                                        solution ) )
                                  componentSolved[compInfo[i].componentid] =
                                      true;
                            }
                         }
                      }
#ifdef PAPILO_TBB
                   }
                   ,tbb::simple_partitioner() );
#endif

               int nsolved = 0;

               int oldndelcols = stats.ndeletedcols;
               int oldndelrows = stats.ndeletedrows;

               auto& lbs = problem.getLowerBounds();
               auto& ubs = problem.getUpperBounds();
               for( int i = 0; i != ncomponents; ++i )
               {
                  if( componentSolved[i] )
                  {
                     ++nsolved;

                     const int* compcols = components.getComponentsCols( i );
                     int numcompcols = components.getComponentsNumCols( i );

                     for( int j = 0; j != numcompcols; ++j )
                     {
                        const int col = compcols[j];
                        lbs[compcols[j]] = solution.primal[col];
                        ubs[compcols[j]] = solution.primal[col];
                        probUpdate.markColFixed( col );
                        if( result.postsolve.postsolveType ==
                            PostsolveType::kFull )
                           result.postsolve.storeDualValue(
                               true, col, solution.reducedCosts[col] );
                     }

                     const int* comprows = components.getComponentsRows( i );
                     int numcomprows = components.getComponentsNumRows( i );

                     for( int j = 0; j != numcomprows; ++j )
                     {
                        probUpdate.markRowRedundant( comprows[j] );
                     }
                  }
               }

               if( nsolved != 0 )
               {
                  if( probUpdate.flush( true ) == PresolveStatus::kInfeasible )
                     assert( false );

                  probUpdate.compress();

                  msg.info( "solved {} components: {} cols fixed, {} rows "
                            "deleted\n",
                            nsolved, stats.ndeletedcols - oldndelcols,
                            stats.ndeletedrows - oldndelrows );
               }
            }
         }

         logStatus( problem, result.postsolve );
         result.status = PresolveStatus::kReduced;
         if( result.postsolve.postsolveType == PostsolveType::kFull )
         {
            auto& coefficients = problem.getObjective().coefficients;
            auto& col_lower = problem.getLowerBounds();
            auto& col_upper = problem.getUpperBounds();
            auto& row_lhs = problem.getConstraintMatrix().getLeftHandSides();
            auto& row_rhs = problem.getConstraintMatrix().getRightHandSides();
            auto& row_flags = problem.getRowFlags();
            auto& col_flags = problem.getColFlags();

            result.postsolve.storeReducedBoundsAndCost(
                col_lower, col_upper, row_lhs, row_rhs, coefficients, row_flags,
                col_flags );
         }

         return result;
      }

      logStatus( problem, result.postsolve );

      // problem was not changed
      result.status = PresolveStatus::kUnchanged;
      return result;
#ifdef PAPILO_TBB
   } );
#endif
}

template <typename REAL>
void
Presolve<REAL>::run_presolvers( const Problem<REAL>& problem,
                                const std::pair<int, int>& presolver_2_run,
                                ProblemUpdate<REAL>& probUpdate,
                                bool& run_sequential, const Timer& timer )
{
#ifndef PAPILO_TBB
   assert(presolveOptions.runs_sequential() == true);
#endif
   if( presolveOptions.runs_sequential() &&
       presolveOptions.apply_results_immediately_if_run_sequentially )
   {
      probUpdate.setPostponeSubstitutions( false );
      for( int i = presolver_2_run.first; i != presolver_2_run.second; ++i )
      {
         results[i] =
             presolvers[i]->run( problem, probUpdate, num, reductions[i], timer );
         apply_result_sequential( i, probUpdate, run_sequential );
         if( results[i] == PresolveStatus::kInfeasible )
            return;
         if( problem.getNRows() == 0 || problem.getNCols() == 0 )
            return;
      }
      PresolveStatus status = probUpdate.trivialPresolve();
      if( is_status_infeasible_or_unbounded( status ) )
      {
         results[presolver_2_run.first] = status;
         return;
      }
      probUpdate.clearStates();
      probUpdate.check_and_compress();
   }
#ifdef PAPILO_TBB
   else
   {
      tbb::parallel_for(
          tbb::blocked_range<int>( presolver_2_run.first,
                                   presolver_2_run.second ),
          [&]( const tbb::blocked_range<int>& r ) {
             for( int i = r.begin(); i != r.end(); ++i )
             {
                results[i] = presolvers[i]->run( problem, probUpdate, num,
                                                 reductions[i], timer );
             }
          },
          tbb::simple_partitioner() );
   }
#endif
}

template <typename REAL>
void
Presolve<REAL>::apply_result_sequential( int index_presolver,
                                          ProblemUpdate<REAL>& probUpdate,
                                          bool& run_sequential )
{
   run_sequential = true;
   apply_reduction_of_solver( probUpdate, index_presolver );
   probUpdate.flushChangedCoeffs();
   if( probUpdate.flush( false ) == PresolveStatus::kInfeasible )
   {
      results[index_presolver] = PresolveStatus::kInfeasible;
      return;
   }
   probUpdate.clearStates();
}

template <typename REAL>
Delegator
Presolve<REAL>::determine_next_round( Problem<REAL>& problem,
                                      ProblemUpdate<REAL>& probUpdate,
                                      const Statistics& roundStats,
                                      const Timer& presolvetimer,
                                      bool unchanged )
{
   if( is_time_exceeded( presolvetimer ) )
      return Delegator::kAbort;

   Delegator next_round = increase_round_if_last_run_was_not_successfull(
       problem, probUpdate, roundStats, unchanged );

   next_round = handle_case_exceeded( next_round );

   assert( next_round != Delegator::kExceeded );
   return next_round;
}

template <typename REAL>
PresolveStatus
Presolve<REAL>::evaluate_and_apply( const Timer& timer, Problem<REAL>& problem,
                                    PresolveResult<REAL>& result,
                                    ProblemUpdate<REAL>& probUpdate,
                                    const Statistics& oldstats,
                                    bool run_sequential )
{
   if( round_to_evaluate == Delegator::kFast )
   {
      probUpdate.clearChangeInfo();
      lastRoundReduced = false;
   }

   result.status = evaluateResults();
   switch( result.status )
   {
   case PresolveStatus::kUnbndOrInfeas:
   case PresolveStatus::kUnbounded:
   case PresolveStatus::kInfeasible:
      printPresolversStats();
      return result.status;
   case PresolveStatus::kUnchanged:
      round_to_evaluate = determine_next_round(
          problem, probUpdate, ( stats - oldstats ), timer, true );
      return result.status;
   case PresolveStatus::kReduced:
      // problem reductions where found by at least one presolver
      PresolveStatus status;
      if( !run_sequential )
         status = apply_all_presolver_reductions( probUpdate );
      else
         status = PresolveStatus::kReduced;
      if( is_status_infeasible_or_unbounded( status ) )
         return status;
      round_to_evaluate = determine_next_round( problem, probUpdate,
                                                ( stats - oldstats ), timer );
      finishRound( probUpdate );
      return status;
   }
   return result.status;
}

template <typename REAL>
bool
Presolve<REAL>::is_status_infeasible_or_unbounded(
    const PresolveStatus& status ) const
{
   return status == PresolveStatus::kUnbndOrInfeas ||
          status == PresolveStatus::kUnbounded ||
          status == PresolveStatus::kInfeasible;
}

template <typename REAL>
PresolveStatus
Presolve<REAL>::apply_all_presolver_reductions(
    ProblemUpdate<REAL>& probUpdate )
{
   probUpdate.setPostponeSubstitutions( true );

   postponedReductionToPresolver.push_back( 0 );

   for( std::size_t i = 0; i < presolvers.size(); ++i )
   {
      apply_reduction_of_solver( probUpdate, i );
      postponedReductionToPresolver.push_back( postponedReductions.size() );
   }

   PresolveStatus status = evaluateResults();
   if( is_status_infeasible_or_unbounded( status ) )
      return status;

   probUpdate.flushChangedCoeffs();

   applyPostponed( probUpdate );

   return probUpdate.flush( true );
}

template <typename REAL>
void
Presolve<REAL>::apply_reduction_of_solver( ProblemUpdate<REAL>& probUpdate,
                                           size_t index_presolver )
{
   if( results[index_presolver] != PresolveStatus::kReduced )
      return;

   Message::debug( this, "applying reductions of presolver {}\n",
                   presolvers[index_presolver]->getName() );

   auto statistics = applyReductions( index_presolver,
                                      reductions[index_presolver], probUpdate );

   // if infeasible it returns -1 -1
   if( statistics.first >= 0 && statistics.second >= 0 )
   {
      presolverStats[index_presolver].first += statistics.first;
      presolverStats[index_presolver].second += statistics.second;
   }
   else
      results[index_presolver] = PresolveStatus::kInfeasible;
}

template <typename REAL>
std::pair<int, int>
Presolve<REAL>::applyReductions( int p, const Reductions<REAL>& reductions_,
                                 ProblemUpdate<REAL>& probUpdate )
{
   int k = 0;
   ApplyResult result;
   int nbtsxAppliedStart = stats.ntsxapplied;
   int nbtsxTotal = 0;

   const auto& reds = reductions_.getReductions();

   msg.detailed( "Presolver {} applying \n", presolvers[p]->getName() );

   for( const auto& transaction : reductions_.getTransactions() )
   {
      int start = transaction.start;
      int end = transaction.end;

      for( ; k != start; ++k )
      {
         result = probUpdate.applyTransaction( &reds[k], &reds[k + 1] );
         if( result == ApplyResult::kApplied )
            ++stats.ntsxapplied;
         else if( result == ApplyResult::kRejected )
            ++stats.ntsxconflicts;
         else if( result == ApplyResult::kInfeasible )
            return std::make_pair( -1, -1 );
         else if( result == ApplyResult::kPostponed )
            postponedReductions.emplace_back( &reds[k], &reds[k + 1] );

         ++nbtsxTotal;
      }

      result = probUpdate.applyTransaction( &reds[start], &reds[end] );
      if( result == ApplyResult::kApplied )
         ++stats.ntsxapplied;
      else if( result == ApplyResult::kRejected )
         ++stats.ntsxconflicts;
      else if( result == ApplyResult::kInfeasible )
         return std::make_pair( -1, -1 );
      else if( result == ApplyResult::kPostponed )
         postponedReductions.emplace_back( &reds[start], &reds[end] );

      k = end;
      ++nbtsxTotal;
   }

   for( ; k != static_cast<int>( reds.size() ); ++k )
   {
      result = probUpdate.applyTransaction( &reds[k], &reds[k + 1] );
      if( result == ApplyResult::kApplied )
         ++stats.ntsxapplied;
      else if( result == ApplyResult::kRejected )
         ++stats.ntsxconflicts;
      else if( result == ApplyResult::kInfeasible )
         return std::make_pair( -1, -1 );
      else if( result == ApplyResult::kPostponed )
         postponedReductions.emplace_back( &reds[k], &reds[k + 1] );

      ++nbtsxTotal;
   }

   return { nbtsxTotal, ( stats.ntsxapplied - nbtsxAppliedStart ) };
}

template <typename REAL>
void
Presolve<REAL>::applyPostponed( ProblemUpdate<REAL>& probUpdate )
{
   probUpdate.setPostponeSubstitutions( false );

   for( int presolver = 0; presolver != (int) presolvers.size(); ++presolver )
   {
      int first = postponedReductionToPresolver[presolver];
      int last = postponedReductionToPresolver[presolver + 1];
      if( first < last )
         msg.detailed( "Presolver {} applying \n",
                       presolvers[presolver]->getName() );
      for( int i = first; i != last; ++i )
      {
         const auto& ptrpair = postponedReductions[i];

         ApplyResult r =
             probUpdate.applyTransaction( ptrpair.first, ptrpair.second );
         if( r == ApplyResult::kApplied )
         {
            ++stats.ntsxapplied;
            ++presolverStats[presolver].second;
         }
         else if( r == ApplyResult::kRejected )
            ++stats.ntsxconflicts;
      }
   }

   postponedReductions.clear();
   postponedReductionToPresolver.clear();
}

template <typename REAL>
void
Presolve<REAL>::finishRound( ProblemUpdate<REAL>& probUpdate )
{
   probUpdate.clearStates();
   probUpdate.check_and_compress();

   for( auto& reduction : reductions )
      reduction.clear();

   std::fill( results.begin(), results.end(), PresolveStatus::kUnchanged );
}

template <typename REAL>
Delegator
Presolve<REAL>::handle_case_exceeded( Delegator& next_round )
{
   if( next_round != Delegator::kExceeded )
      return next_round;

   ++nunsuccessful;

   if( !( rundelayed && ( !lastRoundReduced || nunsuccessful == 2 ) ) )
   {
      printRoundStats( !lastRoundReduced, "Exhaustive" );
      if( !rundelayed )
      {
         msg.info( "activating delayed presolvers\n" );
         for( auto& p : presolvers )
            p->setDelayed( false );
         rundelayed = true;
      }
      ++stats.nrounds;
      return Delegator::kFast;
   }
   printRoundStats( !lastRoundReduced, get_round_type( next_round ) );
   return Delegator::kAbort;
}

template <typename REAL>
bool
Presolve<REAL>::is_time_exceeded( const Timer& presolvetimer ) const
{
   return presolveOptions.tlim != std::numeric_limits<double>::max() &&
          presolvetimer.getTime() >= presolveOptions.tlim;
}

template <typename REAL>
bool
Presolve<REAL>::is_only_slighlty_changes( const Problem<REAL>& problem,
                                          const ProblemUpdate<REAL>& probUpdate,
                                          const Statistics& roundStats ) const
{
   double abort_factor = problem.getNumIntegralCols() == 0
                             ? presolveOptions.lpabortfac
                             : presolveOptions.abortfac;
   return ( 0.1 * roundStats.nboundchgs + roundStats.ndeletedcols ) <=
              abort_factor * probUpdate.getNActiveCols() &&
          ( roundStats.nsidechgs + roundStats.ndeletedrows ) <=
              abort_factor * probUpdate.getNActiveRows() &&
          ( roundStats.ncoefchgs <=
            abort_factor * problem.getConstraintMatrix().getNnz() );
}

template <typename REAL>
Delegator
Presolve<REAL>::increase_round_if_last_run_was_not_successfull(
    const Problem<REAL>& problem, const ProblemUpdate<REAL>& probUpdate,
    const Statistics& roundStats, bool unchanged )
{
   Delegator next_round;
   if( !unchanged )
   {
      if( is_only_slighlty_changes( problem, probUpdate, roundStats ) )
      {
         lastRoundReduced =
             lastRoundReduced || roundStats.nsidechgs > 0 ||
             roundStats.nboundchgs > 0 || roundStats.ndeletedcols > 0 ||
             roundStats.ndeletedrows > 0 || roundStats.ncoefchgs > 0;
         next_round = increase_delegator( round_to_evaluate );
      }
      else
      {
         printRoundStats( false, get_round_type( round_to_evaluate ) );
         lastRoundReduced = true;
         next_round = Delegator::kFast;
         nunsuccessful = 0;
         ++stats.nrounds;
      }
   }
   else
      next_round = increase_delegator( round_to_evaluate );
   return next_round;
}

template <typename REAL>
Delegator
Presolve<REAL>::increase_delegator( Delegator delegator )
{
   switch( delegator )
   {
   case Delegator::kFast:
      return Delegator::kMedium;
   case Delegator::kMedium:
      return Delegator::kExhaustive;
   case Delegator::kAbort:
   case Delegator::kExhaustive:
   case Delegator::kExceeded:
      break;
   }
   return Delegator::kExceeded;
}

template <typename REAL>
PresolveStatus
Presolve<REAL>::evaluateResults()
{
   int largestValue = static_cast<int>( PresolveStatus::kUnchanged );

   for( auto& i : results )
      largestValue = std::max( largestValue, static_cast<int>( i ) );

   return static_cast<PresolveStatus>( largestValue );
}

template <typename REAL>
void
Presolve<REAL>::printRoundStats( bool unchanged, std::string rndtype )
{

   if( unchanged )
   {
      msg.info( "round {:<3} ({:^10}): Unchanged\n", stats.nrounds, rndtype );
      return;
   }

   msg.info( "round {:<3} ({:^10}): {:>4} del cols, {:>4} del rows, "
             "{:>4} chg bounds, {:>4} chg sides, {:>4} chg coeffs, "
             "{:>4} tsx applied, {:>4} tsx conflicts\n",
             stats.nrounds, rndtype, stats.ndeletedcols, stats.ndeletedrows,
             stats.nboundchgs, stats.nsidechgs, stats.ncoefchgs,
             stats.ntsxapplied, stats.ntsxconflicts );
}
template <typename REAL>
void
Presolve<REAL>::printPresolversStats()
{
   msg.info( "presolved {} rounds: {:>4} del cols, {:>4} del rows, "
             "{:>4} chg bounds, {:>4} chg sides, {:>4} chg coeffs, "
             "{:>4} tsx applied, {:>4} tsx conflicts\n",
             stats.nrounds, stats.ndeletedcols, stats.ndeletedrows,
             stats.nboundchgs, stats.nsidechgs, stats.ncoefchgs,
             stats.ntsxapplied, stats.ntsxconflicts );
   msg.info( "\n {:>18} {:>12} {:>18} {:>18} {:>18} {:>18} \n", "presolver",
             "nb calls", "success calls(%)", "nb transactions",
             "tsx applied(%)", "execution time(s)" );
   for( std::size_t i = 0; i < presolvers.size(); ++i )
   {
      presolvers[i]->printStats( msg, presolverStats[i] );
   }

   msg.info( "\n" );
}

template <typename REAL>
void
Presolve<REAL>::logStatus( const Problem<REAL>& problem,
                           const PostsolveStorage<REAL>& postsolveStorage ) const
{
   if(msg.getVerbosityLevel() == VerbosityLevel::kQuiet)
      return;
   msg.info( "reduced problem:\n" );
   msg.info( "  reduced rows:     {}\n", problem.getNRows() );
   msg.info( "  reduced columns:  {}\n", problem.getNCols() );
   msg.info( "  reduced int. columns:  {}\n", problem.getNumIntegralCols() );
   msg.info( "  reduced cont. columns:  {}\n", problem.getNumContinuousCols() );
   msg.info( "  reduced nonzeros: {}\n",
             problem.getConstraintMatrix().getNnz() );
   if( problem.getNCols() == 0)
   {
      // the primaldual can be disabled therefore calculate only primal for obj
      Solution<REAL> solution{};
      SolutionType type = postsolveStorage.postsolveType == PostsolveType::kFull
                              ? SolutionType::kPrimalDual
                              : SolutionType::kPrimal;
      Solution<REAL> empty_sol{ type };
      Postsolve<REAL> postsolve{ msg, num };
      postsolve.undo( empty_sol, solution, postsolveStorage );
      const Problem<REAL>& origprob = postsolveStorage.getOriginalProblem();
      REAL origobj = origprob.computeSolObjective( solution.primal );
      msg.info(
          "problem is solved [optimal solution found] [objective value: {} (double precision)]\n",
          (double) origobj );
   }
}

template <typename REAL>
std::string
Presolve<REAL>::get_round_type( Delegator delegator )
{
   switch( delegator )
   {
   case Delegator::kFast:
      return "Fast";
   case Delegator::kMedium:
      return "Medium";
   case Delegator::kExhaustive:
      return "Exhaustive";
   case Delegator::kExceeded:
      return "Final";
   case Delegator::kAbort:
      break;
   }
   return "Undefined";
}

template <typename REAL>
bool
Presolve<REAL>::are_only_dual_postsolve_presolvers_enabled()
{
   for( int i = 0; i < (int) presolvers.size(); i++ )
   {
      if( presolvers[i]->isEnabled() )
      {
         if( presolvers[i]->getName().compare( "substitution" ) == 0 ||
             presolvers[i]->getName().compare( "sparsify" ) == 0 ||
             presolvers[i]->getName().compare( "dualinfer" ) == 0 ||
             presolvers[i]->getName().compare( "doubletoneq" ) == 0 )
            return false;
      }
   }
   return true;
}

} // namespace papilo

#endif
