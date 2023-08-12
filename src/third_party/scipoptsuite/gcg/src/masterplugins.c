/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program                         */
/*          GCG --- Generic Column Generation                                */
/*                  a Dantzig-Wolfe decomposition based extension            */
/*                  of the branch-cut-and-price framework                    */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/* Copyright (C) 2010-2022 Operations Research, RWTH Aachen University       */
/*                         Zuse Institute Berlin (ZIB)                       */
/*                                                                           */
/* This program is free software; you can redistribute it and/or             */
/* modify it under the terms of the GNU Lesser General Public License        */
/* as published by the Free Software Foundation; either version 3            */
/* of the License, or (at your option) any later version.                    */
/*                                                                           */
/* This program is distributed in the hope that it will be useful,           */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU Lesser General Public License for more details.                       */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with this program; if not, write to the Free Software               */
/* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.*/
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   masterplugins.c
 * @brief  SCIP plugins for generic column generation
 * @author Gerald Gamrath
 * @author Martin Bergner
 */

/*--+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#define USEHEURS 1
#define USESEPA 0
#define USEPROP 1

#include "masterplugins.h"

#include "scip/cons_and.h"
#include "scip/cons_bounddisjunction.h"
#include "scip/cons_conjunction.h"
#include "scip/cons_integral.h"
#include "scip/cons_indicator.h"
#include "scip/cons_knapsack.h"
#include "scip/cons_linear.h"
#include "scip/cons_logicor.h"
#include "scip/cons_or.h"
#include "scip/cons_setppc.h"
#include "scip/cons_varbound.h"
#include "scip/cons_xor.h"

#if USEHEURS
#include "scip/heur_adaptivediving.h"
#include "scip/heur_actconsdiving.h"
#include "scip/heur_alns.h"
#include "scip/heur_bound.h"
#include "scip/heur_clique.h"
#include "scip/heur_coefdiving.h"
#include "scip/heur_completesol.h"
#include "scip/heur_conflictdiving.h"
#include "scip/heur_crossover.h"
#include "scip/heur_dins.h"
#include "scip/heur_distributiondiving.h"
#include "scip/heur_dps.h"
#include "scip/heur_dualval.h"
#include "scip/heur_farkasdiving.h"
#include "scip/heur_feaspump.h"
#include "scip/heur_fixandinfer.h"
#include "scip/heur_fracdiving.h"
#include "scip/heur_gins.h"
#include "scip/heur_guideddiving.h"
#include "scip/heur_indicator.h"
#include "scip/heur_intdiving.h"
#include "scip/heur_intshifting.h"
#include "scip/heur_linesearchdiving.h"
#include "scip/heur_localbranching.h"
#include "scip/heur_locks.h"
#include "scip/heur_lpface.h"
#include "scip/heur_mpec.h"
#include "scip/heur_multistart.h"
#include "scip/heur_mutation.h"
#include "scip/heur_nlpdiving.h"
#include "scip/heur_ofins.h"
#include "scip/heur_objpscostdiving.h"
#include "scip/heur_octane.h"
#include "scip/heur_oneopt.h"
#include "scip/heur_padm.h"
#include "scip/heur_proximity.h"
#include "scip/heur_pscostdiving.h"
#include "scip/heur_randrounding.h"
#include "scip/heur_rens.h"
#include "scip/heur_reoptsols.h"
#include "scip/heur_repair.h"
#include "scip/heur_rins.h"
#include "scip/heur_rootsoldiving.h"
#include "scip/heur_rounding.h"
#include "scip/heur_shiftandpropagate.h"
#include "scip/heur_shifting.h"
#include "scip/heur_simplerounding.h"
#include "scip/heur_subnlp.h"
#include "scip/heur_trivial.h"
#include "scip/heur_trivialnegation.h"
#include "scip/heur_trustregion.h"
#include "scip/heur_trysol.h"
#include "scip/heur_twoopt.h"
#include "scip/heur_undercover.h"
#include "scip/heur_vbounds.h"
#include "scip/heur_veclendiving.h"
#include "scip/heur_zirounding.h"
#include "scip/heur_zeroobj.h"
#endif

#include "scip/presol_implics.h"
#include "scip/presol_inttobinary.h"
#include "presol_roundbound.h"
#include "scip/presol_boundshift.h"

#if USEPROP
#include "scip/prop_dualfix.h"
#include "scip/prop_genvbounds.h"
#include "scip/prop_probing.h"
#include "scip/prop_pseudoobj.h"
#include "scip/prop_rootredcost.h"
#include "scip/prop_redcost.h"
#include "scip/prop_vbounds.h"
#endif

#if USESEPA
#include "scip/sepa_clique.h"
#include "scip/sepa_cmir.h"
#include "scip/sepa_flowcover.h"
#include "scip/sepa_gomory.h"
#include "scip/sepa_impliedbounds.h"
#include "scip/sepa_intobj.h"
#include "scip/sepa_mcf.h"
#include "scip/sepa_oddcycle.h"
#include "scip/sepa_strongcg.h"
#endif

/* Jonas' stuff */
#include "sepa_basis.h"

#include "scip/reader_cip.h"
#include "scip/reader_lp.h"
#include "scip/scipshell.h"

/* GCG specific stuff */
#include "pricer_gcg.h"
#include "nodesel_master.h"
#include "cons_masterbranch.h"
#include "cons_integralorig.h"
#include "sepa_master.h"
#include "branch_ryanfoster.h"
#include "branch_orig.h"
#include "branch_relpsprob.h"
#include "branch_generic.h"
#include "branch_bpstrong.h"
#include "scip/debug.h"
#include "dialog_master.h"
#include "disp_master.h"
#include "solver_knapsack.h"
#include "solver_mip.h"
#include "event_bestsol.h"
#include "event_relaxsol.h"
#include "event_solvingstats.h"
#include "event_display.h"

/* Christian's heuristics */
#include "heur_greedycolsel.h"
#include "heur_masterdiving.h"
#include "heur_mastercoefdiving.h"
#include "heur_masterfracdiving.h"
#include "heur_masterlinesdiving.h"
#include "heur_mastervecldiving.h"
#include "heur_relaxcolsel.h"
#include "heur_restmaster.h"
#include "heur_setcover.h"

#ifdef WITH_CLIQUER
#include "solver_cliquer.h"
#endif

#ifdef WITH_CPLEXSOLVER
#include "solver_cplex.h"
#endif

#include "scip/table_default.h"

/** includes default GCG master plugins */
SCIP_RETCODE GCGincludeMasterPlugins(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( SCIPincludeDialogMaster(scip) );
   SCIP_CALL( SCIPincludeConshdlrLinear(scip) ); /* linear must be first due to constraint upgrading */
   SCIP_CALL( SCIPincludeConshdlrAnd(scip) );
   SCIP_CALL( SCIPincludeConshdlrBounddisjunction(scip) );
   SCIP_CALL( SCIPincludeConshdlrConjunction(scip) );
   SCIP_CALL( SCIPincludeConshdlrIndicator(scip) );
   SCIP_CALL( SCIPincludeConshdlrIntegral(scip) );
   SCIP_CALL( SCIPincludeConshdlrKnapsack(scip) );
   SCIP_CALL( SCIPincludeConshdlrLogicor(scip) );
   SCIP_CALL( SCIPincludeConshdlrOr(scip) );
   SCIP_CALL( SCIPincludeConshdlrSetppc(scip) );
   SCIP_CALL( SCIPincludeConshdlrVarbound(scip) );
   SCIP_CALL( SCIPincludeConshdlrXor(scip) );

   SCIP_CALL( SCIPincludeReaderCip(scip) );
   SCIP_CALL( SCIPincludeReaderLp(scip) );

   SCIP_CALL( SCIPincludePresolBoundshift(scip) );
   SCIP_CALL( SCIPincludePresolImplics(scip) );
   SCIP_CALL( SCIPincludePresolInttobinary(scip) );
   SCIP_CALL( SCIPincludePresolRoundbound(scip) );

#if USEPROP
   SCIP_CALL( SCIPincludePropDualfix(scip) );
   SCIP_CALL( SCIPincludePropGenvbounds(scip) );
   SCIP_CALL( SCIPincludePropProbing(scip) );
   SCIP_CALL( SCIPincludePropPseudoobj(scip) );
   SCIP_CALL( SCIPincludePropRootredcost(scip) );
   SCIP_CALL( SCIPincludePropRedcost(scip) );
   SCIP_CALL( SCIPincludePropVbounds(scip) );
#endif

   SCIP_CALL( SCIPincludeNodeselMaster(scip) );
   SCIP_CALL( SCIPincludeConshdlrIntegralOrig(scip) );
   SCIP_CALL( SCIPincludeBranchruleRyanfoster(scip) );
   SCIP_CALL( SCIPincludeBranchruleOrig(scip) );
   SCIP_CALL( SCIPincludeBranchruleRelpsprob(scip) );
   SCIP_CALL( SCIPincludeBranchruleGeneric(scip) );
   SCIP_CALL( SCIPincludeBranchruleBPStrong(scip) );

#if USEHEURS
   SCIP_CALL( SCIPincludeHeurActconsdiving(scip) );
   SCIP_CALL( SCIPincludeHeurAdaptivediving(scip) );
   SCIP_CALL( SCIPincludeHeurBound(scip) );
   SCIP_CALL( SCIPincludeHeurClique(scip) );
   SCIP_CALL( SCIPincludeHeurCoefdiving(scip) );
   SCIP_CALL( SCIPincludeHeurCompletesol(scip) );
   SCIP_CALL( SCIPincludeHeurConflictdiving(scip) );
   SCIP_CALL( SCIPincludeHeurCrossover(scip) );
   SCIP_CALL( SCIPincludeHeurDins(scip) );
   SCIP_CALL( SCIPincludeHeurDistributiondiving(scip) );
   SCIP_CALL( SCIPincludeHeurDps(scip) );
   SCIP_CALL( SCIPincludeHeurDualval(scip) );
   SCIP_CALL( SCIPincludeHeurFarkasdiving(scip) );
   SCIP_CALL( SCIPincludeHeurFeaspump(scip) );
   SCIP_CALL( SCIPincludeHeurFixandinfer(scip) );
   SCIP_CALL( SCIPincludeHeurFracdiving(scip) );
   SCIP_CALL( SCIPincludeHeurGins(scip) );
   SCIP_CALL( SCIPincludeHeurGuideddiving(scip) );
   SCIP_CALL( SCIPincludeHeurIndicator(scip) );
   SCIP_CALL( SCIPincludeHeurIntdiving(scip) );
   SCIP_CALL( SCIPincludeHeurIntshifting(scip) );
   SCIP_CALL( SCIPincludeHeurLinesearchdiving(scip) );
   SCIP_CALL( SCIPincludeHeurLocalbranching(scip) );
   SCIP_CALL( SCIPincludeHeurLocks(scip) );
   SCIP_CALL( SCIPincludeHeurLpface(scip) );
   SCIP_CALL( SCIPincludeHeurAlns(scip) );
   SCIP_CALL( SCIPincludeHeurMultistart(scip) );
   SCIP_CALL( SCIPincludeHeurMpec(scip) );
   SCIP_CALL( SCIPincludeHeurMutation(scip) );
   SCIP_CALL( SCIPincludeHeurNlpdiving(scip) );
   SCIP_CALL( SCIPincludeHeurObjpscostdiving(scip) );
   SCIP_CALL( SCIPincludeHeurOctane(scip) );
   SCIP_CALL( SCIPincludeHeurOfins(scip) );
   SCIP_CALL( SCIPincludeHeurOneopt(scip) );
   SCIP_CALL( SCIPincludeHeurPADM(scip) );
   SCIP_CALL( SCIPincludeHeurProximity(scip) );
   SCIP_CALL( SCIPincludeHeurPscostdiving(scip) );
   SCIP_CALL( SCIPincludeHeurRandrounding(scip) );
   SCIP_CALL( SCIPincludeHeurRens(scip) );
   SCIP_CALL( SCIPincludeHeurReoptsols(scip) );
   SCIP_CALL( SCIPincludeHeurRepair(scip) );
   SCIP_CALL( SCIPincludeHeurRins(scip) );
   SCIP_CALL( SCIPincludeHeurRootsoldiving(scip) );
   SCIP_CALL( SCIPincludeHeurRounding(scip) );
   SCIP_CALL( SCIPincludeHeurShiftandpropagate(scip) );
   SCIP_CALL( SCIPincludeHeurShifting(scip) );
   SCIP_CALL( SCIPincludeHeurSubNlp(scip) );
   SCIP_CALL( SCIPincludeHeurTrivial(scip) );
   SCIP_CALL( SCIPincludeHeurTrivialnegation(scip) );
   SCIP_CALL( SCIPincludeHeurTrustregion(scip) );
   SCIP_CALL( SCIPincludeHeurTrySol(scip) );
   SCIP_CALL( SCIPincludeHeurTwoopt(scip) );
   SCIP_CALL( SCIPincludeHeurUndercover(scip) );
   SCIP_CALL( SCIPincludeHeurVbounds(scip) );
   SCIP_CALL( SCIPincludeHeurVeclendiving(scip) );
   SCIP_CALL( SCIPincludeHeurZirounding(scip) );
   SCIP_CALL( SCIPincludeHeurZeroobj(scip) );

   SCIP_CALL( SCIPincludeHeurSimplerounding(scip) );

   /* Christian's heuristics */
   SCIP_CALL( SCIPincludeHeurGreedycolsel(scip) );
   SCIP_CALL( SCIPincludeEventHdlrMasterdiving(scip) );
   SCIP_CALL( GCGincludeHeurMastercoefdiving(scip) );
   SCIP_CALL( GCGincludeHeurMasterfracdiving(scip) );
   SCIP_CALL( GCGincludeHeurMasterlinesdiving(scip) );
   SCIP_CALL( GCGincludeHeurMastervecldiving(scip) );
   SCIP_CALL( SCIPincludeHeurRelaxcolsel(scip) );
   SCIP_CALL( SCIPincludeHeurRestmaster(scip) );
   SCIP_CALL( SCIPincludeHeurSetcover(scip) );
#endif

#if USESEPA
   SCIP_CALL( SCIPincludeSepaClique(scip) );
   SCIP_CALL( SCIPincludeSepaCmir(scip) );
   SCIP_CALL( SCIPincludeSepaFlowcover(scip) );
   SCIP_CALL( SCIPincludeSepaGomory(scip) );
   SCIP_CALL( SCIPincludeSepaImpliedbounds(scip) );
   SCIP_CALL( SCIPincludeSepaIntobj(scip) );
   SCIP_CALL( SCIPincludeSepaMcf(scip) );
   SCIP_CALL( SCIPincludeSepaOddcycle(scip) );
   SCIP_CALL( SCIPincludeSepaRedcost(scip) );
   SCIP_CALL( SCIPincludeSepaZerohalf(scip) );
#endif
   SCIP_CALL( SCIPincludeSepaMaster(scip) );
   SCIP_CALL( SCIPincludeDispMaster(scip) );
   SCIP_CALL( SCIPdebugIncludeProp(scip) ); /*lint !e506 !e774*/
   SCIP_CALL( SCIPincludeTableDefault(scip) );

   /* Jonas' stuff */
   SCIP_CALL( SCIPincludeSepaBasis(scip) );

   SCIP_CALL( GCGincludeSolverKnapsack(scip) );
   SCIP_CALL( GCGincludeSolverMip(scip) );

#ifdef WITH_CLIQUER
   SCIP_CALL( GCGincludeSolverCliquer(scip) );
#endif

#ifdef WITH_CPLEXSOLVER
   SCIP_CALL( GCGincludeSolverCplex(scip) );
#endif

   /* include masterbranch constraint handler */
   SCIP_CALL( SCIPincludeConshdlrMasterbranch(scip) );

   SCIP_CALL( SCIPincludeEventHdlrBestsol(scip) );
   SCIP_CALL( SCIPincludeEventHdlrRelaxsol(scip) );
   SCIP_CALL( SCIPincludeEventHdlrSolvingstats(scip) );
   SCIP_CALL( SCIPincludeEventHdlrDisplay(scip) );

   return SCIP_OKAY;
}
