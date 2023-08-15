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

/**@file   objpricer_gcg.h
 * @brief  GCG variable pricer
 * @author Martin Bergner
 * @ingroup PRICERS
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_OBJPRICER_GCG_H_
#define GCG_OBJPRICER_GCG_H_

#include "objscip/objscip.h"
#include "class_pricingtype.h"
#include "class_pricingcontroller.h"
#include "class_stabilization.h"
#include "pub_gcgcol.h"
#include "pub_colpool.h"
#include "pricestore_gcg.h"

/**@defgroup GCGPRICEROBJ GCG Variable Pricer Object
 * @ingroup PRICING_PUB
 * @{
 */

using gcg::Pricingcontroller;
using gcg::Stabilization;

class ObjPricerGcg : public scip::ObjPricer
{
public:
   /*lint --e{1540}*/

   SCIP*                  origprob;           /**< the original program */
   SCIP_PRICERDATA*       pricerdata;         /**< pricerdata data structure */
   GCG_COLPOOL*           colpool;            /**< column pool */
   GCG_PRICESTORE*        pricestore;         /**< price storage */
   static int             threads;

   /** default constructor */
   ObjPricerGcg(
         SCIP* scip, /**< SCIP data structure */
         SCIP*              origscip,           /**< SCIP data structure of original problem */
         const char* name, /**< name of variable pricer */
         const char* desc, /**< description of variable pricer */
         int priority, /**< priority of the variable pricer */
         unsigned int delay, /**< should the pricer be delayed until no other pricers or already existing*/
         SCIP_PRICERDATA *pricerdata /**< pricerdata data structure */
   );
   /** destructor */
   virtual ~ObjPricerGcg()
   {}

   /** destructor of variable pricer to free user data (called when SCIP is exiting) */
   virtual SCIP_DECL_PRICERFREE(scip_free);
   /** initialization method of variable pricer (called after problem was transformed) */
   virtual SCIP_DECL_PRICERINIT(scip_init);
   /** deinitialization method of variable pricer (called before transformed problem is freed) */
   virtual SCIP_DECL_PRICEREXIT(scip_exit);
   /** solving process initialization method of variable pricer (called when branch and bound process is about to begin) */
   virtual SCIP_DECL_PRICERINITSOL(scip_initsol);
   /** solving process deinitialization method of variable pricer (called before branch and bound process data is freed) */
   virtual SCIP_DECL_PRICEREXITSOL(scip_exitsol);
   /** reduced cost pricing method of variable pricer for feasible LPs */
   virtual SCIP_DECL_PRICERREDCOST(scip_redcost);
   /** farkas pricing method of variable pricer for infeasible LPs */
   virtual SCIP_DECL_PRICERFARKAS(scip_farkas);
   inline SCIP_PRICERDATA *getPricerdata()
   {
      return pricerdata;
   }

   /** for a pricing problem, get the dual solution value or Farkas value of the convexity constraint */
   SCIP_Real getConvconsDualsol(
      PricingType*          pricetype,           /**< Farkas or Reduced cost pricing */
      int                   probnr               /**< index of corresponding pricing problem */
      );

   /** computes the pricing problem objectives */
   SCIP_RETCODE setPricingObjs(
      PricingType*          pricetype,          /**< Farkas or Reduced cost pricing */
      SCIP_Bool             stabilize           /**< do we use stabilization ? */
   );

   /** update reduced cost of columns in column pool */
   void updateRedcostColumnPool(
      PricingType*          pricetype           /**< type of pricing: reduced cost or Farkas */
      );
   /** method to price new columns from Column Pool */
   SCIP_RETCODE priceColumnPoolOld(
      PricingType*          pricetype,          /**< type of pricing: reduced cost or Farkas */
      int*                  pnfoundvars         /**< pointer to store number of priced variables */
      );

   /** performs the pricing routine, gets the type of pricing that should be done: farkas or redcost pricing */
   SCIP_RETCODE priceNewVariables(
      PricingType*       pricetype,          /**< type of the pricing */
      SCIP_RESULT*       result,             /**< result pointer */
      double*            lowerbound          /**< lower bound of pricingproblems */
   );

   /** creates a new master variable corresponding to the given solution and problem */
   SCIP_RETCODE createNewMasterVar(
      SCIP*              scip,               /**< SCIP data structure */
      PricingType*       pricetype,          /**< type of the pricing */
      SCIP_SOL*          sol,                /**< solution to compute reduced cost for */
      SCIP_VAR**         solvars,            /**< array of variables with non-zero value in the solution of the pricing problem */
      double*            solvals,            /**< array of values in the solution of the pricing problem for variables in array solvars*/
      int                nsolvars,           /**< number of variables in array solvars */
      unsigned int       solisray,           /**< is the solution a ray? */
      int                prob,               /**< number of the pricing problem the solution belongs to */
      unsigned int       force,              /**< should the given variable be added also if it has non-negative reduced cost? */
      unsigned int*      added,              /**< pointer to store whether the variable was successfully added */
      SCIP_VAR**         addedvar            /**< pointer to store the created variable */
   );

   /** creates a new master variable corresponding to the given gcg column */
   SCIP_RETCODE createNewMasterVarFromGcgCol(
      SCIP*                 scip,               /**< SCIP data structure */
      PricingType*          pricetype,          /**< type of pricing */
      GCG_COL*              gcgcol,             /**< GCG column data structure */
      SCIP_Bool             force,              /**< should the given variable be added also if it has non-negative reduced cost? */
      SCIP_Bool*            added,              /**< pointer to store whether the variable was successfully added */
      SCIP_VAR**            addedvar,           /**< pointer to store the created variable */
      SCIP_Real             score               /**< score of column (or -1.0 if not specified) */
   );

   /* Compute difference of two dual solutions */
   SCIP_RETCODE computeDualDiff(
      SCIP_Real**          dualvals1,           /**< array of dual values for each pricing problem */
      SCIP_Real*           dualconv1,           /**< array of dual solutions for the convexity constraints  */
      SCIP_Real**          dualvals2,           /**< array of dual values for each pricing problem */
      SCIP_Real*           dualconv2,           /**< array of dual solutions for the convexity constraints  */
      SCIP_Real*           dualdiff             /**< pointer to store difference of duals solutions */
   );

   /** the pricing loop: solve the pricing problems */
   SCIP_RETCODE pricingLoop(
      PricingType*   pricetype,          /**< type of pricing */
      SCIP_RESULT*   result,             /**< result pointer */
      int*           nfoundvars,         /**< pointer to store number of found variables */
      SCIP_Real*     lowerbound,         /**< pointer to store lowerbound obtained due to lagrange bound */
      SCIP_Bool*     bestredcostvalid    /**< pointer to store if bestredcost are valid (pp solvedoptimal) */
   );

   const FarkasPricing* getFarkasPricing() const
   {
      return farkaspricing;
   }

   FarkasPricing* getFarkasPricingNonConst()
   {
      return farkaspricing;
   }

   const ReducedCostPricing* getReducedCostPricing() const
   {
      return reducedcostpricing;
   }

   ReducedCostPricing* getReducedCostPricingNonConst()
   {
      return reducedcostpricing;
   }

   SCIP* getOrigprob()
   {
      return origprob;
   }

   /** get the number of columns to be added to the master LP in the current pricing round */
   int getMaxColsRound() const;

   /** get the number of columns per pricing problem to be added to the master LP in the current pricing round */
   int getMaxColsProb() const;

   /** add artificial vars */
   SCIP_RETCODE addArtificialVars();

   /** add trivial sols */
   SCIP_RETCODE addTrivialsols();

   /** create the pointers for the pricing types */
   SCIP_RETCODE createPricingTypes();

   /** create the pricing controller */
   SCIP_RETCODE createPricingcontroller();

   /** create the pointers for the stabilization */
   void createStabilization();

   /** create the pointers for the colpool */
   SCIP_RETCODE createColpool();

   /** create the pointers for the pricestore */
   SCIP_RETCODE createPricestore();

   /** for given columns, (re-)compute and update their reduced costs */
   void updateRedcosts(
      PricingType*          pricetype,          /**< type of pricing */
      GCG_COL**             cols,               /**< columns to compute reduced costs for */
      int                   ncols,              /**< number of columns */
      int*                  nimpcols            /**< pointer to store number of improving columns */
      );

   /** add a new column to the pricing storage */
   SCIP_RETCODE addColToPricestore(
      GCG_COL*              col                 /**< priced col */
      );

   /** for each pricing problem, get the best found column from the pricing storage */
   void getBestCols(
      GCG_COL**             pricingprobcols     /**< array to be filled with best column per pricing problem */
      );

   /** get the sum over the dual values of convexity constraints */
   SCIP_Real getDualconvsum(
      GCG_COL**             bestcols            /**< best columns found per pricing problem */
      );

   /* computes the objective value of the current (stabilized) dual variables) in the dual program */
   SCIP_RETCODE getStabilizedDualObjectiveValue(
      PricingType*       pricetype,          /**< type of pricing */
      SCIP_Real*         stabdualval,        /**< pointer to store stabilized dual objective value */
      SCIP_Bool          stabilize           /**< stabilize? */
      );

   SCIP_Real computeRedCostGcgCol(
      PricingType*          pricetype,          /**< type of pricing */
      GCG_Col*              gcgcol,             /**< gcg column to compute reduced cost for */
      SCIP_Real*            objvalptr           /**< pointer to store the computed objective value */
      ) const;

   /** compute master coefficients of column */
   SCIP_RETCODE computeColMastercoefs(
      GCG_COL*              gcgcol              /**< GCG column data structure */
      );

   /** compute master cut coefficients of column */
   SCIP_RETCODE computeColMastercuts(
      GCG_COL*              gcgcol              /**< GCG column data structure */
      );

private:
   ReducedCostPricing*    reducedcostpricing;
   FarkasPricing*         farkaspricing;
   PricingType*           pricingtype;          /**< current pricing type, or NULL if we are not in pricing */
   Pricingcontroller*     pricingcontroller;
   Stabilization*         stabilization;

   /** free pricing problems */
   SCIP_RETCODE freePricingProblems();

   SCIP_Real computeRedCost(
      PricingType*          pricetype,          /**< type of pricing */
      SCIP_SOL*             sol,                /**< solution to compute reduced cost for */
      SCIP_Bool             solisray,           /**< is the solution a ray? */
      int                   prob,               /**< number of the pricing problem the solution belongs to */
      SCIP_Real*            objvalptr           /**< pointer to store the computed objective value */
   ) const;

   /** for given columns, (re-)compute and update their reduced costs */
   void updateRedcosts(
      PricingType*          pricetype,          /**< type of pricing */
      GCG_COL**             cols,               /**< columns to compute reduced costs for */
      int                   ncols               /**< number of columns */
      ) const;

   /** return TRUE or FALSE whether the master LP is solved to optimality */
   SCIP_Bool isMasterLPOptimal() const;

   /** ensures size of pricedvars array */
   SCIP_RETCODE ensureSizePricedvars(
      int                   size                /**< needed size */
   );

   /** adds new variable to the end of the priced variables array */
   SCIP_RETCODE addVariableToPricedvars(
      SCIP_VAR*             newvar              /**< variable to add */
   );

   /** ensures size of root bounds arrays */
   SCIP_RETCODE ensureSizeRootBounds(
      int                   size                /**< needed size */
   );

   /** adds new bounds to the bound arrays as well as some additional information on dual variables and root lp solution */
   SCIP_RETCODE addRootBounds(
      SCIP_Real             primalbound,        /**< new primal bound for the root master LP */
      SCIP_Real             dualbound           /**< new dual bound for the root master LP */
   );

   /** add master variable to all constraints */
   SCIP_RETCODE addVariableToMasterconstraints(
      SCIP_VAR*             newvar,             /**< The new variable to add */
      int                   prob,               /**< number of the pricing problem the solution belongs to */
      SCIP_VAR**            solvars,            /**< array of variables with non-zero value in the solution of the pricing problem */
      SCIP_Real*            solvals,            /**< array of values in the solution of the pricing problem for variables in array solvars*/
      int                   nsolvars            /**< number of variables in array solvars */
   );

   /** add master variable to all constraints */
   SCIP_RETCODE addVariableToMasterconstraintsFromGCGCol(
      SCIP_VAR*             newvar,             /**< The new variable to add */
      GCG_COL*              gcgcol              /**< GCG column data structure */
      );

   /** add variable with computed coefficients to the master cuts */
   SCIP_RETCODE addVariableToMastercuts(
      SCIP_VAR*             newvar,             /**< The new variable to add */
      int                   prob,               /**< number of the pricing problem the solution belongs to */
      SCIP_VAR**            solvars,            /**< array of variables with non-zero value in the solution of the pricing problem */
      SCIP_Real*            solvals,            /**< array of values in the solution of the pricing problem for variables in array solvars*/
      int                   nsolvars            /**< number of variables in array solvars */
      );

   /** add variable with computed coefficients to the master cuts */
   SCIP_RETCODE addVariableToMastercutsFromGCGCol(
      SCIP_VAR*             newvar,             /**< The new variable to add */
      GCG_COL*              gcgcol              /**< GCG column data structure */
      );

   /**
    * check whether pricing can be aborted:
    * if objective value is always integral and the current node's current
    * lowerbound rounded up equals the current lp objective value rounded
    * up we don't need to continue pricing since the best possible feasible
    * solution must have at least this value
    */
   SCIP_Bool canPricingBeAborted() const;

   /** sorts pricing problems according to their score */
   void sortPricingProblemsByScore() const;

   /** returns the gegeneracy of the masterproblem */
   SCIP_RETCODE computeCurrentDegeneracy(
      double*               degeneracy          /**< pointer to store degeneracy */
   );

   /** set subproblem memory limit */
   SCIP_RETCODE setPricingProblemMemorylimit(
      SCIP*                 pricingscip         /**< SCIP of the pricingproblem */
   );

#ifdef SCIP_DISABLED_CODE
   /** generic method to generate feasible columns from the pricing problem
    * @note This method has to be threadsafe!
    */
   SCIP_RETCODE generateColumnsFromPricingProblem(
      GCG_PRICINGJOB*       pricingjob,         /**< pricing job to be performed */
      PricingType*          pricetype,          /**< type of pricing: reduced cost or Farkas */
      int                   maxcols             /**< size of the cols array to indicate maximum columns */
      );

   /** solves a specific pricing problem
    * @todo simplify
    * @note This method has to be threadsafe!
    */
   SCIP_RETCODE solvePricingProblem(
      GCG_PRICINGJOB*       pricingjob,         /**< pricing job to be performed */
      PricingType*          pricetype,          /**< type of pricing: reduced cost or Farkas */
      int                   maxcols             /**< size of the cols array to indicate maximum columns */
      );
#endif

   /** perform a pricing job, i.e. apply the corresponding solver to the pricing problem
    * @note This method has to be threadsafe!
    */
   SCIP_RETCODE performPricingjob(
      GCG_PRICINGJOB*       pricingjob,         /**< pricing job */
      PricingType*          pricetype,          /**< type of pricing: reduced cost or Farkas */
      GCG_PRICINGSTATUS*    status,             /**< pointer to store pricing status */
      SCIP_Real*            lowerbound          /**< pointer to store the obtained lower bound */
      );

   /** frees all solvers */
   SCIP_RETCODE solversFree();

   /** calls the init method on all solvers */
   SCIP_RETCODE solversInit();

   /** calls the exit method on all solvers */
   SCIP_RETCODE solversExit();

   /** calls the initsol method on all solvers */
   SCIP_RETCODE solversInitsol();

   /** calls the exitsol method of all solvers */
   SCIP_RETCODE solversExitsol();

   /** computes the stack of masterbranch constraints up to the last generic branching node
    * @note This method has to be threadsafe!
    */
   SCIP_RETCODE computeGenericBranchingconssStack(
      PricingType*          pricetype,          /**< type of pricing: reduced cost or Farkas */
      int                   prob,               /**< index of pricing problem */
      SCIP_CONS***          consstack,          /**< stack of branching constraints */
      int*                  nconsstack,         /**< size of the stack */
      SCIP_Real**           consduals           /**< dual values of the masterbranch solutions, or NULL */
   ) const;

   /** add bounds change from constraint from the pricing problem at this node
    * @note This method has to be threadsafe!
    */
   SCIP_RETCODE addBranchingBoundChangesToPricing(
      int                   prob,               /**< index of pricing problem */
      SCIP_CONS*            branchcons          /**< branching constraints from which bound should applied */
   ) const;

   SCIP_RETCODE checkBranchingBoundChanges(
      int                   prob,               /**< index of pricing problem */
      SCIP_SOL*             sol,                /**< solution to check */
      SCIP_CONS*            branchcons,         /**< branching constraints from which bound should applied */
      SCIP_Bool*            feasible            /**< check whether the solution is feasible */
   ) const;

   /** check bounds change from constraint from the pricing problem at this node
    * @note This method has to be threadsafe!
    */
   SCIP_RETCODE checkBranchingBoundChangesGcgCol(
      GCG_COL*              gcgcol,             /**< gcg column to check */
      SCIP_CONS*            branchcons,         /**< branching constraints from which bound should applied */
      SCIP_Bool*            feasible            /**< check whether the solution is feasible */
   ) const;

   SCIP_RETCODE ensureSizeArtificialvars(
      int                   size                /**< needed size */
   );

};
/** @} */
#endif
