/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2022 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   benders_gcg.c
 * @brief  GCG Benders' decomposition algorithm
 * @author Stephen J. Maher
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
/* #define SCIP_DEBUG */
#include <assert.h>
#include <string.h>

#include "benders_gcg.h"

#include "gcg.h"

#include "relax_gcg.h"
#include "scip_misc.h"
#include "pub_gcgvar.h"

#include "scip/cons_linear.h"
#include "scip/pub_var.h"
#include "scip/pub_benders.h"
#include "scip/bendersdefcuts.h"

#define BENDERS_NAME                "gcg"
#define BENDERS_DESC                "Benders' decomposition for the Generic Column Generation package"
#define BENDERS_PRIORITY         1000
#define BENDERS_CUTLP            TRUE   /**< should Benders' cut be generated for LP solutions */
#define BENDERS_CUTPSEUDO        TRUE   /**< should Benders' cut be generated for pseudo solutions */
#define BENDERS_CUTRELAX         TRUE   /**< should Benders' cut be generated for relaxation solutions */
#define BENDERS_SHAREAUXVARS    FALSE   /**< should this Benders' share the highest priority Benders' aux vars */

#define LARGE_VALUE  10000    /**< a large value that is used to create an artificial solution */

/*
 * Data structures
 */

/** Benders' decomposition data */
struct SCIP_BendersData
{
   SCIP*                 origprob;           /**< the SCIP instance of the original problem */
   SCIP_SOL*             relaxsol;           /**< the solution to the original problem related to the relaxation */
};

/*
 * Local methods
 */

/* returns the objective coefficient for the given variable */
static
SCIP_Real varGetObj(
   SCIP_VAR*             var
   )
{
   SCIP_VAR* origvar;
   assert(var != NULL);

   origvar = GCGpricingVarGetOrigvars(var)[0];

   if( GCGoriginalVarIsLinking(origvar) )
      return 0.0;
   else
      return SCIPvarGetObj(origvar);
}

/* Initializes the objective function for all subproblems. */
static
SCIP_RETCODE setSubproblemObjs(
   SCIP_BENDERS*         benders,            /**< the benders' decomposition constraint handler */
   int                   probnumber          /**< the subproblem number */
   )
{
   SCIP* subproblem;
   SCIP_VAR** probvars;
   int nprobvars;
   int i;

   assert(benders != NULL);

   /* changing the variable */
   subproblem = SCIPbendersSubproblem(benders, probnumber);
   assert(subproblem != NULL);

   probvars = SCIPgetVars(subproblem);
   nprobvars = SCIPgetNVars(subproblem);

   for( i = 0; i < nprobvars; i++ )
   {
      assert(GCGvarGetBlock(probvars[i]) == probnumber);
      assert(GCGoriginalVarIsLinking(GCGpricingVarGetOrigvars(probvars[i])[0]) || (GCGvarGetBlock(GCGpricingVarGetOrigvars(probvars[i])[0]) == probnumber));

      SCIP_CALL( SCIPchgVarObj(subproblem, probvars[i], varGetObj(probvars[i])));

      SCIPdebugMessage("pricingobj var <%s> %f\n", SCIPvarGetName(probvars[i]), varGetObj(probvars[i]));
   }

   return SCIP_OKAY;
}

/** sets the pricing problem variable values for the original problem using the decomposed problem solution
 *  There is a mapping between the original problem and the variables from the pricing problems. This mapping is used to
 *  identify the variables of the original problem corresponding to the pricing problem variables.
 *
 *  An artificial solution can be constructed, which is indicated by the vals array provided as NULL. An artifical
 *  solution sets the original problem variables corresponding to pricing problems variables to their bounds. An
 *  artificial solution is created if branching candidates need to be found. Branching candidates only come from the
 *  master problem, so the variable values of the pricing problem variables does not affect the branching variable
 *  selection
 */
static
SCIP_RETCODE setOriginalProblemPricingValues(
   SCIP*                 origprob,           /**< the SCIP instance of the original problem */
   SCIP*                 masterprob,         /**< the Benders' master problem */
   SCIP_BENDERS*         benders,            /**< the Benders' decomposition structure */
   SCIP_SOL*             origsol,            /**< the solution for the original problem */
   SCIP_VAR**            vars,               /**< the variables from the decomposed problem */
   SCIP_Real*            vals,               /**< the solution values of the given problem, can be NULL for an
                                                  artificial solution */
   int                   nvars,              /**< the number of variables */
   SCIP_Bool*            success             /**< were all values set successfully? */
   )
{
   SCIP_VAR** origvars;
   SCIP_Real val;
   int norigvars;
   int i;

   assert(origprob != NULL);
   assert(masterprob != NULL);
   assert(benders != NULL);
   assert(vars != NULL);

   (*success) = TRUE;

   /* looping through all variables to update the values in the original solution */
   for( i = 0; i < nvars; i++ )
   {
      norigvars = GCGpricingVarGetNOrigvars(vars[i]);
      if( norigvars > 0 )
      {
         SCIP_VAR* mastervar;

         origvars = GCGpricingVarGetOrigvars(vars[i]);

         /* all variables should be associated with a single original variable. This is because no reformulation has
          * been performed.
          * TODO: This appears not to be true. Need to find out why multiple original problem variables are associated
          * with a pricing variable. Currently the first original variable is used.
          */
         /* assert(norigvars == 1); */
         assert(GCGvarIsPricing(vars[i]));

         /* for all variables that are from the subproblems, they are set to their bounds if the solution is being
          * created to identify branching candidates. */
         if( vals == NULL )
         {
            if( SCIPisNegative(origprob, SCIPvarGetObj(origvars[0])) )
            {
               val = SCIPvarGetLbGlobal(origvars[0]);
               if( SCIPisInfinity(origprob, -val) )
                  val = -LARGE_VALUE;
            }
            else
            {
               val = SCIPvarGetUbGlobal(origvars[0]);
               if( SCIPisInfinity(origprob, val) )
                  val = LARGE_VALUE;
            }
         }
         else
            val = vals[i];

         /* identifying whether the variable is a master problem variable. The variable is a master problem variable if
          * there is a mapping from the subproblem to the master problem. If a mapping exists, then the variable value
          * is not updated in the master problem
          */
         mastervar = NULL;
         SCIP_CALL( SCIPgetBendersMasterVar(masterprob, benders, vars[i], &mastervar) );
         if( SCIPisInfinity(origprob, val) || SCIPisInfinity(origprob, -val) )
         {
            (*success) = FALSE;
            return SCIP_OKAY;
         }

         SCIPdebugMsg(masterprob, "setting the value of <%s> (dw variable <%s>) to %g in the original solution.\n",
            SCIPvarGetName(origvars[0]), SCIPvarGetName(vars[i]), val);

         /* only update the solution value if the master problem variable does not exist. */
         if( mastervar == NULL )
         {
            SCIP_CALL( SCIPsetSolVal(origprob, origsol, origvars[0], val) );
         }
      }
   }

   return SCIP_OKAY;
}

/** sets the master problem values for the original problem using the decomposed problem solution */
static
SCIP_RETCODE setOriginalProblemMasterValues(
   SCIP*                 origprob,           /**< the SCIP instance of the original problem */
   SCIP*                 masterprob,         /**< the Benders' master problem */
   SCIP_BENDERS*         benders,            /**< the Benders' decomposition structure */
   SCIP_SOL*             origsol,            /**< the solution for the original problem */
   SCIP_VAR**            vars,               /**< the variables from the decomposed problem */
   SCIP_Real*            vals,               /**< the solution values of the given problem, can be NULL */
   int                   nvars               /**< the number of variables */
   )
{
   SCIP_VAR** origvars;
   int norigvars;
   int i;

#ifndef NDEBUG
   SCIP_Real* origvals;
#endif

   assert(origprob != NULL);
   assert(masterprob != NULL);
   assert(benders != NULL);
   assert(vars != NULL);
   assert(vals != NULL);

   /* looping through all variables to update the values in the original solution */
   for( i = 0; i < nvars; i++ )
   {
      norigvars = GCGmasterVarGetNOrigvars(vars[i]);
      if( norigvars > 0 )
      {
         origvars = GCGmasterVarGetOrigvars(vars[i]);

#ifndef NDEBUG
         origvals = GCGmasterVarGetOrigvals(vars[i]);
#endif

         /* all master variables should be associated with a single original variable. This is because no reformulation has
          * been performed. */
         assert(norigvars == 1);
         assert(origvals[0] == 1.0);
         assert(GCGvarIsMaster(vars[i]));
         assert(!SCIPisInfinity(origprob, vals[i]));

         SCIPdebugMsg(masterprob, "setting the value of <%s> (master variable <%s>) to %g in the original solution.\n",
            SCIPvarGetName(origvars[0]), SCIPvarGetName(vars[i]), vals[i]);

         /* only update the solution value of master variables. */
         SCIP_CALL( SCIPsetSolVal(origprob, origsol, origvars[0], vals[i]) );
      }
   }
   return SCIP_OKAY;
}

/** creates an original problem solution from the master and subproblem solutions */
static
SCIP_RETCODE createOriginalProblemSolution(
   SCIP*                 masterprob,         /**< the Benders' master problem */
   SCIP_BENDERS*         benders,            /**< the Benders' decomposition structure */
   SCIP_SOL*             sol,                /**< the solution passed to the Benders' decomposition subproblems. */
   SCIP_Bool             artificial          /**< should an artifical (possibly infeasible) solution be created to
                                                  generate branching candidates */
   )
{
   SCIP* origprob;
   SCIP* subproblem;
   SCIP_BENDERSDATA* bendersdata;
   SCIP_SOL* origsol;
   SCIP_SOL* bestsol;
   SCIP_VAR** vars;
   SCIP_Real* vals;
   int nsubproblems;
   int nvars;
   int i;
   SCIP_Bool stored;
   SCIP_Bool success;

   assert(masterprob != NULL);
   assert(benders != NULL);

   bendersdata = SCIPbendersGetData(benders);

   assert(bendersdata != NULL);

   origprob = bendersdata->origprob;

   success = TRUE;

   /* creating the original problem */
   SCIP_CALL( SCIPcreateSol(origprob, &origsol, GCGrelaxGetProbingheur(origprob)) );

   /* setting the values of the master variables in the original solution */

   /* getting the variable data for the master variables */
   SCIP_CALL( SCIPgetVarsData(masterprob, &vars, &nvars, NULL, NULL, NULL, NULL) );
   assert(vars != NULL);

   /* getting the best solution from the master problem */
   SCIP_CALL( SCIPallocBufferArray(masterprob, &vals, nvars) );
   SCIP_CALL( SCIPgetSolVals(masterprob, sol, nvars, vars, vals) );

   /* setting the values using the master problem solution */
   SCIP_CALL( setOriginalProblemMasterValues(origprob, masterprob, benders, origsol, vars, vals, nvars) );

   /* freeing the values buffer array for use for the pricing problems */
   SCIPfreeBufferArray(masterprob, &vals);

   /* setting the values of the subproblem variables in the original solution */
   nsubproblems = SCIPbendersGetNSubproblems(benders);

   /* looping through all subproblems */
   for( i = 0; i < nsubproblems; i++ )
   {
      /* it is only possible to use the subproblem solutions if the subproblems are enabled. The subproblems are
       * disabled if they have been merged into the master problem.
       */
      if( SCIPbendersSubproblemIsEnabled(benders, i) )
      {
         subproblem = SCIPbendersSubproblem(benders, i);

         /* getting the variable data for the master variables */
         SCIP_CALL( SCIPgetVarsData(subproblem, &vars, &nvars, NULL, NULL, NULL, NULL) );
         assert(vars != NULL);

         /* getting the best solution from the master problem */
         bestsol = SCIPgetBestSol(subproblem);
#ifdef SCIP_DEBUG
         SCIP_CALL( SCIPprintSol(subproblem, bestsol, NULL, FALSE) );
#endif

         /* the branching candidates come from the master problem solution. However, we need a full solution to pass to the
          * original problem to find the branching candidate. So the subproblem variables are set to their bounds, creating
          * a possibly infeasible solution, but with the fractional master problem variables.
          *
          * It may occur that the subproblem has not been solved yet, this can happen if the subproblem is independent.
          * In this case, an artificial solution is created.
          */
         if( artificial || SCIPgetStage(subproblem) == SCIP_STAGE_PROBLEM )
         {
            /* setting the values of the subproblem variables to their bounds. */
            SCIP_CALL( setOriginalProblemPricingValues(origprob, masterprob, benders, origsol, vars, NULL, nvars, &success) );
         }
         else
         {
            SCIP_CALL( SCIPallocBufferArray(subproblem, &vals, nvars) );
            SCIP_CALL( SCIPgetSolVals(subproblem, bestsol, nvars, vars, vals) );

            /* setting the values using the master problem solution */
            SCIP_CALL( setOriginalProblemPricingValues(origprob, masterprob, benders, origsol, vars, vals, nvars, &success) );

            /* freeing the values buffer array for use for the pricing problems */
            SCIPfreeBufferArray(subproblem, &vals);

            if( !success )
               break;
         }
      }
   }

   /* if the values were not set correctly, then the solution is not updated. This should only happen when the timelimit
    * has been exceeded.
    */
   if( !success )
   {
      SCIP_CALL( SCIPfreeSol(origprob, &origsol) );
      return SCIP_OKAY;
   }

#ifdef SCIP_DEBUG
   SCIPdebugMsg(masterprob, "Original Solution\n");
   SCIP_CALL( SCIPprintSol(origprob, origsol, NULL, FALSE) );
#endif

   /* if the solution is NULL, then the solution comes from the relaxation. Thus, it should be stored in the
    * bendersdata. When it is not NULL, then solution comes from a heuristic. So this solution should be passed to the
    * solution storage. */
   if( sol != NULL )
   {
#ifdef SCIP_DEBUG
      SCIP_CALL( SCIPtrySol(origprob, origsol, TRUE, TRUE, TRUE, TRUE, TRUE, &stored) );
#else
      SCIP_CALL( SCIPtrySol(origprob, origsol, FALSE, FALSE, TRUE, TRUE, TRUE, &stored) );
#endif
      if( !stored )
      {
         SCIP_CALL( SCIPcheckSolOrig(origprob, origsol, &stored, TRUE, TRUE) );
      }

      /** @bug The solution doesn't have to be accepted, numerics might bite us, so the transformation might fail.
       *  A remedy could be: Round the values or propagate changes or call a heuristic to fix it.
       */
      SCIP_CALL( SCIPfreeSol(origprob, &origsol) );

      if( stored )
         SCIPdebugMessage("  updated current best primal feasible solution.\n");
   }
   else
   {
      if( bendersdata->relaxsol != NULL )
      {
         SCIP_CALL( SCIPfreeSol(origprob, &bendersdata->relaxsol) );
      }

      bendersdata->relaxsol = origsol;
   }

   return SCIP_OKAY;
}

/** merge a single subproblem into the master problem */
static
SCIP_RETCODE mergeSubproblemIntoMaster(
   SCIP*                 masterprob,         /**< the Benders' master problem */
   SCIP_BENDERS*         benders,            /**< the Benders' decomposition structure */
   int                   probnumber          /**< the index of the subproblem that will be merged */
   )
{
   SCIP* subproblem;
   SCIP_HASHMAP* varmap;
   SCIP_HASHMAP* consmap;
   SCIP_VAR** vars;
   int nvars;
   int i;

   assert(masterprob != NULL);
   assert(benders != NULL);

   subproblem = SCIPbendersSubproblem(benders, probnumber);

   /* allocating the memory for the variable and constraint hashmaps */
   SCIP_CALL( SCIPhashmapCreate(&varmap, SCIPblkmem(masterprob), SCIPgetNVars(subproblem)) );
   SCIP_CALL( SCIPhashmapCreate(&consmap, SCIPblkmem(masterprob), SCIPgetNConss(subproblem)) );

   SCIP_CALL( SCIPmergeBendersSubproblemIntoMaster(masterprob, benders, varmap, consmap, probnumber) );

   SCIP_CALL( SCIPgetVarsData(subproblem, &vars, &nvars, NULL, NULL, NULL, NULL) );
   /* copying the variable data from the pricing variables to the newly created master variables */
   for( i = 0; i < nvars; i++ )
   {
      SCIP_VAR* mastervar;

      mastervar = (SCIP_VAR*) SCIPhashmapGetImage(varmap, vars[i]);
      SCIP_CALL( GCGcopyPricingvarDataToMastervar(masterprob, vars[i], mastervar) );
   }

   /* freeing the variable and constraint hashmaps */
   SCIPhashmapFree(&varmap);
   SCIPhashmapFree(&consmap);

   return SCIP_OKAY;
}

/** performs a merge of subproblems into the master problem. The subproblems are merged into the master problem if the
 * infeasibility can not be resolved through the addition of cuts. This could be because the appropriate cuts are not
 * available in the Benders' decomposition framework, or that the subproblem has been infeasible for a set number of
 * iterations.
 */
static
SCIP_RETCODE mergeSubproblemsIntoMaster(
   SCIP*                 masterprob,         /**< the Benders' master problem */
   SCIP_BENDERS*         benders,            /**< the Benders' decomposition structure */
   int*                  mergecands,         /**< the subproblems that are merge candidates */
   int                   npriomergecands,    /**< the priority merge candidates, these should be merged */
   int                   nmergecands,        /**< the total number of merge candidates */
   SCIP_Bool*            merged              /**< flag to indicate whether a subproblem has been merged into the master */
   )
{
   int i;

   assert(masterprob != NULL);
   assert(benders != NULL);

   (*merged) = FALSE;

   /* adding the priority merge candidates. If there are no priority candidates, then the first merge candidate is
    * added.
    */
   for( i = 0; i < npriomergecands; i++ )
   {
      SCIP_CALL( mergeSubproblemIntoMaster(masterprob, benders, mergecands[i]) );
      (*merged) = TRUE;
   }

   /* if there were no priority candidates and there is a merge candidate, then only a single merge candidate is
    * merged.
    */
   if( !(*merged) && nmergecands > 0 )
   {
      assert(npriomergecands == 0);

      SCIP_CALL( mergeSubproblemIntoMaster(masterprob, benders, mergecands[0]) );
      (*merged) = TRUE;
   }

   return SCIP_OKAY;
}

/*
 * Callback methods for Benders' decomposition
 */

/* TODO: Implement all necessary Benders' decomposition methods. The methods with an #ifdef SCIP_DISABLED_CODE ... #else #define ... are optional */

/** destructor of Benders' decomposition to free user data (called when SCIP is exiting) */
static
SCIP_DECL_BENDERSFREE(bendersFreeGcg)
{  /*lint --e{715}*/
   SCIP_BENDERSDATA* bendersdata;

   assert(scip != NULL);
   assert(benders != NULL);

   bendersdata = SCIPbendersGetData(benders);

   if( bendersdata != NULL )
   {
      SCIPfreeMemory(scip, &bendersdata);
   }

   return SCIP_OKAY;
}

/** presolving initialization method of constraint handler (called when presolving is about to begin) */
static
SCIP_DECL_BENDERSINITPRE(bendersInitpreGcg)
{  /*lint --e{715}*/
   int nsubproblems;
   int i;

   assert(scip != NULL);
   assert(benders != NULL);

   nsubproblems = SCIPbendersGetNSubproblems(benders);

   for( i = 0; i < nsubproblems; i++ )
   {
      SCIP_CALL( GCGaddDataAuxiliaryVar(scip, SCIPbendersGetAuxiliaryVar(benders, i), i) );
   }

   return SCIP_OKAY;
}

/** solving process deinitialization method of Benders' decomposition (called before branch and bound process data is freed) */
static
SCIP_DECL_BENDERSEXITSOL(bendersExitsolGcg)
{  /*lint --e{715}*/
   SCIP_BENDERSDATA* bendersdata;

   assert(scip != NULL);
   assert(benders != NULL);

   bendersdata = SCIPbendersGetData(benders);

   /* freeing the relaxation solution */
   if( bendersdata->relaxsol != NULL )
   {
      SCIP_CALL( SCIPfreeSol(bendersdata->origprob, &bendersdata->relaxsol) );
   }

   return SCIP_OKAY;
}

/** mapping method between the master problem variables and the subproblem variables of Benders' decomposition */
static
SCIP_DECL_BENDERSGETVAR(bendersGetvarGcg)
{  /*lint --e{715}*/
   SCIP_VAR* origvar;

   assert(scip != NULL);
   assert(benders != NULL);
   assert(var != NULL);
   assert(mappedvar != NULL);

   /* if there is no corresponding master variable for the input variable, then NULL is returned */
   (*mappedvar) = NULL;

   /* if the probnumber is -1, then the master variable is requested.
    * if the probnumber >= 0, then the subproblem variable is requested. */
   if( probnumber == -1 )
   {
      /* getting the original variable for the given pricing variable */
      origvar = GCGpricingVarGetOrigvars(var)[0];

      /* checking whether the original variable is associated with any master problem variables. This is identified by
       * retrieving the number of master variables for the given original variable
       * NOTE: previously, only the linking variables were master variables. As such, the following check was to find
       * only the original variables corresponding to linking variables. Since linking constraints, and their associated
       * variables, are also added to the master problem, then the previous check is not valid.
       */
      if( GCGoriginalVarGetNMastervars(origvar) > 0 )
      {
         (*mappedvar) = GCGoriginalVarGetMastervars(origvar)[0];
      }
   }
   else
   {
      assert(probnumber >= 0 && probnumber < SCIPbendersGetNSubproblems(benders));

      /* getting the original variable for the given pricing variable */
      origvar = GCGmasterVarGetOrigvars(var)[0];

      /* checking whether the original variable is associated with any master problem variables. This is identified by
       * retrieving the number of master variables for the given original variable
       * NOTE: previously, only the linking variables were master variables. As such, the following check was to find
       * only the original variables corresponding to linking variables. Since linking constraints, and their associated
       * variables, are also added to the master problem, then the previous check is not valid.
       */
      /* checking whether the original variable is associated with any master problem variables. This is identified by
       * retrieving the number of master variables for the given original variable.
       *
       * If the pricing variable is requested, then there are two options. The first is that the pricing variable is a
       * linking variable. The second is that the pricing variable has been directly copied to the master problem since
       * it was part of the linking constraints.
       */
      if( GCGoriginalVarGetNMastervars(origvar) > 0 )
      {
         /* if the original variable is a linking variable, then the linking pricing variable is retrieved */
         if( GCGoriginalVarIsLinking(origvar) )
            (*mappedvar) = GCGlinkingVarGetPricingVars(origvar)[probnumber];
         else
            (*mappedvar) = GCGoriginalVarGetPricingVar(origvar);
      }
   }

   return SCIP_OKAY;
}

/** the post-execution method for Benders' decomposition */
static
SCIP_DECL_BENDERSPOSTSOLVE(bendersPostsolveGcg)
{  /*lint --e{715}*/
   SCIP_BENDERSDATA* bendersdata;

   assert(benders != NULL);

   bendersdata = SCIPbendersGetData(benders);
   assert(bendersdata != NULL);

   (*merged) = FALSE;

   /* creates a solution to the original problem */
#ifdef SCIP_DEBUG
   SCIPdebugMessage("The master problem solution.\n");
   SCIP_CALL( SCIPprintSol(scip, sol, NULL, FALSE) );
#endif

   /* if there are merge candidates, then they will be merged into the master problem.
    * TODO: maybe check to see whether the merge could be avoided
    */
   if( nmergecands > 0 )
   {
      SCIP_CALL( mergeSubproblemsIntoMaster(scip, benders, mergecands, npriomergecands, nmergecands, merged) );
   }

   if( !infeasible && !(*merged) )
   {
      /* if the problem was found to be infeasible, then an artificial solution is created. */
      SCIP_CALL( createOriginalProblemSolution(scip, benders, sol, infeasible) );
      if( type == SCIP_BENDERSENFOTYPE_LP )
         SCIP_CALL( GCGrelaxUpdateCurrentSol(bendersdata->origprob) );
   }

   return SCIP_OKAY;
}

/** the method for creating the Benders' decomposition subproblem. This method is called during the initialization stage
 *  (after the master problem was transformed)
 *
 *  This method must create the SCIP instance for the subproblem and add the required variables and constraints. In
 *  addition, the settings required for the solving the problem must be set here. However, some settings will be
 *  overridden by the standard solving method included in the Benders' decomposition framework. If a special solving
 *  method is desired, the user can implement the bendersSolvesubDefault callback.
 */
static
SCIP_DECL_BENDERSCREATESUB(bendersCreatesubGcg)
{  /*lint --e{715}*/
   SCIP_BENDERSDATA* bendersdata;
   SCIP* origprob;

   assert(scip != NULL);
   assert(benders != NULL);

   bendersdata = SCIPbendersGetData(benders);
   assert(bendersdata != NULL);

   origprob = bendersdata->origprob;

   SCIP_CALL( SCIPaddBendersSubproblem(scip, benders, GCGgetPricingprob(origprob, probnumber)) );

   /* setting the objective coefficients for the subproblems.
    * This is required because the variables are added to the pricing problems with a zero coefficient. In the DW
    * context, this is appropriate because the objective coefficients are constantly changing. In the BD context, the
    * objective coefficients are static, so they only need to be updated once. */
   SCIP_CALL( setSubproblemObjs(benders, probnumber) );


   return SCIP_OKAY;
}

/*
 * Benders' decomposition specific interface methods
 */

/** creates the gcg Benders' decomposition and includes it in SCIP */
SCIP_RETCODE SCIPincludeBendersGcg(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP*                 origprob            /**< the SCIP instance of the original problem */
   )
{
   SCIP_BENDERSDATA* bendersdata;
   SCIP_BENDERS* benders;

   /* create gcg Benders' decomposition data */
   SCIP_CALL( SCIPallocMemory(scip, &bendersdata) );
   bendersdata->origprob = origprob;
   bendersdata->relaxsol = NULL;

   benders = NULL;

   /* include Benders' decomposition */
   SCIP_CALL( SCIPincludeBendersBasic(scip, &benders, BENDERS_NAME, BENDERS_DESC, BENDERS_PRIORITY,
         BENDERS_CUTLP, BENDERS_CUTPSEUDO, BENDERS_CUTRELAX, BENDERS_SHAREAUXVARS, bendersGetvarGcg,
         bendersCreatesubGcg, bendersdata) );
   assert(benders != NULL);

   /* set non fundamental callbacks via setter functions */
   SCIP_CALL( SCIPsetBendersFree(scip, benders, bendersFreeGcg) );
   SCIP_CALL( SCIPsetBendersInitpre(scip, benders, bendersInitpreGcg) );
   SCIP_CALL( SCIPsetBendersExitsol(scip, benders, bendersExitsolGcg) );
   SCIP_CALL( SCIPsetBendersPostsolve(scip, benders, bendersPostsolveGcg) );

   /* including the default cuts for Benders' decomposition */
   SCIP_CALL( SCIPincludeBendersDefaultCuts(scip, benders) );

   return SCIP_OKAY;
}

/** returns the last relaxation solution */
SCIP_SOL* SCIPbendersGetRelaxSol(
   SCIP_BENDERS*         benders             /**< the Benders' decomposition structure */
   )
{
   SCIP_BENDERSDATA* bendersdata;

   assert(benders != NULL);
   assert(strcmp(SCIPbendersGetName(benders), BENDERS_NAME) == 0);

   bendersdata = SCIPbendersGetData(benders);

   return bendersdata->relaxsol;
}

/** returns the original problem for the given master problem */
SCIP* GCGbendersGetOrigprob(
   SCIP*                 masterprob          /**< the master problem SCIP instance */
   )
{
   SCIP_BENDERS* benders;
   SCIP_BENDERSDATA* bendersdata;

   assert(masterprob != NULL);

   benders = SCIPfindBenders(masterprob, BENDERS_NAME);
   assert(benders != NULL);

   bendersdata = SCIPbendersGetData(benders);
   assert(bendersdata != NULL);

   return bendersdata->origprob;
}
