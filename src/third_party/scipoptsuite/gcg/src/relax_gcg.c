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

/**@file    relax_gcg.c
 * @ingroup RELAXATORS
 * @brief   GCG relaxator
 * @author  Gerald Gamrath
 * @author  Martin Bergner
 * @author  Alexander Gross
 * @author  Michael Bastubbe
 *
 * \bug
 * - The memory limit is not strictly enforced
 * - Dealing with timelimits is a working hack only
 * - CTRL-C handling is very flaky
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

//#define SCIP_DEBUG

#include <string.h>

#include "scip/scipdefplugins.h"
#include "scip/cons_linear.h"
#include "scip/cons_setppc.h"
#include "scip/scip.h"
#include "scip/misc.h"
#include "scip/clock.h"

#include "relax_gcg.h"

#include "struct_branchgcg.h"

#include "cons_origbranch.h"
#include "cons_masterbranch.h"
#include "pricer_gcg.h"
#include "benders_gcg.h"
#include "masterplugins.h"
#include "bendersplugins.h"
#include "cons_decomp.h"
#include "scip_misc.h"

#include "params_visu.h"

#include "gcg.h"

#ifdef WITH_BLISS
#include "pub_bliss.h"
#include "bliss_automorph.h"
#endif

#define RELAX_NAME             "gcg"
#define RELAX_DESC             "relaxator for gcg project representing the master lp"
#define RELAX_PRIORITY         -1
#define RELAX_FREQ             1
#define RELAX_INCLUDESLP       TRUE

#define DEFAULT_DISCRETIZATION TRUE
#define DEFAULT_MIPDISCRETIZATION TRUE
#define DEFAULT_AGGREGATION TRUE
#define DEFAULT_DISPINFOS FALSE
#define DEFAULT_MODE DEC_DECMODE_DANTZIGWOLFE  /**< the decomposition mode that GCG will use. (0: Dantzig-Wolfe (default),
                                                    1: Benders' decomposition, 2: solve original problem) */
#define DEFAULT_BLISS TRUE
#define DEFAULT_BLISS_SEARCH_NODE_LIMIT 0
#define DEFAULT_BLISS_GENERATOR_LIMIT 0
#define DELVARS

/*
 * Data structures
 */

/** relaxator data */

struct SCIP_RelaxData
{
   /* problems and convexity constraints */
   SCIP*                 masterprob;         /**< the master problem */
   SCIP*                 altmasterprob;      /**< the master problem for the alternate decomposition algorithm */
   SCIP**                pricingprobs;       /**< the array of pricing problems */
   int                   npricingprobs;      /**< the number of pricing problems */
   int                   nrelpricingprobs;   /**< the number of relevant pricing problems */
   int*                  blockrepresentative;/**< number of the pricing problem, that represents the i-th problem */
   int*                  nblocksidentical;   /**< number of pricing blocks represented by the i-th pricing problem */
   SCIP_CONS**           convconss;          /**< array of convexity constraints, one for each block */
   int                   ntransvars;         /**< number of variables directly transferred to the master problem */
   int                   nlinkingvars;       /**< number of linking variables */
   int                   nvarlinkconss;      /**< number of constraints that ensure that copies of linking variables have the same value */
   SCIP_Real             pricingprobsmemused; /**< sum of memory used after problem creation stage of all pricing problems */

   /* hashmaps for transformation */
   SCIP_HASHMAP*         hashorig2origvar;   /**< hashmap mapping original variables to themselves */

   /* constraint data */
   SCIP_CONS**           masterconss;        /**< array of constraints in the master problem */
   SCIP_CONS**           origmasterconss;    /**< array of constraints in the original problem that belong to the
                                              * master problem */
   SCIP_CONS**           linearmasterconss;  /**< array of linear constraints equivalent to the cons in
                                              * the original problem that belong to the master problem */
   SCIP_CONS**           varlinkconss;       /**< array of constraints ensuring linking vars equality */
   int*                  varlinkconsblock;   /**< array of constraints ensuring linking vars equality */
   int                   maxmasterconss;     /**< length of the array mastercons */
   int                   nmasterconss;       /**< number of constraints saved in mastercons */

   SCIP_SOL*             currentorigsol;     /**< current lp solution transformed into the original space */
   SCIP_Bool             origsolfeasible;    /**< is the current lp solution primal feasible in the original space? */
   SCIP_Longint          lastmasterlpiters;  /**< number of lp iterations when currentorigsol was updated the last time */
   SCIP_SOL*             lastmastersol;      /**< last feasible master solution that was added to the original problem */
   SCIP_CONS**           markedmasterconss;  /**< array of conss that are marked to be in the master */
   int                   nmarkedmasterconss; /**< number of elements in array of conss that are marked to be in the master */
   int                   maxmarkedmasterconss; /**< capacity of markedmasterconss */
   SCIP_Longint          lastsolvednodenr;   /**< node number of the node that was solved at the last call of the relaxator */

   /* branchrule data */
   GCG_BRANCHRULE**      branchrules;        /**< branching rules registered in the relaxator */
   int                   nbranchrules;       /**< number of branching rules registered in the relaxator */

   /* parameter data */
   SCIP_Bool             discretization;     /**< TRUE: use discretization approach; FALSE: use convexification approach */
   SCIP_Bool             mipdiscretization;  /**< TRUE: use discretization approach in MIPs; FALSE: use convexification approach in MIPs*/
   SCIP_Bool             aggregation;        /**< should identical blocks be aggregated (only for discretization approach)? */
   SCIP_Bool             masterissetpart;    /**< is the master a set partitioning problem? */
   SCIP_Bool             masterissetcover;   /**< is the master a set covering problem? */
   SCIP_Bool             dispinfos;          /**< should additional information be displayed? */
   DEC_DECMODE           mode;               /**< the decomposition mode for GCG. 0: Dantzig-Wolfe (default), 1: Benders' decomposition, 2: automatic */
   int                   origverblevel;      /**< the verbosity level of the original problem */
   SCIP_Bool             usebliss;           /**< should bliss be used to check for identical blocks? */
   int                   searchnodelimit;    /**< bliss search node limit (requires patched bliss version) */
   int                   generatorlimit;     /**< bliss generator limit (requires patched bliss version) */

   /* data for probing */
   SCIP_Bool             masterinprobing;    /**< is the master problem in probing mode? */
   SCIP_HEUR*            probingheur;        /**< heuristic that started probing in master problem, or NULL */
   SCIP_SOL*             storedorigsol;      /**< original solution that was stored before the probing */
   SCIP_Bool             storedfeasibility;  /**< is the stored original solution feasible? */

   /* structure information */
   DEC_DECOMP*           decomp;             /**< structure information */
   SCIP_Bool             relaxisinitialized; /**< indicates whether the relaxator is initialized */

   /* statistical information */
   SCIP_Longint          simplexiters;       /**< cumulative simplex iterations */
   SCIP_CLOCK*           rootnodetime;       /**< time in root node */

   /* visualization parameter */
   GCG_PARAMDATA*        paramsvisu;         /**< parameters for visualization */
};


/*
 * Local methods
 */

/** sets the number of the block, the given original variable belongs to */
static
SCIP_RETCODE setOriginalVarBlockNr(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAXDATA*       relaxdata,          /**< relaxator data data structure */
   SCIP_VAR*             var,                /**< variable to set the block number for */
   int                   newblock            /**< number of the block, the variable belongs to */
   )
{
   int blocknr;

   assert(scip != NULL);
   assert(var != NULL);
   assert(newblock >= 0 || (GCGgetDecompositionMode(scip) == DEC_DECMODE_BENDERS && newblock == -2));

   assert(SCIPvarIsOriginal(var) || SCIPvarGetStatus(var) == SCIP_VARSTATUS_LOOSE || SCIPvarGetStatus(var) == SCIP_VARSTATUS_COLUMN);
   assert(relaxdata != NULL);

   blocknr = GCGvarGetBlock(var);
   assert(GCGvarIsOriginal(var));

   assert(relaxdata->npricingprobs > 0);
   assert(newblock < relaxdata->npricingprobs);
   assert(blocknr >= -2 && blocknr < relaxdata->npricingprobs);

   /* var belongs to no block so far, just set the new block number */
   if( blocknr == -1 )
   {
      assert(newblock >= 0);
      GCGvarSetBlock(var, newblock);
   }
   /* if var already belongs to another block, it is a linking variable */
   else if( blocknr != newblock )
   {
      SCIP_CALL( GCGoriginalVarAddBlock(scip, var, newblock, relaxdata->npricingprobs, relaxdata->mode) );
      assert(newblock == -2 || GCGisLinkingVarInBlock(var, newblock));
      assert(GCGoriginalVarIsLinking(var));
   }
   blocknr = GCGvarGetBlock(var);
   assert(blocknr == -2 || blocknr == newblock);

   return SCIP_OKAY;
}

/** marks the constraint to be transferred to the master problem */
static
SCIP_RETCODE markConsMaster(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAXDATA*       relaxdata,          /**< relaxator data data structure */
   SCIP_CONS*            cons                /**< constraint that is forced to be in the master */
   )
{
#ifndef NDEBUG
   int i;
#endif
   assert(scip != NULL);
   assert(cons != NULL);
   assert(relaxdata != NULL);

   /* allocate array, if not yet done */
   if( relaxdata->markedmasterconss == NULL )
   {
      relaxdata->maxmarkedmasterconss = SCIPcalcMemGrowSize(scip, SCIPgetNConss(scip));
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(relaxdata->markedmasterconss), relaxdata->maxmarkedmasterconss) );
      relaxdata->nmarkedmasterconss = 0;
   }
   assert(relaxdata->nmarkedmasterconss <= SCIPgetNConss(scip));

#ifndef NDEBUG
   /* check that constraints are not marked more than one time */
   for( i = 0; i < relaxdata->nmarkedmasterconss; i++ )
      assert(relaxdata->markedmasterconss[i] != cons);
#endif

   /* save constraint */
   relaxdata->markedmasterconss[relaxdata->nmarkedmasterconss] = cons;
   relaxdata->nmarkedmasterconss++;

   return SCIP_OKAY;
}


/** converts the structure to the GCG format by setting the appropriate blocks and master constraints */
static
SCIP_RETCODE convertStructToGCG(
   SCIP*                 scip,               /**< SCIP data structure          */
   SCIP_RELAXDATA*       relaxdata,          /**< relaxator data structure     */
   DEC_DECOMP*           decomp              /**< decomp data structure        */
   )
{
   int i;
   int j;
   int k;
   int v;
   int nblocks;
   int nvars;
   SCIP_VAR** origvars;
   SCIP_HASHMAP* transvar2origvar;
   SCIP_CONS** linkingconss;
   int nlinkingconss;
   SCIP_VAR** linkingvars;
   int nlinkingvars;
   SCIP_VAR*** subscipvars;
   int* nsubscipvars;
   SCIP_CONS*** subscipconss;
   int* nsubscipconss;

   assert(decomp != NULL);
   assert(relaxdata != NULL);
   assert(scip != NULL);

   assert(DECdecompGetLinkingconss(decomp) != NULL || DECdecompGetNLinkingconss(decomp) == 0);
   assert(DECdecompGetNSubscipvars(decomp) != NULL || DECdecompGetSubscipvars(decomp) == NULL);


   SCIP_CALL( DECdecompAddRemainingConss(scip, decomp) );
   SCIP_CALL( DECdecompCheckConsistency(scip, decomp) );



   origvars = SCIPgetVars(scip);
   nvars = SCIPgetNVars(scip);
   linkingconss = DECdecompGetLinkingconss(decomp);
   nlinkingconss = DECdecompGetNLinkingconss(decomp);
   linkingvars = DECdecompGetLinkingvars(decomp);
   nlinkingvars = DECdecompGetNLinkingvars(decomp);
   subscipvars = DECdecompGetSubscipvars(decomp);
   nsubscipvars = DECdecompGetNSubscipvars(decomp);

   subscipconss = DECdecompGetSubscipconss(decomp);
   nsubscipconss = DECdecompGetNSubscipconss(decomp);
   nblocks = DECdecompGetNBlocks(decomp);

   SCIP_CALL( SCIPhashmapCreate(&transvar2origvar, SCIPblkmem(scip), nvars) );
   relaxdata->npricingprobs = nblocks;
   SCIP_CALL( GCGcreateOrigVarsData(scip) );

   SCIPdebugMessage("Copying structure with %d blocks, %d linking vars and %d linking constraints.\n", nblocks, nlinkingvars, nlinkingconss);

   /* set master constraints */
   for( i = 0; i < nlinkingconss; ++i )
   {
      assert(linkingconss[i] != NULL);
      /* SCIPdebugMessage("\tProcessing linking constraint %s.\n", SCIPconsGetName(linkingconss[i])); */
      if( SCIPconsIsActive(linkingconss[i]) )
      {
         SCIP_CALL( markConsMaster(scip, relaxdata, linkingconss[i]) );
      }
   }

   /* prepare the map from transformed to original variables */
   for( i = 0; i < nvars; ++i )
   {
      SCIP_VAR* transvar;

      SCIP_CALL( SCIPgetTransformedVar(scip, origvars[i], &transvar) );
      assert(transvar != NULL);

      SCIP_CALL( SCIPhashmapInsert(transvar2origvar, transvar, origvars[i]) );
   }

   for( i = 0; i < nblocks; ++i )
   {
      /* SCIPdebugMessage("\tProcessing block %d (%d conss, %d vars).\n", i, nsubscipconss[i], nsubscipvars[i]); */
      assert((subscipvars[i] == NULL) == (nsubscipvars[i] == 0));
      for( j = 0; j < nsubscipvars[i]; ++j )
      {
         SCIP_VAR* relevantvar;
         assert(subscipvars[i][j] != NULL);
         relevantvar = SCIPvarGetProbvar(subscipvars[i][j]);

         /* If there is a corresponding original (untransformed) variable, assign it to the block */
         if( SCIPhashmapGetImage(transvar2origvar, subscipvars[i][j]) != NULL )
         {
            SCIP_VAR* origvar;

            origvar = (SCIP_VAR*) SCIPhashmapGetImage(transvar2origvar, subscipvars[i][j]);
            assert(SCIPvarGetData(origvar) != NULL);

            SCIP_CALL( setOriginalVarBlockNr(scip, relaxdata, origvar, i) );
            SCIPdebugMessage("\t\tOriginal var %s (%p) in block %d\n", SCIPvarGetName(subscipvars[i][j]), (void*) subscipvars[i][j], i);
         }

         /* Assign the corresponding problem variable to the block */
         if( SCIPvarGetData(relevantvar) == NULL )
            SCIP_CALL( GCGorigVarCreateData(scip, relevantvar) );
         SCIP_CALL( setOriginalVarBlockNr(scip, relaxdata, relevantvar, i) );

         SCIPdebugMessage("\t\tTransformed var %s (%p) in block %d\n", SCIPvarGetName(relevantvar), (void*) relevantvar, i);

         assert(SCIPvarGetData(subscipvars[i][j]) != NULL || SCIPvarGetData(relevantvar) != NULL);
      }
   }
   SCIPdebugMessage("\tProcessing linking variables.\n");
   for( i = 0; i < nlinkingvars; ++i )
   {
      int nfound = 0;

      if( GCGoriginalVarIsLinking(linkingvars[i]) )
         continue;

      SCIPdebugMessage("\tDetecting constraint blocks of linking var %s\n", SCIPvarGetName(linkingvars[i]));
      /* HACK; @todo find out constraint blocks more intelligently */
      for( j = 0; j < nblocks; ++j )
      {
         int found = FALSE;
         for( k = 0; k < nsubscipconss[j]; ++k )
         {
            SCIP_VAR** curvars;
            int        ncurvars;
            if( SCIPconsIsDeleted(subscipconss[j][k]) )
               continue;
            ncurvars = GCGconsGetNVars(scip, subscipconss[j][k]);
            curvars = NULL;
            if( ncurvars > 0 )
            {
               SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );
               SCIP_CALL( GCGconsGetVars(scip, subscipconss[j][k], curvars, ncurvars) );

               for( v = 0; v < ncurvars; ++v )
               {
                  if( SCIPvarGetProbvar(curvars[v]) == linkingvars[i] || curvars[v] == linkingvars[i] )
                  {
                     SCIPdebugMessage("\t\t%s is in %d\n", SCIPvarGetName(SCIPvarGetProbvar(curvars[v])), j);
                     assert(SCIPvarGetData(linkingvars[i]) != NULL);
                     SCIP_CALL( setOriginalVarBlockNr(scip, relaxdata, SCIPvarGetProbvar(linkingvars[i]), j) );
                     found = TRUE;
                     break;
                  }
               }

               SCIPfreeBufferArray(scip, &curvars);
            }

            if( found )
            {
               nfound++;
               break;
            }

         }
      }

      /* if the linking variable is only in one block, then it would not have been flagged as a linking variable. In
       * the Benders' decomposition case, then linking variable needs to be flagged as linking so that it is added to
       * the master problem.
       */
      if( nfound == 1 && GCGgetDecompositionMode(scip) == DEC_DECMODE_BENDERS )
      {
         SCIP_CALL( setOriginalVarBlockNr(scip, relaxdata, SCIPvarGetProbvar(linkingvars[i]), -2) );
      }
   }

   SCIPhashmapFree(&transvar2origvar);
   return SCIP_OKAY;
}

/** ensures size of masterconss array */
static
SCIP_RETCODE ensureSizeMasterConss(
   SCIP*                 scip,
   SCIP_RELAXDATA*       relaxdata,
   int                   size
   )
{
   assert(scip != NULL);
   assert(relaxdata != NULL);
   assert(relaxdata->masterconss != NULL);

   if( relaxdata->maxmasterconss < size )
   {
      int newsize = SCIPcalcMemGrowSize(scip, size);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(relaxdata->masterconss), relaxdata->maxmasterconss, newsize) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(relaxdata->origmasterconss), relaxdata->maxmasterconss, newsize) );
      relaxdata->maxmasterconss = newsize;

   }
   assert(relaxdata->maxmasterconss >= size);

   return SCIP_OKAY;
}

/** ensures size of branchrules array: enlarges the array by 1 */
static
SCIP_RETCODE ensureSizeBranchrules(
   SCIP*                 scip,
   SCIP_RELAXDATA*       relaxdata
   )
{
   assert(scip != NULL);
   assert(relaxdata != NULL);
   assert((relaxdata->branchrules == NULL) == (relaxdata->nbranchrules == 0));

   if( relaxdata->nbranchrules == 0 )
   {
      SCIP_CALL( SCIPallocMemoryArray(scip, &(relaxdata->branchrules), 1) ); /*lint !e506*/
   }
   else
   {
      SCIP_CALL( SCIPreallocMemoryArray(scip, &(relaxdata->branchrules), (size_t)relaxdata->nbranchrules+1) );
   }

   return SCIP_OKAY;
}


/** check whether the master problem has a set partitioning or set covering structure */
static
SCIP_RETCODE checkSetppcStructure(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAXDATA*       relaxdata           /**< relaxator data structure */
   )
{
   SCIP_CONS** masterconss;
   int nmasterconss;

   int i;

   assert(relaxdata->decomp != NULL);

   masterconss = DECdecompGetLinkingconss(relaxdata->decomp);
   nmasterconss = DECdecompGetNLinkingconss(relaxdata->decomp);
   assert(nmasterconss >= 0);
   assert(masterconss != NULL || nmasterconss == 0);

   if( nmasterconss == 0 || relaxdata->nvarlinkconss > 0 )
   {
      relaxdata->masterissetcover = FALSE;
      relaxdata->masterissetpart = FALSE;
      return SCIP_OKAY;
   }

   relaxdata->masterissetcover = TRUE;
   relaxdata->masterissetpart = TRUE;

   for( i = 0; i < nmasterconss; ++i )
   {
      assert(masterconss != NULL);

      if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(masterconss[i])), "setppc") == 0 )
      {
         switch( SCIPgetTypeSetppc(scip, masterconss[i]) )
         {
         case SCIP_SETPPCTYPE_COVERING:
            relaxdata->masterissetpart = FALSE;
            break;
         case SCIP_SETPPCTYPE_PARTITIONING:
            relaxdata->masterissetcover = FALSE;
            break;
         case SCIP_SETPPCTYPE_PACKING:
            relaxdata->masterissetcover = FALSE;
            relaxdata->masterissetpart = FALSE;
            break;
         }
      }
      else if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(masterconss[i])), "logicor") == 0 )
      {
         relaxdata->masterissetpart = FALSE;
         break;
      }
      else if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(masterconss[i])), "linear") == 0 )
      {
         SCIP_SETPPCTYPE type;

         if( GCGgetConsIsSetppc(scip, masterconss[i], &type) )
         {
            switch( type )
            {
            case SCIP_SETPPCTYPE_COVERING:
               relaxdata->masterissetpart = FALSE;
               break;
            case SCIP_SETPPCTYPE_PARTITIONING:
               relaxdata->masterissetcover = FALSE;
               break;
            case SCIP_SETPPCTYPE_PACKING:
               relaxdata->masterissetcover = FALSE;
               relaxdata->masterissetpart = FALSE;
               break;
            }
         }
         else
         {
            relaxdata->masterissetcover = FALSE;
            relaxdata->masterissetpart = FALSE;
            break;
         }
      }
      else
      {
         relaxdata->masterissetcover = FALSE;
         relaxdata->masterissetpart = FALSE;
         break;
      }
   }

   if( relaxdata->masterissetcover )
   {
      assert(!relaxdata->masterissetpart);
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Master problem is a set covering problem.\n");
   }
   if( relaxdata->masterissetpart )
   {
      assert(!relaxdata->masterissetcover);
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Master problem is a set partitioning problem.\n");
   }

   return SCIP_OKAY;
}

/** checks whether two arrays of SCIP_Real's are identical */
static
SCIP_Bool realArraysAreEqual(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real*            array1,             /**< first array */
   int                   array1length,       /**< length of first array */
   SCIP_Real*            array2,             /**< second array */
   int                   array2length        /**< length of second array */
   )
{
   int i;

   if( array1length != array2length )
      return FALSE;

   if( array1length == 0 )
      return TRUE;

   assert(array1 != NULL);
   assert(array2 != NULL);

   for( i = 0; i < array1length; i++ )
   {
      if( !SCIPisEQ(scip, array1[i], array2[i]) )
         return FALSE;
   }

   return TRUE;
}

/* checks whether two pricingproblems represent identical blocks */
static
SCIP_RETCODE pricingprobsAreIdenticalFromDetectionInfo(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAXDATA*       relaxdata,          /**< the relaxator's data */
   SCIP_HASHMAP**        hashorig2pricingvar,/**< mapping from orig to pricingvar  */
   int                   probnr1,            /**< number of the first pricingproblem */
   int                   probnr2,            /**< number of the second pricingproblem */
   SCIP_HASHMAP*         varmap,             /**< hashmap mapping the variables of the second pricing problem
                                              *   to those of the first pricing problem */
   SCIP_Bool*            identical           /**< return value: are blocks identical */
   )
{
   SCIP* scip1;
   SCIP* scip2;
   int partialdecid;


   assert(relaxdata != NULL);
   assert(0 <= probnr1 && probnr1 < relaxdata->npricingprobs);
   assert(0 <= probnr2 && probnr2 < relaxdata->npricingprobs);
   assert(varmap != NULL);
   assert(identical != NULL);

   scip1 = relaxdata->pricingprobs[probnr1];
   scip2 = relaxdata->pricingprobs[probnr2];
   assert(scip1 != NULL);
   assert(scip2 != NULL);

   *identical = FALSE;

   /* 1) find partialdec number */

   partialdecid = DECdecompGetPartialdecID(relaxdata->decomp);

   /* 2) are pricingproblems identical for this partialdec? */
   SCIP_CALL( GCGconshdlrDecompArePricingprobsIdenticalForPartialdecid(scip, partialdecid, probnr2, probnr1, identical) );

   /* 3) create varmap if pricing probs are identical */
   if( *identical )
   {
      SCIP_CALL( GCGconshdlrDecompCreateVarmapForPartialdecId(scip, hashorig2pricingvar, partialdecid, probnr2, probnr1, scip2, scip1, varmap) );
   }

   return SCIP_OKAY;
}

/** checks whether there are identical pricing blocks */
static
SCIP_RETCODE checkIdenticalBlocks(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAXDATA*       relaxdata,          /**< the relaxator data data structure*/
   SCIP_HASHMAP**         hashorig2pricingvar /**< mapping from orig to pricingvar for each block */
   )
{
   SCIP_HASHMAP* varmap;
   SCIP_VAR** vars;
   SCIP_VAR* origvar;
   SCIP_VAR* pricingvar;
   int nvars;
   SCIP_Bool identical;

   int i;
   int j;
   int k;

   int nrelevant;

   SCIPdebugMessage("checking identical blocks \n");
   assert(scip != NULL);
   assert(relaxdata != NULL);

   for( i = 0; i < relaxdata->npricingprobs; i++ )
   {
      relaxdata->blockrepresentative[i] = i;
      relaxdata->nblocksidentical[i] = 1;
   }

   relaxdata->nrelpricingprobs = relaxdata->npricingprobs;
   nrelevant = 0;

   if(  ( !relaxdata->discretization || !relaxdata->aggregation ) )
   {
      SCIPdebugMessage("discretization is off, aggregation is off\n");
      return SCIP_OKAY;
   }


   if(  relaxdata->nlinkingvars != 0 )
      {
         SCIPdebugMessage("aggregation is off in presence of linking vars\n");
         return SCIP_OKAY;
      }


   for( i = 0; i < relaxdata->npricingprobs; i++ )
   {
      for( j = 0; j < i && relaxdata->blockrepresentative[i] == i; j++ )
      {
         if( relaxdata->blockrepresentative[j] != j )
            continue;

         SCIP_CALL( SCIPhashmapCreate(&varmap,
               SCIPblkmem(scip),
               5 * SCIPgetNVars(relaxdata->pricingprobs[i])+1) ); /* +1 to deal with empty subproblems */

         assert( SCIPgetNConss(scip) == GCGconshdlrDecompGetNFormerDetectionConssForID(scip, DECdecompGetPartialdecID(relaxdata->decomp)) );
         SCIPdebugMessage( "nconss: %d; ndetectionconss: %d -> using partialdec information for identity test \n", SCIPgetNConss(scip), GCGconshdlrDecompGetNFormerDetectionConssForID(scip, DECdecompGetPartialdecID(relaxdata->decomp) ) );
         SCIP_CALL( pricingprobsAreIdenticalFromDetectionInfo( scip, relaxdata, hashorig2pricingvar, i, j, varmap, &identical ) );

/*
 *  new method of cons_decomp that uses partialdec information
 * 1) check varmap
 * 2) build varmap for partialdecs in partialdec datatstructures
 * 3) translate varmap when transforming partialdec to decomp (store varmap in decomp or partialdec?)
 * 4) write method in cons_decomp using partialdec agg info and varmap*/

         if( identical )
         {
            SCIPdebugMessage("Block %d is identical to block %d!\n", i, j);

            /* save variables in pricing problem variable */
            vars = SCIPgetVars(relaxdata->pricingprobs[i]);
            nvars = SCIPgetNVars(relaxdata->pricingprobs[i]);

            /*
             * quick check whether some of the variables are linking in which case we cannot aggregate
             * this is suboptimal but we use bliss anyway
             */

            for( k = 0; k < nvars; k++ )
            {
               assert(GCGvarIsPricing(vars[k]));
               origvar = GCGpricingVarGetOrigvars(vars[k])[0];
               if( GCGoriginalVarIsLinking(origvar) )
               {
                  SCIPdebugMessage("Var <%s> is linking and can not be aggregated.\n", SCIPvarGetName(origvar));
                  identical = FALSE;
                  break;
               }
            }

            if( !identical )
            {
               break;
            }


            /* block i will be represented by block j */
            relaxdata->blockrepresentative[i] = j;
            relaxdata->nblocksidentical[i] = 0;
            relaxdata->nblocksidentical[j]++;

            for( k = 0; k < nvars; k++ )
            {
               int blocknr;
               assert(GCGvarIsPricing(vars[k]));
               origvar = GCGpricingVarGetOrigvars(vars[k])[0];

               pricingvar = (SCIP_VAR*) SCIPhashmapGetImage(varmap, (void*) vars[k]);
               assert(pricingvar != NULL);
               blocknr = GCGvarGetBlock(pricingvar);

               assert(GCGvarIsPricing(pricingvar));
               assert(GCGvarIsOriginal(origvar));
               assert(GCGoriginalVarGetPricingVar(origvar) != NULL);
               GCGoriginalVarSetPricingVar(origvar, pricingvar);
               SCIP_CALL( GCGpricingVarAddOrigVar(relaxdata->pricingprobs[blocknr], pricingvar, origvar) );
            }

         }
         SCIPhashmapFree(&varmap);

      }
      if( relaxdata->blockrepresentative[i] == i )
      {
         SCIPdebugMessage("Block %d is relevant!\n", i);
         nrelevant++;
      }
   }

   SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Matrix has %d blocks, using %d%s pricing problem%s.\n",
      relaxdata->npricingprobs, nrelevant, (relaxdata->npricingprobs == nrelevant ? "" : " aggregated"), (nrelevant == 1 ? "" : "s"));

   relaxdata->nrelpricingprobs = nrelevant;

   return SCIP_OKAY;
}

/** sets the pricing problem parameters */
static
SCIP_RETCODE setPricingProblemParameters(
   SCIP*                 scip,               /**< SCIP data structure of the pricing problem */
   int                   clocktype,          /**< clocktype to use in the pricing problem */
   SCIP_Real             infinity,           /**< values larger than this are considered infinity in the pricing problem */
   SCIP_Real             epsilon,            /**< absolute values smaller than this are considered zero in the pricing problem */
   SCIP_Real             sumepsilon,         /**< absolute values of sums smaller than this are considered zero in the pricing problem */
   SCIP_Real             feastol,            /**< feasibility tolerance for constraints in the pricing problem */
   SCIP_Real             lpfeastolfactor,    /**< primal feasibility tolerance factor of LP solver in the pricing problem */
   SCIP_Real             dualfeastol,        /**< feasibility tolerance for reduced costs in LP solution in the pricing problem */
   SCIP_Bool             enableppcuts        /**< should ppcuts be stored for sepa_basis */
   )
{
   assert(scip != NULL);

   /* disable conflict analysis */
   SCIP_CALL( SCIPsetBoolParam(scip, "conflict/useprop", FALSE) );
   SCIP_CALL( SCIPsetCharParam(scip, "conflict/useinflp", 'o') );
   SCIP_CALL( SCIPsetCharParam(scip, "conflict/useboundlp", 'o') );
   SCIP_CALL( SCIPsetBoolParam(scip, "conflict/usesb", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(scip, "conflict/usepseudo", FALSE) );

   /* reduce the effort spent for hash tables */
   SCIP_CALL( SCIPsetBoolParam(scip, "misc/usevartable", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(scip, "misc/useconstable", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(scip, "misc/usesmalltables", TRUE) );

   /* disable expensive presolving */
   /* @todo test whether this really helps, perhaps set presolving emphasis to fast? */
   SCIP_CALL( SCIPsetBoolParam(scip, "constraints/linear/presolpairwise", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(scip, "constraints/setppc/presolpairwise", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(scip, "constraints/logicor/presolpairwise", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(scip, "constraints/linear/presolusehashing", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(scip, "constraints/setppc/presolusehashing", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(scip, "constraints/logicor/presolusehashing", FALSE) );

   /* disable dual fixing presolver for the moment, because we want to avoid variables fixed to infinity */
   SCIP_CALL( SCIPsetIntParam(scip, "propagating/dualfix/freq", -1) );
   SCIP_CALL( SCIPsetIntParam(scip, "propagating/dualfix/maxprerounds", 0) );
   SCIP_CALL( SCIPfixParam(scip, "propagating/dualfix/freq") );
   SCIP_CALL( SCIPfixParam(scip, "propagating/dualfix/maxprerounds") );


   /* disable solution storage ! */
   SCIP_CALL( SCIPsetIntParam(scip, "limits/maxorigsol", 0) );

   /* disable multiaggregation because of infinite values */
   SCIP_CALL( SCIPsetBoolParam(scip, "presolving/donotmultaggr", TRUE) );

   /* @todo enable presolving and propagation of xor constraints if bug is fixed */

   /* disable presolving and propagation of xor constraints as work-around for a SCIP bug */
   SCIP_CALL( SCIPsetIntParam(scip, "constraints/xor/maxprerounds", 0) );
   SCIP_CALL( SCIPsetIntParam(scip, "constraints/xor/propfreq", -1) );

   /* disable output to console */
   SCIP_CALL( SCIPsetIntParam(scip, "display/verblevel", (int)SCIP_VERBLEVEL_NONE) );
#if SCIP_VERSION > 210
   SCIP_CALL( SCIPsetBoolParam(scip, "misc/printreason", FALSE) );
#endif
   SCIP_CALL( SCIPsetIntParam(scip, "limits/maxorigsol", 0) );
   SCIP_CALL( SCIPfixParam(scip, "limits/maxorigsol") );

   /* do not abort subproblem on CTRL-C */
   SCIP_CALL( SCIPsetBoolParam(scip, "misc/catchctrlc", FALSE) );

   /* set clock type */
   SCIP_CALL( SCIPsetIntParam(scip, "timing/clocktype", clocktype) );

   SCIP_CALL( SCIPsetBoolParam(scip, "misc/calcintegral", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(scip, "misc/finitesolutionstore", TRUE) );

   SCIP_CALL( SCIPsetRealParam(scip, "numerics/infinity", infinity) );
   SCIP_CALL( SCIPsetRealParam(scip, "numerics/epsilon", epsilon) );
   SCIP_CALL( SCIPsetRealParam(scip, "numerics/sumepsilon", sumepsilon) );
   SCIP_CALL( SCIPsetRealParam(scip, "numerics/feastol", feastol) );
   SCIP_CALL( SCIPsetRealParam(scip, "numerics/lpfeastolfactor", lpfeastolfactor) );
   SCIP_CALL( SCIPsetRealParam(scip, "numerics/dualfeastol", dualfeastol) );

   /* jonas' stuff */
   if( enableppcuts )
   {
      int pscost;
      int prop;

      SCIP_CALL( SCIPgetIntParam(scip, "branching/pscost/priority", &pscost) );
      SCIP_CALL( SCIPgetIntParam(scip, "propagating/maxroundsroot", &prop) );
      SCIP_CALL( SCIPsetIntParam(scip, "branching/pscost/priority", 11000) );
      SCIP_CALL( SCIPsetIntParam(scip, "propagating/maxroundsroot", 0) );
      SCIP_CALL( SCIPsetPresolving(scip, SCIP_PARAMSETTING_OFF, TRUE) );
   }
   return SCIP_OKAY;
}


/** creates a variable in a pricing problem corresponding to the given original variable (belonging to exactly one block) */
static
SCIP_RETCODE createPricingVar(
   SCIP_RELAXDATA*       relaxdata,          /**< relaxator data data structure */
   SCIP_VAR*             origvar             /**< corresponding variable in the original program */
   )
{
   SCIP_VAR* var;
   int pricingprobnr;

   assert(relaxdata != NULL);
   assert(origvar != NULL);

   pricingprobnr = GCGvarGetBlock(origvar);
   assert(pricingprobnr >= 0);

   SCIP_CALL( GCGoriginalVarCreatePricingVar(relaxdata->pricingprobs[pricingprobnr], origvar, &var) );
   assert(var != NULL);

   GCGoriginalVarSetPricingVar(origvar, var);
   SCIP_CALL( SCIPaddVar(relaxdata->pricingprobs[pricingprobnr], var) );
   assert(GCGvarIsPricing(var));
   /* because the variable was added to the problem,
    * it is captured by SCIP and we can safely release it right now
    */
   SCIP_CALL( SCIPreleaseVar(relaxdata->pricingprobs[pricingprobnr], &var) );

   return SCIP_OKAY;
}

/** creates a variable in each of the pricing problems linked by given original variable */
static
SCIP_RETCODE createLinkingPricingVars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAXDATA*       relaxdata,          /**< relaxator data data structure */
   SCIP_VAR*             origvar             /**< corresponding linking variable in the original program */
   )
{
   SCIP_VAR* var;
   SCIP_CONS* linkcons;
#ifndef NDEBUG
   SCIP_CONS** linkconss;
   int nblocks;
#endif
   SCIP_VAR** pricingvars;
   int i;

   assert(origvar != NULL);
   assert(relaxdata != NULL);

   /* get variable data of the original variable */
   assert(GCGvarIsOriginal(origvar));
   assert(GCGoriginalVarIsLinking(origvar));
   pricingvars = GCGlinkingVarGetPricingVars(origvar);

#ifndef NDEBUG
   nblocks = GCGlinkingVarGetNBlocks(origvar);
   /* checks that GCGrelaxSetOriginalVarBlockNr() worked correctly */
   {
      int count;

      linkconss = GCGlinkingVarGetLinkingConss(origvar);
      /* the linking constraints could be NULL if the Benders' decomposition is used. */
      if( linkconss != NULL )
      {
         count = 0;
         for( i = 0; i < relaxdata->npricingprobs; i++ )
         {
            assert(linkconss[i] == NULL);

            if( pricingvars[i] != NULL )
               count++;
         }
         assert(nblocks == count);
      }
   }
#endif

   for( i = 0; i < relaxdata->npricingprobs; ++i )
   {
      if( pricingvars[i] == NULL )
         continue;

      SCIP_CALL( GCGlinkingVarCreatePricingVar(relaxdata->pricingprobs[i], i, origvar, &var) );

      GCGlinkingVarSetPricingVar(origvar, i, var);

      assert(GCGvarIsPricing(var));
      SCIP_CALL( SCIPaddVar(relaxdata->pricingprobs[i], var) );


      if( relaxdata->mode != DEC_DECMODE_BENDERS )
      {
         SCIP_CALL( GCGlinkingVarCreateMasterCons(relaxdata->masterprob, i, origvar, &linkcons) );
         GCGlinkingVarSetLinkingCons(origvar, linkcons, i);
         SCIP_CALL( SCIPaddCons(relaxdata->masterprob, linkcons) );

         SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &relaxdata->varlinkconss, relaxdata->nvarlinkconss, (size_t)relaxdata->nvarlinkconss+1) );
         SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &relaxdata->varlinkconsblock, relaxdata->nvarlinkconss, (size_t)relaxdata->nvarlinkconss+1) );

         relaxdata->varlinkconss[relaxdata->nvarlinkconss] = linkcons;
         relaxdata->varlinkconsblock[relaxdata->nvarlinkconss] = i;
         relaxdata->nvarlinkconss++;
      }

      /* because the variable was added to the problem,
       * it is captured by SCIP and we can safely release it right now
       */
      SCIP_CALL( SCIPreleaseVar(relaxdata->pricingprobs[i], &var) );
   }

#ifndef NDEBUG
   /* checks that createLinkingPricingVars() worked correctly */
   {
      int count;

      linkconss = GCGlinkingVarGetLinkingConss(origvar);
      /* the linking constraints could be NULL if the Benders' decomposition is used. */
      if( linkconss != NULL )
      {
         count = 0;
         for( i = 0; i < relaxdata->npricingprobs; i++ )
         {
            if( pricingvars[i] != NULL )
            {
               count++;
               assert(GCGvarIsPricing(pricingvars[i]));
               assert(relaxdata->mode == DEC_DECMODE_BENDERS || linkconss[i] != NULL);
            }
            else
               assert(linkconss[i] == NULL);
         }
         assert(nblocks == count);
      }
   }
#endif


   return SCIP_OKAY;
}

/** create pricing problem variables */
static
SCIP_RETCODE createPricingVariables(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAXDATA*       relaxdata,          /**< relaxator data data structure */
   SCIP_HASHMAP**        hashorig2pricingvar /**< hashmap mapping original variables to pricing variables */
   )
{
   SCIP_VAR** vars;
   int nvars;
   int v;
   int i;
   int npricingprobs;

   assert(scip != NULL);
   assert(relaxdata != NULL);

   /* create pricing variables and map them to the original variables */
   vars = SCIPgetVars(scip);
   nvars = SCIPgetNVars(scip);
   npricingprobs = relaxdata->npricingprobs;

   for( v = 0; v < nvars; v++ )
   {
      int blocknr;
      SCIP_VAR* probvar;

      assert(SCIPvarIsTransformed(vars[v]));

      probvar = SCIPvarGetProbvar(vars[v]);
      assert(SCIPvarIsTransformed(probvar));
      blocknr = GCGvarGetBlock(probvar);
      if( blocknr == -1 )
      {
         int tempblock;
         tempblock = (int) (size_t) SCIPhashmapGetImage(DECdecompGetVartoblock(relaxdata->decomp), probvar)-1; /*lint !e507*/
         if( tempblock >= DECdecompGetNBlocks(relaxdata->decomp) )
         {
            blocknr = -1;
         }
         else
         {
            SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " changed block number to %d \n", tempblock );
            blocknr = tempblock; /*lint !e806*/
         }
      }

      SCIPdebugMessage("Creating map for (%p, %p) var %s:", (void*)(vars[v]), (void*)(probvar), SCIPvarGetName(probvar));
      assert( !SCIPhashmapExists(relaxdata->hashorig2origvar, probvar) );
      SCIP_CALL( SCIPhashmapInsert(relaxdata->hashorig2origvar, (void*)(probvar), (void*)(probvar)) );

      /* variable belongs to exactly one block --> create corresponding pricing variable*/
      if( blocknr >= 0 )
      {
         SCIPdebugPrintf("block %d", blocknr);

         assert(GCGoriginalVarGetPricingVar(probvar) == NULL);
         SCIP_CALL( createPricingVar(relaxdata, probvar) );
         assert(GCGoriginalVarGetPricingVar(probvar) != NULL);
         assert(hashorig2pricingvar != NULL);
         assert(hashorig2pricingvar[blocknr] != NULL);

         SCIPdebugPrintf("-> %p\n", (void*) GCGoriginalVarGetPricingVar(probvar));

         assert(!SCIPhashmapExists(hashorig2pricingvar[blocknr], probvar));
         SCIP_CALL( SCIPhashmapInsert(hashorig2pricingvar[blocknr], (void*)(probvar),
               (void*)(GCGoriginalVarGetPricingVar(probvar)) ));

         assert(GCGvarIsPricing((SCIP_VAR*) SCIPhashmapGetImage(hashorig2pricingvar[blocknr], probvar)));
      }
      /* variable is a linking variable --> create corresponding pricing variable in all linked blocks
       * and create corresponding linking constraints */
      else if( GCGoriginalVarIsLinking(probvar) )
      {
         SCIP_VAR** pricingvars;
         SCIPdebugPrintf("linking.\n");
         relaxdata->nlinkingvars++;
         SCIP_CALL( createLinkingPricingVars(scip, relaxdata, probvar) );
         assert(GCGlinkingVarGetPricingVars(probvar) != NULL);


         pricingvars = GCGlinkingVarGetPricingVars(probvar);

         for( i = 0; i < npricingprobs; i++ )
         {
            if( pricingvars[i] != NULL )
            {
               assert(GCGvarIsPricing(pricingvars[i]));
               assert(hashorig2pricingvar != NULL);
               assert(hashorig2pricingvar[i] != NULL);
               assert(!SCIPhashmapExists(hashorig2pricingvar[i], probvar));
               SCIP_CALL( SCIPhashmapInsert(hashorig2pricingvar[i], (void*)(probvar),
                     (void*)(pricingvars[i])) );
               assert(GCGvarIsPricing((SCIP_VAR*) SCIPhashmapGetImage(hashorig2pricingvar[i], probvar)));
            }
         }
      }
      else
      {
         assert(GCGvarGetBlock(probvar) == -1);
         assert(GCGoriginalVarGetPricingVar(probvar) == NULL);
         SCIPdebugPrintf("master!\n");
         relaxdata->ntransvars++;
      }
      assert(SCIPhashmapExists(relaxdata->hashorig2origvar, probvar));
   }

   return SCIP_OKAY;
}


/** displays statistics of the pricing problems */
static
SCIP_RETCODE displayPricingStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP**                pricingprobs,       /**< array of pricing problems */
   int                   npricingprobs,      /**< number of pricingproblems */
   int*                  blockrepresentative /**< array of representation information */
)
{
   char name[SCIP_MAXSTRLEN];
   int i;

   assert(scip != NULL);
   assert(pricingprobs != NULL);
   assert(blockrepresentative != NULL);
   assert(npricingprobs > 0);
   for( i = 0; i < npricingprobs; i++ )
   {
      int nbin;
      int nint;
      int nimpl;
      int ncont;

      if( blockrepresentative[i] != i )
         continue;

      SCIP_CALL( SCIPgetVarsData(pricingprobs[i], NULL, NULL, &nbin, &nint, &nimpl, &ncont) );

      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "pricing problem %d: %d conss, %d vars (%d bins, %d ints, %d impls and %d cont)\n", i,
         SCIPgetNConss(pricingprobs[i]), SCIPgetNVars(pricingprobs[i]), nbin, nint, nimpl, ncont);

      (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "pricingprob_%d.lp", i);
      SCIP_CALL( SCIPwriteOrigProblem(pricingprobs[i], name, NULL, FALSE) );
   }

   return SCIP_OKAY;
}


/** allocates initial problem specific data */
static
SCIP_RETCODE initRelaxProblemdata(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAXDATA*       relaxdata           /**< relaxatordata data structure */
   )
{
   assert(scip != NULL);
   assert(relaxdata != NULL);

   /* initialize relaxator data */
   relaxdata->maxmasterconss = 16;
   relaxdata->nmasterconss = 0;

   /* arrays of constraints belonging to the master problems */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(relaxdata->masterconss), relaxdata->maxmasterconss) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(relaxdata->origmasterconss), relaxdata->maxmasterconss) );

   if( relaxdata->npricingprobs > 0 )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(relaxdata->pricingprobs), relaxdata->npricingprobs) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(relaxdata->blockrepresentative), relaxdata->npricingprobs) );
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(relaxdata->nblocksidentical), relaxdata->npricingprobs) );

      /* array for saving convexity constraints belonging to one of the pricing problems */
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(relaxdata->convconss), relaxdata->npricingprobs) );
   }

   SCIP_CALL( SCIPhashmapCreate(&(relaxdata->hashorig2origvar), SCIPblkmem(scip), 10*SCIPgetNVars(scip)+1) );

   return SCIP_OKAY;
}


/** creates the master problem with the specified name */
static
SCIP_RETCODE createMasterProblem(
   SCIP*                 masterscip,         /**< SCIP data structure of master problem */
   const char*           name,               /**< name of the master problem */
   int                   clocktype,          /**< clocktype to use in the master problem */
   SCIP_Real             infinity,           /**< values larger than this are considered infinity in the master problem */
   SCIP_Real             epsilon,            /**< absolute values smaller than this are considered zero in the master problem */
   SCIP_Real             sumepsilon,         /**< absolute values of sums smaller than this are considered zero in the master problem */
   SCIP_Real             feastol,            /**< feasibility tolerance for constraints in the master problem */
   SCIP_Real             lpfeastolfactor,    /**< primal feasibility tolerance factor of LP solver in the master problem */
   SCIP_Real             dualfeastol,        /**< feasibility tolerance for reduced costs in LP solution in the master problem */
   DEC_DECMODE           mode                /**< the decomposition mode */
   )
{
   assert(masterscip != NULL);
   assert(name != NULL);

   SCIP_CALL( SCIPcreateProb(masterscip, name, NULL, NULL, NULL, NULL, NULL, NULL, NULL) );

   /* set clocktype */
   SCIP_CALL( SCIPsetIntParam(masterscip, "timing/clocktype", clocktype) );

   /* set numerical tolerances */
   SCIP_CALL( SCIPsetRealParam(masterscip, "numerics/infinity", infinity) );
   SCIP_CALL( SCIPsetRealParam(masterscip, "numerics/epsilon", epsilon) );
   SCIP_CALL( SCIPsetRealParam(masterscip, "numerics/sumepsilon", sumepsilon) );
   SCIP_CALL( SCIPsetRealParam(masterscip, "numerics/feastol", feastol) );
   SCIP_CALL( SCIPsetRealParam(masterscip, "numerics/lpfeastolfactor", lpfeastolfactor) );
   SCIP_CALL( SCIPsetRealParam(masterscip, "numerics/dualfeastol", dualfeastol) );

   /* disable aggregation and multiaggregation of variables, as this might lead to issues with copied original variables */
   SCIP_CALL( SCIPsetBoolParam(masterscip, "presolving/donotaggr", TRUE) );
   SCIP_CALL( SCIPsetBoolParam(masterscip, "presolving/donotmultaggr", TRUE) );

   /* the following settings are for decomposition, so if the original problem is solved directly, then these settings
    * are not required
    */
   if( mode == DEC_DECMODE_ORIGINAL )
   {
      return SCIP_OKAY;
   }

   if( mode == DEC_DECMODE_DANTZIGWOLFE )
      SCIP_CALL( SCIPactivatePricer(masterscip, SCIPfindPricer(masterscip, "gcg")) );

   /* do not modify the time limit after solving the master problem */
   SCIP_CALL( SCIPsetBoolParam(masterscip, "reoptimization/commontimelimit", FALSE) );

   /* disable aggregation and multiaggregation of variables, as this might lead to issues with copied original variables */
   SCIP_CALL( SCIPsetBoolParam(masterscip, "presolving/donotaggr", TRUE) );
   SCIP_CALL( SCIPsetBoolParam(masterscip, "presolving/donotmultaggr", TRUE) );

   /* for Benders' decomposition, some additional parameter settings are required for the master problem. */
   if( mode == DEC_DECMODE_BENDERS )
   {
      SCIP_CALL( SCIPsetSeparating(masterscip, SCIP_PARAMSETTING_OFF, TRUE) );
      SCIP_CALL( SCIPsetPresolving(masterscip, SCIP_PARAMSETTING_OFF, TRUE) );
      SCIP_CALL( SCIPsetIntParam(masterscip, "presolving/maxrestarts", 0) );
      SCIP_CALL( SCIPsetIntParam(masterscip, "propagating/maxroundsroot", 0) );
      SCIP_CALL( SCIPsetIntParam(masterscip, "heuristics/trysol/freq", 1) );
      SCIP_CALL( SCIPsetBoolParam(masterscip, "constraints/benders/active", TRUE) );
      SCIP_CALL( SCIPsetBoolParam(masterscip, "constraints/benderslp/active", TRUE) );
      SCIP_CALL( SCIPsetBoolParam(masterscip, "benders/gcg/lnscheck", FALSE) );
      SCIP_CALL( SCIPsetIntParam(masterscip, "presolving/maxrounds", 1) );
      SCIP_CALL( SCIPsetIntParam(masterscip, "constraints/benders/maxprerounds", 1) );

      /* the trysol heuristic must have a high priority to ensure the solutions found by the relaxator are added to the
       * original problem
       */
      SCIP_CALL( SCIPsetIntParam(GCGgetOriginalprob(masterscip), "heuristics/trysol/freq", 1) );
   }

   return SCIP_OKAY;
}


/** creates the pricing problem with the specified name */
static
SCIP_RETCODE createPricingProblem(
   SCIP**                pricingscip,        /**< Pricing scip data structure */
   const char*           name,               /**< name of the pricing problem */
   int                   clocktype,          /**< clocktype to use in the pricing problem */
   SCIP_Real             infinity,           /**< values larger than this are considered infinity in the pricing problem */
   SCIP_Real             epsilon,            /**< absolute values smaller than this are considered zero in the pricing problem */
   SCIP_Real             sumepsilon,         /**< absolute values of sums smaller than this are considered zero in the pricing problem */
   SCIP_Real             feastol,            /**< feasibility tolerance for constraints in the pricing problem */
   SCIP_Real             lpfeastolfactor,    /**< primal feasibility tolerance factor of LP solver in the pricing problem */
   SCIP_Real             dualfeastol,        /**< feasibility tolerance for reduced costs in LP solution in the pricing problem */
   SCIP_Bool             enableppcuts        /**< should ppcuts be stored for sepa_basis */
   )
{
   assert(pricingscip != NULL);
   assert(name != NULL);

   SCIP_CALL( SCIPcreate(pricingscip) );
   SCIP_CALL( SCIPincludeDefaultPlugins(*pricingscip) );
   SCIP_CALL( setPricingProblemParameters(*pricingscip, clocktype, infinity, epsilon, sumepsilon, feastol, lpfeastolfactor, dualfeastol, enableppcuts) );
   SCIP_CALL( SCIPcreateProb(*pricingscip, name, NULL, NULL, NULL, NULL, NULL, NULL, NULL) );

   return SCIP_OKAY;
}


/** saves the coefficient of the masterconstraints in the original variable */
static
SCIP_RETCODE saveOriginalVarMastercoeffs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            origvars,           /**< original variables array */
   int                   norigvars,          /**< size of original variables array*/
   int                   nmasterconss,       /**< size of masterconns array */
   SCIP_CONS**           origmasterconss,    /**< orig master constraints array */
   SCIP_CONS**           masterconss         /**< master constraints */
   )
{
   int v;
   int i;

   assert(scip != NULL);
   assert(origvars != NULL || norigvars == 0);
   assert(norigvars >= 0);
   assert(nmasterconss >= 0);
   assert(masterconss != NULL);
   assert(origmasterconss != NULL);

   /* for original variables, save the coefficients in the master problem */
   for( v = 0; v < norigvars; v++ )
   {
      SCIP_VAR* var;
      var = SCIPvarGetProbvar(origvars[v]); /*lint !e613*/
      assert(GCGvarIsOriginal(var));
      assert(GCGoriginalVarGetCoefs(var) == NULL);
      GCGoriginalVarSetNCoefs(var, 0);
   }

   /* save coefs */
   for( i = 0; i < nmasterconss; i++ )
   {
      SCIP_VAR** vars;
      SCIP_Real* vals;
      int nvars;

      nvars = GCGconsGetNVars(scip, origmasterconss[i]);
      SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &vals, nvars) );
      GCGconsGetVars(scip, origmasterconss[i], vars, nvars);
      GCGconsGetVals(scip, origmasterconss[i], vals, nvars);
      for( v = 0; v < nvars; v++ )
      {
         SCIP_CALL( GCGoriginalVarAddCoef(scip, vars[v], vals[v], masterconss[i]) );
      }
      SCIPfreeBufferArray(scip, &vals);
      SCIPfreeBufferArray(scip, &vars);
   }

   return SCIP_OKAY;
}

/** creates the master problem constraints */
static
SCIP_RETCODE createMasterprobConss(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAXDATA*       relaxdata           /**< the relaxator data data structure */
   )
{
   SCIP_CONS** masterconss;
   int nmasterconss;
   SCIP_CONS* mastercons;
   int c;
   char name[SCIP_MAXSTRLEN];

   masterconss = DECdecompGetLinkingconss(relaxdata->decomp);
   nmasterconss = DECdecompGetNLinkingconss(relaxdata->decomp);

 //  assert(SCIPhashmapGetNElements(relaxdata->hashorig2origvar) == SCIPgetNVars(scip));
   for( c = 0; c < nmasterconss; ++c )
   {
      int nconsvars;
      int consvarssize;
      SCIP_VAR** consvars;
      SCIP_Real* consvals;
      SCIP_Bool* releasevars;
      int i;

      if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(masterconss[c])), "origbranch") == 0 )
         continue;

      /* in the Benders' decomposition mode, all variables from the linking constraints need to be added to the master
       * problem. Additionally, if the original problem is solved directly, then we must ensure that all variables are
       * added to the master problem.
       */
      if( GCGgetDecompositionMode(scip) == DEC_DECMODE_BENDERS || GCGgetDecompositionMode(scip) == DEC_DECMODE_ORIGINAL )
      {
         nconsvars = GCGconsGetNVars(scip, masterconss[c]);
         consvarssize = nconsvars;

         SCIP_CALL( SCIPallocBufferArray(scip, &consvars, consvarssize) );
         SCIP_CALL( SCIPallocBufferArray(scip, &consvals, consvarssize) );
         SCIP_CALL( SCIPallocClearBufferArray(scip, &releasevars, consvarssize) );

         SCIP_CALL( GCGconsGetVars(scip, masterconss[c], consvars, nconsvars) );
         SCIP_CALL( GCGconsGetVals(scip, masterconss[c], consvals, nconsvars) );

         for( i = 0; i < nconsvars; i++ )
         {
            SCIP_VAR* origvar;

            /* if the variable is a linking variable or is directly transferred to the master problem, then it is not
             * added to the constraint. This is because the linking variables and the transferred variables are added
             * later in GCGmasterCreateInitialMastervars().
             */
            while( i < nconsvars && (GCGoriginalVarIsLinking(consvars[i]) || GCGoriginalVarIsTransVar(consvars[i])) )
            {
               consvars[i] = consvars[nconsvars - 1];
               consvals[i] = consvals[nconsvars - 1];
               nconsvars--;
            }

            if( i >= nconsvars )
               break;

            /* assigning the origvar to the next variables that is not a linking variable */
            origvar = consvars[i];

            assert(GCGoriginalVarGetNMastervars(origvar) <= 1);

            /* if the original has already has a copy in the master problem, then this is used. Otherwise, the master
             * problem variable is created.
             */
            if( GCGoriginalVarGetNMastervars(origvar) > 0 )
            {
               consvars[i] = GCGoriginalVarGetMastervars(origvar)[0];
               releasevars[i] = FALSE;
            }
            else
            {
               SCIP_CALL( GCGcreateInitialMasterVar(relaxdata->masterprob, consvars[i], &consvars[i]) );
               SCIP_CALL( SCIPaddVar(relaxdata->masterprob, consvars[i]) );

               SCIP_CALL( GCGoriginalVarAddMasterVar(scip, origvar, consvars[i], 1.0) );

               releasevars[i] = TRUE;
            }

            assert(GCGoriginalVarGetNMastervars(origvar) <= 1);
         }
      }
      else
      {
         nconsvars = 0;
         consvars = NULL;
         consvals = NULL;
         releasevars = NULL;
      }

      /* create and add corresponding linear constraint in the master problem */
      (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "m_%s", SCIPconsGetName(masterconss[c]));
      SCIP_CALL( SCIPcreateConsLinear(relaxdata->masterprob, &mastercons, name, nconsvars, consvars, consvals,
            GCGconsGetLhs(scip, masterconss[c]), GCGconsGetRhs(scip, masterconss[c]),
            TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

      SCIP_CALL( SCIPaddCons(relaxdata->masterprob, mastercons) );
      SCIPdebugMessage("Copying %s to masterproblem\n", SCIPconsGetName(masterconss[c]));
      /* store the constraints in the arrays origmasterconss and masterconss in the problem data */
      SCIP_CALL( ensureSizeMasterConss(scip, relaxdata, relaxdata->nmasterconss+1) );
      SCIP_CALL( SCIPcaptureCons(scip, masterconss[c]) );
      relaxdata->origmasterconss[relaxdata->nmasterconss] = masterconss[c];
      relaxdata->masterconss[relaxdata->nmasterconss] = mastercons;
      relaxdata->nmasterconss++;

      /* in the Benders' decomposition mode, the consvars and consvals arrays need to be freed */
      if( GCGgetDecompositionMode(scip) == DEC_DECMODE_BENDERS || GCGgetDecompositionMode(scip) == DEC_DECMODE_ORIGINAL )
      {
         assert(releasevars != NULL);
         assert(consvars != NULL);
         for( i = 0; i < nconsvars; i++ )
         {
            if( releasevars[i] )
            {
               SCIP_CALL( SCIPreleaseVar(relaxdata->masterprob, &consvars[i]) );
            }
         }

         SCIPfreeBufferArray(scip, &releasevars);
         SCIPfreeBufferArray(scip, &consvals);
         SCIPfreeBufferArray(scip, &consvars);
      }
   }
   assert(relaxdata->nmasterconss == nmasterconss);
   SCIP_CALL( saveOriginalVarMastercoeffs(scip, SCIPgetVars(scip), SCIPgetNVars(scip), relaxdata->nmasterconss, relaxdata->origmasterconss, relaxdata->masterconss) );

   return SCIP_OKAY;
}

/** creates the pricing problem constraints */
static
SCIP_RETCODE createPricingprobConss(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAXDATA*       relaxdata,          /**< the relaxator data data structure */
   SCIP_HASHMAP**        hashorig2pricingvar /**< hashmap mapping original to corresponding pricing variables */
   )
{
   SCIP_CONS*** subscipconss;
   int* nsubscipconss;
   SCIP_CONS* newcons;
   SCIP_HASHMAP* hashorig2pricingconstmp;
   int nblocks;
   int b;
   int c;
   char name[SCIP_MAXSTRLEN];
   SCIP_Bool success;

   assert(scip != NULL);
   assert(relaxdata != NULL);

   subscipconss = DECdecompGetSubscipconss(relaxdata->decomp);
   nsubscipconss = DECdecompGetNSubscipconss(relaxdata->decomp);
   nblocks = DECdecompGetNBlocks(relaxdata->decomp);

   SCIP_CALL( SCIPhashmapCreate(&hashorig2pricingconstmp, SCIPblkmem(scip), SCIPgetNConss(scip)) ); /*lint !e613*/

   for( b = 0; b < nblocks; ++b )
   {
      assert(hashorig2pricingvar != NULL);
      for( c = 0; c < nsubscipconss[b]; ++c )
      {
         SCIPdebugMessage("copying %s to pricing problem %d\n", SCIPconsGetName(subscipconss[b][c]), b);
         if( !SCIPconsIsActive(subscipconss[b][c]) )
         {
            SCIPdebugMessage("skipping, cons <%s> inactive\n", SCIPconsGetName(subscipconss[b][c]));
            continue;
         }
         SCIP_CALL( SCIPgetTransformedCons(scip, subscipconss[b][c], &subscipconss[b][c]) );
         assert(subscipconss[b][c] != NULL);

         /* copy the constraint */
         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "p%d_%s", b, SCIPconsGetName(subscipconss[b][c]));
         SCIP_CALL( SCIPgetConsCopy(scip, relaxdata->pricingprobs[b], subscipconss[b][c], &newcons, SCIPconsGetHdlr(subscipconss[b][c]),
               hashorig2pricingvar[b], hashorig2pricingconstmp, name,
               TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, &success) );

         /* constraint was successfully copied */
         assert(success);

         SCIP_CALL( SCIPaddCons(relaxdata->pricingprobs[b], newcons) );


#ifndef NDEBUG
         {
            SCIP_VAR** curvars;
            int ncurvars;

            ncurvars = GCGconsGetNVars(relaxdata->pricingprobs[b], newcons);
            curvars = NULL;
            if( ncurvars > 0 )
            {
               int i;

               SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );
               SCIP_CALL( GCGconsGetVars(relaxdata->pricingprobs[b], newcons, curvars, ncurvars) );

               for( i = 0; i < ncurvars; ++i )
               {
                  if( SCIPisFeasEQ( scip, SCIPvarGetLbGlobal(curvars[i]), SCIPvarGetUbGlobal(curvars[i]) ) && SCIPisFeasEQ( scip, SCIPvarGetUbGlobal(curvars[i]), 0. )  )
                     continue;

                  assert(GCGvarIsPricing(curvars[i]) || ( SCIPvarIsNegated(curvars[i]) && GCGvarIsPricing(SCIPvarGetNegatedVar(curvars[i]) ) ) );
               }

               SCIPfreeBufferArrayNull(scip, &curvars);
            }
         }
#endif
         SCIP_CALL( SCIPreleaseCons(relaxdata->pricingprobs[b], &newcons) );
      }
   }

   SCIPhashmapFree(&hashorig2pricingconstmp);

   return SCIP_OKAY;
}

/** creates the master problem and the pricing problems and copies the constraints into them */
static
SCIP_RETCODE createMaster(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAXDATA*       relaxdata           /**< the relaxator data data structure */
   )
{
   int npricingprobs;
   SCIP_HASHMAP** hashorig2pricingvar;
   SCIP_Bool enableppcuts;
   char name[SCIP_MAXSTRLEN];
   int clocktype;
   SCIP_Real infinity;
   SCIP_Real epsilon;
   SCIP_Real sumepsilon;
   SCIP_Real feastol;
   SCIP_Real lpfeastolfactor;
   SCIP_Real dualfeastol;
   int i;

   assert(scip != NULL);
   assert(relaxdata != NULL);

   assert(relaxdata->decomp != NULL);


   SCIP_CALL( convertStructToGCG(scip, relaxdata, relaxdata->decomp) );

   /* if there are no pricing problems, then the original problem will be solved directly. */
   if( relaxdata->npricingprobs == 0 )
   {
      int origmode;

      /* setting the mode to ORIGINAL */
      SCIP_CALL( SCIPunfixParam(scip, "relaxing/gcg/mode") );
      SCIP_CALL( SCIPgetIntParam(scip, "relaxing/gcg/mode", &origmode) );
      SCIP_CALL( SCIPsetIntParam(scip, "relaxing/gcg/mode", (int) DEC_DECMODE_ORIGINAL) );
      SCIP_CALL( SCIPfixParam(scip, "relaxing/gcg/mode") );

      /* if the original problem is to be solved, then we need to free the currently master problem and create a new
       * SCIP instance
       */

      if( origmode == DEC_DECMODE_DANTZIGWOLFE )
      {
         SCIP* tmpscip;

         /* initialising the master problem */
         SCIP_CALL( SCIPsetIntParam(relaxdata->altmasterprob, "display/verblevel", (int)SCIP_VERBLEVEL_NONE) );
         SCIP_CALL( SCIPsetBoolParam(relaxdata->altmasterprob, "display/relevantstats", FALSE) );

         /* disabling unnecessary display columns */
         SCIP_CALL( SCIPsetIntParam(scip, "display/sumlpiterations/active", 0) );
         SCIP_CALL( SCIPsetIntParam(scip, "display/lpiterations/active", 0) );
         SCIP_CALL( SCIPsetIntParam(scip, "display/degeneracy/active", 0) );

         /* setting the total node limit to 1 for the original SCIP instance. This is because Benders' decomposition solves
          * the MIP within the relaxator of the root node. So no branching in the original problem is required.
          */
         SCIP_CALL( SCIPsetLongintParam(scip, "limits/totalnodes", 1) );

         /* swapping the master problem with the original master problem */
         tmpscip = relaxdata->altmasterprob;
         relaxdata->altmasterprob = relaxdata->masterprob;
         relaxdata->masterprob = tmpscip;
      }

      SCIP_CALL( SCIPsetIntParam(relaxdata->masterprob, "constraints/components/maxprerounds", 0) );
      SCIP_CALL( SCIPsetBoolParam(scip, "relaxing/gcg/discretization", FALSE) );
   }


   npricingprobs = relaxdata->npricingprobs;
   hashorig2pricingvar = NULL;

   if( npricingprobs > 0 )
   {
      /* create hashmaps for mapping from original to pricing variables */
      SCIP_CALL( SCIPallocBufferArray(scip, &(hashorig2pricingvar), npricingprobs) );
   }

   SCIPdebugMessage("Creating master problem...\n");

   SCIP_CALL( initRelaxProblemdata(scip, relaxdata) );

   /* get clocktype of the original SCIP instance in order to use the same clocktype in master and pricing problems */
   SCIP_CALL( SCIPgetIntParam(scip, "timing/clocktype", &clocktype) );

   /* get numerical tolerances of the original SCIP instance in order to use the same numerical tolerances in master and pricing problems */
   SCIP_CALL( SCIPgetRealParam(scip, "numerics/infinity", &infinity) );
   SCIP_CALL( SCIPgetRealParam(scip, "numerics/epsilon", &epsilon) );
   SCIP_CALL( SCIPgetRealParam(scip, "numerics/sumepsilon", &sumepsilon) );
   SCIP_CALL( SCIPgetRealParam(scip, "numerics/feastol", &feastol) );
   SCIP_CALL( SCIPgetRealParam(scip, "numerics/lpfeastolfactor", &lpfeastolfactor) );
   SCIP_CALL( SCIPgetRealParam(scip, "numerics/dualfeastol", &dualfeastol) );

   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "master_%s", SCIPgetProbName(scip));
   SCIP_CALL( createMasterProblem(relaxdata->masterprob, name, clocktype, infinity, epsilon, sumepsilon, feastol,
         lpfeastolfactor, dualfeastol, relaxdata->mode) );

   enableppcuts = FALSE;
   SCIP_CALL( SCIPgetBoolParam(scip, "sepa/basis/enableppcuts", &enableppcuts) );

   /* create the pricing problems */
   for( i = 0; i < npricingprobs; i++ )
   {
      relaxdata->convconss[i] = NULL;
      (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "pricing_block_%d", i);
      SCIP_CALL( createPricingProblem(&(relaxdata->pricingprobs[i]), name, clocktype, infinity, epsilon, sumepsilon,
            feastol, lpfeastolfactor, dualfeastol, enableppcuts) );
      SCIP_CALL( SCIPhashmapCreate(&(hashorig2pricingvar[i]), SCIPblkmem(scip), SCIPgetNVars(scip)) ); /*lint !e613*/

      /* disabling restarts from the tree size estimation */
      SCIP_CALL( SCIPsetCharParam(relaxdata->pricingprobs[i], "estimation/restarts/restartpolicy", 'n') );
   }

   SCIP_CALL( createPricingVariables(scip, relaxdata, hashorig2pricingvar) );

   /* create master and pricing problem constraints
    * If the master problem is solved directly, then we can still call methods creating the pricing problems. These
    * methods check the number of pricing problems and number of blocks.  As such, if the original problem is solved
    * directly, then nothing will happen in these methods
    */
   SCIP_CALL( createMasterprobConss(scip, relaxdata) );
   SCIP_CALL( createPricingprobConss(scip, relaxdata, hashorig2pricingvar) );
   SCIP_CALL( GCGmasterCreateInitialMastervars(relaxdata->masterprob) );

   /* check if the master problem is a set partitioning or set covering problem */
   SCIP_CALL( checkSetppcStructure(scip, relaxdata) );

   /* check for identity of blocks */
   SCIP_CALL( checkIdenticalBlocks(scip, relaxdata, hashorig2pricingvar) );

   /* the convexity constraints are only added in the Dantzig-Wolfe mode */
   if( relaxdata->mode == DEC_DECMODE_DANTZIGWOLFE )
   {
      for( i = 0; i < relaxdata->npricingprobs; i++ )
      {
         if( relaxdata->blockrepresentative[i] != i )
            continue;

         /* create the corresponding convexity constraint */
         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "conv_block_%d", i);
         SCIP_CALL( SCIPcreateConsLinear(relaxdata->masterprob, &(relaxdata->convconss[i]), name, 0, NULL, NULL,
               relaxdata->nblocksidentical[i]*1.0, relaxdata->nblocksidentical[i]*1.0,
               TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );
         SCIP_CALL( SCIPaddCons(relaxdata->masterprob, relaxdata->convconss[i]) );
      }
   }

   /* display statistics */
   if( relaxdata->dispinfos )
   {
      SCIP_CALL( displayPricingStatistics(scip, relaxdata->pricingprobs, relaxdata->npricingprobs, relaxdata->blockrepresentative) );
      SCIP_CALL( SCIPwriteOrigProblem(relaxdata->masterprob, "masterprob.lp", "lp", FALSE) );
   }

   if( hashorig2pricingvar != NULL )
   {
      for( i = 0; i < npricingprobs; i++ )
         SCIPhashmapFree(&(hashorig2pricingvar[i]));

      SCIPfreeBufferArray(scip, &(hashorig2pricingvar));
   }

   /* get used memory and save it for reference */
   for( i = 0; i < npricingprobs; ++i )
   {
      relaxdata->pricingprobsmemused += SCIPgetMemUsed(relaxdata->pricingprobs[i])/1048576.0;
   }

   return SCIP_OKAY;
}

/** combines the solutions from all (disjoint) problems to one solution */
static
SCIP_RETCODE combineSolutions(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            newsol,             /**< pointer to store new solution */
   SCIP**                probs,              /**< array of (solved) subproblems */
   int                   nprobs              /**< number of subproblems */
   )
{
#ifdef SCIP_DEBUG
   int i;
#endif

   int v;
   int nvars;

   SCIP_VAR** vars;
   assert(scip != NULL);
   assert(newsol != NULL);
   assert(probs != NULL);
   assert(nprobs > 0);

   SCIP_CALL( SCIPcreateSol(scip, newsol, NULL) );
   nvars = SCIPgetNVars(scip);
   vars = SCIPgetVars(scip);

#ifdef SCIP_DEBUG
   for( i = 0; i < nprobs; ++i )
   {
      if( probs[i] == NULL )
         continue;

      SCIPprintOrigProblem(probs[i], NULL, "lp", FALSE);
      SCIPprintSol(probs[i], SCIPgetBestSol(probs[i]), NULL, FALSE );
   }
#endif

   for( v = 0; v < nvars; ++v )
   {
      SCIP_VAR* pricingvar;
      int block;

      pricingvar = GCGoriginalVarGetPricingVar(vars[v]);
      block = GCGvarGetBlock(pricingvar);
      assert(block >= 0);
      assert(block < nprobs);
      assert(probs[block] != NULL);

      /* @todo solval should be 0 before, anyway, check it with an assert */
      SCIP_CALL( SCIPincSolVal(scip, *newsol, vars[v], SCIPgetSolVal(probs[block], SCIPgetBestSol(probs[block]), pricingvar)) );
   }
   return SCIP_OKAY;
}

/** sets the pricing objective function to what is necessary */
static
SCIP_RETCODE setPricingObjsOriginal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP**                probs,              /**< array of subproblems */
   int                   nprobs              /**< number of subproblems */
   )
{
   int v;
   int nvars;
   SCIP_VAR** vars;
   int i;

   assert(scip != NULL);
   assert(probs != NULL);
   assert(nprobs > 0);

   nvars = SCIPgetNVars(scip);
   vars = SCIPgetVars(scip);

   /* if the Benders' decomposition is used, then the transformed problem of the subproblems must be freed.
    * This is because the within the create subproblem stage, if the subproblem is an LP, then the SCIP instance is put
    * into probing mode.
    */
   if( GCGgetDecompositionMode(scip) == DEC_DECMODE_BENDERS )
   {
      for( i = 0; i < nprobs; i++ )
      {
         /* if the problem is not in SCIP_STAGE_PROBLEM, then the transformed problem must be freed. The subproblem
          * should also be in probing mode.
          */
         if( SCIPgetStage(probs[i]) != SCIP_STAGE_PROBLEM )
         {
            if( SCIPinProbing(probs[i]) )
            {
               SCIP_CALL( SCIPendProbing(probs[i]) );
            }

            SCIP_CALL( SCIPfreeTransform(probs[i]) );
         }
      }
   }

   for( v = 0; v < nvars; ++v )
   {
      SCIP_VAR* pricingvar;
      SCIP_VAR* origvar;
      SCIP_Real objvalue;

      assert(GCGvarIsOriginal(vars[v]));
      origvar = SCIPvarGetProbvar(vars[v]);

      if( !GCGisPricingprobRelevant(scip, GCGvarGetBlock(origvar)) )
         continue;

      pricingvar = GCGoriginalVarGetPricingVar(origvar);
      assert(pricingvar != NULL);

      objvalue = SCIPvarGetObj(origvar);
      /* SCIPinfoMessage(scip, NULL, "%s: %f block %d\n", SCIPvarGetName(origvar), SCIPvarGetObj(origvar),
         GCGvarGetBlock(origvar)); */
      SCIP_CALL( SCIPchgVarObj(probs[GCGvarGetBlock(pricingvar)], pricingvar, objvalue) );
   }
   return SCIP_OKAY;
}

/** solve a block problem when the decomposition is diagonal */
static
SCIP_RETCODE solveBlockProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP*                 blockprob,          /**< the block problem that will be solved */
   SCIP_RELAXDATA*       relaxdata,          /**< relaxator data structure */
   SCIP_Real             timelimit,          /**< the original problem timelimit */
   int                   blocknum,           /**< the number of the block, -1 for the original problem */
   SCIP_RESULT*          result,             /**< result pointer to indicate success or failure */
   SCIP_Real*            objvalue            /**< the objective function value */
   )
{
   SCIP_Real blocktimelimit;

   assert(scip != NULL);
   assert(result != NULL);
   assert(objvalue != NULL);

   (*result) = SCIP_DIDNOTRUN;

#ifdef SCIP_DEBUG
   char name[SCIP_MAXSTRLEN];
#endif

   if( blockprob == NULL )
   {
      (*result) = SCIP_SUCCESS;
      return SCIP_OKAY;
   }

   if( GCGgetDecompositionMode(scip) != DEC_DECMODE_ORIGINAL )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Solving block %i.\n", blocknum+1);
   }

   SCIP_CALL( SCIPsetIntParam(blockprob, "display/verblevel", relaxdata->origverblevel) );

   /* give the pricing problem 2% more time then the original scip has left */
   if( SCIPgetStage(blockprob) > SCIP_STAGE_PROBLEM )
   {
      if( SCIPisInfinity(scip, timelimit) )
      {
         blocktimelimit = SCIPinfinity(blockprob);
      }
      else
      {
         blocktimelimit = (timelimit - SCIPgetSolvingTime(scip)) * 1.02 + SCIPgetSolvingTime(blockprob);
         blocktimelimit = MIN(SCIPinfinity(blockprob), blocktimelimit); /*lint !e666*/
      }
   }
   else
   {
      if( SCIPisInfinity(scip, timelimit) )
      {
         blocktimelimit = SCIPinfinity(blockprob);
      }
      else
      {
         blocktimelimit = (timelimit - SCIPgetSolvingTime(scip)) * 1.02;
         blocktimelimit = MIN(SCIPinfinity(blockprob), blocktimelimit); /*lint !e666*/
      }
   }

   if( blocktimelimit < 0 )
   {
      (*result) = SCIP_DIDNOTRUN;
      return SCIP_OKAY;
   }

   SCIP_CALL( SCIPsetRealParam(blockprob, "limits/time", blocktimelimit) );

#ifdef SCIP_DEBUG
   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "block_%i.lp", blocknum);
   SCIP_CALL( SCIPwriteOrigProblem(blockprob, name, "lp", FALSE) );
#endif

   if( GCGgetDecompositionMode(scip) == DEC_DECMODE_DANTZIGWOLFE || GCGgetDecompositionMode(scip) == DEC_DECMODE_ORIGINAL )
   {
      SCIP_CALL( SCIPsolve(blockprob) );
   }
   else
   {
      SCIP_BENDERS* benders;
      SCIP_Bool infeasible;

      assert(GCGgetDecompositionMode(scip) == DEC_DECMODE_BENDERS);

      /* retrieving the Benders' decomposition */
      benders = SCIPfindBenders(relaxdata->masterprob, "gcg");

      /* since the diagonal blocks are being solved, this indicates that the subproblems are independent. As such, we
       * can declare this in the Benders' decomposition framework. This allows us to call
       * SCIPsolveBendersSubproblem() without setting up the problem
       */
      SCIPbendersSetSubproblemIsIndependent(benders, blocknum, TRUE);

      /* solving the Benders' decomposition subproblem */
      SCIP_CALL( SCIPsolveBendersSubproblem(relaxdata->masterprob, benders, NULL, blocknum, &infeasible,
            TRUE, NULL) );
   }


   switch( SCIPgetStatus(blockprob) )
   {
      case SCIP_STATUS_UNBOUNDED:
      case SCIP_STATUS_INFORUNBD:
      case SCIP_STATUS_INFEASIBLE:
         /* no other blocks should be solved. */
         *result = SCIP_CUTOFF;
         break;
      case SCIP_STATUS_BESTSOLLIMIT:
      case SCIP_STATUS_MEMLIMIT:
      case SCIP_STATUS_STALLNODELIMIT:
      case SCIP_STATUS_NODELIMIT:
      case SCIP_STATUS_SOLLIMIT:
      case SCIP_STATUS_TIMELIMIT:
         /* no other blocks should be solved. */
         *result = SCIP_DIDNOTRUN;
         break;
      case SCIP_STATUS_GAPLIMIT:
      case SCIP_STATUS_OPTIMAL:
         (*result) = SCIP_SUCCESS;
         (*objvalue) += SCIPgetDualbound(blockprob);
         break;
      default:
         break;
   } /*lint !e788*/

   return SCIP_OKAY;
}

/** frees the block problem */
static
SCIP_RETCODE freeBlockProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP*                 blockprob,          /**< the block problem that will be solved */
   SCIP_RELAXDATA*       relaxdata,          /**< relaxator data structure */
   int                   blocknum            /**< the number of the block, -1 for the original problem */
   )
{
   assert(scip != NULL);

   if( blockprob == NULL )
      return SCIP_OKAY;

   if( GCGgetDecompositionMode(scip) == DEC_DECMODE_DANTZIGWOLFE || GCGgetDecompositionMode(scip) == DEC_DECMODE_ORIGINAL )
   {
      SCIP_CALL( SCIPfreeTransform(blockprob) );
   }
   else
   {
      SCIP_BENDERS* benders;

      assert(GCGgetDecompositionMode(scip) == DEC_DECMODE_BENDERS);

      /* retrieving the Benders' decomposition */
      benders = SCIPfindBenders(relaxdata->masterprob, "gcg");

      /* freeing the Benders' decomposition subproblems */
      SCIP_CALL( SCIPfreeBendersSubproblem(relaxdata->masterprob, benders, blocknum) );
   }

   return SCIP_OKAY;
}

/** solves the blocks diagonal and individually */
static
SCIP_RETCODE solveDiagonalBlocks(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAXDATA*       relaxdata,          /**< relaxator data structure */
   SCIP_RESULT*          result,             /**< result pointer to indicate success or failure */
   SCIP_Real*            lowerbound          /**< lower bound pointer to return the lower bound */
   )
{
   int i;
   SCIP_Real timelimit;
   SCIP_Real objvalue;
   SCIP_SOL *newsol;
   SCIP_Bool isfeasible;
   SCIP_RESULT solveresult;

   /* set objective of pricing problems to original objective */
   if( GCGgetDecompositionMode(scip) != DEC_DECMODE_ORIGINAL )
   {
      SCIP_CALL( setPricingObjsOriginal(scip, relaxdata->pricingprobs, relaxdata->npricingprobs) );
   }

   SCIP_CALL( SCIPgetRealParam(scip, "limits/time", &timelimit) );

   objvalue = 0.0;

   if( GCGgetDecompositionMode(scip) != DEC_DECMODE_ORIGINAL )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Block diagonal structure detected, solving blocks individually.\n");
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "There is an objective function offet of %f.\n", SCIPgetTransObjoffset(scip));
   }

   /* if the original problem is solved directly, then we call  solveBlockProblem with the master problem */
   if( GCGgetDecompositionMode(scip) == DEC_DECMODE_ORIGINAL )
   {
      SCIP_CALL( solveBlockProblem(scip, relaxdata->masterprob, relaxdata, timelimit, -1, &solveresult, &objvalue) );

      if( solveresult == SCIP_CUTOFF || solveresult == SCIP_DIDNOTRUN )
      {
         (*result) = solveresult;
         return SCIP_OKAY;
      }
   }
   else
   {
      /* solve pricing problems one after the other */
      for( i = 0; i < relaxdata->npricingprobs; ++i )
      {
         SCIP_CALL( solveBlockProblem(scip, relaxdata->pricingprobs[i], relaxdata, timelimit, i, &solveresult, &objvalue) );

         if( solveresult == SCIP_CUTOFF || solveresult == SCIP_DIDNOTRUN )
         {
            (*result) = solveresult;
            return SCIP_OKAY;
         }
      }
   }

   /* get solution and glue it together */

   if( GCGgetDecompositionMode(scip) == DEC_DECMODE_ORIGINAL )
   {
      SCIP_CALL( GCGtransformMastersolToOrigsol(scip, SCIPgetBestSol(relaxdata->masterprob), &newsol) );
   }
   else
   {
      SCIP_CALL( combineSolutions(scip, &newsol, relaxdata->pricingprobs, relaxdata->npricingprobs) );
   }

   /* update lower bound pointer and add solution such that this node will be cut off automatically */
   if( SCIPgetObjsense(scip) == SCIP_OBJSENSE_MAXIMIZE )
      *lowerbound = -objvalue;
   else
      *lowerbound = objvalue;

   SCIP_CALL( SCIPcheckSol(scip, newsol, TRUE, TRUE, TRUE, TRUE, TRUE, &isfeasible) );
   assert(isfeasible);

   SCIP_CALL( SCIPtrySolFree(scip, &newsol, FALSE, FALSE, TRUE, TRUE, TRUE, &isfeasible) );

   /** @todo maybe add a constraint here to indicate that it has been decomposed */

   /* if the original problem is solved directly, then we call freeBlockProblem with the master problem */
   if( GCGgetDecompositionMode(scip) != DEC_DECMODE_ORIGINAL )
   {
      /* solve pricing problems one after the other */
      for( i = 0; i < relaxdata->npricingprobs; ++i )
      {
         SCIP_CALL( freeBlockProblem(scip, relaxdata->pricingprobs[i], relaxdata, i) );
      }
   }

   *result = SCIP_SUCCESS;

   return SCIP_OKAY;

}


DEC_DECOMP* GCGgetStructDecomp(
   SCIP*                 scip
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->decomp;
}


/** sets the structure information */
static
SCIP_RETCODE GCGsetStructDecomp(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);
   assert(decomp != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   if( relaxdata->decomp != NULL )
      SCIP_CALL( DECdecompFree(scip, &relaxdata->decomp ) );

   relaxdata->decomp = decomp;

   return SCIP_OKAY;
}


/** transforms the master problem **/
static
SCIP_RETCODE transformMaster(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAX*           relax               /**< relaxator data structure */
   )
{
   SCIP* masterprob;
   SCIP_VAR** vars;
   SCIP_CONS** oldconss;
   SCIP_RELAXDATA* relaxdata;
   int i;
   int nvars;

   assert(scip != NULL);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   masterprob = relaxdata->masterprob;
   assert(masterprob != NULL);
   SCIP_CALL( SCIPtransformProb(masterprob) );
   SCIP_CALL( SCIPduplicateBufferArray(scip, &oldconss, relaxdata->masterconss, relaxdata->nmasterconss) );

   /* transform the master constraints */
   SCIP_CALL( SCIPtransformConss(masterprob, relaxdata->nmasterconss,
                                 relaxdata->masterconss, relaxdata->masterconss) );
   for( i = 0; i < relaxdata->nmasterconss; ++i )
   {
      SCIP_CALL( SCIPreleaseCons(masterprob, &(oldconss[i])) );
   }
   SCIPfreeBufferArray(scip, &oldconss);

   /* transform the convexity constraints */
   for( i = 0; i < relaxdata->npricingprobs; i++ )
   {
      if( relaxdata->convconss[i] != NULL )
      {
         SCIP_CONS* oldcons = relaxdata->convconss[i];
         SCIP_CALL( SCIPreleaseCons(masterprob, &oldcons) );
         SCIP_CALL( SCIPtransformCons(masterprob, relaxdata->convconss[i], &(relaxdata->convconss[i])) );
      }
   }

   nvars = SCIPgetNVars(scip);
   vars = SCIPgetVars(scip);

   /* transform the linking variable constraints */
   for( i = 0; i < nvars; ++i )
   {
      assert(GCGvarIsOriginal(vars[i]));

      if( GCGoriginalVarIsLinking(vars[i]) )
      {
         int j;
         SCIP_CONS** linkconss;
         linkconss = GCGlinkingVarGetLinkingConss(vars[i]);
         /* the linking constraints could be NULL if the Benders' decomposition is used. */
         if( linkconss != NULL )
         {
            for( j = 0; j < relaxdata->npricingprobs; ++j )
            {
               if( linkconss[j] != NULL )
               {
                  SCIP_CONS* tempcons;
                  SCIP_CALL( SCIPtransformCons(masterprob, linkconss[j], &(tempcons)) );
                  GCGlinkingVarSetLinkingCons(vars[i], tempcons, j);
               }
            }
         }
      }
   }
   for( i = 0; i < relaxdata->nvarlinkconss; ++i )
   {
      SCIP_CONS* transcons;

      SCIP_CALL( SCIPgetTransformedCons(masterprob, relaxdata->varlinkconss[i], &transcons) );
      assert(transcons != NULL);

      SCIP_CALL( SCIPreleaseCons(masterprob, &relaxdata->varlinkconss[i]) );
      relaxdata->varlinkconss[i] = transcons;
   }
   return SCIP_OKAY;
}


/** initializes and transforms relaxator data */
static
SCIP_RETCODE initRelaxator(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAX*           relax               /**< relaxator data structure */
   )
{
   SCIP_RELAXDATA* relaxdata;
   int permutationseed;
   int oxfordcomma;

   assert(scip != NULL);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   /* when the original problem should be solved directly, then a decomposition must be made with zero blocks */
   if( GCGgetDecompositionMode(scip) == DEC_DECMODE_ORIGINAL )
   {
      DEC_DECOMP* decomp;
      SCIP_RETCODE retcode;

      assert(relaxdata->decomp == NULL);

      retcode = DECcreateBasicDecomp(scip, &decomp, TRUE);
      assert(retcode == SCIP_OKAY);
      if( retcode != SCIP_OKAY )
      {
         SCIPerrorMessage("Could not add decomp to cons_decomp!\n");
         return SCIP_ERROR;
      }

      assert(decomp != NULL );

      GCGsetStructDecomp(scip, decomp);
   }

   if( relaxdata->decomp == NULL )
   {
      relaxdata->decomp = DECgetBestDecomp(scip, TRUE);
      if( relaxdata->decomp == NULL )
      {
         int partialdecid;
         SCIPwarningMessage(scip, "No complete decomposition available. Creating basic decomposition.\n");
         partialdecid = GCGconshdlrDecompAddBasicPartialdec(scip, TRUE);
         SCIP_CALL( GCGconshdlrDecompSelectPartialdec(scip, partialdecid, TRUE) );

         relaxdata->decomp = DECgetBestDecomp(scip, FALSE);
         assert( relaxdata->decomp != NULL );
      }
   }

   oxfordcomma = 0;
   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "Chosen structure has %d blocks", DECdecompGetNBlocks(relaxdata->decomp));
   /* every master-only variable internally also counts as linking, but should not be reported as linking variable */
   if ( DECdecompGetNLinkingvars(relaxdata->decomp) - DECdecompGetNMastervars(relaxdata->decomp) > 0)
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, ", %d linking variables", DECdecompGetNLinkingvars(relaxdata->decomp) - DECdecompGetNMastervars(relaxdata->decomp));
      ++oxfordcomma;
   }
   if ( DECdecompGetNMastervars(relaxdata->decomp) > 0 )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, ", %d master-only (static) variables", DECdecompGetNMastervars(relaxdata->decomp));
      ++oxfordcomma;
   }
   if ( oxfordcomma > 0 )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, ",");
   }
   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " and %d linking constraints.\n", DECdecompGetNLinkingconss(relaxdata->decomp));
   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "This decomposition has a maxwhite score of %f.\n", DECdecompGetMaxwhiteScore(relaxdata->decomp));

   /* permute the decomposition if the permutation seed is set */
   SCIP_CALL( SCIPgetIntParam(scip, "randomization/permutationseed", &permutationseed) );

   if( permutationseed > 0 )
   {
      SCIP_RANDNUMGEN* randnumgen;

      SCIP_CALL( SCIPcreateRandom(scip, &randnumgen, (unsigned int) permutationseed, TRUE) );
      SCIP_CALL( DECpermuteDecomp(scip, relaxdata->decomp, randnumgen) );
      SCIPfreeRandom(scip, &randnumgen);
   }

   if( relaxdata->discretization && (SCIPgetNContVars(scip) > 0) )
   {
      if( relaxdata->mipdiscretization )
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Warning: Discretization with continuous variables is only an experimental feature.\n");
      }
      else
      {
         SCIP_CALL( SCIPsetBoolParam(scip, "relaxing/gcg/discretization", FALSE) );
         SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Warning: Discretization with continuous variables is disabled by parameter relaxing/gcg/mipdiscretization.\n");
      }
   }

   SCIP_CALL( createMaster(scip, relaxdata) );

   /* for Benders' decomposition, the Benders' plugin must be activated */
   if( relaxdata->mode == DEC_DECMODE_BENDERS )
   {
      SCIP_CALL( SCIPactivateBenders(relaxdata->masterprob, SCIPfindBenders(relaxdata->masterprob, "gcg"),
            relaxdata->npricingprobs) );
   }

   relaxdata->lastsolvednodenr = -1;

   /* set objective limit in master problem if objective limit in original problem is finite */
   if( !SCIPisInfinity(scip, (int) SCIPgetObjsense(scip) * SCIPgetObjlimit(scip)) )
   {
      SCIP_CALL( SCIPsetObjlimit(relaxdata->masterprob, (int) SCIPgetObjsense(scip) * SCIPgetObjlimit(scip)) );
   }

   return SCIP_OKAY;
}

/** initializes relaxator data */
static
void initRelaxdata(
   SCIP_RELAXDATA*       relaxdata           /**< relaxdata data structure */
   )
{
   assert(relaxdata != NULL);

   relaxdata->decomp = NULL;

   relaxdata->blockrepresentative = NULL;
   relaxdata->convconss = NULL;
   relaxdata->hashorig2origvar = NULL;
   relaxdata->lastsolvednodenr = 0;

   relaxdata->origmasterconss = NULL;
   relaxdata->masterconss = NULL;
   relaxdata->nmasterconss = 0;

   relaxdata->npricingprobs = -1;
   relaxdata->pricingprobs = NULL;
   relaxdata->nrelpricingprobs = 0;
   relaxdata->currentorigsol = NULL;
   relaxdata->storedorigsol = NULL;
   relaxdata->origsolfeasible = FALSE;
   relaxdata->storedfeasibility = FALSE;
   relaxdata->nblocksidentical = NULL;

   relaxdata->lastmastersol = NULL;
   relaxdata->lastmasterlpiters = 0;
   relaxdata->markedmasterconss = NULL;
   relaxdata->maxmarkedmasterconss = 0;
   relaxdata->masterinprobing = FALSE;
   relaxdata->probingheur = NULL;

   relaxdata->ntransvars = 0;
   relaxdata->nlinkingvars = 0;
   relaxdata->nvarlinkconss = 0;
   relaxdata->varlinkconss = NULL;
   relaxdata->varlinkconsblock = NULL;
   relaxdata->pricingprobsmemused = 0.0;

   relaxdata->relaxisinitialized = FALSE;
   relaxdata->simplexiters = 0;
   relaxdata->rootnodetime = NULL;
}

/*
 * Callback methods of relaxator
 */

/** destructor of relaxator to free user data (called when SCIP is exiting) */
static
SCIP_DECL_RELAXFREE(relaxFreeGcg)
{
   SCIP_RELAXDATA* relaxdata;

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   /* free visualization parameters */
   if( relaxdata->paramsvisu != NULL )
   {
      GCGVisuFreeParams(scip, relaxdata->paramsvisu);
   }

   /* free master problem */
   if( relaxdata->masterprob != NULL )
   {
      SCIP_CALL( SCIPfree(&(relaxdata->masterprob)) );
   }

   /* free the alternate master problem */
   if( relaxdata->altmasterprob != NULL )
   {
      SCIP_CALL( SCIPfree(&(relaxdata->altmasterprob)) );
   }

   /* free used decomposition */
   if( relaxdata->decomp != NULL )
   {
      SCIP_CALL( DECdecompFree(scip, &relaxdata->decomp) );
   }

   SCIPfreeMemory(scip, &relaxdata);
   return SCIP_OKAY;
}

/** deinitialization method of relaxator (called before transformed problem is freed) */

static
SCIP_DECL_RELAXEXIT(relaxExitGcg)
{
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   if( relaxdata->decomp != NULL )
   {
      SCIP_CALL( DECdecompFree(scip, &relaxdata->decomp) );
      relaxdata->decomp = NULL;
   }

   /* free array for branchrules*/
   if( relaxdata->nbranchrules > 0 )
   {
      int i;

      for( i = 0; i < relaxdata->nbranchrules; i++ )
      {
         SCIPfreeMemory(scip, &(relaxdata->branchrules[i]));
      }
      SCIPfreeMemoryArray(scip, &(relaxdata->branchrules));
   }


   relaxdata->nbranchrules = 0;
   relaxdata->relaxisinitialized = FALSE;

   return SCIP_OKAY;
}


/** initialize the relaxator and master problem for solving the original problem by Dantzig-Wolfe reformulation and
 * Benders' decomposition
 */
static
SCIP_RETCODE initializeMasterProblemSolve(
   SCIP*                 scip,               /**< the SCIP data structure */
   SCIP_RELAX*           relax               /**< the relaxator */
)
{
   SCIP_RELAXDATA* relaxdata;
   SCIP_Bool cutoff;

   assert(scip != NULL);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   /* the relaxator is initialised if it has not been previously initialised */
   if( !relaxdata->relaxisinitialized )
   {
      /* set integral objective status in the extended problem, if possible */
      if( SCIPisObjIntegral(scip) && relaxdata->discretization && SCIPgetNContVars(scip) == 0
          && relaxdata->mode == DEC_DECMODE_DANTZIGWOLFE )
      {
         SCIP_CALL( SCIPsetObjIntegral(relaxdata->masterprob) );
      }
      SCIP_CALL( transformMaster(scip, relax) );
      /* transform the decomposition */
      // SCIP_CALL( DECdecompTransform(scip, relaxdata->decomp) );
      SCIP_CALL( GCGconsOrigbranchAddRootCons(scip) );
      relaxdata->relaxisinitialized = TRUE;
      assert(relaxdata->decomp != NULL);
   }

   if( !SCIPisLPConstructed(scip) ) {
      /* construct the LP in the original problem */
      SCIP_CALL(SCIPconstructLP(scip, &cutoff));
      assert(!cutoff);
      SCIP_CALL(SCIPflushLP(scip));
   }

   return SCIP_OKAY;
}


/** solving process initialization method of relaxator (called when branch and bound process is about to begin) */
static
SCIP_DECL_RELAXINITSOL(relaxInitsolGcg)
{
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);
   assert(relaxdata->masterprob != NULL);

   initRelaxdata(relaxdata);
   SCIP_CALL( SCIPcreateClock(scip, &(relaxdata->rootnodetime)) );

   /* if the master problem decomposition mode is the same as the original SCIP instance mode, then the master problem
    * must be swapped with the alternate master problem.
    */
   if( GCGgetMasterDecompMode(relaxdata->masterprob) != DEC_DECMODE_ORIGINAL &&
      GCGgetMasterDecompMode(relaxdata->masterprob) != GCGgetDecompositionMode(scip) )
   {
      SCIP* tmpscip;

      tmpscip = relaxdata->masterprob;
      relaxdata->masterprob = relaxdata->altmasterprob;
      relaxdata->altmasterprob = tmpscip;
   }

   /* alternative verbosity levels are used for the Benders' decomposition and original mode compared to the Dantzig-Wolfe
    * decomposition mode.
    */
   if( GCGgetDecompositionMode(scip) == DEC_DECMODE_BENDERS || GCGgetDecompositionMode(scip) == DEC_DECMODE_ORIGINAL )
   {
      /* first getting the verbosity level for the original problem before setting it to none. While the verbosity level
       * was collected previously, the user may have changed this in the mean time.
       */
      SCIP_CALL( SCIPgetIntParam(scip, "display/verblevel", &relaxdata->origverblevel) );

      /* deactivating display columns */
      SCIP_CALL( SCIPsetIntParam(scip, "display/sumlpiterations/active", 0) );
      SCIP_CALL( SCIPsetIntParam(scip, "display/lpiterations/active", 0) );
      SCIP_CALL( SCIPsetIntParam(scip, "display/degeneracy/active", 0) );

      /* setting the total node limit to 1 for the original SCIP instance. This is because Benders' decomposition solves
       * the MIP within the relaxator of the root node. So no branching in the original problem is required.
       */
      SCIP_CALL( SCIPsetLongintParam(scip, "limits/totalnodes", 1LL) );
   }

   /* fixing the GCG mode parameter. This ensure that the user does not change this during the solution process. If the
    * mode parameter were to change, the behaviour is unknown.
    */
   SCIP_CALL( SCIPfixParam(scip, "relaxing/gcg/mode") );

   /* Informing the user of the decomposition technique that is being used to solve the original problem */
   SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "\n");
   if( relaxdata->mode == DEC_DECMODE_DANTZIGWOLFE )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "A Dantzig-Wolfe reformulation is applied to solve the original problem.\n");
   }
   else if( relaxdata->mode == DEC_DECMODE_BENDERS )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "A Benders' decomposition is applied to solve the original problem.\n");
   }
   else if( relaxdata->mode == DEC_DECMODE_ORIGINAL )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "No reformulation will be performed. Solving the original model.\n");
   }

   if( !SCIPisStopped(scip) )
      SCIP_CALL( initRelaxator(scip, relax) );

   return SCIP_OKAY;
}


/** solving process deinitialization method of relaxator (called before branch and bound process data is freed) */
static
SCIP_DECL_RELAXEXITSOL(relaxExitsolGcg)
{
   SCIP_RELAXDATA* relaxdata;
   int i;

   assert(scip != NULL);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   if( relaxdata->hashorig2origvar != NULL )
   {
      SCIPhashmapFree(&(relaxdata->hashorig2origvar));
      relaxdata->hashorig2origvar = NULL;
   }

   SCIPfreeBlockMemoryArrayNull(scip, &(relaxdata->markedmasterconss), relaxdata->maxmarkedmasterconss);
   relaxdata->markedmasterconss = NULL;
   relaxdata->maxmarkedmasterconss = 0;

   /* free arrays for constraints */
   for( i = 0; i < relaxdata->nmasterconss; i++ )
   {
      SCIP_CALL( SCIPreleaseCons(scip, &relaxdata->origmasterconss[i]) );
      SCIP_CALL( SCIPreleaseCons(relaxdata->masterprob, &relaxdata->masterconss[i]) );
   }
   for( i = 0; i < relaxdata->npricingprobs; i++ )
   {
      if( relaxdata->convconss[i] != NULL )
         SCIP_CALL( SCIPreleaseCons(relaxdata->masterprob, &relaxdata->convconss[i]) );
   }
   for( i = 0; i < relaxdata->nvarlinkconss; i++ )
   {
      SCIP_CALL( SCIPreleaseCons(relaxdata->masterprob, &relaxdata->varlinkconss[i]) );
   }
   SCIPfreeBlockMemoryArrayNull(scip, &(relaxdata->varlinkconss), relaxdata->nvarlinkconss);
   SCIPfreeBlockMemoryArrayNull(scip, &(relaxdata->varlinkconsblock), relaxdata->nvarlinkconss);
   SCIPfreeBlockMemoryArrayNull(scip, &(relaxdata->origmasterconss), relaxdata->maxmasterconss);
   SCIPfreeBlockMemoryArrayNull(scip, &(relaxdata->masterconss), relaxdata->maxmasterconss);
   SCIPfreeBlockMemoryArrayNull(scip, &(relaxdata->convconss), relaxdata->npricingprobs);

   /* free master problem */
   if( relaxdata->masterprob != NULL )
   {
      SCIP_CALL( SCIPfreeProb(relaxdata->masterprob) );
   }

   /* free pricing problems */
   for( i = relaxdata->npricingprobs - 1; i >= 0 ; i-- )
   {
      SCIP_CALL( SCIPfree(&(relaxdata->pricingprobs[i])) );
   }
   SCIPfreeBlockMemoryArrayNull(scip, &(relaxdata->pricingprobs), relaxdata->npricingprobs);
   SCIPfreeBlockMemoryArrayNull(scip, &(relaxdata->blockrepresentative), relaxdata->npricingprobs);
   SCIPfreeBlockMemoryArrayNull(scip, &(relaxdata->nblocksidentical), relaxdata->npricingprobs);

   /* free solutions */
   if( relaxdata->currentorigsol != NULL )
   {
      SCIP_CALL( SCIPfreeSol(scip, &relaxdata->currentorigsol) );
   }
   if( relaxdata->storedorigsol != NULL )
   {
      SCIP_CALL( SCIPfreeSol(scip, &relaxdata->storedorigsol) );
   }

   if( relaxdata->decomp != NULL )
   {
      SCIP_CALL( DECdecompFree(scip, &relaxdata->decomp) );
      relaxdata->decomp = NULL;
   }

   SCIP_CALL( GCGfreeOrigVarsData(scip) );

   /* free root node clock */
   if( relaxdata->rootnodetime != NULL )
   {
      SCIP_CALL( SCIPfreeClock(scip, &(relaxdata->rootnodetime)) );
   }

   relaxdata->relaxisinitialized = FALSE;

   return SCIP_OKAY;
}


/** method to solve the master problem that is used by Dantzig-Wolfe and Benders' decomposition */
static
SCIP_RETCODE solveMasterProblem(
   SCIP*                 scip,               /**< the SCIP data structure */
   SCIP*                 masterprob,         /**< the master problem SCIP instance */
   SCIP_RELAXDATA*       relaxdata,          /**< the relaxator data */
   SCIP_Longint          nodelimit,          /**< the number of nodes the will be solved in this master problem */
   SCIP_Real*            lowerbound,         /**< the lowerbound computed by the relaxator for the current node */
   SCIP_RESULT*          result              /**< the result of the relaxation call */
   )
{
   SCIP_Real timelimit;
   SCIP_Real memorylimit;

   assert(scip != NULL);
   assert(masterprob != NULL);
   assert(relaxdata != NULL);

   /* update the number of the last solved node */
   relaxdata->lastsolvednodenr = SCIPnodeGetNumber(SCIPgetCurrentNode(scip));

   /* increase the node limit for the master problem by 1 */
   SCIP_CALL( SCIPsetLongintParam(masterprob, "limits/nodes", nodelimit) );


   /* loop to solve the master problem, this is a workaround and does not fix any problem */
   while( !SCIPisStopped(scip) )
   {
      SCIP_Real mastertimelimit = SCIPinfinity(scip);

      /* set memorylimit for master */
      SCIP_CALL( SCIPgetRealParam(scip, "limits/memory", &memorylimit) );
      if( !SCIPisInfinity(scip, memorylimit) )
         memorylimit -= SCIPgetMemUsed(scip)/1048576.0;

      SCIP_CALL( SCIPsetRealParam(masterprob, "limits/memory", memorylimit) );

      SCIP_CALL( SCIPgetRealParam(scip, "limits/time", &timelimit) );
      if( !SCIPisInfinity(scip, timelimit) )
      {

         /* give the master 2% more time then the original scip has left */
         mastertimelimit = (timelimit - SCIPgetSolvingTime(scip)) * 1.02 + SCIPgetSolvingTime(masterprob);
         SCIP_CALL( SCIPsetRealParam(masterprob, "limits/time", mastertimelimit) );

         SCIPdebugMessage("  time limit for master: %f, left: %f, left for original problem: %f\n",
               mastertimelimit,
               mastertimelimit - SCIPgetSolvingTime(masterprob),
               timelimit - SCIPgetSolvingTime(scip));
      }

      /* if we have a blockdetection, see whether the node is block diagonal. Additionally, the solveDiagonalBlocks can
       * be called when the original problem is solved directly.
       */
      if( DECdecompGetType(relaxdata->decomp) == DEC_DECTYPE_DIAGONAL || GCGgetDecompositionMode(scip) == DEC_DECMODE_ORIGINAL )
      {
         SCIP_CALL( solveDiagonalBlocks(scip, relaxdata, result, lowerbound) );
         if( *result == SCIP_SUCCESS || *result == SCIP_CUTOFF )
         {
            *result = SCIP_CUTOFF;
            return SCIP_OKAY;
         }
      }
      /* We are solving the masterproblem regularly */
      else
      {
         SCIP_CALL( SCIPsolve(masterprob) );
      }


      if( SCIPgetStatus(masterprob) != SCIP_STATUS_TIMELIMIT )
      {
         break;
      }

      if( !SCIPisInfinity(scip, timelimit) && !SCIPisStopped(scip) )
         SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "time for master problem was too short, extending time by %f.\n", mastertimelimit - SCIPgetSolvingTime(masterprob));
   }
   if( SCIPgetStatus(masterprob) == SCIP_STATUS_TIMELIMIT && SCIPisStopped(scip) )
   {
      *result = SCIP_DIDNOTRUN;
      return SCIP_OKAY;
   }

   /* set the lower bound pointer */
   if( SCIPgetStage(masterprob) == SCIP_STAGE_SOLVING && GCGmasterIsCurrentSolValid(masterprob) )
   {
      *lowerbound = SCIPgetLocalDualbound(masterprob);
   }
   else
   {
      SCIPdebugMessage("  stage: %d\n", SCIPgetStage(masterprob));
      assert(SCIPgetStatus(masterprob) == SCIP_STATUS_TIMELIMIT || SCIPgetBestSol(masterprob) != NULL || SCIPgetStatus(masterprob) == SCIP_STATUS_INFEASIBLE || SCIPgetStatus(masterprob) == SCIP_STATUS_UNKNOWN);
      if( SCIPgetStatus(masterprob) == SCIP_STATUS_OPTIMAL && GCGmasterIsCurrentSolValid(masterprob) )
         *lowerbound = SCIPgetSolOrigObj(masterprob, SCIPgetBestSol(masterprob));
      else if( SCIPgetStatus(masterprob) == SCIP_STATUS_INFEASIBLE || SCIPgetStatus(masterprob) == SCIP_STATUS_TIMELIMIT || !GCGmasterIsCurrentSolValid(masterprob) )
      {
         SCIP_Real tilim;
         SCIP_CALL( SCIPgetRealParam(masterprob, "limits/time", &tilim) );
         if( tilim-SCIPgetSolvingTime(masterprob) < 0 )
         {
            *result = SCIP_DIDNOTRUN;
            return SCIP_OKAY;
         }
         *lowerbound = SCIPinfinity(scip);
      }
      else if( SCIPgetStatus(masterprob) == SCIP_STATUS_UNKNOWN )
      {
         *result = SCIP_DIDNOTRUN;
         return SCIP_OKAY;
      }
      else
      {
         SCIPwarningMessage(scip, "Stage <%d> is not handled!\n", SCIPgetStage(masterprob));
         *result = SCIP_DIDNOTRUN;
         return SCIP_OKAY;
      }
   }

   SCIPdebugMessage("  update lower bound (value = %g).\n", *lowerbound);

   /* NOTE: All other points when result is set, the function is exited immediately. Ensure that this is checked for
    * future changes to this function
    */
   *result = SCIP_SUCCESS;

   return SCIP_OKAY;
}


/** execution method of the relaxator for Dantzig-Wolfe reformulation */
static
SCIP_RETCODE relaxExecGcgDantzigWolfe(
   SCIP*                 scip,               /**< the SCIP data structure */
   SCIP_RELAX*           relax,              /**< the relaxator */
   SCIP_Real*            lowerbound,         /**< the lowerbound computed by the relaxator for the current node */
   SCIP_RESULT*          result              /**< the result of the relaxation call */
   )
{
   SCIP* masterprob;
   SCIP_RELAXDATA* relaxdata;
   SCIP_Longint oldnnodes;
   SCIP_Longint nodelimit;
   SCIP_Bool stored;

   assert(scip != NULL);
   assert(relax != NULL);
   assert(result != NULL);
   assert(GCGgetDecompositionMode(scip) == DEC_DECMODE_DANTZIGWOLFE);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);
   *result = SCIP_DIDNOTRUN;

   masterprob = relaxdata->masterprob;
   assert(masterprob != NULL);

   /* solve the next node in the master problem */
   SCIPdebugMessage("Solving node %"SCIP_LONGINT_FORMAT"'s relaxation.\n", SCIPnodeGetNumber(SCIPgetCurrentNode(scip)));

   /* only solve the relaxation if it was not yet solved at the current node */
   if( SCIPnodeGetNumber(SCIPgetCurrentNode(scip)) != relaxdata->lastsolvednodenr )
   {
      /* start root node time clock */
      if( SCIPgetRootNode(scip) == SCIPgetCurrentNode(scip) )
      {
         SCIP_CALL( SCIPstartClock(scip, relaxdata->rootnodetime) );
         SCIPdebugMessage("  root node time clock started.\n");
      }

      /* increase the node limit for the master problem by 1 */
      SCIP_CALL( SCIPgetLongintParam(masterprob, "limits/nodes", &oldnnodes) );

      nodelimit = (SCIPgetRootNode(scip) == SCIPgetCurrentNode(scip) ? 1 : oldnnodes + 1);
      /* solving the master problem */
      SCIP_CALL( solveMasterProblem(scip, masterprob, relaxdata, nodelimit, lowerbound, result) );

      if( relaxdata->currentorigsol != NULL )
      {
         SCIP_CALL( SCIPtrySol(scip, relaxdata->currentorigsol, FALSE, FALSE, TRUE, TRUE, TRUE, &stored) );
      }

      /* if a new primal solution was found in the master problem, transfer it to the original problem */
      if( SCIPgetBestSol(relaxdata->masterprob) != NULL && relaxdata->lastmastersol != SCIPgetBestSol(relaxdata->masterprob) && GCGmasterIsCurrentSolValid(masterprob) )
      {
         SCIP_SOL* newsol;

         relaxdata->lastmastersol = SCIPgetBestSol(relaxdata->masterprob);

         SCIP_CALL( GCGtransformMastersolToOrigsol(scip, relaxdata->lastmastersol, &newsol) );
   #ifdef SCIP_DEBUG
         SCIP_CALL( SCIPtrySol(scip, newsol, TRUE, TRUE, TRUE, TRUE, TRUE, &stored) );
   #else
         SCIP_CALL( SCIPtrySol(scip, newsol, FALSE, FALSE, TRUE, TRUE, TRUE, &stored) );
   #endif
         /* only check failed solution if best master solution is valid */
         if( !stored && GCGmasterIsBestsolValid(relaxdata->masterprob) )
         {
            SCIP_CALL( SCIPcheckSolOrig(scip, newsol, &stored, TRUE, TRUE) );
         }
         /** @bug The solution doesn't have to be accepted, numerics might bite us, so the transformation might fail.
          *  A remedy could be: Round the values or propagate changes or call a heuristic to fix it.
          */
         SCIP_CALL( SCIPfreeSol(scip, &newsol) );

         if( stored )
            SCIPdebugMessage("  updated current best primal feasible solution.\n");
      }

      if( GCGconsOrigbranchGetBranchrule(GCGconsOrigbranchGetActiveCons(scip)) != NULL )
      {
         SCIP_CALL( GCGrelaxBranchMasterSolved(scip, GCGconsOrigbranchGetBranchrule(GCGconsOrigbranchGetActiveCons(scip) ),
               GCGconsOrigbranchGetBranchdata(GCGconsOrigbranchGetActiveCons(scip)), *lowerbound) );
      }

      /* stop root node clock */
      if( SCIPgetRootNode(scip) == SCIPgetCurrentNode(scip))
      {
         SCIP_CALL( SCIPstopClock(scip, relaxdata->rootnodetime) );
         SCIPdebugMessage("  root node time clock stopped at %6.2fs.\n", SCIPgetClockTime(scip, relaxdata->rootnodetime));
      }
   }
   else
   {
      SCIPdebugMessage("Problem has been already solved at this node\n");
   }

   return SCIP_OKAY;
}


/** method to solve the master problem for Benders' decomposition and when solving the original problem directly. */
static
SCIP_RETCODE solveMasterProblemAndEvaluate(
   SCIP*                 scip,               /**< the SCIP data structure */
   SCIP_RELAX*           relax,              /**< the relaxator */
   SCIP_Real*            lowerbound,         /**< the lowerbound computed by the relaxator for the current node */
   SCIP_RESULT*          result              /**< the result of the relaxation call */
   )
{
   SCIP* masterprob;
   SCIP_RELAXDATA* relaxdata;
   SCIP_Longint nodelimit;

   assert(scip != NULL);
   assert(relax != NULL);
   assert(result != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);
   *result = SCIP_DIDNOTRUN;

   masterprob = relaxdata->masterprob;
   assert(masterprob != NULL);

   /* solve the next node in the master problem */
   SCIPdebugMessage("Solving node %"SCIP_LONGINT_FORMAT"'s relaxation.\n", SCIPnodeGetNumber(SCIPgetCurrentNode(scip)));

   /* prior to performing the decomposition the original problem verbosity is changed to NONE. This avoids output from
    * the original problem before the decomposition output. Once the decomposition has been performed, then the
    * verbosity level of the original problem is returned to the original verbosity level.
    */
   SCIP_CALL( SCIPsetIntParam(scip, "display/verblevel", relaxdata->origverblevel) );
   SCIP_CALL( SCIPsetIntParam(masterprob, "display/verblevel", relaxdata->origverblevel) );

   /* getting the node limit from the original problem. This is because the master problem is solved to optimality in
    * the execution of the relaxator.
    */
   SCIP_CALL( SCIPgetLongintParam(scip, "limits/nodes", &nodelimit) );

   /* solving the master problem */
   SCIP_CALL( solveMasterProblem(scip, masterprob, relaxdata, nodelimit, lowerbound, result) );

   /* if the master problem has been detected as infeasible, then the result must be set to SCIP_CUTOFF. */
   if( SCIPgetStatus(masterprob) == SCIP_STATUS_INFEASIBLE )
      (*result) = SCIP_CUTOFF;

   /* if the master problem has been solved to optimality, the we cutoff the root node. This informs that original
    * problem that no further processing is required.
    */
   if( SCIPgetStatus(masterprob) == SCIP_STATUS_OPTIMAL )
   {
      (*result) = SCIP_CUTOFF;
   }

   /* if there is no primal solution for the original problem, then the master solution is transferred */
   if( SCIPgetBestSol(relaxdata->masterprob) != NULL && relaxdata->lastmastersol != SCIPgetBestSol(relaxdata->masterprob) )
   {
      SCIP_SOL* newsol;
      SCIP_Bool stored;

      relaxdata->lastmastersol = SCIPgetBestSol(relaxdata->masterprob);

      SCIP_CALL( GCGtransformMastersolToOrigsol(scip, SCIPgetBestSol(relaxdata->masterprob), &newsol) );
#ifdef SCIP_DEBUG
      SCIP_CALL( SCIPtrySol(scip, newsol, TRUE, TRUE, TRUE, TRUE, TRUE, &stored) );
#else
      SCIP_CALL( SCIPtrySol(scip, newsol, FALSE, FALSE, TRUE, TRUE, TRUE, &stored) );
#endif
      /* only check failed solution if best master solution is valid */
      if( !stored && GCGmasterIsBestsolValid(relaxdata->masterprob) )
      {
         SCIP_CALL( SCIPcheckSolOrig(scip, newsol, &stored, TRUE, TRUE) );
      }
      /** @bug The solution doesn't have to be accepted, numerics might bite us, so the transformation might fail.
       *  A remedy could be: Round the values or propagate changes or call a heuristic to fix it.
       */
      SCIP_CALL( SCIPfreeSol(scip, &newsol) );

      if( stored )
         SCIPdebugMessage("  updated current best primal feasible solution.\n");
   }

   /* set the lower bound pointer */
   if( GCGmasterIsCurrentSolValid(masterprob)
      && (SCIPgetStage(masterprob) == SCIP_STAGE_SOLVED || SCIPgetStage(masterprob) == SCIP_STAGE_SOLVING) )
   {
      *lowerbound = SCIPgetDualbound(masterprob);
   }

   /* if the time, memory or node limit is hit in the Original or Benders mode, then we need to interrupt the solve.
    * This is required because the original problem is not solved in either of these modes, so it is not certain that
    * the original SCIP will also exceed the limit (definitely not for the node limit).
    */
   if( SCIPgetStatus(masterprob) == SCIP_STATUS_TIMELIMIT || SCIPgetStatus(masterprob) == SCIP_STATUS_NODELIMIT
      || SCIPgetStatus(masterprob) == SCIP_STATUS_MEMLIMIT )
   {
      SCIP_CALL( SCIPinterruptSolve(scip) );
   }

   /* if the result pointer is DIDNOTRUN, this implies that the master problem was interrupted during solving. Since
    * Benders' decomposition uses a one-tree approach, then the user limits must be adhered to. This means, the if a
    * limit is exceeded, this is still a success for the solving.
    */
   if( (*result) == SCIP_DIDNOTRUN )
      (*result) = SCIP_SUCCESS;

   return SCIP_OKAY;
}

/** execution method of the relaxator for Benders' decomposition */
static
SCIP_RETCODE relaxExecGcgBendersDecomposition(
   SCIP*                 scip,               /**< the SCIP data structure */
   SCIP_RELAX*           relax,              /**< the relaxator */
   SCIP_Real*            lowerbound,         /**< the lowerbound computed by the relaxator for the current node */
   SCIP_RESULT*          result              /**< the result of the relaxation call */
   )
{
   assert(scip != NULL);
   assert(relax != NULL);
   assert(result != NULL);

   SCIP_CALL( solveMasterProblemAndEvaluate(scip, relax, lowerbound, result) );

   return SCIP_OKAY;
}

/** execution method of the relaxator when the original problem is solved directly */
static
SCIP_RETCODE relaxExecGcgOriginalProblem(
   SCIP*                 scip,               /**< the SCIP data structure */
   SCIP_RELAX*           relax,              /**< the relaxator */
   SCIP_Real*            lowerbound,         /**< the lowerbound computed by the relaxator for the current node */
   SCIP_RESULT*          result              /**< the result of the relaxation call */
   )
{
   assert(scip != NULL);
   assert(relax != NULL);
   assert(result != NULL);

   SCIP_CALL( solveMasterProblemAndEvaluate(scip, relax, lowerbound, result) );

   return SCIP_OKAY;
}


/** execution method of relaxator */
static
SCIP_DECL_RELAXEXEC(relaxExecGcg)
{
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);
   assert(relax != NULL);
   assert(result != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   /* checking whether the relaxator needs to be initialised. If so, then the master problem and pricing problems will
    * be created.
    */
   SCIP_CALL( initializeMasterProblemSolve(scip, relax) );

   /* selecting the solving algorithm based upon the decomposition mode selected by the user, or whether the original
    * problem should be solved directly
    */
   if( GCGgetDecompositionMode(scip) == DEC_DECMODE_ORIGINAL )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "There are no pricing problems in the decomposition. The original problem will be solved directly.\n");
      SCIP_CALL( relaxExecGcgOriginalProblem(scip, relax, lowerbound, result) );
   }
   else if( relaxdata->mode == DEC_DECMODE_DANTZIGWOLFE )
   {
      SCIP_CALL( relaxExecGcgDantzigWolfe(scip, relax, lowerbound, result) );
   }
   else if( relaxdata->mode == DEC_DECMODE_BENDERS )
   {
      SCIP_CALL( relaxExecGcgBendersDecomposition(scip, relax, lowerbound, result) );
   }
   else
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_DIALOG, NULL, "Sorry, the automatic selection is not currently available\n");
   }

   return SCIP_OKAY;
}

#define relaxCopyGcg NULL
#define relaxInitGcg NULL


/*
 * relaxator specific interface methods
 */

/** creates the GCG relaxator and includes it in SCIP */
SCIP_RETCODE SCIPincludeRelaxGcg(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAXDATA* relaxdata;

#ifdef WITH_BLISS
   {
      char name[SCIP_MAXSTRLEN];
      GCGgetBlissName(name, SCIP_MAXSTRLEN);
      SCIP_CALL( SCIPincludeExternalCodeInformation(scip, name, "A Tool for Computing Automorphism Groups of Graphs by T. Junttila and P. Kaski (http://www.tcs.hut.fi/Software/bliss/)") );
   }
#endif

#ifdef WITH_CLIQUER
      SCIP_CALL( SCIPincludeExternalCodeInformation(scip, "Cliquer", "A set of C routines for finding cliques in an arbitrary weighted graph by S. Niskanen and P. Ostergard (https://users.aalto.fi/~pat/cliquer.html)") );
#endif

   /* create GCG relaxator data */
   SCIP_CALL( SCIPallocMemory(scip, &relaxdata) );

   relaxdata->decomp = NULL;
   relaxdata->nbranchrules = 0;
   relaxdata->branchrules = NULL;
   relaxdata->masterprob = NULL;
   relaxdata->altmasterprob = NULL;
   relaxdata->paramsvisu = NULL;
   SCIPcreateParamsVisu(scip, &(relaxdata->paramsvisu));
   assert(relaxdata->paramsvisu != NULL);

   initRelaxdata(relaxdata);

   /* include relaxator */
   SCIP_CALL( SCIPincludeRelax(scip, RELAX_NAME, RELAX_DESC, RELAX_PRIORITY, RELAX_FREQ, relaxCopyGcg, relaxFreeGcg, relaxInitGcg,
         relaxExitGcg, relaxInitsolGcg, relaxExitsolGcg, relaxExecGcg, relaxdata) );

   /* inform the main scip, that no LPs should be solved */
   SCIP_CALL( SCIPsetIntParam(scip, "lp/solvefreq", 0) );

   /* Disable restarts */
   SCIP_CALL( SCIPsetIntParam(scip, "presolving/maxrestarts", 0) );
   SCIP_CALL( SCIPsetBoolParam(scip, "misc/calcintegral", FALSE) );

   /* initialize the scip data structure for the master problem. The master problem is initialized as the Dantzig-Wolfe
    * master problem. The alternate master problem is initialized as the Benders' decomposition master problem.
    */
   SCIP_CALL( SCIPcreate(&(relaxdata->masterprob)) );
   SCIP_CALL( SCIPincludePricerGcg(relaxdata->masterprob, scip) );
   SCIP_CALL( GCGincludeMasterPlugins(relaxdata->masterprob) );
   SCIP_CALL( SCIPsetMessagehdlr(relaxdata->masterprob, SCIPgetMessagehdlr(scip)) );

   /* getting the verbosity level of the original problem */
   SCIP_CALL( SCIPgetIntParam(scip, "display/verblevel", &relaxdata->origverblevel) );

   /* disable display output in the master problem */
   SCIP_CALL( SCIPsetIntParam(relaxdata->masterprob, "display/verblevel", (int)SCIP_VERBLEVEL_NONE) );

   /* set parameters in master problem */
   SCIP_CALL( SCIPsetIntParam(relaxdata->masterprob, "pricing/maxvars", INT_MAX) );
   SCIP_CALL( SCIPsetIntParam(relaxdata->masterprob, "pricing/maxvarsroot", INT_MAX) );
   SCIP_CALL( SCIPsetRealParam(relaxdata->masterprob, "pricing/abortfac", 1.0) );
   SCIP_CALL( SCIPsetIntParam(relaxdata->masterprob, "lp/disablecutoff", 1) );
#ifdef DELVARS
   /* set paramteters to allow deletion of variables */
   SCIP_CALL( SCIPsetBoolParam(relaxdata->masterprob, "pricing/delvars", TRUE) );
   SCIP_CALL( SCIPsetBoolParam(relaxdata->masterprob, "pricing/delvarsroot", TRUE) );
   SCIP_CALL( SCIPsetBoolParam(relaxdata->masterprob, "lp/cleanupcols", TRUE) );
   SCIP_CALL( SCIPsetBoolParam(relaxdata->masterprob, "lp/cleanupcolsroot", TRUE) );
#endif

   /* initializing the alternate master problem. The alternate master problem is initially the Benders' decomposition
    * master problem
    */
   SCIP_CALL( SCIPcreate(&(relaxdata->altmasterprob)) );
   SCIP_CALL( SCIPincludeBendersGcg(relaxdata->altmasterprob, scip) );
   SCIP_CALL( GCGincludeBendersPlugins(relaxdata->altmasterprob) );
   SCIP_CALL( SCIPsetMessagehdlr(relaxdata->altmasterprob, SCIPgetMessagehdlr(scip)) );

   SCIP_CALL( SCIPsetIntParam(relaxdata->altmasterprob, "display/verblevel", (int)SCIP_VERBLEVEL_NONE) );
   SCIP_CALL( SCIPsetBoolParam(relaxdata->altmasterprob, "display/relevantstats", FALSE) );

   /* add GCG relaxator parameters */
   SCIP_CALL( SCIPaddBoolParam(scip, "relaxing/gcg/discretization",
         "should discretization (TRUE) or convexification (FALSE) approach be used?",
         &(relaxdata->discretization), FALSE, DEFAULT_DISCRETIZATION, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip, "relaxing/gcg/mipdiscretization",
         "should discretization (TRUE) or convexification (FALSE) approach be used in mixed-integer programs?",
         &(relaxdata->mipdiscretization), FALSE, DEFAULT_MIPDISCRETIZATION, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip, "relaxing/gcg/aggregation",
         "should identical blocks be aggregated (only for discretization approach)?",
         &(relaxdata->aggregation), FALSE, DEFAULT_AGGREGATION, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip, "relaxing/gcg/dispinfos",
         "should additional information about the blocks be displayed?",
         &(relaxdata->dispinfos), FALSE, DEFAULT_DISPINFOS, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(scip, "relaxing/gcg/mode",
            "the decomposition mode that GCG will use. (0: Dantzig-Wolfe (default), 1: Benders' decomposition, "
            "2: no decomposition will be performed)",
            (int*)&(relaxdata->mode), FALSE, (int)DEFAULT_MODE, 0, 2, NULL, NULL) );
#ifdef WITH_BLISS
   SCIP_CALL( SCIPaddBoolParam(scip, "relaxing/gcg/bliss/enabled",
         "should bliss be used to check for identical blocks?",
         &(relaxdata->usebliss), FALSE, DEFAULT_BLISS, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(scip, "relaxing/gcg/bliss/searchnodelimit",
         "bliss search node limit (0: unlimited), requires patched bliss version",
         &(relaxdata->searchnodelimit), TRUE, (int)DEFAULT_BLISS_SEARCH_NODE_LIMIT, 0, INT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(scip, "relaxing/gcg/bliss/generatorlimit",
         "bliss generator limit (0: unlimited), requires patched bliss version or version >= 0.76",
         &(relaxdata->generatorlimit), TRUE, (int)DEFAULT_BLISS_GENERATOR_LIMIT, 0, INT_MAX, NULL, NULL) );
#else
   relaxdata->usebliss = FALSE;
   relaxdata->searchnodelimit = 0;
   relaxdata->generatorlimit = 0;
#endif

   return SCIP_OKAY;
}


/*
 * relaxator specific interface methods for coordination of branching rules
 */

/** includes a branching rule into the relaxator data */
SCIP_RETCODE GCGrelaxIncludeBranchrule(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule for which callback methods are saved */
   GCG_DECL_BRANCHACTIVEMASTER((*branchactivemaster)),/**<  activation method for branchrule */
   GCG_DECL_BRANCHDEACTIVEMASTER((*branchdeactivemaster)),/**<  deactivation method for branchrule */
   GCG_DECL_BRANCHPROPMASTER((*branchpropmaster)),/**<  propagation method for branchrule */
   GCG_DECL_BRANCHMASTERSOLVED((*branchmastersolved)),/**<  master solved method for branchrule */
   GCG_DECL_BRANCHDATADELETE((*branchdatadelete))/**<  branchdata deletion method for branchrule */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   int pos;

   assert(scip != NULL);
   assert(branchrule != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   SCIP_CALL( ensureSizeBranchrules(scip, relaxdata) );

   pos = relaxdata->nbranchrules;

   /* store callback functions */
   SCIP_CALL( SCIPallocMemory(scip, &(relaxdata->branchrules[pos])) ); /*lint !e866*/
   relaxdata->branchrules[pos]->branchrule = branchrule;
   relaxdata->branchrules[pos]->branchactivemaster = branchactivemaster;
   relaxdata->branchrules[pos]->branchdeactivemaster = branchdeactivemaster;
   relaxdata->branchrules[pos]->branchpropmaster = branchpropmaster;
   relaxdata->branchrules[pos]->branchmastersolved = branchmastersolved;
   relaxdata->branchrules[pos]->branchdatadelete = branchdatadelete;
   relaxdata->nbranchrules++;

   return SCIP_OKAY;
}

/** perform activation method of the given branchrule for the given branchdata */
SCIP_RETCODE GCGrelaxBranchActiveMaster(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule that did the branching */
   GCG_BRANCHDATA*       branchdata          /**< data representing the branching decision */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   int i;

   assert(scip != NULL);
   assert(branchrule != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   /* search for the branching rule in the branchrules array */
   for( i = 0; i < relaxdata->nbranchrules; i++ )
   {
      if( branchrule == relaxdata->branchrules[i]->branchrule )
      {
         /* call activation method of branching rule */
         if( relaxdata->branchrules[i]->branchactivemaster != NULL )
            SCIP_CALL( relaxdata->branchrules[i]->branchactivemaster(relaxdata->masterprob, branchdata) );

         break;
      }
   }

   assert(i < relaxdata->nbranchrules);

   return SCIP_OKAY;
}

/** perform deactivation method of the given branchrule for the given branchdata */
SCIP_RETCODE GCGrelaxBranchDeactiveMaster(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule that did the branching */
   GCG_BRANCHDATA*       branchdata          /**< data representing the branching decision */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   int i;

   assert(scip != NULL);
   assert(branchrule != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   /* search for the branching rule in the branchrules array */
   for( i = 0; i < relaxdata->nbranchrules; i++ )
   {
      if( branchrule == relaxdata->branchrules[i]->branchrule )
      {
         /* call deactivation method of branching rule */
         if( relaxdata->branchrules[i]->branchdeactivemaster != NULL )
            SCIP_CALL( relaxdata->branchrules[i]->branchdeactivemaster(relaxdata->masterprob, branchdata) );

         break;
      }
   }

   assert(i < relaxdata->nbranchrules);

   return SCIP_OKAY;
}

/** perform propagation method of the given branchrule for the given branchdata */
SCIP_RETCODE GCGrelaxBranchPropMaster(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule that did the branching */
   GCG_BRANCHDATA*       branchdata,         /**< data representing the branching decision */
   SCIP_RESULT*          result              /**< pointer to store the result of the propagation call */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   int i;

   assert(scip != NULL);
   assert(branchrule != NULL);
   assert(result != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   *result = SCIP_DIDNOTRUN;

   /* search for the branching rule in the branchrules array */
   for( i = 0; i < relaxdata->nbranchrules; i++ )
   {
      if( branchrule == relaxdata->branchrules[i]->branchrule )
      {
         /* call propagation method of branching rule*/
         if( relaxdata->branchrules[i]->branchpropmaster != NULL )
            SCIP_CALL( relaxdata->branchrules[i]->branchpropmaster(relaxdata->masterprob, branchdata, result) );

         break;
      }
   }

   assert(i < relaxdata->nbranchrules);

   return SCIP_OKAY;
}

/** frees branching data created by the given branchrule */
SCIP_RETCODE GCGrelaxBranchDataDelete(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule that did the branching */
   GCG_BRANCHDATA**      branchdata          /**< data representing the branching decision */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   int i;

   assert(scip != NULL);
   assert(branchrule != NULL);
   assert(branchdata != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   /* search for the branching rule in the branchrules array */
   for( i = 0; i < relaxdata->nbranchrules; i++ )
   {
      if( branchrule == relaxdata->branchrules[i]->branchrule )
      {
         /* call branchrule data deletion method of the branching rule */
         if( relaxdata->branchrules[i]->branchdatadelete != NULL )
            SCIP_CALL( relaxdata->branchrules[i]->branchdatadelete(scip, branchdata) );
         else
         {
            if( *branchdata != NULL )
            {
               SCIPfreeMemory(GCGgetMasterprob(scip), branchdata);
            }
         }
         break;
      }
   }

   assert(i < relaxdata->nbranchrules);

   return SCIP_OKAY;
}

/** perform method of the given branchrule that is called after the master LP is solved */
SCIP_RETCODE GCGrelaxBranchMasterSolved(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule that did the branching */
   GCG_BRANCHDATA*       branchdata,         /**< data representing the branching decision */
   SCIP_Real             newlowerbound       /**< the new local lowerbound */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   int i;

   assert(scip != NULL);
   assert(branchrule != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   /* search for the branching rule in the branchrules array */
   for( i = 0; i < relaxdata->nbranchrules; i++ )
   {
      if( branchrule == relaxdata->branchrules[i]->branchrule )
      {
         /* call master problem solved method of the branching rule */
         if( relaxdata->branchrules[i]->branchmastersolved != NULL )
            SCIP_CALL( relaxdata->branchrules[i]->branchmastersolved(scip, branchdata, newlowerbound) );

         break;
      }
   }

   assert(i < relaxdata->nbranchrules);

   return SCIP_OKAY;
}

/** transforms a constraint of the original problem into the master variable space
 *  and stores information about the constraints in the variable */
SCIP_RETCODE GCGrelaxTransOrigToMasterCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< the constraint that should be transformed */
   SCIP_CONS**           transcons           /**< pointer to store the transformed constraint */
   )
{

   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   SCIP_CONS* mastercons;
   char name[SCIP_MAXSTRLEN];

   SCIP_VAR** mastervars;
   int nmastervars;
   SCIP_VAR** consvars;
   SCIP_Real* consvals;
   int nconsvars;
   int v;
   int i;
   int j;

   assert(scip != NULL);
   assert(cons != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   /* create and add corresponding linear constraint in the master problem */
   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "m_%s", SCIPconsGetName(cons));
   SCIP_CALL( SCIPcreateConsLinear(relaxdata->masterprob, &mastercons, name, 0, NULL, NULL,
         GCGconsGetLhs(scip, cons), GCGconsGetRhs(scip, cons),
         TRUE, TRUE, TRUE, TRUE, TRUE, SCIPconsIsLocal(cons), TRUE, FALSE, FALSE,
         SCIPconsIsStickingAtNode(cons)) );

   /* now compute coefficients of the master variables in the master constraint */
   mastervars = SCIPgetVars(relaxdata->masterprob);
   nmastervars = SCIPgetNVars(relaxdata->masterprob);

   consvars = SCIPgetVarsLinear(scip, cons);
   nconsvars = SCIPgetNVarsLinear(scip, cons);
   consvals = SCIPgetValsLinear(scip, cons);


   /* add coefs of the original variables in the constraint to their variable data */
   for( v = 0; v < nconsvars; v++ )
   {
      SCIP_CALL( GCGoriginalVarAddCoef(scip, consvars[v], consvals[v], mastercons) );
   }

   /* add master variables to the corresponding master constraint */
   for( v = 0; v < nmastervars; v++ )
   {
      SCIP_VAR** origvars;
      SCIP_Real* origvals;
      int norigvars;
      SCIP_Real coef = 0.0;

      origvars = GCGmasterVarGetOrigvars(mastervars[v]);
      norigvars = GCGmasterVarGetNOrigvars(mastervars[v]);
      origvals = GCGmasterVarGetOrigvals(mastervars[v]);

      for( i = 0; i < norigvars; i++ )
         for( j = 0; j < nconsvars; j++ )
            if( consvars[j] == origvars[i] )
               coef += consvals[j] * origvals[i];

      if( !SCIPisFeasZero(scip, coef) )
      {
         SCIP_CALL( SCIPaddCoefLinear(relaxdata->masterprob, mastercons, mastervars[v], coef) );
      }
   }

   /* store the constraints in the arrays origmasterconss and masterconss in the problem data */
   SCIP_CALL( ensureSizeMasterConss(scip, relaxdata, relaxdata->nmasterconss+1) );
   SCIP_CALL( SCIPcaptureCons(scip, cons) );
   relaxdata->origmasterconss[relaxdata->nmasterconss] = cons;
   relaxdata->masterconss[relaxdata->nmasterconss] = mastercons;

   SCIP_CALL( GCGmasterAddMasterconsToHashmap(relaxdata->masterprob, relaxdata->masterconss[relaxdata->nmasterconss],
         relaxdata->nmasterconss) );

   relaxdata->nmasterconss++;

   *transcons = mastercons;

   return SCIP_OKAY;
}

/** returns the original problem for the given master problem */
SCIP* GCGgetOriginalprob(
   SCIP*                 masterprob          /**< the SCIP data structure for the master problem */
   )
{
   SCIP* origprob;
   SCIP_BENDERS* benders;
   SCIP_PRICER* pricer;

   assert(masterprob != NULL);

   /* retrieving the Benders' decomposition and the pricer plugins. There should only be one or the other for a given
    * master problem. If there are both, then an error is returned */
   benders = SCIPfindBenders(masterprob, "gcg");
   pricer = SCIPfindPricer(masterprob, "gcg");
   assert((benders != NULL && pricer == NULL) || (pricer != NULL && benders == NULL));

   origprob = NULL;
   if( benders != NULL && pricer == NULL )
   {
      origprob = GCGbendersGetOrigprob(masterprob);
      assert(GCGgetDecompositionMode(origprob) == DEC_DECMODE_BENDERS
         || GCGgetDecompositionMode(origprob) == DEC_DECMODE_ORIGINAL);
   }
   else if( pricer != NULL && benders == NULL )
   {
      origprob = GCGmasterGetOrigprob(masterprob);
      assert(GCGgetDecompositionMode(origprob) == DEC_DECMODE_DANTZIGWOLFE);
   }
   else
   {
      SCIPerrorMessage("There must exist either a pricer or a benders or neither, not both.\n");
   }

   return origprob;
}

/** returns the master problem */
SCIP* GCGgetMasterprob(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->masterprob;
}

/** returns the pricing problem of the given number */
SCIP* GCGgetPricingprob(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   pricingprobnr       /**< number of the pricing problem */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->pricingprobs[pricingprobnr];
}

/** returns the number of relevant pricing problems */
int GCGgetNRelPricingprobs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   assert(relaxdata->nrelpricingprobs >= -1);
   return relaxdata->nrelpricingprobs;
}

/** returns the number of pricing problems */
int GCGgetNPricingprobs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   assert(relaxdata->npricingprobs >= -1);
   return relaxdata->npricingprobs;
}

/** returns TRUE iff the pricing problem of the given number is relevant, that means is not identical to
 *  another and represented by it */
SCIP_Bool GCGisPricingprobRelevant(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   pricingprobnr       /**< number of the pricing problem */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return (relaxdata->blockrepresentative[pricingprobnr] == pricingprobnr);

}

/**
 *  for a given block, return the block by which it is represented
 */
int GCGgetBlockRepresentative(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   pricingprobnr       /**< number of the pricing problem */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   if( pricingprobnr == -1 )
      return -1;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   assert(pricingprobnr >= 0);
   assert(pricingprobnr < relaxdata->npricingprobs);
   assert(relaxdata->nblocksidentical[pricingprobnr] >= 0);
   assert((relaxdata->blockrepresentative[pricingprobnr] == pricingprobnr)
      == (relaxdata->nblocksidentical[pricingprobnr] > 0));

   return relaxdata->blockrepresentative[pricingprobnr];
}

/** returns the number of blocks in the original formulation, that are represented by
 *  the pricingprob with the given number */
int GCGgetNIdenticalBlocks(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   pricingprobnr       /**< number of the pricing problem */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);
   assert(pricingprobnr >= 0);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);
   assert(pricingprobnr <= relaxdata->npricingprobs);
   assert(relaxdata->nblocksidentical[pricingprobnr] >= 0);
   assert((relaxdata->blockrepresentative[pricingprobnr] == pricingprobnr)
      == (relaxdata->nblocksidentical[pricingprobnr] > 0));

   return relaxdata->nblocksidentical[pricingprobnr];

}

/** returns the number of constraints in the master problem */
int GCGgetNMasterConss(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->nmasterconss;
}

/** returns the contraints in the master problem */
SCIP_CONS** GCGgetMasterConss(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->masterconss;
}

/** returns the linking constraints in the original problem that correspond to the constraints in the master problem */
SCIP_CONS** GCGgetOrigMasterConss(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->origmasterconss;
}

/** returns the convexity constraint for the given block */
SCIP_CONS* GCGgetConvCons(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   blocknr             /**< the number of the block for which we
                                              *   need the convexity constraint */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);
   assert(blocknr >= 0);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);
   assert(blocknr < relaxdata->npricingprobs);

   return relaxdata->convconss[blocknr];
}

/** returns the visualization parameters */
GCG_PARAMDATA* GCGgetParamsVisu(
   SCIP*                 scip               /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   GCG_PARAMDATA* paramdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);
   assert(relaxdata->paramsvisu != NULL);

   paramdata = relaxdata->paramsvisu;
   assert(paramdata != NULL);

   return paramdata;
}

/** returns the current solution for the original problem */
SCIP_SOL* GCGrelaxGetCurrentOrigSol(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->currentorigsol;
}

/** returns whether the current solution is primal feasible in the original problem */
SCIP_Bool GCGrelaxIsOrigSolFeasible(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->origsolfeasible;
}

/** returns whether the master problem is a set covering problem */
SCIP_Bool GCGisMasterSetCovering(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->masterissetcover;
}

/** returns whether the master problem is a set partitioning problem */
SCIP_Bool GCGisMasterSetPartitioning(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->masterissetpart;
}

/** start probing mode on both the original and master problems
 *
 *  @note This mode is intended for working on the original variables but using the master LP;
 *        it currently only supports bound changes on the original variables,
 *        but no additional rows
 */
SCIP_RETCODE GCGrelaxStartProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR*            probingheur         /**< heuristic that started probing mode, or NULL */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   SCIP* masterprob;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   if( relaxdata->masterinprobing )
   {
      SCIPerrorMessage("already in GCG probing mode\n");
      return SCIP_INVALIDCALL;
   }

   masterprob = relaxdata->masterprob;
   assert(masterprob != NULL);

   /* start probing in both the original and the master problem */
   SCIP_CALL( SCIPstartProbing(scip) );
   SCIP_CALL( SCIPstartProbing(masterprob) );

   relaxdata->masterinprobing = TRUE;
   relaxdata->probingheur = probingheur;

   /* remember the current original solution */
   assert(relaxdata->storedorigsol == NULL);
   if( relaxdata->currentorigsol != NULL )
   {
      SCIP_CALL( SCIPcreateSolCopy(scip, &relaxdata->storedorigsol, relaxdata->currentorigsol) );
      relaxdata->storedfeasibility = relaxdata->origsolfeasible;
   }

   return SCIP_OKAY;
}

/** returns the  heuristic that started probing in the master problem, or NULL */
SCIP_HEUR* GCGrelaxGetProbingheur(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->probingheur;
}

/** add a new probing node the original problem together with an original branching constraint
 *
 *  @note A corresponding probing node must be added to the master problem right before solving the probing LP
 */
SCIP_RETCODE GCGrelaxNewProbingnodeOrig(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   SCIP_CONS* probingcons;
   SCIP_NODE* probingnode;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   if( !relaxdata->masterinprobing )
   {
      SCIPerrorMessage("not in GCG probing mode\n");
      return SCIP_INVALIDCALL;
   }

   if( SCIPgetProbingDepth(scip) != SCIPgetProbingDepth(GCGgetMasterprob(scip)) )
   {
      SCIPerrorMessage("original and master problem not at same probing depth\n");
      return SCIP_INVALIDCALL;
   }

   /* add a probing node in the original problem together with an original branching constraint */
   SCIP_CALL( SCIPnewProbingNode(scip) );
   probingnode = SCIPgetCurrentNode(scip);
   SCIP_CALL( GCGcreateConsOrigbranch(scip, &probingcons, "probingcons", probingnode,
      GCGconsOrigbranchGetActiveCons(scip), NULL, NULL) );
   SCIP_CALL( SCIPaddConsNode(scip, probingnode, probingcons, NULL) );
   SCIP_CALL( SCIPreleaseCons(scip, &probingcons) );


   return SCIP_OKAY;
}

/** add a new probing node the master problem together with a master branching constraint
 *  which ensures that bound changes are transferred to master and pricing problems
 *
 *  @note A corresponding probing node must have been added to the original problem beforehand;
 *        furthermore, this method must be called after bound changes to the original problem have been made
 */
SCIP_RETCODE GCGrelaxNewProbingnodeMaster(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   SCIP* masterprob;
   SCIP_CONS* probingcons;
   SCIP_NODE* probingnode;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   if( !relaxdata->masterinprobing )
   {
      SCIPerrorMessage("not in GCG probing mode\n");
      return SCIP_INVALIDCALL;
   }

   masterprob = relaxdata->masterprob;
   assert(masterprob != NULL);

   if( SCIPgetProbingDepth(scip) != SCIPgetProbingDepth(masterprob) + 1 )
   {
      SCIPerrorMessage("master probing node must be created after original probing node\n");
      return SCIP_INVALIDCALL;
   }

   /* add a probing node in the master problem together with a master branching constraint */
   SCIP_CALL( SCIPnewProbingNode(masterprob) );
   probingnode = SCIPgetCurrentNode(masterprob);
   assert(GCGconsMasterbranchGetActiveCons(masterprob) != NULL);
   SCIP_CALL( GCGcreateConsMasterbranch(masterprob, &probingcons, "mprobingcons", probingnode,
      GCGconsMasterbranchGetActiveCons(masterprob), NULL, NULL, NULL, 0, 0) );
   SCIP_CALL( SCIPaddConsNode(masterprob, probingnode, probingcons, NULL) );
   SCIP_CALL( SCIPreleaseCons(masterprob, &probingcons) );

   return SCIP_OKAY;
}

/** add a new probing node the master problem together with a master branching constraint
 *  which ensures that bound changes are transferred to master and pricing problems as well as additional
 *  constraints
 *
 *  @note A corresponding probing node must have been added to the original problem beforehand;
 *        furthermore, this method must be called after bound changes to the original problem have been made
 */
SCIP_RETCODE GCGrelaxNewProbingnodeMasterCons(
   SCIP*                 scip,                /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< pointer to the branching rule */
   GCG_BRANCHDATA*       branchdata,         /**< branching data */
   SCIP_CONS**           origbranchconss,    /**< original constraints enforcing the branching decision */
   int                   norigbranchconss,   /**< number of original constraints */
   int                   maxorigbranchconss  /**< capacity of origbranchconss */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   SCIP* masterprob;
   SCIP_CONS* probingcons;
   SCIP_NODE* probingnode;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   if( !relaxdata->masterinprobing )
   {
      SCIPerrorMessage("not in GCG probing mode\n");
      return SCIP_INVALIDCALL;
   }

   masterprob = relaxdata->masterprob;
   assert(masterprob != NULL);

   if( SCIPgetProbingDepth(scip) != SCIPgetProbingDepth(masterprob) + 1 )
   {
      SCIPerrorMessage("master probing node must be created after original probing node\n");
      return SCIP_INVALIDCALL;
   }

   /* add a probing node in the master problem together with a master branching constraint */
   SCIP_CALL( SCIPnewProbingNode(masterprob) );
   probingnode = SCIPgetCurrentNode(masterprob);
   assert(GCGconsMasterbranchGetActiveCons(masterprob) != NULL);
   SCIP_CALL( GCGcreateConsMasterbranch(masterprob, &probingcons, "mprobingcons", probingnode,
      GCGconsMasterbranchGetActiveCons(masterprob), branchrule, branchdata, origbranchconss, norigbranchconss, maxorigbranchconss) );
   SCIP_CALL( SCIPaddConsNode(masterprob, probingnode, probingcons, NULL) );
   SCIP_CALL( SCIPreleaseCons(masterprob, &probingcons) );

   return SCIP_OKAY;
}

/** add probing nodes to both the original and master problem;
 *  furthermore, add origbranch and masterbranch constraints to transfer branching decisions
 *  from the original to the master problem
 */
SCIP_RETCODE GCGrelaxBacktrackProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   probingdepth        /**< probing depth of the node in the probing path that should be reactivated */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   SCIP* masterprob;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   if( !relaxdata->masterinprobing )
   {
      SCIPerrorMessage("not in GCG probing mode\n");
      return SCIP_INVALIDCALL;
   }

   masterprob = relaxdata->masterprob;
   assert(masterprob != NULL);

   SCIP_CALL( SCIPbacktrackProbing(scip, probingdepth) );
   SCIP_CALL( SCIPbacktrackProbing(masterprob, probingdepth) );

   return SCIP_OKAY;
}

/** solve the master probing LP with or without pricing */
static
SCIP_RETCODE performProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   maxlpiterations,    /**< maximum number of lp iterations allowed */
   int                   maxpricerounds,     /**< maximum number of pricing rounds allowed */
   SCIP_Bool             usepricing,         /**< should the LP be solved with or without pricing? */
   SCIP_Longint*         nlpiterations,      /**< pointer to store the number of performed LP iterations (or NULL) */
   int*                  npricerounds,       /**< pointer to store the number of performed pricing rounds (or NULL) */
   SCIP_Real*            lpobjvalue,         /**< pointer to store the lp obj value if lp was solved */
   SCIP_Bool*            lpsolved,           /**< pointer to store whether the lp was solved */
   SCIP_Bool*            lperror,            /**< pointer to store whether an unresolved LP error occured or the
                                              *   solving process should be stopped (e.g., due to a time limit) */
   SCIP_Bool*            cutoff              /**< pointer to store whether the probing direction is infeasible */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   SCIP* masterprob;
   SCIP_LPSOLSTAT lpsolstat;
   SCIP_Longint oldnlpiters;
   int oldpricerounds;
   SCIP_Longint nodelimit;

   assert(scip != NULL);

   /* get the relaxator */
   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   /* get the relaxator data */
   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   if( !relaxdata->masterinprobing )
   {
      SCIPerrorMessage("not in GCG probing mode\n");
      return SCIP_INVALIDCALL;
   }

   /* get master problem */
   masterprob = relaxdata->masterprob;
   assert(masterprob != NULL);

   /* increase node limit for the master problem by 1 */
   SCIP_CALL( SCIPgetLongintParam(masterprob, "limits/nodes", &nodelimit) );
   SCIP_CALL( SCIPsetLongintParam(masterprob, "limits/nodes", nodelimit + 1) );

   /* propagate probing bound changes to the master problem */
   SCIP_CALL( SCIPpropagateProbing(masterprob, -1, cutoff, NULL) );
   assert(!(*cutoff));

   /* remember LP iterations and pricing rounds before LP solving */
   oldnlpiters = SCIPgetNLPIterations(masterprob);
   oldpricerounds = SCIPgetNPriceRounds(masterprob);

   *lpobjvalue = 0.0;
   *lpsolved = FALSE;

   /* solve the probing LP */
   if( usepricing )
   {
      /* LP iterations are unlimited when probing LP is solved with pricing */
      assert(maxlpiterations == -1);
      SCIP_CALL( SCIPsolveProbingLPWithPricing(masterprob, FALSE, TRUE, maxpricerounds, lperror, NULL) );
   }
   else
   {
      assert(maxpricerounds == 0);
      SCIP_CALL( SCIPsolveProbingLP(masterprob, maxlpiterations, lperror, NULL) );
   }
   lpsolstat = SCIPgetLPSolstat(masterprob);

   /* reset the node limit */
   SCIP_CALL( SCIPsetLongintParam(masterprob, "limits/nodes", nodelimit) );

   /* calculate number of LP iterations and pricing rounds performed */
   if( nlpiterations != NULL )
      *nlpiterations = SCIPgetNLPIterations(masterprob) - oldnlpiters;
   if( npricerounds != NULL )
      *npricerounds = SCIPgetNPriceRounds(masterprob) - oldpricerounds;

   if( !(*lperror) )
   {
      /* get LP solution status, objective value */
      *cutoff = *cutoff || (lpsolstat == SCIP_LPSOLSTAT_OBJLIMIT || lpsolstat == SCIP_LPSOLSTAT_INFEASIBLE);
      if( lpsolstat == SCIP_LPSOLSTAT_OPTIMAL )
      {
         SCIPdebugMessage("lpobjval = %g\n", SCIPgetLPObjval(masterprob));
         *lpobjvalue = SCIPgetLPObjval(masterprob);
         *lpsolved = TRUE;
         SCIP_CALL( GCGrelaxUpdateCurrentSol(scip) );
      }
   }
   else
   {
      SCIPdebugMessage("something went wrong, an lp error occurred\n");
   }

   return SCIP_OKAY;
}


/** solve the master probing LP without pricing */
SCIP_RETCODE GCGrelaxPerformProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   maxlpiterations,    /**< maximum number of lp iterations allowed */
   SCIP_Longint*         nlpiterations,      /**< pointer to store the number of performed LP iterations (or NULL) */
   SCIP_Real*            lpobjvalue,         /**< pointer to store the lp obj value if lp was solved */
   SCIP_Bool*            lpsolved,           /**< pointer to store whether the lp was solved */
   SCIP_Bool*            lperror,            /**< pointer to store whether an unresolved LP error occured or the
                                              *   solving process should be stopped (e.g., due to a time limit) */
   SCIP_Bool*            cutoff              /**< pointer to store whether the probing direction is infeasible */
   )
{
   SCIP_CALL( performProbing(scip, maxlpiterations, 0, FALSE, nlpiterations,
         NULL, lpobjvalue, lpsolved, lperror, cutoff) );

   return SCIP_OKAY;
}


/** solve the master probing LP with pricing */
SCIP_RETCODE GCGrelaxPerformProbingWithPricing(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   maxpricerounds,     /**< maximum number of pricing rounds allowed */
   SCIP_Longint*         nlpiterations,      /**< pointer to store the number of performed LP iterations (or NULL) */
   int*                  npricerounds,       /**< pointer to store the number of performed pricing rounds (or NULL) */
   SCIP_Real*            lpobjvalue,         /**< pointer to store the lp obj value if lp was solved */
   SCIP_Bool*            lpsolved,           /**< pointer to store whether the lp was solved */
   SCIP_Bool*            lperror,            /**< pointer to store whether an unresolved LP error occured or the
                                              *   solving process should be stopped (e.g., due to a time limit) */
   SCIP_Bool*            cutoff              /**< pointer to store whether the probing direction is infeasible */
   )
{
   SCIP_CALL( performProbing(scip, -1, maxpricerounds, TRUE, nlpiterations,
         npricerounds, lpobjvalue, lpsolved, lperror, cutoff) );

   return SCIP_OKAY;
}

/** end probing mode in both the original and master problems */
SCIP_RETCODE GCGrelaxEndProbing(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   SCIP* masterprob;

   SCIP_VAR** vars;
   int nvars;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   if( !relaxdata->masterinprobing )
   {
      SCIPerrorMessage("not in GCG probing mode\n");
      return SCIP_INVALIDCALL;
   }

   masterprob = relaxdata->masterprob;
   assert(masterprob != NULL);

   SCIP_CALL( SCIPgetVarsData(scip, &vars, &nvars, NULL, NULL, NULL, NULL) );
   assert(vars != NULL);
   assert(nvars >= 0);

   SCIP_CALL( SCIPendProbing(masterprob) );
   SCIP_CALL( SCIPendProbing(scip) );

   relaxdata->masterinprobing = FALSE;
   relaxdata->probingheur = NULL;

   /* if a new primal solution was found in the master problem, transfer it to the original problem
    * @todo: this is probably not necessary anymore since it is done by an event handler
    */
   if( SCIPgetBestSol(masterprob) != NULL && relaxdata->lastmastersol != SCIPgetBestSol(masterprob) )
   {
      SCIP_SOL* newsol;
      SCIP_Bool stored;

      relaxdata->lastmastersol = SCIPgetBestSol(masterprob);

      SCIP_CALL( GCGtransformMastersolToOrigsol(scip, relaxdata->lastmastersol, &newsol) );

      SCIP_CALL( SCIPtrySol(scip, newsol, FALSE, FALSE, TRUE, TRUE, TRUE, &stored) );
      if( !stored )
      {
         SCIP_CALL( SCIPcheckSolOrig(scip, newsol, &stored, TRUE, TRUE) );
      }
      assert(stored);
      SCIP_CALL( SCIPfreeSol(scip, &newsol) );

      SCIPdebugMessage("probing finished in master problem\n");
   }

   /* restore old relaxation solution and branching candidates */
   if( relaxdata->currentorigsol != NULL )
   {
      SCIPdebugMessage("Freeing previous solution origsol\n");
      SCIP_CALL( SCIPfreeSol(scip, &(relaxdata->currentorigsol)) );
   }
   SCIPclearExternBranchCands(scip);

   if( relaxdata->storedorigsol != NULL )
   {
      int i;

      SCIP_CALL( SCIPcreateSol(scip, &relaxdata->currentorigsol, NULL) );
      SCIP_CALL( SCIPsetRelaxSolValsSol(scip, relax, relaxdata->storedorigsol, RELAX_INCLUDESLP) );

      for( i = 0; i < nvars; i++ )
      {
         SCIP_VAR* var;
         SCIP_Real solval;

         var = vars[i];
         solval = SCIPgetSolVal(scip, relaxdata->storedorigsol, var);

         SCIP_CALL( SCIPsetSolVal(scip, relaxdata->currentorigsol, var, solval) );

         if( SCIPvarGetType(var) <= SCIP_VARTYPE_INTEGER && !SCIPisFeasIntegral(scip, solval) )
         {
            assert(!SCIPisEQ(scip, SCIPvarGetLbLocal(var), SCIPvarGetUbLocal(var)));
            SCIP_CALL( SCIPaddExternBranchCand(scip, var, solval - SCIPfloor(scip, solval), solval) );
         }
      }
      assert(SCIPisFeasEQ(scip, SCIPgetRelaxSolObj(scip), SCIPgetSolTransObj(scip, relaxdata->currentorigsol)));

      SCIP_CALL( SCIPfreeSol(scip, &relaxdata->storedorigsol) );

      relaxdata->origsolfeasible = relaxdata->storedfeasibility;
   }

   /** @todo solve master problem again */

   return SCIP_OKAY;
}


/** checks whether a variable shoudl be added as an external branching candidate, if so it is added */
static
SCIP_RETCODE checkAndAddExternalBranchingCandidate(
   SCIP*                 scip,               /**< the SCIP data structure */
   SCIP_VAR*             var                 /**< the variable to check whether to add as a branching candidate */
   )
{
   assert(scip != NULL);
   assert(var != NULL);

   if( SCIPvarGetType(var) <= SCIP_VARTYPE_INTEGER && !SCIPisFeasIntegral(scip, SCIPgetRelaxSolVal(scip, var)) )
   {
      if( SCIPisEQ(scip, SCIPvarGetLbLocal(var), SCIPvarGetUbLocal(var)) )
      {
         SCIPdebugMessage("lblocal = %g, ublocal = %g\n", SCIPvarGetLbLocal(var), SCIPvarGetUbLocal(var));
         SCIPdebugMessage("var = %s, vartype = %d, val = %g\n", SCIPvarGetName(var), SCIPvarGetType(var),
            SCIPgetRelaxSolVal(scip, var));
      }

      assert(!SCIPisEQ(scip, SCIPvarGetLbLocal(var), SCIPvarGetUbLocal(var)));

      SCIP_CALL( SCIPaddExternBranchCand(scip, var, SCIPgetRelaxSolVal(scip, var) -
            SCIPfloor(scip, SCIPgetRelaxSolVal(scip, var)), SCIPgetRelaxSolVal(scip, var)) );
   }

   return SCIP_OKAY;
}



/** transforms the current solution of the master problem into the original problem's space
 *  and saves this solution as currentsol in the relaxator's data
 */
SCIP_RETCODE GCGrelaxUpdateCurrentSol(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   SCIP_VAR** origvars;
   int norigvars;
   SCIP_Bool stored;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   origvars = SCIPgetVars(scip);
   norigvars = SCIPgetNVars(scip);
   assert(origvars != NULL);

   relaxdata->origsolfeasible = FALSE;

   /* if the master problem has not been solved, don't try to update the solution */
   if( SCIPgetStage(relaxdata->masterprob) == SCIP_STAGE_TRANSFORMED )
      return SCIP_OKAY;

   /* free previous solution and clear branching candidates */
   if( relaxdata->currentorigsol != NULL )
   {
      SCIPdebugMessage("Freeing previous solution origsol\n");
      SCIP_CALL( SCIPfreeSol(scip, &(relaxdata->currentorigsol)) );
   }
   SCIPclearExternBranchCands(scip);

   if( SCIPgetStage(relaxdata->masterprob) == SCIP_STAGE_SOLVED || SCIPgetLPSolstat(relaxdata->masterprob) == SCIP_LPSOLSTAT_OPTIMAL )
   {
      SCIP_SOL* mastersol;

      relaxdata->lastmasterlpiters = SCIPgetNLPIterations(relaxdata->masterprob);

      /* create new solution */
      if( SCIPgetStage(relaxdata->masterprob) == SCIP_STAGE_SOLVING )
      {
         SCIPdebugMessage("Masterproblem still solving, mastersol = NULL\n");
         mastersol = NULL;
      }
      else if( SCIPgetStage(relaxdata->masterprob) == SCIP_STAGE_SOLVED )
      {
         mastersol = SCIPgetBestSol(relaxdata->masterprob);
         if( mastersol == NULL )
         {
            SCIPdebugMessage("Masterproblem solved, no master sol present\n");
            return SCIP_OKAY;
         }
         SCIPdebugMessage("Masterproblem solved, mastersol = %p\n", (void*) mastersol);
      }
      else
      {
         SCIPdebugMessage("stage in master not solving and not solved!\n");
         return SCIP_OKAY;
      }

      if( !SCIPisInfinity(scip, SCIPgetSolOrigObj(relaxdata->masterprob, mastersol)) && GCGmasterIsSolValid(relaxdata->masterprob, mastersol) )
      {
         int i;
         int j;

         /* transform the master solution to the original variable space */
         SCIP_CALL( GCGtransformMastersolToOrigsol(scip, mastersol, &(relaxdata->currentorigsol)) );

         /* store the solution as relaxation solution */
         SCIP_CALL( SCIPsetRelaxSolValsSol(scip, relax, relaxdata->currentorigsol, RELAX_INCLUDESLP) );
         assert(SCIPisEQ(scip, SCIPgetRelaxSolObj(scip), SCIPgetSolTransObj(scip, relaxdata->currentorigsol)));

         if( GCGgetDecompositionMode(scip) == DEC_DECMODE_BENDERS )
            SCIP_CALL( SCIPtrySol(scip, relaxdata->currentorigsol, FALSE, FALSE, TRUE, TRUE, TRUE, &stored) );
         else
            SCIP_CALL( SCIPcheckSolOrig(scip, relaxdata->currentorigsol, &stored, FALSE, TRUE) );

         SCIPdebugMessage("updated current original LP solution, %s feasible in the original problem!\n",
            (stored ? "" : "not"));

         if( stored )
            relaxdata->origsolfeasible = TRUE;

         /* in the case of Benders decomposition, only the master variables can be added as branching candidates */
         if( GCGgetDecompositionMode(scip) == DEC_DECMODE_BENDERS )
         {
            SCIP* masterprob;
            SCIP_VAR** mastervars;
            SCIP_VAR** masterorigvars;
            int nmastervars;
            int nmasterorigvars;

            /* retrieving the master problem */
            masterprob = GCGgetMasterprob(scip);

            /* get variables of the master problem and their solution values */
            SCIP_CALL( SCIPgetVarsData(masterprob, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );

            /* looping over all master variables to get the original variable for branching candidates */
            for( i = 0; i < nmastervars; i++ )
            {
               masterorigvars = GCGmasterVarGetOrigvars(mastervars[i]);
               nmasterorigvars = GCGmasterVarGetNOrigvars(mastervars[i]);

               for( j = 0; j < nmasterorigvars; j++ )
                  SCIP_CALL( checkAndAddExternalBranchingCandidate(scip, masterorigvars[j]) );
            }
         }
         else
         {
            assert( GCGgetDecompositionMode(scip) == DEC_DECMODE_DANTZIGWOLFE );
            /* store branching candidates */
            for( i = 0; i < norigvars; i++ )
               SCIP_CALL( checkAndAddExternalBranchingCandidate(scip, origvars[i]) );
         }

         SCIPdebugMessage("updated relaxation branching candidates\n");
      }
   }

   return SCIP_OKAY;
}


/** gets the total memory used after problem creation stage for all pricingproblems */
SCIP_Real GCGgetPricingprobsMemUsed(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   int p;
   SCIP_Real memused;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   memused = 0.0;

   /* @todo replace the computation by relaxdata->pricingprobsmemused if we can assure that the memory
    * used by the pricing problems is constant */

   /* compute memory that is used by all pricing problems */
   for( p = 0; p < relaxdata->npricingprobs; ++p )
   {
      memused += SCIPgetMemUsed(relaxdata->pricingprobs[p])/1048576.0;
   }

   return memused;
}

/** returns whether the relaxator has been initialized */
SCIP_Bool GCGrelaxIsInitialized(
   SCIP*                 scip                /**< SCIP data structure */
   )
{

   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->relaxisinitialized;
}

/** returns the average degeneracy */
SCIP_Real GCGgetDegeneracy(
   SCIP*                 scip                /**< SCIP data structure */
   )
{

   SCIP_Real degeneracy;
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);
   degeneracy = 0.0;
   if( relaxdata->masterprob != NULL )
   {
      degeneracy = GCGmasterGetDegeneracy(relaxdata->masterprob);
      if( SCIPisInfinity(relaxdata->masterprob, degeneracy) )
         degeneracy = SCIPinfinity(scip);
   }
   return degeneracy;
}

/** return linking constraints for variables */
SCIP_CONS** GCGgetVarLinkingconss(
   SCIP*                 scip                /**< SCIP data structure */
  )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->varlinkconss;
}

/** return blocks of linking constraints for variables */
int* GCGgetVarLinkingconssBlock(
   SCIP*                 scip                /**< SCIP data structure */
  )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->varlinkconsblock;
}

/** return number of linking constraints for variables */
int GCGgetNVarLinkingconss(
   SCIP*                 scip                /**< SCIP data structure */
  )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->nvarlinkconss;
}

/** return number of linking variables */
int GCGgetNLinkingvars(
   SCIP*                 scip                /**< SCIP data structure */
  )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->nlinkingvars;
}

/** return number of variables directly transferred to the master problem */
int GCGgetNTransvars(
   SCIP*                 scip                /**< SCIP data structure */
  )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->ntransvars;
}

/** returns the relaxation solution from the Benders' decomposition */
SCIP_SOL* GCGgetBendersRelaxationSol(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;
   SCIP_BENDERS* benders;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   benders = SCIPfindBenders(relaxdata->masterprob, "gcg");
   assert(benders != NULL);

   return SCIPbendersGetRelaxSol(benders);
}

/** returns the decomposition mode */
DEC_DECMODE GCGgetDecompositionMode(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return (DEC_DECMODE)relaxdata->mode;
}

/** returns the decomposition mode of the master problem. The mode is given by the existence of either the GCG pricer or
 * the GCG Benders' decomposition plugins.
 */
DEC_DECMODE GCGgetMasterDecompMode(
   SCIP*                 masterprob          /**< the master problem SCIP instance */
   )
{
   SCIP_BENDERS* benders;
   SCIP_PRICER* pricer;
   DEC_DECMODE mode;

   assert(masterprob != NULL);

   /* retrieving the Benders' decomposition and the pricer plugins. There should only be one or the other for a given
    * master problem. If there are both, then an error is returned */
   benders = SCIPfindBenders(masterprob, "gcg");
   pricer = SCIPfindPricer(masterprob, "gcg");
   assert((benders != NULL && pricer == NULL) || (pricer != NULL && benders == NULL));

   if( benders != NULL )
   {
      /* both the Benders' master and the original master have the Benders' decomposition included. */
      if( SCIPgetNActiveBenders(masterprob) > 0 )
         mode = DEC_DECMODE_BENDERS;
      else
         mode = DEC_DECMODE_ORIGINAL;
   }
   else if( pricer != NULL )
      mode = DEC_DECMODE_DANTZIGWOLFE;
   else
   {
      mode = DEC_DECMODE_UNKNOWN;
      SCIPerrorMessage("Sorry, the decomposition mode of the master problem is invalid. This should not happen.");
      SCIPABORT();
   }

   return mode;
}

/** return root node time clock */
SCIP_CLOCK* GCGgetRootNodeTime(
   SCIP*                scip              /**< SCIP data structure */
   )
{
   SCIP_RELAX* relax;
   SCIP_RELAXDATA* relaxdata;

   assert(scip != NULL);

   relax = SCIPfindRelax(scip, RELAX_NAME);
   assert(relax != NULL);

   relaxdata = SCIPrelaxGetData(relax);
   assert(relaxdata != NULL);

   return relaxdata->rootnodetime;
}

SCIP_RETCODE GCGtransformProb(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   switch( SCIPgetStage(scip) )
   {
   case SCIP_STAGE_INIT:
      SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "no problem exists\n");
      break;

   case SCIP_STAGE_PROBLEM:
      SCIP_CALL( SCIPconshdlrDecompRepairConsNames(scip) );
      SCIP_CALL( SCIPtransformProb(scip) );
      break;

   case SCIP_STAGE_TRANSFORMED:
      SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "problem is already transformed\n");
      break;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_INITPRESOLVE:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_EXITPRESOLVE:
   case SCIP_STAGE_INITSOLVE:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
   case SCIP_STAGE_EXITSOLVE:
   case SCIP_STAGE_FREETRANS:
   case SCIP_STAGE_FREE:
   default:
      SCIPerrorMessage("invalid SCIP stage\n");
      return SCIP_INVALIDCALL;
   }

   return SCIP_OKAY;
}

SCIP_RETCODE GCGpresolve(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   switch( SCIPgetStage(scip) )
   {
   case SCIP_STAGE_INIT:
      SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "no problem exists\n");
      break;

   case SCIP_STAGE_PROBLEM:
      SCIP_CALL( GCGtransformProb(scip) );
      assert(SCIPgetStage(scip) == SCIP_STAGE_TRANSFORMED);

      /*lint -fallthrough*/

   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
      SCIP_CALL( SCIPpresolve(scip) );
      SCIP_CALL( GCGconshdlrDecompTranslateOrigPartialdecs(scip) );
      break;

   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
      SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "problem is already presolved\n");
      break;

   case SCIP_STAGE_SOLVED:
      SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "problem is already solved\n");
      break;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_INITPRESOLVE:
   case SCIP_STAGE_EXITPRESOLVE:
   case SCIP_STAGE_INITSOLVE:
   case SCIP_STAGE_EXITSOLVE:
   case SCIP_STAGE_FREETRANS:
   case SCIP_STAGE_FREE:
   default:
      SCIPerrorMessage("invalid SCIP stage\n");
      return SCIP_INVALIDCALL;
   }

   return SCIP_OKAY;
}

SCIP_RETCODE GCGdetect(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RESULT result;

   switch( SCIPgetStage(scip) )
   {
   case SCIP_STAGE_INIT:
      SCIPverbMessage(scip, SCIP_VERBLEVEL_DIALOG, NULL, "no problem exists\n");
      break;

   case SCIP_STAGE_PROBLEM:
      SCIP_CALL( GCGtransformProb(scip) );
      assert(SCIPgetStage(scip) == SCIP_STAGE_TRANSFORMED);

      /*lint -fallthrough*/

   case SCIP_STAGE_TRANSFORMED:
      if( GCGdetectionTookPlace(scip, TRUE) )
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_DIALOG, NULL, "The detection for the original problem took place already.\n");
      }
      else
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_DIALOG, NULL, "starting detection\n");
         SCIP_CALL( DECdetectStructure(scip, &result) );
      }
      break;
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
      if( GCGdetectionTookPlace(scip, FALSE) )
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_DIALOG, NULL, "The detection for the presolved problem took place already.\n");
      }
      else
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_DIALOG, NULL, "starting detection\n");
         SCIP_CALL( DECdetectStructure(scip, &result) );
      }
      break;
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_INITPRESOLVE:
   case SCIP_STAGE_EXITPRESOLVE:
   case SCIP_STAGE_INITSOLVE:
   case SCIP_STAGE_EXITSOLVE:
   case SCIP_STAGE_FREETRANS:
   case SCIP_STAGE_FREE:
   default:
      SCIPerrorMessage("invalid SCIP stage\n");
      return SCIP_INVALIDCALL;
   }

   return SCIP_OKAY;
}

SCIP_RETCODE GCGsolve(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_RESULT result;
   int presolrounds;
   SCIP_Bool exit = FALSE;

   presolrounds = -1;

   assert(GCGconshdlrDecompCheckConsistency(scip) );

   while( !exit )
   {
      switch( SCIPgetStage(scip) )
      {
      case SCIP_STAGE_INIT:
         SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "No problem exists\n");
         exit = TRUE;
         break;

      case SCIP_STAGE_PROBLEM:
         SCIP_CALL( GCGtransformProb(scip) );
         assert(SCIPgetStage(scip) == SCIP_STAGE_TRANSFORMED);

         /*lint -fallthrough*/

      case SCIP_STAGE_TRANSFORMED:
      case SCIP_STAGE_PRESOLVING:
         if( GCGconshdlrDecompOrigPartialdecExists(scip) )
         {
            SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "there is an original decomposition and problem is not presolved yet -> disable presolving and start optimizing (rerun with presolve command before detect command for detecting in presolved problem)  \n");
            SCIP_CALL( SCIPgetIntParam(scip, "presolving/maxrounds", &presolrounds) );
            SCIP_CALL( SCIPsetIntParam(scip, "presolving/maxrounds", 0) );
         }
         SCIP_CALL( GCGpresolve(scip) );
         assert(SCIPgetStage(scip) > SCIP_STAGE_PRESOLVING);

         break;

      case SCIP_STAGE_PRESOLVED:
         assert(GCGconshdlrDecompCheckConsistency(scip) );

         if( !GCGdetectionTookPlace(scip, TRUE) && !GCGdetectionTookPlace(scip, FALSE) && GCGconshdlrDecompGetNFinishedPartialdecsTransformed(scip) == 0 )
         {
            SCIP_CALL( DECdetectStructure(scip, &result) );
            if( result == SCIP_DIDNOTFIND )
            {
               DEC_DECOMP* bestdecomp;
               bestdecomp = DECgetBestDecomp(scip, TRUE);
               assert(bestdecomp == NULL && (GCGdetectionTookPlace(scip, TRUE) || GCGdetectionTookPlace(scip, FALSE)));
               DECdecompFree(scip, &bestdecomp);
               SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "No decomposition exists or could be detected. Solution process started with original problem...\n");
            }
         }
         else if( !GCGdetectionTookPlace(scip, TRUE) && !GCGdetectionTookPlace(scip, FALSE) && GCGconshdlrDecompGetNFinishedPartialdecsTransformed(scip) > 0 )
         {
   #ifndef NDEBUG
            DEC_DECOMP* bestdecomp;
            bestdecomp = DECgetBestDecomp(scip, TRUE);
            assert(bestdecomp != NULL);
            DECdecompFree(scip, &bestdecomp);
   #endif
            SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "Preexisting decomposition found. Solution process started...\n");
         }
         else if( GCGconshdlrDecompGetNFinishedPartialdecsTransformed(scip) == 0 )
         {
            SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "No decomposition exists or could be detected. Solution process started with original problem...\n");
         }
         assert(GCGconshdlrDecompCheckConsistency(scip));
         assert(SCIPgetNConss(scip) == SCIPgetNActiveConss(scip));

         /*lint -fallthrough*/
      case SCIP_STAGE_SOLVING:
         SCIP_CALL( SCIPsolve(scip) );
         exit = TRUE;
         break;

      case SCIP_STAGE_SOLVED:
         SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "Problem is already solved\n");
         exit = TRUE;
         break;

      case SCIP_STAGE_TRANSFORMING:
      case SCIP_STAGE_INITPRESOLVE:
      case SCIP_STAGE_EXITPRESOLVE:
      case SCIP_STAGE_INITSOLVE:
      case SCIP_STAGE_EXITSOLVE:
      case SCIP_STAGE_FREETRANS:
      case SCIP_STAGE_FREE:
      default:
         SCIPerrorMessage("invalid SCIP stage <%d>\n", SCIPgetStage(scip));
         return SCIP_INVALIDCALL;
      }
   }

   if( presolrounds != -1 )
   {
      SCIP_CALL( SCIPsetIntParam(scip, "presolving/maxrounds", presolrounds) );
   }

   return SCIP_OKAY;
}

SCIP_Real GCGgetDualbound(
   SCIP*                scip              /**< SCIP data structure */
   )
{
   SCIP* masterprob;
   SCIP_Real dualbound;

   assert(scip != NULL);

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   dualbound = SCIPgetDualbound(scip);

   /* @todo find a better way to do this */
   if( SCIPgetStage(masterprob) >= SCIP_STAGE_SOLVING )
   {
      SCIP_Real masterdualbound;

      masterdualbound = SCIPgetDualbound(masterprob);
      masterdualbound = SCIPretransformObj(scip, masterdualbound);
      dualbound = MAX(dualbound, masterdualbound);
   }

   return dualbound;
}

SCIP_Real GCGgetPrimalbound(
   SCIP*                scip              /**< SCIP data structure */
   )
{
   SCIP* masterprob;
   SCIP_Real primalbound;

   assert(scip != NULL);

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   primalbound = SCIPgetPrimalbound(scip);

   /* @todo find a better way to do this */
   if( SCIPgetStage(masterprob) >= SCIP_STAGE_SOLVING && GCGmasterIsBestsolValid(masterprob) )
   {
      SCIP_Real masterprimalbound;
      masterprimalbound = SCIPgetPrimalbound(masterprob);
      masterprimalbound = SCIPretransformObj(scip, masterprimalbound);

      primalbound = MIN(primalbound, masterprimalbound);
   }

   return primalbound;
}

SCIP_Real GCGgetGap(
   SCIP*                scip              /**< SCIP data structure */
   )
{
   SCIP_Real dualbound;
   SCIP_Real primalbound;
   SCIP_Real gap;

   assert(scip != NULL);

   primalbound = GCGgetPrimalbound(scip);
   dualbound = GCGgetDualbound(scip);

   /* this is the gap calculation from SCIPgetGap() */
   if( SCIPisEQ(scip, primalbound, dualbound) )
      gap = 0.0;
   else if( SCIPisZero(scip, dualbound )
      || SCIPisZero(scip, primalbound)
      || SCIPisInfinity(scip, REALABS(primalbound))
      || SCIPisInfinity(scip, REALABS(dualbound))
      || primalbound * dualbound < 0.0 )
      gap = SCIPinfinity(scip);
   else
   {
      SCIP_Real absdual = REALABS(dualbound);
      SCIP_Real absprimal = REALABS(primalbound);

      gap = REALABS((primalbound - dualbound)/MIN(absdual, absprimal));
   }

   return gap;
}
