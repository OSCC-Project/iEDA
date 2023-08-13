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

/**@file    branch_bpstrong.c
 * @ingroup BRANCHINGRULES
 * @brief   generic branch-and-price strong branching heuristics
 * @author  Oliver Gaul
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
/*#define SCIP_DEBUG*/
#include <assert.h>

#include "branch_bpstrong.h"
#include "type_branchgcg.h"
#include "gcg.h"
#include "branch_orig.h"

#include <string.h>

#include "gcg.h"
#include "branch_relpsprob.h"
#include "cons_integralorig.h"
#include "cons_masterbranch.h"
#include "cons_origbranch.h"
#include "relax_gcg.h"
#include "pricer_gcg.h"
#include "type_branchgcg.h"

#include "scip/cons_linear.h"
#include "scip/scipdefplugins.h"
#include "scip/var.h"

#define BRANCHRULE_NAME "bpstrong"                              /**< name of branching rule */
#define BRANCHRULE_DESC "strong branching for branch-and-price" /**< short description of branching rule */
#define BRANCHRULE_PRIORITY -536870912                          /**< priority of this branching rule */
#define BRANCHRULE_MAXDEPTH 0                                   /**< maximal depth level of the branching rule */
#define BRANCHRULE_MAXBOUNDDIST 0.0                             /**< maximal relative distance from current node's
                                                                   * dual bound to primal bound compared to best node's
                                                                   * dual bound for applying branching */
#define DEFAULT_ENFORCEBYCONS      FALSE
#define DEFAULT_MOSTFRAC           FALSE
#define DEFAULT_USEPSEUDO          TRUE
#define DEFAULT_USEPSSTRONG        FALSE

#define DEFAULT_USESTRONG          FALSE
#define DEFAULT_STRONGLITE         FALSE
#define DEFAULT_STRONGTRAIN        FALSE
#define DEFAULT_IMMEDIATEINF       TRUE
#define DEFAULT_MAXSBLPITERS       INT_MAX
#define DEFAULT_MAXSBPRICEROUNDS   INT_MAX

#define DEFAULT_RFUSEPSEUDOCOSTS   TRUE
#define DEFAULT_RFUSEMOSTFRAC      FALSE

#define DEFAULT_REEVALAGE          1
#define DEFAULT_MINCOLGENCANDS     4
#define DEFAULT_HISTWEIGHT         0.5
#define DEFAULT_MAXLOOKAHEAD       8
#define DEFAULT_LOOKAHEADSCALES    0.5
#define DEFAULT_MINPHASE0DEPTH     0
#define DEFAULT_MAXPHASE1DEPTH     4
#define DEFAULT_MAXPHASE2DEPTH     3
#define DEFAULT_DEPTHLOGWEIGHT     0.5
#define DEFAULT_DEPTHLOGBASE       3.5
#define DEFAULT_DEPTHLOGPHASE0FRAC 0
#define DEFAULT_DEPTHLOGPHASE2FRAC 0.75

#define DEFAULT_CLOSEPERCENTAGE    0.90
#define DEFAULT_MAXCONSECHEURCLOSE 4

#define DEFAULT_SBPSEUDOCOSTWEIGHT 1

#define DEFAULT_PHASE1RELIABLE     INT_MAX
#define DEFAULT_PHASE2RELIABLE     INT_MAX

#define DEFAULT_FORCEP0            FALSE

#define ORIG                       0
#define RYANFOSTER                 1
#define GENERIC                    2


/** branching data for branching decisions (for Ryan-Foster branching) */
struct GCG_BranchData
{
   SCIP_VAR*             var1;               /**< first original variable on which the branching is done */
   SCIP_VAR*             var2;               /**< second original variable on which the branching is done */
   SCIP_Bool             same;               /**< should each master var contain either both or none of the vars? */
   int                   blocknr;            /**< number of the block in which branching was performed */
   SCIP_CONS*            pricecons;          /**< constraint enforcing the branching restriction in the pricing
                                                  problem */
};

/* stores candidates and their corresponding index to insert them into a hashtable */
typedef struct VarTuple{
   SCIP_VAR* var1;
   SCIP_VAR* var2;
   int index;
} VarTuple;

/** branching rule data */
struct SCIP_BranchruleData
{
   int                   lastcand;              /**< last evaluated candidate of last branching rule execution */
   int                   nvars;                 /**< the number of candidates currently in the hashtable */
   int                   maxvars;               /**< the maximal number of cands that were in the hashtable at the same
                                                   * time */
   int                   maxcands;              /**< (realistically) the maximum total amount of candidates */
   SCIP_HASHTABLE*       candhashtable;         /**< hashtable mapping candidates to their index */
   VarTuple              **vartuples;            /**< all VarTuples that are in candhashtable */
   SCIP_Real             *score;                /**< the candidates' last scores */
   int                   *uniqueblockflags;     /**< flags assigned by assignUniqueBlockFlags() */
   SCIP_Real             *strongbranchscore;    /**< the candidates' last scores from strong branching with column
                                                   * generation */
   SCIP_Bool             *sbscoreisrecent;      /**< was the score saved in strongbranchscore computed in a parent of
                                                   * the current node where all node on the path to the parent were
                                                   * created for domain reduction due to infeasibility? */
   int                   *lastevalnode;         /**< the last node at which the candidates were evaluated */

   int                   nphase1lps;            /**< number of phase 1 lps solved */
   int                   nphase2lps;            /**< number of phase 2 lps solved */
   SCIP_Longint          nsblpiterations;       /**< total number of strong branching lp iterations during phase 1 */
   int                   nsbpricerounds;        /**< total number of strong branching pricing rounds */

   int                   initiator;             /**< the identifier of the branching rule that initiated strong
                                                   * branching */
   SCIP_BRANCHRULE*      initiatorbranchrule;   /**< the branching rule that initiated strong branching */

   SCIP_Bool             mostfrac;              /**< should most infeasible/fractional branching be used in phase 0? */
   SCIP_Bool             usepseudocosts;        /**< should pseudocost branching be used in phase 0? */

   SCIP_Bool             usestronglite;         /**< should strong branching use column generation during variable
                                                   * evaluation? */
   SCIP_Bool             usestrongtrain;        /**< should strong branching run as precise as possible
                                                   * (to generate more valuable training data, currently not
                                                   * implemented)? */
   SCIP_Bool             immediateinf;          /**< should infeasibility detected during strong branching be handled
                                                   * immediately, or only if the variable is selected? */
   SCIP_Longint          maxsblpiters;          /**< maximum number of strong branching lp iterations, set to 2*avg lp
                                                   * iterations if <= 0 */
   int                   maxsbpricerounds;      /**< maximum number of strong branching pricing rounds, set to 2*avg lp
                                                   * iterations if <= 0 */
   int                   reevalage;             /**< how many times can bounds be changed due to infeasibility during
                                                   * strong branching until an already evaluated variable needs to be
                                                   * reevaluated? */
   int                   mincolgencands;        /**< minimum number of variables for phase 2 to be executed, otherwise
                                                   * the best candidate from phase 1 will be chosen */

   int                   minphase0depth;        /**< minimum tree depth from which on phase 0 is performed (~ hybrid
                                                   * branching) */
   int                   maxphase1depth;        /**< maximum tree depth up to which phase 1 is performed (~ hybrid
                                                   * branching) */
   int                   maxphase2depth;        /**< maximum tree depth up to which phase 2 is performed (~ hybrid
                                                   * branching) */

   int                   minphase0outcands;     /**< minimum number of output candidates from phase 0 */
   int                   maxphase0outcands;     /**< maximum number of output candidates from phase 0 */
   SCIP_Real             maxphase0outcandsfrac; /**< maximum number of output candidates from phase 0 as fraction of
                                                   * total cands */
   SCIP_Real             phase1gapweight;       /**< how much impact should the node gap have on the number of
                                                   * precisely evaluated candidates in phase 1? */

   int                   minphase1outcands;     /**< minimum number of output candidates from phase 1 */
   int                   maxphase1outcands;     /**< maximum number of output candidates from phase 1 */
   SCIP_Real             maxphase1outcandsfrac; /**< maximum number of output candidates from phase 0 as fraction of
                                                   * phase 1 candidates */
   SCIP_Real             phase2gapweight;       /**< how much impact should the node gap have on the number of
                                                   * precisely evaluated candidates in phase 2? */
   int                   maxlookahead;          /**< maximum number of non-improving candidates until phase 2 is
                                                   * stopped */
   SCIP_Real             lookaheadscales;       /**< how much should the look ahead scale with the overall evaluation
                                                   * effort? (0 = not at all, 1 = fully) */

   SCIP_Real             histweight;            /**< how many candidates should be chosen based on historical strong
                                                   * branching scores as opposed to current heuristic scores in phase 0
                                                   * (e.g. 0.5 = 50%)? */

   SCIP_Real             closepercentage;       /**< what percentage of the strong branching score of the candidate
                                                   * that was selected does the best candidate according to the phase 0
                                                   * heuristic need to have to be considered close? */
   int                   consecheurclose;       /**< how many times in a row the best candidate according to the phase
                                                   * 0 heuristic was close to that selected by SBw/CG */
   int                   maxconsecheurclose;    /**< how many times in a row can the heuristic be close before strong
                                                   * branching is stopped? */

   SCIP_Bool             done;                  /**< has any of the permanent stopping criteria been reached? */

   SCIP_Real             sbpseudocostweight;    /**< with how much weight should strong branching scores be considered
                                                   * for pseudocost scores? */

   int                   phase1reliable;        /**< min count of pseudocost scores for a variable to be considered
                                                   * reliable in phase 1 (~ reliability branching) */
   int                   phase2reliable;        /**< min count of pseudocost scores for a variable to be considered
                                                   * reliable in phase 2 (~ reliability branching) */

   SCIP_Bool             forcephase0;           /**< should phase 0 be performed even if the number of input candidates
                                                   * is already lower or equal to the number of output candidates? */

   SCIP_Bool             initialized;           /**< has the branching rule been initialized? */
};

/* needed for compare_function (for now) */
SCIP_BRANCHRULEDATA*     this_branchruledata;

/*
 * Hash functions
 */

/** gets the hash key of a variable tuple */
static
SCIP_DECL_HASHGETKEY(hashGetKeyVars)
{
   VarTuple* vars;

   vars = (VarTuple*)elem;
   assert(vars != NULL);

   /* the key of a variable tuple is the variable tuple itself */
   return vars;
}

/** returns TRUE iff both variable tuples contain the same variables (ignoring order) */
static
SCIP_DECL_HASHKEYEQ(hashKeyEqVars)
{
   VarTuple* vars1;
   VarTuple* vars2;

   vars1 = (VarTuple*)key1;
   vars2 = (VarTuple*)key2;
   assert(vars1 != NULL);
   assert(vars2 != NULL);

   /* check if first variable is equal */
   if( SCIPvarGetIndex(vars1->var1) != SCIPvarGetIndex(vars2->var1) )
      return FALSE;

   /* second variable might be NULL */
   if( (vars1->var2 == NULL) && (vars2->var2 == NULL) )
      return TRUE;

   if( (vars1->var2 == NULL) || (vars2->var2 == NULL) )
      return FALSE;

   /* check if second variable is equal */
   if( SCIPvarGetIndex(vars1->var2) != SCIPvarGetIndex(vars2->var2) )
      return FALSE;

   return TRUE;
}

static
SCIP_DECL_HASHKEYVAL(hashKeyValVars)
{
   VarTuple* vars;
   unsigned int hashvalue;
   SCIP_VAR* var1;
   SCIP_VAR* var2;

   vars = (VarTuple*)key;
   assert(vars != NULL);

   var1 = vars->var1;
   var2 = vars->var2;

   /* return hashvalue of indices */
   if( var2 == NULL )
      hashvalue = SCIPhashSignature64( SCIPvarGetIndex(var1) );
   else
      hashvalue = SCIPhashTwo( MIN( SCIPvarGetIndex(var1), SCIPvarGetIndex(var2) ), MAX( SCIPvarGetIndex(var1), SCIPvarGetIndex(var2) ) );

   return hashvalue;
}

/* calculates the number of needed candidates based on the min and max number of candidates as well as the node gap */
static
int calculateNCands(
   SCIP*                 scip,               /**< scip data structure */
   SCIP_BRANCHRULEDATA*  branchruledata,     /**< strong branching branchruledata */
   SCIP_Real             nodegap,            /**< node gap in current focus node */
   int                   phase,              /**< phase we are calculating this for */
   int                   ncands              /**< number of input candidates for the phase */
)
{
   int min;
   int max;
   int dif;
   SCIP_Real gapweight;
   SCIP_Real candfrac;

   if( phase == 0 )
   {
      min = branchruledata->minphase0outcands;
      max = branchruledata->maxphase0outcands;
      candfrac = branchruledata->maxphase0outcandsfrac;
      gapweight = branchruledata->phase1gapweight;
   }
   else
   {
      min = branchruledata->minphase1outcands;
      max = branchruledata->maxphase1outcands;
      candfrac = branchruledata->maxphase1outcandsfrac;
      gapweight = branchruledata->phase2gapweight;
   }

   dif = max-min;

   assert(min >= 1);

   return MIN( candfrac*ncands,
               min + (int) SCIPceil(scip, MIN( dif, dif * nodegap * gapweight + dif * (1-gapweight) )) );
}

/* assigns a flag to the given branching candidate based on the block it is in
 *
 * return  1: integer variables belonging to a unique block with fractional value
 * return  0: variables that belong to no block but were directly transferred to the
 *            master problem and which have a fractional value in the current solution
 * return -1: neither
 */
static
int assignUniqueBlockFlags(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             branchcand          /**< branching candidate to be considered */
)
{
   assert(GCGvarIsOriginal(branchcand));

   for ( int iter = 0; iter <= 1; iter++ )
   {
      /* continue if variable belongs to a block in second iteration*/
      if (iter == 0)
      {
         /* variable belongs to no block */
         if ( GCGvarGetBlock(branchcand) == -1 )
            continue;

         /* block is not unique (non-linking variables) */
         if ( !GCGoriginalVarIsLinking(branchcand) && GCGgetNIdenticalBlocks(scip, GCGvarGetBlock(branchcand)) != 1 )
            continue;

         /* check that blocks of linking variable are unique */
         if ( GCGoriginalVarIsLinking(branchcand) )
         {
            int nvarblocks;
            int *varblocks;
            SCIP_Bool unique;
            int j;

            nvarblocks = GCGlinkingVarGetNBlocks(branchcand);
            SCIP_CALL( SCIPallocBufferArray(scip, &varblocks, nvarblocks) );
            SCIP_CALL( GCGlinkingVarGetBlocks(branchcand, nvarblocks, varblocks) );

            unique = TRUE;
            for ( j = 0; j < nvarblocks; ++j )
               if ( GCGgetNIdenticalBlocks(scip, varblocks[j]) != 1 )
                  unique = FALSE;

            SCIPfreeBufferArray(scip, &varblocks);

            if( !unique )
               continue;
         }
         /* candidate is valid in first iteration */
         return 1;
      }
      else /* iter == 1 */
      {
         if ( GCGvarGetBlock(branchcand) != -1 )
            return -1;

         /* candidate is valid in second iteration */
         return 0;
      }
   }
   return -1;
}

/** adds branching candidates to branchruledata to collect infos about it */
static
SCIP_RETCODE addBranchcandsToData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_VAR**            var1s,              /**< first parts of branching candidates */
   SCIP_VAR**            var2s,              /**< second parts of branching candidates */
   int                   ncands              /**< number of priority branching candidates */
   )
{
   SCIP* masterscip;
   SCIP_BRANCHRULEDATA* branchruledata;
   int i;
   VarTuple* vartuple = NULL;
   int nvars;
   int newsize;

   /* get branching rule data */
   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);

   masterscip = GCGgetMasterprob(scip);

   /* if var is not in hashtable, insert it */
   for( i = 0; i < ncands; i++ )
   {
      nvars = branchruledata->nvars;

      /* if variable is not in hashmtable insert it, initialize its array entries, and increase array sizes */
      SCIP_CALL( SCIPallocBlockMemory(masterscip, &vartuple) );
      vartuple->var1 = var1s[i];
      vartuple->var2 = var2s!=NULL? var2s[i] : NULL;
      vartuple->index = nvars;

      if( !SCIPhashtableExists(branchruledata->candhashtable, (void *) vartuple) )
      {
         newsize = SCIPcalcMemGrowSize(masterscip, nvars + 1);
         SCIP_CALL( SCIPreallocBlockMemoryArray(masterscip, &branchruledata->strongbranchscore,
                     branchruledata->maxvars, newsize) );
         SCIP_CALL( SCIPreallocBlockMemoryArray(masterscip, &branchruledata->sbscoreisrecent,
                     branchruledata->maxvars, newsize) );
         SCIP_CALL( SCIPreallocBlockMemoryArray(masterscip, &branchruledata->lastevalnode, branchruledata->maxvars,
                     newsize) );
         SCIP_CALL( SCIPreallocBlockMemoryArray(masterscip, &branchruledata->uniqueblockflags,
                     branchruledata->maxvars, newsize) );
         SCIP_CALL( SCIPreallocBlockMemoryArray(masterscip, &branchruledata->vartuples,
                     branchruledata->maxvars, newsize) );
         branchruledata->maxvars = newsize;

         branchruledata->strongbranchscore[nvars] = -1;
         branchruledata->sbscoreisrecent[nvars] = FALSE;
         branchruledata->lastevalnode[nvars] = -1;
         branchruledata->uniqueblockflags[nvars] = -2;
         branchruledata->vartuples[nvars] = vartuple;
         SCIP_CALL( SCIPhashtableInsert(branchruledata->candhashtable,
                                          (void*) branchruledata->vartuples[nvars]) );

         assert(SCIPhashtableExists(branchruledata->candhashtable, (void*) branchruledata->vartuples[nvars])
                  && ( (VarTuple *) SCIPhashtableRetrieve(branchruledata->candhashtable,
                                          (void*) branchruledata->vartuples[nvars]) )->index == nvars);

         ++(branchruledata->nvars);
      }
      else
      {
         SCIPfreeBlockMemory(masterscip, &vartuple);
      }
   }

   return SCIP_OKAY;
}

/* compare two indices corresponding to entries in branchruledata->score, returns TRUE iff the first elements score is
 * larger
 */
static int score_compare_function(
   const void            *index1,            /**< index in branchruledata->score of first element */
   const void            *index2             /**< index in branchruledata->score of first element */
   )
{
   return this_branchruledata->score[*(int *)index1] > this_branchruledata->score[*(int *)index2]? -1 : 1;
}

/* compare two indices based on descending numerical order, returns TRUE iff the first index is smaller */
static int geq_compare_function(
   const void            *index1,            /**< first index */
   const void            *index2             /**< second index */
   )
{
   return *(int *)index1 < *(int *)index2? -1 : 1;
}

/* creates a probing node that is "equal" to the same or differ branch of a given Ryan-Foster candidate, mostly copied
 * from the corresponding method in branch_ryanfoster.c
 */
static
SCIP_RETCODE newProbingNodeRyanfosterMaster(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_VAR*             ovar1,              /**< first original variable */
   SCIP_VAR*             ovar2,              /**< second original variable */
   int                   blocknr,            /**< number of the pricing block */
   SCIP_Bool             same                /**< do we want to create the same (TRUE) or differ (FALSE) branch? */
   )
{
   SCIP* masterscip;
   SCIP_VAR* pricingvar1;
   SCIP_VAR* pricingvar2;
   GCG_BRANCHDATA* branchdata;
   char name[SCIP_MAXSTRLEN];

   SCIP_VAR** origvars1;
   SCIP_VAR** origvars2;
   int norigvars;
   int maxorigvars;
   int v;

   SCIP_CONS** origbranchconss;


   assert(scip != NULL);
   assert(branchrule != NULL);
   assert(ovar1 != NULL);
   assert(ovar2 != NULL);
   assert(GCGvarIsOriginal(ovar1));
   assert(GCGvarIsOriginal(ovar2));

   origbranchconss = NULL;

   masterscip = GCGgetMasterprob(scip);
   assert(masterscip != NULL);

   /* for cons_masterbranch */

   /* allocate branchdata for same child and store information */
   SCIP_CALL( SCIPallocBlockMemory(scip, &branchdata) );
   branchdata->var1 = ovar1;
   branchdata->var2 = ovar2;
   branchdata->same = same;
   branchdata->blocknr = blocknr;
   branchdata->pricecons = NULL;

   /* define name for origbranch constraints */
   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s(%s,%s)", same? "same" : "differ", SCIPvarGetName(branchdata->var1),
      SCIPvarGetName(branchdata->var2));

   pricingvar1 = GCGoriginalVarGetPricingVar(branchdata->var1);
   pricingvar2 = GCGoriginalVarGetPricingVar(branchdata->var2);
   assert(GCGvarIsPricing(pricingvar1));
   assert(GCGvarIsPricing(pricingvar2));
   assert(GCGvarGetBlock(pricingvar1) == GCGvarGetBlock(pricingvar2));
   assert(GCGpricingVarGetNOrigvars(pricingvar1) == GCGpricingVarGetNOrigvars(pricingvar2));

   norigvars = GCGpricingVarGetNOrigvars(pricingvar1);
   assert(norigvars == GCGpricingVarGetNOrigvars(pricingvar2));

   origvars1 = GCGpricingVarGetOrigvars(pricingvar1);
   origvars2 = GCGpricingVarGetOrigvars(pricingvar2);

   if( norigvars > 0 )
   {
      maxorigvars = SCIPcalcMemGrowSize(masterscip, norigvars);
      SCIP_CALL( SCIPallocBlockMemoryArray(masterscip, &origbranchconss, maxorigvars) );
   }
   else
   {
      maxorigvars = 0;
   }

   /* add branching decision as varbound constraints to original problem */
   for( v = 0; v < norigvars; v++ )
   {
      SCIP_CONS* origcons;

      assert(GCGvarGetBlock(origvars1[v]) == GCGvarGetBlock(origvars2[v]));
      assert(origbranchconss != NULL);

      /* create constraint for same-child */
      SCIP_CALL( SCIPcreateConsVarbound(scip, &origcons, name, origvars1[v], origvars2[v],
            same? -1.0 : 1.0, same ? 0.0 : -SCIPinfinity(scip), same? 0.0 : 1.0, TRUE, TRUE,
            TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );

      origbranchconss[v] = origcons;
   }

   /* create and add the masterbranch constraints */
   SCIP_CALL( GCGrelaxNewProbingnodeMasterCons(scip, branchrule, branchdata, origbranchconss, norigvars,
                                               maxorigvars) );

   return SCIP_OKAY;
}

/* executes strong branching on one variable, with or without pricing */
static
SCIP_RETCODE executeStrongBranching(
    SCIP                 *scip,              /* SCIP data structure */
    SCIP_BRANCHRULE*     branchrule,         /* pointer to the branching rule */
    SCIP_VAR             *branchvar1,        /* first variable to get strong branching values for */
    SCIP_VAR             *branchvar2,        /* second variable to get strong branching values for */
    SCIP_Real            solval1,            /* value of the first variable in the current solution */
    SCIP_Real            solval2,            /* value of the second variable in the current solution */
    int                  candinfo,           /* additional intager information about the candidate */
    SCIP_Bool            pricing,            /* should pricing be applied? */
    int                  maxpricingrounds,   /* maximal number of pricing rounds, -1 for no limit */
    SCIP_Real            *up,                /* stores dual bound for up/same child */
    SCIP_Real            *down,              /* stores dual bound for down/differ child */
    SCIP_Bool            *upvalid,           /* stores whether the up/samebranch was solved properly */
    SCIP_Bool            *downvalid,         /* stores whether the down/differbranch was solved properly */
    SCIP_Bool            *upinf,             /* stores whether the up/samebranch is infeasible */
    SCIP_Bool            *downinf            /* stores whether the down/differbranch is infeasible */
)
{
   /* get bound values */
   SCIP_BRANCHRULEDATA* branchruledata;

   SCIP_Bool cutoff;
   SCIP_Bool lperror;
   SCIP_Bool lpsolved;

   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);

   *downvalid = FALSE;
   *upvalid = FALSE;
   *downinf = FALSE;
   *upinf = FALSE;

   assert(scip != NULL);

   /* probe for each child node */
   for( int cnode = 0; cnode <= 1; cnode++ )
   {
      /* start probing */
      SCIP_CALL( GCGrelaxStartProbing(scip, NULL) );
      SCIP_CALL( GCGrelaxNewProbingnodeOrig(scip) );

      cutoff = FALSE;
      lperror = FALSE;
      lpsolved = FALSE;

      if( branchruledata->initiator == ORIG )
      {
         if( cnode == 0 )
         {
            SCIP_CALL( SCIPchgVarUbProbing(scip, branchvar1, SCIPfeasFloor(scip, solval1)) );
         }
         else
         {
            SCIP_CALL( SCIPchgVarLbProbing(scip, branchvar1, SCIPfeasCeil(scip, solval1)) );
         }
      }

      /* propagate the new b&b-node */
      SCIP_CALL( SCIPpropagateProbing(scip, -1, &cutoff, NULL) );

      /* solve the LP with or without pricing */
      if( !cutoff )
      {
         if( branchruledata->initiator == RYANFOSTER )
         {
            SCIP_CALL( newProbingNodeRyanfosterMaster(scip, branchruledata->initiatorbranchrule, branchvar1,
                                                      branchvar2, candinfo, cnode == 1) );
         }
         else
         {
            SCIP_CALL( GCGrelaxNewProbingnodeMaster(scip) );
         }

         if( pricing )
         {
            int npricerounds;

            SCIP_CALL( GCGrelaxPerformProbingWithPricing(scip, branchruledata->maxsbpricerounds, NULL, &npricerounds,
                                                         cnode == 0? down : up, &lpsolved, &lperror, &cutoff) );
            branchruledata->nphase2lps++;
            branchruledata->nsbpricerounds += npricerounds;
         }
         else
         {
            SCIP_Longint nlpiterations;

            SCIP_CALL( GCGrelaxPerformProbing(scip, branchruledata->maxsblpiters, &nlpiterations,
                                              cnode == 0? down : up, &lpsolved, &lperror, &cutoff) );
            branchruledata->nphase1lps++;
            branchruledata->nsblpiterations += nlpiterations;
         }
      }

      if( cnode == 0 )
      {
         *downvalid = lpsolved;
         *downinf = cutoff && pricing;
      }
      else
      {
         *upvalid = lpsolved;
         *upinf = cutoff && pricing;
      }

      SCIP_CALL( GCGrelaxEndProbing(scip) );
   }
   return SCIP_OKAY;
}

/* Returns true iff the second node is a k-successor of the to the first number corresponding node
 * (i.e. iff there are at most k edges between them)
 */
static
SCIP_Bool isKAncestor(
    SCIP*                scip,               /**< SCIP data structure */
    int                  ancestornodenr,     /**< number of the supposed ancestor */
    SCIP_NODE            *successornode,     /**< the supposed successor */
    int                  k                   /**< maximal allowed distance between the nodes */
)
{
   SCIP_NODE* curnode;
   curnode = successornode;

   for( int i = 0; i <= k && SCIPnodeGetNumber(curnode) >= ancestornodenr; i++ )
   {
      if( SCIPnodeGetNumber(curnode) == ancestornodenr )
         return TRUE;

      if( SCIPnodeGetNumber(curnode) == 1 )
         break;

      curnode = SCIPnodeGetParent(curnode);
   }

   return FALSE;
}

/* Evaluates the given variable based on a score function of choice. Higher scores are given to better variables. */
static
SCIP_Real score_function(
    SCIP                 *scip,              /**< SCIP data structure */
    SCIP_BRANCHRULE*     branchrule,         /**< pointer to the branching rule */
    SCIP_VAR             *var1,              /**< first var to be scored */
    SCIP_VAR             *var2,              /**< second var to be scored */
    SCIP_Real            solval1,            /**< the first var's current solution value */
    SCIP_Real            solval2,            /**< the second var's current solution value */
    int                  candinfo,           /**< additional integer information about the candidate */
    SCIP_Bool            useheuristic,       /**< should heuristics be used instead of strong branching? */
    SCIP_Bool            usehistorical,      /**< should historical data from phase 2 be used as heuristic? */
    SCIP_Bool            usecolgen,          /**< should column generation be used during strong branching? */
    SCIP_Real            *score,             /**< stores the computed score */
    SCIP_Bool            *upinf,             /**< stores whether the upbranch is infeasible */
    SCIP_Bool            *downinf            /**< stores whether the downbranch is infeasible */
)
{
   SCIP_BRANCHRULEDATA* branchruledata;
   VarTuple vartuple = {var1, var2, 0};

   branchruledata = SCIPbranchruleGetData(branchrule);
   assert(branchruledata != NULL);

   /* define score functions and calculate score for all variables for sorting dependent on used heuristic */
   /* phase 0 */
   if( useheuristic)
   {
      if( usehistorical )
      {
         int hashindex;

         assert(SCIPhashtableExists(branchruledata->candhashtable, (void *) &vartuple));
         hashindex = ((VarTuple *) SCIPhashtableRetrieve(branchruledata->candhashtable, (void *) &vartuple))->index;

         return branchruledata->strongbranchscore[hashindex];
      }
      else if( branchruledata->usepseudocosts )
      {
         *score = SCIPgetVarPseudocostScore(scip, var1, solval1);
         if( var2 != NULL )
            *score = *score * SCIPgetVarPseudocostScore(scip, var2, solval2);
      }
      else
      {
         if( !branchruledata->mostfrac )
            return 1;

         *score = solval1 - SCIPfloor(scip, solval1);
         *score = MIN( *score, 1.0 - *score );

         if( var2 != NULL )
         {
            SCIP_Real frac2;

            frac2 = solval2 - SCIPfloor(scip, solval2);
            *score = *score * MIN( frac2, 1.0 - frac2 );
         }
      }
   }
   else
   /* phases 1 & 2 */
   {
      SCIP* masterscip;

      int hashindex;
      int currentnodenr;

      SCIP_Real down;
      SCIP_Real up;
      SCIP_Real downgain;
      SCIP_Real upgain;
      SCIP_Bool upvalid;
      SCIP_Bool downvalid;
      SCIP_Real lpobjval;

      SCIP_Real frac;

      /* get master problem */
      masterscip = GCGgetMasterprob(scip);
      assert(masterscip != NULL);

      assert(SCIPhashtableExists(branchruledata->candhashtable, &vartuple));
      hashindex = ((VarTuple *) SCIPhashtableRetrieve(branchruledata->candhashtable, (void *) &vartuple))->index;
      currentnodenr = SCIPnodeGetNumber(SCIPgetFocusNode(scip));

      if( !usecolgen
          || !branchruledata->sbscoreisrecent[hashindex]
          || !isKAncestor(scip, branchruledata->lastevalnode[hashindex], SCIPgetFocusNode(scip),
                          branchruledata->reevalage) )
      {
         up = -SCIPinfinity(scip);
         down = -SCIPinfinity(scip);

         lpobjval = SCIPgetLPObjval(masterscip);

         /* usecolgen is True for phase 1 and False for phase 2 */
         SCIP_CALL( executeStrongBranching(scip, branchrule, var1, var2, solval1, solval2, candinfo, usecolgen, -1,
                                           &up, &down, &upvalid, &downvalid, upinf, downinf) );

         down = downvalid? down : upvalid? up : 0;
         up = upvalid? up : down;

         downgain = MAX(down - lpobjval, 0.0);
         upgain = MAX(up - lpobjval, 0.0);

         *score = SCIPgetBranchScore(scip, var1, downgain, upgain);

         if( usecolgen && upvalid && downvalid )
         {
            frac = solval1 - SCIPfloor(scip, solval1);
            if( !*upinf && !*downinf )
            {
               branchruledata->strongbranchscore[hashindex] = *score;
               branchruledata->sbscoreisrecent[hashindex] = TRUE;
               branchruledata->lastevalnode[hashindex] = currentnodenr;
            }

            if( branchruledata->initiator == ORIG )
            {
               /* update pseudocost scores */
               if( !*upinf )
               {
                  SCIP_CALL( SCIPupdateVarPseudocost(scip, var1, 1.0-frac, upgain, 1.0) );
               }

               if( !*downinf)
               {
                  SCIP_CALL( SCIPupdateVarPseudocost(scip, var1, 0.0-frac, downgain, 1.0) );
               }
            }
         }
      }
      else
      {
         *score = branchruledata->strongbranchscore[hashindex];
      }
   }

   return SCIP_OKAY;
}

/** branching method for relaxation solutions */
static
SCIP_RETCODE selectCandidate(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< pointer to the branching rule */
   SCIP_VAR**            cand1s,             /**< first variable candidates */
   SCIP_VAR**            cand2s,             /**< second variable candidates (each cand2 corresponds to exactly one
                                                * cand1 and vice versa) */
   int*                  candinfos,          /**< additional information for each candidate */
   int                   ncands,             /**< number of input candidates */
   SCIP_VAR**            outcand1,           /**< pointer to store the pointer of the first selected variable */
   SCIP_VAR**            outcand2,           /**< pointer to store the pointer of the second selected variable (if
                                                * applicable) */
   int*                  outcandinfo,        /**< pointer to store additional (integer) info */
   SCIP_Bool*            bestupinf,          /**< pointer to store whether strong branching detected infeasibility in
                                                * the upbranch */
   SCIP_Bool*            bestdowninf,        /**< pointer to store whether strong branching detected infeasibility in
                                                * the downbranch */
   SCIP_RESULT*          result              /**< pointer to store the result of the branching call */
   )
{
   SCIP* masterscip;
   SCIP_BRANCHRULEDATA* branchruledata;

   /* branching candidates */
   SCIP_VAR** branchcands;
   SCIP_Real* branchcandssol;
   int npriobranchcands;

   SCIP_HASHMAP* solhashmap;

   int hashindex;

   /* values for choosing the variable to branch on */
   SCIP_Real maxscore;
   SCIP_Real score;

   /* variables for controlling the evaluation effort */
   int lookahead;
   int lastimproved;

   int depth;
   int phase0nneededcands;

   SCIP_Real minpscount;

   SCIP_Real nodegap;
   SCIP_Real upperbound;
   SCIP_Real nodelowerbound;

   int nneededcands;

   int heurincumbentindex;
   SCIP_Real heurincumbentsbscore;

   /* infeasibility results during strong branching */
   SCIP_Bool upinf;
   SCIP_Bool downinf;

   /* storing best candidates */
   int *indices;
   int nvalidcands;

   /* storing best candidates based on historical strong branching scores */
   int *histindices;
   int nvalidhistcands;
   int nneededhistcands;

   VarTuple vartuple = {NULL,NULL,0};

   assert(branchrule != NULL);
   assert(strcmp(SCIPbranchruleGetName(branchrule), BRANCHRULE_NAME) == 0);
   assert(scip != NULL);
   assert(result != NULL);
   assert(SCIPisRelaxSolValid(scip));

   branchruledata = SCIPbranchruleGetData(branchrule);

   assert(branchruledata->maxphase1depth + 1 >= branchruledata->minphase0depth ||
          branchruledata->maxphase2depth + 1 >= branchruledata->minphase0depth);

   *result = SCIP_DIDNOTRUN;

   /* get master problem */
   masterscip = GCGgetMasterprob(scip);
   assert(masterscip != NULL);

   heurincumbentindex = -1;

   /* get the branching candidates */
   SCIP_CALL( SCIPgetExternBranchCands(scip, &branchcands, &branchcandssol, NULL, NULL,
         &npriobranchcands, NULL, NULL, NULL) );

   if( branchruledata->initiator == ORIG )
   {
      cand1s = branchcands;
      ncands = npriobranchcands;
   }
   else if( branchruledata->initiator == RYANFOSTER && (branchruledata->usepseudocosts || branchruledata->mostfrac ) )
   {
      SCIP_CALL( SCIPhashmapCreate(&solhashmap, SCIPblkmem(scip), npriobranchcands) );
      for( int r = 0; r<npriobranchcands; r++ )
      {
         SCIP_CALL( SCIPhashmapInsertReal(solhashmap, branchcands[r], branchcandssol[r]) );
      }
   }

   *outcand1 = NULL;

   maxscore = -1.0;

   upinf = FALSE;
   downinf = FALSE;
   *bestupinf = FALSE;
   *bestdowninf = FALSE;

   depth = SCIPnodeGetDepth(SCIPgetFocusNode(scip));

   /* set maximum strong branching lp iterations and pricing rounds to 2 times the average unless the value is
    * fixed in the settings (as it is done in SCIP)
    */
   SCIP_CALL( SCIPgetLongintParam(scip, "branching/bp_strong/maxsblpiters", &branchruledata->maxsblpiters) );
   if( branchruledata->maxsblpiters == 0 )
   {
      SCIP_Longint nlpiterations;
      SCIP_Longint nlps;
      SCIP_Longint maxlpiters;

      nlpiterations = branchruledata->nsblpiterations;
      nlps = branchruledata->nphase1lps;
      if( nlps == 0 )
      {
         nlpiterations = SCIPgetNNodeInitLPIterations(masterscip);
         nlps = SCIPgetNNodeInitLPs(masterscip);
         if( nlps == 0 )
         {
            nlpiterations = 1000;
            nlps = 1;
         }
      }
      assert(nlps >= 1);
      maxlpiters = (int)(2*nlpiterations / nlps);
      maxlpiters = (int)((SCIP_Real)maxlpiters * (1.0 + 10.0 / SCIPgetNNodes(masterscip)));
      branchruledata->maxsblpiters = maxlpiters;
   }

   SCIP_CALL( SCIPgetIntParam(scip, "branching/bp_strong/maxsbpricerounds", &branchruledata->maxsbpricerounds) );
   if( branchruledata->maxsbpricerounds == 0 )
   {
      SCIP_Longint npricerounds;
      SCIP_Longint nlps;
      SCIP_Longint maxpricerounds;

      npricerounds = branchruledata->nsbpricerounds;
      nlps = branchruledata->nphase2lps;
      if( nlps == 0 )
      {
         npricerounds = SCIPgetNNodeInitLPIterations(masterscip);
         nlps = SCIPgetNNodeInitLPs(masterscip);
         if( nlps == 0 )
         {
            npricerounds = 100000;
            nlps = 1;
         }
      }
      assert(nlps >= 1);
      maxpricerounds = (int)(2 * npricerounds / nlps);
      maxpricerounds = (int)((SCIP_Real)maxpricerounds * (1.0 + 10.0 / SCIPgetNNodes(masterscip)));
      branchruledata->maxsbpricerounds = maxpricerounds;
   }

   upperbound = SCIPgetUpperbound(scip);
   nodelowerbound = SCIPnodeGetLowerbound( SCIPgetFocusNode(scip) );
   nodegap = ((upperbound >= 0) == (nodelowerbound >= 0) && MIN( ABS( upperbound ), ABS( nodelowerbound ) ) != 0)?
             MIN( ABS( (upperbound-nodelowerbound) / MIN( ABS( upperbound ), ABS( nodelowerbound ) ) ), 1 ) : 1;
   assert(0 <= nodegap && nodegap <= 1);

   /* number of candidates we evaluate precisely should be based on the likely relevance of this branching decision
    * via the nodegap */
   nneededcands = calculateNCands(scip, branchruledata, nodegap, 0, ncands);

   /* insert branchcands into hashtable */
   SCIP_CALL( addBranchcandsToData(scip, branchrule, cand1s, cand2s, ncands) );

   SCIP_CALL( SCIPallocBufferArray(masterscip, &branchruledata->score, ncands) );
   for( int init = 0; init < ncands; ++init )
   {
      vartuple.var1 = cand1s[init];
      vartuple.var2 = cand2s == NULL? NULL : cand2s[init];
      hashindex = ((VarTuple *) SCIPhashtableRetrieve(branchruledata->candhashtable, (void *) &vartuple))->index;

      branchruledata->score[init] = branchruledata->strongbranchscore[hashindex];
   }

   /* allocate memory */
   SCIP_CALL( SCIPallocBufferArray(masterscip, &indices, ncands) );
   SCIP_CALL( SCIPallocBufferArray(masterscip, &histindices, ncands) );
   indices[0] = 0;

   if( branchruledata->initiator == ORIG )
   {
      nvalidcands = 0;
      nvalidhistcands = 0;

      /* iter = 0: integer variables belonging to a unique block with fractional value,
       * iter = 1: we did not find enough variables to branch on so far, so we look for integer variables that belong
       * to no block but were directly transferred to the master problem and which have a fractional value in the
       * current solution
       */
      for( int iter = 0; iter <= 1 && nvalidcands < nneededcands; iter++ )
      {
         for( int i = 0; i < ncands; i++ )
         {
            vartuple.var1 = cand1s[i];
            vartuple.var2 = NULL;

            hashindex = ((VarTuple *) SCIPhashtableRetrieve(branchruledata->candhashtable, (void *) &vartuple))->index;

            if (iter == 0)
            {
               if( branchruledata->uniqueblockflags[hashindex] < -1 )
               {
                  branchruledata->uniqueblockflags[hashindex] = assignUniqueBlockFlags(scip, cand1s[i]);
               }

               if( branchruledata->uniqueblockflags[hashindex] == 1 )
               {
                  indices[nvalidcands] = i;
                  nvalidcands++;

                  if( branchruledata->strongbranchscore[hashindex] != -1)
                  {
                     histindices[nvalidhistcands] = i;
                     nvalidhistcands++;
                  }
               }
            }
            else if( nvalidcands == 0 )
            {
               if( branchruledata->uniqueblockflags[hashindex] == 0 )
               {
                  indices[nvalidcands] = i;
                  nvalidcands++;
                  if( branchruledata->strongbranchscore[hashindex] != -1 )
                  {
                     histindices[nvalidhistcands] = i;
                     nvalidhistcands++;
                  }
               }
            }
         }
      }

      if( nvalidcands == 0 )
      {
         SCIPfreeBufferArray(masterscip, &indices);
         SCIPfreeBufferArray(masterscip, &histindices);
         SCIPfreeBufferArray(masterscip, &branchruledata->score);
         return SCIP_OKAY;
      }
   }
   else
   {
      nvalidhistcands = 0;
      for( int i=0; i<ncands; i++ )
      {
         indices[i] = i;
         if( branchruledata->score[i] != -1 )
         {
            histindices[nvalidhistcands] = i;
            nvalidhistcands++;
         }
      }
      nvalidcands = ncands;
   }

   /* the number of candidates we select based on historical strong branching scores needs to depend on the number of
    * candidates for which we have historical scores, otherwise some candidates would be selected simply because they
    * have been scored before
    */
   nneededhistcands = (int) SCIPfloor(scip, MIN( (SCIP_Real)nvalidhistcands / (SCIP_Real)(nvalidcands+nvalidhistcands),
                                                 branchruledata->histweight ) * nvalidcands);
   qsort(histindices, nvalidhistcands, sizeof(int), score_compare_function);
   qsort(histindices, nneededhistcands, sizeof(int), geq_compare_function);

   /* go through the three phases:
    * - phase 0: select a first selection of candidates based on some traditional variable selection
    *            heuristic, some (half) of the candidates are new, and some are selected based on previous calls
    * - phase 1: filter the best candidates by evaluating the Master LP, w/o column and cut generation
    * - phase 2: select the best of the candidates from phase 1 by solving the Master LP with column and cut generation
    */
   for( int phase = 0; phase <= 2; phase++ )
   {
      /* skip phase 1 if we are below its max depth */
      if( depth > branchruledata->maxphase1depth && phase == 1 )
         phase = 2;

      switch( phase )
      {
         case 0:
            ncands = nvalidcands;

            /* necessary in case we skip phase 0 */
            phase0nneededcands = nneededcands;

            /* skip phase 0 we are too high in the tree, and phases 1 and 2 if we are too low */
            if( branchruledata->minphase0depth > depth )
            {
               nneededcands = ncands;
            }
            else if( depth > branchruledata->maxphase1depth )
            {
               if( depth > branchruledata->maxphase2depth )
               {
                  nneededcands = 1;

                  /* strong branching can be fully stopped if all open nodes are below the max depth */
                  if( SCIPgetEffectiveRootDepth(scip) > branchruledata->maxphase1depth &&
                      SCIPgetEffectiveRootDepth(scip) > branchruledata->maxphase2depth )
                     branchruledata->done = TRUE;
               }
               else
               {
                  /* we only want to skip phase 1, so we need to set nneededcands to the number of output candidates
                   * for phase 1
                   */
                  nneededcands = calculateNCands(scip, branchruledata, nodegap, 1, phase0nneededcands);
               }
            }

            break;

         case 1:
            nneededcands = calculateNCands(scip, branchruledata, nodegap, 1, phase0nneededcands);

            /* skip phase 2 if we are in lite mode,
             * or if the number of available candidates is lower than the min amount for phase 2,
             * or if we are too low in the tree
             */
            if( branchruledata->usestronglite
               || nneededcands < branchruledata->mincolgencands
               || ncands < branchruledata->mincolgencands
               || depth > branchruledata->maxphase2depth )
               nneededcands = 1;

            break;

         case 2:
            nneededcands = 1;
            lastimproved = 0;

            /* the lookahead can be partially based on the overall evaluation effort for phase 2 */
            lookahead = branchruledata->maxlookahead;
            if( lookahead && branchruledata->lookaheadscales>0 )
            {
               lookahead = MAX( 1, (int) SCIPround(scip,
                                                  (SCIP_Real) ((1-branchruledata->lookaheadscales) * lookahead) -
                                                  (SCIP_Real) (branchruledata->lookaheadscales *
                                                   ncands / branchruledata->maxphase1outcands) * lookahead) );
            }
            break;
      }

      if( nneededcands >= ncands && (phase != 0 || !branchruledata->forcephase0) )
         continue;

      /* compute scores */
      for( int i = 0, c=branchruledata->lastcand; i < ncands; i++, c++ )
      {
         c = c % ncands;

         /* select the variable as new best candidate (if it is) if we look for only one candidate,
          * or remember its score if we look for multiple
          */
         if( branchruledata->initiator == ORIG )
         {
            minpscount = MIN( SCIPgetVarPseudocostCount(scip, cand1s[indices[c]], 0),
                              SCIPgetVarPseudocostCount(scip, cand1s[indices[c]], 1) );

            /* only call strong branching if this variable is not sufficiently reliable yet */
            if(  phase == 0 ||
                (phase == 1 && minpscount < branchruledata->phase1reliable) ||
                (phase == 2 && minpscount < branchruledata->phase2reliable)  )
            {
            SCIP_CALL( score_function(scip, branchrule, cand1s[indices[c]], NULL, branchcandssol[indices[c]], 0, 0,
                                      phase == 0, FALSE, phase == 2 && !branchruledata->usestronglite,
                                      &score, &upinf, &downinf) );
            }
            else
            {
               score = branchruledata->score[indices[c]];
            }
         }
         else
         {
            SCIP_CALL( score_function(scip, branchrule, cand1s[indices[c]], cand2s[indices[c]],
                                      SCIPhashmapGetImageReal(solhashmap, cand1s[indices[c]]),
                                      SCIPhashmapGetImageReal(solhashmap, cand2s[indices[c]]),
                                      candinfos[indices[c]], phase == 0, FALSE,
                                      phase == 2 && !branchruledata->usestronglite, &score, &upinf, &downinf) );
         }

         /* variable pointers for orig candidates sometimes change during probing in strong branching */
         if( branchruledata->initiator == ORIG && phase >= 1 )
         {
            SCIP_CALL( SCIPgetExternBranchCands(scip, &cand1s, &branchcandssol, NULL, NULL,
               NULL, NULL, NULL, NULL) );
         }

         /* handle infeasibility detected during strong branching */
         if( phase == 2 && !branchruledata->usestronglite && branchruledata->immediateinf && (upinf || downinf) )
         {
            if( upinf && downinf )
            {
               for( int k = 0; k < branchruledata->maxvars; k++ )
               {
                  branchruledata->sbscoreisrecent[k] = FALSE;
               }
               *result = SCIP_CUTOFF;

               SCIPfreeBufferArray(masterscip, &indices);
               SCIPfreeBufferArray(masterscip, &histindices);
               SCIPfreeBufferArray(masterscip, &branchruledata->score);

               *bestupinf = TRUE;
               *bestdowninf = TRUE;

               SCIPdebugMessage("Strong branching detected current node to be infeasible!\n");
               return SCIP_OKAY;
            }

            branchruledata->lastcand = c;
            indices[0] = indices[c];
            *bestupinf = upinf;
            *bestdowninf = downinf;
            break;
         }

         /* store strong branching score of the candidate that was selected by the heuristic */
         if( phase>0 && heurincumbentindex == indices[c] )
               heurincumbentsbscore = score;

         if( nneededcands == 1 )
         {
            if( score > maxscore )
            {
               lastimproved = 0;
               indices[0] = indices[c];
               maxscore = score;
               *bestupinf = upinf;
               *bestdowninf = downinf;
            }
            /* if the last improving candidate is lookahead or more steps away, abort phase 2 */
            else
            {
               lastimproved++;
               if( lookahead && lastimproved >= lookahead && phase == 2 )
               {
                  break;
               }
            }

         }
         else
         {
            branchruledata->score[indices[c]] = score;
         }
      }

      if( nneededcands > 1 )
      {
         qsort(indices, ncands, sizeof(int), score_compare_function);
         ncands = MIN( ncands, nneededcands );

         if( phase == 0 )
         {
            heurincumbentindex = indices[0];

            if( nneededhistcands )
            {
               /* swap out the worst performing "new" candidates with the best performing historical candidates */
               int *indicescopy;
               int pos;

               SCIP_CALL( SCIPallocBufferArray(masterscip, &indicescopy, ncands) );
               pos = nneededhistcands;

               for( int i = 0; i<ncands; i++ )
               {
                  indicescopy[i] = indices[i];
               }

               for( int i = 0; i<nneededhistcands; i++ )
               {
                  indices[i] = histindices[i];
               }

               /* concatenate the two arrays, while avoiding duplicates */
               for( int i = 0; i<ncands && pos<=ncands; i++ )
               {
                  for( int j = 0; j <= nneededhistcands; j++ )
                  {
                     if( j == nneededhistcands )
                     {
                        indices[pos] = indicescopy[i];
                        pos++;
                     }
                     else if( indices[j] == indicescopy[i] )
                        break;
                  }
               }

               SCIPfreeBufferArray(masterscip, &indicescopy);
            }
         }
      }
      else
      {
         break;
      }
   }

   *outcand1 = cand1s[indices[0]];
   if( branchruledata->initiator == RYANFOSTER )
   {
      *outcand2 = cand2s[indices[0]];
      *outcandinfo = candinfos[indices[0]];
   }

   /* free memory */
   SCIPfreeBufferArray(masterscip, &indices);
   SCIPfreeBufferArray(masterscip, &histindices);
   SCIPfreeBufferArray(masterscip, &branchruledata->score);

   if( *outcand1 == NULL )
   {
      SCIPdebugMessage("Strong branching could not find a variable to branch on!\n");
      return SCIP_OKAY;
   }

   if( branchruledata->initiator == ORIG )
   {
      SCIPdebugMessage("Strong branching selected variable %s%s\n",
                       SCIPvarGetName(*outcand1),
                       (*bestupinf || *bestdowninf)? ", branching on which is infeasible in one direction" : "");
   }
   else
   {
      SCIPdebugMessage("Strong branching selected variables %s and %s%s\n",
                       SCIPvarGetName(*outcand1), SCIPvarGetName(*outcand2),
                       (*bestupinf || *bestdowninf)? ", branching on which is infeasible in one direction" : "");
   }

   if( branchruledata->initiator == RYANFOSTER && (branchruledata->usepseudocosts || branchruledata->mostfrac) )
   {
      SCIPhashmapFree(&solhashmap);
   }

   if( !*bestupinf && !*bestdowninf )
   {
      /* if the heuristic was close multiple times in a row, stop strong branching */
      if( branchruledata->maxconsecheurclose >= 0
          && heurincumbentsbscore >= branchruledata->closepercentage * maxscore )
      {
         branchruledata->consecheurclose++;
         if( branchruledata->consecheurclose >= branchruledata->maxconsecheurclose )
            branchruledata->done = TRUE;
      }
      else
         branchruledata->consecheurclose = 0;

      for( int i=0; i<branchruledata->maxvars; i++ )
      {
         branchruledata->sbscoreisrecent[i] = FALSE;
      }
   }

   *result = SCIP_BRANCHED;

   return SCIP_OKAY;
}

/*
 * Callback methods
 */
#define branchDeactiveMasterBPStrong NULL
#define branchPropMasterBPStrong NULL
#define branchActiveMasterBPStrong NULL
#define branchMasterSolvedBPStrong NULL
#define branchDataDeleteBPStrong NULL

#define branchExeclpBPStrong NULL
#define branchExecextBPStrong NULL
#define branchExecpsBPStrong NULL

/** free remaining allocated memory */
static
SCIP_DECL_BRANCHFREE(branchFreeBPStrong)
{
   SCIP_BRANCHRULEDATA* branchruledata;
   SCIP_HASHTABLE* candhashtable;
   int i;

   branchruledata = SCIPbranchruleGetData(branchrule);

   if( branchruledata->initialized )
   {
      candhashtable = branchruledata->candhashtable;

      SCIPfreeBlockMemoryArray(scip, &branchruledata->lastevalnode, branchruledata->maxvars);
      SCIPfreeBlockMemoryArray(scip, &branchruledata->sbscoreisrecent, branchruledata->maxvars);
      SCIPfreeBlockMemoryArray(scip, &branchruledata->strongbranchscore, branchruledata->maxvars);
      SCIPfreeBlockMemoryArray(scip, &branchruledata->uniqueblockflags, branchruledata->maxvars);

      for( i=0; i<branchruledata->nvars; i++ )
      {
          SCIPfreeBlockMemory(scip, &(branchruledata->vartuples[i]));
      }

      SCIPfreeBlockMemoryArray(scip, &branchruledata->vartuples, branchruledata->maxvars);

      if( branchruledata->candhashtable != NULL )
      {
         SCIPhashtableFree(&candhashtable);
      }
   }

   SCIPfreeBlockMemory(scip, &branchruledata);
   SCIPbranchruleSetData(branchrule, NULL);

   return SCIP_OKAY;
}

/** initialization method of branching rule (called after problem was transformed) */
static
SCIP_DECL_BRANCHINIT(branchInitBPStrong)
{
   SCIP* origprob;
   SCIP_BRANCHRULEDATA* branchruledata;
   SCIP_HASHTABLE* candhashtable;

   int i;
   int maxcands;

   SCIP_Real logweight;
   SCIP_Real logbase;
   SCIP_Real phase0frac;
   SCIP_Real phase2frac;

   SCIP_Real phase1depth;

   origprob = GCGmasterGetOrigprob(scip);
   assert(branchrule != NULL);
   assert(origprob != NULL);

   SCIPdebugMessage("Init BPStrong branching rule\n");

   SCIP_CALL( GCGrelaxIncludeBranchrule( origprob, branchrule, branchActiveMasterBPStrong,
         branchDeactiveMasterBPStrong, branchPropMasterBPStrong, branchMasterSolvedBPStrong,
         branchDataDeleteBPStrong) );

   branchruledata = SCIPbranchruleGetData(branchrule);

   /* free data if we already solved another instance but branchFreeBPStrong was not called inbetween */
   if( branchruledata->initialized )
   {
      candhashtable = branchruledata->candhashtable;

      for( i=0; i<branchruledata->nvars; i++ )
      {
         SCIPfreeBlockMemory(scip, &branchruledata->vartuples[i]);
      }

      SCIPfreeBlockMemoryArray(scip, &branchruledata->vartuples, branchruledata->maxvars);
      SCIPfreeBlockMemoryArray(scip, &branchruledata->lastevalnode, branchruledata->maxvars);
      SCIPfreeBlockMemoryArray(scip, &branchruledata->sbscoreisrecent, branchruledata->maxvars);
      SCIPfreeBlockMemoryArray(scip, &branchruledata->strongbranchscore, branchruledata->maxvars);
      SCIPfreeBlockMemoryArray(scip, &branchruledata->uniqueblockflags, branchruledata->maxvars);

      if( branchruledata->candhashtable != NULL )
      {
         SCIPhashtableFree(&candhashtable);
      }
   }

   branchruledata->lastcand = 0;
   branchruledata->nvars = 0;
   branchruledata->maxvars = 0;
   branchruledata->initiator = -1;

   branchruledata->nphase1lps = 0;
   branchruledata->nphase2lps = 0;
   branchruledata->nsblpiterations = 0;
   branchruledata->nsbpricerounds = 0;

   branchruledata->consecheurclose = 0;
   branchruledata->done = FALSE;

   SCIP_CALL( SCIPgetRealParam(origprob, "branching/bp_strong/depthlogweight", &logweight) );
   if( logweight > 0 )
   {
      SCIP_CALL( SCIPgetRealParam(origprob, "branching/bp_strong/depthlogbase", &logbase) );
      SCIP_CALL( SCIPgetRealParam(origprob, "branching/bp_strong/depthlogphase0frac", &phase0frac) );
      SCIP_CALL( SCIPgetRealParam(origprob, "branching/bp_strong/depthlogphase2frac", &phase2frac) );
      branchruledata->maxcands = SCIPgetNIntVars(origprob) + SCIPgetNBinVars(origprob);

      phase1depth = log(branchruledata->maxcands)/log(logbase);

      branchruledata->minphase0depth = (int) SCIPround(origprob, (1-logweight) * branchruledata->minphase0depth
                                                                 + logweight * phase0frac * phase1depth);
      branchruledata->maxphase1depth = (int) SCIPround(origprob, (1-logweight) * branchruledata->maxphase1depth
                                                                 + logweight * phase1depth);
      branchruledata->maxphase2depth = (int) SCIPround(origprob, (1-logweight) * branchruledata->maxphase2depth
                                                                 + logweight * phase2frac * phase1depth);
   }

   /* create hash table TODO: better inital table sizes */
   maxcands = branchruledata->maxcands;
   SCIP_CALL( SCIPhashtableCreate(&(branchruledata->candhashtable), SCIPblkmem(scip),
                              branchruledata->initiator == RYANFOSTER? maxcands*2 : maxcands,
                              hashGetKeyVars, hashKeyEqVars, hashKeyValVars, (void*) scip) );

   assert(branchruledata->candhashtable != NULL);

   /* create arrays */
   branchruledata->nvars = 0;
   branchruledata->maxvars = SCIPcalcMemGrowSize(scip, 0);
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &branchruledata->uniqueblockflags, branchruledata->maxvars) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &branchruledata->strongbranchscore, branchruledata->maxvars) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &branchruledata->sbscoreisrecent, branchruledata->maxvars) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &branchruledata->lastevalnode, branchruledata->maxvars) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &branchruledata->vartuples, branchruledata->maxvars) );

   branchruledata->initialized = TRUE;

   this_branchruledata = branchruledata;

   return SCIP_OKAY;
}

/** creates the b&p strong-branching branching rule and includes it in SCIP */
SCIP_RETCODE SCIPincludeBranchruleBPStrong(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP* origscip;
   SCIP_BRANCHRULE* branchrule;
   SCIP_BRANCHRULEDATA* branchruledata;

   SCIPdebugMessage("Include BPStrong branching rule\n");

   /* get original problem */
   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   /* alloc branching rule data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &branchruledata) );

   /* include branching rule */
   SCIP_CALL( SCIPincludeBranchruleBasic(scip, &branchrule, BRANCHRULE_NAME, BRANCHRULE_DESC, BRANCHRULE_PRIORITY,
            BRANCHRULE_MAXDEPTH, BRANCHRULE_MAXBOUNDDIST, branchruledata) );
   assert(branchrule != NULL);

   branchruledata->initialized = FALSE;

   /* set non fundamental callbacks via setter functions */
   SCIP_CALL( SCIPsetBranchruleInit(scip, branchrule, branchInitBPStrong) );
   SCIP_CALL( SCIPsetBranchruleFree(scip, branchrule, branchFreeBPStrong) );

   /* add branching rule parameters */
   SCIP_CALL( SCIPaddBoolParam(origscip, "branching/bp_strong/stronglite",
         "should strong branching use column generation during variable evaluation?",
         &branchruledata->usestronglite, FALSE, DEFAULT_STRONGLITE, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(origscip, "branching/bp_strong/strongtraining",
         "should strong branching run as precise as possible (to generate more valuable training data)?",
         &branchruledata->usestrongtrain, FALSE, DEFAULT_STRONGTRAIN, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(origscip, "branching/bp_strong/immediateinf",
         "should infeasibility detected during strong branching be handled immediately, or only if the candidate is selected?",
         &branchruledata->immediateinf, FALSE, DEFAULT_IMMEDIATEINF, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(origscip, "branching/bp_strong/reevalage",
         "how many times can bounds be changed due to infeasibility during strong branching until an already evaluated variable needs to be reevaluated?",
         &branchruledata->reevalage, FALSE, DEFAULT_REEVALAGE, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(origscip, "branching/bp_strong/mincolgencands",
         "minimum number of variables for phase 2 to be executed, otherwise the best candidate from phase 1 will be chosen",
         &branchruledata->mincolgencands, FALSE, DEFAULT_MINCOLGENCANDS, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origscip, "branching/bp_strong/histweight",
         "how many candidates should be chosen based on historical strong branching scores as opposed to current heuristic scores in phase 0 (e.g. 0.5 = 50%)?",
         &branchruledata->histweight, FALSE, DEFAULT_HISTWEIGHT, 0, 1, NULL, NULL) );

   SCIP_CALL( SCIPaddLongintParam(origscip, "branching/bp_strong/maxsblpiters",
         "maximum number of strong branching lp iterations, set to 2*avg lp iterations if <= 0",
         NULL, FALSE, DEFAULT_MAXSBLPITERS, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(origscip, "branching/bp_strong/maxsbpricerounds",
         "maximum number of strong branching price rounds, set to 2*avg lp iterations if <= 0",
         NULL, FALSE, DEFAULT_MAXSBPRICEROUNDS, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(origscip, "branching/bp_strong/maxlookahead",
         "maximum number of non-improving candidates until phase 2 is stopped",
         &branchruledata->maxlookahead, FALSE, DEFAULT_MAXLOOKAHEAD, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origscip, "branching/bp_strong/lookaheadscales",
         "how much should the lookahead scale with the overall evaluation effort? (0 = not at all, 1 = fully)",
         &branchruledata->lookaheadscales, FALSE, DEFAULT_LOOKAHEADSCALES, 0, 1, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(origscip, "branching/bp_strong/minphase0depth",
         "minimum tree depth from which on phase 0 is performed (intended for heuristics like pseudocost branching)",
         &branchruledata->minphase0depth, FALSE, DEFAULT_MINPHASE0DEPTH, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(origscip, "branching/bp_strong/maxphase1depth",
         "maximum tree depth up to which phase 1 is performed (intended for heuristics like pseudocost branching)",
         &branchruledata->maxphase1depth, FALSE, DEFAULT_MAXPHASE1DEPTH, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(origscip, "branching/bp_strong/maxphase2depth",
         "maximum tree depth up to which phase 2 is performed (intended for heuristics like pseudocost branching)",
         &branchruledata->maxphase2depth, FALSE, DEFAULT_MAXPHASE2DEPTH, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origscip, "branching/bp_strong/depthlogweight",
         "how much should the logarithm of the number of variables influence the depth for hybrid branching? (0 = not at all, 1 = fully)",
         NULL, FALSE, DEFAULT_DEPTHLOGWEIGHT, 0, 1, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origscip, "branching/bp_strong/depthlogbase",
         "what should be the base of the logarithm that is used to compute the depth of hybrid branching?",
         NULL, FALSE, DEFAULT_DEPTHLOGBASE, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origscip, "branching/bp_strong/depthlogphase0frac",
         "if using a logarithm to compute the depth of hybrid branching, what should be the fraction of the depth assigned to phase 1 that is assigned to phase 0?",
         NULL, FALSE, DEFAULT_DEPTHLOGPHASE0FRAC, 0, 1, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origscip, "branching/bp_strong/depthlogphase2frac",
         "if using a logarithm to compute the depth of hybrid branching, what should be the fraction of the depth assigned to phase 1 that is assigned to phase 2?",
         NULL, FALSE, DEFAULT_DEPTHLOGPHASE2FRAC, 0, 1, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origscip, "branching/bp_strong/closepercentage",
         "what percentage of the strong branching score of the candidate that was selected does the heuristic's incumbent need to be considered close (e.g. 0.5 = 50%)?",
         &branchruledata->closepercentage, FALSE, DEFAULT_CLOSEPERCENTAGE, 0, 1, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(origscip, "branching/bp_strong/maxconsecheurclose",
         "how many times in a row can the heuristic be close before strong branching is stopped?",
         &branchruledata->maxconsecheurclose, FALSE, DEFAULT_MAXCONSECHEURCLOSE, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origscip, "branching/bp_strong/sbpseudocostweight",
         "with how much weight should strong branching scores be considered for pseudocost scores?",
         &branchruledata->sbpseudocostweight, FALSE, DEFAULT_SBPSEUDOCOSTWEIGHT, 0, 1, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(origscip, "branching/bp_strong/phase1reliable",
         "min count of pseudocost scores for a variable to be considered reliable in phase 1",
         &branchruledata->phase1reliable, FALSE, DEFAULT_PHASE1RELIABLE, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(origscip, "branching/bp_strong/phase2reliable",
         "min count of pseudocost scores for a variable to be considered reliable in phase 2",
         &branchruledata->phase2reliable, FALSE, DEFAULT_PHASE2RELIABLE, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(origscip, "branching/bp_strong/forcep0",
         "should phase 0 be performed even if the number of input candidates is already lower or equal to the number of output candidates?",
         &branchruledata->forcephase0, FALSE, DEFAULT_FORCEP0, NULL, NULL) );


   SCIP_CALL( SCIPaddBoolParam(origscip, "branching/bp_strong/ryanfoster/usepseudocosts",
         "should single-variable-pseudocosts be used as a heuristic for strong branching for Ryan-Foster branching?",
         NULL, FALSE, DEFAULT_RFUSEPSEUDOCOSTS, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(origscip, "branching/bp_strong/ryanfoster/usemostfrac",
         "should single-variable-fractionality be used as a heuristic for strong branching for Ryan-Foster branching?",
         NULL, FALSE, DEFAULT_RFUSEMOSTFRAC, NULL, NULL) );


   /* notify cons_integralorig about the branching rule */
   SCIP_CALL( GCGconsIntegralorigAddBranchrule(scip, branchrule) );

   return SCIP_OKAY;
}

SCIP_RETCODE
GCGbranchSelectCandidateStrongBranchingOrig(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE       *origbranchrule,    /**< pointer storing original branching rule */
   SCIP_VAR              **branchvar,        /**< pointer to store output var pointer */
   SCIP_Bool             *upinf,             /**< pointer to store whether strong branching detected infeasibility in
                                                * the upbranch */
   SCIP_Bool             *downinf,           /**< pointer to store whether strong branching detected infeasibility in
                                                * the downbranch */
   SCIP_RESULT           *result,            /**< pointer to store result */
   SCIP_Bool             *stillusestrong     /**< pointer to store whether strong branching has reached a permanent
                                                * stopping condition for orig */
)
{
   SCIP_BRANCHRULEDATA *branchruledata;
   SCIP_BRANCHRULE *branchrule;
   SCIP* masterscip;

   masterscip = GCGgetMasterprob(scip);
   branchrule = SCIPfindBranchrule(masterscip, BRANCHRULE_NAME);
   assert(branchrule != NULL);

   branchruledata = SCIPbranchruleGetData(branchrule);

   if( branchruledata->initiator != ORIG )
   {
      branchruledata->initiator = ORIG;

      SCIP_CALL( SCIPgetBoolParam(scip, "branching/orig/usepseudocosts", &branchruledata->usepseudocosts) );
      SCIP_CALL( SCIPgetBoolParam(scip, "branching/orig/mostfrac", &branchruledata->mostfrac) );

      SCIP_CALL( SCIPgetIntParam(scip, "branching/orig/minphase0outcands", &branchruledata->minphase0outcands) );
      SCIP_CALL( SCIPgetIntParam(scip, "branching/orig/maxphase0outcands", &branchruledata->maxphase0outcands) );
      SCIP_CALL( SCIPgetRealParam(scip, "branching/orig/maxphase0outcandsfrac",
                                  &branchruledata->maxphase0outcandsfrac) );
      SCIP_CALL( SCIPgetRealParam(scip, "branching/orig/phase1gapweight", &branchruledata->phase1gapweight) );

      SCIP_CALL( SCIPgetIntParam(scip, "branching/orig/minphase1outcands", &branchruledata->minphase1outcands) );
      SCIP_CALL( SCIPgetIntParam(scip, "branching/orig/maxphase1outcands", &branchruledata->maxphase1outcands) );
      SCIP_CALL( SCIPgetRealParam(scip, "branching/orig/maxphase1outcandsfrac",
                                  &branchruledata->maxphase1outcandsfrac) );
      SCIP_CALL( SCIPgetRealParam(scip, "branching/orig/phase2gapweight", &branchruledata->phase2gapweight) );

      assert(branchruledata->maxphase0outcands >= branchruledata->minphase0outcands);
      assert(branchruledata->maxphase1outcands >= branchruledata->minphase1outcands);
   }

   selectCandidate(scip, branchrule, NULL, NULL, NULL, 0, branchvar, NULL, NULL, upinf, downinf, result);

   *stillusestrong = !branchruledata->done;

   return SCIP_OKAY;
}

/** interface method for Ryan-Foster branching to strong branching heuristics */
SCIP_RETCODE
GCGbranchSelectCandidateStrongBranchingRyanfoster(
   SCIP*                 scip,               /**< original SCIP data structure */
   SCIP_BRANCHRULE*      rfbranchrule,       /**< Ryan-Foster branchrule */
   SCIP_VAR              **ovar1s,           /**< first elements of candidate pairs */
   SCIP_VAR              **ovar2s,           /**< second elements of candidate pairs */
   int                   *nspricingblock,    /**< pricing block numbers corresponding to input pairs */
   int                   npairs,             /**< number of input pairs */
   SCIP_VAR              **ovar1,            /**< pointer to store output var 1 pointer */
   SCIP_VAR              **ovar2,            /**< pointer to store output var 2 pointer */
   int                   *pricingblock,      /**< pointer to store output pricing block number */
   SCIP_Bool             *sameinf,           /**< pointer to store whether strong branching detected infeasibility in
                                                * the same branch */
   SCIP_Bool             *differinf,         /**< pointer to store whether strong branching detected infeasibility in
                                                * the differ branch */
   SCIP_RESULT           *result,            /**< pointer to store result */
   SCIP_Bool             *stillusestrong     /**< pointer to store whether strong branching has reached a permanent
                                                * stopping condition for Ryan-Foster */
)
{
   SCIP_BRANCHRULEDATA *branchruledata;
   SCIP_BRANCHRULE *branchrule;
   SCIP* masterscip;

   masterscip = GCGgetMasterprob(scip);
   branchrule = SCIPfindBranchrule(masterscip, BRANCHRULE_NAME);
   assert(branchrule != NULL);

   branchruledata = SCIPbranchruleGetData(branchrule);

   if( branchruledata->initiator != RYANFOSTER )
   {
      branchruledata->initiator = RYANFOSTER;
      branchruledata->initiatorbranchrule = rfbranchrule;
      SCIP_CALL( SCIPgetBoolParam(scip, "branching/bp_strong/ryanfoster/usepseudocosts",
                                  &branchruledata->usepseudocosts) );
      SCIP_CALL( SCIPgetBoolParam(scip, "branching/bp_strong/ryanfoster/usemostfrac", &branchruledata->mostfrac) );

      SCIP_CALL( SCIPgetIntParam(scip, "branching/ryanfoster/minphase0outcands",
                                 &branchruledata->minphase0outcands) );
      SCIP_CALL( SCIPgetIntParam(scip, "branching/ryanfoster/maxphase0outcands",
                                 &branchruledata->maxphase0outcands) );
      SCIP_CALL( SCIPgetRealParam(scip, "branching/ryanfoster/maxphase0outcandsfrac",
                                  &branchruledata->maxphase0outcandsfrac) );
      SCIP_CALL( SCIPgetRealParam(scip, "branching/ryanfoster/phase1gapweight",
                                  &branchruledata->phase1gapweight) );

      SCIP_CALL( SCIPgetIntParam(scip, "branching/ryanfoster/minphase1outcands",
                                 &branchruledata->minphase1outcands) );
      SCIP_CALL( SCIPgetIntParam(scip, "branching/ryanfoster/maxphase1outcands",
                                 &branchruledata->maxphase1outcands) );
      SCIP_CALL( SCIPgetRealParam(scip, "branching/ryanfoster/maxphase1outcandsfrac",
                                  &branchruledata->maxphase1outcandsfrac) );
      SCIP_CALL( SCIPgetRealParam(scip, "branching/ryanfoster/phase2gapweight",
                                  &branchruledata->phase2gapweight) );

      assert(branchruledata->maxphase0outcands >= branchruledata->minphase0outcands);
      assert(branchruledata->maxphase1outcands >= branchruledata->minphase1outcands);
   }

   selectCandidate(scip, branchrule, ovar1s, ovar2s, nspricingblock, npairs,
                   ovar1, ovar2, pricingblock, sameinf, differinf, result);

   *stillusestrong = !branchruledata->done;

   assert(*ovar1 != NULL);
   assert(*ovar2 != NULL);

   return SCIP_OKAY;
}