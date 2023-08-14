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

/**@file   dec_isomorph.cpp
 * @ingroup DETECTORS
 * @brief  detector for pricing problems that can be aggregated (uses bliss)
 * @author Martin Bergner
 * @author Daniel Peters
 * @author Jonas Witt
 * @author Michael Bastubbe
 *
 * @note requires package to be installed: BLISS, requires flag to be set: `BLISS=true`
 *
 * This detector finds subproblems that can be aggregated thus reducing the symmetry of the problem using color preserving
 * automorphisms and bliss.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
/* #define SCIP_DEBUG */

#include "dec_isomorph.h"
#include "pub_decomp.h"
#include "cons_decomp.h"
#include "scip_misc.h"
#include "gcg.h"
#include "class_partialdecomp.h"
#include "class_detprobdata.h"
#include "scip/clock.h"

#include "bliss/graph.hh"
#include "pub_gcgvar.h"
#include <cstring>
#include <cassert>
#include <algorithm>
#include <iostream>

#include "pub_bliss.h"

/* constraint handler properties */
#define DEC_DETECTORNAME          "isomorph"  /**< name of detector */
#define DEC_DESC                  "detector for pricing problems suitable for aggregation" /**< description of detector */
#define DEC_FREQCALLROUND         1           /**< frequency the detector gets called in detection loop ,ie it is called in round r if and only if minCallRound <= r <= maxCallRound AND  (r - minCallRound) mod freqCallRound == 0 */
#define DEC_MAXCALLROUND          0           /**< last round the detector gets called */
#define DEC_MINCALLROUND          0           /**< first round the detector gets called */
#define DEC_FREQCALLROUNDORIGINAL 1           /**< frequency the detector gets called in detection loop while detecting the original problem */
#define DEC_MAXCALLROUNDORIGINAL  0           /**< last round the detector gets called while detecting the original problem */
#define DEC_MINCALLROUNDORIGINAL  0           /**< first round the detector gets called while detecting the original problem */
#define DEC_PRIORITY              100         /**< priority of the constraint handler for separation */
#define DEC_DECCHAR               'I'         /**< display character of detector */

#define DEC_ENABLED               FALSE       /**< should the detection be enabled */
#define DEC_ENABLEDFINISHING      FALSE       /**< should the finishing be enabled */
#define DEC_ENABLEDPOSTPROCESSING FALSE       /**< should the postprocessing be enabled */
#define DEC_SKIP                  TRUE        /**< should the detector be skipped if others found decompositions */
#define DEC_USEFULRECALL          FALSE       /**< is it useful to call this detector on a descendant of the propagated partialdec */

#define DEFAULT_MAXDECOMPSEXACT  6            /**< default maximum number of decompositions */
#define DEFAULT_MAXDECOMPSEXTEND 4            /**< default maximum number of decompositions */

#define SET_MULTIPLEFORSIZETRANSF 12500

/*
 * Data structures
 */

/** detector data */
struct DEC_DetectorData
{
   SCIP_RESULT          result;            /**< result pointer to indicate success or failure */
   int                  maxdecompsexact;   /**< maximum number of decompositions for exact emthod */
   int                  maxdecompsextend;  /**< maximum number of decompositions for extend method*/
};

typedef struct struct_hook AUT_HOOK;

/** saves information of the permutation */
struct struct_hook
{
   SCIP_Bool aut;                            /**< true if there is an automorphism */
   unsigned int n;                           /**< number of permutations */
   SCIP* scip;                               /**< scip to search for automorphisms */
   int* conssperm;                           /**< permutations of conss*/
   gcg::PARTIALDECOMP* partialdec;           /**< partialdec to propagate */
   gcg::DETPROBDATA* detprobdata;            /**< detection process information and data */

   /** constructor for the hook struct*/
   struct_hook(SCIP_Bool aut,  /**< true if there is an automorphism */
      unsigned int       n,                  /**< number of permutations */
      SCIP*              scip                /**< scip to search for automorphisms */
   );

   /** constructor for the hook struct with a partialdec */
   struct_hook(SCIP_Bool aut,  /**< true if there is an automorphism */
      unsigned int          n,                  /**< number of permutations */
      SCIP*                 scip,               /**< scip to search for automorphisms */
      gcg::PARTIALDECOMP*   partialdec,         /**< partial decomposition */
      gcg::DETPROBDATA*     detprobdata         /**< detection process information and data */
   );

   ~struct_hook();
   /** getter for the bool aut */
   SCIP_Bool getBool();

   /** setter for the bool aut */
   void setBool(SCIP_Bool aut);

   /** getter for the SCIP */
   SCIP* getScip();

   /** getter for the partialdec */
   gcg::PARTIALDECOMP* getPartialdec();

   /** getter for the detprobdata */
   gcg::DETPROBDATA* getDetprobdata();
};

SCIP* struct_hook::getScip()
{
   return this->scip;
}

gcg::PARTIALDECOMP* struct_hook::getPartialdec()
{
   return this->partialdec;
}

gcg::DETPROBDATA* struct_hook::getDetprobdata()
{
   return this->detprobdata;
}

SCIP_Bool struct_hook::getBool()
{
   return aut;
}

void struct_hook::setBool( SCIP_Bool aut_ )
{
   aut = aut_;
}




/** method to calculate the greatest common divisor */
static
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

/** constructor of the hook struct */
struct_hook::struct_hook(
   SCIP_Bool             aut_,               /**< true if there is an automorphism */
   unsigned int          n_,                 /**< number of permutations */
   SCIP*                 scip_               /**< array of scips to search for automorphisms */
   ) : conssperm(NULL)
{
   aut = aut_;
   n = n_;
   scip = scip_;
   SCIP_CALL_ABORT( SCIPallocMemoryArray(scip, &conssperm, SCIPgetNConss(scip)) ); /*lint !e666*/
   partialdec = NULL;
   detprobdata = NULL;
}

/** constructor of the hook struct */
struct_hook::struct_hook(
   SCIP_Bool             aut_,               /**< true if there is an automorphism */
   unsigned int          n_,                 /**< number of permutations */
   SCIP*                 scip_,              /**< array of scips to search for automorphisms */
   gcg::PARTIALDECOMP*   partialdec_,        /**< partialdec to propagate */
   gcg::DETPROBDATA*     detprobdata_        /**< detection process information and data */
   ) : conssperm(NULL)
{
   aut = aut_;
   n = n_;
   scip = scip_;
   SCIP_CALL_ABORT( SCIPallocMemoryArray(scip, &conssperm, detprobdata_->getNConss() ) ); /*lint !e666*/
   partialdec = partialdec_;
   detprobdata = detprobdata_;
}

struct_hook::~struct_hook()
{   /*lint -esym(1540,struct_hook::conssperm) */
   if( conssperm != NULL )
      SCIPfreeMemoryArrayNull(scip, &conssperm);
   conssperm  = NULL;
   scip = NULL;
}

/** hook function to save the permutation of the graph */
static
void fhookForPartialdecs(
   void*                 user_param,         /**< data structure to save hashmaps with permutation */
   unsigned int          N,                  /**< number of permutations */
   const unsigned int*   aut                 /**< array of permutations */
   )
{ /*lint -e715*/
   int i;
   int nconss;
   AUT_HOOK* hook = (AUT_HOOK*) user_param;
   int auti;
   int ind;
   gcg::PARTIALDECOMP*  partialdec;
   gcg::DETPROBDATA*       detprobdata;

   partialdec = hook->getPartialdec();
   detprobdata = hook->getDetprobdata() ;
   assert(partialdec != NULL);
   assert(detprobdata != NULL);
   nconss = partialdec->getNOpenconss();

   for( i = 0; i < nconss; i++ )
   {
      SCIP_CONS* cons = detprobdata->getCons(partialdec->getOpenconss()[i]);
      assert(aut[i] < INT_MAX);
      if( (size_t) i != aut[i])
      {
         auti = (int) aut[i];

         SCIPdebugMessage("%d <%s> <-> %d <%s>\n", i, SCIPconsGetName(cons), auti,
            SCIPconsGetName(detprobdata->getCons(partialdec->getOpenconss()[auti])));

         ind = MIN(i, auti);

         if( hook->conssperm[i] != -1)
            ind = MIN(ind, hook->conssperm[i]);
         if( hook->conssperm[auti] != -1 )
            ind = MIN(ind, hook->conssperm[auti]);

         hook->conssperm[i] = ind;
         hook->conssperm[auti] = ind;
         hook->setBool(TRUE);
      }
   }
}

static
SCIP_RETCODE allocMemory(
    SCIP*                scip,               /**< SCIP data structure */
    AUT_COLOR*           colorinfo,          /**< struct to save intermediate information */
    int                  nconss,             /**< number of constraints */
    int                  nvars               /**< number of variables */
    )
{
   SCIP_CALL( SCIPallocMemoryArray(scip, &colorinfo->ptrarraycoefs, nvars) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &colorinfo->ptrarrayvars, nvars) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &colorinfo->ptrarrayconss, nconss) );
   colorinfo->alloccoefsarray = nvars;
   return SCIP_OKAY;
}

/** destructor for colorinfo */
static
void freeMemory(
   SCIP*                 scip,               /**< SCIP data structure */
   AUT_COLOR*            colorinfo           /**< struct to save intermediate information */
)
{
   int i;

   for( i = 0; i < colorinfo->lenvarsarray; i++ )
   {
      AUT_VAR* svar = (AUT_VAR*) colorinfo->ptrarrayvars[i];
      delete svar;
   }
   for( i = 0; i < colorinfo->lenconssarray; i++ )
   {
      AUT_CONS* scons = (AUT_CONS*) colorinfo->ptrarrayconss[i];
      delete scons;
   }
   for( i = 0; i < colorinfo->lencoefsarray; i++ )
   {
      AUT_COEF* scoef = (AUT_COEF*) colorinfo->ptrarraycoefs[i];
      delete scoef;
   }

   SCIPfreeMemoryArray(scip, &colorinfo->ptrarraycoefs);
   SCIPfreeMemoryArray(scip, &colorinfo->ptrarrayconss);
   SCIPfreeMemoryArray(scip, &colorinfo->ptrarrayvars);
}

/** set up a help structure for graph creation (for partialdecs) */
static
SCIP_RETCODE setupArrays(
   SCIP*                 scip,               /**< SCIP to compare */
   AUT_COLOR*            colorinfo,          /**< data structure to save intermediate data */
   SCIP_RESULT*          result,             /**< result pointer to indicate success or failure */
   gcg::PARTIALDECOMP*   partialdec,         /**< partialdec to set up help structure for */
   gcg::DETPROBDATA*     detprobdata         /**< detection process information and data */
   )
{ /*lint -esym(593,scoef) */
   int i;
   int j;
   int nconss;
   int nvars;
   AUT_COEF* scoef;
   AUT_CONS* scons;
   SCIP_Bool added;
   SCIP_Bool onlysign;

   //allocate max n of coefarray, varsarray, and boundsarray in scip
   nconss = partialdec->getNOpenconss();
   nvars = partialdec->getNVars();
   SCIP_CALL( allocMemory(scip, colorinfo, nconss, nvars) );

   onlysign = colorinfo->getOnlySign();

   //save the properties of variables in a struct array and in a sorted pointer array
   for( i = 0; i < nvars; i++ )
   {
      SCIP_VAR* var = detprobdata->getVar(i);
      AUT_VAR* svar = new AUT_VAR(scip, var);
      //add to pointer array iff it doesn't exist
      SCIP_CALL( colorinfo->insert(svar, &added) );
      SCIPdebugMessage("%s color %d %d\n", SCIPvarGetName(var), colorinfo->get(*svar), colorinfo->color);
      //otherwise free allocated memory
      if( !added )
         delete svar;
   }

   //save the properties of constraints in a struct array and in a sorted pointer array
   for( i = 0; i < nconss && *result == SCIP_SUCCESS; i++ )
   {
      int consindex = partialdec->getOpenconss()[i];
      SCIP_CONS* cons = detprobdata->getCons(consindex);

      int ncurvars = detprobdata->getNVarsForCons(consindex);
      if( ncurvars == 0 )
         continue;

      scons = new AUT_CONS(scip, cons);
      //add to pointer array iff it doesn't exist
      SCIPdebugMessage("nconss %d %d\n", nconss, *result);
      SCIP_CALL( colorinfo->insert(scons, &added) );
      SCIPdebugMessage("%s color %d %d\n", SCIPconsGetName(cons), colorinfo->get(*scons), colorinfo->color);
      //otherwise free allocated memory
      if( !added )
         delete scons;

      //save the properties of variables of the constraints in a struct array and in a sorted pointer array
      for( j = 0; j < ncurvars; j++ )
      {
         added = FALSE;

         if( !onlysign )
         {
            scoef = new AUT_COEF(scip, detprobdata->getValsForCons(consindex)[j]);
         }
         else
         {
            if( SCIPisPositive(scip, detprobdata->getValsForCons(consindex)[j]) )
               scoef = new AUT_COEF(scip, 1.0);
            else if( SCIPisNegative(scip, detprobdata->getValsForCons(consindex)[j]) )
               scoef = new AUT_COEF(scip, -1.0);
            else
               scoef = new AUT_COEF(scip, 0.0);
         }

         //test, whether the coefficient is not zero
         if( !SCIPisZero(scip, scoef->getVal()) )
         {
            //add to pointer array iff it doesn't exist
            SCIP_CALL( colorinfo->insert(scoef, &added) );
            SCIPdebugMessage("%f color %d %d\n", scoef->getVal(), colorinfo->get(*scoef), colorinfo->color);
         }
         //otherwise free allocated memory
         if( !added )
            delete scoef;

      }
   }
   return SCIP_OKAY;
}

/** create a graph out of an array of scips (for partialdecs) */
static
SCIP_RETCODE createGraph(
   SCIP*                 scip,               /**< SCIP to compare */
   AUT_COLOR             colorinfo,          /**< data structure to save intermediate data */
   bliss::Graph*         graph,              /**< graph needed for discovering isomorphism */
   SCIP_RESULT*          result,             /**< result pointer to indicate success or failure */
   gcg::PARTIALDECOMP*   partialdec,         /**< partialdec to create graph for */
   gcg::DETPROBDATA*     detprobdata         /**< detection process information and data */
   )
{
   int i;
   int j;
   int z;
   int nvars;
   int nconss;
   int ncurvars;
   int curvar;
   int color;
   unsigned int nnodes;
   SCIP_Bool onlysign;
   nnodes = 0;
   //building the graph out of the arrays
   bliss::Graph* h = graph;
   nconss = partialdec->getNOpenconss();
   nvars = partialdec->getNVars();
   z = 0;
   onlysign = colorinfo.getOnlySign();

   //add a node for every constraint
   for( i = 0; i < nconss && *result == SCIP_SUCCESS; i++ )
   {
      ncurvars = detprobdata->getNVarsForCons(partialdec->getOpenconss()[i]);
      SCIP_CONS* cons = detprobdata->getCons(partialdec->getOpenconss()[i]);

      AUT_CONS scons(scip, cons);
      color = colorinfo.get(scons);

      if( color == -1 )
      {
         *result = SCIP_DIDNOTFIND;
         break;
      }

      assert(color >= 0);
      (void)h->add_vertex((unsigned int) color);
      nnodes++;
   }
   //add a node for every variable
   for( i = 0; i < nvars && *result == SCIP_SUCCESS; i++ )
   {
      SCIP_VAR* var = detprobdata->getVar(i);
      AUT_VAR svar(scip, var);
      color = colorinfo.get(svar);

      if( color == -1 )
      {
         *result = SCIP_DIDNOTFIND;
         break;
      }
      (void) h->add_vertex((unsigned int) (colorinfo.getLenCons() + color));
      nnodes++;
   }
   //connecting the nodes with an additional node in the middle
   //it is necessary, since only nodes have colors
   for( i = 0; i < nconss && *result == SCIP_SUCCESS; i++ )
   {
      int consindex = partialdec->getOpenconss()[i];
      SCIP_CONS* cons = detprobdata->getCons(consindex);
      AUT_CONS scons(scip, cons);
      ncurvars = detprobdata->getNVarsForCons(partialdec->getOpenconss()[i]);
      if( ncurvars == 0 )
         continue;

      for( j = 0; j < ncurvars; j++ )
      {
         int varindex = detprobdata->getVarsForCons(consindex)[j];
         SCIP_VAR* var = detprobdata->getVar(varindex);


//              if( SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED)
//                 SCIPgetProbvarSum(scip, &(curvars[j]), &(curvals[j]), &constant);

         SCIP_Real val;

         if( !onlysign )
         {
            val = detprobdata->getValsForCons(consindex)[j];
         }
         else
         {
            if( SCIPisPositive(scip, detprobdata->getValsForCons(consindex)[j]) )
               val = 1.0;
            else if( SCIPisNegative(scip, detprobdata->getValsForCons(consindex)[j]) )
               val = -1.0;
            else
               val = 0.0;
         }
         *result = SCIP_SUCCESS;

         AUT_COEF scoef(scip, val);
         AUT_VAR svar(scip, var);

         color = colorinfo.get(scoef);

         if( color == -1 )
         {
            *result = SCIP_DIDNOTFIND;
            break;
         }

         curvar = SCIPvarGetProbindex(var);
         (void) h->add_vertex((unsigned int) (colorinfo.getLenCons() + colorinfo.getLenVar() + color)); /*lint !e864 */
         nnodes++;
         h->add_edge((unsigned int)i, (unsigned int) (nconss + nvars + z));
              h->add_edge((unsigned int) (nconss + nvars + z), (unsigned int) (nconss + curvar));
              SCIPdebugMessage(
                    "nz: c <%s> (id: %d, colour: %d) -> nz (id: %d) (value: %f, colour: %d) -> var <%s> (id: %d, colour: %d) \n",
                    SCIPconsGetName(cons), i, colorinfo.get(scons),
                    nconss + nvars + z, scoef.getVal(),
                    color + colorinfo.getLenCons() + colorinfo.getLenVar(), /*lint !e864 */
                    SCIPvarGetName(var), nconss + curvar,
                    colorinfo.get(svar) + colorinfo.getLenCons());  /*lint !e864 */
              z++;

      }


   }
   SCIPdebugMessage("Iteration 1: nnodes = %ud, Cons = %d, Vars = %d\n", nnodes, colorinfo.getLenCons(), colorinfo.getLenVar()); /*lint !e864 */
   assert(*result == SCIP_SUCCESS && nnodes == h->get_nof_vertices());

   //free all allocated memory
   freeMemory(scip, &colorinfo);
   return SCIP_OKAY;
}


/** creates a partialdec with provided constraints in the master
 * The function will put the remaining constraints in one or more pricing problems
 * depending on whether the subproblems decompose with no variables in common.
 */
SCIP_RETCODE createPartialdecFromMasterconss(
   SCIP*                 scip,                /**< SCIP data structure */
   gcg::PARTIALDECOMP**  newPartialdec,       /**< partialdec data structure */
   int*                  masterconss,         /**< constraints to be put in the master */
   int                   nmasterconss,        /**< number of constraints in the master */
   gcg::PARTIALDECOMP*   partialdec,          /**< partialdec to propagate */
   gcg::DETPROBDATA*     detprobdata,         /**< detection process information and data */
   SCIP_Bool             exact                /** does this partialdec stems from exact graph construction ( or was onlysign = TRUE ) was used */
   )
{


   char decinfo[SCIP_MAXSTRLEN];
   int nconss;
   int nvars;
   int nblocks;
   int* blockrepresentative = NULL;
   int nextblock = 1;
   SCIP_Bool* consismaster = NULL;
   int i, j;
   int* vartoblock = NULL;
   int ncurvars;

   std::vector<int> constoblock( detprobdata->getNConss(), -1);
   std::vector<int> newconstoblock( detprobdata->getNConss(), -1);

   assert(scip != NULL);
   assert(nmasterconss == 0 || masterconss != NULL);
   assert(SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED);

   nconss = partialdec->getNOpenconss();
   nvars = partialdec->getNVars();

   assert( nmasterconss <= nconss );

   nblocks = nconss-nmasterconss+1;
   assert(nblocks > 0);

   SCIP_CALL( SCIPallocMemoryArray(scip, &blockrepresentative, nblocks) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &consismaster, nconss) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &vartoblock, nvars) );

   for( i = 0; i < nmasterconss; ++i )
   {
      constoblock[masterconss[i]]  = nblocks+1;
   }

   for( i = 0; i < nconss; ++i )
   {
      consismaster[i] = ( constoblock[partialdec->getOpenconss()[i]] != -1 );
   }

   for( i = 0; i < nvars; ++i )
   {
      vartoblock[i] = -1;
   }

   for( i = 0; i < nblocks; ++i )
   {
      blockrepresentative[i] = -1;
   }

   /* assign constraints to representatives */

   /* go through the all constraints */
   for( i = 0; i < nconss; ++i )
   {
      int consblock;
      int cons = partialdec->getOpenconss()[i];

      if( consismaster[i] )
         continue;

      /* get variables of constraint; ignore empty constraints */
      ncurvars = detprobdata->getNVarsForCons(partialdec->getOpenconss()[i]);
      assert(ncurvars >= 0);

      assert( constoblock[cons] == -1 );

      /* if there are no variables, put it in the first block, otherwise put it in the next block */
      if( ncurvars == 0 )
         consblock = -1;
      else
         consblock = nextblock;

      /* go through all variables */
      for( j = 0; j < ncurvars; ++j )
      {
         int var;
         int varblock;
         var = detprobdata->getVarsForCons(cons)[j];

         assert(var >= 0);

         /** @todo what about deleted variables? */
         /* get block of variable */
         varblock = vartoblock[var];

         /* if variable is already assigned to a block, assign constraint to that block */
         if( varblock > -1 && varblock != consblock )
         {
            consblock = MIN(consblock, blockrepresentative[varblock]);
            SCIPdebugPrintf("still in block %d.\n", varblock);
         }
         else if( varblock == -1 )
         {
            /* if variable is free, assign it to the new block for this constraint */
            varblock = consblock;
            assert(varblock > 0);
            assert(varblock <= nextblock);
            vartoblock[var] = varblock;
            SCIPdebugPrintf("new in block %d.\n", varblock);
         }
         else
         {
            assert((varblock > 0) && (consblock == varblock));
            SCIPdebugPrintf("no change.\n");
         }

         SCIPdebugPrintf("VARINDEX: %d (%d)\n", var, vartoblock[var]);
      }

      /* if the constraint belongs to a new block, mark it as such */
      if( consblock == nextblock )
      {
         assert(consblock > 0);
         blockrepresentative[consblock] = consblock;
         assert(blockrepresentative[consblock] > 0);
         assert(blockrepresentative[consblock] <= nextblock);
         ++(nextblock);
      }

      SCIPdebugMessage("Cons %s will be in block %d (next %d)\n", SCIPconsGetName(detprobdata->getCons(cons)), consblock, nextblock);

      for( j = 0; j < ncurvars; ++j )
      {
         int var;
         int oldblock;
         var = detprobdata->getVarsForCons(cons)[j];

         oldblock = vartoblock[var];
         assert((oldblock > 0) && (oldblock <= nextblock));

         SCIPdebugMessage("\tVar %s ", SCIPvarGetName(detprobdata->getVar(var)));
         if( oldblock != consblock )
         {
            SCIPdebugPrintf("reset from %d to block %d.\n", oldblock, consblock);
            vartoblock[var] = consblock;
            SCIPdebugPrintf("VARINDEX: %d (%d)\n", var, consblock);

            if( (blockrepresentative[oldblock] != -1) && (blockrepresentative[oldblock] > blockrepresentative[consblock]) )
            {
               int oldrepr;
               oldrepr = blockrepresentative[oldblock];
               SCIPdebugMessage("\t\tBlock representative from block %d changed from %d to %d.\n", oldblock, blockrepresentative[oldblock], consblock);
               assert(consblock > 0);
               blockrepresentative[oldblock] = consblock;
               if( (oldrepr != consblock) && (oldrepr != oldblock) )
               {
                  blockrepresentative[oldrepr] = consblock;
                  SCIPdebugMessage("\t\tBlock representative from block %d changed from %d to %d.\n", oldrepr, blockrepresentative[oldrepr], consblock);
               }
            }
         }
         else
         {
            SCIPdebugPrintf("will not be changed from %d to %d.\n", oldblock, consblock);
         }
      }
      assert(consblock >= 1 || consblock == -1);
      assert(consblock <= nextblock);

      /* store the constraint block */
      if( consblock != -1 )
      {
         SCIPdebugMessage("cons %s in block %d\n", SCIPconsGetName(detprobdata->getCons(cons)), consblock);
         constoblock[cons] = consblock;
      }
      else
      {
         SCIPdebugMessage("ignoring %s\n", SCIPconsGetName(detprobdata->getCons(cons)));
      }
   }

   /* postprocess blockrepresentatives */

   int tempblock = 1;
   int maxblock = nextblock;

   assert(maxblock >= 1);
   assert(blockrepresentative != NULL );
   //SCIPdebugPrintf("Blocks: ");

   for( i = 1; i < maxblock; ++i )
   {
      /* forward replace the representatives */
      assert(blockrepresentative[i] >= 0);
      assert(blockrepresentative[i] < maxblock);
      if( blockrepresentative[i] != i )
         blockrepresentative[i] = blockrepresentative[blockrepresentative[i]];
      else
      {
         blockrepresentative[i] = tempblock;
         ++tempblock;
      }
      /* It is crucial that this condition holds */
      assert(blockrepresentative[i] <= i);
 //     SCIPdebugPrintf("%d ", blockrepresentative[i]);
   }
//   SCIPdebugPrintf("\n");

   /* convert temporary data to detectordata */

   /* fillout Constoblock */
   /* convert temporary data to detectordata */
   for( i = 0; i < nconss; ++i )
   {
      int consblock;

      int cons = partialdec->getOpenconss()[i];

      if( consismaster[i] )
      {
         /* notation is misleading: masterconss are only potential master constraints */
         /* SCIP_CALL( SCIPhashmapInsert(newconstoblock, (void*) (size_t) cons, (void*) (size_t) (nblocks+1)) ); */
         continue;
      }

      if( constoblock[cons] == -1)
               continue;

      consblock = constoblock[cons]; /*lint !e507*/
      assert(consblock > 0);
      consblock = blockrepresentative[consblock];
      assert(consblock <= nblocks);
      newconstoblock[cons] = consblock;
      SCIPdebugMessage("%d %s\n", consblock, SCIPconsGetName(detprobdata->getCons(cons)));
   }
   (*newPartialdec) = new gcg::PARTIALDECOMP(partialdec);
   SCIP_CALL( (*newPartialdec)->assignPartialdecFromConstoblockVector(newconstoblock, nblocks) );

   (*newPartialdec)->considerImplicits();
   (*newPartialdec)->refineToBlocks();

   if( exact )
      (void) SCIPsnprintf(decinfo, SCIP_MAXSTRLEN, "isomorph\\_exact");
   else
      (void) SCIPsnprintf(decinfo, SCIP_MAXSTRLEN, "isomorph\\_extended" );
   (*newPartialdec)->addDetectorChainInfo(decinfo);



   //(*newPartialdec)->showScatterPlot(detprobdata);

   SCIPfreeMemoryArray(scip, &vartoblock);
   SCIPfreeMemoryArray(scip, &consismaster);
   SCIPfreeMemoryArray(scip, &blockrepresentative);

   return SCIP_OKAY;
}

/** destructor of detector to free user data (called when GCG is exiting) */
static
DEC_DECL_FREEDETECTOR(detectorFreeIsomorph)
{ /*lint --e{715}*/
   DEC_DETECTORDATA *detectordata;

   assert(scip != NULL);
   assert(detector != NULL);

   assert(strcmp(DECdetectorGetName(detector), DEC_DETECTORNAME) == 0);

   detectordata = DECdetectorGetData(detector);
   assert(detectordata != NULL);

   SCIPfreeMemory(scip, &detectordata);

   return SCIP_OKAY;
}

/** detector initialization method (called after problem was transformed) */
static
DEC_DECL_INITDETECTOR(detectorInitIsomorph)
{ /*lint --e{715}*/
   DEC_DETECTORDATA *detectordata;

   assert(scip != NULL);
   assert(detector != NULL);

   assert(strcmp(DECdetectorGetName(detector), DEC_DETECTORNAME) == 0);

   detectordata = DECdetectorGetData(detector);
   assert(detectordata != NULL);

   detectordata->result = SCIP_SUCCESS;

   return SCIP_OKAY;
}

/** renumbers the permutations from 0 to n-1 and returns the number of permutations
 * @return the number of permutations
 */
int renumberPermutations(
   int*                  permutation,        /**< the permutation */
   int                   permsize            /**< size of the permutation */
)
{
   // renumbering from 0 to number of permutations
   int nperms = -1;

   for( int i = 0; i < permsize; i++ )
   {
      SCIPdebugMessage("%d: %d -> ", i, permutation[i]);
      if( permutation[i] == -1 )
      {
         SCIPdebugPrintf("%d\n", permutation[i]);
         continue;
      }

      if( permutation[i] > nperms && permutation[permutation[i]] > nperms )
      {
         nperms++;
         permutation[i] = nperms;
      }
      else
      {
         permutation[i] = permutation[permutation[i]];
      }
      SCIPdebugPrintf("%d\n", permutation[i]);
   }

   return nperms+1;
}

/** collapses the permutation, if possible */
void collapsePermutation(
   int*                  permutation,        /**< the permutation */
   int                   permsize            /**< size of the permutation */
)
{
   int tmp = 0;
   // assign to a permutation circle only one number
   for( int i = 0; i < permsize; i++ )
   {
      if( permutation[i] != -1 && permutation[i] != i )
      {
         tmp = permutation[i];
         permutation[i] = permutation[tmp];
      }
      SCIPdebugMessage("%d %d\n",i, permutation[i]);

   }
}

/** method to enumerate all subsets */
static
std::vector< std::vector<int> > getAllSubsets(std::vector<int> set)
{
    std::vector< std::vector<int> > subset;
    std::vector<int> empty;
    subset.push_back( empty );

    for ( size_t i = 0; i < set.size(); ++i )
    {
        std::vector< std::vector<int> > subsetTemp = subset;

        for (size_t j = 0; j < subsetTemp.size(); ++j)
            subsetTemp[j].push_back( set[i] );

        for (size_t j = 0; j < subsetTemp.size(); ++j)
            subset.push_back( subsetTemp[j] );
    }
    return subset;
}

/** reorder such that the best permutation is represented by 0, the second best by 1, etc. */
SCIP_RETCODE reorderPermutations(
   SCIP*                 scip,               /**< SCIP data structure */
   gcg::DETPROBDATA*     detprobdata,        /**< detection process information and data */
   int*                  permutation,        /**< the permutation */
   int                   permsize,           /**< size of the permutation */
   int                   nperms              /**< number of permutations */
)
{
   int i;
   int* count = NULL;
   int* order = NULL;
   int* invorder = NULL;

   assert(scip != NULL);
   assert(permutation != NULL);
   assert(permsize > 0);
   assert(nperms > 0);

   SCIP_CALL( SCIPallocMemoryArray(scip, &count, nperms) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &order, nperms) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &invorder, nperms) );
   BMSclearMemoryArray(count, nperms);
   BMSclearMemoryArray(order, nperms);
   BMSclearMemoryArray(invorder, nperms);

   /* initialize order array that will give a mapping from new to old representatives */
   for( i = 0; i < nperms; ++i )
   {
      order[i] = i;
   }

   /* count sizes of orbits */
   for( i = 0; i < permsize; ++i )
   {
      if( permutation[i] >= 0 )
      {
         count[permutation[i]] += 1;

         SCIPdebugMessage("permutation[i] = %d; count %d\n", permutation[i], count[permutation[i]]);
      }
   }

   /* sort count and order array */
   SCIPsortDownIntInt(count, order, nperms);

#ifdef SCIP_DEBUG

   for( i = 0; i < nperms; ++i )
   {
      SCIPdebugMessage("count[%d] = %d, order[%d] = %d\n", i, count[i], i, order[i]);
   }
#endif

   /* compute invorder array that gives a mapping from old to new representatives */
   for( i = 0; i < nperms; ++i )
   {
      invorder[order[i]] = i;
      SCIPdebugMessage("invorder[%d] = %d\n", order[i], invorder[order[i]]);
   }

   SCIPdebugMessage("Best permutation with orbit of size %d, best %d\n", count[0], order[0]);

   /* update representatives of constraints */
   for( i = 0; i < permsize; ++i )
   {
      if( permutation[i] >= 0 )
         permutation[i] = invorder[permutation[i]];
   }


   std::vector<int> orbitsizes(0);

   /* compute invorder array that gives a mapping from old to new representatives */
   for( i = 0; i < nperms; ++i )
   {
      int orbitsize;
      orbitsize = count[i];

      /* find orbitsize or not */
      std::vector<int>::const_iterator orbitsizesIter = orbitsizes.begin();
      for(; orbitsizesIter != orbitsizes.end(); ++orbitsizesIter)
      {
        if(*orbitsizesIter == orbitsize)
           break;
      }

      if( orbitsizesIter  == orbitsizes.end()  )
      {
         GCGconshdlrDecompAddCandidatesNBlocks(scip, detprobdata->isAssignedToOrigProb(), orbitsize);

         orbitsizes.push_back(orbitsize);
      }
   }
   std::vector< std::vector<int> > subsetsOfOrbitsizes = getAllSubsets(orbitsizes);

   for(size_t subset = 0; subset < subsetsOfOrbitsizes.size(); ++subset)
   {
      int greatestCD = 1;

      if(subsetsOfOrbitsizes[subset].size() == 0 || subsetsOfOrbitsizes[subset].size() == 1)
           continue;

      greatestCD = gcd(subsetsOfOrbitsizes[subset][0], subsetsOfOrbitsizes[subset][1]  );

      for( size_t j = 2; j < subsetsOfOrbitsizes[subset].size() ; ++j)
      {
         greatestCD = gcd( greatestCD, subsetsOfOrbitsizes[subset][j] );
      }

      GCGconshdlrDecompAddCandidatesNBlocks(scip, detprobdata->isAssignedToOrigProb(), greatestCD);
   }


   SCIPfreeMemoryArray(scip, &count);
   SCIPfreeMemoryArray(scip, &order);
   SCIPfreeMemoryArray(scip, &invorder);

   return SCIP_OKAY;
}


/** detection function of isomorph detector for partialdecs */
static
SCIP_RETCODE detectIsomorph(
   SCIP*                 scip,               /**< SCIP data structure */
   PARTIALDEC_DETECTION_DATA* detectiondata, /**< detection data */
   DEC_DETECTORDATA*     detectordata,       /**< detector data structure */
   SCIP_RESULT*          result,             /**< pointer to store result */
   SCIP_Bool             onlysign,           /**< use only sign of coefficients instead of coefficients? */
   int                   maxdecomps          /**< maximum number of new decompositions */
   )
{
   SCIP_CLOCK* temporaryClock;
   SCIP_CALL_ABORT( SCIPcreateClock(scip, &temporaryClock) );
   SCIP_CALL_ABORT( SCIPstartClock(scip, temporaryClock) );

   bliss::Graph graph;
   bliss::Stats bstats;
   AUT_HOOK *ptrhook;
   AUT_COLOR *colorinfo;
   gcg::PARTIALDECOMP* partialdec = detectiondata->workonpartialdec;
   gcg::DETPROBDATA* detprobdata = detectiondata->detprobdata;

   int nconss = partialdec->getNOpenconss();
   int i;

   detectordata->result = SCIP_SUCCESS;

   colorinfo = new AUT_COLOR();

   colorinfo->setOnlySign(onlysign);


   if( !onlysign )
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Detecting aggregatable structure: ");
   else
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Detecting almost aggregatable structure: ");

   SCIP_CALL( setupArrays(scip, colorinfo, &detectordata->result, partialdec, detprobdata) );
   SCIP_CALL( createGraph(scip, *colorinfo, &graph, &detectordata->result, partialdec, detprobdata) );

   ptrhook = new AUT_HOOK(FALSE, graph.get_nof_vertices(), scip, partialdec, detprobdata);
   for( i = 0; i < nconss; i++ )
   {
      ptrhook->conssperm[i] = -1;
   }

#if BLISS_VERSION_MAJOR >= 1 || BLISS_VERSION_MINOR >= 76
   auto report = [&](unsigned int n, const unsigned int* aut) {
      fhookForPartialdecs((void*)ptrhook, n, aut);
   };

   graph.find_automorphisms(bstats, report);
#else
   graph.find_automorphisms(bstats, fhookForPartialdecs, ptrhook);
#endif

   if( !ptrhook->getBool() )
      detectordata->result = SCIP_DIDNOTFIND;

   if( detectordata->result == SCIP_SUCCESS )
   {
      int nperms;
      int nmasterconss;
      int* masterconss = NULL;
      int p;

      // assign to a permutation circle only one number
      collapsePermutation(ptrhook->conssperm, nconss);
      // renumbering from 0 to number of permutations
      nperms = renumberPermutations(ptrhook->conssperm, nconss);

      // reorder decomposition (corresponding to orbit size)
      SCIP_CALL( reorderPermutations(scip, detprobdata, ptrhook->conssperm, nconss, nperms) );

      SCIP_CALL( SCIPreallocMemoryArray(scip, &(detectiondata->newpartialdecs), detectiondata->nnewpartialdecs + MIN(maxdecomps, nperms)) ); /*lint !e506*/

      int pos = detectiondata->nnewpartialdecs;

      for( p = 0; p < nperms && pos < maxdecomps; ++p )
      {
         SCIP_CALL( SCIPallocMemoryArray(scip, &masterconss, nconss) );

         SCIPdebugMessage("masterconss of partialdec %d:\n", p);

         nmasterconss = 0;
         for( i = 0; i < nconss; i++ )
         {
            if( p != ptrhook->conssperm[i] )
            {
               masterconss[nmasterconss] = partialdec->getOpenconss()[i];
               SCIPdebugMessage("%s\n", SCIPconsGetName(detprobdata->getCons(masterconss[nmasterconss])));
               nmasterconss++;
            }
         }
         SCIPdebugMessage("%d\n", nmasterconss);

         if( nmasterconss < nconss )
         {
            SCIP_Bool isduplicate;
            int q;

            SCIP_CALL( createPartialdecFromMasterconss(scip, &(detectiondata->newpartialdecs[pos]), masterconss, nmasterconss, partialdec, detprobdata, !onlysign) );

            isduplicate = FALSE;
            for( q = 0; q < pos && !isduplicate; ++q )
            {
               SCIP_CALL( detectiondata->newpartialdecs[pos]->isEqual(detectiondata->newpartialdecs[q], &isduplicate, TRUE) );
            }

            if( isduplicate )
            {
               delete detectiondata->newpartialdecs[pos];
            }
            else
            {
               ++pos;
            }

            SCIPfreeMemoryArray(scip, &masterconss);
         }

         else
         {
            SCIPfreeMemoryArray(scip, &masterconss);
            SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock ) );
            continue;
         }
      }
      detectiondata->nnewpartialdecs = pos;

      if( detectiondata->nnewpartialdecs > 0 )
      {
         SCIP_CALL( SCIPreallocMemoryArray(scip, &(detectiondata->newpartialdecs), detectiondata->nnewpartialdecs) );
      }

      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "found %d (new) decompositions.\n", detectiondata->nnewpartialdecs);
   }
   else
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "not found.\n");
   }

   delete colorinfo;
   delete ptrhook;

   SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock ) );

   if( detectiondata->nnewpartialdecs == 0 )
   {
      SCIPfreeMemoryArrayNull(scip, &(detectiondata->newpartialdecs));
   }

   detectiondata->detectiontime = SCIPgetClockTime(scip, temporaryClock);
   for( i = 0; i < detectiondata->nnewpartialdecs; ++i )
   {
      detectiondata->newpartialdecs[i]->addClockTime(detectiondata->detectiontime / detectiondata->nnewpartialdecs);
   }

   *result = detectordata->result;

   SCIP_CALL_ABORT(SCIPfreeClock(scip, &temporaryClock) );

   return SCIP_OKAY;
}


DEC_DECL_PROPAGATEPARTIALDEC(detectorPropagatePartialdecIsomorph)
{
   *result = SCIP_DIDNOTFIND;
   DEC_DETECTORDATA* detectordata = DECdetectorGetData(detector);
   gcg::PARTIALDECOMP* partialdec = partialdecdetectiondata->workonpartialdec ;

   partialdecdetectiondata->nnewpartialdecs = 0;
   partialdecdetectiondata->newpartialdecs = NULL;

   if(partialdec->getNBlocks() != 0 || partialdec->getNOpenvars() != partialdec->getNVars())
   {
      *result = SCIP_SUCCESS;
      return SCIP_OKAY;
   }

   if( detectordata->maxdecompsextend > 0 )
      SCIP_CALL( detectIsomorph(scip, partialdecdetectiondata, detectordata, result, TRUE, detectordata->maxdecompsextend) );

   if( detectordata->maxdecompsexact > 0 )
      SCIP_CALL( detectIsomorph(scip, partialdecdetectiondata, detectordata, result, FALSE, detectordata->maxdecompsexact) );

   for( int i = 0; i < partialdecdetectiondata->nnewpartialdecs; ++i )
   {
      partialdecdetectiondata->newpartialdecs[i]->refineToMaster();
   }

   return SCIP_OKAY;
}
#define detectorExitIsomorph NULL

#define detectorFinishPartialdecIsomorph NULL

#define detectorPostprocessPartialdecIsomorph NULL

static
DEC_DECL_SETPARAMAGGRESSIVE(setParamAggressiveIsomorph)
{
   char setstr[SCIP_MAXSTRLEN];
   const char* name = DECdetectorGetName(detector);
   int newval;
   SCIP_Real modifier;

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, TRUE) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE ) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxcallround", name);
   SCIP_CALL( SCIPgetIntParam(scip, setstr, &newval) );
   ++newval;
   SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
   SCIPinfoMessage(scip, NULL, "After Setting %s = %d\n", setstr, newval);


   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/origmaxcallround", name);
   SCIP_CALL( SCIPgetIntParam(scip, setstr, &newval) );
   ++newval;
   SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
   SCIPinfoMessage(scip, NULL, "%s = %d\n", setstr, newval);

   /* check if no problem is read */
   if( SCIPgetStage(scip) < SCIP_STAGE_PROBLEM )
   {
      newval = MAX( 0, DEFAULT_MAXDECOMPSEXACT   );
      (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxdecompsexact", name);
      SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
      SCIPinfoMessage(scip, NULL, "%s = %d\n", setstr, newval);

      newval = MAX( 0, DEFAULT_MAXDECOMPSEXTEND  );
      (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxdecompsextend", name);
      SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
      SCIPinfoMessage(scip, NULL, "%s = %d\n", setstr, newval);
      return SCIP_OKAY;
   }

   modifier = ( (SCIP_Real)SCIPgetNConss(scip) + (SCIP_Real)SCIPgetNVars(scip) ) / SET_MULTIPLEFORSIZETRANSF;

   modifier = log(modifier) / log(2);

   if (!SCIPisFeasPositive(scip, modifier) )
      modifier = -1.;

   modifier = SCIPfloor(scip, modifier);
   modifier += 0;

   newval = MAX( 0, DEFAULT_MAXDECOMPSEXACT - modifier );
   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxdecompsexact", name);
   SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
   SCIPinfoMessage(scip, NULL, "%s = %d\n", setstr, newval);

   newval = MAX( 0, DEFAULT_MAXDECOMPSEXTEND - modifier );
   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxdecompsextend", name);
   SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
   SCIPinfoMessage(scip, NULL, "%s = %d\n", setstr, newval);

   return SCIP_OKAY;
}


static
DEC_DECL_SETPARAMDEFAULT(setParamDefaultIsomorph)
{
   char setstr[SCIP_MAXSTRLEN];
   int newval;
   SCIP_Real modifier;

   const char* name = DECdetectorGetName(detector);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, DEC_ENABLED) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, DEC_ENABLEDFINISHING ) );


   /* check if no problem is read */
   if( SCIPgetStage(scip) < SCIP_STAGE_PROBLEM )
   {
      newval = MAX( 0, DEFAULT_MAXDECOMPSEXACT   );
      (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxdecompsexact", name);
      SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
      SCIPinfoMessage(scip, NULL, "%s = %d\n", setstr, newval);

      newval = MAX( 0, DEFAULT_MAXDECOMPSEXTEND  );
      (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxdecompsextend", name);
      SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
      SCIPinfoMessage(scip, NULL, "%s = %d\n", setstr, newval);
      return SCIP_OKAY;
   }


   modifier = ( (SCIP_Real)SCIPgetNConss(scip) + (SCIP_Real)SCIPgetNVars(scip) ) / SET_MULTIPLEFORSIZETRANSF;

   modifier = log(modifier) / log(2);

   if (!SCIPisFeasPositive(scip, modifier) )
      modifier = -1.;

   modifier = SCIPfloor(scip, modifier);
   modifier += 2;

   newval = MAX( 0, DEFAULT_MAXDECOMPSEXACT - modifier );
   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxdecompsexact", name);
   SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
   SCIPinfoMessage(scip, NULL, "%s = %d\n", setstr, newval);

   newval = MAX( 0, DEFAULT_MAXDECOMPSEXTEND - modifier );
   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxdecompsextend", name);
   SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
   SCIPinfoMessage(scip, NULL, "%s = %d\n", setstr, newval);

   return SCIP_OKAY;
}

static
DEC_DECL_SETPARAMFAST(setParamFastIsomorph)
{
   char setstr[SCIP_MAXSTRLEN];
   int newval;

   const char* name = DECdetectorGetName(detector);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE ) );

   /* check if no problem is read */
   if( SCIPgetStage(scip) < SCIP_STAGE_PROBLEM )
   {
      newval = MAX( 0, DEFAULT_MAXDECOMPSEXACT   );
      (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxdecompsexact", name);
      SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
      SCIPinfoMessage(scip, NULL, "%s = %d\n", setstr, newval);

      newval = MAX( 0, DEFAULT_MAXDECOMPSEXTEND  );
      (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxdecompsextend", name);
      SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
      SCIPinfoMessage(scip, NULL, "%s = %d\n", setstr, newval);
      return SCIP_OKAY;
   }



   newval = ( (SCIP_Real)SCIPgetNConss(scip) + (SCIP_Real)SCIPgetNVars(scip) >  SET_MULTIPLEFORSIZETRANSF) ? 0 : 1;
   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxdecompsexact", name);
   SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
   SCIPinfoMessage(scip, NULL, "%s = %d\n", setstr, newval);

   newval = 0;
   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxdecompsextend", name);
   SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
   SCIPinfoMessage(scip, NULL, "%s = %d\n", setstr, newval);

   return SCIP_OKAY;

}


/*
 * detector specific interface methods
 */


/** creates the handler for isomorph subproblems and includes it in SCIP */
extern "C"
SCIP_RETCODE SCIPincludeDetectorIsomorphism(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   DEC_DETECTORDATA* detectordata;

   detectordata = NULL;

   SCIP_CALL( SCIPallocMemory(scip, &detectordata) );
   assert(detectordata != NULL);



   SCIP_CALL( DECincludeDetector(scip, DEC_DETECTORNAME, DEC_DECCHAR, DEC_DESC, DEC_FREQCALLROUND, DEC_MAXCALLROUND, DEC_MINCALLROUND, DEC_FREQCALLROUNDORIGINAL, DEC_MAXCALLROUNDORIGINAL, DEC_MINCALLROUNDORIGINAL, DEC_PRIORITY, DEC_ENABLED, DEC_ENABLEDFINISHING,DEC_ENABLEDPOSTPROCESSING, DEC_SKIP, DEC_USEFULRECALL,
      detectordata, detectorFreeIsomorph, detectorInitIsomorph, detectorExitIsomorph, detectorPropagatePartialdecIsomorph, detectorFinishPartialdecIsomorph, detectorPostprocessPartialdecIsomorph, setParamAggressiveIsomorph, setParamDefaultIsomorph, setParamFastIsomorph) );

   /* add isomorph constraint handler parameters */
   SCIP_CALL( SCIPaddIntParam(scip, "detection/detectors/isomorph/maxdecompsexact",
      "Maximum number of solutions/decompositions with exact detection", &detectordata->maxdecompsexact, FALSE,
      DEFAULT_MAXDECOMPSEXACT, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip, "detection/detectors/isomorph/maxdecompsextend",
      "Maximum number of solutions/decompositions with extended detection", &detectordata->maxdecompsextend, FALSE,
      DEFAULT_MAXDECOMPSEXTEND, 0, INT_MAX, NULL, NULL) );

   return SCIP_OKAY;
}
