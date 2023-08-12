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

/**@file   decomp.c
 * @brief  generic methods for working with different decomposition structures
 * @author Martin Bergner
 * @author Michael Bastubbe
 *
 * Various methods to work with the decomp structure
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

//#define SCIP_DEBUG

#include "decomp.h"
#include "gcg.h"
#include "cons_decomp.h"
#include "scip/scip.h"
#include "struct_decomp.h"
#include "scip_misc.h"
#include "relax_gcg.h"


#include <assert.h>

typedef struct {
   SCIP_Real mean;
   SCIP_Real median;
   SCIP_Real max;
   SCIP_Real min;
} DEC_STATISTIC;

#define ELEM_SWAP(a,b) { register SCIP_Real t=(a);(a)=(b);(b)=t; }

static
SCIP_Real quick_select_median(SCIP_Real arr[], int n)
{
   int low, high;
   int median;

   low = 0;
   high = n - 1;
   median = high / 2;

   for( ;; )
   {
      int middle, ll, hh;
      if( high <= low ) /* One element only */
         return arr[median];

      if( high == low + 1 ) /* Two elements only */
      {
         if( arr[low] > arr[high] )
            ELEM_SWAP(arr[low], arr[high]);
         return arr[median];
      }

      /* Find median of low, middle and high items; swap into position low */
      middle = (low + high) / 2;
      if( arr[middle] > arr[high] )
         ELEM_SWAP(arr[middle], arr[high]);
      if( arr[low] > arr[high] )
         ELEM_SWAP(arr[low], arr[high]);
      if( arr[middle] > arr[low] )
         ELEM_SWAP(arr[middle], arr[low]);
      /* Swap low item (now in position middle) into position (low+1) */
      ELEM_SWAP(arr[middle], arr[(size_t) (low + 1)]);
      /* Nibble from each end towards middle, swapping items when stuck */
      ll = low + 1;
      hh = high;
      for( ;; )
      {
         do
            ll++;
         while( arr[low] > arr[ll] );
         do
            hh--;
         while( arr[hh] > arr[low] );
         if( hh < ll )
            break;
         ELEM_SWAP(arr[ll], arr[hh]);
      }
      /* Swap middle item (in position low) back into correct position */
      ELEM_SWAP(arr[low], arr[hh]);

      /* Re-set active partition */
      if( hh <= median )
         low = ll;
      if( hh >= median )
         high = hh - 1;
   }
}


/** fill out subscipvars arrays from the information from vartoblock */
static
SCIP_RETCODE fillOutVarsFromVartoblock(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_HASHMAP*         vartoblock,         /**< variable to block hashmap */
   int                   nblocks,            /**< number of blocks */
   SCIP_VAR**            vars,               /**< variable array */
   int                   nvars,              /**< number of variables */
   SCIP_Bool*            haslinking          /**< returns whether there are linking variables */
   )
{
   SCIP_VAR*** subscipvars;
   int* nsubscipvars;

   SCIP_VAR** linkingvars;
   int nlinkingvars;
   int nmastervars;
   int i;

   assert(scip != NULL);
   assert(decomp != NULL);
   assert(vartoblock != NULL);
   assert(nblocks >= 0);
   assert(vars != NULL);

   nlinkingvars = 0;
   nmastervars = 0;

   *haslinking = FALSE;

   if( nvars == 0 )
      return SCIP_OKAY;

   SCIP_CALL( SCIPallocBufferArray(scip, &linkingvars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &nsubscipvars, nblocks) );
   SCIP_CALL( SCIPallocBufferArray(scip, &subscipvars, nblocks) );

   for( i = 0; i < nblocks; ++i )
   {
      SCIP_CALL( SCIPallocBufferArray(scip, &subscipvars[i], nvars) ); /*lint !e866*/
      nsubscipvars[i] = 0;
   }

   /* handle variables */
   for( i = 0; i < nvars; ++i )
   {
      int block;
      SCIP_VAR* var;

      var = vars[i];
      assert(var != NULL);
      assert(SCIPvarIsActive(var));

      if( !SCIPhashmapExists(vartoblock, var) )
         block = nblocks+1;
      else
      {
         block = (int)(size_t)SCIPhashmapGetImage(vartoblock, var); /*lint !e507*/
      }

      assert(block > 0 && block <= nblocks+2);

      /* if variable belongs to a block */
      if( block <= nblocks )
      {
         SCIPdebugMessage("var %s in block %d.\n", SCIPvarGetName(var), block-1);
         subscipvars[block-1][nsubscipvars[block-1]] = var;
         ++(nsubscipvars[block-1]);
      }
      else /* variable is linking or master*/
      {
         assert(block == nblocks+1 || block == nblocks+2 );

         if( block == nblocks+2 )
            SCIPdebugMessage("var %s is linking.\n", SCIPvarGetName(var));
         else
         {
            SCIPdebugMessage("var %s is in master only.\n", SCIPvarGetName(var));
            ++nmastervars;
         }
         linkingvars[nlinkingvars] = var;
         ++nlinkingvars;
      }
   }

   if( nlinkingvars > 0 )
   {
      SCIP_CALL( DECdecompSetLinkingvars(scip, decomp, linkingvars, nlinkingvars, 0, nmastervars) );
      *haslinking = TRUE;
   }

   for( i = nblocks-1; i >= 0; --i )
   {
      if( nsubscipvars[i] == 0 )
      {
         SCIPfreeBufferArray(scip, &subscipvars[i]);
         subscipvars[i] = NULL;
      }
   }
   if( nblocks > 0 )
   {
      SCIP_CALL( DECdecompSetSubscipvars(scip, decomp, subscipvars, nsubscipvars) );
   }
   DECdecompSetVartoblock(decomp, vartoblock);

   for( i = nblocks-1; i >= 0; --i )
   {
     SCIPfreeBufferArrayNull(scip, &subscipvars[i]);
   }

   SCIPfreeBufferArray(scip, &subscipvars);
   SCIPfreeBufferArray(scip, &nsubscipvars);
   SCIPfreeBufferArray(scip, &linkingvars);

   return SCIP_OKAY;
}


/** fill out subscipcons arrays from the information from constoblock */
static
SCIP_RETCODE fillOutConsFromConstoblock(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_HASHMAP*         constoblock,        /**< constraint to block hashmap */
   int                   nblocks,            /**< number of blocks */
   SCIP_CONS**           conss,              /**< constraint array */
   int                   nconss,             /**< number of constraints */
   SCIP_Bool*            haslinking          /**< returns whether there are linking constraints */
   )
{
   SCIP_RETCODE retcode;

   SCIP_CONS*** subscipconss;
   int* nsubscipconss;

   SCIP_CONS** linkingconss;
   int nlinkingconss;
   int i;
   assert(scip != NULL);
   assert(decomp != NULL);
   assert(constoblock != NULL);
   assert(nblocks >= 0);
   assert(conss != NULL);
   assert(haslinking != NULL);

   *haslinking = FALSE;
   retcode = SCIP_OKAY;

   DECdecompSetConstoblock(decomp, constoblock);

   if( nconss == 0 )
      return retcode;

   SCIP_CALL( SCIPallocBufferArray(scip, &linkingconss, nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &nsubscipconss, nblocks) );
   SCIP_CALL( SCIPallocBufferArray(scip, &subscipconss, nblocks) );

   for( i = 0; i < nblocks; ++i )
   {
      SCIP_CALL( SCIPallocBufferArray(scip, &subscipconss[i], nconss) ); /*lint !e866*/
      nsubscipconss[i] = 0;
   }

   nlinkingconss = 0;

   /* handle constraints */
   for( i = 0; i < nconss; ++i )
   {
      int block;
      SCIP_CONS* cons;
      int nvars;
      SCIP_Bool success;

      cons = conss[i];
      assert(cons != NULL);

      if( !SCIPhashmapExists(decomp->constoblock, cons) )
      {
         block = nblocks+1;
         SCIP_CALL( SCIPhashmapInsert(decomp->constoblock, cons, (void*) (size_t) block) );
      }
      else
      {
         block = (int)(size_t)SCIPhashmapGetImage(decomp->constoblock, cons); /*lint !e507*/
      }

      assert(block > 0 && block <= nblocks+1);

      SCIP_CALL( SCIPgetConsNVars(scip, cons, &nvars, &success) );
      assert(success);
      if( nvars == 0 )
         continue;

      /* if constraint belongs to a block */
      if( block <= nblocks )
      {
         SCIPdebugMessage("cons %s in block %d.\n", SCIPconsGetName(cons), block-1);
         subscipconss[block-1][nsubscipconss[block-1]] = cons;
         ++(nsubscipconss[block-1]);
      }
      else /* constraint is linking */
      {
         SCIPdebugMessage("cons %s is linking.\n", SCIPconsGetName(cons));
         assert(block == nblocks+1);
         linkingconss[nlinkingconss] = cons;
         ++nlinkingconss;
      }
   }

   if( nlinkingconss > 0 )
   {
      retcode = DECdecompSetLinkingconss(scip, decomp, linkingconss, nlinkingconss);
      *haslinking = TRUE;
   }
   if( nblocks > 0 )
   {
      retcode = DECdecompSetSubscipconss(scip, decomp, subscipconss, nsubscipconss);
   }

   for( i = nblocks-1; i >= 0; --i )
   {
     SCIPfreeBufferArray(scip, &subscipconss[i]);
   }

   SCIPfreeBufferArray(scip, &subscipconss);
   SCIPfreeBufferArray(scip, &nsubscipconss);
   SCIPfreeBufferArray(scip, &linkingconss);

   return retcode;
}

/** removes a variable from the linking variable array */
static
SCIP_RETCODE removeFromLinkingvars(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_VAR*             var,                /**< variable to remove */
   SCIP_Bool*            success             /**< indicates whether the variable was successfully removed */
   )
{
   int v;
   int linkingvarsize;
   assert(scip != NULL);
   assert(decomp != NULL);
   assert(var != NULL);
   assert(success != NULL);

   *success = FALSE;
   linkingvarsize = decomp->nlinkingvars;

   for( v = 0; v < decomp->nlinkingvars; ++v )
   {
      if( decomp->linkingvars[v] == var )
      {
         decomp->linkingvars[v] = decomp->linkingvars[decomp->nlinkingvars-1];
         decomp->nlinkingvars -= 1;
         *success = TRUE;
      }
   }

   if( *success )
   {
      if( decomp->nlinkingvars == 0 )
      {
         SCIPfreeBlockMemoryArrayNull(scip, &decomp->linkingvars, SCIPcalcMemGrowSize(scip, linkingvarsize));
         if( DECdecompGetNLinkingconss(decomp) == 0 )
         {
            SCIP_CALL( DECdecompSetType(decomp, DEC_DECTYPE_DIAGONAL) );
         }
         else
         {
            SCIP_CALL( DECdecompSetType(decomp, DEC_DECTYPE_BORDERED) );
         }
      }
      else
      {
         int oldsize = SCIPcalcMemGrowSize(scip, linkingvarsize);
         int newsize = SCIPcalcMemGrowSize(scip, decomp->nlinkingvars);
         SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &decomp->linkingvars, oldsize, newsize) );
      }
   }
   return SCIP_OKAY;
}

/** for a given constraint, check which of its variables were previously determined to be copied directly to the master,
 * and assign them to the block to which the constraint belongs
 */
static
SCIP_RETCODE assignConsvarsToBlock(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_CONS*            cons,               /**< constraint whose variables should be assigned to its block */
   int                   block               /**< block to which the constraint has been assigned */
   )
{
   SCIP_VAR** curvars;
   int ncurvars;

   SCIP_Bool success;
   int v;

   curvars = NULL;
   ncurvars = 0;

   SCIP_CALL( SCIPgetConsNVars(scip, cons, &ncurvars, &success) );
   assert(success);
   SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );
   SCIP_CALL( SCIPgetConsVars(scip, cons, curvars, ncurvars, &success) );
   assert(success);

   for( v = 0; v < ncurvars; ++v )
   {
      SCIP_VAR* probvar = SCIPvarGetProbvar(curvars[v]);

      if( SCIPvarGetStatus(probvar) == SCIP_VARSTATUS_FIXED )
         continue;

      assert(SCIPhashmapExists(decomp->vartoblock, probvar));
      /* if variable is in master only, move to subproblem */
      if( (int) (size_t) SCIPhashmapGetImage(decomp->vartoblock, probvar) == decomp->nblocks+1 ) /*lint !e507 */
      {
         int oldsize;
         int newsize;
         oldsize = SCIPcalcMemGrowSize(scip, decomp->nsubscipvars[block]);
         newsize = SCIPcalcMemGrowSize(scip, decomp->nsubscipvars[block] + 1);
         SCIP_CALL( SCIPhashmapSetImage(decomp->vartoblock, probvar, (void*) (size_t)(block+1)) );
         SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &decomp->subscipvars[block], oldsize, newsize) ) /*lint !e866 */;
         decomp->subscipvars[block][decomp->nsubscipvars[block]] = probvar;
         decomp->nsubscipvars[block] += 1;
         SCIP_CALL( removeFromLinkingvars(scip, decomp, probvar, &success) );
         assert(success);
      }
   }

   SCIPfreeBufferArrayNull(scip, &curvars);

   return SCIP_OKAY;
}


const char *DECgetStrType(
   DEC_DECTYPE type
   )
{
   const char * names[] = { "unknown", "arrowhead", "staircase", "diagonal", "bordered" };
   return names[type];
}

/** initializes the decomposition to absolutely nothing */
SCIP_RETCODE DECdecompCreate(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP**          decdecomp           /**< pointer to the decomposition data structure */
   )
{
   DEC_DECOMP* decomp;

   int ncalls;

   assert(scip != NULL);
   assert(decdecomp != NULL);

   SCIP_CALL( SCIPallocMemory(scip, decdecomp) );
   assert(*decdecomp != NULL);
   decomp = *decdecomp;

   decomp->type = DEC_DECTYPE_UNKNOWN;
   decomp->constoblock = NULL;
   decomp->vartoblock = NULL;
   decomp->subscipvars = NULL;
   decomp->subscipconss = NULL;
   decomp->nsubscipconss = NULL;
   decomp->nsubscipvars = NULL;
   decomp->linkingconss = NULL;
   decomp->nlinkingconss = 0;
   decomp->linkingvars = NULL;
   decomp->nlinkingvars = 0;
   decomp->nfixedlinkingvars = 0;
   decomp->stairlinkingvars = NULL;
   decomp->nstairlinkingvars = NULL;
   decomp->nblocks = 0;
   decomp->presolved = (SCIPgetStage(scip) >= SCIP_STAGE_PRESOLVING);
   decomp->consindex = NULL;
   decomp->varindex = NULL;
   decomp->detector = NULL;
   decomp->nmastervars = 0;

   decomp->detectorchain = NULL;
   decomp->sizedetectorchain = 0;
   decomp->detectorchainstring = NULL;
   decomp->partialdecid = -1;
   decomp->detectorclocktimes = NULL;
   decomp->pctvarstoborder= NULL;
   decomp->pctconsstoborder= NULL;
   decomp->pctvarstoblock= NULL;
   decomp->pctconsstoblock= NULL;
   decomp->pctvarsfromopen= NULL;
   decomp->pctconssfromopen= NULL;
   decomp->nnewblocks= NULL;
   decomp->maxwhitescore = -1.;

   ncalls = GCGconshdlrDecompIncreaseNCallsCreateDecomp(scip);

   SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL, "ncalls of createdecompfrompartialdec: %d \n", ncalls);

   return SCIP_OKAY;
}

/** frees the decdecomp structure */
SCIP_RETCODE DECdecompFree(
   SCIP*                 scip,               /**< pointer to the SCIP instance */
   DEC_DECOMP**          decdecomp           /**< pointer to the decomposition data structure */
   )
{
   DEC_DECOMP* decomp;
   int i;
   int j;
   int ncalls;

   assert( scip!= NULL );
   assert( decdecomp != NULL);
   decomp = *decdecomp;

   assert(decomp != NULL);

   for( i = 0; i < decomp->nblocks; ++i )
   {
      if( decomp->nsubscipvars != NULL )
      {
         for( j = 0; j < decomp->nsubscipvars[i]; ++j )
         {
            if( decomp->subscipvars[i][j] != NULL )
            {
               SCIP_CALL( SCIPreleaseVar(scip, &(decomp->subscipvars[i][j])) );
            }
         }

         SCIPfreeBlockMemoryArrayNull(scip, &(decomp->subscipvars[i]), SCIPcalcMemGrowSize(scip, decomp->nsubscipvars[i])); /*lint !e866*/

      }
      if( decomp->nsubscipconss != NULL )
      {
         for( j = 0; j < decomp->nsubscipconss[i]; ++j )
         {
            if( decomp->subscipconss[i][j] != NULL )
            {
               SCIP_CALL( SCIPreleaseCons(scip, &(decomp->subscipconss[i][j])) );
            }
         }
         SCIPfreeBlockMemoryArrayNull(scip, &decomp->subscipconss[i], SCIPcalcMemGrowSize(scip, decomp->nsubscipconss[i])); /*lint !e866*/
      }
   }

   if( decomp->linkingvars != NULL )
   {
      for( i = 0; i < decomp->nlinkingvars; ++i )
      {
         if( decomp->linkingvars[i] != NULL )
         {
            if( decomp->linkingvars[i] != NULL )
            {
               SCIP_CALL( SCIPreleaseVar(scip, &(decomp->linkingvars[i])) );
            }
         }
      }
   }

   if( decomp->stairlinkingvars != NULL )
      for( i = 0; i < decomp->nblocks-1; ++i )
      {
         for( j = 0; j < decomp->nstairlinkingvars[i]; ++j )
         {
            if( decomp->stairlinkingvars[i][j] != NULL )
            {
               SCIP_CALL( SCIPreleaseVar(scip, &(decomp->stairlinkingvars[i][j])) );
            }
         }
         SCIPfreeBlockMemoryArrayNull(scip, &decomp->stairlinkingvars[i], SCIPcalcMemGrowSize(scip, decomp->nstairlinkingvars[i])); /*lint !e866*/
      }

   /* free hashmaps if they are not NULL */
   if( decomp->constoblock != NULL )
      SCIPhashmapFree(&decomp->constoblock);
   if( decomp->vartoblock != NULL )
      SCIPhashmapFree(&decomp->vartoblock);
   if( decomp->varindex != NULL )
      SCIPhashmapFree(&decomp->varindex);
   if( decomp->consindex != NULL )
      SCIPhashmapFree(&decomp->consindex);

   for( i = 0; i < decomp->nlinkingconss; ++i )
   {
      SCIP_CALL( SCIPreleaseCons(scip, &(decomp->linkingconss[i])) );
   }
   SCIPfreeBlockMemoryArrayNull(scip, &decomp->subscipvars, decomp->nblocks);
   SCIPfreeBlockMemoryArrayNull(scip, &decomp->nsubscipvars, decomp->nblocks);
   SCIPfreeBlockMemoryArrayNull(scip, &decomp->subscipconss, decomp->nblocks);
   SCIPfreeBlockMemoryArrayNull(scip, &decomp->nsubscipconss,  decomp->nblocks);
   SCIPfreeBlockMemoryArrayNull(scip, &decomp->linkingvars, SCIPcalcMemGrowSize(scip, decomp->nlinkingvars));
   SCIPfreeBlockMemoryArrayNull(scip, &decomp->stairlinkingvars, decomp->nblocks);
   SCIPfreeBlockMemoryArrayNull(scip, &decomp->nstairlinkingvars,decomp->nblocks);
   SCIPfreeBlockMemoryArrayNull(scip, &decomp->linkingconss, SCIPcalcMemGrowSize(scip, decomp->nlinkingconss));
   if( decomp->detectorchain != NULL )
   {
      SCIPfreeBlockMemoryArrayNull(scip, &decomp->detectorchain, SCIPcalcMemGrowSize(scip,decomp->sizedetectorchain ) );
      SCIPfreeBlockMemoryArrayNull(scip, &decomp->detectorclocktimes, SCIPcalcMemGrowSize(scip,decomp->sizedetectorchain ) );
      SCIPfreeBlockMemoryArrayNull(scip, &decomp->pctvarstoborder, SCIPcalcMemGrowSize(scip,decomp->sizedetectorchain ) );
      SCIPfreeBlockMemoryArrayNull(scip, &decomp->pctconsstoborder, SCIPcalcMemGrowSize(scip,decomp->sizedetectorchain ) );
      SCIPfreeBlockMemoryArrayNull(scip, &decomp->pctvarstoblock, SCIPcalcMemGrowSize(scip,decomp->sizedetectorchain ) );
      SCIPfreeBlockMemoryArrayNull(scip, &decomp->pctconsstoblock, SCIPcalcMemGrowSize(scip,decomp->sizedetectorchain ) );
      SCIPfreeBlockMemoryArrayNull(scip, &decomp->pctvarsfromopen, SCIPcalcMemGrowSize(scip,decomp->sizedetectorchain ) );
      SCIPfreeBlockMemoryArrayNull(scip, &decomp->pctconssfromopen, SCIPcalcMemGrowSize(scip,decomp->sizedetectorchain ) );
      SCIPfreeBlockMemoryArrayNull(scip, &decomp->nnewblocks, SCIPcalcMemGrowSize(scip,decomp->sizedetectorchain ) );
      SCIPfreeBlockMemoryArrayNull(scip, &decomp->detectorchainstring, SCIP_MAXSTRLEN );
   }

   SCIPfreeMemoryNull(scip, decdecomp);

   ncalls = GCGconshdlrDecompDecreaseNCallsCreateDecomp(scip);

   SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL, "ncalls of createdecompfrompartialdec: %d \n", ncalls);

   return SCIP_OKAY;
}

/** sets the type of the decomposition */
SCIP_RETCODE DECdecompSetType(
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   DEC_DECTYPE           type               /**< type of the decomposition */
   )
{
   SCIP_Bool valid;

   assert(decomp != NULL);

   switch( type )
   {
   case DEC_DECTYPE_DIAGONAL:
      valid = decomp->nlinkingconss == 0 && decomp->linkingconss == NULL;
      valid = valid && decomp->nlinkingvars == 0 && decomp->linkingvars == NULL;
      break;
   case DEC_DECTYPE_ARROWHEAD:
      valid = TRUE;
      break;
   case DEC_DECTYPE_UNKNOWN:
      valid = FALSE;
      break;
   case DEC_DECTYPE_BORDERED:
      valid = decomp->nlinkingvars == 0 && decomp->linkingvars == NULL;
      break;
   case DEC_DECTYPE_STAIRCASE:
      valid = decomp->nlinkingconss == 0 && decomp->linkingconss == NULL;
      break;
   default:
      valid = FALSE;
      break;
   }

   if( !valid )
   {
      SCIPerrorMessage("The decomposition is not of the given type!\n");
      return SCIP_INVALIDDATA;
   }

   decomp->type = type;

   return SCIP_OKAY;
}

/** gets the type of the decomposition */
DEC_DECTYPE DECdecompGetType(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);

   return decomp->type;
}


SCIP_Real DECdecompGetMaxwhiteScore(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);

   return decomp->maxwhitescore;
}


/** sets the presolved flag for decomposition */
void DECdecompSetPresolved(
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_Bool             presolved           /**< presolved flag for decomposition */
   )
{
   assert(decomp != NULL);

   decomp->presolved = presolved;
}

/** gets the presolved flag for decomposition */
SCIP_Bool DECdecompGetPresolved(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);

   return decomp->presolved;
}

/** sets the number of blocks for decomposition */
void DECdecompSetNBlocks(
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   int                   nblocks             /**< number of blocks for decomposition */
   )
{
   assert(decomp != NULL);
   assert(nblocks >= 0);

   decomp->nblocks = nblocks;
}

/** gets the number of blocks for decomposition */
int DECdecompGetNBlocks(
      DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);

   return decomp->nblocks;
}

/** copies the input subscipvars array to the given decomposition */
SCIP_RETCODE DECdecompSetSubscipvars(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_VAR***           subscipvars,        /**< subscipvars array  */
   int*                  nsubscipvars        /**< number of subscipvars per block */
   )
{
   SCIP_Bool valid;
   int i;
   int b;
   assert(scip != NULL);
   assert(decomp != NULL);
   assert(subscipvars != NULL);
   assert(nsubscipvars != NULL);
   assert(decomp->nblocks >= 0);

   assert(decomp->subscipvars == NULL);
   assert(decomp->nsubscipvars == NULL);

   if( decomp->nblocks == 0 )
      return SCIP_OKAY;

   valid = TRUE;

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &decomp->subscipvars, decomp->nblocks) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &decomp->nsubscipvars, decomp->nblocks) );

   assert(decomp->subscipvars != NULL);
   assert(decomp->nsubscipvars != NULL);

   BMSclearMemoryArray(decomp->subscipvars, decomp->nblocks);
   BMSclearMemoryArray(decomp->nsubscipvars, decomp->nblocks);

   for( b = 0; b < decomp->nblocks; ++b )
   {
      assert((subscipvars[b] == NULL) == (nsubscipvars[b] == 0));
      decomp->nsubscipvars[b] = nsubscipvars[b];

      if( nsubscipvars[b] < 0 )
      {
         SCIPerrorMessage("Number of variables per subproblem must be nonnegative.\n");
         valid = FALSE;
      }
      else if( nsubscipvars[b] > 0 )
      {
         int size;
         assert(subscipvars[b] != NULL);
         size = SCIPcalcMemGrowSize(scip, nsubscipvars[b]);
         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &decomp->subscipvars[b], size) ); /*lint !e866*/
         BMScopyMemoryArray(decomp->subscipvars[b],subscipvars[b], nsubscipvars[b]); /*lint !e866*/

         for( i = 0; i < nsubscipvars[b]; ++i )
         {
            SCIP_CALL( SCIPcaptureVar(scip, decomp->subscipvars[b][i]) );
         }
      }
   }

   if( !valid )
   {
      return SCIP_INVALIDDATA;
   }


   return SCIP_OKAY;
}

/** returns the subscipvars array of the given decomposition */
SCIP_VAR*** DECdecompGetSubscipvars(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);

   return decomp->subscipvars;
}

/** returns the nsubscipvars array of the given decomposition */
int* DECdecompGetNSubscipvars(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);

   return decomp->nsubscipvars;
}

/** copies the input subscipconss array to the given decomposition */
SCIP_RETCODE DECdecompSetSubscipconss(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_CONS***          subscipconss,       /**< subscipconss array  */
   int*                  nsubscipconss       /**< number of subscipconss per block */
   )
{
   SCIP_Bool valid;
   int i;
   int b;
   assert(scip != NULL);
   assert(decomp != NULL);
   assert(subscipconss != NULL);
   assert(nsubscipconss != NULL);

   assert(decomp->nblocks >= 0);
   assert(decomp->subscipconss == NULL);
   assert(decomp->nsubscipconss == NULL);

   valid = TRUE;

   if( decomp->nblocks == 0)
      return SCIP_OKAY;

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &decomp->subscipconss, decomp->nblocks) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &decomp->nsubscipconss, decomp->nblocks) );

   assert(decomp->subscipconss != NULL);
   assert(decomp->nsubscipconss != NULL);

   BMSclearMemoryArray(decomp->subscipconss, decomp->nblocks);
   BMSclearMemoryArray(decomp->nsubscipconss, decomp->nblocks);

   for( b = 0; b < decomp->nblocks; ++b )
   {
      if( nsubscipconss[b] <= 0 || subscipconss[b] == NULL )
      {
         SCIPerrorMessage("Block %d is empty and thus invalid. Each block needs at least one constraint.\n", b);
         valid = FALSE;
      }

      decomp->nsubscipconss[b] = nsubscipconss[b];

      if( nsubscipconss[b] > 0 )
      {
         int size;

         assert(subscipconss[b] != NULL);
         size = SCIPcalcMemGrowSize(scip, nsubscipconss[b]);
         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &decomp->subscipconss[b], size) ); /*lint !e866*/
         BMScopyMemoryArray(decomp->subscipconss[b], subscipconss[b], nsubscipconss[b]); /*lint !e866*/
         for( i = 0; i < nsubscipconss[b]; ++i )
         {
            SCIP_CALL( SCIPcaptureCons(scip, decomp->subscipconss[b][i]) );
         }
      }
   }

   if( !valid )
      return SCIP_INVALIDDATA;

   return SCIP_OKAY;
}

/** returns the subscipconss array of the given decomposition */
SCIP_CONS*** DECdecompGetSubscipconss(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);
   return decomp->subscipconss;
}

/** returns the nsubscipconss array of the given decomposition */
int* DECdecompGetNSubscipconss(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);
   return decomp->nsubscipconss;
}

/** copies the input linkingconss array to the given decomposition */
SCIP_RETCODE DECdecompSetLinkingconss(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_CONS**           linkingconss,       /**< linkingconss array  */
   int                   nlinkingconss       /**< number of linkingconss per block */
   )
{
   assert(scip != NULL);
   assert(decomp != NULL);
   assert(linkingconss != NULL);
   assert(nlinkingconss >= 0);

   assert(decomp->linkingconss == NULL);
   assert(decomp->nlinkingconss == 0);

   decomp->nlinkingconss = nlinkingconss;

   if( nlinkingconss > 0 )
   {
      int i;
      int size;
      assert(linkingconss != NULL);
      size = SCIPcalcMemGrowSize(scip, nlinkingconss);
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &decomp->linkingconss, size) );
      BMScopyMemoryArray(decomp->linkingconss, linkingconss, nlinkingconss);

      for( i = 0; i < nlinkingconss; ++i )
      {
         SCIP_CALL( SCIPcaptureCons(scip, decomp->linkingconss[i]) );
      }
   }

   if( (linkingconss == NULL) !=  (nlinkingconss == 0) )
   {
      SCIPerrorMessage("Number of linking constraints and linking constraint array are inconsistent.\n");
      return SCIP_INVALIDDATA;
   }
   return SCIP_OKAY;
}

/** returns the linkingconss array of the given decomposition */
SCIP_CONS** DECdecompGetLinkingconss(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);

   return decomp->linkingconss;
}

/** returns the nlinkingconss array of the given decomposition */
int DECdecompGetNLinkingconss(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);
   assert(decomp->nlinkingconss >= 0);

   return decomp->nlinkingconss;
}


/** copies the input linkingvars array to the given decdecomp structure */
SCIP_RETCODE DECdecompSetLinkingvars(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_VAR**            linkingvars,        /**< linkingvars array  */
   int                   nlinkingvars,       /**< number of total linkingvars (including fixed linking vars,  ) */
   int                   nfixedlinkingvars,  /**< number of fixed linking variables */
   int                   nmastervars         /**< number of linking variables that are purely master variables */

   )
{
   assert(scip != NULL);
   assert(decomp != NULL);
   assert(linkingvars != NULL || nlinkingvars == 0);

   assert(decomp->linkingvars == NULL);
   assert(decomp->nlinkingvars == 0);

   decomp->nlinkingvars = nlinkingvars;
   decomp->nmastervars = nmastervars;
   decomp->nfixedlinkingvars = nfixedlinkingvars;

   if( nlinkingvars > 0 )
   {
      int i;
      int size;
      assert(linkingvars != NULL);
      size = SCIPcalcMemGrowSize(scip, nlinkingvars);
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &decomp->linkingvars, size) );
      BMScopyMemoryArray(decomp->linkingvars, linkingvars, nlinkingvars);

      for( i = 0; i < nlinkingvars; ++i )
      {
         SCIP_CALL( SCIPcaptureVar(scip, decomp->linkingvars[i]) );
      }
   }

   if( (linkingvars == NULL) != (nlinkingvars == 0) )
   {
      SCIPerrorMessage("Number of linking variables and linking variable array are inconsistent.\n");
      return SCIP_INVALIDDATA;
   }

   return SCIP_OKAY;
}


/** returns the linkingvars array of the given decomposition */
SCIP_VAR** DECdecompGetLinkingvars(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);

   return decomp->linkingvars;
}

/** returns the nlinkingvars array of the given decomposition */
int DECdecompGetNLinkingvars(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);
   assert(decomp->nlinkingvars >= 0);

   return decomp->nlinkingvars;
}

/** returns the nlinkingvars array of the given decomposition */
int DECdecompGetNFixedLinkingvars(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);
   assert(decomp->nfixedlinkingvars >= 0);

   return decomp->nfixedlinkingvars;
}


/** returns the number of linking variables that are purely master ("static") variables of the given decomposition */
int DECdecompGetNMastervars(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);
   assert(decomp->nmastervars >= 0);

   return decomp->nmastervars;
}


/** copies the input stairlinkingvars array to the given decomposition */
SCIP_RETCODE DECdecompSetStairlinkingvars(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_VAR***           stairlinkingvars,   /**< stairlinkingvars array  */
   int*                  nstairlinkingvars   /**< number of linkingvars per block */
   )
{
   SCIP_Bool valid;
   int b;
   int i;
   assert(scip != NULL);
   assert(decomp != NULL);
   assert(stairlinkingvars != NULL);
   assert(nstairlinkingvars != NULL);

   assert(decomp->nblocks >= 0);

   assert(decomp->stairlinkingvars == NULL);
   assert(decomp->nstairlinkingvars == NULL);

   valid = TRUE; /**@todo A valid check needs to be implemented */

   if( decomp->nblocks == 0 )
      return SCIP_OKAY;

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &decomp->stairlinkingvars, decomp->nblocks) ); /* this is more efficient */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &decomp->nstairlinkingvars, decomp->nblocks) ); /* this is more efficient */

   assert(decomp->stairlinkingvars != NULL);
   assert(decomp->nstairlinkingvars != NULL);

   BMSclearMemoryArray(decomp->stairlinkingvars, decomp->nblocks);
   BMSclearMemoryArray(decomp->nstairlinkingvars, decomp->nblocks);

   for( b = 0; b < decomp->nblocks-1; ++b )
   {
      assert(nstairlinkingvars[b] > 0 || stairlinkingvars[b] == NULL);
      decomp->nstairlinkingvars[b] = nstairlinkingvars[b];
      if( stairlinkingvars[b] != NULL )
      {
         int size = SCIPcalcMemGrowSize(scip, nstairlinkingvars[b]);
         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(decomp->stairlinkingvars[b]), size) ); /*lint !e866 */
         BMScopyMemoryArray(decomp->stairlinkingvars[b], stairlinkingvars[b], nstairlinkingvars[b]); /*lint !e866 */
      }
      else
      {
         decomp->stairlinkingvars[b] = NULL;
      }
   }

   decomp->nstairlinkingvars[decomp->nblocks - 1] = 0;
   decomp->stairlinkingvars[decomp->nblocks -1] = NULL;

   for( b = 0; b < decomp->nblocks-1; ++b )
   {
      for( i = 0; i < nstairlinkingvars[b]; ++i )
      {
         assert(stairlinkingvars[b] != NULL);
         SCIP_CALL( SCIPcaptureVar(scip, decomp->stairlinkingvars[b][i]) );
      }
   }
   if( !valid ) /*lint !e774 it is always true, see above */
   {
      SCIPerrorMessage("The staircase linking variables are inconsistent.\n");
      return SCIP_INVALIDDATA;
   }
   return SCIP_OKAY;
}

/** returns the stairlinkingvars array of the given decomposition */
SCIP_VAR*** DECdecompGetStairlinkingvars(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);
   return decomp->stairlinkingvars;
}

/** returns the nstairlinkingvars array of the given decomposition */
int* DECdecompGetNStairlinkingvars(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);
   assert(decomp->nstairlinkingvars != NULL );
   return decomp->nstairlinkingvars;
}

/** returns the total number of stairlinkingvars array of the given decomposition */
int DECdecompGetNTotalStairlinkingvars(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   int sum;
   int b;

   sum = 0;

   for ( b = 0; b < DECdecompGetNBlocks(decomp); ++b)
         sum += DECdecompGetNStairlinkingvars(decomp)[b];

   return sum;
}


/** sets the vartoblock hashmap of the given decomposition */
void DECdecompSetVartoblock(
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_HASHMAP*         vartoblock          /**< Vartoblock hashmap */
   )
{
   assert(decomp != NULL);
   assert(vartoblock != NULL);

   decomp->vartoblock = vartoblock;
}

/** returns the vartoblock hashmap of the given decomposition */
SCIP_HASHMAP* DECdecompGetVartoblock(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);

   return decomp->vartoblock;
}

/** sets the constoblock hashmap of the given decomposition */
void DECdecompSetConstoblock(
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_HASHMAP*         constoblock         /**< Constoblock hashmap */
   )
{
   assert(decomp != NULL);
   assert(constoblock != NULL);

   decomp->constoblock = constoblock;
}

/** returns the constoblock hashmap of the given decomposition */
SCIP_HASHMAP* DECdecompGetConstoblock(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);

   return decomp->constoblock;
}

/** sets the varindex hashmap of the given decomposition */
void DECdecompSetVarindex(
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_HASHMAP*         varindex            /**< Varindex hashmap */
   )
{
   assert(decomp != NULL);
   assert(varindex != NULL);
   decomp->varindex = varindex;
}

/** returns the varindex hashmap of the given decomposition */
SCIP_HASHMAP* DECdecompGetVarindex(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);
   return decomp->varindex;
}

/** sets the consindex hashmap of the given decomposition */
void DECdecompSetConsindex(
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_HASHMAP*         consindex           /**< Consindex hashmap */
   )
{
   assert(decomp != NULL);
   assert(consindex != NULL);
   decomp->consindex = consindex;
}

/** returns the consindex hashmap of the given decomposition */
SCIP_HASHMAP* DECdecompGetConsindex(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);
   return decomp->consindex;
}

/** completely initializes decomposition structure from the values of the hashmaps */
SCIP_RETCODE DECfilloutDecompFromHashmaps(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_HASHMAP*         vartoblock,         /**< variable to block hashmap */
   SCIP_HASHMAP*         constoblock,        /**< constraint to block hashmap */
   int                   nblocks,            /**< number of blocks */
   SCIP_Bool             staircase           /**< should the decomposition be a staircase structure */
   )
{
   SCIP_HASHMAP* varindex;
   SCIP_HASHMAP* consindex;
   int* nsubscipconss;
   int* nsubscipvars;
   int* nstairlinkingvars;
   SCIP_VAR*** stairlinkingvars;
   SCIP_CONS*** subscipconss;
   SCIP_Bool success;
   int cindex;
   int cumindex;
   SCIP_Bool haslinking;
   int i;
   int b;
   SCIP_VAR** curvars;
   int ncurvars;
   int j;

   SCIP_VAR** vars;
   int nvars;
   SCIP_CONS** conss;
   int nconss;

   assert(scip != NULL);
   assert(decomp != NULL);
   assert(vartoblock != NULL);
   assert(constoblock != NULL);
   assert(nblocks >= 0);

   conss = SCIPgetConss(scip);
   nconss = SCIPgetNConss(scip);
   vars = SCIPgetVars(scip);
   nvars = SCIPgetNVars(scip);

   assert(vars != NULL);
   assert(conss != NULL);

   DECdecompSetNBlocks(decomp, nblocks);

   SCIP_CALL( DECdecompSetType(decomp, DEC_DECTYPE_DIAGONAL) );
   SCIP_CALL_QUIET( fillOutConsFromConstoblock(scip, decomp, constoblock, nblocks, conss, nconss, &haslinking) );

   if( haslinking )
   {
      SCIPdebugMessage("Decomposition has linking constraints and is bordered.\n");
      SCIP_CALL( DECdecompSetType(decomp, DEC_DECTYPE_BORDERED) );
   }

   SCIP_CALL( fillOutVarsFromVartoblock(scip,  decomp, vartoblock, nblocks, vars, nvars, &haslinking) );

   if( haslinking )
   {
      SCIPdebugMessage("Decomposition has linking variables and is arrowhead.\n");
      SCIP_CALL( DECdecompSetType(decomp, DEC_DECTYPE_ARROWHEAD) );
   }

   if( !staircase )
   {
      SCIP_CALL( DECdecompCheckConsistency(scip, decomp) );
      return SCIP_OKAY;
   }

   SCIP_CALL( SCIPhashmapCreate(&varindex, SCIPblkmem(scip), nvars) );
   SCIP_CALL( SCIPhashmapCreate(&consindex, SCIPblkmem(scip), nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &stairlinkingvars, nblocks) );
   SCIP_CALL( SCIPallocBufferArray(scip, &nstairlinkingvars, nblocks) );

   for( i = 0; i < nblocks; ++i )
   {
      SCIP_CALL( SCIPallocBufferArray(scip, &(stairlinkingvars[i]), nvars) ); /*lint !e866*/
      nstairlinkingvars[i] = 0;
   }

   nsubscipconss = DECdecompGetNSubscipconss(decomp);
   subscipconss = DECdecompGetSubscipconss(decomp);
   nsubscipvars = DECdecompGetNSubscipvars(decomp);

   cindex = 0;
   cumindex = 0;

   /* try to deduce staircase map */
   for( b = 0; b < nblocks; ++b )
   {
      int idx = 0;
      SCIPdebugMessage("block %d (%d vars):\n", b, nsubscipvars[b]);

      for( i = 0; i < nsubscipconss[b]; ++i )
      {

         int linkindex = 0;
         SCIP_CONS* cons = subscipconss[b][i];

         SCIP_CALL( SCIPhashmapInsert(consindex, cons, (void*)(size_t) (cindex+1)) );
         ++cindex;
         SCIP_CALL( SCIPgetConsNVars(scip, cons, &ncurvars, &success) );
         assert(success);

         SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );

         SCIP_CALL( SCIPgetConsVars(scip, cons, curvars, ncurvars, &success) );
         assert(success);

         for( j = 0; j < ncurvars; ++j )
         {
            SCIP_VAR* probvar = SCIPvarGetProbvar(curvars[j]);

            if( SCIPvarGetStatus(probvar) == SCIP_VARSTATUS_FIXED )
               continue;

            /* if the variable is linking */
            if( (int)(size_t)SCIPhashmapGetImage(vartoblock, probvar) >= nblocks+1 ) /*lint !e507*/
            {
               /* if it has not been already assigned, it links to the next block */
               if( !SCIPhashmapExists(varindex, probvar) )
               {
                  int vindex = cumindex+nsubscipvars[b]+linkindex+1;
                  SCIPdebugMessage("assigning link var <%s> to index <%d>\n", SCIPvarGetName(probvar), vindex);
                  SCIP_CALL( SCIPhashmapInsert(varindex, probvar, (void*)(size_t)(vindex)) );
                  stairlinkingvars[b][nstairlinkingvars[b]] = probvar;
                  ++(nstairlinkingvars[b]);
                  linkindex++;
               }
            }
            else
            {
               if( !SCIPhashmapExists(varindex, probvar) )
               {
                  int vindex = cumindex+idx+1;
                  assert(((int) (size_t) SCIPhashmapGetImage(vartoblock, probvar)) -1 == b);  /*lint !e507*/
                  SCIPdebugMessage("assigning block var <%s> to index <%d>\n", SCIPvarGetName(probvar), vindex);
                  SCIP_CALL( SCIPhashmapInsert(varindex, probvar, (void*)(size_t)(vindex)) );
                  ++idx;
               }
            }
         }
         SCIPfreeBufferArray(scip, &curvars);
      }
      if( b < nblocks-1 )
      {
         cumindex += nsubscipvars[b] + nstairlinkingvars[b];
      }
   }

   DECdecompSetVarindex(decomp, varindex);
   DECdecompSetConsindex(decomp, consindex);
   SCIP_CALL( DECdecompSetType(decomp, DEC_DECTYPE_STAIRCASE) );

   for( b = nblocks-1; b >= 0; --b )
   {
      if( nstairlinkingvars[b] == 0 )
      {
         SCIPfreeBufferArrayNull(scip, &(stairlinkingvars[b]));
      }
   }

   SCIP_CALL( DECdecompSetStairlinkingvars(scip, decomp, stairlinkingvars, nstairlinkingvars) );

   for( b = nblocks-1; b >= 0; --b )
   {
      SCIPfreeBufferArrayNull(scip, &stairlinkingvars[b]);
   }
   SCIPfreeBufferArray(scip, &nstairlinkingvars);
   SCIPfreeBufferArray(scip, &stairlinkingvars);

   SCIP_CALL( DECdecompCheckConsistency(scip, decomp) );

   return SCIP_OKAY;
}

/** completely fills out decomposition structure from only the constraint partition in the following manner:
 *  given constraint block/border assignment (by constoblock), one gets the following assignment of probvars:
 *  let C(j) be the set of constraints containing variable j, set block of j to
 *  (i)   constoblock(i) iff constoblock(i1) == constoblock(i2) for all i1,i2 in C(j) with constoblock(i1) != nblocks+1 && constoblock(i2) != nblocks+1
 *  (ii)  nblocks+2 ["linking var"] iff exists i1,i2 with constoblock(i1) != constoblock(i2) && constoblock(i1) != nblocks+1 && constoblock(i2) != nblocks+1
 *  (iii) nblocks+1 ["master var"] iff constoblock(i) == nblocks+1 for all i in C(j)
 */
SCIP_RETCODE DECfilloutDecompFromConstoblock(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_HASHMAP*         constoblock,        /**< constraint to block hashmap, start with 1 for first block and nblocks+1 for linking constraints */
   int                   nblocks,            /**< number of blocks */
   SCIP_Bool             staircase           /**< should the decomposition be a staircase structure */
   )
{
   SCIP_HASHMAP* vartoblock;
   int i;
   int j;
   SCIP_VAR** vars;
   int nvars;
   SCIP_CONS** conss;
   int nconss;
   SCIP_VAR** curvars;
   int ncurvars;
   SCIP_Bool haslinking;
   SCIP_Bool success;
   SCIP_RETCODE retcode;

   assert(scip != NULL);
   assert(decomp != NULL);
   assert(constoblock != NULL);
   assert(nblocks >= 0);

   conss = SCIPgetConss(scip);
   nconss = SCIPgetNConss(scip);
   vars = SCIPgetVars(scip);
   nvars = SCIPgetNVars(scip);

   assert(vars != NULL);
   assert(nvars > 0);
   assert(conss != NULL);
   assert(nconss > 0);

   SCIP_CALL( SCIPhashmapCreate(&vartoblock, SCIPblkmem(scip), nvars) );
   haslinking = FALSE;
   for( i = 0; i < nconss; ++i )
   {
      int consblock;

      SCIP_CALL( SCIPgetConsNVars(scip, conss[i], &ncurvars, &success) );
      assert(success);

      if( ncurvars == 0 )
         continue;

      consblock = (int)(size_t)SCIPhashmapGetImage(constoblock, conss[i]);  /*lint !e507*/

      assert(consblock > 0 && consblock <= nblocks+1);
      if( consblock == nblocks+1 )
      {
         SCIPdebugMessage("cons <%s> is linking and need not be handled\n", SCIPconsGetName(conss[i]));
         continue;
      }

      SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );

      SCIP_CALL( SCIPgetConsVars(scip, conss[i], curvars, ncurvars, &success) );
      assert(success);
      SCIPdebugMessage("cons <%s> (%d vars) is in block %d.\n", SCIPconsGetName(conss[i]), ncurvars, consblock);
      assert(consblock <= nblocks);

      for( j = 0; j < ncurvars; ++j )
      {
         int varblock;
         SCIP_VAR* probvar = SCIPvarGetProbvar(curvars[j]);

         if( SCIPvarGetStatus(probvar) == SCIP_VARSTATUS_FIXED )
            continue;

         assert(SCIPvarIsActive(probvar));

         if( SCIPhashmapExists(vartoblock, probvar) )
            varblock = (int) (size_t) SCIPhashmapGetImage(vartoblock, probvar); /*lint !e507*/
         else
            varblock = nblocks+1;

         /* The variable is currently in no block */
         if( varblock == nblocks+1 )
         {
            SCIPdebugMessage(" var <%s> not been handled before, adding to block %d\n", SCIPvarGetName(probvar), consblock);
            SCIP_CALL( SCIPhashmapSetImage(vartoblock, probvar, (void*) (size_t) consblock) );
         }
         /* The variable is already in a different block */
         else if( varblock != consblock )
         {
            assert(varblock <= nblocks || varblock == nblocks+2);
            SCIPdebugMessage(" var <%s> has been handled before, adding to linking (%d != %d)\n", SCIPvarGetName(probvar), consblock, varblock);
            SCIP_CALL( SCIPhashmapSetImage(vartoblock, probvar, (void*) (size_t) (nblocks+2)) );
            haslinking = TRUE;
         }
         else
         {
            assert(consblock == varblock);
            SCIPdebugMessage(" var <%s> is handled and in same block as cons (%d == %d).\n", SCIPvarGetName(probvar), consblock, varblock);
         }
      }

      SCIPfreeBufferArray(scip, &curvars);
   }

   /* Handle variables that do not appear in any pricing problem, those will be copied directly to the master */
   for( i = 0; i < nvars; ++i )
   {
      if( !SCIPhashmapExists(vartoblock, vars[i]) )
      {
         SCIPdebugMessage(" var <%s> not handled at all and now in master\n", SCIPvarGetName(vars[i]));
         SCIP_CALL( SCIPhashmapSetImage(vartoblock, vars[i], (void*) (size_t) (nblocks+1)) );
      }
   }

   retcode = DECfilloutDecompFromHashmaps(scip, decomp, vartoblock, constoblock, nblocks, staircase && haslinking);
   if( retcode != SCIP_OKAY )
   {
      SCIPhashmapFree(&vartoblock);
      return retcode;
   }

   return SCIP_OKAY;
}

/** sets the detector for the given decomposition */
void DECdecompSetDetector(
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   DEC_DETECTOR*         detector            /**< detector data structure */
   )
{
   assert(decomp != NULL);

   decomp->detector = detector;
}

/** gets the detector for the given decomposition */
DEC_DETECTOR* DECdecompGetDetector(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);

   return decomp->detector;
}

/** gets the detectors for the given decomposition */
DEC_DETECTOR** DECdecompGetDetectorChain(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);

   return decomp->detectorchain;
}


SCIP_RETCODE DECdecompSetDetectorChain(
   SCIP*                 scip,
   DEC_DECOMP*           decomp,
   DEC_DETECTOR**        detectors,
   int                   ndetectors
   )
{
   int d;

   /* resize detectorchain */
   int size = SCIPcalcMemGrowSize( scip, ndetectors);
   SCIP_CALL_ABORT( SCIPallocBlockMemoryArray( scip, &decomp->detectorchain, size ) );

   /* clear former */
   BMSclearMemoryArray(decomp->detectorchain, size);

   for( d = 0; d < ndetectors; ++d )
   {
      decomp->detectorchain[d] = detectors[d];
   }

   decomp->sizedetectorchain = ndetectors;
   return SCIP_OKAY;
}


/** gets the number of detectors for the given decomposition */
int DECdecompGetDetectorChainSize(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);

   return decomp->sizedetectorchain;
}


/** sets the id of the original partialdec */
void DECdecompSetPartialdecID(
   DEC_DECOMP*           decomp,              /**< decomposition data structure */
   int                   id                   /**< ID of partialdec */
   )
{
   assert(decomp != NULL);
   assert(id >= 0);

   decomp->partialdecid = id;
}

/** gets the id of the original partialdec */
int DECdecompGetPartialdecID(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   assert(decomp != NULL);
   return decomp->partialdecid;
}


/** sets the detector clock times of the detectors of the detector chain */
extern
void DECdecompSetDetectorClockTimes(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_Real*            detectorClockTimes  /**< time used by the detectors */
   )
{
   int d;

   int size;
   size = SCIPcalcMemGrowSize(scip, decomp->sizedetectorchain);

   assert(decomp->sizedetectorchain > 0);
   SCIP_CALL_ABORT( SCIPallocBlockMemoryArray(scip, &decomp->detectorclocktimes, size) );

   BMSclearMemoryArray(decomp->detectorclocktimes, size);

   for ( d = 0; d < decomp->sizedetectorchain; ++d )
   {
      decomp->detectorclocktimes[d] = detectorClockTimes[d];
   }

   return;
}

/** gets the detector clock times of the detectors of the detector chain */
SCIP_Real* DECdecompGetDetectorClockTimes(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   return decomp->detectorclocktimes;
}

/** sets the detector clock times of the detectors of the detector chain */
extern
SCIP_RETCODE DECdecompSetDetectorChainString(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,              /**< decomposition data structure */
   const char*           detectorchainstring  /**< string for the detector information working on that decomposition */
   )
{
   SCIP_CALL (SCIPduplicateBlockMemoryArray(scip, &(decomp->detectorchainstring), detectorchainstring, SCIP_MAXSTRLEN ) );
   return SCIP_OKAY;

}

/** sets the detector clock times of the detectors of the detector chain */
extern
char* DECdecompGetDetectorChainString(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   return decomp->detectorchainstring;
}


/** sets the percentages of variables assigned to the border of the corresponding detectors (of the detector chain) on this decomposition */
void DECdecompSetDetectorPctVarsToBorder(
   SCIP*                 scip,              /**< SCIP data structure */
   DEC_DECOMP*           decomp,            /**< decomposition data structure */
   SCIP_Real*            pctVarsToBorder    /**< percentage of variables assigned to border */
   )
{
   int d;
   int size;
   size = SCIPcalcMemGrowSize(scip, decomp->sizedetectorchain);

   assert(decomp->sizedetectorchain > 0);

   SCIP_CALL_ABORT( SCIPallocBlockMemoryArray(scip, &decomp->pctvarstoborder, size) );

   BMSclearMemoryArray(decomp->pctvarstoborder, size);

   for ( d = 0; d < decomp->sizedetectorchain; ++d )
   {
      decomp->pctvarstoborder[d] = pctVarsToBorder[d];
   }

   return;


}

/** gets the percentages of variables assigned to the border of the corresponding detectors (of the detector chain) on this decomposition */
extern
SCIP_Real* DECdecompGetDetectorPctVarsToBorder(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   return decomp->pctvarstoborder;
}

/** sets the percentages of constraints assigned to the border of the corresponding detectors (of the detector chain) on this decomposition */
void DECdecompSetDetectorPctConssToBorder(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_Real*            pctConssToBorder    /**< percentage of constraints assigned to border */
   )
{

   int d;
   int size;
   size = SCIPcalcMemGrowSize(scip, decomp->sizedetectorchain);


   assert(decomp->sizedetectorchain > 0);

   SCIP_CALL_ABORT( SCIPallocBlockMemoryArray(scip, &decomp->pctconsstoborder, size) );

   BMSclearMemoryArray(decomp->pctconsstoborder, size);

   for ( d = 0; d < decomp->sizedetectorchain; ++d )
   {
      decomp->pctconsstoborder[d] = pctConssToBorder[d];
   }

   return;
}

/** gets the percentages of constraints assigned to the border of the corresponding detectors (of the detector chain) on this decomposition */
SCIP_Real* DECdecompGetDetectorPctConssToBorder(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   return decomp->pctconsstoborder;
}

/** sets the percentages of variables assigned to some block of the corresponding detectors (of the detector chain) on this decomposition */
void DECdecompSetDetectorPctVarsToBlock(
   SCIP*                 scip,             /**< SCIP data structure */
   DEC_DECOMP*           decomp,           /**< decomposition data structure */
   SCIP_Real*            pctVarsToBlock    /**< percentage of variables assigned to some block in the detector chain */
   )
{
   int d;
   int size;

    assert(decomp->sizedetectorchain > 0);

    size = SCIPcalcMemGrowSize(scip, decomp->sizedetectorchain);


    SCIP_CALL_ABORT( SCIPallocBlockMemoryArray(scip, &decomp->pctvarstoblock, size) );

    BMSclearMemoryArray(decomp->pctvarstoblock, size);

    for ( d = 0; d < decomp->sizedetectorchain; ++d )
    {
       decomp->pctvarstoblock[d] = pctVarsToBlock[d];
    }

    return;
 }

/** gets the percentages of variables assigned to some block of the corresponding detectors (of the detector chain) on this decomposition */
SCIP_Real* DECdecompGetDetectorPctVarsToBlock(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   return decomp->pctvarstoblock;
}

/** sets the percentages of constraints assigned to some block of the corresponding detectors (of the detector chain) on this decomposition */
void DECdecompSetDetectorPctConssToBlock(
   SCIP*                 scip,              /**< SCIP data structure */
   DEC_DECOMP*           decomp,            /**< decomposition data structure */
   SCIP_Real*            pctConssToBlock    /**< percentage of constraints assigned to some block in the detector chain */
   )
{
   int d;

   int size;

   assert(decomp->sizedetectorchain > 0);

   size = SCIPcalcMemGrowSize(scip, decomp->sizedetectorchain);

   SCIP_CALL_ABORT( SCIPallocBlockMemoryArray(scip, &decomp->pctconsstoblock, size) );

   BMSclearMemoryArray(decomp->pctconsstoblock, size);

   for ( d = 0; d < decomp->sizedetectorchain; ++d )
   {
      decomp->pctconsstoblock[d] = pctConssToBlock[d];
      }

   return;
}

/** gets the percentages of constraints assigned to some block of the corresponding detectors (of the detector chain) on this decomposition */
extern
SCIP_Real* DECdecompGetDetectorPctConssToBlock(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   return decomp->pctconsstoblock;
}


/** sets the percentages of variables assigned to some block of the corresponding detectors (of the detector chain) on this decomposition */
void DECdecompSetDetectorPctVarsFromOpen(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_Real*            pctVarsFromOpen     /**< percentage of open variables assigned to some block in the detector chain */
   )
{
   int d;
   int size;

   assert(decomp->sizedetectorchain > 0);


   size = SCIPcalcMemGrowSize(scip, decomp->sizedetectorchain);


   SCIP_CALL_ABORT( SCIPallocBlockMemoryArray(scip, &decomp->pctvarsfromopen, size) );

   BMSclearMemoryArray(decomp->pctvarsfromopen, size);

   for ( d = 0; d < decomp->sizedetectorchain; ++d )
   {
      decomp->pctvarsfromopen[d] = pctVarsFromOpen[d];
   }

   return;
}

/** gets the percentages of variables assigned to some block of the corresponding detectors (of the detector chain) on this decomposition */
SCIP_Real* DECdecompGetDetectorPctVarsFromOpen(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   return decomp->pctvarsfromopen;
}

/** sets the percentages of constraints assigned to some block of the corresponding detectors (of the detector chain) on this decomposition */
void DECdecompSetDetectorPctConssFromOpen(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_Real*            pctConssFromOpen    /**< percentage of open variables assigned to some block in the detector chain */
   )
{
   int d;
   int size;

   assert(decomp->sizedetectorchain > 0);


   size = SCIPcalcMemGrowSize(scip, decomp->sizedetectorchain);


   SCIP_CALL_ABORT( SCIPallocBlockMemoryArray(scip, &decomp->pctconssfromopen, size) );

   BMSclearMemoryArray(decomp->pctconssfromopen, size);

   for ( d = 0; d < decomp->sizedetectorchain; ++d )
   {
      decomp->pctconssfromopen[d] = pctConssFromOpen[d];
   }

   return;
}

/** gets the percentages of constraints assigned to some block of the corresponding detectors (of the detector chain)
 *  on this decomposition */
SCIP_Real* DECdecompGetDetectorPctConssFromOpen(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   return decomp->pctconssfromopen;
}

/** sets the number of new blocks of the corresponding detectors (of the detector chain) on this decomposition */
void DECdecompSetNNewBlocks(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   int*                  nNewBlocks          /**< number of newly found blocks in this decomposition */
   )
{
   int d;
   int size;

   assert(decomp->sizedetectorchain > 0);

   size = SCIPcalcMemGrowSize(scip, decomp->sizedetectorchain);

   SCIP_CALL_ABORT( SCIPallocBlockMemoryArray(scip, &decomp->nnewblocks, size) );

   BMSclearMemoryArray(decomp->nnewblocks, size);

   for ( d = 0; d < decomp->sizedetectorchain; ++d )
   {
      decomp->nnewblocks[d] = nNewBlocks[d];
   }

   return;
}

/** gets the number of new blocks corresponding detectors (of the detector chain) on this decomposition */
int* DECdecompGetNNewBlocks(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   return decomp->nnewblocks;
}





/** transforms all constraints and variables, updating the arrays */
SCIP_RETCODE DECdecompTransform(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   int b;
   int c;
   int v;
   SCIP_HASHMAP* newconstoblock;
   SCIP_HASHMAP* newvartoblock;
   SCIP_VAR* newvar;

   assert(SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED);

   SCIP_CALL( SCIPhashmapCreate(&newconstoblock, SCIPblkmem(scip), SCIPgetNConss(scip)) );
   SCIP_CALL( SCIPhashmapCreate(&newvartoblock, SCIPblkmem(scip), SCIPgetNVars(scip)) );

   /* transform all constraints and put them into constoblock */
   for( b = 0; b < decomp->nblocks; ++b )
   {
      for( c = 0; c < decomp->nsubscipconss[b]; ++c )
      {
         SCIP_CONS* newcons;
         SCIPdebugMessage("%d, %d: %s (%p, %s)\n", b, c, SCIPconsGetName(decomp->subscipconss[b][c]),
            (void*) decomp->subscipconss[b][c], SCIPconsIsTransformed(decomp->subscipconss[b][c])?"t":"o" );
         assert(decomp->subscipconss[b][c] != NULL);
         newcons = SCIPfindCons(scip, SCIPconsGetName(decomp->subscipconss[b][c]));
         if( newcons != decomp->subscipconss[b][c] )
         {
            SCIP_CALL( SCIPcaptureCons(scip, newcons) );
            SCIP_CALL( SCIPreleaseCons(scip, &(decomp->subscipconss[b][c])) );
            decomp->subscipconss[b][c] = newcons;
         }
         assert(decomp->subscipconss[b][c] != NULL);
         assert(!SCIPhashmapExists(newconstoblock, decomp->subscipconss[b][c]));
         SCIP_CALL( SCIPhashmapSetImage(newconstoblock, decomp->subscipconss[b][c], (void*) (size_t) (b+1)) );
      }
   }
   /* transform all variables and put them into vartoblock */
   for( b = 0; b < decomp->nblocks; ++b )
   {
      int idx;
      for( v = 0, idx = 0; v < decomp->nsubscipvars[b]; ++v )
      {
         assert(decomp->subscipvars[b][v] != NULL);

         SCIPdebugMessage("%d, %d: %s (%p, %s)\n", b, v, SCIPvarGetName(decomp->subscipvars[b][v]),
            (void*)decomp->subscipvars[b][v], SCIPvarIsTransformed(decomp->subscipvars[b][v])?"t":"o" );

         /* make sure that newvar is a transformed variable */
         SCIP_CALL( SCIPgetTransformedVar(scip, decomp->subscipvars[b][v], &newvar) );
         SCIP_CALL( SCIPreleaseVar(scip, &(decomp->subscipvars[b][v])) );

         assert(newvar != NULL);
         assert(SCIPvarIsTransformed(newvar));

         newvar = SCIPvarGetProbvar(newvar);
         assert(newvar != NULL);

         /* the probvar can also be fixed, in which case we do not need it in the block; furthermore, multiple variables
          * can resolve to the same active problem variable, so we check whether we already handled the variable
          * @todo: why do we ignore fixed variables? They could still be present in constraints?
          */
         if( SCIPvarIsActive(newvar) && !SCIPhashmapExists(newvartoblock, newvar) )
         {
            decomp->subscipvars[b][idx] = newvar;
            SCIP_CALL( SCIPcaptureVar(scip, newvar) );
            SCIPdebugMessage("%d, %d: %s (%p, %s)\n", b, v, SCIPvarGetName(decomp->subscipvars[b][idx]),
               (void*)decomp->subscipvars[b][idx], SCIPvarIsTransformed(decomp->subscipvars[b][idx])?"t":"o" );

            assert(decomp->subscipvars[b][idx] != NULL);
            assert(!SCIPhashmapExists(newvartoblock, decomp->subscipvars[b][idx]));
            SCIP_CALL( SCIPhashmapSetImage(newvartoblock, decomp->subscipvars[b][idx], (void*) (size_t) (b+1)) );
            ++idx;
         }
      }
      decomp->nsubscipvars[b] = idx;
   }

   /* transform all linking constraints */
   for( c = 0; c < decomp->nlinkingconss; ++c )
   {
      SCIP_CONS* newcons;

      SCIPdebugMessage("m, %d: %s (%s)\n", c, SCIPconsGetName(decomp->linkingconss[c]), SCIPconsIsTransformed(decomp->linkingconss[c])?"t":"o" );
      assert(decomp->linkingconss[c] != NULL);
      newcons = SCIPfindCons(scip, SCIPconsGetName(decomp->linkingconss[c]));
      if( newcons != decomp->linkingconss[c] )
      {
         SCIP_CALL( SCIPcaptureCons(scip, newcons) );
         SCIP_CALL( SCIPreleaseCons(scip, &(decomp->linkingconss[c])) );
         decomp->linkingconss[c] = newcons;
      }
      SCIP_CALL( SCIPhashmapSetImage(newconstoblock, decomp->linkingconss[c],(void*) (size_t) (decomp->nblocks+1) ) );

      assert(decomp->linkingconss[c] != NULL);
   }

   /* transform all linking variables */
   for( v = 0; v < decomp->nlinkingvars; ++v )
   {
      int block;
      SCIPdebugMessage("m, %d: %s (%p, %s)\n", v, SCIPvarGetName(decomp->linkingvars[v]),
         (void*)decomp->linkingvars[v], SCIPvarIsTransformed(decomp->linkingvars[v])?"t":"o");
      assert(decomp->linkingvars[v] != NULL);

      if( !SCIPvarIsTransformed(decomp->linkingvars[v]) )
      {
         SCIP_CALL( SCIPgetTransformedVar(scip, decomp->linkingvars[v], &newvar) );
         newvar = SCIPvarGetProbvar(newvar);
      }
      else
         newvar = decomp->linkingvars[v];

      block = (int) (size_t) SCIPhashmapGetImage(decomp->vartoblock, decomp->linkingvars[v]); /*lint !e507*/
      assert(block == decomp->nblocks +1 || block == decomp->nblocks +2);
      assert(newvar != NULL);
      assert(SCIPvarIsTransformed(newvar));
      SCIP_CALL( SCIPreleaseVar(scip, &(decomp->linkingvars[v])) );

      decomp->linkingvars[v] = newvar;
      SCIP_CALL( SCIPcaptureVar(scip, decomp->linkingvars[v]) );
      SCIP_CALL( SCIPhashmapSetImage(newvartoblock, decomp->linkingvars[v], (void*) (size_t) (block) ) );
      SCIPdebugMessage("m, %d: %s (%p, %s)\n", v, SCIPvarGetName(decomp->linkingvars[v]),
         (void*)decomp->linkingvars[v], SCIPvarIsTransformed(decomp->linkingvars[v])?"t":"o");
      assert(decomp->linkingvars[v] != NULL);
   }

   SCIPhashmapFree(&decomp->constoblock);
   decomp->constoblock = newconstoblock;
   SCIPhashmapFree(&decomp->vartoblock);
   decomp->vartoblock = newvartoblock;

   SCIP_CALL( DECdecompCheckConsistency(scip, decomp) );

   return SCIP_OKAY;
}

/**
 * Remove all those constraints that were removed from the problem after the decomposition had been created
 */
SCIP_RETCODE DECdecompRemoveDeletedConss(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decdecomp           /**< decomposition data structure */
   )
{
   int block;

   int c;
   int pos;

   assert(scip != NULL);
   assert(decdecomp != NULL);

   for( block = 0; block < decdecomp->nblocks; ++block )
   {
      for( c = 0, pos = 0; c < decdecomp->nsubscipconss[block]; ++c )
      {
         if( !SCIPconsIsDeleted(decdecomp->subscipconss[block][c]) )
         {
            decdecomp->subscipconss[block][pos] = decdecomp->subscipconss[block][c];
            ++pos;
         }
         else
         {
            SCIP_CALL( SCIPreleaseCons(scip, &decdecomp->subscipconss[block][c]) );
         }
      }
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &decdecomp->subscipconss[block],
         SCIPcalcMemGrowSize(scip, decdecomp->nsubscipconss[block]), SCIPcalcMemGrowSize(scip, pos)) );
      decdecomp->nsubscipconss[block] = pos;
   }

   for( c = 0, pos = 0; c < decdecomp->nlinkingconss; ++c )
   {
      if( !SCIPconsIsDeleted(decdecomp->linkingconss[c]) )
      {
         decdecomp->linkingconss[pos] = decdecomp->linkingconss[c];
         ++pos;
      }
      else
      {
         SCIP_CALL( SCIPreleaseCons(scip, &decdecomp->linkingconss[c]) );
      }
   }

   if( pos != decdecomp->nlinkingconss && decdecomp->linkingconss != NULL )
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &decdecomp->linkingconss,
         SCIPcalcMemGrowSize(scip, decdecomp->nlinkingconss), SCIPcalcMemGrowSize(scip, pos)) );
   decdecomp->nlinkingconss = pos;

   return SCIP_OKAY;
}

/**
 * Adds all those constraints that were added to the problem after the decomposition had been created
 */
SCIP_RETCODE DECdecompAddRemainingConss(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decdecomp           /**< decomposition data structure */
   )
{
   int c;

   assert(scip != NULL);
   assert(decdecomp != NULL);

   for( c = 0; c < SCIPgetNConss(scip); ++c )
   {
      SCIP_CONS* cons;
      cons = SCIPgetConss(scip)[c];


      if( !GCGisConsGCGCons(cons) && !SCIPhashmapExists(DECdecompGetConstoblock(decdecomp), cons) )
      {
         int block;
         SCIP_CALL( DECdetermineConsBlock(scip, decdecomp, cons, &block) );
         SCIPdebugMessage("add remaining: cons <%s> in block %d/%d\n", SCIPconsGetName(cons), block, DECdecompGetNBlocks(decdecomp) );

         /* If the constraint has only variables appearing in the master only,
          * we assign it to the master rather than creating a new block
          */
         if( block == -1 || (block >= 0 && block == DECdecompGetNBlocks(decdecomp) ) )
         {
            if( decdecomp->nlinkingconss == 0 )
            {
               int newsize;
               newsize = SCIPcalcMemGrowSize(scip, 1);
               SCIP_CALL( SCIPallocBlockMemoryArray(scip, &decdecomp->linkingconss, newsize) );

               switch( decdecomp->type )
               {
               case DEC_DECTYPE_DIAGONAL:
                  decdecomp->type = DEC_DECTYPE_BORDERED;
                  SCIPwarningMessage(scip, "Decomposition type changed to 'bordered' due to an added constraint.\n");
                  break;
               case DEC_DECTYPE_STAIRCASE:
                  decdecomp->type = DEC_DECTYPE_ARROWHEAD;
                  SCIPwarningMessage(scip, "Decomposition type changed to 'arrowhead' due to an added constraint.\n");
                  break;
               default:
                  break;
               }
            }
            else
            {
               int oldsize;
               int newsize;
               oldsize = SCIPcalcMemGrowSize(scip, decdecomp->nlinkingconss);
               newsize = SCIPcalcMemGrowSize(scip, decdecomp->nlinkingconss+1);
               SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &decdecomp->linkingconss, oldsize, newsize) );
            }
            decdecomp->linkingconss[decdecomp->nlinkingconss] = cons;
            decdecomp->nlinkingconss += 1;
            SCIP_CALL( SCIPhashmapInsert(decdecomp->constoblock, cons, (void*) (size_t) (DECdecompGetNBlocks(decdecomp)+1)) );
         }
         else
         {
            int oldsize;
            int newsize;
            assert(block>=0);

            oldsize = SCIPcalcMemGrowSize(scip, decdecomp->nsubscipconss[block]);
            newsize = SCIPcalcMemGrowSize(scip, decdecomp->nsubscipconss[block]+1);

            assert(decdecomp->nsubscipconss[block] > 0);
            SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &decdecomp->subscipconss[block], oldsize, newsize) ); /*lint !e866*/
            decdecomp->subscipconss[block][decdecomp->nsubscipconss[block]] = cons;
            decdecomp->nsubscipconss[block] += 1;
            SCIP_CALL( SCIPhashmapInsert(decdecomp->constoblock, cons, (void*) (size_t) (block+1)) );
            SCIP_CALL( assignConsvarsToBlock(scip, decdecomp, cons, block) );
         }
         SCIP_CALL( SCIPcaptureCons(scip, cons) );
      }
   }


   return SCIP_OKAY;
}

/** checks the consistency of the data structure
 *
 *  In particular, it checks whether the redundant information in the structure agree and
 *  whether the variables in the structure are both existant in the arrays and in the problem
 */
SCIP_RETCODE DECdecompCheckConsistency(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decdecomp           /**< decomposition data structure */
   )
{
#ifndef NDEBUG
   int c;
   int b;
   int v;


   SCIPdebugMessage("Problem is %stransformed\n", SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED ? "": "not ");

   for( v = 0; v < SCIPgetNVars(scip); ++v )
   {
      if( SCIPisEQ(scip, SCIPvarGetLbGlobal(SCIPgetVars(scip)[v]), SCIPvarGetUbGlobal(SCIPgetVars(scip)[v]) ) && SCIPisEQ(scip, SCIPvarGetUbGlobal(SCIPgetVars(scip)[v]), 0. ) )
         continue;
      assert(SCIPhashmapExists(DECdecompGetVartoblock(decdecomp), SCIPgetVars(scip)[v]));
   }

   for( c = 0; c < SCIPgetNConss(scip); ++c )
   {
      if( !GCGisConsGCGCons(SCIPgetConss(scip)[c]) )
      {
         assert(SCIPhashmapExists(DECdecompGetConstoblock(decdecomp), SCIPgetConss(scip)[c]));
      }
   }

   /* Check whether subscipcons are correct */
   for( b = 0; b < DECdecompGetNBlocks(decdecomp); ++b )
   {
      for( c = 0; c < DECdecompGetNSubscipconss(decdecomp)[b]; ++c )
      {
         SCIP_VAR** curvars;
         int ncurvars;
         SCIP_CONS* cons = DECdecompGetSubscipconss(decdecomp)[b][c];

         SCIPdebugMessage("Cons <%s> in block %d = %d\n", SCIPconsGetName(cons), b, ((int) (size_t) SCIPhashmapGetImage(DECdecompGetConstoblock(decdecomp), cons)) -1);  /*lint !e507*/
         // @todo: remove if check when SCIPfindCons() can be called in stage SCIP_STAGE_INITSOLVE
         if( SCIPgetStage(scip) != SCIP_STAGE_INITSOLVE)
            assert(SCIPfindCons(scip, SCIPconsGetName(cons)) != NULL);
         assert(((int) (size_t) SCIPhashmapGetImage(DECdecompGetConstoblock(decdecomp), cons)) - 1 == b); /*lint !e507*/
         ncurvars = GCGconsGetNVars(scip, cons);
         if ( ncurvars == 0 )
            continue;
         SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );
         SCIP_CALL( GCGconsGetVars(scip, cons, curvars, ncurvars) );

         for( v = 0; v < ncurvars; ++v )
         {
            int varblock;
            SCIP_VAR* var = SCIPvarGetProbvar(curvars[v]);

            if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_FIXED || SCIPvarGetLbGlobal(var) == SCIPvarGetUbGlobal(var)  )
               continue;

            varblock = ((int) (size_t) SCIPhashmapGetImage(DECdecompGetVartoblock(decdecomp), var)) - 1;  /*lint !e507*/
            SCIPdebugMessage("\tVar <%s> in block %d = %d\n", SCIPvarGetName(var), b, varblock);

            assert(SCIPfindVar(scip, SCIPvarGetName(var)) != NULL);
            assert(SCIPvarIsActive(var));
            assert(varblock == b || varblock == DECdecompGetNBlocks(decdecomp)+1 );

         }
         SCIPfreeBufferArray(scip, &curvars);
      }

      assert((DECdecompGetSubscipvars(decdecomp)[b] == NULL) == (DECdecompGetNSubscipvars(decdecomp)[b] == 0));


      for( v = 0; v < DECdecompGetNSubscipvars(decdecomp)[b]; ++v )
      {
         int varblock;
         SCIP_VAR* var = DECdecompGetSubscipvars(decdecomp)[b][v];
         varblock = ((int) (size_t) SCIPhashmapGetImage(DECdecompGetVartoblock(decdecomp), var)) - 1; /*lint !e507*/
         SCIPdebugMessage("Var <%s> in block %d = %d\n", SCIPvarGetName(var), b, varblock);
         assert(SCIPfindVar(scip, SCIPvarGetName(var)) != NULL);
         assert(SCIPvarIsActive(var));
         assert(varblock == b || varblock == DECdecompGetNBlocks(decdecomp)+1);
      }
   }

   /* check linking constraints and variables */
   for( v = 0; v < DECdecompGetNLinkingvars(decdecomp); ++v )
   {
      int varblock;
      varblock = (int) (size_t) SCIPhashmapGetImage(DECdecompGetVartoblock(decdecomp), DECdecompGetLinkingvars(decdecomp)[v]); /*lint !e507*/
      assert( varblock == DECdecompGetNBlocks(decdecomp) +1 || varblock == DECdecompGetNBlocks(decdecomp)+2); /*lint !e507*/
   }
   for (c = 0; c < DECdecompGetNLinkingconss(decdecomp); ++c)
   {
      assert(((int) (size_t) SCIPhashmapGetImage(DECdecompGetConstoblock(decdecomp), DECdecompGetLinkingconss(decdecomp)[c])) -1 ==  DECdecompGetNBlocks(decdecomp)); /*lint !e507*/
   }

   switch( DECdecompGetType(decdecomp) )
   {
   case DEC_DECTYPE_UNKNOWN:
         assert(FALSE);
      break;
   case DEC_DECTYPE_ARROWHEAD:
      assert(DECdecompGetNLinkingvars(decdecomp) > 0 || DECdecompGetNTotalStairlinkingvars(decdecomp) > 0);
      break;
   case DEC_DECTYPE_BORDERED:
      assert(DECdecompGetNLinkingvars(decdecomp) == 0 && DECdecompGetNLinkingconss(decdecomp) > 0);
      break;
   case DEC_DECTYPE_DIAGONAL:
      assert(DECdecompGetNLinkingvars(decdecomp) == 0 && DECdecompGetNLinkingconss(decdecomp) == 0);
      break;
   case DEC_DECTYPE_STAIRCASE:
      assert(DECdecompGetNLinkingvars(decdecomp) > 0 && DECdecompGetNLinkingconss(decdecomp) == 0);
      break;
   default:
         assert(FALSE);
         break;
   }

#endif
   return SCIP_OKAY;
}

/** creates a decomposition with all constraints in the master */
SCIP_RETCODE DECcreateBasicDecomp(
   SCIP*                 scip,                /**< SCIP data structure */
   DEC_DECOMP**          decomp,              /**< decomposition data structure */
   SCIP_Bool             solveorigprob        /**< is the original problem being solved? */
   )
{
   SCIP_HASHMAP* constoblock;
   SCIP_HASHMAP* vartoblock;
   SCIP_CONS** conss;
   SCIP_VAR**  vars;
   SCIP_Bool haslinking;
   int nblocks;
   int nconss;
   int nvars;
   int c;
   int v;

   assert(scip != NULL);
   assert(decomp != NULL);

   SCIP_CALL( DECdecompCreate(scip, decomp) );
   conss = SCIPgetConss(scip);
   nconss = SCIPgetNConss(scip);
   vars = SCIPgetVars(scip);
   nvars = SCIPgetNVars(scip);

   SCIP_CALL( SCIPhashmapCreate(&constoblock, SCIPblkmem(scip), nconss) );
   SCIP_CALL( SCIPhashmapCreate(&vartoblock, SCIPblkmem(scip), nvars) );
   haslinking = FALSE;


   for( c = 0; c < nconss; ++c )
   {
      if( GCGisConsGCGCons(conss[c]) )
         continue;
      SCIP_CALL( SCIPhashmapInsert(constoblock, conss[c], (void*) (size_t) 1 ) );
   }

   for( v = 0; v < nvars; ++v )
   {
      SCIP_VAR* probvar = SCIPvarGetProbvar(vars[v]);

      if( SCIPvarGetStatus(probvar) == SCIP_VARSTATUS_FIXED )
         continue;
      SCIP_CALL( SCIPhashmapInsert(vartoblock, probvar, (void*) (size_t) 1 ) );
      }

   if( solveorigprob || SCIPgetNVars(scip) == 0 || SCIPgetNConss(scip) == 0 )
      nblocks = 0;
   else
      nblocks = 1;

   DECfilloutDecompFromHashmaps(scip, *decomp, vartoblock, constoblock, nblocks, haslinking);

   DECdecompSetPresolved(*decomp, TRUE);

   return SCIP_OKAY;
}

/**
 * processes block representatives
 *
 * @return returns the number of blocks
 */
static
int processBlockRepresentatives(
   int                   maxblock,           /**< maximal number of blocks */
   int*                  blockrepresentative /**< array blockrepresentatives */
   )
{
   int i;
   int tempblock = 1;

   assert(maxblock >= 1);
   assert(blockrepresentative != NULL );
   SCIPdebugPrintf("Blocks: ");

   /* postprocess blockrepresentatives */
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
      SCIPdebugPrintf("%d ", blockrepresentative[i]);
   }
   SCIPdebugPrintf("\n");
   return tempblock-1;
}

/** */
static
SCIP_RETCODE assignConstraintsToRepresentatives(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           conss,              /**< array of all constraints */
   int                   nconss,             /**< number of constraints */
   SCIP_Bool*            consismaster,       /**< array of flags whether a constraint belongs to the master problem */
   SCIP_HASHMAP*         constoblock,        /**< hashmap from constraints to block numbers, to be filled */
   int*                  vartoblock,         /**< array mapping variables to block numbers, initially all -1, to be set */
   int*                  nextblock,          /**< index of next free block to which no constraints have been assigned yet */
   int*                  blockrepresentative /**< array of blockrepresentatives */
   )
{

   int i;
   int j;
   SCIP_VAR** curvars;
   int ncurvars;

   conss = SCIPgetConss(scip);
   nconss = SCIPgetNConss(scip);

   /* go through the all constraints */
   for( i = 0; i < nconss; ++i )
   {
      int consblock;
      SCIP_CONS* cons = conss[i];

      assert(cons != NULL);
      if( GCGisConsGCGCons(cons) )
         continue;

      if( consismaster[i] )
         continue;

      /* get variables of constraint; ignore empty constraints */
      ncurvars = GCGconsGetNVars(scip, cons);
      curvars = NULL;
      if( ncurvars > 0 )
      {
         SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );
         SCIP_CALL( GCGconsGetVars(scip, cons, curvars, ncurvars) );
      }
      assert(ncurvars >= 0);
      assert(ncurvars <= SCIPgetNVars(scip));
      assert(curvars != NULL || ncurvars == 0);

      assert(SCIPhashmapGetImage(constoblock, cons) == NULL);

      /* if there are no variables, put it in the first block, otherwise put it in the next block */
      if( ncurvars == 0 )
         consblock = -1;
      else
         consblock = *nextblock;

      /* go through all variables */
      for( j = 0; j < ncurvars; ++j )
      {
         SCIP_VAR* probvar;
         int varindex;
         int varblock;

         assert(curvars != NULL);
         probvar = SCIPvarGetProbvar(curvars[j]);
         assert(probvar != NULL);

         /* ignore variables which have been fixed during presolving */
         if( SCIPvarGetStatus(probvar) == SCIP_VARSTATUS_FIXED )
            continue;

         varindex = SCIPvarGetProbindex(probvar);
         assert(varindex >= 0);
         assert(varindex < SCIPgetNVars(scip));

         /** @todo what about deleted variables? */
         /* get block of variable */
         varblock = vartoblock[varindex];

         SCIPdebugMessage("\tVar %s (%d): ", SCIPvarGetName(probvar), varblock);
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
            assert(varblock <= *nextblock);
            vartoblock[varindex] = varblock;
            SCIPdebugPrintf("new in block %d.\n", varblock);
         }
         else
         {
            assert((varblock > 0) && (consblock == varblock));
            SCIPdebugPrintf("no change.\n");
         }

         SCIPdebugPrintf("VARINDEX: %d (%d)\n", varindex, vartoblock[varindex]);
      }

      /* if the constraint belongs to a new block, mark it as such */
      if( consblock == *nextblock )
      {
         assert(consblock > 0);
         blockrepresentative[consblock] = consblock;
         assert(blockrepresentative[consblock] > 0);
         assert(blockrepresentative[consblock] <= *nextblock);
         ++(*nextblock);
      }

      SCIPdebugMessage("Cons %s will be in block %d (next %d)\n", SCIPconsGetName(cons), consblock, *nextblock);

      for( j = 0; j < ncurvars; ++j )
      {
         SCIP_VAR* probvar;
         int varindex;
         int oldblock;

         assert(curvars != NULL);
         probvar = SCIPvarGetProbvar(curvars[j]);
         assert(probvar != NULL);

         /* ignore variables which have been fixed during presolving */
         if( SCIPvarGetStatus(probvar) == SCIP_VARSTATUS_FIXED )
            continue;

         varindex = SCIPvarGetProbindex(probvar);
         assert(varindex >= 0);
         assert(varindex < SCIPgetNVars(scip));

         oldblock = vartoblock[varindex];
         assert((oldblock > 0) && (oldblock <= *nextblock));

         SCIPdebugMessage("\tVar %s ", SCIPvarGetName(probvar));
         if( oldblock != consblock )
         {
            SCIPdebugPrintf("reset from %d to block %d.\n", oldblock, consblock);
            vartoblock[varindex] = consblock;
            SCIPdebugPrintf("VARINDEX: %d (%d)\n", varindex, consblock);

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

      SCIPfreeBufferArrayNull(scip, &curvars);
      assert(consblock >= 1 || consblock == -1);
      assert(consblock <= *nextblock);

      /* store the constraint block */
      if( consblock != -1 )
      {
         SCIPdebugMessage("cons %s in block %d\n", SCIPconsGetName(cons), consblock);
         SCIP_CALL( SCIPhashmapInsert(constoblock, cons, (void*)(size_t)consblock) );
      }
      else
      {
         SCIPdebugMessage("ignoring %s\n", SCIPconsGetName(cons));
      }
   }

   return SCIP_OKAY;
}

/** */
static
SCIP_RETCODE fillConstoblock(
   SCIP_CONS**           conss,              /**< array of all constraints */
   int                   nconss,             /**< number of constraints */
   SCIP_Bool*            consismaster,       /**< array of flags whether a constraint belongs to the master problem */
   int                   nblocks,            /**< number of blocks */
   SCIP_HASHMAP*         constoblock,        /**< current hashmap from constraints to block numbers */
   SCIP_HASHMAP*         newconstoblock,     /**< new hashmap from constraints to block numbers, to be filled */
   int*                  blockrepresentative /**< array of blockrepresentatives */
   )
{
   int i;

   /* convert temporary data to detectordata */
   for( i = 0; i < nconss; ++i )
   {
      int consblock;

      SCIP_CONS* cons = conss[i];

      if( GCGisConsGCGCons(cons) )
         continue;

      if( consismaster[i] )
      {
         SCIP_CALL( SCIPhashmapInsert(newconstoblock, cons, (void*) (size_t) (nblocks+1)) );
         continue;
      }

      if( !SCIPhashmapExists(constoblock, cons) )
         continue;

      consblock = (int) (size_t) SCIPhashmapGetImage(constoblock, cons); /*lint !e507*/
      assert(consblock > 0);
      consblock = blockrepresentative[consblock];
      assert(consblock <= nblocks);
      SCIP_CALL( SCIPhashmapInsert(newconstoblock, cons, (void*)(size_t)consblock) );
      SCIPdebugMessage("%d %s\n", consblock, SCIPconsGetName(cons));
   }
   return SCIP_OKAY;
}

/** creates a decomposition with provided constraints in the master
 * The function will put the remaining constraints in one or more pricing problems
 * depending on whether the subproblems decompose with no variables in common.
 */
SCIP_RETCODE DECcreateDecompFromMasterconss(
   SCIP*                 scip,                /**< SCIP data structure */
   DEC_DECOMP**          decomp,              /**< decomposition data structure */
   SCIP_CONS**           masterconss,         /**< constraints to be put in the master */
   int                   nmasterconss         /**< number of constraints in the master */
   )
{
   SCIP_HASHMAP* constoblock;
   SCIP_HASHMAP* newconstoblock;
   SCIP_CONS** conss;
   int nconss;
   int nvars;
   int nblocks;
   int* blockrepresentative;
   int nextblock = 1;
   SCIP_Bool* consismaster;
   int i;
   int* vartoblock;

   assert(scip != NULL);
   assert(decomp != NULL);
   assert(nmasterconss == 0 || masterconss != NULL);
   assert(SCIPgetStage(scip) >= SCIP_STAGE_TRANSFORMED);

   conss = SCIPgetConss(scip);
   nconss = SCIPgetNConss(scip);
   nvars = SCIPgetNVars(scip);

   assert( nmasterconss <= nconss );

   if( GCGisConsGCGCons(conss[nconss-1]) )
      --nconss;

   nblocks = nconss-nmasterconss+1;
   assert(nblocks > 0);

   SCIP_CALL( SCIPallocBufferArray(scip, &blockrepresentative, nblocks) );
   SCIP_CALL( SCIPallocBufferArray(scip, &consismaster, nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &vartoblock, nvars) );
   SCIP_CALL( SCIPhashmapCreate(&constoblock, SCIPblkmem(scip), nconss) );
   SCIP_CALL( SCIPhashmapCreate(&newconstoblock, SCIPblkmem(scip), nconss) );

   for( i = 0; i < nmasterconss; ++i )
   {
      SCIP_CALL( SCIPhashmapInsert(constoblock, masterconss[i], (void*) (size_t) (nblocks+1)) );
   }

   for( i = 0; i < nconss; ++i )
   {
      assert(!GCGisConsGCGCons(conss[i]));
      consismaster[i] = SCIPhashmapExists(constoblock, conss[i]);
   }

   for( i = 0; i < nvars; ++i )
   {
      vartoblock[i] = -1;
   }

   for( i = 0; i < nblocks; ++i )
   {
      blockrepresentative[i] = -1;
   }

   SCIP_CALL( assignConstraintsToRepresentatives(scip, conss, nconss, consismaster, constoblock, vartoblock, &nextblock, blockrepresentative) );

   /* postprocess blockrepresentatives */
   nblocks = processBlockRepresentatives(nextblock, blockrepresentative);

   /* convert temporary data to detectordata */
   SCIP_CALL( fillConstoblock(conss, nconss, consismaster, nblocks, constoblock, newconstoblock, blockrepresentative) );
   SCIP_CALL( DECdecompCreate(scip, decomp) );
   SCIP_CALL( DECfilloutDecompFromConstoblock(scip, *decomp, newconstoblock, nblocks, FALSE) );

   SCIPfreeBufferArray(scip, &vartoblock);
   SCIPfreeBufferArray(scip, &consismaster);
   SCIPfreeBufferArray(scip, &blockrepresentative);
   SCIPhashmapFree(&constoblock);

   return SCIP_OKAY;
}

/** increase the corresponding count of the variable stats*/
static
void incVarsData(
   SCIP_VAR*              var,                /**< variable to consider */
   int*                   nbinvars,           /**< pointer to array of size nproblems to store number of binary subproblem vars */
   int*                   nintvars,           /**< pointer to array of size nproblems to store number of integer subproblem vars */
   int*                   nimplvars,          /**< pointer to array of size nproblems to store number of implied subproblem vars */
   int*                   ncontvars,          /**< pointer to array of size nproblems to store number of continues subproblem vars */
   int                    nproblems,          /**< size of the arrays*/
   int                    i                   /**< index of the array to increase */
)
{
   assert(var != NULL);
   assert(i >= 0);
   assert(i < nproblems);

   if( nbinvars != NULL && (SCIPvarGetType(var) == SCIP_VARTYPE_BINARY || SCIPvarIsBinary(var)) )
   {
      ++(nbinvars[i]);
      assert(nbinvars[i] > 0);
   }
   if( nintvars != NULL && (SCIPvarGetType(var) == SCIP_VARTYPE_INTEGER && !SCIPvarIsBinary(var)) )
   {
      ++(nintvars[i]);
      assert(nintvars[i] > 0);
   }
   if( nimplvars != NULL && (SCIPvarGetType(var) == SCIP_VARTYPE_IMPLINT) )
   {
      ++(nimplvars[i]);
      assert(nimplvars[i] > 0);
   }
   if( ncontvars != NULL && (SCIPvarGetType(var) == SCIP_VARTYPE_CONTINUOUS) )
   {
      ++(ncontvars[i]);
      assert(ncontvars[i] > 0);
   }
}

/* score methods */

/** return the number of variables and binary, integer, implied integer, continuous variables of all subproblems */
void DECgetSubproblemVarsData(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   int*                  nvars,              /**< pointer to array of size nproblems to store number of subproblem vars or NULL */
   int*                  nbinvars,           /**< pointer to array of size nproblems to store number of binary subproblem vars or NULL */
   int*                  nintvars,           /**< pointer to array of size nproblems to store number of integer subproblem vars or NULL */
   int*                  nimplvars,          /**< pointer to array of size nproblems to store number of implied subproblem vars or NULL */
   int*                  ncontvars,          /**< pointer to array of size nproblems to store number of continuous subproblem vars or NULL */
   int                   nproblems           /**< size of the arrays*/
)
{
   int i;
   int j;

   assert(scip != NULL);
   assert(decomp != NULL);
   assert(nproblems > 0);

   assert(DECdecompGetType(decomp) != DEC_DECTYPE_UNKNOWN);
   if( nvars != NULL )
      BMSclearMemoryArray(nvars, nproblems);
   if( nbinvars != NULL )
      BMSclearMemoryArray(nbinvars, nproblems);
   if( nintvars != NULL )
      BMSclearMemoryArray(nintvars, nproblems);
   if( nimplvars != NULL )
      BMSclearMemoryArray(nimplvars, nproblems);
   if( ncontvars != NULL )
      BMSclearMemoryArray(ncontvars, nproblems);

   for( i = 0; i < nproblems; ++i )
   {
      SCIP_VAR*** subscipvars;
      int* nsubscipvars;

      nsubscipvars = DECdecompGetNSubscipvars(decomp);
      subscipvars = DECdecompGetSubscipvars(decomp);
      if( nvars != NULL )
         nvars[i] = nsubscipvars[i];

      for( j = 0; j < nsubscipvars[i]; ++j )
      {
         incVarsData(subscipvars[i][j], nbinvars, nintvars, nimplvars, ncontvars, nproblems, i);
      }
   }
}


/** return the number of variables and binary, integer, implied integer, continuous variables of the master */
void DECgetLinkingVarsData(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   int*                  nvars,              /**< pointer to store number of linking vars or NULL */
   int*                  nbinvars,           /**< pointer to store number of binary linking vars or NULL */
   int*                  nintvars,           /**< pointer to store number of integer linking vars or NULL */
   int*                  nimplvars,          /**< pointer to store number of implied linking vars or NULL */
   int*                  ncontvars           /**< pointer to store number of continuous linking vars or NULL */
)
{
   int i;
   SCIP_VAR** linkingvars;
   int nlinkingvars;

   assert(scip != NULL);
   assert(decomp != NULL);

   assert(DECdecompGetType(decomp) != DEC_DECTYPE_UNKNOWN);

   nlinkingvars = DECdecompGetNLinkingvars(decomp);
   linkingvars = DECdecompGetLinkingvars(decomp);

   if( nvars != NULL )
      *nvars = nlinkingvars;
   if( nbinvars != NULL )
      *nbinvars = 0;
   if( nintvars != NULL )
      *nintvars = 0;
   if( nimplvars != NULL )
      *nimplvars = 0;
   if( ncontvars != NULL )
      *ncontvars = 0;


   for( i = 0; i < nlinkingvars; ++i )
   {
      incVarsData(linkingvars[i], nbinvars, nintvars, nimplvars, ncontvars, 1, 0);
   }
}

/**
 * returns the number of nonzeros of each column of the constraint matrix both in the subproblem and in the master
 * @note For linking variables, the number of nonzeros in the subproblems corresponds to the number on nonzeros
 * in the border
 *
 * @note The arrays have to be allocated by the caller
 *
 * @pre This function assumes that constraints are partitioned in the decomp structure, no constraint is present in more than one block
 *
 */
SCIP_RETCODE DECgetDensityData(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_VAR**            vars,               /**< pointer to array store variables belonging to density */
   int                   nvars,              /**< number of variables */
   SCIP_CONS**           conss,              /**< pointer to array to store constraints belonging to the density */
   int                   nconss,             /**< number of constraints */
   int*                  varsubproblemdensity, /**< pointer to array to store the nonzeros for the subproblems */
   int*                  varmasterdensity,   /**< pointer to array to store the nonzeros for the master */
   int*                  conssubproblemdensity, /**< pointer to array to store the nonzeros for the subproblems */
   int*                  consmasterdensity   /**< pointer to array to store the nonzeros for the master */
)
{
   int nlinkingconss;
   SCIP_HASHMAP* vartoblock;
   SCIP_CONS** curconss;

   int ncurvars;
   SCIP_VAR** curvars;
   SCIP_Bool success;

   int i;
   int j;
   int v;
   int c;

   assert(scip != NULL);
   assert(decomp != NULL);
   assert(vars != NULL);
   assert(nvars > 0);
   assert(conss != NULL);
   assert(varsubproblemdensity != NULL);
   assert(varmasterdensity != NULL);
   assert(conssubproblemdensity != NULL);
   assert(consmasterdensity != NULL);

   /* make sure the passed data is initialised to 0 */
   BMSclearMemoryArray(vars, nvars);
   BMSclearMemoryArray(conss, nconss);
   BMSclearMemoryArray(varsubproblemdensity, nvars);
   BMSclearMemoryArray(varmasterdensity, nvars);
   BMSclearMemoryArray(conssubproblemdensity, nconss);
   BMSclearMemoryArray(consmasterdensity, nconss);

   BMScopyMemoryArray(vars, SCIPgetVars(scip), nvars);

   vartoblock = DECdecompGetVartoblock(decomp);
   c = 0;
   for( i = 0; i < DECdecompGetNBlocks(decomp); ++i )
   {
      curconss = DECdecompGetSubscipconss(decomp)[i];
      assert(curconss != NULL);

      for( j = 0; j < DECdecompGetNSubscipconss(decomp)[i]; ++j )
      {
         assert(c < nconss); /* This assertion and the logic forbids constraints in more than one block */
         conss[c] = curconss[j];

         SCIP_CALL( SCIPgetConsNVars(scip, curconss[j], &ncurvars, &success) );
         assert(success);
         SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );
         SCIP_CALL( SCIPgetConsVars(scip, curconss[j], curvars, ncurvars, &success) );
         assert(success);

         for( v = 0; v < ncurvars; ++v )
         {
            SCIP_VAR* var;
            int block;
            int probindex;

            var = curvars[v];
            var = SCIPvarGetProbvar(var);
            probindex = SCIPvarGetProbindex(var);

            if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_FIXED )
               continue;

            assert(probindex >= 0);
            assert(probindex < nvars);
            varsubproblemdensity[probindex] += 1;
            assert(varsubproblemdensity[probindex] > 0);
            block = (int) (size_t) SCIPhashmapGetImage(vartoblock, var); /*lint !e507*/
            assert(block > 0);

            if( block <= DECdecompGetNBlocks(decomp) )
            {
               conssubproblemdensity[c] +=1;
            }
            else
            {
               consmasterdensity[c] += 1;
            }
         }

         SCIPfreeBufferArray(scip, &curvars);
         c++;
      }
   }

   nlinkingconss = DECdecompGetNLinkingconss(decomp);
   curconss = DECdecompGetLinkingconss(decomp);

   for( j = 0; j < nlinkingconss; ++j )
   {
      assert(c < nconss); /* This assertion and the logic forbids constraints in more than one block */
      SCIP_CALL( SCIPgetConsNVars(scip, curconss[j], &ncurvars, &success) );
      assert(success);
      SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );
      SCIP_CALL( SCIPgetConsVars(scip, curconss[j], curvars, ncurvars, &success) );
      assert(success);

      conss[c] = curconss[j];

      for( v = 0; v < ncurvars; ++v )
      {
         SCIP_VAR* var;
         int probindex;

         var = curvars[v];
         var = SCIPvarGetProbvar(var);
         probindex = SCIPvarGetProbindex(var);

         if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_FIXED )
            continue;

         assert(probindex >= 0);
         assert(probindex < nvars);
         varmasterdensity[probindex] += 1;
         assert(varmasterdensity[probindex] > 0);
         SCIPdebugMessage("Var <%s> appears in cons <%s>, total count: %d\n", SCIPvarGetName(var), SCIPconsGetName(curconss[j]), varmasterdensity[probindex]);
      }

      consmasterdensity[c] = ncurvars;
      c++;

      SCIPfreeBufferArray(scip, &curvars);
   }

   return SCIP_OKAY;
}

/** helper function to increase correct lock */
static
void increaseLock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             lhs,                /**< left side of constraint */
   SCIP_Real             coef,               /**< coefficient of variable in constraint */
   SCIP_Real             rhs,                /**< right side of constraint */
   int*                  downlock,           /**< pointer to store downlock */
   int*                  uplock              /**< pointer to store uplock */
   )
{
   assert(scip != NULL);
   assert(downlock != NULL);
   assert(uplock != NULL);

   if( !SCIPisInfinity(scip, -lhs) )
   {
      if( SCIPisPositive(scip, coef) )
         ++(*downlock);
      if( SCIPisNegative(scip, coef) )
         ++(*uplock);

   }
   if( !SCIPisInfinity(scip, rhs) )
   {
      if( SCIPisPositive(scip, coef) )
         ++(*uplock);
      if( SCIPisNegative(scip, coef) )
         ++(*downlock);
   }

}

/**
 *  calculates the number of up and down locks of variables for a given decomposition in both the original problem and the pricingproblems
 *
 *  @note All arrays need to be allocated by the caller
 *
 *  @warning This function needs a lot of memory (nvars*nblocks+1) array entries
 */
SCIP_RETCODE DECgetVarLockData(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_VAR**            vars,               /**< pointer to array store variables belonging to density */
   int                   nvars,              /**< number of variables */
   int                   nsubproblems,       /**< number of sub problems */
   int**                 subsciplocksdown,   /**< pointer to two dimensional array to store the down locks for the subproblems */
   int**                 subsciplocksup,     /**< pointer to two dimensional array to store the down locks for the subproblems */
   int*                  masterlocksdown,    /**< pointer to array to store the down locks for the master */
   int*                  masterlocksup       /**< pointer to array to store the down locks for the master */
   )
{
   int nlinkingconss;
   SCIP_CONS** curconss;
   SCIP_VAR** curvars;
   SCIP_Real* curvals;
   int ncurvars;
   SCIP_Real lhs;
   SCIP_Real rhs;

   SCIP_Bool success;

   int i;
   int j;
   int v;

   assert(scip != NULL);
   assert(decomp != NULL);
   assert(vars != NULL);
   assert(nvars > 0);
   assert(nvars == SCIPgetNVars(scip));
   assert(subsciplocksdown != NULL);
   assert(subsciplocksup != NULL);
   assert(masterlocksdown != NULL);
   assert(masterlocksup != NULL);

   /* make sure the passed data is initialised to 0 */
   BMSclearMemoryArray(vars, nvars);
   BMScopyMemoryArray(vars, SCIPgetVars(scip), nvars);
   BMSclearMemoryArray(masterlocksdown, nvars);
   BMSclearMemoryArray(masterlocksup, nvars);
   for( i = 0; i < nsubproblems; ++i )
   {
      BMSclearMemoryArray(subsciplocksdown[i], nvars); /*lint !e866*/
      BMSclearMemoryArray(subsciplocksup[i], nvars); /*lint !e866*/
   }

   for( i = 0; i < DECdecompGetNBlocks(decomp); ++i )
   {
      curconss = DECdecompGetSubscipconss(decomp)[i];
      assert(curconss != NULL);

      for( j = 0; j < DECdecompGetNSubscipconss(decomp)[i]; ++j )
      {

         SCIP_CALL( SCIPgetConsNVars(scip, curconss[j], &ncurvars, &success) );
         assert(success);
         SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );
         SCIP_CALL( SCIPallocBufferArray(scip, &curvals, ncurvars) );

         SCIP_CALL( GCGconsGetVals(scip, curconss[j], curvals, ncurvars) );
         SCIP_CALL( SCIPgetConsVars(scip, curconss[j], curvars, ncurvars, &success) );
         assert(success);

         rhs = GCGconsGetRhs(scip, curconss[j]);
         lhs = GCGconsGetLhs(scip, curconss[j]);

         for( v = 0; v < ncurvars; ++v )
         {
            SCIP_VAR* var;
            int probindex;

            var = curvars[v];
            var = SCIPvarGetProbvar(var);
            probindex = SCIPvarGetProbindex(var);

            if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_FIXED )
               continue;

            assert(probindex >= 0);
            assert(probindex < nvars);
            assert(SCIPhashmapExists(DECdecompGetVartoblock(decomp), var));

            increaseLock(scip, lhs, curvals[v], rhs, &(subsciplocksdown[i][probindex]), &(subsciplocksup[i][probindex]));
         }

         SCIPfreeBufferArray(scip, &curvals);
         SCIPfreeBufferArray(scip, &curvars);
      }
   }

   nlinkingconss = DECdecompGetNLinkingvars(decomp);
   curconss = DECdecompGetLinkingconss(decomp);
   for( j = 0; j < nlinkingconss; ++j )
   {
      SCIP_CALL( SCIPgetConsNVars(scip, curconss[j], &ncurvars, &success) );
      assert(success);
      SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &curvals, ncurvars) );

      SCIP_CALL( GCGconsGetVals(scip, curconss[j], curvals, ncurvars) );
      SCIP_CALL( SCIPgetConsVars(scip, curconss[j], curvars, ncurvars, &success) );
      assert(success);

      rhs = GCGconsGetRhs(scip, curconss[j]);
      lhs = GCGconsGetLhs(scip, curconss[j]);

      for( v = 0; v < ncurvars; ++v )
      {
         SCIP_VAR* var;
         int probindex;

         var = curvars[v];
         var = SCIPvarGetProbvar(var);
         probindex = SCIPvarGetProbindex(var);

         if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_FIXED )
            continue;

         assert(probindex >= 0);
         assert(probindex < nvars);

         increaseLock(scip, lhs, curvals[v], rhs, &(masterlocksdown[probindex]), &(masterlocksup[probindex]));
      }

      SCIPfreeBufferArray(scip, &curvals);
      SCIPfreeBufferArray(scip, &curvars);
   }

   return SCIP_OKAY;
}


/** sets the score of the given decomposition based on the border, the average density score and the ratio of
 * linking variables
 */
void DECsetMaxWhiteScore(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decdecomp,          /**< decomposition data structure */
   SCIP_Real             maxwhitescore       /**< score related to max white measure (i.e. fraction of white (nonblock and nonborder) matrix area ) */
   )
{
   assert(maxwhitescore >= 0);

   decdecomp->maxwhitescore = maxwhitescore;

   return;
}


/** computes the score of the given decomposition based on the border, the average density score and the ratio of
 * linking variables
 */
SCIP_Real DECgetMaxWhiteScore(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decdecomp           /**< decomposition data structure */
   )
{
   DEC_SCORES score;

   if( decdecomp->maxwhitescore == -1.)
      DECevaluateDecomposition(scip, decdecomp, &score);

   assert(decdecomp->maxwhitescore >= 0);

   return decdecomp->maxwhitescore;
}

/** computes the score of the given decomposition based on the border, the average density score and the ratio of
 * linking variables
 */
SCIP_RETCODE DECevaluateDecomposition(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decdecomp,          /**< decomposition data structure */
   DEC_SCORES*           score               /**< returns the score of the decomposition */
   )
{
   SCIP_Longint matrixarea;
   SCIP_Longint borderarea;
   int nvars;
   int nconss;
   int i;
   int j;
   int k;
   /*   int blockarea; */
   SCIP_Real varratio;
   int* nzblocks;
   int nblocks;
   int* nlinkvarsblocks;
   int* nvarsblocks;
   SCIP_Real* blockdensities;
   int* blocksizes;
   SCIP_Real density;
   SCIP_Real blackarea;

   SCIP_Real alphaborderarea;
   SCIP_Real alphalinking;
   SCIP_Real alphadensity;

   alphaborderarea = 0.6;
   alphalinking = 0.2 ;
   alphadensity  = 0.2;
   blackarea = 0.;


   assert(scip != NULL);
   assert(score != NULL);

   nvars = SCIPgetNVars(scip);
   nconss = SCIPgetNConss(scip);

   nblocks = DECdecompGetNBlocks(decdecomp);

   SCIP_CALL( SCIPallocBufferArray(scip, &nzblocks, nblocks) );
   SCIP_CALL( SCIPallocBufferArray(scip, &nlinkvarsblocks, nblocks) );
   SCIP_CALL( SCIPallocBufferArray(scip, &blockdensities, nblocks) );
   SCIP_CALL( SCIPallocBufferArray(scip, &blocksizes, nblocks) );
   SCIP_CALL( SCIPallocBufferArray(scip, &nvarsblocks, nblocks) );
   /*
    * 3 Scores
    *
    * - Area percentage (min)
    * - block density (max)
    * - \pi_b {v_b|v_b is linking}/#vb (min)
    */

   /* calculate matrix area */
   matrixarea = (SCIP_Longint) nvars*nconss;

   blackarea += ( DECdecompGetNLinkingvars(decdecomp) - DECdecompGetNMastervars(decdecomp) ) * nconss;
   blackarea += DECdecompGetNLinkingconss(decdecomp) * nvars;

   blackarea -= (DECdecompGetNLinkingvars(decdecomp) - DECdecompGetNMastervars(decdecomp) ) * DECdecompGetNLinkingconss(decdecomp);


   /* calculate slave sizes, nonzeros and linkingvars */
   for( i = 0; i < nblocks; ++i )
   {
      SCIP_CONS** curconss;
      int ncurconss;
      int nvarsblock;
      SCIP_Bool *ishandled;

      SCIP_CALL( SCIPallocBufferArray(scip, &ishandled, nvars) );
      nvarsblock = 0;
      nzblocks[i] = 0;
      nlinkvarsblocks[i] = 0;
      blackarea +=  DECdecompGetNSubscipconss(decdecomp)[i] * ( DECdecompGetNSubscipvars(decdecomp)[i] );

      for( j = 0; j < nvars; ++j )
      {
         ishandled[j] = FALSE;
      }
      curconss = DECdecompGetSubscipconss(decdecomp)[i];
      ncurconss = DECdecompGetNSubscipconss(decdecomp)[i];

      for( j = 0; j < ncurconss; ++j )
      {
         SCIP_VAR** curvars;
         SCIP_VAR* var;
         int ncurvars;
         ncurvars = GCGconsGetNVars(scip, curconss[j]);
         if ( ncurvars == 0 )
            continue;
         SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );
         SCIP_CALL( GCGconsGetVars(scip, curconss[j], curvars, ncurvars) );

         for( k = 0; k < ncurvars; ++k )
         {
            int block;
            if( !GCGisVarRelevant(curvars[k]) )
               continue;

            var = SCIPvarGetProbvar(curvars[k]);
            assert(var != NULL);
            if( !GCGisVarRelevant(var) )
               continue;

            assert(SCIPvarIsActive(var));
            assert(!SCIPvarIsDeleted(var));
            ++(nzblocks[i]);
            if( !SCIPhashmapExists(DECdecompGetVartoblock(decdecomp), var) )
            {
               block = (int)(size_t) SCIPhashmapGetImage(DECdecompGetVartoblock(decdecomp), curvars[k]); /*lint !e507*/
            }
            else
            {
               assert(SCIPhashmapExists(DECdecompGetVartoblock(decdecomp), var));
               block = (int)(size_t) SCIPhashmapGetImage(DECdecompGetVartoblock(decdecomp), var); /*lint !e507*/
            }

            if( block == nblocks+1 && ishandled[SCIPvarGetProbindex(var)] == FALSE )
            {
               ++(nlinkvarsblocks[i]);
            }
            ishandled[SCIPvarGetProbindex(var)] = TRUE;
         }

         SCIPfreeBufferArray(scip, &curvars);
      }

      for( j = 0; j < nvars; ++j )
      {
         if( ishandled[j] )
         {
            ++nvarsblock;
         }
      }

      blocksizes[i] = nvarsblock*ncurconss;
      nvarsblocks[i] = nvarsblock;
      if( blocksizes[i] > 0 )
      {
         blockdensities[i] = 1.0*nzblocks[i]/blocksizes[i];
      }
      else
      {
         blockdensities[i] = 0.0;
      }

      assert(blockdensities[i] >= 0 && blockdensities[i] <= 1.0);
      SCIPfreeBufferArray(scip, &ishandled);
   }

   borderarea = (SCIP_Longint) DECdecompGetNLinkingconss(decdecomp)*nvars + (SCIP_Longint) DECdecompGetNLinkingvars(decdecomp)*(nconss-DECdecompGetNLinkingconss(decdecomp));

   density = 1E20;
   varratio = 1.0;
   for( i = 0; i < nblocks; ++i )
   {
      density = MIN(density, blockdensities[i]);

      if( DECdecompGetNLinkingvars(decdecomp) > 0 )
      {
         varratio *= 1.0*nlinkvarsblocks[i]/DECdecompGetNLinkingvars(decdecomp);
      }
      else
      {
         varratio = 0;
      }
   }

   score->linkingscore = (0.5+0.5*varratio);
   score->borderscore = (1.0*(borderarea)/matrixarea);
   score->densityscore = (1-density);
   //score->maxwhitescore = blackarea/( nconss * nvars );

   //decdecomp->maxwhitescore = score->maxwhitescore;


   switch( DECdecompGetType(decdecomp) )
   {
   case DEC_DECTYPE_ARROWHEAD:
      score->totalscore = alphaborderarea*(score->borderscore) + alphalinking*(score->linkingscore) + alphadensity*(score->densityscore);
/*      score->totalscore = score->borderscore*score->linkingscore*score->densityscore; */
      break;
   case DEC_DECTYPE_BORDERED:
      score->totalscore = alphaborderarea*(score->borderscore) + alphalinking*(score->linkingscore) + alphadensity*(score->densityscore);
      /*       score->totalscore = score->borderscore*score->linkingscore*score->densityscore; */
      break;
   case DEC_DECTYPE_DIAGONAL:
      if(nblocks == 1 || nblocks == 0)
         score->totalscore = 1.0;
      else
         score->totalscore = 0.0;
      break;
   case DEC_DECTYPE_STAIRCASE:
      score->totalscore = alphaborderarea*(score->borderscore) + alphalinking*(score->linkingscore) + 0.2*(score->densityscore);
/*       score->totalscore = score->borderscore*score->linkingscore*score->densityscore; */
      break;
   case DEC_DECTYPE_UNKNOWN:
      SCIPerrorMessage("Decomposition type is %s, cannot compute score\n", DECgetStrType(DECdecompGetType(decdecomp)));
      assert(FALSE);
      break;
   default:
      SCIPerrorMessage("No rule for this decomposition type, cannot compute score\n");
      assert(FALSE);
      break;
   }
   if(nblocks == 1 || nblocks == 0)
      score->totalscore = 1.0;

   if( nblocks == 0 || nblocks == 1)
	   score->totalscore = 1;

   SCIPfreeBufferArray(scip, &nvarsblocks);
   SCIPfreeBufferArray(scip, &blocksizes);
   SCIPfreeBufferArray(scip, &blockdensities);
   SCIPfreeBufferArray(scip, &nlinkvarsblocks);
   SCIPfreeBufferArray(scip, &nzblocks);
   return SCIP_OKAY;
}

/** compute the density of variables in blocks and master */
static
SCIP_RETCODE computeVarDensities(
      SCIP*              scip,               /**< SCIP data structure */
      DEC_DECOMP*        decomp,             /**< decomposition data structure */
      int*               varprobdensity,     /**< density information */
      int*               varmasterdensity,   /**< density information */
      SCIP_VAR**         vars,               /**< array of variables */
      int                nvars,              /**< number of variables */
      DEC_STATISTIC*     blockvardensities,  /**< array of statistic structs to store density information of each block */
      DEC_STATISTIC*     mastervardensity,   /**< pointer to store density information of master variables*/
      int                nblocks             /**< number of blocks */
   )
{
   int v;
   int b;
   SCIP_Real** vardistribution;
   int* nvardistribution;
   SCIP_Real* mastervardistribution;

   SCIP_Real max = 0;
   SCIP_Real min = 1.0;
   SCIP_Real median = 0;
   SCIP_Real mean = 0;

   assert(scip != NULL);
   assert(decomp != NULL);

   assert(vars != NULL);
   assert(nvars > 0);
   assert(blockvardensities != NULL);
   assert(mastervardensity != NULL);
   assert(nblocks >= 0);

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &vardistribution, nblocks) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &nvardistribution, nblocks) );

   BMSclearMemoryArray(vardistribution, nblocks);
   BMSclearMemoryArray(nvardistribution, nblocks);

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &mastervardistribution, nvars) );
   BMSclearMemoryArray(mastervardistribution, nvars);

   for( b = 0; b < nblocks; ++b )
   {
      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &vardistribution[b], DECdecompGetNSubscipvars(decomp)[b]) ); /*lint !e866 !e666*/
      BMSclearMemoryArray(vardistribution[b], DECdecompGetNSubscipvars(decomp)[b]); /*lint !e866 */
   }

   for( v = 0; v < nvars; ++v )
   {
      int block = ((int) (size_t) SCIPhashmapGetImage(DECdecompGetVartoblock(decomp), (vars[v]))) - 1; /*lint !e507 */
      assert(block >= 0);
      SCIPdebugMessage("Var <%s>:", SCIPvarGetName(vars[v]));


      mastervardistribution[v] = 1.0*varmasterdensity[v]/DECdecompGetNLinkingconss(decomp);
      SCIPdebugPrintf("master %d ", varmasterdensity[v]);

      if( block < nblocks )
      {
         vardistribution[block][nvardistribution[block]] = 1.0*varprobdensity[v]/DECdecompGetNSubscipconss(decomp)[block];
         SCIPdebugPrintf("block %d %.3f\n", block, vardistribution[block][nvardistribution[block]]);
         ++(nvardistribution[block]);
      }
      else
      {
         SCIPdebugPrintf("\n");
      }
   }

   for( b = 0; b < nblocks; ++b )
   {
      int ncurvars = DECdecompGetNSubscipvars(decomp)[b];

      max = 0;
      min = 1.0;
      median = 0;
      mean = 0;



      SCIPdebugMessage("block %d:", b);
      for( v = 0; v < ncurvars; ++v )
      {

         SCIPdebugPrintf(" <%s> %.3f", SCIPvarGetName(DECdecompGetSubscipvars(decomp)[b][v]), vardistribution[b][v]);
         max = MAX(max, vardistribution[b][v]);
         min = MIN(min, vardistribution[b][v]);
         mean += 1.0*vardistribution[b][v]/ncurvars;

      }
      if( ncurvars > 0 )
         median = quick_select_median(vardistribution[b], ncurvars);

      SCIPdebugPrintf("\nmin: %.3f, max: %.3f, median: %.3f, mean: %.3f\n", min, max, median, mean);

      blockvardensities[b].max = max;
      blockvardensities[b].min = min;
      blockvardensities[b].median = median;
      blockvardensities[b].mean = mean;
   }
   max = 0;
   min = 1.0;
   mean = 0;

   SCIPdebugMessage("master:");

   for( v = 0; v < nvars; ++v )
   {

      SCIPdebugPrintf(" <%s> %.3f", SCIPvarGetName(vars[v]), mastervardistribution[v]);
      max = MAX(max, mastervardistribution[v]);
      min = MIN(min, mastervardistribution[v]);
      mean += 1.0*mastervardistribution[v]/nvars;

   }
   median = quick_select_median(mastervardistribution, nvars);
   SCIPdebugPrintf("\nmin: %.3f, max: %.3f, median: %.3f, mean: %.3f\n", min, max, median, mean);


   mastervardensity->max = max;
   mastervardensity->min = min;
   mastervardensity->median = median;
   mastervardensity->mean = mean;

   for( b = 0; b < nblocks; ++b )
   {
      SCIPfreeBlockMemoryArray(scip, &vardistribution[b], DECdecompGetNSubscipvars(decomp)[b]); /*lint !e866 */
   }

   SCIPfreeBlockMemoryArray(scip, &mastervardistribution, nvars);

   SCIPfreeBlockMemoryArray(scip, &nvardistribution, nblocks);
   SCIPfreeBlockMemoryArray(scip, &vardistribution, nblocks);

   return SCIP_OKAY;
}

/** returns the number of constraints saved in the decomposition */
int DECdecompGetNConss(
   DEC_DECOMP*           decomp              /**< decomposition data structure */
   )
{
   int b;
   int nconss = 0;
   assert(decomp != NULL);

   for( b = 0; b < DECdecompGetNBlocks(decomp); ++b )
      nconss += DECdecompGetNSubscipconss(decomp)[b];

   nconss += DECdecompGetNLinkingconss(decomp);
   return nconss;
}

/** computes nonzero elements of a given constraint, separated into linking variables and normal vars */
static
SCIP_RETCODE computeConssNzeros(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_CONS*            cons,               /**< SCIP data structure */
   int*                  nzeros,             /**< pointer to store nonzero elements */
   int*                  nintzeros,          /**< pointer to store integer nonzeros */
   int*                  nbzeros,            /**< pointer to store border nonzero elements */
   int*                  nintbzeros          /**< pointer to store border integer nonzeros */
)
{
   int v;
   int ncurvars;
   SCIP_VAR** curvars = NULL;
   SCIP_Real* curvals = NULL;

   assert(scip != NULL);
   assert(decomp != NULL);
   assert(cons != NULL);
   assert(nzeros != NULL);
   assert(nintzeros != NULL);
   assert(nbzeros != NULL);
   assert(nintbzeros != NULL);

   ncurvars = GCGconsGetNVars(scip, cons);
   SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &curvals, ncurvars) );

   SCIP_CALL( GCGconsGetVars(scip, cons, curvars, ncurvars) );
   SCIP_CALL( GCGconsGetVals(scip, cons, curvals, ncurvars) );

   for( v = 0; v < ncurvars; ++v )
   {
      int block;
      SCIP_VAR* curvar;
      if( SCIPisZero(scip, curvals[v]) )
         continue;

      curvar = SCIPvarGetProbvar(curvars[v]);

      if( SCIPvarGetStatus(curvar) == SCIP_VARSTATUS_FIXED )
         continue;

      block = ((int) (size_t) SCIPhashmapGetImage(DECdecompGetVartoblock(decomp), (curvar))) - 1; /*lint !e507 */
      assert(block >= 0);

      if( block > DECdecompGetNBlocks(decomp) )
      {
         if( SCIPvarGetType(curvar) == SCIP_VARTYPE_BINARY || SCIPvarGetType(curvar) == SCIP_VARTYPE_INTEGER )
            *nintbzeros += 1;

         *nbzeros += 1;
      }
      else
      {
         if( SCIPvarGetType(curvar) == SCIP_VARTYPE_BINARY || SCIPvarGetType(curvar) == SCIP_VARTYPE_INTEGER )
            *nintzeros += 1;

         *nzeros += 1;
      }
   }

   SCIPfreeBufferArrayNull(scip, &curvals);
   SCIPfreeBufferArrayNull(scip, &curvars);

   return SCIP_OKAY;
}

/** computes nonzero elements of the pricing problems and the master */
static
SCIP_RETCODE computeNonzeros(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   int*                  mnzeros,            /**< number of nonzero elements in row border */
   int*                  mintnzeros,         /**< number of integral nonzero elements in row border */
   int*                  lnzeros,            /**< number of nonzero elements in column border */
   int*                  lintnzeros,         /**< number of integral nonzero elements in column border */
   int*                  nonzeros,           /**< number of nonzero elements per pricing problem */
   int*                  intnzeros           /**< number of integral nonzero elements per pricing problem */
   )
{
   int c;
   int b;

   assert(scip != NULL);
   assert(decomp != NULL);
   assert(mnzeros != NULL);
   assert(mintnzeros != NULL);
   assert(lnzeros != NULL);
   assert(lintnzeros != NULL);
   assert(nonzeros != NULL);
   assert(intnzeros != NULL);

   for( b = 0; b < DECdecompGetNBlocks(decomp); ++b )
   {
      SCIP_CONS** subscipconss = DECdecompGetSubscipconss(decomp)[b];
      int nsubscipconss = DECdecompGetNSubscipconss(decomp)[b];
      for( c = 0; c < nsubscipconss; ++c )
      {
         SCIP_CALL( computeConssNzeros(scip, decomp, subscipconss[c], &(nonzeros[b]), &(intnzeros[b]), lnzeros, lintnzeros ) );
      }
   }

   for( c = 0; c < DECdecompGetNLinkingconss(decomp); ++c )
   {
      SCIP_CALL( computeConssNzeros(scip, decomp, DECdecompGetLinkingconss(decomp)[c], mnzeros, mintnzeros, lnzeros, lintnzeros ) );
   }

   return SCIP_OKAY;
}

/** display statistics about the decomposition */
SCIP_RETCODE GCGprintDecompStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file or NULL for standard output */
   DEC_DECOMP*           decomp              /**< decomp that should be evaluated */
   )
{
   DEC_SCORES scores;
   SCIP_VAR** vars;
   SCIP_CONS** conss;

   int nvars;
   int nconss;

   int* nallvars;
   int* nbinvars;
   int* nintvars;
   int* nimplvars;
   int* ncontvars;

   int nblocks;
   int nblocksrelevant;
   int nlinkvars;
   int nstaticvars;
   int nlinkbinvar;
   int nlinkintvars;
   int nlinkimplvars;
   int nlinkcontvars;
   int b;

   int* varprobdensity;
   int* varmasterdensity;
   int* consprobsensity;
   int* consmasterdensity;

   DEC_STATISTIC* blockvardensities;
   DEC_STATISTIC* blockconsdensities;
   DEC_STATISTIC mastervardensity;

   int mnzeros;
   int mintnzeros;
   int lnzeros;
   int lintnzeros;

   int* nonzeros;
   int* intnzeros;

   assert(scip != NULL);
   assert(decomp != NULL);

   nblocks = DECdecompGetNBlocks(decomp);
   nvars = SCIPgetNVars(scip);
   nconss = DECdecompGetNConss(decomp);

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &nallvars, nblocks) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &nbinvars, nblocks) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &nintvars, nblocks) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &nimplvars, nblocks) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &ncontvars, nblocks) );

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &blockvardensities, nblocks) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &blockconsdensities, nblocks) );

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &varprobdensity, nvars) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &varmasterdensity, nvars) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &vars, nvars) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &conss, nconss) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &consprobsensity, nconss) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &consmasterdensity, nconss) );

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &nonzeros, nblocks) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &intnzeros, nblocks) );

   BMSclearMemoryArray(nonzeros, nblocks);
   BMSclearMemoryArray(intnzeros, nblocks);

   mnzeros = 0;
   mintnzeros = 0;
   lnzeros = 0;
   lintnzeros = 0;

   SCIP_CALL( DECevaluateDecomposition(scip, decomp, &scores) );

   DECgetSubproblemVarsData(scip, decomp, nallvars, nbinvars, nintvars, nimplvars, ncontvars, nblocks);
   DECgetLinkingVarsData(scip, decomp, &nlinkvars, &nlinkbinvar, &nlinkintvars, &nlinkimplvars, &nlinkcontvars);
   nlinkvars = nlinkvars - DECdecompGetNMastervars(decomp);
   nstaticvars = DECdecompGetNMastervars(decomp);

   SCIP_CALL( DECgetDensityData(scip, decomp, vars, nvars, conss, nconss, varprobdensity, varmasterdensity, consprobsensity, consmasterdensity) );

   SCIP_CALL( computeVarDensities(scip, decomp, varprobdensity, varmasterdensity, vars, nvars, blockvardensities, &mastervardensity, nblocks) );
   SCIP_CALL( computeNonzeros(scip, decomp, &mnzeros, &mintnzeros, &lnzeros, &lintnzeros, nonzeros, intnzeros) );

   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Decomp statistics  :\n");
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "  type             : %10s\n", DECgetStrType(DECdecompGetType(decomp)));
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "  detector         : %10s\n", decomp->detectorchainstring == NULL? "provided": decomp->detectorchainstring);
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "  blocks           : %10d\n", DECdecompGetNBlocks(decomp));

   nblocksrelevant = nblocks;
   if( SCIPgetStage(GCGgetMasterprob(scip)) >= SCIP_STAGE_PRESOLVED )
   {
      for( b = 0; b < nblocks; ++b )
      {
         if( GCGgetNIdenticalBlocks(scip, b) == 0 )
            nblocksrelevant -= 1;
      }
   }
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "  aggr. blocks     : %10d\n", nblocksrelevant);

   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Master statistics  :  nlinkvars  nstatvars  nbinvars  nintvars nimplvars  ncontvars   nconss  nonzeros intnzeros   bnzeros bintnzeros min(dens) max(dens) medi(dens) mean(dens)\n");
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "  master           : %10d %10d %9d %9d %9d %10d %8d %9d %9d %9d %10d %9.3f %9.3f %10.3f %9.3f\n", nlinkvars, nstaticvars,
         nlinkbinvar, nlinkintvars, nlinkimplvars, nlinkcontvars, DECdecompGetNLinkingconss(decomp),
         mnzeros, mintnzeros, lnzeros, lintnzeros, mastervardensity.min, mastervardensity.max, mastervardensity.median, mastervardensity.mean);

   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Pricing statistics :      nvars   nbinvars   nintvars  nimplvars  ncontvars     nconss   nonzeros  intnzeros  min(dens)  max(dens) medi(dens) mean(dens)  identical\n");
   for( b = 0; b < nblocks; ++b )
   {
      int identical = 0;
      SCIP_Bool relevant = TRUE;

      if( SCIPgetStage(GCGgetMasterprob(scip)) >= SCIP_STAGE_PRESOLVED )
      {
         relevant =  GCGisPricingprobRelevant(scip, b);
         identical = GCGgetNIdenticalBlocks(scip, b);
      }
      if( relevant )
      {
         SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, " %10d        : %10d %10d %10d %10d %10d %10d %10d %10d %10.3f %10.3f %10.3f %10.3f %10d\n", b+1, nallvars[b], nbinvars[b], nintvars[b], nimplvars[b], ncontvars[b],
               DECdecompGetNSubscipconss(decomp)[b], nonzeros[b], intnzeros[b], blockvardensities[b].min, blockvardensities[b].max, blockvardensities[b].median, blockvardensities[b].mean, identical);
      }
   }

   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Decomp Scores      :\n");
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "  border area      : %10.3f\n", scores.borderscore);
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "  avg. density     : %10.3f\n", scores.densityscore);
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "  linking score    : %10.3f\n", scores.linkingscore);

   SCIPfreeBlockMemoryArray(scip, &vars, nvars);
   SCIPfreeBlockMemoryArray(scip, &conss, nconss);


   SCIPfreeBlockMemoryArray(scip, &intnzeros, nblocks);
   SCIPfreeBlockMemoryArray(scip, &nonzeros, nblocks);

   SCIPfreeBlockMemoryArray(scip, &varprobdensity, nvars);
   SCIPfreeBlockMemoryArray(scip, &varmasterdensity, nvars);
   SCIPfreeBlockMemoryArray(scip, &consprobsensity, nconss);
   SCIPfreeBlockMemoryArray(scip, &consmasterdensity, nconss);

   SCIPfreeBlockMemoryArray(scip, &blockvardensities, nblocks);
   SCIPfreeBlockMemoryArray(scip, &blockconsdensities, nblocks);

   SCIPfreeBlockMemoryArray(scip, &nallvars, nblocks);
   SCIPfreeBlockMemoryArray(scip, &nbinvars, nblocks);
   SCIPfreeBlockMemoryArray(scip, &nintvars, nblocks);
   SCIPfreeBlockMemoryArray(scip, &nimplvars, nblocks);
   SCIPfreeBlockMemoryArray(scip, &ncontvars, nblocks);

   return SCIP_OKAY;
}

/** returns whether both structures lead to the same decomposition */
SCIP_Bool DECdecompositionsAreEqual(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp1,            /**< first decomp data structure */
   DEC_DECOMP*           decomp2             /**< second decomp data structure */
)
{
   SCIP_HASHMAP* constoblock1;
   SCIP_HASHMAP* constoblock2;

   SCIP_HASHMAP* vartoblock1;
   SCIP_HASHMAP* vartoblock2;

   SCIP_CONS** conss;
   int nconss;

   SCIP_VAR** vars;
   int nvars;
   int i;

   assert(scip != NULL);
   assert(decomp1 != NULL);
   assert(decomp2 != NULL);

   if( DECdecompGetNBlocks(decomp1) != DECdecompGetNBlocks(decomp2) )
   {
      return FALSE;
   }

   conss = SCIPgetConss(scip);
   nconss = SCIPgetNConss(scip);

   vars = SCIPgetVars(scip);
   nvars = SCIPgetNVars(scip);

   constoblock1 = DECdecompGetConstoblock(decomp1);
   constoblock2 = DECdecompGetConstoblock(decomp2);
   assert(constoblock1 != NULL);
   assert(constoblock2 != NULL);

   vartoblock1 = DECdecompGetVartoblock(decomp1);
   vartoblock2 = DECdecompGetVartoblock(decomp2);
   assert(vartoblock1 != NULL);
   assert(vartoblock2 != NULL);

   vartoblock1 = DECdecompGetVartoblock(decomp1);
   vartoblock2 = DECdecompGetVartoblock(decomp2);

   for( i = 0; i < nconss; ++i )
   {
      if( SCIPhashmapGetImage(constoblock1, conss[i]) != SCIPhashmapGetImage(constoblock2, conss[i]) )
         return FALSE;
   }

   for( i = 0; i < nvars; ++i )
   {
      if( SCIPhashmapGetImage(vartoblock1, vars[i]) != SCIPhashmapGetImage(vartoblock2, vars[i]) )
         return FALSE;
   }

   return TRUE;
}

/** filters similar decompositions from a given list and moves them to the end
 * @return the number of unique decompositions
 */
int DECfilterSimilarDecompositions(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP**          decs,               /**< array of decompositions */
   int                   ndecs               /**< number of decompositions */
)
{
   int i;
   int j;
   int nunique;
   assert(scip != NULL);
   assert(decs != NULL);
   assert(ndecs > 0);

   nunique = ndecs;
   for( i = 0; i < nunique; ++i )
   {
      /*lint -e{850} j is modified in the body of the for loop */
      for( j = i+1; j < nunique; ++j )
      {
         DEC_DECOMP* tmp;
         if( DECdecompositionsAreEqual(scip, decs[i], decs[j]) )
         {
            tmp = decs[nunique-1];
            decs[nunique-1] = decs[j];
            decs[j] = tmp;
            --nunique;
            --j;
         }
      }
   }
   return nunique;
}

/** returns the number of the block that the constraint is with respect to the decomposition; set
 * *block = -2, if it has no variables
 * *block = -1, if it has only variables belonging only to the master (meaning that this constraint should build a new block)
 * *block in [0,...,nblocks-1] if it only contains variables of a particular block (plus linking variables)
 * *block = nblocks, if it contains
 *   - either variables from more than one block (plus linking variables or master only variables)
 *   - or linking variables only
 */
/** @todo: For linking variables, we should check which blocks they actually link */
/** @todo: maybe this is possible in such a way that a staircase structure is preserved */
SCIP_RETCODE DECdetermineConsBlock(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_CONS*            cons,               /**< constraint to check */
   int*                  block              /**< block of the constraint (or nblocks for master) */
)
{
   SCIP_VAR** curvars = NULL;
   int ncurvars = 0;
   SCIP_Bool success = FALSE;
   int i;
   int nblocks ;

   int nmastervars = 0;
   int npricingvars = 0;

   SCIP_HASHMAP* vartoblock;
   assert(scip != NULL);
   assert(decomp != NULL);
   assert(cons != NULL);
   assert(block != NULL);

   *block = -2;

   SCIP_CALL( SCIPgetConsNVars(scip, cons, &ncurvars, &success) );
   assert(success);

   if( ncurvars == 0 )
      return SCIP_OKAY;

   vartoblock= DECdecompGetVartoblock(decomp);
   assert(vartoblock != NULL);

   nblocks = DECdecompGetNBlocks(decomp);

   SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );
   SCIP_CALL( SCIPgetConsVars(scip, cons, curvars, ncurvars, &success) );
   assert(success);

   for( i = 0; i < ncurvars && *block != nblocks; ++i )
   {
      SCIP_VAR* var;
      int varblock;

      var = SCIPvarGetProbvar(curvars[i]);

      if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_FIXED )
         continue;

      assert(SCIPhashmapExists(vartoblock, var));
      varblock = ((int) (size_t) SCIPhashmapGetImage(vartoblock, var))-1; /*lint !e507 */

      /* if variable is linking skip*/
      if( varblock == nblocks+1 )
      {
         continue;
      }
      else if( varblock == nblocks )
      {
         ++nmastervars;
         continue;
      }
      else if( *block != varblock )
      {
         ++npricingvars;
         if( *block < 0 )
            *block = varblock;
         else
         {
            assert(*block != nblocks);
            *block = nblocks;
            break;
         }
      }
   }

   SCIPfreeBufferArrayNull(scip, &curvars);

   if( ncurvars > 0 && *block == -2 )
      *block = nblocks;

   if( npricingvars == 0 && nmastervars > 0 )
      *block = -1;



   return SCIP_OKAY;
}

/** move a master constraint to pricing problem */
SCIP_RETCODE DECdecompMoveLinkingConsToPricing(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   int                   consindex,          /**< index of constraint to move */
   int                   block               /**< block of the pricing problem where to move */
   )
{
   SCIP_CONS* linkcons;
   int oldsize;
   int newsize;

   assert(scip != NULL);
   assert(decomp != NULL);
   assert(consindex >= 0 && consindex < decomp->nlinkingconss);
   assert(block >= 0 && block < decomp->nblocks);

   linkcons = decomp->linkingconss[consindex];

   decomp->linkingconss[consindex] =  decomp->linkingconss[decomp->nlinkingconss-1];
   decomp->nlinkingconss -= 1;

   oldsize = SCIPcalcMemGrowSize(scip, decomp->nsubscipconss[block]);
   newsize = SCIPcalcMemGrowSize(scip, decomp->nsubscipconss[block]+1);
   SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &decomp->subscipconss[block], oldsize, newsize) ); /*lint !e866 */
   decomp->subscipconss[block][decomp->nsubscipconss[block]] = linkcons;
   decomp->nsubscipconss[block] += 1;

   SCIP_CALL( SCIPhashmapSetImage(decomp->constoblock, linkcons, (void*) (size_t)((size_t)block+1)) );
   SCIP_CALL( assignConsvarsToBlock(scip, decomp, linkcons, block) );

   return SCIP_OKAY;
}


/** tries to assign masterconss to pricing problem */
SCIP_RETCODE DECtryAssignMasterconssToExistingPricing(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   int*                  transferred         /**< number of master constraints reassigned */
   )
{
   int c;
   int linkingconssize;
   assert(scip != NULL);
   assert(decomp != NULL);
   assert(transferred != NULL);
   linkingconssize = decomp->nlinkingconss;
   *transferred = 0;

   /*lint -e{850} c is modified in the body of the for loop */
   for( c = 0; c < decomp->nlinkingconss; ++c )
   {
      int block;
      SCIP_CALL( DECdetermineConsBlock(scip, decomp, decomp->linkingconss[c], &block) );

      if( block == DECdecompGetNBlocks(decomp) || block < 0 )
      {
         continue;
      }

      SCIP_CALL( DECdecompMoveLinkingConsToPricing(scip, decomp, c, block) );
      --c;
      *transferred += 1;
   }

   if( *transferred > 0 )
   {
      if( decomp->nlinkingconss > 0 )
      {
         int oldsize = SCIPcalcMemGrowSize(scip, linkingconssize);
         int newsize = SCIPcalcMemGrowSize(scip, decomp->nlinkingconss);
         SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &decomp->linkingconss, oldsize, newsize) );
      }
      else
      {
         SCIPfreeBlockMemoryArrayNull(scip, &decomp->linkingconss, SCIPcalcMemGrowSize(scip, linkingconssize));
      }
   }

   return SCIP_OKAY;
}

/** tries to assign masterconss to new pricing problem */
SCIP_RETCODE DECtryAssignMasterconssToNewPricing(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   DEC_DECOMP**          newdecomp,          /**< new decomposition, if successful */
   int*                  transferred         /**< number of master constraints reassigned */
   )
{
   int c;

   assert(scip != NULL);
   assert(decomp != NULL);
   assert(newdecomp != NULL);
   assert(transferred != NULL);

   *newdecomp = NULL;
   *transferred = 0;

   for( c = 0; c < decomp->nlinkingconss; ++c )
   {
      int block;
      int i;
      int nconss;
      SCIP_HASHMAP* constoblock;
      SCIP_CALL( DECdetermineConsBlock(scip, decomp, decomp->linkingconss[c], &block) );

      if( block >= 0 )
      {
         continue;
      }
      SCIPdebugMessage("Cons <%s> in new pricing problem\n", SCIPconsGetName(decomp->linkingconss[c]));
      nconss = SCIPgetNConss(scip);
      SCIP_CALL( DECdecompCreate(scip, newdecomp) );
      SCIP_CALL( SCIPhashmapCreate(&constoblock, SCIPblkmem(scip), SCIPgetNConss(scip)) );

      for( i = 0; i < nconss; ++i )
      {
         int consblock;
         SCIP_CONS* cons = SCIPgetConss(scip)[i];
         assert(SCIPhashmapExists(decomp->constoblock, cons));
         consblock = (int) (size_t) SCIPhashmapGetImage(decomp->constoblock, cons); /*lint !e507 */
         SCIPdebugMessage("Cons <%s> %d -> %d\n", SCIPconsGetName(cons), consblock, consblock+1);

         SCIP_CALL( SCIPhashmapSetImage(constoblock, cons, (void*) (size_t) (consblock+1)) );
      }
      SCIP_CALL( SCIPhashmapSetImage(constoblock, decomp->linkingconss[c], (void*) (size_t) (1)) );
      SCIPdebugMessage("Cons <%s>    -> %d\n", SCIPconsGetName(decomp->linkingconss[c]), 1);

      SCIP_CALL( DECfilloutDecompFromConstoblock(scip, *newdecomp, constoblock, decomp->nblocks+1, FALSE) );
      *transferred += 1;
      break;
   }

   return SCIP_OKAY;
}

/** polish the decomposition and try to greedily assign master constraints to pricing problem where useful */
SCIP_RETCODE DECcreatePolishedDecomp(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   DEC_DECOMP**          newdecomp           /**< new decomposition, if successful */
   )
{
   int transferredexisting = 0;
   int transferrednew = 0;
   DEC_DECOMP* origdecomp = decomp;
   DEC_DECOMP* tempdecomp = NULL;

   assert(scip != NULL);
   assert(decomp != NULL);
   assert(newdecomp != NULL);

   if( DECdecompGetNBlocks(decomp) == 1 )
   {
      *newdecomp = NULL;
      return SCIP_OKAY;
   }
   *newdecomp = decomp;

   do
   {
      SCIP_CALL( DECtryAssignMasterconssToExistingPricing(scip, *newdecomp, &transferredexisting) );
      SCIPdebugMessage("%d conss transferred to existing pricing\n", transferredexisting);
      SCIP_CALL( DECtryAssignMasterconssToNewPricing(scip, *newdecomp, &tempdecomp, &transferrednew) );
      SCIPdebugMessage("%d conss transferred to new pricing\n", transferrednew);
      if( transferrednew > 0 )
      {
         if( *newdecomp != origdecomp )
         {
            SCIP_CALL( DECdecompFree(scip, newdecomp) );
         }
         *newdecomp = tempdecomp;
      }
   } while( transferredexisting > 0 || transferrednew > 0 );

   if( *newdecomp == origdecomp )
   {
      *newdecomp = NULL;
   }

   return SCIP_OKAY;
}

/** permutes the decomposition according to the permutation seed */
SCIP_RETCODE DECpermuteDecomp(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DECOMP*           decomp,             /**< decomposition data structure */
   SCIP_RANDNUMGEN*      randnumgen          /**< random number generator */
   )
{
   int b;
   int npricingprobs;
   assert(scip != NULL);
   assert(decomp != NULL);

   npricingprobs = DECdecompGetNBlocks(decomp);

   /* Permute individual variables and constraints of pricing problems */
   for( b = 0; b < npricingprobs; ++b )
   {
      SCIP_CONS*** subscipconss;
      SCIP_VAR*** subscipvars;
      int *nsubscipconss = DECdecompGetNSubscipconss(decomp);
      int *nsubscipvars = DECdecompGetNSubscipvars(decomp);
      subscipconss = DECdecompGetSubscipconss(decomp);

      SCIPrandomPermuteArray(randnumgen, (void**)(subscipconss[b]), 0, nsubscipconss[b]);

      subscipvars = DECdecompGetSubscipvars(decomp);
      SCIPrandomPermuteArray(randnumgen, (void**)(subscipvars[b]), 0, nsubscipvars[b]);
   }

   if( DECdecompGetNLinkingconss(decomp) > 0 )
   {
      SCIP_CONS** linkingconss = DECdecompGetLinkingconss(decomp);
      SCIPrandomPermuteArray(randnumgen, (void**)linkingconss, 0, DECdecompGetNLinkingconss(decomp));
   }

   if( DECdecompGetNLinkingvars(decomp) > 0 )
   {
      SCIP_VAR** linkingvars = DECdecompGetLinkingvars(decomp);;
      SCIPrandomPermuteArray(randnumgen, (void**)linkingvars, 0, DECdecompGetNLinkingvars(decomp));
   }

   SCIP_CALL( DECdecompCheckConsistency(scip, decomp) );
   return SCIP_OKAY;
}
