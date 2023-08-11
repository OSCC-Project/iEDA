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

/**@file   branch_generic.c
 * @ingroup BRANCHINGRULES
 * @brief  branching rule based on vanderbeck's generic branching scheme
 * @author Marcel Schmickerath
 * @author Martin Bergner
 * @author Jonas Witt
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

/*#define SCIP_DEBUG*/

#include "branch_generic.h"
#include "relax_gcg.h"
#include "cons_masterbranch.h"
#include "cons_origbranch.h"
#include "pricer_gcg.h"
#include "scip/cons_linear.h"
#include "type_branchgcg.h"
#include "gcg.h"
#include "cons_integralorig.h"
#include "gcgsort.h"

#include "scip/nodesel_bfs.h"
#include "scip/nodesel_dfs.h"
#include "scip/nodesel_estimate.h"
#include "scip/nodesel_hybridestim.h"
#include "scip/nodesel_restartdfs.h"
#include "scip/branch_allfullstrong.h"
#include "scip/branch_fullstrong.h"
#include "scip/branch_inference.h"
#include "scip/branch_mostinf.h"
#include "scip/branch_leastinf.h"
#include "scip/branch_pscost.h"
#include "scip/branch_random.h"
#include "scip/branch_relpscost.h"

#include <assert.h>
#include <string.h>


#define BRANCHRULE_NAME          "generic"
#define BRANCHRULE_DESC          "generic branching rule by Vanderbeck"
#define BRANCHRULE_PRIORITY      -100
#define BRANCHRULE_MAXDEPTH      -1
#define BRANCHRULE_MAXBOUNDDIST  1.0


#define EVENTHDLR_NAME         "genericbranchvaradd"
#define EVENTHDLR_DESC         "event handler for adding a new generated mastervar into the right branching constraints by using Vanderbecks generic branching scheme"

/** branching data */
struct GCG_BranchData
{
   GCG_COMPSEQUENCE**    C;                  /**< S[k] bound sequence for block k */ /* !!! sort of each C[i] = S is important !!! */
   SCIP_Real             lhs;                /**< lefthandside of the constraint corresponding to the bound sequence C */
   SCIP_CONS*            mastercons;         /**< constraint enforcing the branching restriction in the master problem */
   GCG_COMPSEQUENCE*     consS;              /**< component bound sequence which induce the current branching constraint */
   int                   consSsize;          /**< size of the component bound sequence */
   int                   maxconsS;           /**< size of consS */
   int                   consblocknr;        /**< id of the pricing problem (or block) to which this branching constraint belongs */
   int                   nvars;              /**< number of master variables the last time the node has been visited */
};

/** set of component bounds in separate */
struct GCG_Record
{
   GCG_COMPSEQUENCE**   record;              /**< array of component bound sequences in record */
   int                  recordsize;          /**< number of component bound sequences in record */
   int                  recordcapacity;      /**< capacity of record */
   int*                 sequencesizes;       /**< array of sizes of component bound sequences */
   int*                 capacities;          /**< array of capacities of component bound sequences */
};
typedef struct GCG_Record GCG_RECORD;

/*
 * Callback methods
 */

/* define not used callback as NULL*/
#define branchFreeGeneric NULL
#define branchExitGeneric NULL
#define branchInitsolGeneric NULL
#define branchExitsolGeneric NULL


/** computes the generator of mastervar for the entry in origvar
 * @return entry of the generator corresponding to origvar */
static
SCIP_Real getGeneratorEntry(
   SCIP_VAR*             mastervar,          /**< current mastervariable */
   SCIP_VAR*             origvar             /**< corresponding origvar */
   )
{
   int i;
   SCIP_VAR** origvars;
   SCIP_Real* origvals;
   int norigvars;

   assert(mastervar != NULL);
   assert(origvar != NULL);

   origvars = GCGmasterVarGetOrigvars(mastervar);
   norigvars = GCGmasterVarGetNOrigvars(mastervar);
   origvals = GCGmasterVarGetOrigvals(mastervar);

   for( i = 0; i < norigvars; ++i )
   {
      if( origvars[i] == origvar )
      {
         return origvals[i];
      }
   }

   return 0.0;
}

/** adds a variable to a branching constraint */
static
SCIP_RETCODE addVarToMasterbranch(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             mastervar,          /**< the variable to add */
   GCG_BRANCHDATA*       branchdata,         /**< branching data structure where the variable should be added */
   SCIP_Bool*            added               /**< whether the variable was added */
)
{
   int p;

   SCIP_Bool varinS = TRUE;

   assert(scip != NULL);
   assert(mastervar != NULL);
   assert(branchdata != NULL);
   assert(added != NULL);

   *added = FALSE;

   if( GCGvarGetBlock(mastervar) == -1 || GCGbranchGenericBranchdataGetConsblocknr(branchdata) == -3 || !GCGisMasterVarInBlock(mastervar, GCGbranchGenericBranchdataGetConsblocknr(branchdata)) )
      return SCIP_OKAY;

   SCIPdebugMessage("consSsize = %d\n", GCGbranchGenericBranchdataGetConsSsize(branchdata));

   for( p = 0; p < GCGbranchGenericBranchdataGetConsSsize(branchdata); ++p )
   {
      SCIP_Real generatorentry;

      generatorentry = getGeneratorEntry(mastervar, GCGbranchGenericBranchdataGetConsS(branchdata)[p].component);

      if( GCGbranchGenericBranchdataGetConsS(branchdata)[p].sense == GCG_COMPSENSE_GE )
      {
         if( SCIPisLT(scip, generatorentry, GCGbranchGenericBranchdataGetConsS(branchdata)[p].bound) )
         {
            varinS = FALSE;
            break;
         }
      }
      else
      {
         if( SCIPisGE(scip, generatorentry, GCGbranchGenericBranchdataGetConsS(branchdata)[p].bound) )
         {
            varinS = FALSE;
            break;
         }
      }
   }

   if( varinS )
   {
      SCIPdebugMessage("mastervar is added\n");
      SCIP_CALL( SCIPaddCoefLinear(scip, GCGbranchGenericBranchdataGetMastercons(branchdata), mastervar, 1.0) );
      *added = TRUE;
   }

   return SCIP_OKAY;
}

/** creates the constraint for branching directly on a master variable */
static
SCIP_RETCODE createDirectBranchingCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node to add constraint */
   GCG_BRANCHDATA*       branchdata          /**< branching data structure */
)
{
   char name[SCIP_MAXSTRLEN];

   assert(scip != NULL);
   assert(node != NULL);
   assert(branchdata->consblocknr == -3);
   assert(branchdata->consSsize == 1);

   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "directchild(%d, %g) sense = %d", branchdata->consSsize, branchdata->consS[0].bound, branchdata->consS[0].sense);

   /*  create constraint for child */
   if( branchdata->consS[0].sense == GCG_COMPSENSE_GE )
   {
      SCIP_CALL( SCIPcreateConsLinear(scip, &(branchdata->mastercons), name, 0, NULL, NULL, branchdata->consS[0].bound, SCIPinfinity(scip), TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE) );
   } else
   {
      SCIP_CALL( SCIPcreateConsLinear(scip, &(branchdata->mastercons), name, 0, NULL, NULL, -SCIPinfinity(scip), branchdata->consS[0].bound-1, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE) );
   }
   assert(GCGvarIsMaster(branchdata->consS[0].component));
   SCIP_CALL( SCIPaddCoefLinear(scip, branchdata->mastercons, branchdata->consS[0].component, 1.0) );

   /* add constraint to the master problem that enforces the branching decision */
   SCIP_CALL( SCIPaddConsNode(scip, node, branchdata->mastercons, NULL) );

   return SCIP_OKAY;
}

/** creates the constraint for branching directly on a master variable */
static
SCIP_RETCODE createBranchingCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node to add constraint */
   GCG_BRANCHDATA*       branchdata          /**< branching data structure */
)
{
   char name[SCIP_MAXSTRLEN];

   assert(scip != NULL);
   assert(node != NULL);
   assert(branchdata != NULL);

   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "child(%d, %g)", branchdata->consSsize, branchdata->lhs);

   assert(branchdata->mastercons == NULL);

   /*  create constraint for child */
   SCIP_CALL( SCIPcreateConsLinear(scip, &(branchdata->mastercons), name, 0, NULL, NULL,
         branchdata->lhs, SCIPinfinity(scip), TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE) );

   SCIP_CALL( SCIPaddConsNode(scip, node, branchdata->mastercons, NULL) );

   return SCIP_OKAY;
}
/** solving process initialization method of event handler (called when branch and bound process is about to begin) */
static
SCIP_DECL_EVENTINITSOL(eventInitsolGenericbranchvaradd)
{  /*lint --e{715}*/
    assert(scip != NULL);
    assert(eventhdlr != NULL);
    assert(strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_NAME) == 0);

    /* notify SCIP that your event handler wants to react on the event type */
    SCIP_CALL( SCIPcatchEvent( scip, SCIP_EVENTTYPE_VARADDED, eventhdlr, NULL, NULL) );

    return SCIP_OKAY;
}

/** solving process deinitialization method of event handler (called before branch and bound process data is freed) */
static
SCIP_DECL_EVENTEXITSOL(eventExitsolGenericbranchvaradd)
{  /*lint --e{715}*/
    assert(scip != NULL);
    assert(eventhdlr != NULL);
    assert(strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_NAME) == 0);

    /* notify SCIP that your event handler wants to drop the event type */
    SCIP_CALL( SCIPdropEvent( scip, SCIP_EVENTTYPE_VARADDED, eventhdlr, NULL, -1) );

    return SCIP_OKAY;
}

/** execution method of event handler */
static
SCIP_DECL_EVENTEXEC(eventExecGenericbranchvaradd)
{  /*lint --e{715}*/
   SCIP* origscip;
   SCIP_CONS* masterbranchcons;
   SCIP_CONS* parentcons;
   SCIP_VAR* mastervar;
   SCIP_VAR** allorigvars;
   SCIP_VAR** mastervars;
   GCG_BRANCHDATA* branchdata;
   int allnorigvars;
   int nmastervars;

   assert(eventhdlr != NULL);
   assert(strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_NAME) == 0);
   assert(event != NULL);
   assert(scip != NULL);
   assert(SCIPeventGetType(event) == SCIP_EVENTTYPE_VARADDED);

   mastervar = SCIPeventGetVar(event);
   if( !GCGvarIsMaster(mastervar) )
      return SCIP_OKAY;

   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   /*   SCIPdebugMessage("exec method of event_genericbranchvaradd\n"); */

   masterbranchcons = GCGconsMasterbranchGetActiveCons(scip);
   assert(masterbranchcons != NULL);

   /* if branch rule is not generic, abort */
   if( !GCGisBranchruleGeneric(GCGconsMasterbranchGetBranchrule(masterbranchcons)) )
      return SCIP_OKAY;

   SCIP_CALL( SCIPgetVarsData(origscip, &allorigvars, &allnorigvars, NULL, NULL, NULL, NULL) );
   SCIP_CALL( SCIPgetVarsData(scip, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );

   parentcons = masterbranchcons;
   branchdata = GCGconsMasterbranchGetBranchdata(parentcons);


   if( GCGvarIsMaster(mastervar) && GCGconsMasterbranchGetBranchrule(parentcons) != NULL )
   {
      SCIP_Bool added = FALSE;
      SCIPdebugMessage("Mastervar <%s>\n", SCIPvarGetName(mastervar));
      while( parentcons != NULL && branchdata != NULL
            && GCGbranchGenericBranchdataGetConsS(branchdata) != NULL && GCGbranchGenericBranchdataGetConsSsize(branchdata) > 0 )
      {
         if( GCGconsMasterbranchGetBranchrule(parentcons) == NULL || strcmp(SCIPbranchruleGetName(GCGconsMasterbranchGetBranchrule(parentcons)), "generic") != 0 )
            break;

         assert(branchdata != NULL);


         if( (GCGbranchGenericBranchdataGetConsblocknr(branchdata) != GCGvarGetBlock(mastervar) && GCGvarGetBlock(mastervar) != -1 )
               || (GCGvarGetBlock(mastervar) == -1 && !GCGmasterVarIsLinking(mastervar)) )
         {
            parentcons = GCGconsMasterbranchGetParentcons(parentcons);

            if( parentcons != NULL )
               branchdata = GCGconsMasterbranchGetBranchdata(parentcons);

            continue;
         }

         SCIP_CALL( addVarToMasterbranch(scip, mastervar, branchdata, &added) );

         parentcons = GCGconsMasterbranchGetParentcons(parentcons);
         branchdata = GCGconsMasterbranchGetBranchdata(parentcons);
      }
   }

   return SCIP_OKAY;
}


/*
 * branching specific interface methods
 */

/** method for initializing the set of respected indices */
static
SCIP_RETCODE InitIndexSet(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            F,                  /**< array of fractional mastervars */
   int                   Fsize,              /**< number of fractional mastervars */
   SCIP_VAR***           IndexSet,           /**< set to initialize */
   int*                  IndexSetSize        /**< size of the index set */
   )
{
   int i;

   assert( scip != NULL);
   assert( F != NULL);
   assert( Fsize > 0);
   assert( IndexSet != NULL);
   assert( IndexSetSize != NULL);

   *IndexSet = NULL;
   *IndexSetSize = 0;


   for( i = 0; i < Fsize; ++i )
   {
      int j;
      SCIP_VAR** origvars = GCGmasterVarGetOrigvars(F[i]);
      int norigvars = GCGmasterVarGetNOrigvars(F[i]);

      if( *IndexSetSize == 0 && norigvars > 0 )
      {
         /* TODO: allocate memory for norigvars although there might be slots for continuous variables which are not needed? */
         SCIP_CALL( SCIPallocBufferArray(scip, IndexSet, 1) );
         for( j = 0; j < norigvars; ++j )
         {
            if( SCIPvarGetType(origvars[j]) > SCIP_VARTYPE_INTEGER )
               continue;

            if( *IndexSetSize > 0 )
               SCIP_CALL( SCIPreallocBufferArray(scip, IndexSet, *IndexSetSize + 1) );

            (*IndexSet)[*IndexSetSize] = origvars[j];
            ++(*IndexSetSize);
         }
      }
      else
      {
         for( j = 0; j < norigvars; ++j )
         {
            int k;
            int oldsize = *IndexSetSize;

            if( SCIPvarGetType(origvars[j]) > SCIP_VARTYPE_INTEGER )
               continue;

            for( k = 0; k < oldsize; ++k )
            {
               /*  if variable already in union */
               if( (*IndexSet)[k] == origvars[j] )
               {
                  break;
               }
               if( k == oldsize-1 )
               {
                  /*  add variable to the end */
                  ++(*IndexSetSize);
                  SCIP_CALL( SCIPreallocBufferArray(scip, IndexSet, *IndexSetSize) );
                  (*IndexSet)[*IndexSetSize-1] = origvars[j];
               }
            }
         }
      }
   }

   return SCIP_OKAY;
}

/** method for calculating the median over all fractional components values using
 * the quickselect algorithm (or a variant of it)
 *
 * This method will change the array
 *
 * @return median or if the median is the minimum return ceil(arithm middle)
 */
static
SCIP_Real GetMedian(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real*            array,              /**< array to find the median in (will be destroyed) */
   int                   arraysize,          /**< size of the array */
   SCIP_Real             min                 /**< minimum of array */
   )
{
   SCIP_Real Median;
   SCIP_Real swap;
   int l;
   int r;
   int i;
   int MedianIndex;
   SCIP_Real arithmMiddle;

   assert(scip != NULL);
   assert(array != NULL);
   assert(arraysize > 0);

   r = arraysize -1;
   l = 0;
   arithmMiddle = 0;

   if( arraysize & 1 )
      MedianIndex = arraysize/2;
   else
      MedianIndex = arraysize/2 -1;

   while( l < r-1 )
   {
      int j = r;
      Median = array[MedianIndex];
      i = l;

      do
      {
         while( SCIPisLT(scip, array[i], Median) )
            ++i;
         while( SCIPisGT(scip, array[j], Median) )
            --j;
         if( i <=j )
         {
            swap = array[i];
            array[i] = array[j];
            array[j] = swap;
            ++i;
            --j;
         }
      } while( i <=j );
      if( j < MedianIndex )
         l = i;
      if( i > MedianIndex )
         r = j;
   }
   Median = array[MedianIndex];

   if( SCIPisEQ(scip, Median, min) )
   {
      for( i=0; i<arraysize; ++i )
         arithmMiddle += 1.0*array[i]/arraysize;

      Median = SCIPceil(scip, arithmMiddle);
   }

   return Median;
}

/** comparefunction for lexicographical sort */
static
GCG_DECL_SORTPTRCOMP(ptrcomp)
{
   SCIP* origprob = (SCIP *) userdata;
   GCG_STRIP* strip1;
   GCG_STRIP* strip2;
   SCIP_VAR* mastervar1;
   SCIP_VAR* mastervar2;
   SCIP_VAR** origvars;
   int norigvars;
   int i;

   strip1 = (GCG_STRIP*) elem1;
   strip2 = (GCG_STRIP*) elem2;

   mastervar1 = strip1->mastervar;
   mastervar2 = strip2->mastervar;

   assert(mastervar1 != NULL);
   assert(mastervar2 != NULL);

   if( GCGvarGetBlock(mastervar1) == -1 )
   {
      SCIPdebugMessage("linkingvar\n");
      assert(GCGmasterVarIsLinking(mastervar1));
   }
   if( GCGvarGetBlock(mastervar2) == -1 )
   {
      SCIPdebugMessage("linkingvar\n");
      assert(GCGmasterVarIsLinking(mastervar2));
   }

   origvars = SCIPgetVars(origprob);
   norigvars = SCIPgetNVars(origprob);

   for( i = 0; i < norigvars; ++i )
   {
      if( SCIPvarGetType(origvars[i]) > SCIP_VARTYPE_INTEGER )
         continue;

      if( SCIPisFeasGT(origprob, getGeneratorEntry(mastervar1, origvars[i]), getGeneratorEntry(mastervar2, origvars[i])) )
         return -1;

      if( SCIPisFeasLT(origprob, getGeneratorEntry(mastervar1, origvars[i]), getGeneratorEntry(mastervar2, origvars[i])) )
         return 1;
   }

   return 0;
}

/** lexicographical sort using scipsort
 * This method will change the array
 */
static
SCIP_RETCODE LexicographicSort(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_STRIP**           array,              /**< array to sort (will be changed) */
   int                   arraysize           /**< size of the array */
   )
{

   assert(array != NULL);
   assert(arraysize > 0);

   SCIPdebugMessage("Lexicographic sorting\n");

   GCGsortPtr((void**)array, ptrcomp, scip, arraysize );

   return SCIP_OKAY;
}


/** compare function for ILO: returns 1 if bd1 < bd2 else -1 with respect to bound sequence */
static
int ILOcomp(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             mastervar1,         /**< first strip */
   SCIP_VAR*             mastervar2,         /**< second strip */
   GCG_COMPSEQUENCE**    C,                  /**< component bound sequence to compare with */
   int                   NBoundsequences,    /**< size of the bound sequence */
   int*                  sequencesizes,      /**< sizes of the bound sequences */
   int                   p                   /**< current depth in C*/
   )
{
   SCIP_Real ivalue;
   int j;
   int k;
   int Nupper;
   int Nlower;
   int returnvalue;
   GCG_COMPSEQUENCE** CopyC;
   SCIP_VAR* origvar;
   int* newsequencesizes;
   GCG_STRIP* strip1;
   GCG_STRIP* strip2;

   k = 0;
   Nupper = 0;
   Nlower = 0;
   newsequencesizes = NULL;
   strip1 = NULL;
   strip2 = NULL;

   /* lexicographic Order? */
   if( C == NULL || NBoundsequences <= 1 )
   {
      SCIP_CALL_ABORT( SCIPallocBuffer(scip, &strip1) );
      SCIP_CALL_ABORT( SCIPallocBuffer(scip, &strip2) );

      strip1->scip = scip;
      strip2->scip = scip;
      strip1->C = NULL;
      strip2->C = NULL;
      strip1->Csize = 0;
      strip2->Csize = 0;
      strip1->sequencesizes = NULL;
      strip2->sequencesizes = NULL;
      strip1->mastervar = mastervar1;
      strip2->mastervar = mastervar2;

      returnvalue = (*ptrcomp)(scip, strip1, strip2);

      SCIPfreeBuffer(scip, &strip1);
      SCIPfreeBuffer(scip, &strip2);

      return returnvalue;
   }

   assert(C != NULL);
   assert(NBoundsequences > 0);

   /* find i which is in all S in C on position p */
   while( sequencesizes[k] < p )
   {
      ++k;
      assert(k < NBoundsequences);
   }
   origvar = C[k][p-1].component;
   assert(origvar != NULL);
   assert(SCIPvarGetType(origvar) <= SCIP_VARTYPE_INTEGER);
   ivalue = C[k][p-1].bound;

   /* calculate subset of C */
   for( j=0; j< NBoundsequences; ++j )
   {
      if( sequencesizes[j] >= p )
      {
         assert(C[j][p-1].component == origvar);
         if( C[j][p-1].sense == GCG_COMPSENSE_GE )
            ++Nupper;
         else
            ++Nlower;
      }
   }

   if( SCIPisGE(scip, getGeneratorEntry(mastervar1, origvar), ivalue) && SCIPisGE(scip, getGeneratorEntry(mastervar2, origvar), ivalue) )
   {
      k = 0;
      SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &CopyC, Nupper) );
      SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &newsequencesizes, Nupper) );
      for( j = 0; j < NBoundsequences; ++j )
      {
         if( sequencesizes[j] >= p )
            assert(C[j][p-1].component == origvar);

         if( sequencesizes[j] >= p && C[j][p-1].sense == GCG_COMPSENSE_GE )
         {
            int l;
            CopyC[k] = NULL;
            SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &(CopyC[k]), sequencesizes[j]) ); /*lint !e866*/
            for( l = 0; l < sequencesizes[j]; ++l )
            {
               CopyC[k][l] = C[j][l];
            }
            newsequencesizes[k] = sequencesizes[j];
            ++k;
         }
      }

      if( k != Nupper )
      {
         SCIPdebugMessage("k = %d, Nupper+1 =%d\n", k, Nupper+1);
      }

      if( Nupper != 0 )
         assert( k == Nupper );

      returnvalue = ILOcomp(scip, mastervar1, mastervar2, CopyC, Nupper, newsequencesizes, p+1);

      for( j = Nupper - 1; j >= 0; --j )
      {
         SCIPfreeBufferArray(scip, &(CopyC[j]) );
      }
      SCIPfreeBufferArray(scip, &newsequencesizes);
      SCIPfreeBufferArray(scip, &CopyC);

      return returnvalue;
   }


   if( SCIPisLT(scip, getGeneratorEntry(mastervar1, origvar), ivalue) && SCIPisLT(scip, getGeneratorEntry(mastervar2, origvar), ivalue) )
   {
      SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &CopyC, Nlower) );
      SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &newsequencesizes, Nlower) );

      k = 0;
      for( j = 0; j < NBoundsequences; ++j )
      {
         if( sequencesizes[j] >= p )
            assert(C[j][p-1].component == origvar);

         if( sequencesizes[j] >= p && C[j][p-1].sense != GCG_COMPSENSE_GE )
         {
            int l;
            CopyC[k] = NULL;
            SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &(CopyC[k]), sequencesizes[j]) ); /*lint !e866*/
            for( l = 0; l < sequencesizes[j]; ++l )
            {
               CopyC[k][l] = C[j][l];
            }
            newsequencesizes[k] = sequencesizes[j];
            ++k;
         }
      }

      if( k != Nlower )
      {
         SCIPdebugMessage("k = %d, Nlower+1 =%d\n", k, Nlower+1);
      }

      if( Nlower != 0 )
         assert( k == Nlower);

      returnvalue = ILOcomp( scip, mastervar1, mastervar2, CopyC, Nlower, newsequencesizes, p+1);

      for( j = Nlower - 1; j >= 0; --j )
      {

         SCIPfreeBufferArray(scip, &(CopyC[j]) );
      }
      SCIPfreeBufferArray(scip, &newsequencesizes);
      SCIPfreeBufferArray(scip, &CopyC);

      return returnvalue;
   }
   if( SCIPisGT(scip, getGeneratorEntry(mastervar1, origvar), getGeneratorEntry(mastervar2, origvar)) )
      return 1;
   else
      return -1;
}

/** comparefunction for induced lexicographical sort */
static
SCIP_DECL_SORTPTRCOMP(ptrilocomp)
{
   GCG_STRIP* strip1;
   GCG_STRIP* strip2;
   int returnvalue;

   strip1 = (GCG_STRIP*) elem1;
   strip2 = (GCG_STRIP*) elem2;

   returnvalue = ILOcomp(strip1->scip, strip1->mastervar, strip2->mastervar, strip1->C, strip1->Csize, strip1->sequencesizes, 1);

   return returnvalue;
}

/** induced lexicographical sort */
static
SCIP_RETCODE InducedLexicographicSort(
   SCIP*                 scip,               /**< SCIP ptr*/
   GCG_STRIP**           array,              /**< array of strips to sort in ILO*/
   int                   arraysize,          /**< size of the array */
   GCG_COMPSEQUENCE**    C,                  /**< current set o comp bound sequences*/
   int                   NBoundsequences,    /**< size of C */
   int*                  sequencesizes       /**< sizes of the sequences in C */
   )
{
   int i;

   SCIPdebugMessage("Induced Lexicographic sorting\n");

   if( NBoundsequences == 0 )
      return LexicographicSort( scip, array, arraysize );
   assert( C!= NULL );

   assert(arraysize > 0);
   if( arraysize <= 1 )
      return SCIP_OKAY;

   assert(array != NULL);
   for( i = 0; i < arraysize; ++i )
   {
      array[i]->scip = scip;
      array[i]->Csize = NBoundsequences;
      array[i]->sequencesizes = sequencesizes;
      array[i]->C = C;
   }

   SCIPsortPtr((void**)array, ptrilocomp, arraysize);

   return SCIP_OKAY;
}

/** partitions the strip according to the priority */
static
SCIP_RETCODE partition(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            J,                  /**< array of variables which belong to the discriminating components */
   int*                  Jsize,              /**< pointer to the number of discriminating components */
   SCIP_Longint*         priority,           /**< branching priorities */
   SCIP_VAR**            F,                  /**< set of fractional solutions satisfying bounds */
   int                   Fsize,              /**< size of list of fractional solutions satisfying bounds */
   SCIP_VAR**            origvar,            /**< pointer to store the variable which belongs to a discriminating component with maximum priority */
   SCIP_Real*            median              /**< pointer to store the median of fractional solutions satisfying bounds */
   )
{
   int j;
   int l;
   SCIP_Real min;
   SCIP_Real* compvalues;

   do
   {
      SCIP_Longint maxPriority = SCIP_LONGINT_MIN;
      min = SCIPinfinity(scip);

      /* max-min priority */
      for ( j = 0; j < *Jsize; ++j )
      {
         assert(SCIPvarGetType(J[j]) <= SCIP_VARTYPE_INTEGER);

         if ( priority[j] > maxPriority )
         {
            maxPriority = priority[j];
            *origvar = J[j];
         }
      }
      SCIP_CALL( SCIPallocBufferArray(scip, &compvalues, Fsize) );
      for ( l = 0; l < Fsize; ++l )
      {
         compvalues[l] = getGeneratorEntry(F[l], *origvar);
         if ( SCIPisLT(scip, compvalues[l], min) )
            min = compvalues[l];
      }
      *median = GetMedian(scip, compvalues, Fsize, min);
      SCIPfreeBufferArray(scip, &compvalues);

      assert(!SCIPisInfinity(scip, min));

      if ( !SCIPisEQ(scip, *median, 0.0) )
      {
         SCIPdebugMessage("median = %g\n", *median);
         SCIPdebugMessage("min = %g\n", min);
         SCIPdebugMessage("Jsize = %d\n", *Jsize);
      }

      if ( SCIPisEQ(scip, *median, min) )
      {
         /* here with max-min priority */
         for ( j = 0; j < *Jsize; ++j )
         {
            if ( *origvar == J[j] )
            {
               assert(priority[j] == 0);
               J[j] = J[*Jsize - 1];
               priority[j] = priority[*Jsize - 1];
               break;
            }
         }
         if( j < *Jsize )
            *Jsize = *Jsize-1;
      }
      assert(*Jsize >= 0);

   }while ( SCIPisEQ(scip, *median, min) && *Jsize > 0);

   return SCIP_OKAY;
}

/** add identified sequence to record */
static
SCIP_RETCODE addToRecord(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_RECORD*           record,             /**< record of identified sequences */
   GCG_COMPSEQUENCE*     S,                  /**< bound restriction sequence */
   int                   Ssize               /**< size of bound restriction sequence */
)
{
   int i;

   SCIPdebugMessage("recordsize=%d, Ssize=%d\n", record->recordsize, Ssize);

   if(record->recordcapacity < record->recordsize + 1)
   {
      i = SCIPcalcMemGrowSize(scip, record->recordsize + 1);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(record->record), record->recordcapacity, i) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(record->sequencesizes), record->recordcapacity, i) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(record->capacities), record->recordcapacity, i) );
      record->recordcapacity = i;
   }

   i = SCIPcalcMemGrowSize(scip, Ssize);
   record->capacities[record->recordsize] = i;
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(record->record[record->recordsize]), i) ); /*lint !e866*/
   for( i=0; i<Ssize;++i )
   {
      record->record[record->recordsize][i].component = S[i].component;
      record->record[record->recordsize][i].sense = S[i].sense;
      record->record[record->recordsize][i].bound = S[i].bound;
   }

   record->sequencesizes[record->recordsize] = Ssize; /* +1 ? */

   record->recordsize++;

   return SCIP_OKAY;
}


/** separation at the root node */
static
SCIP_RETCODE Separate(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            F,                  /**< fractional strips respecting bound restrictions */
   int                   Fsize,              /**< size of the strips */
   SCIP_VAR**            IndexSet,           /**< index set */
   int                   IndexSetSize,       /**< size of index set */
   GCG_COMPSEQUENCE*     S,                  /**< ordered set of bound restrictions */
   int                   Ssize,              /**< size of the ordered set */
   GCG_RECORD*           record              /**< identified bound sequences */
   )
{
   int j;
   int k;
   int l;
   int Jsize;
   SCIP_VAR** J;
   SCIP_Real median;
   int Fupper;
   int Flower;
   SCIP_Longint* priority;
   SCIP_VAR* origvar;
   SCIP_VAR** copyF;
   GCG_COMPSEQUENCE* upperLowerS;
   GCG_COMPSEQUENCE* upperS;
   SCIP_Real* alpha;
   SCIP_Real* compvalues;
   SCIP_Real muF;
   SCIP_Bool found;

   assert(scip != NULL);
   assert((Fsize == 0) == (F == NULL));

   Jsize = 0;
   Fupper = 0;
   Flower = 0;
   found = FALSE;
   priority = NULL;
   compvalues = NULL;
   J = NULL;
   origvar = NULL;
   copyF = NULL;
   upperLowerS = NULL;
   upperS = NULL;
   alpha = NULL;

   SCIPdebugMessage("Separate with ");

   /* if there are no fractional columns or potential columns, return */
   if( Fsize == 0 || IndexSetSize == 0 )
   {
      SCIPdebugPrintf("nothing, no fractional columns\n");
      return SCIP_OKAY;
   }

   assert( F != NULL );
   assert( IndexSet != NULL );

   muF = 0.0;
   for( j = 0; j < Fsize; ++j )
      muF += SCIPgetSolVal(GCGgetMasterprob(scip), NULL, F[j]);

   SCIPdebugPrintf("Fsize = %d; Ssize = %d, IndexSetSize = %d, nuF=%.6g \n", Fsize, Ssize, IndexSetSize, muF);

   /* detect fractional alpha_i */
   SCIP_CALL( SCIPallocBufferArray(scip, &alpha, IndexSetSize) );

   for( k = 0; k < IndexSetSize; ++k )
   {
      GCG_COMPSEQUENCE* copyS;
      SCIP_Real min;
      min = SCIPinfinity(scip);

      origvar = IndexSet[k];
      copyS = NULL;
      alpha[k] = 0.0;

      if( SCIPvarGetType(origvar) > SCIP_VARTYPE_INTEGER )
         continue;

      SCIP_CALL( SCIPallocBufferArray(scip, &compvalues, Fsize) );
      for( l=0; l<Fsize; ++l )
      {
         compvalues[l] = getGeneratorEntry(F[l], origvar);
         if( SCIPisLT(scip, compvalues[l], min) )
            min = compvalues[l];
      }

      median = GetMedian(scip, compvalues, Fsize, min);
      SCIPfreeBufferArray(scip, &compvalues);
      compvalues = NULL;

      for( j = 0; j < Fsize; ++j )
      {
         SCIP_Real generatorentry;

         generatorentry = getGeneratorEntry(F[j], origvar);

         if( SCIPisGE(scip, generatorentry, median) )
         {
            alpha[k] += SCIPgetSolVal(GCGgetMasterprob(scip), NULL, F[j]);
         }
      }

      if( SCIPisGT(scip, alpha[k], 0.0) && SCIPisLT(scip, alpha[k], muF) )
      {
         ++Jsize;
      }

      if( !SCIPisFeasIntegral(scip, alpha[k]) )
      {
         SCIPdebugMessage("alpha[%d] = %g\n", k, alpha[k]);
         found = TRUE;

         /* ********************************** *
          *   add the current pair to record   *
          * ********************************** */

         /* copy S */
         SCIP_CALL( SCIPallocBufferArray(scip, &copyS, (size_t)Ssize+1) );

         for( l = 0; l < Ssize; ++l )
         {
            copyS[l] = S[l];
         }

         /* create temporary array to compute median */
         SCIP_CALL( SCIPallocBufferArray(scip, &compvalues, Fsize) );

         for( l = 0; l < Fsize; ++l )
         {
            compvalues[l] = getGeneratorEntry(F[l], origvar);

            if( SCIPisLT(scip, compvalues[l], min) )
               min = compvalues[l];
         }

         assert( SCIPisEQ(scip, median, GetMedian(scip, compvalues, Fsize, min)));
         median = GetMedian(scip, compvalues, Fsize, min);
         SCIPfreeBufferArray(scip, &compvalues);
         compvalues = NULL;

         SCIPdebugMessage("new median is %g, comp=%s, Ssize=%d\n", median, SCIPvarGetName(origvar), Ssize);

         /* add last bound change to the copy of S */
         copyS[Ssize].component = origvar;
         copyS[Ssize].sense = GCG_COMPSENSE_GE;
         copyS[Ssize].bound = median;

         /* add identified sequence to record */
         SCIP_CALL( addToRecord(scip, record, copyS, Ssize+1) );


         /* ********************************** *
          *  end adding to record              *
          * ********************************** */
         SCIPfreeBufferArrayNull(scip, &copyS);
      }
   }

   if( found )
   {
      SCIPfreeBufferArrayNull(scip, &alpha);

      SCIPdebugMessage("one S found with size %d\n", record->sequencesizes[record->recordsize-1]);

      return SCIP_OKAY;
   }


   /* ********************************** *
    *  discriminating components         *
    * ********************************** */

   /** @todo mb: this is a filter */
   SCIP_CALL( SCIPallocBufferArray(scip, &J, Jsize) );
   j=0;
   for( k = 0; k < IndexSetSize; ++k )
   {
      if( SCIPisGT(scip, alpha[k], 0.0) && SCIPisLT(scip, alpha[k], muF) )
      {
         J[j] = IndexSet[k];
         ++j;
      }
   }
   assert( j == Jsize );

   /* ********************************** *
    *  compute priority  (max-min)       *
    * ********************************** */

   SCIP_CALL( SCIPallocBufferArray(scip, &priority, Jsize) );

   for( j = 0; j < Jsize; ++j )
   {
      SCIP_Longint maxcomp;
      SCIP_Longint mincomp;

      maxcomp = SCIP_LONGINT_MIN;
      mincomp = SCIP_LONGINT_MAX;

      origvar = J[j];

      for( l = 0; l < Fsize; ++l )
      {
         SCIP_Longint generatorentry;

         assert(SCIPisIntegral(scip, getGeneratorEntry(F[l], origvar)));

         generatorentry = (SCIP_Longint) (getGeneratorEntry(F[l], origvar) + 0.5);

         if( generatorentry > maxcomp )
            maxcomp = generatorentry;

         if( generatorentry < mincomp )
            mincomp = generatorentry;
      }

      priority[j] = maxcomp -mincomp;
   }

   SCIP_CALL( partition(scip, J, &Jsize, priority, F, Fsize, &origvar, &median) );

   /* this is a copy of S for the recursive call below */
   SCIP_CALL( SCIPallocBufferArray(scip, &upperLowerS, (size_t)Ssize+1) );
   SCIP_CALL( SCIPallocBufferArray(scip, &upperS, (size_t)Ssize+1) );

   for( l = 0; l < Ssize; ++l )
   {
      upperLowerS[l] = S[l];
      upperS[l] = S[l];
   }

   upperLowerS[Ssize].component = origvar;/* i; */
   upperS[Ssize].component = origvar;
   upperLowerS[Ssize].sense = GCG_COMPSENSE_LT;
   upperS[Ssize].sense = GCG_COMPSENSE_GE;
   upperLowerS[Ssize].bound = median;
   upperS[Ssize].bound = median;

   for( k = 0; k < Fsize; ++k )
   {
      if( SCIPisGE(scip, getGeneratorEntry(F[k], origvar), median) )
         ++Fupper;
      else
         ++Flower;
   }

   /* ********************************** *
    *  choose smallest partition         *
    * ********************************** */

   SCIP_CALL( SCIPallocBufferArray(scip, &copyF, Fsize) );

   if( Flower > 0 )
   {
      j = 0;

      for( k = 0; k < Fsize; ++k )
      {
         if( SCIPisLT(scip, getGeneratorEntry(F[k], origvar), median) )
         {
            copyF[j] = F[k];
            ++j;
         }
      }

      /*Fsize = Flower;*/
      assert(j < Fsize+1);

      SCIP_CALL( Separate( scip, copyF, Flower, J, Jsize, upperLowerS, Ssize+1, record) );
   }

   if( Fupper > 0 )
   {
      upperLowerS[Ssize].sense = GCG_COMPSENSE_GE;
      j = 0;

      for( k = 0; k < Fsize; ++k )
      {
         if( SCIPisGE(scip, getGeneratorEntry(F[k], origvar), median) )
         {
            copyF[j] = F[k];
            ++j;
         }
      }

      /*Fsize = Fupper;*/
      assert(j < Fsize+1);

      SCIP_CALL( Separate( scip, copyF, Fupper, J, Jsize, upperS, Ssize+1, record) );
   }

   SCIPfreeBufferArrayNull(scip, &copyF);
   SCIPfreeBufferArrayNull(scip, &upperS);
   SCIPfreeBufferArrayNull(scip, &upperLowerS);
   SCIPfreeBufferArray(scip, &priority);
   SCIPfreeBufferArrayNull(scip, &J);
   SCIPfreeBufferArray(scip, &alpha);

   return SCIP_OKAY;
}

/** choose a component bound sequence to create branching */
static
SCIP_RETCODE ChoseS(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_RECORD**          record,             /**< candidate of bound sequences */
   GCG_COMPSEQUENCE**    S,                  /**< pointer to return chosen bound sequence */
   int*                  Ssize               /**< size of the chosen bound sequence */
   )
{
   int minSizeOfMaxPriority;  /* needed if the last comp priority is equal to the one in other bound sequences */
   SCIP_Longint maxPriority;
   int i;
   int Index;

   minSizeOfMaxPriority = INT_MAX;
   maxPriority = SCIP_LONGINT_MIN;
   Index = -1;

   SCIPdebugMessage("Chose S \n");

   assert((*record)->recordsize > 0);

   for( i = 0; i < (*record)->recordsize; ++i )
   {
      assert((*record)->sequencesizes != NULL );
      assert((*record)->sequencesizes[i] > 0);

      if( maxPriority <= 1 ) /*  later by pseudocosts e.g. */
      {
         if( maxPriority < 1 )
         {
            maxPriority = 1; /*  only choose here first smallest S */
            minSizeOfMaxPriority = (*record)->sequencesizes[i];
            Index = i;
         }
         else if( (*record)->sequencesizes[i] < minSizeOfMaxPriority )
         {
               minSizeOfMaxPriority = (*record)->sequencesizes[i];
               Index = i;
         }
      }
   }
   assert(maxPriority > SCIP_LONGINT_MIN);
   assert(minSizeOfMaxPriority < INT_MAX);
   assert(Index >= 0);

   *Ssize = minSizeOfMaxPriority;
   SCIP_CALL( SCIPallocBufferArray(scip, S, *Ssize) );

   for( i = 0; i < *Ssize; ++i )
   {
      (*S)[i] =  (*record)->record[Index][i];
   }

   assert(S!=NULL);
   assert(*S!=NULL);

   /* free record */
   for( i = 0; i < (*record)->recordsize; ++i )
   {
      SCIPfreeBlockMemoryArray(scip, &((*record)->record[i]), (*record)->capacities[i] );
   }
   SCIPfreeBlockMemoryArray(scip, &((*record)->record), (*record)->recordcapacity );

   SCIPdebugMessage("with size %d \n", *Ssize);

   assert(*S!=NULL);

   return SCIP_OKAY;
}

/** updates the new set of sequences C in CopyC and the corresponding size array newsequencesizes
 *  returns the size of CopyC */
static
int computeNewSequence(
   int                   Csize,              /**< size of the sequence */
   int                   p,                  /**< index of node */
   SCIP_VAR*             origvar,            /**< another index generatorentry */
   int*                  sequencesizes,      /**< size of the sequences */
   GCG_COMPSEQUENCE**    C,                  /**< original sequence */
   GCG_COMPSEQUENCE**    CopyC,              /**< new sequence */
   int*                  newsequencesizes,   /**< output parameter for the new sequence */
   GCG_COMPSENSE         sense               /**< sense of the comparison */
   )
{
   int j;
   int k;

   for( k = 0, j = 0; j < Csize; ++j )
   {
      if ( sequencesizes[j] >= p )
         assert(C[j][p-1].component == origvar);

      if ( sequencesizes[j] >= p && C[j][p-1].sense == sense )
      {
         CopyC[k] = C[j];
         newsequencesizes[k] = sequencesizes[j];
         ++k;
      }
   }

   return k;
}

/** auxilary function to compute alpha for given index */
static
double computeAlpha(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   Fsize,              /**< size of F */
   GCG_COMPSENSE         isense,             /**< sense of the bound for origvar */
   double                ivalue,             /**< value of the bound for origvar */
   SCIP_VAR*             origvar,            /**< index of the variable */
   SCIP_VAR**            F                   /**< current fractional mastervars*/
   )
{
   int j;
   SCIP_Real alpha_i = 0.0;

   for ( j = 0; j < Fsize; ++j )
   {
      SCIP_Real generatorentry;

      generatorentry = getGeneratorEntry(F[j], origvar);

      if ( (isense == GCG_COMPSENSE_GE && SCIPisGE(scip, generatorentry, ivalue)) ||
           (isense == GCG_COMPSENSE_LT && SCIPisLT(scip, generatorentry, ivalue)) )
      {
         alpha_i += SCIPgetSolVal(GCGgetMasterprob(scip), NULL, F[j]);
      }
   }

   return alpha_i;
}

/** separation at a node other than the root node */
static
SCIP_RETCODE Explore(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_COMPSEQUENCE**    C,                  /**< array of component bounds sequences*/
   int                   Csize,              /**< number of component bounds sequences*/
   int*                  sequencesizes,      /**< array of sizes of component bounds sequences */
   int                   p,                  /**< depth of recursive call */
   SCIP_VAR**            F,                  /**< strip of fractional columns */
   int                   Fsize,              /**< size of the strips */
   SCIP_VAR**            IndexSet,           /**< array of original variables which belong to the fractional master columns */
   int                   IndexSetSize,       /**< number of original variables which belong to fractional master columns. Note that IndexSetSize >= Fsize */
   GCG_COMPSEQUENCE**    S,                  /**< component sequences */
   int*                  Ssize,              /**< length of component sequences */
   GCG_RECORD*           record              /**< array of sets of component bounds which we might use for separation */
   )
{
   int j;
   int k;
   SCIP_Real ivalue;
   GCG_COMPSENSE isense;
   SCIP_Real median;
   int Fupper;
   int Flower;
   int Cupper;
   int Clower;
   int lowerSsize;
   SCIP_VAR** copyF;
   GCG_COMPSEQUENCE* copyS;
   GCG_COMPSEQUENCE* lowerS;
   GCG_COMPSEQUENCE** CopyC;
   int* newsequencesizes;
   SCIP_VAR* origvar;
   SCIP_Real alpha_i;
   SCIP_Real  muF;
   SCIP_Bool found;

   k = 0;
   muF = 0;
   Fupper = 0;
   Flower = 0;
   Cupper = 0;
   Clower = 0;
   newsequencesizes = NULL;
   copyF = NULL;
   CopyC = NULL;
   copyS = NULL;
   lowerS =  NULL;
   found = FALSE;

   SCIPdebugMessage("Explore\n");

   SCIPdebugMessage("with Fsize = %d, Csize = %d, Ssize = %d, p = %d\n", Fsize, Csize, *Ssize, p);

   /* *************************************** *
    *   if C=Ø, call separate and return that *
    * *************************************** */

   if( C == NULL || Fsize==0 || IndexSetSize==0 || Csize == 0 )
   {
      /* SCIPdebugMessage("go to Separate\n"); */
      assert(S != NULL);

      SCIP_CALL( Separate( scip, F, Fsize, IndexSet, IndexSetSize, *S, *Ssize, record) );

      if( *Ssize > 0 && *S != NULL )
      {
         SCIPfreeBufferArrayNull(scip, S);
         *S = NULL;
         *Ssize = 0;
      }
      return SCIP_OKAY;
   }

   assert(C != NULL);
   assert(Csize > 0);
   assert(F != NULL);
   assert(IndexSet != NULL);
   assert(sequencesizes != NULL);

   /* ******************************************* *
    * find i which is in all S in C on position p *
    * ******************************************* */

   while( sequencesizes[k] < p )
   {
    /*   SCIPdebugMessage("sequencesizes[%d] = %d\n", k, sequencesizes[k]); */
      ++k;

      if( k >= Csize )
      {
         SCIPdebugMessage("no %dth element bounded\n", p);
         assert(S != NULL);
         SCIP_CALL( Separate( scip, F, Fsize, IndexSet, IndexSetSize, *S, *Ssize, record) );

         if( *Ssize > 0 && *S != NULL )
         {
            SCIPfreeBufferArrayNull(scip, S);
            *S = NULL;
            *Ssize = 0;
         }

         return SCIP_OKAY;
      }

      assert( k < Csize );
   }

   origvar = C[k][p-1].component;
   isense = C[k][p-1].sense;
   ivalue = C[k][p-1].bound;

   assert(origvar != NULL);
   assert(SCIPvarGetType(origvar) <= SCIP_VARTYPE_INTEGER);

   SCIPdebugMessage("orivar = %s; ivalue = %g\n", SCIPvarGetName(origvar), ivalue);

   for( j = 0; j < Fsize; ++j )
   {
      muF += SCIPgetSolVal(GCGgetMasterprob(scip), NULL, F[j]);
   }

   SCIPdebugMessage("muF = %g\n", muF);

   /* ******************************************* *
    * compute alpha_i                             *
    * ******************************************* */

   alpha_i = computeAlpha(scip, Fsize, isense, ivalue, origvar, F);

   SCIPdebugMessage("alpha(%s) = %g\n", SCIPvarGetName(origvar), alpha_i);

   /* ******************************************* *
    * if f > 0, add pair to record                *
    * ******************************************* */

   if( !SCIPisFeasIntegral(scip, alpha_i) )
   {
      int l;

      found = TRUE;
       SCIPdebugMessage("fractional alpha(%s) = %g\n", SCIPvarGetName(origvar), alpha_i);

      /* ******************************************* *
       * add to record                               *
       * ******************************************* */

         SCIP_CALL( SCIPallocBufferArray(scip, &copyS, (size_t)*Ssize+1) );
         for( l = 0; l < *Ssize; ++l )
         {
            copyS[l] = (*S)[l];
         }
         copyS[*Ssize].component = origvar;
         copyS[*Ssize].sense = isense;
         copyS[*Ssize].bound = ivalue;
         SCIP_CALL( addToRecord(scip, record, copyS, *Ssize+1) );
   }

   if( found )
   {
      SCIPdebugMessage("found fractional alpha\n");
      SCIPfreeBufferArrayNull(scip, &copyS);
      return SCIP_OKAY;
   }

   /* add bound to the end of S */
   ++(*Ssize);
   assert(S != NULL );

   SCIP_CALL( SCIPreallocBufferArray(scip, S, *Ssize) );

   median = ivalue;
   (*S)[*Ssize-1].component = origvar;
   (*S)[*Ssize-1].sense = GCG_COMPSENSE_GE;
   (*S)[*Ssize-1].bound = median;

   SCIP_CALL( SCIPallocBufferArray(scip, &lowerS, *Ssize) );

   for( k = 0; k < *Ssize-1; ++k )
   {
      lowerS[k].component = (*S)[k].component;
      lowerS[k].sense = (*S)[k].sense;
      lowerS[k].bound = (*S)[k].bound;
   }

   lowerSsize = *Ssize;
   lowerS[lowerSsize-1].component = origvar;
   lowerS[lowerSsize-1].sense = GCG_COMPSENSE_LT;
   lowerS[lowerSsize-1].bound = median;

   for( k = 0; k < Fsize; ++k )
   {
      if( SCIPisGE(scip, getGeneratorEntry(F[k], origvar), median) )
         ++Fupper;
      else
         ++Flower;
   }

   /* calculate subset of C */
   for( j = 0; j < Csize; ++j )
   {
      if( sequencesizes[j] >= p )
      {
         if( C[j][p-1].sense == GCG_COMPSENSE_GE )
         {
            ++Cupper;
         }
         else
         {
            ++Clower;
            assert( C[j][p-1].sense == GCG_COMPSENSE_LT );
         }
      }
   }

   SCIPdebugMessage("Cupper = %d, Clower = %d\n", Cupper, Clower);

   if( SCIPisLE(scip, alpha_i, 0.0) && Fupper != 0 )
      Flower = INT_MAX;
   if( SCIPisEQ(scip, alpha_i, muF) && Flower != 0 )
      Fupper = INT_MAX;

   if( Fupper > 0  && Fupper != INT_MAX )
   {
      SCIPdebugMessage("chose upper bound Fupper = %d, Cupper = %d\n", Fupper, Cupper);

      SCIP_CALL( SCIPallocBufferArray(scip, &copyF, Fupper) );
      for( j = 0, k = 0; k < Fsize; ++k )
      {
         if( SCIPisGE(scip, getGeneratorEntry(F[k], origvar), median) )
         {
            copyF[j] = F[k];
            ++j;
         }
      }

      /* new C */
      if( Fupper > 0 )
      {
         SCIP_CALL( SCIPallocBufferArray(scip, &CopyC, Cupper) );
         SCIP_CALL( SCIPallocBufferArray(scip, &newsequencesizes, Cupper) );
         k = computeNewSequence(Csize, p, origvar, sequencesizes, C, CopyC, newsequencesizes, GCG_COMPSENSE_GE);
         if( k != Cupper )
         {
            SCIPdebugMessage("k = %d, p = %d\n", k, p);
         }
         assert(k == Cupper);
      }
      else
      {
         CopyC = NULL;
      }

      SCIP_CALL( Explore( scip, CopyC, Cupper, newsequencesizes, p+1, copyF, Fupper, IndexSet, IndexSetSize, S, Ssize, record) );
   }

   if( Flower > 0 && Flower != INT_MAX )
   {
      SCIPdebugMessage("chose lower bound Flower = %d Clower = %d\n", Flower, Clower);

      SCIP_CALL( SCIPreallocBufferArray(scip, &copyF, Flower) );

      j = 0;
      for( k = 0; k < Fsize; ++k )
      {
         if( SCIPisLT(scip, getGeneratorEntry(F[k], origvar), median) )
         {
            copyF[j] = F[k];
            ++j;
         }
      }

      /* new C */
      if( Flower > 0 )
      {
         SCIP_CALL( SCIPreallocBufferArray(scip, &CopyC, Clower) );
         SCIP_CALL( SCIPreallocBufferArray(scip, &newsequencesizes, Clower) );
         k = computeNewSequence(Csize, p, origvar, sequencesizes, C, CopyC, newsequencesizes, GCG_COMPSENSE_LT);
         if( k != Clower )
         {
            SCIPdebugMessage("k = %d, p = %d\n", k, p);
         }
         assert(k == Clower);
      }
      else
      {
         CopyC = NULL;
      }

      SCIP_CALL( Explore( scip, CopyC, Clower, newsequencesizes, p+1, copyF, Flower, IndexSet, IndexSetSize, &lowerS, &lowerSsize, record) );
   }

   SCIPfreeBufferArrayNull(scip, &newsequencesizes);
   SCIPfreeBufferArrayNull(scip, &CopyC);
   SCIPfreeBufferArrayNull(scip, &copyF);
   SCIPfreeBufferArrayNull(scip, &lowerS);
   SCIPfreeBufferArrayNull(scip, &copyS);

   if( *Ssize > 0 && *S != NULL )
   {
      SCIPfreeBufferArrayNull(scip, S);

      *Ssize = 0;
   }

   return SCIP_OKAY;
}

/** callup method for seperate
 * decides whether Separate or Explore should be done */
static
SCIP_RETCODE ChooseSeparateMethod(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            F                   /**< strip of fractional columns */,
   int                   Fsize,              /**< size of the strips */
   GCG_COMPSEQUENCE**    S,                  /**< array of existing bound sequences */
   int*                  Ssize,              /**< size of existing bound sequences */
   GCG_COMPSEQUENCE**    C,                  /**< array of component bounds sequences*/
   int                   Csize,              /**< number of component bounds sequences*/
   int*                  CompSizes,          /**< array of sizes of component bounds sequences */
   int                   blocknr,            /**< id of the pricing problem (or block) in which we want to branch */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_RESULT*          result,             /**< pointer to store the result of the branching call */
   int**                 checkedblocks,      /**< array to store which blocks have been checked */
   int*                  ncheckedblocks,     /**< number of blocks that have beend checked */
   GCG_STRIP****         checkedblockssortstrips, /**< sorted strips of checked blocks */
   int**                 checkedblocksnsortstrips /**< size of the strips */
   )
{
   int i;
   SCIP_VAR** IndexSet;
   SCIP_VAR** mastervars;
   int IndexSetSize;
   GCG_RECORD* record;
   int exploreSsize;
   GCG_COMPSEQUENCE* exploreS;
   GCG_STRIP** strips;
   int nstrips;
   int nmastervars;

   assert(Fsize > 0);
   assert(F != NULL);
   IndexSetSize = 0;
   exploreSsize = 0;
   exploreS = NULL;
   record = NULL;
   IndexSet = NULL;
   strips = NULL;
   nstrips = 0;

   SCIPdebugMessage("Calling Separate\n");

   SCIP_CALL( SCIPallocBuffer(scip, &record) );
   record->recordsize = 0;
   record->record = NULL;
   record->sequencesizes = NULL;
   record->capacities = NULL;
   record->recordcapacity = 0;

   /* calculate IndexSet */
   SCIP_CALL( InitIndexSet(scip, F, Fsize, &IndexSet, &IndexSetSize) );
   assert(IndexSetSize > 0);
   assert(IndexSet != NULL);

   /* rootnode? */
   if( Csize<=0 )
      SCIP_CALL( Separate( scip, F, Fsize, IndexSet, IndexSetSize, NULL, 0, record) );
   else
   {
      assert( C!=NULL );
      SCIP_CALL( Explore( scip, C, Csize, CompSizes, 1, F, Fsize, IndexSet, IndexSetSize, &exploreS, &exploreSsize, record) );

      SCIPfreeBufferArrayNull(scip, &exploreS);
   }

   assert(record != NULL);

   if( record->recordsize <= 0 )
   {
      SCIP_CALL( SCIPgetVarsData(GCGgetMasterprob(scip), &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );

      ++(*ncheckedblocks);
      assert((*ncheckedblocks) <= GCGgetNPricingprobs(scip)+1);

      if( (*ncheckedblocks) > 1 )
      {
         SCIP_CALL( SCIPreallocBufferArray(scip, checkedblocks, *ncheckedblocks) );
         SCIP_CALL( SCIPreallocBufferArray(scip, checkedblocksnsortstrips, *ncheckedblocks) );
         SCIP_CALL( SCIPreallocBufferArray(scip, checkedblockssortstrips, *ncheckedblocks) );
      }
      else
      {
         SCIP_CALL( SCIPallocBufferArray(scip, checkedblocks, *ncheckedblocks) );
         SCIP_CALL( SCIPallocBufferArray(scip, checkedblocksnsortstrips, *ncheckedblocks) );
         SCIP_CALL( SCIPallocBufferArray(scip, checkedblockssortstrips, *ncheckedblocks) );
      }

      (*checkedblocks)[(*ncheckedblocks)-1] = blocknr;

      for( i=0; i<nmastervars; ++i )
      {
         if( GCGisMasterVarInBlock(mastervars[i], blocknr) )
         {
            ++nstrips;

            SCIP_CALL( SCIPreallocBufferArray(scip, &strips, nstrips) );

            assert(strips != NULL);

            SCIP_CALL( SCIPallocBuffer(scip, &(strips[nstrips-1])) ); /*lint !e866*/
            assert(strips[nstrips-1] != NULL);

            strips[nstrips-1]->C = NULL;
            strips[nstrips-1]->mastervar = mastervars[i];
            strips[nstrips-1]->Csize = 0;
            strips[nstrips-1]->sequencesizes = NULL;
            strips[nstrips-1]->scip = NULL;
         }
      }

      SCIP_CALL( InducedLexicographicSort(scip, strips, nstrips, C, Csize, CompSizes) );

      (*checkedblocksnsortstrips)[(*ncheckedblocks)-1] = nstrips;

      SCIP_CALL( SCIPallocBufferArray(scip, &((*checkedblockssortstrips)[(*ncheckedblocks)-1]), nstrips) ); /*lint !e866*/

      /* sort the direct copied origvars at the end */

      for( i = 0; i < nstrips; ++i )
      {
         assert(strips != NULL);

         SCIP_CALL( SCIPallocBuffer(scip, &(*checkedblockssortstrips[*ncheckedblocks-1][i])) ); /*lint !e866*/
         *checkedblockssortstrips[*ncheckedblocks-1][i] = strips[i];
      }

      for( i=0; i<nstrips; ++i )
      {
         assert(strips != NULL);
         SCIPfreeBuffer(scip, &(strips[i]));
         strips[i] = NULL;
      }
      SCIPfreeBufferArrayNull(scip, &strips);

      /*choose new block */
      SCIP_CALL( GCGbranchGenericInitbranch(GCGgetMasterprob(scip), branchrule, result, checkedblocks, ncheckedblocks, checkedblockssortstrips, checkedblocksnsortstrips) );

   }
   else
   {
      if( ncheckedblocks != NULL && (*ncheckedblocks) > 0 )
      {
         for( i = (*ncheckedblocks) - 1; i >= 0; --i )
         {
            int j;

            for( j = (*checkedblocksnsortstrips)[i] - 1; j >= 0; --j )
            {
               SCIPfreeBuffer(scip, &((*checkedblockssortstrips)[i][j]) );
            }

            SCIPfreeBufferArray(scip, &((*checkedblockssortstrips)[i]) );
         }

         SCIPfreeBufferArray(scip, checkedblockssortstrips);
         SCIPfreeBufferArray(scip, checkedblocksnsortstrips);
         SCIPfreeBufferArray(scip, checkedblocks);
         *ncheckedblocks = 0;
      }
   }

   if( record->recordsize > 0 )
   {
      SCIP_CALL( ChoseS( scip, &record, S, Ssize) );
      assert(*S != NULL);
   }


   SCIPfreeBufferArray(scip, &IndexSet);
   if( record != NULL )
   {
      SCIPfreeBlockMemoryArrayNull(scip, &record->capacities, record->recordcapacity);
      SCIPfreeBlockMemoryArrayNull(scip, &record->sequencesizes, record->recordcapacity);
      SCIPfreeBlockMemoryArrayNull(scip, &record->record, record->recordcapacity);
      SCIPfreeBuffer(scip, &record);
   }
   return SCIP_OKAY;
}

/** callback deletion method for branching data*/
static
GCG_DECL_BRANCHDATADELETE(branchDataDeleteGeneric)
{
   assert(scip != NULL);
   assert(branchdata != NULL);

   if( *branchdata == NULL )
   {
      SCIPdebugMessage("branchDataDeleteGeneric: cannot delete empty branchdata\n");

      return SCIP_OKAY;
   }

   if( (*branchdata)->mastercons != NULL )
   {
      SCIPdebugMessage("branchDataDeleteGeneric: child blocknr %d, %s\n", (*branchdata)->consblocknr,
         SCIPconsGetName((*branchdata)->mastercons) );
   }
   else
   {
      SCIPdebugMessage("branchDataDeleteGeneric: child blocknr %d, empty mastercons\n", (*branchdata)->consblocknr);
   }

   /* release constraint that enforces the branching decision */
   if( (*branchdata)->mastercons != NULL )
   {
      SCIP_CALL( SCIPreleaseCons(GCGgetMasterprob(scip), &(*branchdata)->mastercons) );
      (*branchdata)->mastercons = NULL;
   }

   if( (*branchdata)->consS != NULL && (*branchdata)->consSsize > 0 )
   {
      SCIPfreeBlockMemoryArrayNull(scip, &((*branchdata)->consS), (*branchdata)->maxconsS);
      (*branchdata)->consS = NULL;
      (*branchdata)->consSsize = 0;
   }

   SCIPfreeBlockMemoryNull(scip, branchdata);
   *branchdata = NULL;

   return SCIP_OKAY;
}

/** check method for pruning ChildS directly on childnodes
 *  @returns TRUE if node is pruned */
static
SCIP_Bool checkchildconsS(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             lhs,                /**< lhs for childnode which is checkes to be pruned */
   GCG_COMPSEQUENCE*     childS,             /**< component bound sequence defining the childnode */
   int                   childSsize,         /**< size of component bound sequence */
   SCIP_CONS*            parentcons,         /**< constraint of parent node in B&B tree */
   int                   childBlocknr        /**< number of the block for the child node */
   )
{
   int i;
   int nchildren;

   nchildren = GCGconsMasterbranchGetNChildconss(parentcons);
   assert(nchildren>0);

   for( i=0; i<nchildren; ++i )
   {
      SCIP_CONS* childcons;
      GCG_BRANCHDATA* branchdata;
      SCIP_Bool same;
      int j;

      same = TRUE;
      childcons = GCGconsMasterbranchGetChildcons(parentcons, i);
      if( childcons == NULL )
         continue;

      if( GCGconsMasterbranchGetBranchrule(childcons) != NULL && strcmp(SCIPbranchruleGetName(GCGconsMasterbranchGetBranchrule(childcons)), "generic") != 0 )
         continue;

      branchdata = GCGconsMasterbranchGetBranchdata(childcons);
      assert(branchdata != NULL);

      if( childBlocknr != branchdata->consblocknr || childSsize != branchdata->consSsize || !SCIPisEQ(scip, lhs, branchdata->lhs) )
         continue;

      assert(childSsize > 0 && branchdata->consSsize > 0);

      for( j=0; j< childSsize; ++j )
      {
         if( childS[j].component != branchdata->consS[j].component
            || childS[j].sense != branchdata->consS[j].sense
            || !SCIPisEQ(scip, childS[j].bound, branchdata->consS[j].bound) )
         {
            same = FALSE;
            break;
         }
      }

      if( same )
      {
         SCIPdebugMessage("child pruned \n");
         return TRUE;
      }
   }
   return FALSE;
}

/** check method for pruning ChildS indirectly by parentnodes
 *  retrun TRUE if node is pruned */
static
SCIP_Bool pruneChildNodeByDominanceGeneric(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             lhs,                /**< lhs for childnode which is checkes to be pruned */
   GCG_COMPSEQUENCE*     childS,             /**< Component Bound Sequence defining the childnode */
   int                   childSsize,         /**< size of component bound sequence */
   SCIP_CONS*            masterbranchcons,   /**< master branching constraint */
   int                   childBlocknr        /**< number of the block for the childnode */
   )
{
   SCIP_CONS* cons;

   SCIPdebugMessage("Prune by dominance\n");
   cons = GCGconsMasterbranchGetParentcons(masterbranchcons);

   if( cons == NULL )
   {
      SCIPdebugMessage("cons == NULL, not pruned\n");
      return FALSE;
   }
   while( cons != NULL )
   {
      GCG_BRANCHDATA* parentdata;
      SCIP_Bool ispruned;

      parentdata = GCGconsMasterbranchGetBranchdata(cons);
      if( parentdata == NULL )
      {
         /* root node: check children for pruning */
         return checkchildconsS(scip, lhs, childS, childSsize, cons, childBlocknr);
      }
      if( strcmp(SCIPbranchruleGetName(GCGconsMasterbranchGetBranchrule(cons)), "generic") != 0 )
         return checkchildconsS(scip, lhs, childS, childSsize, cons, childBlocknr);

      ispruned = checkchildconsS(scip, lhs, childS, childSsize, cons, childBlocknr);

      if( ispruned )
      {
         return TRUE;
      }

      cons = GCGconsMasterbranchGetParentcons(cons);
   }

   SCIPdebugMessage("child not pruned\n");
   return FALSE;
}

/** initialize branchdata at the node */
static
SCIP_RETCODE initNodeBranchdata(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_BRANCHDATA**      nodebranchdata,     /**< branching data to set */
   int                   blocknr             /**< block we are branching in */
   )
{
   SCIP_CALL( SCIPallocBlockMemory(scip, nodebranchdata) );

   (*nodebranchdata)->consblocknr = blocknr;
   (*nodebranchdata)->mastercons = NULL;
   (*nodebranchdata)->consS = NULL;
   (*nodebranchdata)->C = NULL;
   (*nodebranchdata)->maxconsS = 0;
   (*nodebranchdata)->consSsize = 0;
   (*nodebranchdata)->nvars = 0;

   return SCIP_OKAY;
}

/** for given component bound sequence S, create |S|+1 Vanderbeck branching nodes */
static
SCIP_RETCODE createChildNodesGeneric(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   GCG_COMPSEQUENCE*     S,                  /**< Component Bound Sequence defining the nodes */
   int                   Ssize,              /**< size of S */
   int                   blocknr,            /**< number of the block */
   SCIP_CONS*            masterbranchcons,   /**< current masterbranchcons*/
   SCIP_RESULT*          result              /**< pointer to store the result of the branching call */
   )
{
#ifdef SCIP_DEBUG
   SCIP_Real identicalcontrol = 0;
#endif
   SCIP*  masterscip;
   int i;
   int p;
   SCIP_Real pL;
   SCIP_Real L;
   SCIP_Real lhsSum;
   int nmastervars;
   int nmastervars2;
   int ncopymastervars;
   int nbranchcands;
   int nchildnodes;
   SCIP_VAR** mastervars;
   SCIP_VAR** mastervars2;
   SCIP_VAR** branchcands;

   assert(scip != NULL);
   assert(Ssize > 0);
   assert(S != NULL);

   lhsSum = 0;
   nchildnodes = 0;
   L = 0;

   pL = GCGgetNIdenticalBlocks(scip, blocknr);
   SCIPdebugMessage("Vanderbeck branching rule Node creation for blocknr %d with %.1f identical blocks \n", blocknr, pL);


   /*  get variable data of the master problem */
   masterscip = GCGgetMasterprob(scip);
   assert(masterscip != NULL);
   SCIP_CALL( SCIPgetVarsData(masterscip, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );
   nmastervars2 = nmastervars;
   assert(nmastervars >= 0);
   SCIP_CALL( SCIPduplicateBufferArray(scip, &mastervars2, mastervars, nmastervars) );

   SCIP_CALL( SCIPgetLPBranchCands(masterscip, &branchcands, NULL, NULL, &nbranchcands, NULL, NULL) );

   SCIPdebugMessage("Vanderbeck branching rule: creating %d nodes\n", Ssize+1);

   for( p=0; p<Ssize+1; ++p )
   {
      GCG_BRANCHDATA* branchchilddata;
      SCIP_NODE* child;
      SCIP_CONS* childcons;
      int k;

      SCIP_Real lhs;
      branchchilddata = NULL;

      /*  allocate branchdata for child and store information */
      SCIP_CALL( initNodeBranchdata(scip, &branchchilddata, blocknr) );

      if( p == Ssize )
      {
         branchchilddata->maxconsS = SCIPcalcMemGrowSize(scip, Ssize);
         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(branchchilddata->consS), branchchilddata->maxconsS) );
         branchchilddata->consSsize = Ssize;
      }
      else
      {
         branchchilddata->maxconsS = SCIPcalcMemGrowSize(scip, p+1);
         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(branchchilddata->consS), branchchilddata->maxconsS) );
         branchchilddata->consSsize = p+1;
      }
      for( k = 0; k <= p; ++k )
      {
         GCG_COMPSEQUENCE compBound;

         if( k == Ssize )
         {
            assert( p == Ssize );
            compBound = S[k-1];
            branchchilddata->consS[k-1] = compBound;
         }
         else
         {
            compBound = S[k];
            if( k >= p )
            {
               if( S[p].sense == GCG_COMPSENSE_GE )
                  compBound.sense = GCG_COMPSENSE_LT;
               else
                  compBound.sense = GCG_COMPSENSE_GE;
            }
            branchchilddata->consS[k] = compBound;
         }
      }

      /* last node? */
      if( p == Ssize )
      {
         lhs = pL;
      }
      else
      {
         /* calculate mu */
         SCIP_Real mu = 0.0;

         ncopymastervars = nmastervars2;
         for( i = 0; i < ncopymastervars; ++i )
         {
            SCIP_VAR* swap;

            if( i >= nmastervars2 )
               break;

            if( GCGisMasterVarInBlock(mastervars2[i], blocknr) )
            {
               SCIP_Real generator_i = getGeneratorEntry(mastervars2[i], S[p].component);

               if( (S[p].sense == GCG_COMPSENSE_GE && SCIPisGE(scip, generator_i, S[p].bound)) ||
                  (S[p].sense == GCG_COMPSENSE_LT && SCIPisLT(scip, generator_i, S[p].bound) ) )
               {
                  mu += SCIPgetSolVal(masterscip, NULL, mastervars2[i]);
               }
               else if( ncopymastervars > 0 )
               {
                  swap = mastervars2[i];
                  mastervars2[i] = mastervars2[nmastervars2-1];
                  mastervars2[nmastervars2-1] = swap;
                  --nmastervars2;
                  --i;
               }
            }
            else if( nmastervars2 > 0 )
            {
               swap = mastervars2[i];
               mastervars2[i] = mastervars2[nmastervars2-1];
               mastervars2[nmastervars2-1] = swap;
               --nmastervars2;
               --i;
            }
         }

         if( p == Ssize-1 ) /*lint !e866 !e850*/
         {
            L = SCIPceil(scip, mu);
            SCIPdebugMessage("mu = %g, \n", mu);
            assert(!SCIPisFeasIntegral(scip,mu));
         }
         else
         {
            L = mu;
            SCIPdebugMessage("mu = %g should be integer, \n", mu);
            assert(SCIPisFeasIntegral(scip,mu));
         }
         lhs = pL-L+1;
      }
      SCIPdebugMessage("pL = %g \n", pL);
      pL = L;

      branchchilddata->lhs = lhs;
      SCIPdebugMessage("L = %g, \n", L);
      SCIPdebugMessage("lhs set to %g \n", lhs);
      assert(SCIPisFeasIntegral(scip, lhs));
      lhsSum += lhs;


      if( masterbranchcons == NULL || !pruneChildNodeByDominanceGeneric(scip, lhs, branchchilddata->consS, branchchilddata->consSsize, masterbranchcons, blocknr) )
      {
         if( masterbranchcons != NULL )
         {
            char childname[SCIP_MAXSTRLEN];
            ++nchildnodes;

            assert(branchchilddata != NULL);

            /* define names for origbranch constraints */
            (void) SCIPsnprintf(childname, SCIP_MAXSTRLEN, "node(%d, %d) (last comp=%s %s %g) >= %g", blocknr, p+1,
               SCIPvarGetName(branchchilddata->consS[branchchilddata->consSsize-1].component),
               branchchilddata->consS[branchchilddata->consSsize-1].sense == GCG_COMPSENSE_GE? ">=": "<",
               branchchilddata->consS[branchchilddata->consSsize-1].bound,
               branchchilddata->lhs);

            SCIP_CALL( SCIPcreateChild(masterscip, &child, 0.0, SCIPgetLocalTransEstimate(masterscip)) );
            SCIP_CALL( GCGcreateConsMasterbranch(masterscip, &childcons, childname, child,
               GCGconsMasterbranchGetActiveCons(masterscip), branchrule, branchchilddata, NULL, 0, 0) );
            SCIP_CALL( SCIPaddConsNode(masterscip, child, childcons, NULL) );

            SCIP_CALL( createBranchingCons(masterscip, child, branchchilddata) );

            /*  release constraints */
            SCIP_CALL( SCIPreleaseCons(masterscip, &childcons) );
         }
      }
      else
      {
         SCIPfreeBlockMemoryArrayNull(scip, &(branchchilddata->consS), branchchilddata->maxconsS);
         SCIPfreeBlockMemoryNull(scip, &branchchilddata);
      }
   }
   SCIPdebugMessage("lhsSum = %g\n", lhsSum);

#ifdef SCIP_DEBUG
   SCIP_CALL( SCIPgetVarsData(masterscip, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );

  for( i = 0; i < nmastervars; ++i )
  {
     SCIP_VAR* mastervar = mastervars[i];

     if( GCGisMasterVarInBlock(mastervar, blocknr) )
     {
        identicalcontrol += SCIPgetSolVal(masterscip, NULL, mastervar);
     }

  }
  if( !SCIPisEQ(scip, identicalcontrol, GCGgetNIdenticalBlocks(scip, blocknr)) )
  {
     SCIPdebugMessage("width of the block is only %g\n", identicalcontrol);
  }

  assert( SCIPisEQ(scip, identicalcontrol, GCGgetNIdenticalBlocks(scip, blocknr)) );
#endif

   assert( SCIPisEQ(scip, lhsSum, 1.0*(GCGgetNIdenticalBlocks(scip, blocknr) + Ssize)) );

   SCIPfreeBufferArray(scip, &mastervars2);

   if( nchildnodes <= 0 )
   {
      SCIPdebugMessage("node cut off, since all childnodes have been pruned\n");

      *result = SCIP_CUTOFF;
   }

   return SCIP_OKAY;
}

/** branching on copied origvar directly in master */
static
SCIP_RETCODE branchDirectlyOnMastervar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             mastervar,          /**< master variable */
   SCIP_BRANCHRULE*      branchrule          /**< branching rule */
   )
{
   SCIP* masterscip;
   GCG_BRANCHDATA* branchupchilddata;
   GCG_BRANCHDATA* branchdownchilddata;
   SCIP_NODE* upchild;
   SCIP_NODE* downchild;
   SCIP_CONS* upchildcons;
   SCIP_CONS* downchildcons;
   char upchildname[SCIP_MAXSTRLEN];
   char downchildname[SCIP_MAXSTRLEN];
   int bound;

   masterscip = GCGgetMasterprob(scip);
   assert(masterscip != NULL);

   bound = (int) (SCIPceil( scip, SCIPgetSolVal(masterscip, NULL, mastervar)) + 0.5); /*lint -e524*/

   /*  allocate branchdata for child and store information */
   SCIP_CALL( initNodeBranchdata(scip, &branchupchilddata, -3) );
   SCIP_CALL( initNodeBranchdata(scip, &branchdownchilddata, -3) );

   branchupchilddata->maxconsS = SCIPcalcMemGrowSize(scip, 1);
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(branchupchilddata->consS), branchupchilddata->maxconsS) ); /*lint !e506*/
   branchupchilddata->consSsize = 1;

   branchdownchilddata->maxconsS = branchupchilddata->maxconsS;
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(branchdownchilddata->consS), branchdownchilddata->maxconsS) ); /*lint !e506*/
      branchdownchilddata->consSsize = 1;

   branchupchilddata->consS[0].component = mastervar;
   branchupchilddata->consS[0].sense = GCG_COMPSENSE_GE;
   branchupchilddata->consS[0].bound = bound;

   branchdownchilddata->consS[0].component = mastervar;
   branchdownchilddata->consS[0].sense = GCG_COMPSENSE_LT;
   branchdownchilddata->consS[0].bound = bound;


   assert(branchupchilddata != NULL);
   assert(branchdownchilddata != NULL);

   (void) SCIPsnprintf(upchildname, SCIP_MAXSTRLEN, "node(-3, %f) direct up on comp=%s", branchupchilddata->consS[0].bound,
               SCIPvarGetName(branchupchilddata->consS[branchupchilddata->consSsize-1].component));
   (void) SCIPsnprintf(downchildname, SCIP_MAXSTRLEN, "node(-3, %f) direct up on comp=%s", branchdownchilddata->consS[0].bound,
               SCIPvarGetName(branchdownchilddata->consS[branchdownchilddata->consSsize-1].component));

   SCIP_CALL( SCIPcreateChild(masterscip, &upchild, 0.0, SCIPgetLocalTransEstimate(masterscip)) );
   SCIP_CALL( GCGcreateConsMasterbranch(masterscip, &upchildcons, upchildname, upchild,
      GCGconsMasterbranchGetActiveCons(masterscip), branchrule, branchupchilddata, NULL, 0, 0) );
   SCIP_CALL( SCIPaddConsNode(masterscip, upchild, upchildcons, NULL) );

   SCIP_CALL( SCIPcreateChild(masterscip, &downchild, 0.0, SCIPgetLocalTransEstimate(masterscip)) );
   SCIP_CALL( GCGcreateConsMasterbranch(masterscip, &downchildcons, downchildname, downchild,
      GCGconsMasterbranchGetActiveCons(masterscip), branchrule, branchdownchilddata, NULL, 0, 0) );
   SCIP_CALL( SCIPaddConsNode(masterscip, downchild, downchildcons, NULL) );

   /*  create branching constraint in master */
   SCIP_CALL( createDirectBranchingCons(masterscip, upchild, branchupchilddata) );
   SCIP_CALL( createDirectBranchingCons(masterscip, downchild, branchdownchilddata) );

   /*  release constraints */
   SCIP_CALL( SCIPreleaseCons(masterscip, &upchildcons) );
   SCIP_CALL( SCIPreleaseCons(masterscip, &downchildcons) );

   return SCIP_OKAY;
}

/** prepares information for using the generic branching scheme */
SCIP_RETCODE GCGbranchGenericInitbranch(
   SCIP*                 masterscip,         /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_RESULT*          result,             /**< pointer to store the result of the branching call */
   int**                 checkedblocks,      /**< blocks that have been checked */
   int*                  ncheckedblocks,     /**< number of checked blocks */
   GCG_STRIP****         checkedblockssortstrips, /**< sorted strips of checked blocks */
   int**                 checkedblocksnsortstrips /**< sizes of the strips */
   )
{
   SCIP* origscip;
   SCIP_VAR** branchcands;
   SCIP_VAR** allorigvars;
   SCIP_VAR** mastervars;
   int nmastervars;
   SCIP_CONS* masterbranchcons;
   int nbranchcands;
   GCG_BRANCHDATA* branchdata;
   SCIP_VAR* mastervar;
   SCIP_Real mastervarValue;
   GCG_COMPSEQUENCE* S;
   GCG_COMPSEQUENCE** C;
   SCIP_VAR** F;
   int Ssize;
   int Fsize;
   int* sequencesizes;
   int blocknr;
   int i;
   int j;
   int allnorigvars;
#ifndef NDEBUG
   SCIP_Bool foundblocknr = FALSE;
#endif

   SCIP_Bool discretization;

   blocknr = -2;
   Ssize = 0;
   Fsize = 0;
   S = NULL;
   C = NULL;
   F = NULL;
   sequencesizes = NULL;

   assert(masterscip != NULL);

   SCIPdebugMessage("get information for Vanderbecks generic branching\n");

   origscip = GCGmasterGetOrigprob(masterscip);

   SCIP_CALL( SCIPgetBoolParam(origscip, "relaxing/gcg/discretization", &discretization) );

   assert(origscip != NULL);
   SCIP_CALL( SCIPgetLPBranchCands(masterscip, &branchcands, NULL, NULL, &nbranchcands, NULL, NULL) );

   SCIP_CALL( SCIPgetVarsData(origscip, &allorigvars, &allnorigvars, NULL, NULL, NULL, NULL) );
   SCIP_CALL( SCIPgetVarsData(masterscip, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );

   /* in case original problem contains continuous variables, there are no branching cands */
   assert(nbranchcands > 0 || (discretization && SCIPgetNContVars(origscip) > 0));
   mastervar = NULL;

   /* loop over all branching candidates */
   for( i = 0; i < nbranchcands && (!discretization || SCIPgetNContVars(origscip) == 0); ++i )
   {
      int k;
      mastervar = branchcands[i];
      assert(GCGvarIsMaster(mastervar));

      /* if we have a master variable, we branch on it */
      if( GCGvarGetBlock(mastervar) == -1 )
      {
         assert(!GCGmasterVarIsArtificial(mastervar));
#ifndef NDEBUG
         foundblocknr = TRUE;
#endif
         blocknr = -1;
         break;
      }

      /* else, check if the candidate is in an unchecked block */
      for( j = 0; j < GCGgetNPricingprobs(origscip); ++j )
      {
         SCIP_Bool checked = FALSE;
         for( k = 0; ncheckedblocks != NULL && k < (*ncheckedblocks); ++k )
         {
            /* if the block has been checked, no need to check master variable */
            if( (*checkedblocks)[k] == j )
            {
               checked = TRUE;
               break;
            }
         }

         if( checked )
            continue;
         /* else the block has not been checked and the variable is in it , we have a candidate */
         else if( GCGisMasterVarInBlock(mastervar, j) )
         {
#ifndef NDEBUG
            foundblocknr = TRUE;
#endif
            blocknr = j;
            break;
         }
      }
   }
   assert(foundblocknr || blocknr == -1  || (discretization && SCIPgetNContVars(origscip) > 0));
   assert(i <= nbranchcands); /* else all blocks has been checked and we can observe an integer solution */

   /* in case of continuous origvar look for "fractional" blocks using the representation (currentorigsol) in the original problem */
   if(discretization && SCIPgetNContVars(origscip) > 0)
   {
      int norigvars;
      SCIP_VAR** origvars;
      SCIP_VAR* origvar;

      norigvars = SCIPgetNVars(origscip);
      origvars = SCIPgetVars(origscip);

      nbranchcands = SCIPgetNVars(masterscip);
      branchcands = SCIPgetVars(masterscip);

      assert(nbranchcands > 0);

      for( i = 0; i < norigvars; ++i )
      {
         int k;
         SCIP_Bool checked;
         origvar = origvars[i];

         if( SCIPvarGetType(origvar) > SCIP_VARTYPE_INTEGER )
            continue;

         if( SCIPisIntegral(origscip, SCIPgetSolVal(origscip, GCGrelaxGetCurrentOrigSol(origscip), origvar)) )
            continue;

         blocknr = GCGgetBlockRepresentative(origscip, GCGvarGetBlock(origvar));

         SCIPdebugMessage("Variable %s belonging to block %d with representative %d is not integral!\n", SCIPvarGetName(origvar), GCGvarGetBlock(origvar), blocknr);

         if( blocknr == -1 )
         {
            assert(GCGoriginalVarGetNMastervars(origvar) == 1);
            mastervar = GCGoriginalVarGetMastervars(origvar)[0];
            break;
         }

         checked = FALSE;
         for( k = 0; ncheckedblocks != NULL && k < (*ncheckedblocks); ++k )
         {
            /* if the block has been checked, no need to check master variable */
            if( (*checkedblocks)[k] == blocknr )
            {
               checked = TRUE;
               break;
            }
         }

         if( checked )
         {
            continue;
         }
         else
         {
            break;
         }
      }
   }

   if( blocknr < -1 )
   {
      SCIP_Bool rays;

      SCIPdebugMessage("Generic branching rule could not find variables to branch on!\n");

      SCIP_CALL( GCGpricerExistRays(masterscip, &rays) );

      if( rays )
         SCIPwarningMessage(masterscip, "Generic branching is not compatible with unbounded problems!\n");

      return SCIP_ERROR;
   }

   /* a special case; branch on copy of an origvar directly */
   if( blocknr == -1 )
   {
      assert(!GCGmasterVarIsLinking(mastervar));
      SCIPdebugMessage("branching on master variable\n");
      SCIP_CALL( branchDirectlyOnMastervar(origscip, mastervar, branchrule) );
      return SCIP_OKAY;
   }

   masterbranchcons = GCGconsMasterbranchGetActiveCons(masterscip);
   SCIPdebugMessage("branching in block %d \n", blocknr);

   /* calculate F and the strips */
   for( i = 0; i < nbranchcands; ++i )
   {
      mastervar = branchcands[i];
      assert(GCGvarIsMaster(mastervar));

      if( GCGisMasterVarInBlock(mastervar, blocknr) )
      {
         mastervarValue = SCIPgetSolVal(masterscip, NULL, mastervar);
         if( !SCIPisFeasIntegral(masterscip, mastervarValue) )
         {

            SCIP_CALL( SCIPreallocBufferArray(origscip, &F, (size_t)Fsize+1) );

            F[Fsize] = mastervar;
            ++Fsize;
         }
      }
   }

   /* old data to regard? */
   if( masterbranchcons != NULL && GCGconsMasterbranchGetBranchdata(masterbranchcons) != NULL )
   {
      /* calculate C */
      int c;
      int Csize = 0;
      SCIP_CONS* parentcons = masterbranchcons;

      while( parentcons != NULL && GCGconsMasterbranchGetBranchrule(parentcons) != NULL
         && strcmp(SCIPbranchruleGetName(GCGconsMasterbranchGetBranchrule(parentcons)), "generic") == 0)
      {
         branchdata = GCGconsMasterbranchGetBranchdata(parentcons);
         if( branchdata == NULL )
         {
            SCIPdebugMessage("branchdata is NULL\n");
            break;
         }
         if( branchdata->consS == NULL || branchdata->consSsize == 0 )
            break;
         if( branchdata->consblocknr != blocknr )
         {
            parentcons = GCGconsMasterbranchGetParentcons(parentcons);
            continue;
         }

         if( Csize == 0 )
         {
            assert(branchdata != NULL);
            assert(branchdata->consSsize > 0);
            Csize = 1;
            SCIP_CALL( SCIPallocBufferArray(origscip, &C, Csize) );
            SCIP_CALL( SCIPallocBufferArray(origscip, &sequencesizes, Csize) );
            assert(sequencesizes != NULL);
            C[0] = NULL;
            SCIP_CALL( SCIPallocBufferArray(origscip, &(C[0]), branchdata->consSsize) );
            for( i = 0; i < branchdata->consSsize; ++i )
            {
               C[0][i] = branchdata->consS[i];
            }
            sequencesizes[0] = branchdata->consSsize;
            parentcons = GCGconsMasterbranchGetParentcons(parentcons);
         }
         else
         {
            /* S not yet in C ? */
            SCIP_Bool SinC = FALSE;
            for( c = 0; c < Csize && !SinC; ++c )
            {
               SinC = TRUE;
               assert(sequencesizes != NULL);
               if( branchdata->consSsize == sequencesizes[c] )
               {
                  for( i = 0; i < branchdata->consSsize; ++i )
                  {
                     assert(C != NULL);
                     if( branchdata->consS[i].component != C[c][i].component || branchdata->consS[i].sense != C[c][i].sense || !SCIPisEQ(origscip, branchdata->consS[i].bound, C[c][i].bound) )
                     {
                        SinC = FALSE;
                        break;
                     }
                  }
               }
               else
                  SinC = FALSE;
            }
            if( !SinC )
            {
               ++Csize;
               SCIP_CALL( SCIPreallocBufferArray(origscip, &C, Csize) );
               SCIP_CALL( SCIPreallocBufferArray(origscip, &sequencesizes, Csize) );
               assert(sequencesizes != NULL);
               C[Csize-1] = NULL;
               SCIP_CALL( SCIPallocBufferArray(origscip, &(C[Csize-1]), branchdata->consSsize) ); /*lint !e866*/

               /** @todo copy memory */
               for( i = 0; i < branchdata->consSsize; ++i )
               {
                  C[Csize-1][i] = branchdata->consS[i];
               }
               sequencesizes[Csize-1] = branchdata->consSsize;
            }
            parentcons = GCGconsMasterbranchGetParentcons(parentcons);
         }
      }

      if( C != NULL )
      {
         SCIPdebugMessage("Csize = %d\n", Csize);

         for( i = 0; i < Csize; ++i )
         {
            assert(sequencesizes != NULL);

            for( c = 0; c < sequencesizes[i]; ++c )
            {
               SCIPdebugMessage("C[%d][%d].component = %s\n", i, c, SCIPvarGetName(C[i][c].component) );
               SCIPdebugMessage("C[%d][%d].sense = %d\n", i, c, C[i][c].sense);
               SCIPdebugMessage("C[%d][%d].bound = %.6g\n", i, c, C[i][c].bound);
            }
         }
         /* SCIP_CALL( InducedLexicographicSort(scip, F, Fsize, C, Csize, sequencesizes) ); */
         SCIP_CALL( ChooseSeparateMethod(origscip, F, Fsize, &S, &Ssize, C, Csize, sequencesizes, blocknr, branchrule, result, checkedblocks,
            ncheckedblocks, checkedblockssortstrips, checkedblocksnsortstrips) );
      }
      else
      {
         SCIPdebugMessage("C == NULL\n");
         /* SCIP_CALL( InducedLexicographicSort( scip, F, Fsize, NULL, 0, NULL ) ); */
         SCIP_CALL( ChooseSeparateMethod( origscip, F, Fsize, &S, &Ssize, NULL, 0, NULL, blocknr, branchrule, result,
               checkedblocks, ncheckedblocks, checkedblockssortstrips, checkedblocksnsortstrips) );
      }
      if( sequencesizes != NULL )
      {
         assert(Csize > 0);
         SCIPfreeBufferArray(origscip, &sequencesizes);
      }
      for( i = Csize - 1; i >= 0; --i )
      {
         assert(C != NULL );
         SCIPfreeBufferArrayNull(origscip, &(C[i]));
      }
      if( C != NULL )
      {
         assert( Csize > 0);
         SCIPfreeBufferArrayNull(origscip, &C);
      }
   }
   else
   {
      SCIPdebugMessage("root node\n");
      /* SCIP_CALL( InducedLexicographicSort( scip, F, Fsize, NULL, 0, NULL ) ); */
      SCIP_CALL( ChooseSeparateMethod( origscip, F, Fsize, &S, &Ssize, NULL, 0, NULL, blocknr, branchrule, result, checkedblocks,
         ncheckedblocks, checkedblockssortstrips, checkedblocksnsortstrips) );
   }

   /* create the |S|+1 child nodes in the branch-and-bound tree */
   if( S != NULL && Ssize > 0 )
   {
      SCIP_CALL( createChildNodesGeneric(origscip, branchrule, S, Ssize, blocknr, masterbranchcons, result) );
   }

   SCIPfreeBufferArrayNull(origscip, &S);
   SCIPdebugMessage("free F\n");
   SCIPfreeBufferArray(origscip, &F);

   return SCIP_OKAY;
}

/* from branch_master */
static
SCIP_RETCODE GCGincludeMasterCopyPlugins(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( SCIPincludeNodeselBfs(scip) );
   SCIP_CALL( SCIPincludeNodeselDfs(scip) );
   SCIP_CALL( SCIPincludeNodeselEstimate(scip) );
   SCIP_CALL( SCIPincludeNodeselHybridestim(scip) );
   SCIP_CALL( SCIPincludeNodeselRestartdfs(scip) );
   SCIP_CALL( SCIPincludeBranchruleAllfullstrong(scip) );
   SCIP_CALL( SCIPincludeBranchruleFullstrong(scip) );
   SCIP_CALL( SCIPincludeBranchruleInference(scip) );
   SCIP_CALL( SCIPincludeBranchruleMostinf(scip) );
   SCIP_CALL( SCIPincludeBranchruleLeastinf(scip) );
   SCIP_CALL( SCIPincludeBranchrulePscost(scip) );
   SCIP_CALL( SCIPincludeBranchruleRandom(scip) );
   SCIP_CALL( SCIPincludeBranchruleRelpscost(scip) );
   return SCIP_OKAY;
}
/** copy method for master branching rule */
static
SCIP_DECL_BRANCHCOPY(branchCopyGeneric)
{
   assert(branchrule != NULL);
   assert(scip != NULL);
   SCIP_CALL( GCGincludeMasterCopyPlugins(scip) );
   return SCIP_OKAY;
}

/** callback activation method */
static
GCG_DECL_BRANCHACTIVEMASTER(branchActiveMasterGeneric)
{

   SCIP_VAR** mastervars;
   int nmastervars;
   int i;
   int nvarsadded;

   assert(scip != NULL);
   assert(GCGisMaster(scip));
   assert(branchdata != NULL);
   assert(branchdata->mastercons != NULL);

   SCIPdebugMessage("branchActiveMasterGeneric: Block %d, Ssize %d\n", branchdata->consblocknr, branchdata->consSsize);

   if( branchdata->nvars >= SCIPgetNVars(scip) )
      return SCIP_OKAY;

   nmastervars = SCIPgetNVars(scip);
   mastervars = SCIPgetVars(scip);
   nvarsadded = 0;

   for( i = branchdata->nvars; i < nmastervars; ++i )
   {
      SCIP_Bool added = FALSE;
      assert(mastervars[i] != NULL);
      assert(GCGvarIsMaster(mastervars[i]));

      SCIP_CALL( addVarToMasterbranch(scip, mastervars[i], branchdata, &added) );
      if( added )
         ++nvarsadded;

   }
   SCIPdebugMessage("%d/%d vars added with lhs=%g\n", nvarsadded, nmastervars-branchdata->nvars, branchdata->lhs);
   branchdata->nvars = nmastervars;

   return SCIP_OKAY;
}

/** callback deactivation method */
static
GCG_DECL_BRANCHDEACTIVEMASTER(branchDeactiveMasterGeneric)
{
   assert(scip != NULL);
   assert(GCGisMaster(scip));
   assert(branchdata != NULL);
   assert(branchdata->mastercons != NULL);

   SCIPdebugMessage("branchDeactiveMasterGeneric: Block %d, Ssize %d\n", branchdata->consblocknr, branchdata->consSsize);

   /* set number of variables since last call */
   branchdata->nvars = SCIPgetNVars(scip);
   return SCIP_OKAY;
}



/** callback propagation method */
static
GCG_DECL_BRANCHPROPMASTER(branchPropMasterGeneric)
{
   assert(scip != NULL);
   assert(branchdata != NULL);
   assert(branchdata->mastercons != NULL);
   assert(branchdata->consS != NULL);

   /* SCIPdebugMessage("branchPropMasterGeneric: Block %d ,Ssize %d)\n", branchdata->consblocknr, branchdata->consSsize); */
   *result = SCIP_DIDNOTFIND;
   return SCIP_OKAY;
}

/** branching execution method for fractional LP solutions */
static
SCIP_DECL_BRANCHEXECLP(branchExeclpGeneric)
{  /*lint --e{715}*/
   SCIP* origscip;
   SCIP_Bool discretization;

   int* checkedblocks;
   int ncheckedblocks;
   GCG_STRIP*** checkedblockssortstrips;
   int* checkedblocksnsortstrips;

   assert(branchrule != NULL);
   assert(strcmp(SCIPbranchruleGetName(branchrule), BRANCHRULE_NAME) == 0);
   assert(scip != NULL);
   assert(result != NULL);

   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   SCIPdebugMessage("Execrel method of Vanderbecks generic branching\n");

   *result = SCIP_DIDNOTRUN;

   /* the branching scheme only works for the discretization approach */
   SCIP_CALL( SCIPgetBoolParam(origscip, "relaxing/gcg/discretization", &discretization) );
   if( !discretization )
   {
      SCIPdebugMessage("Generic branching only for discretization approach\n");
      return SCIP_OKAY;
   }

   if( GCGisMasterSetCovering(origscip) || GCGisMasterSetPartitioning(origscip) )
   {
      SCIPdebugMessage("Generic branching executed on a set covering or set partitioning problem\n");
   }

   if( GCGrelaxIsOrigSolFeasible(origscip) )
   {
      SCIPdebugMessage("node cut off, since origsol was feasible, solval = %f\n",
         SCIPgetSolOrigObj(origscip, GCGrelaxGetCurrentOrigSol(origscip)));

      *result = SCIP_DIDNOTFIND;
      return SCIP_OKAY;
   }

   *result = SCIP_BRANCHED;

   checkedblocks = NULL;
   ncheckedblocks = 0;
   checkedblockssortstrips = NULL;
   checkedblocksnsortstrips = NULL;

   SCIP_CALL( GCGbranchGenericInitbranch(scip, branchrule, result, &checkedblocks, &ncheckedblocks, &checkedblockssortstrips, &checkedblocksnsortstrips) );

   return SCIP_OKAY;
}

/** branching execution method for relaxation solutions */
static
SCIP_DECL_BRANCHEXECEXT(branchExecextGeneric)
{  /*lint --e{715}*/
   SCIPdebugMessage("Execext method of generic branching\n");

   *result = SCIP_DIDNOTRUN;

   return SCIP_OKAY;
}

/** branching execution method for not completely fixed pseudo solutions
 * @bug this needs to be implemented: #32
 */

static
SCIP_DECL_BRANCHEXECPS(branchExecpsGeneric)
{  /*lint --e{715}*/
   assert(branchrule != NULL);
   assert(strcmp(SCIPbranchruleGetName(branchrule), BRANCHRULE_NAME) == 0);
   assert(scip != NULL);
   assert(result != NULL);

   SCIPdebugMessage("Execps method of Vanderbecks generic branching\n");

   if( SCIPisStopped(scip) )
   {
      SCIPwarningMessage(scip, "No branching could be created, solving process cannot be restarted...\n" );

      *result = SCIP_DIDNOTRUN;
      return SCIP_OKAY;
   }
   else
   {
      SCIPerrorMessage("This method is not implemented, aborting since we cannot recover!\n");
      SCIPdialogMessage(scip, NULL, "Due to numerical issues, the problem could not be solved.\n");
      SCIPdialogMessage(scip, NULL, "You can try to disable discretization and aggregation and resolve the problem.\n");

      *result = SCIP_DIDNOTRUN;
      return SCIP_ERROR;
   }
}

/** initialization method of branching rule (called after problem was transformed) */
static
SCIP_DECL_BRANCHINIT(branchInitGeneric)
{
   SCIP* origscip;

   origscip = GCGmasterGetOrigprob(scip);
   assert(branchrule != NULL);
   assert(origscip != NULL);

   SCIPdebugMessage("Init method of Vanderbecks generic branching\n");

   SCIP_CALL( GCGrelaxIncludeBranchrule(origscip, branchrule, branchActiveMasterGeneric,
         branchDeactiveMasterGeneric, branchPropMasterGeneric, NULL, branchDataDeleteGeneric) );

   return SCIP_OKAY;
}

/** creates the generic branching rule and includes it in SCIP */
SCIP_RETCODE SCIPincludeBranchruleGeneric(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_BRANCHRULEDATA* branchruledata;
   SCIP_BRANCHRULE* branchrule;

   /* create branching rule data */
   branchruledata = NULL;

   SCIPdebugMessage("Include method of Vanderbecks generic branching\n");

   /* include branching rule */
   SCIP_CALL( SCIPincludeBranchrule(scip, BRANCHRULE_NAME, BRANCHRULE_DESC, BRANCHRULE_PRIORITY,
         BRANCHRULE_MAXDEPTH, BRANCHRULE_MAXBOUNDDIST, branchCopyGeneric,
         branchFreeGeneric, branchInitGeneric, branchExitGeneric, branchInitsolGeneric,
         branchExitsolGeneric, branchExeclpGeneric, branchExecextGeneric, branchExecpsGeneric,
         branchruledata) );

   /* include event handler for adding generated mastervars to the branching constraints */
   SCIP_CALL( SCIPincludeEventhdlr(scip, EVENTHDLR_NAME, EVENTHDLR_DESC,
         NULL, NULL, NULL, NULL, eventInitsolGenericbranchvaradd, eventExitsolGenericbranchvaradd,
         NULL, eventExecGenericbranchvaradd,
         NULL) );

   branchrule = SCIPfindBranchrule(scip, BRANCHRULE_NAME);
   assert(branchrule != NULL);

   SCIP_CALL( GCGconsIntegralorigAddBranchrule(scip, branchrule) );

   return SCIP_OKAY;
}

/** initializes branchdata */
SCIP_RETCODE GCGbranchGenericCreateBranchdata(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_BRANCHDATA**      branchdata          /**< branching data to initialize */
   )
{
   assert(scip != NULL);
   assert(branchdata != NULL);

   SCIP_CALL( SCIPallocBlockMemory(scip, branchdata) );
   (*branchdata)->consS = NULL;
   (*branchdata)->consSsize = 0;
   (*branchdata)->maxconsS = 0;
   (*branchdata)->C = NULL;
   (*branchdata)->mastercons = NULL;
   (*branchdata)->consblocknr = -2;

   return SCIP_OKAY;
}

/** get component bound sequence */
GCG_COMPSEQUENCE* GCGbranchGenericBranchdataGetConsS(
   GCG_BRANCHDATA*      branchdata          /**< branching data */
   )
{
   assert(branchdata != NULL);
   return branchdata->consS;
}

/** get size of component bound sequence */
int GCGbranchGenericBranchdataGetConsSsize(
   GCG_BRANCHDATA*      branchdata          /**< branching data */
   )
{
   assert(branchdata != NULL);
   return branchdata->consSsize;
}

/** get id of pricing problem (or block) to which the constraint belongs */
int GCGbranchGenericBranchdataGetConsblocknr(
   GCG_BRANCHDATA*      branchdata          /**< branching data */
   )
{
   assert(branchdata != NULL);
   return branchdata->consblocknr;
}

/** get master constraint */
SCIP_CONS* GCGbranchGenericBranchdataGetMastercons(
   GCG_BRANCHDATA*      branchdata          /**< branching data */
   )
{
   assert(branchdata != NULL);
   return branchdata->mastercons;
}

/** returns true when the branch rule is the generic branchrule */
SCIP_Bool GCGisBranchruleGeneric(
   SCIP_BRANCHRULE*      branchrule          /**< branchrule to check */
)
{
   return (branchrule != NULL) && (strcmp(BRANCHRULE_NAME, SCIPbranchruleGetName(branchrule)) == 0);
}
