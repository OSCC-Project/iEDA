/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*          This file is part of the program and software framework          */
/*                    UG --- Ubquity Generator Framework                     */
/*                                                                           */
/*  Copyright Written by Yuji Shinano <shinano@zib.de>,                      */
/*            Copyright (C) 2021 by Zuse Institute Berlin,                   */
/*            licensed under LGPL version 3 or later.                        */
/*            Commercial licenses are available through <licenses@zib.de>    */
/*                                                                           */
/* This code is free software; you can redistribute it and/or                */
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
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.     */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   objnodesel.cpp
 * @brief  C++ wrapper for node selectors
 * @author Yuji Shinano
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <cassert>
#include "scipParaObjNodesel.h"

using namespace ParaSCIP;

/** node selection method of node selector */
SCIP_DECL_NODESELSELECT(ScipParaObjNodesel::scip_select)
{  /*lint --e{715}*/
   // SCIP_NODESELDATA* nodeseldata;

   assert(nodesel != NULL);
   assert(strcmp(SCIPnodeselGetName(nodesel), "ScipParaObjNodesel") == 0);
   assert(scip != NULL);
   assert(selnode != NULL);

   *selnode = NULL;

   /* get node selector user data */
   // nodeseldata = SCIPnodeselGetData(nodesel);
   // assert(nodeseldata != NULL);

   SCIP_NODE* node;

   /* we want to plunge again: prefer children over siblings, and siblings over leaves,
    * but only select a child or sibling, if its dual bound is small enough;
    * prefer using nodes with higher node selection priority assigned by the branching rule
    */
   node = SCIPgetPrioChild(scip);
   if( node != NULL )
   {
      *selnode = node;
      SCIPdebugMessage("  -> selected prio child: lower=%g\n", SCIPnodeGetLowerbound(*selnode));
   }
   else
   {
      node = SCIPgetBestChild(scip);
      if( node != NULL )
      {
         *selnode = node;
         SCIPdebugMessage("  -> selected best child: lower=%g\n", SCIPnodeGetLowerbound(*selnode));
      }
      else
      {
         node = SCIPgetPrioSibling(scip);
         if( node != NULL )
         {
            *selnode = node;
            SCIPdebugMessage("  -> selected prio sibling: lower=%g\n", SCIPnodeGetLowerbound(*selnode));
         }
         else
         {
            node = SCIPgetBestSibling(scip);
            if( node != NULL )
            {
               *selnode = node;
               SCIPdebugMessage("  -> selected best sibling: lower=%g\n", SCIPnodeGetLowerbound(*selnode));
            }
            else
            {
               *selnode = SCIPgetBestNode(scip);
               SCIPdebugMessage("  -> selected best leaf: lower=%g\n",
                  *selnode != NULL ? SCIPnodeGetLowerbound(*selnode) : SCIPinfinity(scip));
            }
         }
      }
   }

   return SCIP_OKAY;
}


/** node comparison method of node selector */
SCIP_DECL_NODESELCOMP(ScipParaObjNodesel::scip_comp)
{

   SCIP_Real lowerbound1;
   SCIP_Real lowerbound2;

   assert(nodesel != NULL);
   assert(strcmp(SCIPnodeselGetName(nodesel), "ScipParaObjNodesel") == 0);
   assert(scip != NULL);

   lowerbound1 = SCIPnodeGetLowerbound(node1);
   lowerbound2 = SCIPnodeGetLowerbound(node2);
   if( SCIPisLT(scip, lowerbound1, lowerbound2) )
      return -1;
   else if( SCIPisGT(scip, lowerbound1, lowerbound2) )
      return +1;
   else
   {
      int nBranchVars1 = getNBoundChanges(node1);
      int nBranchVars2 = getNBoundChanges(node2);
      if( nBranchVars1 < nBranchVars2 )
         return -1;
      else if( nBranchVars1 > nBranchVars2 )
         return +1;
      else
      {
         /** worse one might be better to transfer */
         SCIP_Real estimate1;
         SCIP_Real estimate2;

         estimate1 = SCIPnodeGetEstimate(node1);
         estimate2 = SCIPnodeGetEstimate(node2);
         if( (SCIPisInfinity(scip,  estimate1) && SCIPisInfinity(scip,  estimate2)) ||
             (SCIPisInfinity(scip, -estimate1) && SCIPisInfinity(scip, -estimate2)) ||
             SCIPisEQ(scip, estimate1, estimate2) )
         {
            SCIP_NODETYPE nodetype1;
            SCIP_NODETYPE nodetype2;

            nodetype1 = SCIPnodeGetType(node1);
            nodetype2 = SCIPnodeGetType(node2);
            if( nodetype1 == SCIP_NODETYPE_CHILD && nodetype2 != SCIP_NODETYPE_CHILD )
               return +1;
            else if( nodetype1 != SCIP_NODETYPE_CHILD && nodetype2 == SCIP_NODETYPE_CHILD )
               return -1;
            else if( nodetype1 == SCIP_NODETYPE_SIBLING && nodetype2 != SCIP_NODETYPE_SIBLING )
               return +1;
            else if( nodetype1 != SCIP_NODETYPE_SIBLING && nodetype2 == SCIP_NODETYPE_SIBLING )
               return -1;
            else
            {
               int depth1;
               int depth2;

               depth1 = SCIPnodeGetDepth(node1);
               depth2 = SCIPnodeGetDepth(node2);
               if( depth1 < depth2 )
                  return +1;
               else if( depth1 > depth2 )
                  return -1;
               else
                  return 0;
            }
         }
         if( SCIPisLT(scip, estimate1, estimate2) )
            return +1;

         assert(SCIPisGT(scip, estimate1, estimate2));
         return -1;
      }
   }
}

int
ScipParaObjNodesel::getNBoundChanges(
      SCIP_NODE* node 
      )
{
   SCIP *scip = scipParaSolver->getScip();
   int depth = SCIPnodeGetDepth( node );
   SCIP_VAR **branchVars = new SCIP_VAR*[depth];
   SCIP_Real *branchBounds = new SCIP_Real[depth];
   SCIP_BOUNDTYPE *boundTypes = new  SCIP_BOUNDTYPE[depth];
   int nBranchVars;
   SCIPnodeGetAncestorBranchings( node, branchVars, branchBounds, boundTypes, &nBranchVars, depth );
   if( nBranchVars > depth )  // did not have enough memory, then reallocate
   {
      delete [] branchVars;
      delete [] branchBounds;
      delete [] boundTypes;
      branchVars = new SCIP_VAR*[nBranchVars];
      branchBounds = new SCIP_Real[nBranchVars];
      boundTypes = new  SCIP_BOUNDTYPE[nBranchVars];
      SCIPnodeGetAncestorBranchings( node, branchVars, branchBounds, boundTypes, &nBranchVars, nBranchVars );
   }

   if( scipParaSolver->getParaParamSet()->getBoolParamValue(UG::AllBoundChangesTransfer) &&
         !( scipParaSolver->getParaParamSet()->getBoolParamValue(UG::NoAllBoundChangesTransferInRacing) &&
               scipParaSolver->isRacingStage() ) )
   {
      int nVars = SCIPgetNVars(scip);
      SCIP_VAR **vars = SCIPgetVars(scip);
      int *iBranchVars = new int[nBranchVars];
      /* create the variable mapping hash map */
      SCIP_HASHMAP* varmapLb;
      SCIP_HASHMAP* varmapUb;
      SCIP_CALL_ABORT( SCIPhashmapCreate(&varmapLb, SCIPblkmem(scip), nVars) );
      SCIP_CALL_ABORT( SCIPhashmapCreate(&varmapUb, SCIPblkmem(scip), nVars) );
      for( int i = 0; i < nBranchVars; i++ )
      {
         iBranchVars[i] = i;
         if( boundTypes[i] == SCIP_BOUNDTYPE_LOWER )
         {
            if( !SCIPhashmapGetImage(varmapLb, branchVars[i]) )
            {
               SCIP_CALL_ABORT( SCIPhashmapInsert(varmapLb, branchVars[i], &iBranchVars[i] ) );
            }
         }
         else
         {
            if( !SCIPhashmapGetImage(varmapUb, branchVars[i]) )
            {
               SCIP_CALL_ABORT( SCIPhashmapInsert(varmapUb, branchVars[i], &iBranchVars[i] ) );
            }
         }
      }
      SCIP_VAR **preBranchVars = branchVars;
      SCIP_Real *preBranchBounds = branchBounds;
      SCIP_BOUNDTYPE *preBboundTypes = boundTypes;
      branchVars = new SCIP_VAR*[nBranchVars+nVars*2];
      branchBounds = new SCIP_Real[nBranchVars+nVars*2];
      boundTypes = new  SCIP_BOUNDTYPE[nBranchVars+nVars*2];
      for( int i = 0; i < nBranchVars; i++ )
      {
         branchVars[i] = preBranchVars[i];
         branchBounds[i] = preBranchBounds[i];
         boundTypes[i] = preBboundTypes[i];
      }
      int *iBranchVar = NULL;
      for( int i = 0; i < nVars; i++ )
      {
         iBranchVar =  (int *)SCIPhashmapGetImage(varmapLb, vars[i]);
         if( iBranchVar )
         {
            // assert( EPSGE(preBranchBounds[*iBranchVar], SCIPvarGetLbLocal(vars[i]) ,DEFAULT_NUM_EPSILON ) );
            if( EPSLT(preBranchBounds[*iBranchVar], SCIPvarGetLbLocal(vars[i]) ,DEFAULT_NUM_EPSILON ) )
            {
               branchBounds[*iBranchVar] = SCIPvarGetLbLocal(vars[i]);  // node is current node
               // if ( EPSGT(branchBounds[*iBranchVar], SCIPvarGetUbGlobal(vars[i]), DEFAULT_NUM_EPSILON) ) abort();
            }
         }
         else
         {
            if( EPSGT( SCIPvarGetLbLocal(vars[i]), SCIPvarGetLbGlobal(vars[i]), MINEPSILON ) )
            {
               branchVars[nBranchVars] = vars[i];
               branchBounds[nBranchVars] = SCIPvarGetLbLocal(vars[i]);
               boundTypes[nBranchVars] = SCIP_BOUNDTYPE_LOWER;
               // if ( EPSGT(branchBounds[nBranchVars], SCIPvarGetUbGlobal(vars[i]), DEFAULT_NUM_EPSILON) ) abort();
               nBranchVars++;
            }
         }
         iBranchVar = (int *)SCIPhashmapGetImage(varmapUb, vars[i]);
         if( iBranchVar )
         {
            // assert( EPSLE(preBranchBounds[*iBranchVar], SCIPvarGetUbLocal(vars[i]) ,DEFAULT_NUM_EPSILON ) );
            if( EPSGT(preBranchBounds[*iBranchVar], SCIPvarGetUbLocal(vars[i]) ,DEFAULT_NUM_EPSILON ) )
            {
               branchBounds[*iBranchVar] = SCIPvarGetUbLocal(vars[i]); // node is current node
               if ( EPSLT(branchBounds[*iBranchVar], SCIPvarGetLbGlobal(vars[i]),DEFAULT_NUM_EPSILON) ) abort();
            }
         }
         else
         {
            if( EPSLT( SCIPvarGetUbLocal(vars[i]), SCIPvarGetUbGlobal(vars[i]), MINEPSILON ) )
            {
               branchVars[nBranchVars] = vars[i];
               branchBounds[nBranchVars] = SCIPvarGetUbLocal(vars[i]);
               boundTypes[nBranchVars] = SCIP_BOUNDTYPE_UPPER;
               if ( EPSLT(branchBounds[nBranchVars], SCIPvarGetLbGlobal(vars[i]),DEFAULT_NUM_EPSILON) ) abort();
               nBranchVars++;
            }
         }
      }
      SCIPhashmapFree(&varmapLb);
      SCIPhashmapFree(&varmapUb);
      delete [] preBranchVars;
      delete [] preBranchBounds;
      delete [] preBboundTypes;
      delete [] iBranchVars;
   }

   /** check root node solvability */
   delete [] branchVars;
   delete [] branchBounds;
   delete [] boundTypes;

   return nBranchVars;;
}
