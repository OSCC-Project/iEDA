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

/**@file   scipParaObjSelfSplitNodesel.cpp
 * @brief  node selector for self-split ramp-up
 * @author Yuji Shinano
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <cassert>
#include "scip/scip.h"
#include "ug_bb/bbParaComm.h"
#include "scipParaObjSelfSplitNodesel.h"

#if SCIP_APIVERSION >= 101

using namespace ParaSCIP;

/*
 * Callback methods of node selector
 */

/** node selection method of node selector */
SCIP_DECL_NODESELSELECT(ScipParaObjSelfSplitNodesel::scip_select)
{  /*lint --e{715}*/
   assert(nodesel != NULL);
   assert(strcmp(SCIPnodeselGetName(nodesel), "ScipParaObjSelfSplitNodeSel") == 0);
   assert(scip != NULL);
   assert(selnode != NULL);

   *selnode = NULL;

   if( sampling )
   {
      SCIP_NODE* node;
      node = SCIPgetBestNode(scip);
      if( node == NULL )
      {
         *selnode = NULL;
         return SCIP_OKAY;
      }
      /* if( SCIPnodeGetDepth(SCIPgetBestNode(scip)) > nodeseldata->depthlimit ) */
      if( SCIPnodeGetDepth(node) > depthlimit )
      {
         SCIP_NODE** leaves;
         SCIP_NODE** children;
         SCIP_NODE** siblings;
         SCIP_NODE** nodes;
         int* depths;
         int i;
         int offset;

         int nleaves;
         int nsiblings;
         int nchildren;
         int nnodes;

         /* collect leaves, children and siblings data */
         SCIP_CALL( SCIPgetOpenNodesData(scip, &leaves, &children, &siblings, &nleaves, &nchildren, &nsiblings) );
         nnodes = nleaves + nchildren + nsiblings;
         assert(nnodes == SCIPgetNNodesLeft(scip));

         // printf(">>>>>>>>>>>>>>Ending sampling phase: %d open nodes\n", nnodes);

         SCIP_CALL( SCIPallocMemoryArray(scip, &nodes, nnodes) );
         SCIP_CALL( SCIPallocMemoryArray(scip, &depths, nnodes) );

         for(i = 0; i < nchildren; i++)
         {
            nodes[i] = children[i];
            depths[i] = SCIPnodeGetDepth(children[i]);
         }

         for(i = nchildren; i < nchildren+nsiblings; i++)
         {
            nodes[i] = siblings[i-nchildren];
            depths[i] = SCIPnodeGetDepth(siblings[i-nchildren]);
         }

         offset = nchildren+nsiblings;
         for(i = offset; i < nnodes; i++)
         {
            nodes[i] = leaves[i-offset];
            depths[i] = SCIPnodeGetDepth(leaves[i-offset]);
         }

         SCIPsortIntPtr(depths, (void**)nodes, nnodes);

         for(i = 0; i < nnodes; i++)
         {
            if( (i % selfsplitsize) == selfsplitrank )
            {
               keepParaNode( scip, depths[i], nodes[i] );
            }
            SCIP_CALL( SCIPcutoffNode(scip,nodes[i]));
         }

         SCIP_CALL( SCIPpruneTree(scip) );
         assert( SCIPgetNNodesLeft(scip) == 0 );

         SCIPfreeMemoryArray(scip, &depths);
         SCIPfreeMemoryArray(scip, &nodes);

         sampling = false;

         return SCIP_OKAY;
      }
      else
      {
         /* siblings come before leaves at the same level. Sometimes it can occur that no leaves are left except for children */
         *selnode = SCIPgetBestSibling(scip);
         if( *selnode == NULL )
         {
            *selnode = SCIPgetBestLeaf(scip);
            if( *selnode == NULL )
               *selnode=SCIPgetBestChild(scip);
         }
         if( *selnode != NULL )
         {
            SCIPdebugMessage("Selecting next node number %" SCIP_LONGINT_FORMAT " at depth %d\n", SCIPnodeGetNumber(*selnode), SCIPnodeGetDepth(*selnode));
         }
         return SCIP_OKAY;
      }
   }

   if( *selnode == NULL )
   {
      *selnode = SCIPgetBestNode(scip);
   }
   // SCIP_CALL( SCIPselectNodeEstimate(scip, nodesel_estimate, selnode) );

   return SCIP_OKAY;
}


/** node comparison method of node selector */
SCIP_DECL_NODESELCOMP(ScipParaObjSelfSplitNodesel::scip_comp)
{  /*lint --e{715}*/

   /* @todo implement and call interface functions of preferred node selectors in SCIP */
   if( sampling )
   {
      int depth1;
      int depth2;

      depth1 = SCIPnodeGetDepth(node1);
      depth2 = SCIPnodeGetDepth(node2);

      /* if depths differ, prefer node with smaller depth */
      if( depth1 < depth2 )
         return -1;
      else if( depth1 > depth2 )
         return +1;
      else
      {
         /* depths are equal; prefer node with smaller number */
         SCIP_Longint number1;
         SCIP_Longint number2;

         number1 = SCIPnodeGetNumber(node1);
         number2 = SCIPnodeGetNumber(node2);
         assert(number1 != number2);

         if( number1 < number2 )
            return -1;
         else
            return +1;
      }
   }
   else
   {
      SCIP_Real estimate1;
      SCIP_Real estimate2;


      estimate1 = SCIPnodeGetEstimate(node1);
      estimate2 = SCIPnodeGetEstimate(node2);
      if( (SCIPisInfinity(scip,  estimate1) && SCIPisInfinity(scip,  estimate2)) ||
         (SCIPisInfinity(scip, -estimate1) && SCIPisInfinity(scip, -estimate2)) ||
         SCIPisEQ(scip, estimate1, estimate2) )
      {
         SCIP_Real lowerbound1;
         SCIP_Real lowerbound2;

         lowerbound1 = SCIPnodeGetLowerbound(node1);
         lowerbound2 = SCIPnodeGetLowerbound(node2);
         if( SCIPisLT(scip, lowerbound1, lowerbound2) )
            return -1;
         else if( SCIPisGT(scip, lowerbound1, lowerbound2) )
            return +1;
         else
         {
            SCIP_NODETYPE nodetype1;
            SCIP_NODETYPE nodetype2;

            nodetype1 = SCIPnodeGetType(node1);
            nodetype2 = SCIPnodeGetType(node2);
            if( nodetype1 == SCIP_NODETYPE_CHILD && nodetype2 != SCIP_NODETYPE_CHILD )
               return -1;
            else if( nodetype1 != SCIP_NODETYPE_CHILD && nodetype2 == SCIP_NODETYPE_CHILD )
               return +1;
            else if( nodetype1 == SCIP_NODETYPE_SIBLING && nodetype2 != SCIP_NODETYPE_SIBLING )
               return -1;
            else if( nodetype1 != SCIP_NODETYPE_SIBLING && nodetype2 == SCIP_NODETYPE_SIBLING )
               return +1;
            else
            {
               int depth1;
               int depth2;
               depth1 = SCIPnodeGetDepth(node1);
               depth2 = SCIPnodeGetDepth(node2);
               if( depth1 < depth2 )
                  return -1;
               else if( depth1 > depth2 )
                  return +1;
               else
                  return 0;
            }
         }
      }

      if( SCIPisLT(scip, estimate1, estimate2) )
         return -1;

      assert(SCIPisGT(scip, estimate1, estimate2));
      return +1;
   }
}

void
ScipParaObjSelfSplitNodesel::keepParaNode(
      SCIP *scip,
      int depth,
      SCIP_NODE *node
      )
{
   assert( depth == SCIPnodeGetDepth( node ) );
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

   int nVars = SCIPgetNVars(scip);
   // SCIP_VAR **vars = SCIPgetVars(scip);
   int *iBranchVars = new int[nBranchVars];
   // create the variable mapping hash map
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

   /*
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
      if( scipParaSolver->isCopyIncreasedVariables() &&
         scipParaSolver->getOriginalIndex(i) >= scipParaSolver->getNOrgVars() )
      {
         continue;
      }
      iBranchVar =  (int *)SCIPhashmapGetImage(varmapLb, vars[i]);
      if( iBranchVar )
      {
         if( EPSLT(preBranchBounds[*iBranchVar], SCIPvarGetLbGlobal(vars[i]) ,DEFAULT_NUM_EPSILON ) )
         {
            branchBounds[*iBranchVar] = SCIPvarGetLbGlobal(vars[i]);  // node is current node
            if ( EPSGT(branchBounds[*iBranchVar], SCIPvarGetUbGlobal(vars[i]), DEFAULT_NUM_EPSILON) ) abort();
         }
      }
      else
      {
         if( EPSGT( SCIPvarGetLbGlobal(vars[i]), scipParaSolver->getOrgVarLb(i), MINEPSILON ) )
         {
            branchVars[nBranchVars] = vars[i];
            branchBounds[nBranchVars] = SCIPvarGetLbGlobal(vars[i]);
            boundTypes[nBranchVars] = SCIP_BOUNDTYPE_LOWER;
            if ( EPSGT(branchBounds[nBranchVars], SCIPvarGetUbGlobal(vars[i]), DEFAULT_NUM_EPSILON) ) abort();
            if ( EPSGT(branchBounds[nBranchVars], scipParaSolver->getOrgVarUb(i), DEFAULT_NUM_EPSILON) ) abort();
            nBranchVars++;
            // std::cout << "********* GLOBAL is updated!*************" << std::endl;
         }
      }
      iBranchVar = (int *)SCIPhashmapGetImage(varmapUb, vars[i]);
      if( iBranchVar )
      {
         if( EPSGT(preBranchBounds[*iBranchVar], SCIPvarGetUbGlobal(vars[i]) ,DEFAULT_NUM_EPSILON ) )
         {
            branchBounds[*iBranchVar] = SCIPvarGetUbGlobal(vars[i]);
            if ( EPSLT(branchBounds[*iBranchVar], SCIPvarGetLbGlobal(vars[i]),DEFAULT_NUM_EPSILON) ) abort();
         }
      }
      else
      {
         if( EPSLT( SCIPvarGetUbGlobal(vars[i]), scipParaSolver->getOrgVarUb(i), MINEPSILON ) )
         {
            branchVars[nBranchVars] = vars[i];
            branchBounds[nBranchVars] = SCIPvarGetUbGlobal(vars[i]);
            boundTypes[nBranchVars] = SCIP_BOUNDTYPE_UPPER;
            if ( EPSLT(branchBounds[nBranchVars], SCIPvarGetLbGlobal(vars[i]),DEFAULT_NUM_EPSILON) ) abort();
            if ( EPSLT(branchBounds[nBranchVars], scipParaSolver->getOrgVarLb(i),DEFAULT_NUM_EPSILON) ) abort();
            nBranchVars++;
            // std::cout << "********* GLOBAL is updated!*************" << std::endl;
         }
      }
   }
   */
   SCIPhashmapFree(&varmapLb);
   SCIPhashmapFree(&varmapUb);
   /*
   delete [] preBranchVars;
   delete [] preBranchBounds;
   delete [] preBboundTypes;
   */
   delete [] iBranchVars;

   if( scipParaSolver->isCopyIncreasedVariables() ) // this may not need, but only for this error occurred so far.
   {
      if( !ifFeasibleInOriginalProblem(scip, nBranchVars, branchVars, branchBounds) )
      {
         delete [] branchVars;
         delete [] branchBounds;
         delete [] boundTypes;
         return;
      }
   }

   SCIP_CONS** addedcons = 0;
   int addedconsssize = SCIPnodeGetNAddedConss(node);
   int naddedconss = 0;
   if( addedconsssize > 0 )
   {
      SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &addedcons, addedconsssize) );
      SCIPnodeGetAddedConss(node, addedcons, &naddedconss, addedconsssize);
   }

   assert( scipParaSolver->getParentDiffSubproblem() == 0 );

   DEF_SCIP_PARA_COMM( scipParaComm, paraComm);
   ScipParaDiffSubproblem *diffSubproblem = scipParaComm->createScipParaDiffSubproblem(
         scip,
         scipParaSolver,
         nBranchVars,
         branchVars,
         branchBounds,
         boundTypes,
         naddedconss,
         addedcons
         );

   if( naddedconss  )
   {
      SCIPfreeBufferArray(scip, &addedcons);
   }


   long long n = SCIPnodeGetNumber( node );
   // Cutoff looks mess-up this assert(SCIPisFeasGE(scip, SCIPretransformObj(scip,SCIPnodeGetLowerbound(node)), SCIPgetDualbound(scip)));
   double dualBound = std::min(SCIPretransformObj(scip, SCIPnodeGetLowerbound( node )), SCIPgetDualbound(scip));
   // std::cout << "SCIPretransformObj(scip, SCIPnodeGetLowerbound( node )) = " <<  SCIPretransformObj(scip, SCIPnodeGetLowerbound( node )) << ", SCIPgetDualbound(scip) = " << SCIPgetDualbound(scip) << std::endl;
   // if( SCIPisObjIntegral(scip) )
   // {
   //    dualBound = ceil(dualBound);
   // }
   // assert(SCIPisFeasGE(scip, SCIPnodeGetLowerbound(node) , SCIPgetLowerbound(scip)));
   double estimateValue = SCIPnodeGetEstimate( node );
   // assert(SCIPisFeasGE(scip, estimateValue, SCIPnodeGetLowerbound(node) ));
#ifdef UG_DEBUG_SOLUTION
   SCIP_Bool valid = 0;
   SCIP_CALL_ABORT( SCIPdebugSolIsValidInSubtree(scip, &valid) );
   diffSubproblem->setOptimalSolIndicator(valid);
   std::cout << "* R." << scipParaSolver->getRank() << ", debug = " << SCIPdebugSolIsEnabled(scip) << ", valid = " << valid << std::endl;
#endif
   scipParaSolver->keepParaNode(n, depth, dualBound, estimateValue, diffSubproblem);

   /** remove the node sent from SCIP environment */
#ifdef UG_DEBUG_SOLUTION
   if( valid )
   {
      SCIPdebugSolDisable(scip);
      std::cout << "R." << paraComm->getRank() << ": disable debug, node which contains optmal solution is sent." << std::endl;
   }
#endif

   delete [] branchVars;
   delete [] branchBounds;
   delete [] boundTypes;

}

bool
ScipParaObjSelfSplitNodesel::ifFeasibleInOriginalProblem(
      SCIP *scip,
      int nBranchVars,
      SCIP_VAR **branchVars,
      SCIP_Real *inBranchBounds)
{

   bool feasible = true;
   SCIP_Real *branchBounds = new SCIP_Real[nBranchVars];
   for( int v = nBranchVars -1 ; v >= 0; --v )
   {
      SCIP_VAR *transformVar = branchVars[v];
      SCIP_Real scalar = 1.0;
      SCIP_Real constant = 0.0;
      SCIP_CALL_ABORT( SCIPvarGetOrigvarSum(&transformVar, &scalar, &constant ) );
      if( transformVar == NULL ) continue;
      // assert( scalar == 1.0 && constant == 0.0 );
      branchBounds[v] = ( inBranchBounds[v] - constant ) / scalar;
      if( SCIPvarGetType(transformVar) != SCIP_VARTYPE_CONTINUOUS
          && SCIPvarGetProbindex(transformVar) < scipParaSolver->getNOrgVars() )
      {
         if( !(SCIPisLE(scip,scipParaSolver->getOrgVarLb(SCIPvarGetProbindex(transformVar)), branchBounds[v]) &&
               SCIPisGE(scip,scipParaSolver->getOrgVarUb(SCIPvarGetProbindex(transformVar)), branchBounds[v])) )
         {
            feasible = false;
            break;
         }
      }
   }
   delete [] branchBounds;

   return feasible;
}

#endif

