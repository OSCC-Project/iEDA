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

/**@file    bbParaNodesMerger.cpp
 * @brief   Nodes Merger.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "bbParaParamSet.h"
#include "bbParaNodesMerger.h"
#include "bbParaNode.h"
#include "bbParaNodePool.h"
#include "bbParaDiffSubproblem.h"

using namespace UG;

void
BbParaNodesMerger::addNodeToMergeNodeStructs(
      BbParaNode *node
      )
{
   // Mergeing nodes look better to be restricted. It has to be tested
   if( nBoundChangesOfBestNode < 0 )
   {
      nBoundChangesOfBestNode = dynamic_cast<BbParaDiffSubproblem *>(node->getDiffSubproblem())->getNBoundChanges();
   }
   if( nBoundChangesOfBestNode > 0 &&
         node->getDiffSubproblem() && dynamic_cast<BbParaDiffSubproblem *>(node->getDiffSubproblem())->getNBoundChanges() <= nBoundChangesOfBestNode )
   {
      /*
      if( logSolvingStatusFlag )
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " The node " << node->toSimpleString() << " is out of merge candidates."
         << std::endl;
      }
      */
      return;   // prohibit to generate the same merging node twice
   }
   /*
   if( node->getMergingStatus() == 1 )
   {
      if( logSolvingStatusFlag )
      {
         *osLogSolvingStatus << paraTimer->getElapsedTime()
         << " The node " << node->toSimpleString() << " is out of merge candidates."
         << std::endl;
      }
      return;   // prohibit to generate the same merging node twice
   }
   */

   double startTime = paraTimer->getElapsedTime();
   //
   // create mergeNodeInfo and linked to paraNode
   //
   BbParaMergeNodeInfo *mNode = new BbParaMergeNodeInfo();
   mNode->status = BbParaMergeNodeInfo::PARA_MERGING;
   mNode->nSameValueVariables = -1;
   mNode->nMergedNodes = -1;
   mNode->keyIndex = -1;
   mNode->nFixedVariables = 0;
   mNode->fixedVariables = 0;
   mNode->mergedTo = 0;
   mNode->paraNode = node;
   mNode->origDiffSubproblem = 0;
   mNode->mergedDiffSubproblem = 0;
   mNode->next = 0;
   node->setMergingStatus(0);      // checking
   node->setMergeNodeInfo(mNode);  // set merge node info.

   //
   // make the same value fixed variables links
   //
   /* get fixed variables array */
   if( node->getDiffSubproblem() )
   {
      mNode->nFixedVariables = dynamic_cast<BbParaDiffSubproblem *>(node->getDiffSubproblem())->getFixedVariables(
            instance,
            &(mNode->fixedVariables));
   }
   else
   {
      mNode->nFixedVariables = 0;
   }
   if( mNode->nFixedVariables == 0 )  // cannot merge!
   {
      delete mNode;
      dynamic_cast<BbParaNode *>(node)->setMergingStatus(3);     // cannot be merged
      dynamic_cast<BbParaNode *>(node)->setMergeNodeInfo(0);
      addingNodeToMergeStructTime += paraTimer->getElapsedTime() - startTime;
      return;
   }

   //
   // add mergeNode to mergeNodeInfo list
   //
   if( mergeInfoTail == 0 )
   {
      mergeInfoTail = mNode;
      mergeInfoHead = mNode;
   }
   else
   {
      mergeInfoTail->next = mNode;
      mergeInfoTail = mNode;
   }

   for( int i = 0; i < mNode->nFixedVariables; i++ )
   {
      mNode->fixedVariables[i].mnode = mNode;
      BbParaFixedValue *fixedValue = 0;
      if( varIndexTable[mNode->fixedVariables[i].index] == 0 )
      {
         fixedValue = new BbParaFixedValue();
         fixedValue->value = mNode->fixedVariables[i].value;
         fixedValue->head = 0;
         fixedValue->tail = 0;
         fixedValue->next = 0;
         varIndexTable[mNode->fixedVariables[i].index] = fixedValue;
      }
      else
      {
         BbParaFixedValue *prev = varIndexTable[mNode->fixedVariables[i].index];
         for( fixedValue = varIndexTable[mNode->fixedVariables[i].index];
               fixedValue != 0 &&  !EPSEQ( fixedValue->value, mNode->fixedVariables[i].value, DEFAULT_NUM_EPSILON );
               fixedValue = fixedValue->next )
         {
            prev = fixedValue;
         }
         if( fixedValue == 0 )
         {
            fixedValue = new BbParaFixedValue();
            fixedValue->value = mNode->fixedVariables[i].value;
            fixedValue->head = 0;
            fixedValue->tail = 0;
            fixedValue->next = 0;
            prev->next = fixedValue;
         }
      }
      assert( fixedValue );
      if( fixedValue->tail == 0 )
      {
         fixedValue->head = &(mNode->fixedVariables[i]);
         fixedValue->tail = &(mNode->fixedVariables[i]);
      }
      else
      {
         fixedValue->tail->next = &(mNode->fixedVariables[i]);
         fixedValue->tail->next->prev = fixedValue->tail;
         fixedValue->tail = &(mNode->fixedVariables[i]);
      }
      for( BbParaFixedVariable *p = fixedValue->head; p != fixedValue->tail; p = p->next )
      {
         (p->nSameValue)++;
      }
   }

   addingNodeToMergeStructTime += paraTimer->getElapsedTime() - startTime;

}

void
BbParaNodesMerger::generateMergeNodesCandidates(
      ParaComm          *paraComm,
      ParaInitiator     *paraInitiator
      )
{
   double startTime = paraTimer->getElapsedTime();

   BbParaMergeNodeInfo *mPre = mergeInfoHead;
   BbParaMergeNodeInfo *mNode = mergeInfoHead;
   mNode = mergeInfoHead;
   while( mNode )
   {
      assert( mNode->paraNode->getMergeNodeInfo() == mNode );
      if( mNode->status == BbParaMergeNodeInfo::PARA_MERGING && mNode->nMergedNodes < 0 )
      {
         // make sorted variables list
         std::multimap<int, BbParaSortedVariable, std::greater<int> > descendent;
         for( int i = 0; i < mNode->nFixedVariables; i++ )
         {
            BbParaSortedVariable sortedVar;
            sortedVar.idxInFixedVariabes = i;
            sortedVar.fixedVariable = &(mNode->fixedVariables[i]);
            descendent.insert(std::make_pair(mNode->fixedVariables[i].nSameValue, sortedVar));

         }
         //
         //  try to make merge candidates
         //
         std::multimap<int, BbParaSortedVariable, std::greater<int> >::iterator pos;
         pos = descendent.begin();
         mNode->keyIndex = pos->second.idxInFixedVariabes;
         mNode->nSameValueVariables = 1;
         BbParaFixedVariable *traverse = mNode->fixedVariables[mNode->keyIndex].next;
         int nmNodes = 0;
         for( ;
               traverse;
               traverse = traverse->next )
         {
            if( traverse->mnode->status == BbParaMergeNodeInfo::PARA_MERGING && traverse->mnode->nMergedNodes < 0 )
            {
               assert( traverse->mnode != mNode );
               traverse->mnode->mergedTo = mNode;
               traverse->mnode->nMergedNodes = 0;
               traverse->mnode->nSameValueVariables = 1;
               nmNodes++;
            }
         }
         ++pos;
         for( ; pos != descendent.end(); ++pos )
         {
            // check if there are merged nodes in case adding one more variable
            for( traverse = mNode->fixedVariables[pos->second.idxInFixedVariabes].next;
                  traverse;
                  traverse = traverse->next )
            {
               if( traverse->mnode->nMergedNodes == 0 && traverse->mnode->mergedTo == mNode )
               {
                  if( traverse->mnode->nSameValueVariables == mNode->nSameValueVariables )
                  {
                     break;   // at least one node can be merged
                  }
               }
            }
            if( traverse == 0 )  // cannot merge any nodes
            {
               break;
            }

            // merge nodes
            mNode->nSameValueVariables++;
            for( traverse = mNode->fixedVariables[pos->second.idxInFixedVariabes].next;
                  traverse;
                  traverse = traverse->next )
            {
               if( traverse->mnode->nMergedNodes == 0 && traverse->mnode->mergedTo == mNode )
               {
                  if( traverse->mnode->nSameValueVariables == (mNode->nSameValueVariables - 1) )
                  {
                     traverse->mnode->nSameValueVariables++;
                  }
               }
            }
         }

         // if the number of fixed variables is too small, then the merged node is not created
         if( nmNodes < 2 ||     // no merging nodes
               static_cast<int>((mNode->nFixedVariables)*paraParamSet->getRealParamValue(FixedVariablesRatioInMerging)) < 1 || //  0 same value variables are not allowed
               mNode->nSameValueVariables < (mNode->nFixedVariables)*paraParamSet->getRealParamValue(FixedVariablesRatioInMerging) ||
               ( nBoundChangesOfBestNode > 0 && mNode->nSameValueVariables <= nBoundChangesOfBestNode )
            )
         {
            for( BbParaFixedVariable *cleanup = mNode->fixedVariables[mNode->keyIndex].next;
                  cleanup;
                  cleanup = cleanup->next )
            {
               if( cleanup->mnode->mergedTo == mNode )
               {
                  cleanup->mnode->nSameValueVariables = -1;
                  cleanup->mnode->nMergedNodes = -1;
                  cleanup->mnode->keyIndex = -1;
                  cleanup->mnode->mergedTo = 0;
                  assert( cleanup->mnode->status == BbParaMergeNodeInfo::PARA_MERGING );
               }
            }
            assert( !(mNode->origDiffSubproblem) );
            assert( !(mNode->mergedDiffSubproblem) );
            mNode->paraNode->setMergeNodeInfo(0);
            mNode->paraNode->setMergingStatus(3);   // cannot merged
            BbParaMergeNodeInfo *doomed = mNode;
            if( mNode == mergeInfoHead )
            {
               mergeInfoHead = mNode->next;
               mPre = mergeInfoHead;
               mNode = mergeInfoHead;
            }
            else
            {
               mPre->next = mNode->next;
               mNode = mNode->next;
            }
            if( mNode == mergeInfoTail )
            {
               mergeInfoTail = mPre;
            }
            deleteMergeNodeInfo(doomed);
         }
         else  // cleanup and merge nodes
         {
            int nMergedNodes = 0;
            for( BbParaFixedVariable *cleanup = mNode->fixedVariables[mNode->keyIndex].next;
                  cleanup;
                  cleanup = cleanup->next )
            {
               if( cleanup->mnode->mergedTo == mNode )
               {
                  if( mNode->nSameValueVariables == cleanup->mnode->nSameValueVariables )
                  {
                     nMergedNodes++;
                     cleanup->mnode->status = BbParaMergeNodeInfo::PARA_MERGE_CHECKING_TO_OTHER_NODE;
                  }
                  else
                  {
                     assert( cleanup->mnode->status == BbParaMergeNodeInfo::PARA_MERGING );
                     cleanup->mnode->nSameValueVariables = -1;
                     cleanup->mnode->nMergedNodes = -1;
                     cleanup->mnode->keyIndex = -1;
                     cleanup->mnode->mergedTo = 0;
                  }
               }
            }
            mNode->nMergedNodes = nMergedNodes;
            assert(nMergedNodes > 0);
            int n = 0;
            BbParaFixedVariable *fixedVariables = new BbParaFixedVariable[mNode->nSameValueVariables];
            for( pos = descendent.begin(); pos != descendent.end(); ++pos )
            {
               fixedVariables[n] = *(pos->second.fixedVariable);
               n++;
               if( n == mNode->nSameValueVariables ) break;
            }
            mNode->origDiffSubproblem = mNode->paraNode->getDiffSubproblem();
            mNode->mergedDiffSubproblem = mNode->origDiffSubproblem->createDiffSubproblem(paraComm, paraInitiator, n, fixedVariables );
            delete [] fixedVariables;
            mNode->paraNode->setDiffSubproblem(mNode->mergedDiffSubproblem);
            mNode->status = BbParaMergeNodeInfo::PARA_MERGED_RPRESENTATIVE;
            assert( mNode->mergedTo == 0 );
            mPre = mNode;
            mNode = mNode->next;
         }
      }
      else
      {
         mPre = mNode;
         mNode = mNode->next;
      }
   }

   // remove data which are not used anymore.
   if( varIndexTable )
   {
      for( int i = 0; i < instance->getVarIndexRange(); i++ )
      {
         if( varIndexTable[i] )
         {
            while ( varIndexTable[i] )
            {
               BbParaFixedValue *del = varIndexTable[i];
               varIndexTable[i] = varIndexTable[i]->next;
               delete del;
            }
         }
      }
      delete [] varIndexTable;
      varIndexTable = 0;
   }

   generateMergeNodesCandidatesTime += paraTimer->getElapsedTime() - startTime;
}

void
BbParaNodesMerger::regenerateMergeNodesCandidates(
      BbParaNode    *node,
      ParaComm      *paraComm,
      ParaInitiator *paraInitiator
      )
{
   double startTime = paraTimer->getElapsedTime();

   BbParaMergeNodeInfo *mNode = dynamic_cast<BbParaNode *>(node)->getMergeNodeInfo();
   assert(mNode);
   assert(mNode->paraNode == node);
   assert(mNode->status == BbParaMergeNodeInfo::PARA_MERGED_RPRESENTATIVE);
   assert(mNode->mergedTo == 0);
   dynamic_cast<BbParaNode *>(node)->setMergeNodeInfo(0);
   dynamic_cast<BbParaNode *>(node)->resetDualBoundValue();
   node->setDiffSubproblem(mNode->origDiffSubproblem);
   delete mNode->mergedDiffSubproblem;
   mNode->mergedDiffSubproblem = 0;
   assert( mNode->status != BbParaMergeNodeInfo::PARA_MERGING );
   // set new range
   mergeInfoHead =  0;
   mergeInfoTail = 0;
   BbParaMergeNodeInfo *mPrev = 0;
   for( BbParaFixedVariable *traverse = mNode->fixedVariables[mNode->keyIndex].next;
         traverse;
         traverse = traverse->next )
   {
      if( mergeInfoTail )
      {
         mPrev->next = traverse->mnode;
         mergeInfoTail = traverse->mnode;
         mPrev = traverse->mnode;
      }
      if( mNode == traverse->mnode->mergedTo )
      {
         if( !mergeInfoHead )
         {
            mergeInfoHead = traverse->mnode;
            mergeInfoTail = traverse->mnode;
            mPrev = traverse->mnode;
         }
         assert( traverse->mnode->status == BbParaMergeNodeInfo::PARA_MERGE_CHECKING_TO_OTHER_NODE );
      }
   }
   if( mergeInfoHead )
   {
      assert(mergeInfoTail);
      mergeInfoTail->next = 0;
   }

   // remove mnode
   mNode->paraNode->setMergingStatus(-1);  // no merging node
   deleteMergeNodeInfo(mNode);
   if( mergeInfoHead )
   {
      generateMergeNodesCandidates(paraComm, paraInitiator);
   }

   regenerateMergeNodesCandidatesTime += paraTimer->getElapsedTime() - startTime;
}

void
BbParaNodesMerger::deleteMergeNodeInfo(
      BbParaMergeNodeInfo *mNode
      )
{

   if( mNode->nMergedNodes == 0 && mNode->mergedTo )
   {
      assert(mNode->status == BbParaMergeNodeInfo::PARA_MERGE_CHECKING_TO_OTHER_NODE);
      assert(mNode->mergedTo->status ==  BbParaMergeNodeInfo::PARA_MERGED_RPRESENTATIVE);
      assert(mNode->mergedTo->mergedTo == 0);
      mNode->mergedTo->nMergedNodes--;
      if( mNode->mergedTo->nMergedNodes == 0 && mNode->mergedTo->paraNode->getMergeNodeInfo() )
      {
         assert(mNode->mergedTo == mNode->mergedTo->paraNode->getMergeNodeInfo() );
         mNode->mergedTo->paraNode->setDiffSubproblem(mNode->mergedTo->origDiffSubproblem);
         mNode->mergedTo->paraNode->setMergeNodeInfo(0);
         mNode->mergedTo->paraNode->setMergingStatus(-1);
         delete mNode->mergedTo->mergedDiffSubproblem;
         mNode->mergedTo->mergedDiffSubproblem = 0;
         mNode->mergedTo->origDiffSubproblem = 0;
         deleteMergeNodeInfo(mNode->mergedTo);
      }
      mNode->mergedTo = 0;

   }

   if( mNode->status == BbParaMergeNodeInfo::PARA_MERGED_RPRESENTATIVE)
   {
      assert( mNode->mergedTo == 0 );
      if( mNode->paraNode->getMergingStatus() == -1 )  // merging failed
      {
         for( BbParaFixedVariable *traverse = mNode->fixedVariables[mNode->keyIndex].next;
               traverse;
               traverse = traverse->next )
         {
            if( traverse->mnode->nMergedNodes == 0 && mNode == traverse->mnode->mergedTo )
            {
               traverse->mnode->mergedTo->nMergedNodes--;
               if( traverse->mnode->mergedTo->nMergedNodes == 0 && traverse->mnode->mergedTo->paraNode->getMergeNodeInfo() )
               {
                  // NOTE; mNode == traverse->mnode->mergedTo:
                  traverse->mnode->mergedTo->paraNode->setDiffSubproblem(traverse->mnode->mergedTo->origDiffSubproblem);
                  traverse->mnode->mergedTo->paraNode->setMergeNodeInfo(0);
                  traverse->mnode->mergedTo->paraNode->setMergingStatus(-1);
                  delete traverse->mnode->mergedTo->mergedDiffSubproblem;
                  traverse->mnode->mergedTo->mergedDiffSubproblem = 0;
                  traverse->mnode->mergedTo->origDiffSubproblem = 0;
                  assert( traverse->mnode->mergedTo->mergedTo == 0);
                  deleteMergeNodeInfo(traverse->mnode->mergedTo);
               }
               traverse->mnode->mergedTo = 0;
               traverse->mnode->paraNode->setMergingStatus(0);
               traverse->mnode->status = BbParaMergeNodeInfo::PARA_MERGING;
               traverse->mnode->nMergedNodes = -1;
               traverse->mnode->nSameValueVariables = -1;
               traverse->mnode->keyIndex = -1;
            }
         }
      }
      else
      {
         for( BbParaFixedVariable *traverse = mNode->fixedVariables[mNode->keyIndex].next;
               traverse;
               traverse = traverse->next )
         {
            if( traverse->mnode->nMergedNodes == 0 && mNode == traverse->mnode->mergedTo )
            {
               traverse->mnode->mergedTo->nMergedNodes--;
               if( traverse->mnode->mergedTo->nMergedNodes == 0 && traverse->mnode->mergedTo->paraNode->getMergeNodeInfo() )
               {
                  // NOTE; mNode == traverse->mnode->mergedTo:
                  traverse->mnode->mergedTo->paraNode->setDiffSubproblem(traverse->mnode->mergedTo->origDiffSubproblem);
                  traverse->mnode->mergedTo->paraNode->setMergeNodeInfo(0);
                  traverse->mnode->mergedTo->paraNode->setMergingStatus(-1);
                  delete traverse->mnode->mergedDiffSubproblem;
                  traverse->mnode->mergedTo->mergedDiffSubproblem = 0;
                  traverse->mnode->mergedTo->origDiffSubproblem = 0;
                  assert( traverse->mnode->mergedTo->mergedTo == 0);
                  deleteMergeNodeInfo(traverse->mnode->mergedTo);
               }
               traverse->mnode->mergedTo = 0;
               if( traverse->mnode->paraNode->getDualBoundValue() < mNode->paraNode->getDualBoundValue() )
               {
                  traverse->mnode->paraNode->setMergingStatus(0);
                  traverse->mnode->status = BbParaMergeNodeInfo::PARA_MERGING;
                  traverse->mnode->nMergedNodes = -1;
                  traverse->mnode->nSameValueVariables = -1;
                  traverse->mnode->keyIndex = -1;
               }
               else
               {
                  traverse->mnode->paraNode->setMergingStatus(4);  // merging representative was deleted -> this node should be deleted
                  traverse->mnode->status = BbParaMergeNodeInfo::PARA_DELETED;
                  traverse->mnode->nMergedNodes = -1;
                  traverse->mnode->nSameValueVariables = -1;
                  traverse->mnode->keyIndex = -1;
               }
            }
         }
      }
   }

   if( mNode->fixedVariables )
   {
      for( int i = 0; i < mNode->nFixedVariables; i++ )
      {
         for( BbParaFixedVariable *traverse = mNode->fixedVariables[i].prev;
               traverse;
               traverse = traverse->prev
               )
         {
            traverse->nSameValue--;
         }
         if( mNode->fixedVariables[i].prev )
         {
            mNode->fixedVariables[i].prev->next = mNode->fixedVariables[i].next;
            if( mNode->fixedVariables[i].next )
            {
               mNode->fixedVariables[i].next->prev = mNode->fixedVariables[i].prev;
            }
            else
            {
               mNode->fixedVariables[i].prev->next = 0;
            }
         }
         else  // prev == 0
         {
            if( mNode->fixedVariables[i].next )
            {
               mNode->fixedVariables[i].next->prev = 0;
            }
         }

      }
      delete [] mNode->fixedVariables;
   }
   delete mNode;
}

int
BbParaNodesMerger::mergeNodes(
      BbParaNode *node,
      BbParaNodePool *paraNodePool
      )
{
   double startTime = paraTimer->getElapsedTime();

   BbParaMergeNodeInfo *mNode = dynamic_cast<BbParaNode *>(node)->getMergeNodeInfo();
   assert( mNode->status == BbParaMergeNodeInfo::PARA_MERGED_RPRESENTATIVE );
   assert( mNode->mergedTo == 0 );
   BbParaMergedNodeListElement *head = 0;
   BbParaMergedNodeListElement *cur = 0;
   int nMerged = 0;
   for( BbParaFixedVariable *traverse = mNode->fixedVariables[mNode->keyIndex].next;
         traverse;
         traverse = traverse->next )
   {
      if( traverse->mnode->nMergedNodes == 0 && mNode == traverse->mnode->mergedTo )
      {
         if( head == 0 )
         {
            head = cur = new BbParaMergedNodeListElement();
         }
         else
         {
            cur->next = new BbParaMergedNodeListElement();
            cur = cur->next;
         }
         cur->node = traverse->mnode->paraNode;
         dynamic_cast<BbParaNode *>(cur->node)->setMergingStatus(2);
         cur->next = 0;
         nMerged++;
#ifdef UG_DEBUG_SOLUTION
         if( cur->node->getDiffSubproblem() && cur->node->getDiffSubproblem()->isOptimalSolIncluded() )
         {
            assert(node->getDiffSubproblem());
            node->getDiffSubproblem()->setOptimalSolIndicator(1);
         }
#endif
      }
   }
   assert( mNode->nMergedNodes == nMerged);
   int delNodes = 0;
   if ( head )
   {
      delNodes = paraNodePool->removeMergedNodes(head);
   }
   assert( delNodes == nMerged );
   if( delNodes != nMerged )
   {
	   THROW_LOGICAL_ERROR4("delNodes != nMerged, delNodes = ", delNodes, ", nMerged = ", nMerged );
   }
   node->setDiffSubproblem(mNode->mergedDiffSubproblem);
   delete mNode->origDiffSubproblem;
   dynamic_cast<BbParaNode *>(node)->setMergeNodeInfo(0);
   dynamic_cast<BbParaNode *>(node)->setMergingStatus(1);
   assert(mNode->mergedTo == 0);
   mNode->mergedDiffSubproblem = 0;
   mNode->origDiffSubproblem = 0;
   deleteMergeNodeInfo(mNode);
//   lcts.nDeletedByMerging += nMerged;
//   if( logSolvingStatusFlag )
//   {
//      *osLogSolvingStatus << (nMerged + 1) <<
//            " nodes are merged at " <<
//            paraTimer->getElapsedTime() << " seconds."
//            << "Dual bound: " << dynamic_cast<BbParaInitiator *>(paraInitiator)->convertToExternalValue(
//                  dynamic_cast<BbParaNode *>(node)->getDualBoundValue() )
//            << std::endl;
//   }

   mergeNodeTime += paraTimer->getElapsedTime() - startTime;

   return nMerged;

}
