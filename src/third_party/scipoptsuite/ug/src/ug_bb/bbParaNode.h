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

/**@file    paraNode.h
 * @brief   Base class for BbParaNode.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_NODE_H__
#define __BB_PARA_NODE_H__

#include <cassert>
#include <iostream>
#include <fstream>
#include <map>
#include "ug/paraDef.h"
#include "ug/paraComm.h"
#ifdef UG_WITH_ZLIB
#include "ug/gzstream.h"
#endif
#include "ug/paraTask.h"
#include "bbParaDiffSubproblem.h"
#include "bbParaNodesMerger.h"


namespace UG
{


///
/// class BbParaNode
///
class BbParaNode : public ParaTask
{

protected:

   int             depth;                  ///< depth from the root node of original tree
   double          dualBoundValue;         ///< dual bound value
   double          initialDualBoundValue;  ///< dual bound value when this node is created
                                           ///< This value is updated to precise one when there is guarantee
   int             basisInfo;              ///< indicate if basis information is including or not
   int             mergingStatus;          ///< merging status:
                                           ///<       -1 - no merging node,
                                           ///<        0 - checking,
                                           ///<        1 - merged (representative)
                                           ///<        2 - merged to the other node
                                           ///<        3 - cannot be merged
                                           ///<        4 - merging representative was deleted
   BbParaMergeNodeInfo *mergeNodeInfo;     ///< pointer to mergeNodeInfo. Not zero means merging
   bool            nodesAreCollected;      ///< subproblems generated from this nodes are collected at interruption.
                                           ///<        this field is not transferred
public:

   BbParaNode      *next;                  ///< this pointer is used in case of self-split ramp-up in LC
                                           ///<        this field is not transferred

   ///
   /// default constructor
   ///
   BbParaNode(
         )
         : ParaTask(),
           depth(-1),
           dualBoundValue(-DBL_MAX),
           initialDualBoundValue(0.0),
           basisInfo(0),
		     mergingStatus(-1),
		     mergeNodeInfo(0),
		     nodesAreCollected(false),
		     next(0)
   {
   }

   ///
   ///  constructor
   ///
   BbParaNode(
         TaskId inNodeId,                         ///< node id
         TaskId inGeneratorNodeId,                ///< generator node id
         int inDepth,                             ///< depth in global search tree
         double inDualBoundValue,                 ///< dual bound value
         double inOriginalDualBoundValue,         ///< original dual bound value when the node is generated
         double inEstimatedValue,                 ///< estimated value
         ParaDiffSubproblem *inDiffSubproblem     ///< pointer to BbParaDiffSubproblem object
         )
         : ParaTask(inNodeId, inGeneratorNodeId, inEstimatedValue, inDiffSubproblem),
           depth(inDepth),
           dualBoundValue(inDualBoundValue),
           initialDualBoundValue(inOriginalDualBoundValue),
           basisInfo(0),
		     mergingStatus(-1),
		     mergeNodeInfo(0),
		     nodesAreCollected(false),
		     next(0)
   {
   }

   ///
   ///  destructor
   ///
   virtual ~BbParaNode(
         )
   {
      assert((mergingStatus != -1) || (mergingStatus == -1 && mergeNodeInfo == 0) );

      if( ancestor )
      {
         ParaTaskGenealogicalLocalPtr *localPtrAncestor = dynamic_cast< ParaTaskGenealogicalLocalPtr * >(ancestor);
         if( !descendants.empty() )
         {
            std::map< TaskId, ParaTaskGenealogicalPtrPtr >::iterator pos;
            for( pos = descendants.begin(); pos != descendants.end(); )
            {
               if( pos->second->getType() == ParaTaskLocalPtr )
               {
                  ParaTaskGenealogicalLocalPtr *localPtrDescendant = dynamic_cast< ParaTaskGenealogicalLocalPtr * >(pos->second);
                  assert( localPtrDescendant->getPointerValue()->ancestor->getTaskId() == taskId );
                  assert( localPtrAncestor->getTaskId() == localPtrAncestor->getPointerValue()->taskId );
                  assert( localPtrDescendant->getTaskId() == localPtrDescendant->getPointerValue()->taskId );
                  localPtrDescendant->getPointerValue()->setAncestor(
                        new ParaTaskGenealogicalLocalPtr( localPtrAncestor->getTaskId(), localPtrAncestor->getPointerValue() ) );
                  localPtrAncestor->getPointerValue()->addDescendant(
                        new ParaTaskGenealogicalLocalPtr( localPtrDescendant->getTaskId(), localPtrDescendant->getPointerValue() ) );
               }
               else
               {  /** not implemented yet **/
                  ABORT_LOGICAL_ERROR1("remote pointer is not implemented yet, but it is called!");
               }
               delete pos->second;
               descendants.erase(pos++);
            }
         }
         if( ancestor->getType() == ParaTaskLocalPtr )
         {
             assert( localPtrAncestor->getTaskId() == localPtrAncestor->getPointerValue()->taskId );
             localPtrAncestor->getPointerValue()->removeDescendant(taskId);
         }
         else
         {   /** not implemented yet **/
             ABORT_LOGICAL_ERROR1("remote pointer is not implemented yet, but it is called!");
         }
         delete ancestor;
      }
      else
      {
         if( !descendants.empty() )
         {
            std::map< TaskId, ParaTaskGenealogicalPtrPtr >::iterator pos;
            for( pos = descendants.begin(); pos != descendants.end(); )
            {
               if( pos->second->getType() == ParaTaskLocalPtr )
               {
                  ParaTaskGenealogicalLocalPtr *localPtrDescendant = dynamic_cast< ParaTaskGenealogicalLocalPtr * >(pos->second);
                  localPtrDescendant->getPointerValue()->setAncestor(0);
               }
               else
               {  /** not implemented yet **/
                  ABORT_LOGICAL_ERROR1("remote pointer is not implemented yet, but it is called!");
               }
               delete pos->second;
               descendants.erase(pos++);
            }
         }
      }
      if( mergeNodeInfo )
      {
         if( mergeNodeInfo->nMergedNodes == 0 && mergeNodeInfo->mergedTo )
         {
            assert(mergeNodeInfo->status == BbParaMergeNodeInfo::PARA_MERGE_CHECKING_TO_OTHER_NODE);
            assert(mergeNodeInfo->mergedTo->status ==  BbParaMergeNodeInfo::PARA_MERGED_RPRESENTATIVE);
            mergeNodeInfo->mergedTo->nMergedNodes--;
         }

         assert( mergeNodeInfo->status != BbParaMergeNodeInfo::PARA_MERGING);

         if( mergeNodeInfo->status == BbParaMergeNodeInfo::PARA_MERGED_RPRESENTATIVE )
         {
            for( BbParaFixedVariable *traverse = mergeNodeInfo->fixedVariables[mergeNodeInfo->keyIndex].next;
                  traverse;
                  traverse = traverse->next )
            {
               if( traverse->mnode->nMergedNodes == 0 && mergeNodeInfo == traverse->mnode->mergedTo )
               {
                  traverse->mnode->mergedTo->nMergedNodes--;
                  traverse->mnode->mergedTo = 0;
                  if( traverse->mnode->paraNode->getDualBoundValue() < mergeNodeInfo->paraNode->getDualBoundValue() )
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

         if( mergeNodeInfo->fixedVariables )
         {
            for( int i = 0; i < mergeNodeInfo->nFixedVariables; i++ )
            {
               for( BbParaFixedVariable *traverse = mergeNodeInfo->fixedVariables[i].prev;
                     traverse;
                     traverse = traverse->prev
                     )
               {
                  traverse->nSameValue--;
               }
               if( mergeNodeInfo->fixedVariables[i].prev )
               {
                  mergeNodeInfo->fixedVariables[i].prev->next = mergeNodeInfo->fixedVariables[i].next;
                  if( mergeNodeInfo->fixedVariables[i].next )
                  {
                     mergeNodeInfo->fixedVariables[i].next->prev = mergeNodeInfo->fixedVariables[i].prev;
                  }
                  else
                  {
                     mergeNodeInfo->fixedVariables[i].prev->next = 0;
                  }
               }
               else
               {
                  if( mergeNodeInfo->fixedVariables[i].next )
                  {
                     mergeNodeInfo->fixedVariables[i].next->prev = 0;
                  }
               }
            }
            delete [] mergeNodeInfo->fixedVariables;
         }
         if( mergeNodeInfo->origDiffSubproblem )
         {
            if( mergeNodeInfo->origDiffSubproblem != diffSubproblem )
            {
               delete mergeNodeInfo->origDiffSubproblem;
            }
         }
         delete mergeNodeInfo;
      }
   }

   ///
   /// getter of depth
   /// @return depth of this node in global tree
   ///
   int getDepth(
         )
   {
      return depth;
   }

   ///
   /// setter of depth
   ///
   void setDepth(
         int inDepth               ///< depth
         )
   {
      depth = inDepth;
   }

   ///
   /// getter of dual bound value
   /// @return dual bound value
   ///
   double getDualBoundValue(
         )
   {
      return dualBoundValue;
   }

   ///
   /// getter of initial dual bound value
   /// @return initial dual bound value
   ///
   ///
   double getInitialDualBoundValue(
         )
   {
      return initialDualBoundValue;
   }

   ///
   /// setter of dual bound value
   ///
   void setDualBoundValue(
         double inDualBoundValue       ///< dual bound value
         )
   {
      dualBoundValue = inDualBoundValue;
   }

   ///
   /// setter of initial dual bound value
   ///
   void setInitialDualBoundValue(
         double inTrueDualBoundValue     ///< inital dual bound value
         )
   {
      initialDualBoundValue = inTrueDualBoundValue;
   }

   ///
   /// reset dual bound value
   ///
   void resetDualBoundValue(
         )
   {
      dualBoundValue = initialDualBoundValue;
   }

   ///
   /// getter of diffSubproblem
   /// @return pointer to BbParaDiffSubproblem object
   ///
   BbParaDiffSubproblem *getDiffSubproblem(
         )
   {
      return dynamic_cast<BbParaDiffSubproblem *>(diffSubproblem);
   }

   ///
   /// setter of diffSubproblem */
   ///
   void setDiffSubproblem(
         BbParaDiffSubproblem *inDiffSubproblem    ///< pointer to BbParaDiffSubproblem object
         )
   {
      diffSubproblem = inDiffSubproblem;
   }

   ///
   /// getter of ancestor
   /// @return ancestor BbParaNodeGenealogicalPtr
   ///
   ParaTaskGenealogicalPtr *getAncestor(
         )
   {
      return ancestor;
   }

   ///
   /// setter of ancestor
   ///
   void setAncestor(
         ParaTaskGenealogicalPtr *inAncestor   ///< ancestor BbParaNodeGenealogicalPtr
         )
   {
      if( ancestor ) delete ancestor;
      ancestor = inAncestor;
   }

   ///
   /// remove a descendant
   ///
   void removeDescendant(
         TaskId removeNodeId                  ///< node id to remove
         )
   {
      std::map< TaskId, ParaTaskGenealogicalPtrPtr >::iterator pos;
      pos = descendants.find(removeNodeId);
      if( pos != descendants.end() )
      {
         delete pos->second;
         descendants.erase(pos);
      }
      else
      {
         for( pos = descendants.begin(); pos != descendants.end(); )
         {
            if( pos->second->getType() == ParaTaskLocalPtr )
            {
               ParaTaskGenealogicalLocalPtr *localPtrDescendant = dynamic_cast< ParaTaskGenealogicalLocalPtr * >(pos->second);
               std::cout << "Descendant NodeId = " << localPtrDescendant->getTaskId().toString() << std::endl;
            }
            else
            {
               /** not implemented yet */
            }
            pos++;
         }
         THROW_LOGICAL_ERROR1("invalid NodeId removed!");
      }
   }

   ///
   /// check if this node has descendant or not
   /// @return true if it has descendant
   ///
   bool hasDescendant(
         )
   {
      return !(descendants.empty());
   }

   ///
   /// add a descendant
   ///
   void addDescendant(
         ParaTaskGenealogicalPtr *inDescendant   ///< descendant BbParaNodeGenealogicalPtr
         )
   {
      descendants.insert(std::make_pair(inDescendant->getTaskId(),inDescendant));
   }

   ///
   /// update initial dual bound value by checking descendant dual bound values
   ///
   void updateInitialDualBoundToSubtreeDualBound(
         )
   {
      // dualBoundValue = getMinimumDualBoundInDesendants(dualBoundValue);
      /** More accurate dual bound of this node is obtained */
      initialDualBoundValue = getMinimumDualBoundInDesendants(dualBoundValue);
   }

   ///
   /// get minimum dual bound value in descendants
   /// @return dual bound value
   ///
   double getMinimumDualBoundInDesendants(
         double value
         )
   {
      if( descendants.empty() ) return value;
      std::map< TaskId, ParaTaskGenealogicalPtrPtr >::iterator pos;
      for( pos = descendants.begin(); pos != descendants.end(); )
      {
         if( pos->second->getType() == ParaTaskLocalPtr )
         {
            ParaTaskGenealogicalLocalPtr *localPtrDescendant = dynamic_cast< ParaTaskGenealogicalLocalPtr * >(pos->second);
            value = std::min( value, dynamic_cast<BbParaNode *>(localPtrDescendant->getPointerValue())->getDualBoundValue() );
            value = std::min(
                  value,
                  dynamic_cast<BbParaNode *>(localPtrDescendant->getPointerValue())->getMinimumDualBoundInDesendants(value));
         }
         else
         {
            /** not implemented yet */
         }
         pos++;
      }
      return value;
   }

   ///
   /// clone this BbParaNode
   /// @return pointer to cloned BbParaNode object
   ///
   virtual BbParaNode* clone(
         ParaComm *comm          ///< communicator used
         ) = 0;

   ///
   /// broadcast this object
   /// @return always 0 (for future extensions)
   ///
   virtual int bcast(
         ParaComm *comm,        ///< communicator used
         int root               ///< root rank of broadcast
         ) = 0;

   ///
   /// send this object
   /// @return always 0 (for future extensions)
   ///
   virtual int send(
         ParaComm *comm,        ///< communicator used
         int destination        ///< destination rank
         ) = 0;

   ///
   /// receive this object
   /// @return always 0 (for future extensions)
   ///
   virtual int receive(
         ParaComm *comm,        ///< communicator used
         int source             ///< source rank
         ) = 0;

   ///
   /// send new subtree root node
   /// @return always 0 (for future extensions)
   ///
   virtual int sendNewSubtreeRoot(
         ParaComm *comm,        ///< communicator used
         int destination        ///< destination rank
         ) = 0;

   ///
   /// send subtree root to be removed
   /// @return always 0 (for future extensions)
   ///
   virtual int sendSubtreeRootNodeId(
         ParaComm *comm,               ///< communicator used
         int destination,              ///< destination rank
         int tag                       ///< tag of message
         ) = 0;

//   ///
//   /// send subtree root to be reassigned
//   /// @return always 0 (for future extensions)
//   ///
//   virtual int sendReassignSelfSplitSubtreeRoot(
//         ParaComm *comm,               ///< communicator used
//         int destination               ///< destination rank
//         ) = 0;

   ///
   /// receive this object
   /// @return always 0 (for future extensions)
   ///
   virtual int receiveNewSubtreeRoot(
         ParaComm *comm,        ///< communicator used
         int source             ///< source rank
         ) = 0;

   ///
   /// receive this object node Id
   /// @return always 0 (for future extensions)
   ///
   virtual int receiveSubtreeRootNodeId(
         ParaComm *comm,                ///< communicator used
         int source,                    ///< source rank
         int tag                        ///< tag of message
         ) = 0;

//   ///
//   /// receive this object node Id
//   /// @return always 0 (for future extensions)
//   ///
//   virtual int receiveReassignSelfSplitSubtreeRoot(
//         ParaComm *comm,                ///< communicator used
//         int source                      ///< source rank
//         ) = 0;



#ifdef UG_WITH_ZLIB

   ///
   /// write to checkpoint file
   ///
   void write(
         gzstream::ogzstream &out   ///< gzstream for output
         );

   ///
   /// read from checkpoint file
   ///
   bool read(
         ParaComm *comm,            ///< communicator used
         gzstream::igzstream &in,   ///< gzstream for input
         bool onlyBoundChanges      ///< indicate if only bound changes are read or not
         );

#endif

   ///
   /// stringfy BbParaNode
   /// @return string to show inside of this object
   ///
   const std::string toString(
         )
   {
      std::ostringstream s;
      s << "BbParaNodeId = " << (taskId.toString()) << ", GeneratorNodeId = " << (generatorTaskId.toString())
      << ", depth = " << depth << ", dual bound value = " << dualBoundValue
      << ", initialDualBoundValue = " << initialDualBoundValue
      << ", estimated value = " << estimatedValue << std::endl;
      if( diffSubproblem )
      {
         s << diffSubproblem->toString();
      }
      return s.str();
   }

   ///
   /// stringfy BbParaNode as simple string
   /// @return string to show inside of this object
   ///
   const std::string toSimpleString(
         )
   {
      std::ostringstream s;
      s << taskId.toString()
            << ", "
            << generatorTaskId.toString()
            << ", ";
      if( diffSubproblem )
      {
         s << dynamic_cast<BbParaDiffSubproblem *>(diffSubproblem)->getNBoundChanges();
      }
      else
      {
         s << 0;
      }
      s << ", " << depth
            << ", "
            << initialDualBoundValue
            << ", "
            << dualBoundValue;
      return s.str();
   }

   ///
   /// set merge node information to this BbParaNode object
   ///
   void setMergeNodeInfo(
         BbParaMergeNodeInfo *mNode      ///< pointer to merge node information struct
         )
   {
      assert(mergingStatus != -1);
      mergeNodeInfo = mNode;
   }

   ///
   /// get merge node information struct
   /// @return pointer to merge node information struct
   ///
   BbParaMergeNodeInfo *getMergeNodeInfo(
         )
   {
      return mergeNodeInfo;
   }

   ///
   /// set merging status
   ///
   void setMergingStatus(
         int status                  ///< merging status
         )
   {
      mergingStatus = status;
   }

   ///
   /// get merging status
   /// @return merging status
   ///
   int getMergingStatus(
         )
   {
      return mergingStatus;
   }

   ///
   /// check if nodes are collected or not
   /// @return true if they are collected
   ///
   bool areNodesCollected(
         )
   {
      return nodesAreCollected;
   }

   ///
   /// set all nodes are collected
   /// TODO: this function has to be investigated
   ///
   void collectsNodes(
         )
   {
      nodesAreCollected = true;
   }

   ///
   /// check if this node's id is the same as that of argument ParaNode's task id
   /// @return true if the both are the same
   ///
   bool isSameNodeIdAs(
         const BbParaNode& inNode      ///< ParaNode
         )
   {
      if( taskId.subtaskId == inNode.taskId.subtaskId && generatorTaskId == inNode.generatorTaskId )
         return true;
      else return false;
   }


};

typedef BbParaNode *BbParaNodePtr;

}

#endif // __BB_PARA_NODE_H__

