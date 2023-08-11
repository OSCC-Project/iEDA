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

/**@file    paraMergeNodesStructs.h
 * @brief   Structs used for merging nodes.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_NODES_MERGER_H__
#define __BB_PARA_NODES_MERGER_H__

#include<cassert>
#include<ug/paraTimer.h>
#include<ug/paraInitiator.h>
#include<ug/paraParamSet.h>
#include "bbParaInstance.h"

namespace UG
{

class BbParaNode;
class BbParaDiffSubproblem;
class BbParaNodePool;

typedef struct BbParaFixedValue_            BbParaFixedValue;
typedef struct BbParaMergeNodeInfo_         BbParaMergeNodeInfo;
typedef struct BbParaFixedVariable_         BbParaFixedVariable;
typedef struct BbParaSortedVariable_        BbParaSortedVariable;
typedef struct BbParaFixedValue_ *          BbParaFixedValuePtr;
typedef struct BbParaMergedNodeListElement_ BbParaMergedNodeListElement;

///
/// Fixed value struct
///
struct BbParaFixedValue_ {
   double            value;                 ///< value for a fixed variable
   BbParaFixedVariable *head;               ///< point the head of the ParaFixedVariable
   BbParaFixedVariable *tail;               ///< point the tail of the ParaFixedVarialbe
   BbParaFixedValue    *next;               ///< point next ParaFixedValue struct
};

///
/// Merge node information struct
///
struct BbParaMergeNodeInfo_ {
   enum {
      PARA_MERGING,                            ///< in merging process
      PARA_MERGED_RPRESENTATIVE,               ///< representative node for merging
      PARA_MERGE_CHECKING_TO_OTHER_NODE,       ///< checking possibility to merge with the other nodes
      PARA_MERGED_TO_OTHER_NODE,               ///< merged to the other node
      PARA_CANNOT_MERGE,                       ///< cannot merge to the other node
      PARA_DELETED                             ///< this node is deleted
   }   status;                                 ///< status of this ParaMargeNodeInfo
   int nSameValueVariables;                    ///< the number of fixed values which are the same as those of the merged node
                                               ///<     - This value < 0 means that this node is not merging target
   int nMergedNodes;                           ///< the number of merged nodes with this node.
                                               ///<     - This value > 0 : head
                                               ///<     - This value = 0 : merging to the other node
                                               ///<     - This value < 0 : no merging node
   int keyIndex;                               ///< The fixedVar of this index can reach all merging nodes
   int nFixedVariables;                        ///< the number of fixed variables
   BbParaFixedVariable *fixedVariables;        ///< array of fixed variable info
   BbParaMergeNodeInfo *mergedTo;              ///< pointer to merge node info to which this node is merged */
   BbParaNode *paraNode;                       ///< BbParaNode corresponding to this ParaMergeModeInfo */
   BbParaDiffSubproblem *origDiffSubproblem;   ///< original DiffSubproblem */
   BbParaDiffSubproblem *mergedDiffSubproblem; ///< merged DiffSubproblem, in case this node is merged and this is the head */
   BbParaMergeNodeInfo *next;                  ///< pointer to the next ParaMergeNodeInfo */
};

///
/// Fixed variable struct
///
struct BbParaFixedVariable_ {
   int    nSameValue;                       ///< the number of same value fixed variables in the following nodes
   int    index;                            ///< index of the variable among all solvers
   double value;                            ///< fixed value
   BbParaMergeNodeInfo *mnode;              ///< pointer to merge node info struct to which this info is belonging
   BbParaFixedVariable *next;               ///< pointer to the next node which has the same fixed value
   BbParaFixedVariable *prev;               ///< pointer to the previous node which has the same fixed value
};

///
/// Sorted variable struct
///
struct BbParaSortedVariable_ {
   int   idxInFixedVariabes;                ///< index in the fixedVariables array
   BbParaFixedVariable *fixedVariable;      ///< pointer to the fixedVariable
};

///
/// merged node list element stract
struct BbParaMergedNodeListElement_ {
   BbParaNode  *node;                       ///< pointer to BbParaNode object
   BbParaMergedNodeListElement *next;       ///< pointer to the next ParaMergedNodeListElement
};

class BbParaNodesMerger
{
   int varIndexRange;                       ///< variable index range
   int nBoundChangesOfBestNode;             ///< bound changes of the best node
   ParaTimer           *paraTimer;          ///< normal timer used
   BbParaInstance      *instance;           ///< pointer to ParaInstance object
   ParaParamSet        *paraParamSet;       ///< pointer to ParaParamSet object
   BbParaFixedValue    **varIndexTable;     ///< variable indices table.
   BbParaMergeNodeInfo *mergeInfoHead;      ///< head of BbParaMergeNodeInfo list
   BbParaMergeNodeInfo *mergeInfoTail;      ///< tail of BbParaMergeNodeInfo list
   /// times
   double addingNodeToMergeStructTime;        ///< accumulate time to add Node to merge struct
   double generateMergeNodesCandidatesTime;   ///< accumulate time to generate merge nodes candidates
   double regenerateMergeNodesCandidatesTime; ///< accumulate time to regenerate merge nodes candidates
   double mergeNodeTime;                      ///< accumulate time to make a merged node

public:

   BbParaNodesMerger(
         int inVarIndexRange,
         int inNBoundChangesOfBestNode,
         ParaTimer *inParaTimer,
         BbParaInstance *inParaInstance,
         ParaParamSet *inParaParamSet
         )
         :
         varIndexRange(inVarIndexRange),
         nBoundChangesOfBestNode(inNBoundChangesOfBestNode),
         paraTimer(inParaTimer),
         instance(inParaInstance),
         paraParamSet(inParaParamSet),
         varIndexTable(0),
         mergeInfoHead(0),
         mergeInfoTail(0),
         addingNodeToMergeStructTime(0.0),
         generateMergeNodesCandidatesTime(0.0),
         regenerateMergeNodesCandidatesTime(0.0),
         mergeNodeTime(0.0)
   {
      assert( varIndexRange > 0 );
      varIndexTable = new BbParaFixedValuePtr[varIndexRange];
      for( int i = 0; i < varIndexRange; i++ )
      {
         varIndexTable[i] = 0;
      }
      mergeInfoHead = 0;
      mergeInfoTail = 0;
   }

   ~BbParaNodesMerger(
         )
   {
   }

   ///
   /// add a node to nodes merger
   ///
   void addNodeToMergeNodeStructs(
         BbParaNode *node                     ///< pointer to BbParaNode object to be merged
         );

   ///
   /// generate merge nodes candidates
   ///
   void generateMergeNodesCandidates(
         ParaComm          *paraComm,         ///< pointer to paraComm object
         ParaInitiator     *paraInitiator     ///< pointer to ParaInitiatior object, this can be 0, if it is not
         );

   ///
   /// regenerate merge nodes candidates
   ///
   void regenerateMergeNodesCandidates(
         BbParaNode        *node,             ///< pointer to BbParaNode object to be removed from this merger
         ParaComm          *paraComm,         ///< pointer to paraComm object
         ParaInitiator     *paraInitiator     ///< pointer to ParaInitiatior object, this can be 0, if it is not
         );

   ///
   /// delete merge node info
   ///
   void deleteMergeNodeInfo(
         BbParaMergeNodeInfo *mNode           ///< pointer to BbParaMergeNodeInfo object to be removed
         );

   ///
   /// make a merge node
   /// @return the number of nodes merged
   ///
   int mergeNodes(
         BbParaNode *node,                   ///< pointer to BbParaNode object which is the representative
         BbParaNodePool *paraNodePool        ///< pointer to BbParaNodePool object
         );

   ///
   /// getter of addingNodeToMergeStructTime
   /// @return addingNodeToMergeStructTime
   ///
   double getAddingNodeToMergeStructTime(
         )
   {
      return addingNodeToMergeStructTime;
   }

   ///
   /// getter of generateMergeNodesCandidatesTime
   /// @return generateMergeNodesCandidatesTime
   ///
   double getGenerateMergeNodesCandidatesTime(
         )
   {
      return generateMergeNodesCandidatesTime;
   }

   ///
   /// getter of regenerateMergeNodesCandidatesTime
   /// @return regenerateMergeNodesCandidatesTime
   ///
   double getRegenerateMergeNodesCandidatesTime(
         )
   {
      return regenerateMergeNodesCandidatesTime;
   }

   ///
   /// getter of mergeNodeTime
   /// @return mergeNodeTime
   ///
   double getMergeNodeTime(
         )
   {
      return mergeNodeTime;
   }

};

}

#endif // __BB_PARA_NODES_MERGER_H__
