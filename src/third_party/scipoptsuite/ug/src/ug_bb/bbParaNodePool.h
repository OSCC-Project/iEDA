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

/**@file    paraNodePool.h
 * @brief   BbParaNode Pool.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_NODE_POOL_H__
#define __BB_PARA_NODE_POOL_H__
#include <map>
#include <queue>
#include <cfloat>
#include <cmath>
#include "bbParaInstance.h"
#include "bbParaNodesMerger.h"
#include "bbParaNode.h"

namespace UG
{

static const double eps = 1.0e-12;


///
/// class BbParaNodeSortCriterion
///
class BbParaNodeSortCriterion
{

public:

   ///
   /// ()operator
   /// @return true if n1 < n2
   ///
   bool operator()(
         const BbParaNodePtr& n1,       ///< BbParaNode pointer n1
         const BbParaNodePtr& n2        ///< BbParaNode pointer n2
         ) const
   {

      return EPSLT(n1->getDualBoundValue(),n2->getDualBoundValue(),eps) ||
              ( EPSEQ(n1->getDualBoundValue(),n2->getDualBoundValue(),eps) &&
                  n1->getDiffSubproblem() && n2->getDiffSubproblem() &&
                  n1->getDiffSubproblem()->getNBoundChanges() < n2->getDiffSubproblem()->getNBoundChanges() );

   }
};

///
/// class BbParaNodeSortCriterionForCleanUp
///
class BbParaNodeSortCriterionForCleanUp
{

public:

   ///
   /// ()operator
   /// @return true if n1 > n2
   ///
   bool operator()(
         const BbParaNodePtr& n1,      ///< BbParaNode pointer n1
         const BbParaNodePtr& n2       ///< BbParaNode pointer n2
         ) const
   {

      return EPSGT(n1->getDualBoundValue(),n2->getDualBoundValue(),eps) ||
              ( EPSEQ(n1->getDualBoundValue(),n2->getDualBoundValue(),eps) &&
                  n1->getDiffSubproblem() && n2->getDiffSubproblem() &&
                  n1->getDiffSubproblem()->getNBoundChanges() > n2->getDiffSubproblem()->getNBoundChanges() );

   }
};

///
/// class BbParaNodePool
///
class BbParaNodePool
{

protected:

   double        bgap;               ///< gap which can be considered as good
   size_t        maxUsageOfPool;     ///< maximum usage of this pool

public:

   ///
   /// constructor
   ///
   BbParaNodePool(
         double inBgap                ///< gap which can be considered as good
         )
         : bgap(inBgap),
           maxUsageOfPool(0)
   {
   }

   ///
   /// destructor
   ///
   virtual ~BbParaNodePool(
         )
   {
   }

   ///
   /// insert BbParaNode to this pool
   ///
   virtual void insert(
         BbParaNodePtr node             ///< pointer to BbParaNode object
         ) = 0;

   ///
   /// check if this pool is empty or not
   /// @return true if this pool is empty
   ///
   virtual bool isEmpty(
         ) = 0;

   ///
   /// extract a BbParaNode object from this pool
   /// @return pointer to BbParaNode object extracted
   ///
   virtual BbParaNodePtr extractNode(
         ) = 0;

   ///
   /// extract a BbParaNode object with the lowest priority from this pool
   /// @return pointer to BbParaNode object extracted
   ///
   virtual BbParaNodePtr extractWorstNode(
         ) = 0;

   ///
   /// extract a BbParaNode object randomly from this pool
   /// @return pointer to BbParaNode object extracted
   ///
   virtual BbParaNodePtr extractNodeRandomly(
         ) = 0;

   ///
   /// get best dual bound value of BbParaNode object in this pool
   /// @return best dual bound value
   ///
   virtual double getBestDualBoundValue(
         ) = 0;

   ///
   /// get number of good (heavy) BbParaNodes in this pool
   /// @return the number of good BbParaNodes
   ///
   virtual unsigned int getNumOfGoodNodes(
         double globalBestBound        ///< global best bound value to evaluate goodness
         ) = 0;

   ///
   /// get number of BbParaNodes in this pool
   /// @return number of BbParaNodes
   ///
   virtual size_t getNumOfNodes(
         ) = 0;

   ///
   /// remove bounded BbParaNodes by given incumbnet value
   /// @return the number of bounded BbParaNodes
   ///
   virtual int removeBoundedNodes(
         double incumbentValue        ///< incumbent value
         ) = 0;

   ///
   /// update dual bound values for saving BbParaNodes to checkpoint file
   ///
   virtual void updateDualBoundsForSavingNodes(
         ) = 0;

#ifdef UG_WITH_ZLIB

   ///
   /// write BbParaNodes to checkpoint file
   /// @return number of BbParaNodes written
   ///
   virtual int writeBbParaNodesToCheckpointFile(
         gzstream::ogzstream &out                  ///< gzstream for output
         ) = 0;

#endif

   ///
   /// remove merged BbParaNodes from this pool
   /// @return
   ///
   virtual int removeMergedNodes(
         BbParaMergedNodeListElement *head         ///< head of Merged BbParaNodes list
         ) = 0;

   ///
   /// stringfy this object
   /// @return string which shows inside of this object as string
   ///
   virtual const std::string toString(
         ) = 0;

   ///
   /// get maximum usage of this pool
   /// @return the maximum number of BbParaNodes that were in this pool
   ///
   size_t getMaxUsageOfPool(
         )
   {
      return maxUsageOfPool;
   }

};

///
/// class BbParaNodePoolForMinimization
/// @note only minimization pool was written, since all problem is converted to minimization problem inside of UG solvers
///
class BbParaNodePoolForMinimization : virtual public BbParaNodePool
{

   std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion > ascendingPool;    ///< asnding pool

public:

   ///
   /// constructor
   ///
   BbParaNodePoolForMinimization(
         double inBgap            ///< gap to evaluate a BbParaNode is good or not
         )
         : BbParaNodePool(inBgap)
   {
   }

   ///
   /// destructor
   ///
   ~BbParaNodePoolForMinimization(
         )
   {
      if( ascendingPool.size() > 0 )
      {
         std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion >::iterator p;
         for( p = ascendingPool.begin(); p != ascendingPool.end(); )
         {
            if( p->second ) delete p->second;
            ascendingPool.erase(p++);
         }
      }
   }

   ///
   /// insert a BbParaNode object to this pool
   ///
   void insert(
         BbParaNodePtr paraNode          ///< pointer to BbParaNode object to insert
         )
   {
      ascendingPool.insert(std::make_pair(paraNode,paraNode));
      if( maxUsageOfPool < ascendingPool.size() )
      {
         maxUsageOfPool = ascendingPool.size();
      }
   }

   ///
   /// check if this pool is empty or not
   /// @return true if this pool is empty
   ///
   bool isEmpty(
         )
   {
      return ( ascendingPool.size() == 0 );
   }

   ///
   /// extract a BbParaNode from this pool
   /// @return pointer to BbParaNode object extracted
   ///
   BbParaNodePtr extractNode(
         )
   {
      BbParaNodePtr extracted = 0;
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion >::iterator p;
      p = ascendingPool.begin();
      while( p != ascendingPool.end() )
      {
         if( p->second->getMergeNodeInfo() )
         {
            assert( p->second->getMergeNodeInfo()->status != BbParaMergeNodeInfo::PARA_MERGING );
            if( p->second->getMergingStatus() == 4 )  // merging representative was already deleted
            {
               delete p->second;              // this delete is not counted in statistics
               ascendingPool.erase(p++);
            }
            else
            {
               if( p->second->getMergeNodeInfo()->status == BbParaMergeNodeInfo::PARA_MERGE_CHECKING_TO_OTHER_NODE )
               {
                  assert(dynamic_cast<BbParaNodePtr>(p->second)->getMergeNodeInfo()->mergedTo->status == BbParaMergeNodeInfo::PARA_MERGED_RPRESENTATIVE );
                  p++;
               }
               else
               {
                  extracted = p->second;
                  ascendingPool.erase(p);
                  break;
               }
            }
         }
         else
         {
            extracted = p->second;
            ascendingPool.erase(p);
            break;
         }
      }
      assert( ( p == ascendingPool.end() && (!extracted) ) || ( p != ascendingPool.end() && extracted ) );
      return extracted;
   }

   ///
   /// extract a BbParaNode object with the lowest priority from this pool
   /// @return pointer to BbParaNode object extracted
   /// TODO: need to debug
   BbParaNodePtr extractWorstNode(
         )
   {
      BbParaNodePtr extracted = 0;
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion >::reverse_iterator rp;
      rp = ascendingPool.rbegin();
      //      while( rp != ascendingPool.rend() )
      if( rp != ascendingPool.rend() )
      {
         assert( !(rp->second->getMergeNodeInfo()) );
         extracted = rp->second;
         //         ascendingPool.erase(--(rp.base()));
         ascendingPool.erase((rp.base()));
         //         break;
      }
      // assert( ( rp == ascendingPool.rend() && (!extracted) ) || ( rp != ascendingPool.rend() && extracted ) );
      return extracted;
   }

   ///
   /// get a BbParaNode object, which is expected to extract from this pool
   /// @return pointer to BbParaNode object, which is expected to extract (Note: the BbParaNode object stays in pool)
   ///
   BbParaNodePtr getNextNode(
         )
   {
      BbParaNodePtr node = 0;
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion >::iterator p;
      p = ascendingPool.begin();
      while( p != ascendingPool.end() )
      {
         assert( !(p->second->getMergeNodeInfo()) ); // should not have this info.
         node = p->second;
         break;
      }
      assert( ( p == ascendingPool.end() && (!node) ) || ( p != ascendingPool.end() && node ) );
      return node;
   }


   ///
   /// extract a BbParaNode object randomly from this pool
   /// @return pointer to BbParaNode object extracted
   ///
   BbParaNodePtr extractNodeRandomly(
         )
   {
      BbParaNodePtr extracted = 0;
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion >::iterator p;
      p = ascendingPool.begin();
      size_t nNodes = ascendingPool.size();
      if( nNodes == 0 ) return extracted;
      int pos = 0;
      if( nNodes > 10 )
      {
         // select nodes randomly
         pos = rand()% static_cast<int>( nNodes * 0.8 ) + static_cast<int>(nNodes*0.1);
         if( pos < (static_cast<int>(nNodes*0.1)) || pos > nNodes*0.9 )
         {
            THROW_LOGICAL_ERROR4("Invalid pos in extractNodeRandomly in BbParaNodePool: pos = ", pos, ", nNodes = ", nNodes);
         }
         for( int j = 0; j < pos; j++ )
         {
            p++;
         }
      }
      assert( p != ascendingPool.end() );
      if( p == ascendingPool.end() ) // should not happen
      {
         p = ascendingPool.begin();
      }
      if( p->second->getMergeNodeInfo() )
      {
         assert( p->second->getMergeNodeInfo()->status != BbParaMergeNodeInfo::PARA_MERGING );
         if( p->second->getMergingStatus() == 4 )  // merging representative was already deleted
         {
            delete p->second;              // this delete is not counted in statistics
            ascendingPool.erase(p);
         }
         else
         {
            if( p->second->getMergeNodeInfo()->status == BbParaMergeNodeInfo::PARA_MERGE_CHECKING_TO_OTHER_NODE )
            {
               assert(dynamic_cast<BbParaNodePtr>(p->second)->getMergeNodeInfo()->mergedTo->status == BbParaMergeNodeInfo::PARA_MERGED_RPRESENTATIVE );
               std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion >::iterator pp;
               pp = ascendingPool.begin();
               while( pp != p )
               {
                  if( dynamic_cast<BbParaNodePtr>(pp->second) == dynamic_cast<BbParaNodePtr>(p->second)->getMergeNodeInfo()->mergedTo->paraNode )
                  {
                     extracted = pp->second;
                     ascendingPool.erase(pp);
                     break;
                  }
                  pp++;
               }
            }
            else
            {
               extracted = p->second;
               ascendingPool.erase(p);
            }
         }
      }
      else
      {
         extracted = p->second;
         ascendingPool.erase(p);
      }
      if( !extracted )
      {
         /** check nodes from the head of this pool */
         p = ascendingPool.begin();
         while( p != ascendingPool.end() )
         {
            if( p->second->getMergeNodeInfo() )
            {
               assert( p->second->getMergeNodeInfo()->status != BbParaMergeNodeInfo::PARA_MERGING );
               if( p->second->getMergingStatus() == 4 )  // merging representative was already deleted
               {
                  delete p->second;              // this delete is not counted in statistics
                  ascendingPool.erase(p++);
               }
               else
               {
                  if( p->second->getMergeNodeInfo()->status == BbParaMergeNodeInfo::PARA_MERGE_CHECKING_TO_OTHER_NODE )
                  {
                     assert(dynamic_cast<BbParaNodePtr>(p->second)->getMergeNodeInfo()->mergedTo->status == BbParaMergeNodeInfo::PARA_MERGED_RPRESENTATIVE );
                     p++;
                  }
                  else
                  {
                     extracted = p->second;
                     ascendingPool.erase(p);
                     break;
                  }
               }
            }
            else
            {
               extracted = p->second;
               ascendingPool.erase(p);
               break;
            }
         }
      }
      assert( ( p == ascendingPool.end() && (!extracted) ) || ( p != ascendingPool.end() && extracted ) );
      return extracted;
   }

   ///
   /// update dual bound values for saving BbParaNodes to checkpoint file
   ///
   void updateDualBoundsForSavingNodes(
         )
   {
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion >::iterator p;
      for( p = ascendingPool.begin(); p != ascendingPool.end(); ++p )
      {
         if( p->second->getAncestor() == 0 )
         {
            p->second->updateInitialDualBoundToSubtreeDualBound();
         }
      }
   }

#ifdef UG_WITH_ZLIB

   ///
   /// write BbParaNodes to checkpoint file
   /// @return number of BbParaNodes written
   ///
   int writeBbParaNodesToCheckpointFile(
         gzstream::ogzstream &out       ///< gzstream for output
         )
   {
      int n = 0;
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion >::iterator p;
      for( p = ascendingPool.begin(); p != ascendingPool.end(); ++p )
      {
         if( p->second->getAncestor() == 0 )
         {
            p->second->write(out);
            n++;
         }
      }
      return n;
   }

#endif

   ///
   /// get best dual bound value of BbParaNode object in this pool
   /// @return best dual bound value
   ///
   double getBestDualBoundValue(
         )
   {
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion >::iterator p;
      p = ascendingPool.begin();
      if( p != ascendingPool.end() )
      {
         return p->second->getDualBoundValue();
      }
      else
      {
         return DBL_MAX;  // no nodes exist
      }
   }

   ///
   /// get number of good (heavy) BbParaNodes in this pool
   /// @return the number of good BbParaNodes
   ///
   unsigned int getNumOfGoodNodes(
         double globalBestBound          ///< global best bound value to evaluate goodness
         )
   {
      /** The following code is not a good idea,
       * because only a node is received from a solver, LC can switch out
      if( globalBestBound > getBestDualBoundValue() )
         globalBestBound = getBestDualBoundValue();
      */
//
//      Old code:  we had an issue when Pool size is huge
//
//      int num = 0;
//      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion >::iterator p;
//      for( p = ascendingPool.begin(); p != ascendingPool.end() &&
//              ( ( ( p->second->getDualBoundValue() ) - globalBestBound ) /
//                    std::max( fabs(globalBestBound) , 1.0 ) ) < bgap;
//            ++p )
//      {
//         num++;
//      }

      int num2 = 0;
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion >::reverse_iterator rp;
      for( rp = ascendingPool.rbegin(); rp != ascendingPool.rend() &&
              ( ( ( rp->second->getDualBoundValue() ) - globalBestBound ) /
                    std::max( fabs(globalBestBound) , 1.0 ) ) > bgap;
            ++rp )
      {
         num2++;
      }
      int num3 = ascendingPool.size() - num2;
//      assert( num == num3 );
//      std::cout << "################### num = " << num << ", num3 = " << num3 << std::endl;

      return num3;
   }

   ///
   /// get number of BbParaNodes in this pool
   /// @return number of BbParaNodes
   ///
   size_t getNumOfNodes(
         )
   {
      return ascendingPool.size();
   }

   ///
   /// remove bounded BbParaNodes by given incumbnet value
   /// @return the number of bounded BbParaNodes
   ///
   int removeBoundedNodes(
         double incumbentValue        ///< incumbent value
         )
   {
      int nDeleted = 0;
      if( ascendingPool.size() > 0 )
      {
         std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion >::iterator p;
         for( p = ascendingPool.begin(); p != ascendingPool.end(); )
         {
            assert( p->second );
            if( !p->second->getMergeNodeInfo() )
            {
               if( p->second->getDualBoundValue() > incumbentValue || p->second->getMergingStatus() == 4 )
               {
                  nDeleted++;
                  delete p->second;
                  ascendingPool.erase(p++);
               }
               else
               {
                  p++;
               }
            }
            else
            {
               if( p->second->getMergeNodeInfo()->status == BbParaMergeNodeInfo::PARA_MERGE_CHECKING_TO_OTHER_NODE )
               {
                  if( p->second->getDualBoundValue() > incumbentValue || p->second->getMergingStatus() == 4 )
                   {
                      nDeleted++;
                      delete p->second;
                      ascendingPool.erase(p++);
                   }
                   else
                   {
                      p++;
                   }
               }
               else
               {
                  p++;
               }
            }
         }
      }
      return nDeleted;
   }

   ///
   /// remove merged BbParaNodes from this pool
   /// @return the number of BbParaNodes removed
   ///
   int removeMergedNodes(
         BbParaMergedNodeListElement *head           ///< head of Merged BbParaNodes list
         )
   {
      int nDeleted = 0;
      assert( ascendingPool.size() > 0 );
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion >::iterator p;
      for( p = ascendingPool.begin(); p != ascendingPool.end() && head; )
      {
         assert( p->second );
         if( p->second->getMergingStatus() == 2 )
         {
            BbParaMergedNodeListElement *prev = head;
            BbParaMergedNodeListElement *cur = head;
            for( ; cur; cur=cur->next )
            {
               if( p->second == cur->node )
               {
                  break;
               }
               prev = cur;
            }
            assert(cur);
            if( prev == head )
            {
               if( cur == prev )
               {
                  head = head->next;
               }
               else
               {
                  assert( cur == prev->next );
                  prev->next = prev->next->next;
               }
            }
            else
            {
               assert( cur == prev->next );
               prev->next = prev->next->next;
            }
            assert(  p->second == cur->node );
            assert( cur->node->getMergeNodeInfo() );
            delete cur;
            nDeleted++;
            delete p->second;
            ascendingPool.erase(p++);
         }
         else
         {
            p++;
         }
      }
      assert(!head);
      return nDeleted;
   }

   ///
   /// stringfy this object
   /// @return string which shows inside of this object as string
   ///
   const std::string toString(
         )
   {
      std::ostringstream s;
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterion >::iterator p;
      for( p = ascendingPool.begin(); p != ascendingPool.end(); ++p )
      {
         s << p->second->toString();
      }
      return s.str();
   }
};

///
/// class BbParaNodePoolForCleanUp
///
class BbParaNodePoolForCleanUp : virtual public BbParaNodePool
{
   std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterionForCleanUp > descendingPool;

public:

   ///
   /// constructor
   ///
   BbParaNodePoolForCleanUp(
         double inBgap                ///< gap which can be considered as good
         )
         : BbParaNodePool(inBgap)
   {
   }

   ///
   /// destructor
   ///
   ~BbParaNodePoolForCleanUp(
         )
   {
      if( descendingPool.size() > 0 )
      {
         std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterionForCleanUp >::iterator p;
         for( p = descendingPool.begin(); p != descendingPool.end(); )
         {
            if( p->second ) delete p->second;
            descendingPool.erase(p++);
         }
      }
   }

   ///
   /// insert BbParaNode to this pool
   ///
   void insert(
         BbParaNodePtr paraNode             ///< pointer to BbParaNode object
         )
   {
      descendingPool.insert(std::make_pair(paraNode,paraNode));
      if( maxUsageOfPool < descendingPool.size() )
      {
         maxUsageOfPool = descendingPool.size();
      }
   }

   ///
   /// check if this pool is empty or not
   /// @return true if this pool is empty
   ///
   bool isEmpty(
         )
   {
      return ( descendingPool.size() == 0 );
   }

   ///
   /// extract a BbParaNode object from this pool
   /// @return pointer to BbParaNode object extracted
   ///
   BbParaNodePtr extractNode(
         )
   {
      BbParaNodePtr extracted = 0;
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterionForCleanUp >::iterator p;
      p = descendingPool.begin();
      while( p != descendingPool.end() )
      {
         if( p->second->getMergeNodeInfo() )
         {
            THROW_LOGICAL_ERROR1("Nodes merging was used in clean up process!");
         }
         extracted = p->second;
         descendingPool.erase(p);
         break;
      }
      assert( ( p == descendingPool.end() && (!extracted) ) || ( p != descendingPool.end() && extracted ) );
      return extracted;
   }

   ///
   /// extract a BbParaNode object with the lowest priority from this pool
   /// @return pointer to BbParaNode object extracted
   ///
   BbParaNodePtr extractWorstNode(
         )
   {
      std::cerr << "***** LOGICAL ERROR : should not be called *****" << std::endl;
      abort();
   }

   ///
   /// extract a BbParaNode object randomly from this pool
   /// @return pointer to BbParaNode object extracted
   ///
   BbParaNodePtr extractNodeRandomly(
         )
   {
      BbParaNodePtr extracted = 0;
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterionForCleanUp >::iterator p;
      p = descendingPool.begin();
      while( p != descendingPool.end() )
      {
         if( p->second->getMergeNodeInfo() )
         {
            THROW_LOGICAL_ERROR1("Nodes merging was used in clean up process!");
         }
         extracted = p->second;
         descendingPool.erase(p);
         break;
      }
      assert( ( p == descendingPool.end() && (!extracted) ) || ( p != descendingPool.end() && extracted ) );
      return extracted;
   }

   ///
   /// update dual bound values for saving BbParaNodes to checkpoint file
   ///
   void updateDualBoundsForSavingNodes(
         )
   {
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterionForCleanUp >::iterator p;
      for( p = descendingPool.begin(); p != descendingPool.end(); ++p )
      {
         if( p->second->getAncestor() == 0 )
         {
            p->second->updateInitialDualBoundToSubtreeDualBound();
         }
      }
   }

#ifdef UG_WITH_ZLIB

   ///
   /// write BbParaNodes to checkpoint file
   /// @return number of BbParaNodes written
   ///
   int writeBbParaNodesToCheckpointFile(
         gzstream::ogzstream &out          ///< gzstram for output
         )
   {
      int n = 0;
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterionForCleanUp >::iterator p;
      for( p = descendingPool.begin(); p != descendingPool.end(); ++p )
      {
         if( p->second->getAncestor() == 0 )
         {
            p->second->write(out);
            n++;
         }
      }
      return n;
   }

#endif

   ///
   /// get best dual bound value of BbParaNode object in this pool
   /// @return best dual bound value
   ///
   double getBestDualBoundValue(
         )
   {
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterionForCleanUp >::reverse_iterator rp;
      rp = descendingPool.rbegin();
      if( rp != descendingPool.rend() )
      {
         return rp->second->getDualBoundValue();
      }
      else
      {
         return DBL_MAX;  // no nodes exist
      }
   }

   ///
   /// get number of good (heavy) BbParaNodes in this pool
   /// @return the number of good BbParaNodes
   ///
   unsigned int getNumOfGoodNodes(
         double globalBestBound            ///< global best bound value to evaluate goodness
         )
   {
      /** The following code is not a good idea,
       * because only a node is received from a solver, LC can switch out
      if( globalBestBound > getBestDualBoundValue() )
         globalBestBound = getBestDualBoundValue();
      */
      int num = 0;
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterionForCleanUp >::reverse_iterator rp;
      for( rp = descendingPool.rbegin(); rp != descendingPool.rend() &&
              ( ( ( rp->second->getDualBoundValue() ) - globalBestBound ) /
                    std::max( fabs(globalBestBound) , 1.0 ) ) < bgap;
            ++rp )
      {
         num++;
      }
      return num;
   }

   ///
   /// get number of BbParaNodes in this pool
   /// @return number of BbParaNodes
   ///
   size_t getNumOfNodes(
         )
   {
      return descendingPool.size();
   }

   ///
   /// remove bounded BbParaNodes by given incumbnet value
   /// @return the number of bounded BbParaNodes
   ///
   int removeBoundedNodes(
         double incumbentValue        ///< incumbent value
         )
   {
      int nDeleted = 0;
      if( descendingPool.size() > 0 )
      {
         std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterionForCleanUp >::iterator p;
         for( p = descendingPool.begin(); p != descendingPool.end(); )
         {
            assert( p->second );
            if( !p->second->getMergeNodeInfo() )
            {
               if( p->second->getDualBoundValue() > incumbentValue )
               {
                  nDeleted++;
                  delete p->second;
                  descendingPool.erase(p++);
               }
               else
               {
                  p++;
               }
            }
            else
            {
               THROW_LOGICAL_ERROR1("Nodes merging was used in clean up process!");
            }
         }
      }
      return nDeleted;
   }

   ///
   /// remove merged BbParaNodes from this pool
   /// @return the number of BbParaNodes removed
   ///
   int removeMergedNodes(
         BbParaMergedNodeListElement *head           ///< head of Merged BbParaNodes list
         )
   {
      THROW_LOGICAL_ERROR1("Nodes merging was used in clean up process!");
   }

   ///
   /// stringfy this object
   /// @return string which shows inside of this object as string
   ///
   const std::string toString(
         )
   {
      std::ostringstream s;
      std::multimap<BbParaNodePtr, BbParaNodePtr, BbParaNodeSortCriterionForCleanUp >::iterator p;
      for( p = descendingPool.begin(); p != descendingPool.end(); ++p )
      {
         s << p->second->toString();
      }
      return s.str();
   }
};

}

#endif // __BB_PARA_NODE_POOL_H__
