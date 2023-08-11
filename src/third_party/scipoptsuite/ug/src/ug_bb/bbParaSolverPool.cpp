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

/**@file    paraSolverPool.cpp
 * @brief   Solver pool.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <cassert>
#include <string>
#include <cstring>
#include <sstream>
#include <list>
#include <map>
#include <cmath>
#include "ug/paraDef.h"
#include "bbParaComm.h"
#include "bbParaParamSet.h"
#include "bbParaSolverPool.h"

using namespace std;
using namespace UG;

/** constructor of selection heap */
SelectionHeap::SelectionHeap(
      std::size_t size                               /**< maximum size of this heap */
      )
{
   maxHeapSize = size;
   heapSize = 0;
   heap = new BbParaSolverPoolElementPtr[size+1]; /* index 0 is a sentinel */
}

/** destructor of selection heap */
SelectionHeap::~SelectionHeap(
      )
{
   delete[] heap;
}
/** resize selection heap */
void
SelectionHeap::resize(
      std::size_t size                               /**< new heap size */
      )
{
   BbParaSolverPoolElementPtr *tempHeap =
           new BbParaSolverPoolElementPtr[size+1]; /* index 0 is a sentinel */
   std::memcpy(tempHeap, heap, (unsigned long int)(sizeof(BbParaSolverPoolElementPtr) * (maxHeapSize+1)));
   delete[] heap;
   heap = tempHeap;
   maxHeapSize = size;
}

/** insert BbParaSolverPoolElementPtr to selection heap */
SelectionHeap::ResultOfInsert
SelectionHeap::insert(
      BbParaSolverPoolElementPtr inSolver      /**< BbParaSolverPoolElementPtr to be inserted */
      )
{
   if( heapSize >= maxHeapSize ) return FAILED_BY_FULL;
   heap[++heapSize] = inSolver;
   upHeap(heapSize);
   return SUCCEEDED;
}

/** remove the top priority element form selection heap */
BbParaSolverPoolElementPtr
SelectionHeap::remove(
      )
{
   BbParaSolverPoolElementPtr solver = heap[1];

   heap[1] = heap[heapSize];
   heapSize--;
   downHeap(1);
   if( heapSize == 0 )
   {
       heap[1] = 0;
   }
   solver->setSelectionHeapElement(0);
   return solver;
}


/** stringfy selection heap */
const std::string
SelectionHeap::toString(
      )
{
   std::ostringstream os;
   os << "--- selection heap ---" << std::endl;
   os << "maxHeapSize: " << maxHeapSize << std::endl;
   os << "heapSize: " << heapSize << std::endl;
   for( std::size_t i = 1; i <= heapSize; i++ )
   {
      os << "heap[" << i << "]->rank: "
         << heap[i]->getRank() << std::endl
         << "heap[" << i << "]->status: "
         << static_cast<int>(heap[i]->getStatus()) << std::endl
         << "heap[" << i << "]->bestBound: "
         << heap[i]->getBestDualBoundValue() << std::endl
         << "heap[" << i << "]->numOfNodesLeft: "
         << heap[i]->getNumOfNodesLeft()  << std::endl
         << "heap[" << i << "]->numOfDiff: "
         << heap[i]->getNumOfDiffNodesLeft() <<  std::endl
         << "heap[" << i << "]->collectingMode: "
         << heap[i]->isInCollectingMode()  << std::endl;
   }
   return os.str();
}

/** constructor of DescendingSelectionHeap */
DescendingSelectionHeap::DescendingSelectionHeap(
      std::size_t size                                 /**< maximum size of this heap */
      )
      : SelectionHeap(size)
{
}

/** update dual bound value of the solver in heap */
void
DescendingSelectionHeap::updateDualBoundValue(
      BbParaSolverPoolElementPtr solver,         /**< pointer to solver pool element whose dual bound is updated */
      double               newDualBoundValue   /**< new dual bound value */
      )
{
   int pos = solver->getSelectionHeapElement() - heap;
   if( solver->getBestDualBoundValue() < newDualBoundValue )
   {
      solver->setBestDualBoundValue(newDualBoundValue);
      upHeap(pos);
   }
   else
   {
      solver->setBestDualBoundValue(newDualBoundValue);
      downHeap(pos);
   }
}

/** delete BbParaSolverPoolElement */
void
DescendingSelectionHeap::deleteElement(
      BbParaSolverPoolElementPtr solver        /**< BbParaSolverPoolElement to be deleted */
      )
{
   std::size_t pos = (solver->getSelectionHeapElement()) - heap;

   if( pos == heapSize )
   {
      /* no need to rearrange heap element */
      heap[heapSize--] = 0;
   }
   else
   {
      if( heap[pos]->getBestDualBoundValue() < heap[heapSize]->getBestDualBoundValue() )
      {
         heap[pos] = heap[heapSize];
         heap[heapSize--] = 0;
         upHeap(pos);
      }
      else
      {
         heap[pos] = heap[heapSize];
         heap[heapSize--] = 0;
         downHeap(pos);
      }
   }
   solver->setSelectionHeapElement(0);
}


/** up heap */
void
DescendingSelectionHeap::upHeap(
   std::size_t pos                                      /**< up heap this position element */
   )
{
   BbParaSolverPoolElementPtr she;

   she = heap[pos];
   heap[0] = NULL;
   while ( heap[pos/2] != NULL &&
         ( heap[pos/2]->getBestDualBoundValue()
               < she->getBestDualBoundValue() ) )
   {
      heap[pos] = heap[pos/2];
      heap[pos]->setSelectionHeapElement(&heap[pos]);
      pos = pos/2;
   }
   heap[pos] = she;
   heap[pos]->setSelectionHeapElement(&heap[pos]);
}

/** down heap */
void
DescendingSelectionHeap::downHeap(
   std::size_t pos                                       /**< down heap this position element */
   )
{
   std::size_t j;
   BbParaSolverPoolElementPtr she;

   she = heap[pos];
   while ( pos <= (heapSize/2) )
   {
      j = pos + pos;
      if( j < heapSize &&
            ( heap[j]->getBestDualBoundValue()
                  < heap[j+1]->getBestDualBoundValue() ) ) j++;
      if( she->getBestDualBoundValue()
            > heap[j]->getBestDualBoundValue() )
         break;
      heap[pos] = heap[j];
      heap[pos]->setSelectionHeapElement(&heap[pos]);
      pos = j;
   }
   heap[pos] = she;
   heap[pos]->setSelectionHeapElement(&heap[pos]);
}

/** constructor of AscendingSelectionHeap */
AscendingSelectionHeap::AscendingSelectionHeap(
      std::size_t size                                   /**< maximum size of this heap */
      )
      : SelectionHeap(size)
{
}

/** update dual bound value of the solver in heap */
void
AscendingSelectionHeap::updateDualBoundValue(
      BbParaSolverPoolElementPtr solver,           /**< pointer to solver pool element whose dual bound is updated */
      double newDualBoundValue                   /**< new dual bound value */
      )
{
   int pos = solver->getSelectionHeapElement() - heap;
   if( solver->getBestDualBoundValue() > newDualBoundValue )
   {
      solver->setBestDualBoundValue(newDualBoundValue);
      upHeap(pos);
   }
   else
   {
      solver->setBestDualBoundValue(newDualBoundValue);
      downHeap(pos);
   }
}

/** delete BbParaSolverPoolElement */
void
AscendingSelectionHeap::deleteElement(
      BbParaSolverPoolElementPtr solver        /**< BbParaSolverPoolElement to be deleted */
      )
{
   std::size_t pos = (solver->getSelectionHeapElement()) - heap;

   if( pos == heapSize )
   {
      /* no need to rearrange heap element */
      heap[heapSize--] = 0;
   }
   else
   {
      if( heap[pos]->getBestDualBoundValue() > heap[heapSize]->getBestDualBoundValue() )
      {
         heap[pos] = heap[heapSize];
         heap[heapSize--] = 0;
         upHeap(pos);
      }
      else
      {
         heap[pos] = heap[heapSize];
         heap[heapSize--] = 0;
         downHeap(pos);
      }
   }
   solver->setSelectionHeapElement(0);
}

/** up heap */
void
AscendingSelectionHeap::upHeap(
   std::size_t pos                                        /**< up heap this position element */
){
   BbParaSolverPoolElementPtr she;

   she = heap[pos];
   heap[0] = NULL;

   while ( heap[pos/2] != NULL &&
         ( heap[pos/2]->getBestDualBoundValue()
               > she->getBestDualBoundValue() ) )
   {
      heap[pos] = heap[pos/2];
      heap[pos]->setSelectionHeapElement(&heap[pos]);
      pos = pos/2;
   }
   heap[pos] = she;
   heap[pos]->setSelectionHeapElement(&heap[pos]);
}

/** down heap */
void
AscendingSelectionHeap::downHeap(
   std::size_t pos                                       /**< down heap this position element */
){
   std::size_t j;
   BbParaSolverPoolElementPtr she;

   she = heap[pos];
   while ( pos <= (heapSize/2) )
   {
      j = pos + pos;
      if( j < heapSize &&
            ( heap[j]->getBestDualBoundValue()
                  > heap[j+1]->getBestDualBoundValue() ) ) j++;
      if( she->getBestDualBoundValue()
            < heap[j]->getBestDualBoundValue()  )
         break;
      heap[pos] = heap[j];
      heap[pos]->setSelectionHeapElement(&heap[pos]);
      pos = j;
   }
   heap[pos] = she;
   heap[pos]->setSelectionHeapElement(&heap[pos]);
}

/** constructor of selection heap */
CollectingModeSolverHeap::CollectingModeSolverHeap(
      std::size_t size                               /**< maximum size of this heap */
      )
{
   maxHeapSize = size;
   heapSize = 0;
   heap = new BbParaSolverPoolElementPtr[size+1]; /* index 0 is a sentinel */
}

/** destructor of selection heap */
CollectingModeSolverHeap::~CollectingModeSolverHeap(
      )
{
   delete[] heap;
}
/** resize selection heap */
void
CollectingModeSolverHeap::resize(
      std::size_t size                               /**< new heap size */
      )
{
   BbParaSolverPoolElementPtr *tempHeap =
           new BbParaSolverPoolElementPtr[size+1]; /* index 0 is a sentinel */
   std::memcpy(tempHeap, heap, (unsigned long int)(sizeof(BbParaSolverPoolElementPtr) * (maxHeapSize+1)));
   delete[] heap;
   heap = tempHeap;
   maxHeapSize = size;
}

/** insert BbParaSolverPoolElementPtr to selection heap */
CollectingModeSolverHeap::ResultOfInsert
CollectingModeSolverHeap::insert(
      BbParaSolverPoolElementPtr inSolver      /**< BbParaSolverPoolElementPtr to be inserted */
      )
{
   if( heapSize >= maxHeapSize ) return FAILED_BY_FULL;
   heap[++heapSize] = inSolver;
   upHeap(heapSize);
   return SUCCEEDED;
}

/** remove the top priority element form selection heap */
BbParaSolverPoolElementPtr
CollectingModeSolverHeap::remove(
      )
{
   BbParaSolverPoolElementPtr solver = heap[1];

   heap[1] = heap[heapSize];
   heapSize--;
   downHeap(1);
   if( heapSize == 0 )
   {
       heap[1] = 0;
   }
   solver->setCollectingModeSolverHeapElement(0);
   return solver;
}


/** stringfy selection heap */
const std::string
CollectingModeSolverHeap::toString(
      )
{
   std::ostringstream os;
   os << "--- selection heap ---" << std::endl;
   os << "maxHeapSize: " << maxHeapSize << std::endl;
   os << "heapSize: " << heapSize << std::endl;
   for( std::size_t i = 1; i <= heapSize; i++ )
   {
      os << "heap[" << i << "]->rank: "
         << heap[i]->getRank() << std::endl
         << "heap[" << i << "]->status: "
         << static_cast<int>(heap[i]->getStatus()) << std::endl
         << "heap[" << i << "]->bestBound: "
         << heap[i]->getBestDualBoundValue() << std::endl
         << "heap[" << i << "]->numOfNodesLeft: "
         << heap[i]->getNumOfNodesLeft()  << std::endl
         << "heap[" << i << "]->numOfDiff: "
         << heap[i]->getNumOfDiffNodesLeft() <<  std::endl
         << "heap[" << i << "]->collectingMode: "
         << heap[i]->isInCollectingMode()  << std::endl;
   }
   return os.str();
}

/** constructor of DescendingCollectingModeSolverHeap */
DescendingCollectingModeSolverHeap::DescendingCollectingModeSolverHeap(
      std::size_t size                                 /**< maximum size of this heap */
      )
      : CollectingModeSolverHeap(size)
{
}

/** update dual bound value of the solver in heap */
void
DescendingCollectingModeSolverHeap::updateDualBoundValue(
      BbParaSolverPoolElementPtr solver,         /**< pointer to solver pool element whose dual bound is updated */
      double               newDualBoundValue   /**< new dual bound value */
      )
{
   int pos = solver->getCollectingModeSolverHeapElement() - heap;
   if( solver->getBestDualBoundValue() < newDualBoundValue )
   {
      solver->setBestDualBoundValue(newDualBoundValue);
      upHeap(pos);
   }
   else
   {
      solver->setBestDualBoundValue(newDualBoundValue);
      downHeap(pos);
   }
}

/** delete BbParaSolverPoolElement */
void
DescendingCollectingModeSolverHeap::deleteElement(
      BbParaSolverPoolElementPtr solver        /**< BbParaSolverPoolElement to be deleted */
      )
{
   std::size_t pos = (solver->getCollectingModeSolverHeapElement()) - heap;

   if( pos == heapSize )
   {
      /* no need to rearrange heap element */
      heap[heapSize--] = 0;
   }
   else
   {
      if( heap[pos]->getBestDualBoundValue() < heap[heapSize]->getBestDualBoundValue() )
      {
         heap[pos] = heap[heapSize];
         heap[heapSize--] = 0;
         upHeap(pos);
      }
      else
      {
         heap[pos] = heap[heapSize];
         heap[heapSize--] = 0;
         downHeap(pos);
      }
   }
   solver->setCollectingModeSolverHeapElement(0);
}


/** up heap */
void
DescendingCollectingModeSolverHeap::upHeap(
   std::size_t pos                                      /**< up heap this position element */
   )
{
   BbParaSolverPoolElementPtr she;

   she = heap[pos];
   heap[0] = NULL;
   while ( heap[pos/2] != NULL &&
         ( heap[pos/2]->getBestDualBoundValue()
               < she->getBestDualBoundValue() ) )
   {
      heap[pos] = heap[pos/2];
      heap[pos]->setCollectingModeSolverHeapElement(&heap[pos]);
      pos = pos/2;
   }
   heap[pos] = she;
   heap[pos]->setCollectingModeSolverHeapElement(&heap[pos]);
}

/** down heap */
void
DescendingCollectingModeSolverHeap::downHeap(
   std::size_t pos                                       /**< down heap this position element */
   )
{
   std::size_t j;
   BbParaSolverPoolElementPtr she;

   she = heap[pos];
   while ( pos <= (heapSize/2) )
   {
      j = pos + pos;
      if( j < heapSize &&
            ( heap[j]->getBestDualBoundValue()
                  < heap[j+1]->getBestDualBoundValue() ) ) j++;
      if( she->getBestDualBoundValue()
            > heap[j]->getBestDualBoundValue() )
         break;
      heap[pos] = heap[j];
      heap[pos]->setCollectingModeSolverHeapElement(&heap[pos]);
      pos = j;
   }
   heap[pos] = she;
   heap[pos]->setCollectingModeSolverHeapElement(&heap[pos]);
}

/** constructor of AscendingCollectingModeSolverHeap */
AscendingCollectingModeSolverHeap::AscendingCollectingModeSolverHeap(
      std::size_t size                                   /**< maximum size of this heap */
      )
      : CollectingModeSolverHeap(size)
{
}

/** update dual bound value of the solver in heap */
void
AscendingCollectingModeSolverHeap::updateDualBoundValue(
      BbParaSolverPoolElementPtr solver,           /**< pointer to solver pool element whose dual bound is updated */
      double newDualBoundValue                   /**< new dual bound value */
      )
{
   int pos = solver->getCollectingModeSolverHeapElement() - heap;
   if( solver->getBestDualBoundValue() > newDualBoundValue )
   {
      solver->setBestDualBoundValue(newDualBoundValue);
      upHeap(pos);
   }
   else
   {
      solver->setBestDualBoundValue(newDualBoundValue);
      downHeap(pos);
   }
}

/** delete BbParaSolverPoolElement */
void
AscendingCollectingModeSolverHeap::deleteElement(
      BbParaSolverPoolElementPtr solver        /**< BbParaSolverPoolElement to be deleted */
      )
{
   std::size_t pos = (solver->getCollectingModeSolverHeapElement()) - heap;

   if( pos == heapSize )
   {
      /* no need to rearrange heap element */
      heap[heapSize--] = 0;
   }
   else
   {
      if( heap[pos]->getBestDualBoundValue() > heap[heapSize]->getBestDualBoundValue() )
      {
         heap[pos] = heap[heapSize];
         heap[heapSize--] = 0;
         upHeap(pos);
      }
      else
      {
         heap[pos] = heap[heapSize];
         heap[heapSize--] = 0;
         downHeap(pos);
      }
   }
   solver->setCollectingModeSolverHeapElement(0);
}

/** up heap */
void
AscendingCollectingModeSolverHeap::upHeap(
   std::size_t pos                                        /**< up heap this position element */
){
   BbParaSolverPoolElementPtr she;

   she = heap[pos];
   heap[0] = NULL;

   while ( heap[pos/2] != NULL &&
         ( heap[pos/2]->getBestDualBoundValue()
               > she->getBestDualBoundValue() ) )
   {
      heap[pos] = heap[pos/2];
      heap[pos]->setCollectingModeSolverHeapElement(&heap[pos]);
      pos = pos/2;
   }
   heap[pos] = she;
   heap[pos]->setCollectingModeSolverHeapElement(&heap[pos]);
}

/** down heap */
void
AscendingCollectingModeSolverHeap::downHeap(
   std::size_t pos                                       /**< down heap this position element */
){
   std::size_t j;
   BbParaSolverPoolElementPtr she;

   she = heap[pos];
   while ( pos <= (heapSize/2) )
   {
      j = pos + pos;
      if( j < heapSize &&
            ( heap[j]->getBestDualBoundValue()
                  > heap[j+1]->getBestDualBoundValue() ) ) j++;
      if( she->getBestDualBoundValue()
            < heap[j]->getBestDualBoundValue()  )
         break;
      heap[pos] = heap[j];
      heap[pos]->setCollectingModeSolverHeapElement(&heap[pos]);
      pos = j;
   }
   heap[pos] = she;
   heap[pos]->setCollectingModeSolverHeapElement(&heap[pos]);
}

/** activate Solver specified by its rank */
void
BbParaSolverPool::activateSolver(
      int rank,                                 ///< rank of the solver to be activated
      BbParaNode *paraNode,                     ///< paraNode to be solved in the solver
      int nGoodNodesInNodePool,
      double averageDualBoundGain
      )
{

   active = true;                               // activate this solver pool

   if( rank < originRank || rank >= (signed)(originRank + nSolvers)  )
   {
      THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
   }
   map<int, BbParaSolverPoolElementPtr>::iterator p;
   if( pool[SOLVER_POOL_INDEX( rank )]->getStatus() == Inactive )
   {
      p = inactiveSolvers.find(rank);
      if( p != inactiveSolvers.end() )
      {
         if( p->second->getRank() != rank ||
               pool[SOLVER_POOL_INDEX( rank )]->getRank() != rank )
         {
            THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
         }
         inactiveSolvers.erase(p);
      }
      else
      {
         THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
      }
   }
   else
   {
      THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
   }
   activeSolvers.insert(make_pair(rank,pool[SOLVER_POOL_INDEX( rank )]));
   pool[SOLVER_POOL_INDEX( rank )]->activate(paraNode);
   if( paraParams->getBoolParamValue(ControlCollectingModeOnSolverSide) )
   {
      pool[SOLVER_POOL_INDEX( rank )]->prohibitCollectingMode();
   }
   nNodesInSolvers++;
   selectionHeap->insert(pool[SOLVER_POOL_INDEX( rank )]);  // this should be called after activate: dual bound value need to be set
   if( paraParams->getBoolParamValue(DualBoundGainTest) )
   {
      if( ( (nDualBoundGainTesting*2) + nGoodNodesInNodePool ) < getNumInactiveSolvers() +
               paraParams->getIntParamValue(NChangeIntoCollectingMode)*mCollectingNodes*paraParams->getRealParamValue(MultiplierForCollectingMode) )  // could be doubled
      {
         //if( beforeInitialCollect )
         // {
         //   averageDualBoundGain*=2;
         //}
         PARA_COMM_CALL(
               paraComm->send(&averageDualBoundGain, 1, ParaDOUBLE, rank, TagTestDualBoundGain )
         );
         pool[SOLVER_POOL_INDEX( rank )]->setDualBoundGainTesting(true);
         nDualBoundGainTesting++;
         // std::cout << "S." << rank << " Test dual bound gain" << std::endl;
      }
      else
      {
         PARA_COMM_CALL(
               paraComm->send(NULL, 0, ParaBYTE, rank, TagNoTestDualBoundGain )
         );
         // std::cout << "S." << rank << " No test dual bound gain" << std::endl;
      }
   }
   paraNode->send(paraComm, rank);
}

/** activate Solver specified by its rank and number of nodes left. This is for the racing winner. */
void
BbParaSolverPool::activateSolver(
      int rank,                                 ///< rank of the solver to be activated */
      BbParaNode *paraNode,                     ///< paraNode to be solved in the solver */
      int nNodesLeft                            ///< number of nodes left in the Solver */
      )
{

   active = true;                               // activate this solver pool

   if( rank < originRank || rank >= (signed)(originRank + nSolvers)  )
   {
      THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
   }
   map<int, BbParaSolverPoolElementPtr>::iterator p;
   if( pool[SOLVER_POOL_INDEX( rank )]->getStatus() == Inactive )
   {
      p = inactiveSolvers.find(rank);
      if( p != inactiveSolvers.end() )
      {
         if( p->second->getRank() != rank ||
               pool[SOLVER_POOL_INDEX( rank )]->getRank() != rank )
         {
            THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
         }
         inactiveSolvers.erase(p);
      }
      else
      {
         THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
      }
   }
   else
   {
      THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
   }
   activeSolvers.insert(make_pair(rank,pool[SOLVER_POOL_INDEX( rank )]));
   pool[SOLVER_POOL_INDEX( rank )]->activate(paraNode);
   pool[SOLVER_POOL_INDEX( rank )]->setNumOfNodesLeft(nNodesLeft);
   /** DO NOT PROHIBIT COLLECTING MODE. THIS IS ACTIVATION ROUTIN FOR RACING WIINGER */
   nNodesInSolvers += nNodesLeft;
   selectionHeap->insert(pool[SOLVER_POOL_INDEX( rank )]);  // this should be called after activate: dual bound value need to be set
}


/** activate any Solver which is idle */
int
BbParaSolverPool::activateSolver(
      BbParaNode *paraNode,                        ///< paraNode to be solved in an activated solver
      BbParaRacingSolverPool *paraRacingSolverPool,///< paraRacingSolverPool to check if the solver is not solving root node
      bool     rampUpPhase,
      int nGoodNodesInNodePool,
      double averageDualBoundGain
      )
{
   map<int, BbParaSolverPoolElementPtr>::iterator p;
   int rank;

   active = true;   // activate this solver pool

   if( !paraRacingSolverPool )
   {
      p = inactiveSolvers.begin();
      if( p == inactiveSolvers.end() )
      {
         THROW_LOGICAL_ERROR1("No inactive Solver");
      }
      rank = p->second->getRank();
   }
   else
   {
      for( p = inactiveSolvers.begin(); p != inactiveSolvers.end(); ++p )
      {
         if( !paraRacingSolverPool->isSolverActive(p->second->getRank()) )
         {
            // at least one notification message is received
            rank =  p->second->getRank();
            break;
         }
      }
      if( p == inactiveSolvers.end() )
      {
         return -1;
      }
   }
   activeSolvers.insert(make_pair(rank,pool[SOLVER_POOL_INDEX( rank )]));
   pool[SOLVER_POOL_INDEX( rank )]->activate(paraNode);
   if( paraParams->getBoolParamValue(ControlCollectingModeOnSolverSide) )
   {
      pool[SOLVER_POOL_INDEX( rank )]->prohibitCollectingMode();
   }
   nNodesInSolvers++;
   selectionHeap->insert(pool[SOLVER_POOL_INDEX( rank )]);  // this should be called after activate: dual bound value need to be set
   inactiveSolvers.erase(p);
   if( paraParams->getBoolParamValue(LightWeightRootNodeProcess) &&
         (!rampUpPhase) && (!paraRacingSolverPool ) &&      // already ramp-up ( and no racing solvers )
         inactiveSolvers.size() >
            ( nSolvers * paraParams->getRealParamValue(RatioToApplyLightWeightRootProcess) )
                                                            // specified idle solver exists
         )
   {
      PARA_COMM_CALL(
            paraComm->send(NULL, 0, ParaBYTE, rank, TagLightWeightRootNodeProcess)
      );
   }
   if( paraParams->getBoolParamValue(DualBoundGainTest) )
   {
      if( ( (nDualBoundGainTesting*2) + nGoodNodesInNodePool ) < getNumInactiveSolvers() +
               paraParams->getIntParamValue(NChangeIntoCollectingMode)*mCollectingNodes*paraParams->getRealParamValue(MultiplierForCollectingMode) )  // could be doubled
      {
         //if( beforeInitialCollect )
         // {
         //   averageDualBoundGain*=2;
         //}
         PARA_COMM_CALL(
               paraComm->send(&averageDualBoundGain, 1, ParaDOUBLE, rank, TagTestDualBoundGain )
         );
         pool[SOLVER_POOL_INDEX( rank )]->setDualBoundGainTesting(true);
         nDualBoundGainTesting++;
         // std::cout << "S." << rank << " Test dual bound gain" << std::endl;
      }
      else
      {
         PARA_COMM_CALL(
               paraComm->send(NULL, 0, ParaBYTE, rank, TagNoTestDualBoundGain )
         );
         // std::cout << "S." << rank << " No test dual bound gain" << std::endl;
      }
   }
   paraNode->send(paraComm, rank);
   assert( ( paraNode->getMergingStatus() != 0 && ( !paraNode->getMergeNodeInfo() ) ) ||
         ( paraNode->getMergingStatus() == 0 && paraNode->getMergeNodeInfo() ) );
   return rank;
}

void
BbParaSolverPool::addNewSubtreeRootNode(
      int rank,                                 ///< rank of the solver, in which the subtree root node is added
      BbParaNode *paraNode                      ///< paraNode to be added as the subtree root
      )
{
   if( rank < originRank || rank >= (signed)(originRank + nSolvers)  )
   {
      THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
   }
   map<int, BbParaSolverPoolElementPtr>::iterator p;
   if ( pool[SOLVER_POOL_INDEX( rank )]->getStatus() == Active )
   {
      p = activeSolvers.find(rank);
      if( p != activeSolvers.end() )
      {
         if( p->second->getRank() != rank ||
               pool[SOLVER_POOL_INDEX( rank )]->getRank() != rank )
         {
            THROW_LOGICAL_ERROR2("Rank = ", rank);
         }
         if( collectingMode || p->second->isCandidateOfCollecting() )
         {
            THROW_LOGICAL_ERROR3("Rank = ", rank, " should not be in colleting mode and shuould not be a candidate of collecting.");
         }
         p->second->addSubtreeRoot(paraNode);
         // std::cout << "ADD: " << paraNode->toString() << std::endl;
      }
      else
      {
         THROW_LOGICAL_ERROR2("Rank = ", rank);
      }
   }
   else
   {
      if( pool[SOLVER_POOL_INDEX( rank )]->getStatus() != InterruptRequested &&
          pool[SOLVER_POOL_INDEX( rank )]->getStatus() != TerminateRequested && 
          pool[SOLVER_POOL_INDEX( rank )]->getStatus() != Terminated )
      {
	 std::cout << "Status = " << static_cast<int>(pool[SOLVER_POOL_INDEX( rank )]->getStatus()) << std::endl;
         THROW_LOGICAL_ERROR2("Rank = ", rank);
      }
   }
}

void
BbParaSolverPool::makeSubtreeRootNodeCurrent(
      int rank,                                 ///< rank of the solver, in which the subtree root node is removed
      BbParaNode *paraNode                      ///< paraNode to be removed
      )
{
   if( rank < originRank || rank >= (signed)(originRank + nSolvers)  )
   {
      THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
   }
   map<int, BbParaSolverPoolElementPtr>::iterator p;
   if ( pool[SOLVER_POOL_INDEX( rank )]->getStatus() == Active )
   {
      p = activeSolvers.find(rank);
      if( p != activeSolvers.end() )
      {
         if( p->second->getRank() != rank ||
               pool[SOLVER_POOL_INDEX( rank )]->getRank() != rank )
         {
            THROW_LOGICAL_ERROR2("Rank = ", rank);
         }

         // if( collectingMode || p->second->isCandidateOfCollecting() )
         // {
         //    THROW_LOGICAL_ERROR3("Rank = ", rank, " should not be in colleting mode and shuould not be a candidate of collecting.");
         // }
         // std::cout << "REMOVE: " << paraNode->toString() << std::endl;
         p->second->makeSubtreeRootCurrent(paraNode);
      }
      else
      {
         THROW_LOGICAL_ERROR2("Rank = ", rank);
      }
   }
   else
   {
      if( pool[SOLVER_POOL_INDEX( rank )]->getStatus() != InterruptRequested &&
          pool[SOLVER_POOL_INDEX( rank )]->getStatus() != TerminateRequested && 
          pool[SOLVER_POOL_INDEX( rank )]->getStatus() != Terminated )
      {
	 std::cout << "Status = " << static_cast<int>(pool[SOLVER_POOL_INDEX( rank )]->getStatus()) << std::endl;
         THROW_LOGICAL_ERROR2("Rank = ", rank);
      }
   }
}

void
BbParaSolverPool::removeSubtreeRootNode(
      int rank,                                 ///< rank of the solver, in which the subtree root node is removed
      BbParaNode *paraNode                      ///< paraNode to be removed
      )
{
   if( rank < originRank || rank >= (signed)(originRank + nSolvers)  )
   {
      THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
   }
   map<int, BbParaSolverPoolElementPtr>::iterator p;
   if ( pool[SOLVER_POOL_INDEX( rank )]->getStatus() == Active )
   {
      p = activeSolvers.find(rank);
      if( p != activeSolvers.end() )
      {
         if( p->second->getRank() != rank ||
               pool[SOLVER_POOL_INDEX( rank )]->getRank() != rank )
         {
            THROW_LOGICAL_ERROR2("Rank = ", rank);
         }

         // if( collectingMode || p->second->isCandidateOfCollecting() )
         // {
         //    THROW_LOGICAL_ERROR3("Rank = ", rank, " should not be in colleting mode and shuould not be a candidate of collecting.");
         // }
         // std::cout << "REMOVE: " << paraNode->toString() << std::endl;
         p->second->removeSubtreeRoot(paraNode);
      }
      else
      {
         THROW_LOGICAL_ERROR2("Rank = ", rank);
      }
   }
   else
   {
      if( pool[SOLVER_POOL_INDEX( rank )]->getStatus() != InterruptRequested &&
          pool[SOLVER_POOL_INDEX( rank )]->getStatus() != TerminateRequested && 
          pool[SOLVER_POOL_INDEX( rank )]->getStatus() != Terminated )
      {
	 std::cout << "Status = " << static_cast<int>(pool[SOLVER_POOL_INDEX( rank )]->getStatus()) << std::endl;
         THROW_LOGICAL_ERROR2("Rank = ", rank);
      }
   }
}

//BbParaNode *
//BbParaSolverPool::removeAllSubtreeRootNode(
//      int rank,                                 ///< rank of the solver, in which the subtree root node is removed
//      BbParaNode *paraNode                      ///< paraNode to be removed
//      )
//{
//   if( rank < originRank || rank >= (signed)(originRank + nSolvers)  )
//   {
//      THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
//   }
//   map<int, BbParaSolverPoolElementPtr>::iterator p;
//   if ( pool[SOLVER_POOL_INDEX( rank )]->getStatus() == Active )
//   {
//      p = activeSolvers.find(rank);
//      if( p != activeSolvers.end() )
//      {
//         if( p->second->getRank() != rank ||
//               pool[SOLVER_POOL_INDEX( rank )]->getRank() != rank )
//         {
//            THROW_LOGICAL_ERROR2("Rank = ", rank);
//         }
//
//         // if( collectingMode || p->second->isCandidateOfCollecting() )
//         // {
//         //    THROW_LOGICAL_ERROR3("Rank = ", rank, " should not be in colleting mode and shuould not be a candidate of collecting.");
//         // }
//         // std::cout << "REMOVE: " << paraNode->toString() << std::endl;
//         p->second->removeSubtreeRoot(paraNode);
//      }
//      else
//      {
//         THROW_LOGICAL_ERROR2("Rank = ", rank);
//      }
//   }
//   else
//   {
//      THROW_LOGICAL_ERROR2("Rank = ", rank);
//   }
//   BbParaNode *node = pool[SOLVER_POOL_INDEX(rank)]->extractCurrentNodeGeneratedBySelfSplit();
//   return node;
//}

BbParaNode *
BbParaSolverPool::extractSelfSplitSubtreeRootNode(
      int rank,                                 ///< rank of the solver, in which the subtree root node is removed
      BbParaNode *paraNode                      ///< paraNode to be removed
      )
{
   if( rank < originRank || rank >= (signed)(originRank + nSolvers)  )
   {
      THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
   }
   map<int, BbParaSolverPoolElementPtr>::iterator p;
   if ( pool[SOLVER_POOL_INDEX( rank )]->getStatus() == Active )
   {
      p = activeSolvers.find(rank);
      if( p != activeSolvers.end() )
      {
         if( p->second->getRank() != rank ||
               pool[SOLVER_POOL_INDEX( rank )]->getRank() != rank )
         {
            THROW_LOGICAL_ERROR2("Rank = ", rank);
         }

         // if( collectingMode || p->second->isCandidateOfCollecting() )
         // {
         //    THROW_LOGICAL_ERROR3("Rank = ", rank, " should not be in colleting mode and shuould not be a candidate of collecting.");
         // }

         BbParaNode *node = p->second->extractSubtreeRoot(paraNode);
         node->next = 0;
         // std::cout << "EXTRACT: " << node->toString() << std::endl;
         return node;
      }
      else
      {
         THROW_LOGICAL_ERROR2("Rank = ", rank);
      }
   }
   else
   {
      THROW_LOGICAL_ERROR2("Rank = ", rank);
   }
}

BbParaNode *
BbParaSolverPool::getSelfSplitSubtreeRootNodes(
      int rank                                 ///< rank of the solver, in which the subtree root node is removed
      )
{
   if( rank < originRank || rank >= (signed)(originRank + nSolvers)  )
   {
      THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
   }
   map<int, BbParaSolverPoolElementPtr>::iterator p;
   if ( pool[SOLVER_POOL_INDEX( rank )]->getStatus() == Active )
   {
      p = activeSolvers.find(rank);
      if( p != activeSolvers.end() )
      {
         if( p->second->getRank() != rank ||
               pool[SOLVER_POOL_INDEX( rank )]->getRank() != rank )
         {
            THROW_LOGICAL_ERROR2("Rank = ", rank);
         }

         // if( collectingMode || p->second->isCandidateOfCollecting() )
         // {
         //    THROW_LOGICAL_ERROR3("Rank = ", rank, " should not be in colleting mode and shuould not be a candidate of collecting.");
         // }
         return p->second->getSelfSplitNodes();
      }
      else
      {
         THROW_LOGICAL_ERROR2("Rank = ", rank);
      }
   }
   else
   {
      THROW_LOGICAL_ERROR2("Rank = ", rank);
   }
}

BbParaNode *
BbParaSolverPool::extractSelfSplitSubtreeRootNodes(
      int rank                                 ///< rank of the solver, in which the subtree root node is removed
      )
{
   if( rank < originRank || rank >= (signed)(originRank + nSolvers)  )
   {
      THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
   }
   return pool[SOLVER_POOL_INDEX( rank )]->extractSelfSplitNodes();
}

void
BbParaSolverPool::deleteCurrentSubtreeRootNode(
      int rank                                 ///< rank of the solver, in which the subtree root node is removed
      )
{
   if( rank < originRank || rank >= (signed)(originRank + nSolvers)  )
   {
      THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
   }
   map<int, BbParaSolverPoolElementPtr>::iterator p;
   if ( pool[SOLVER_POOL_INDEX( rank )]->getStatus() == Active )
   {
      p = activeSolvers.find(rank);
      if( p != activeSolvers.end() )
      {
         if( p->second->getRank() != rank ||
               pool[SOLVER_POOL_INDEX( rank )]->getRank() != rank )
         {
            THROW_LOGICAL_ERROR2("Rank = ", rank);
         }

         // if( collectingMode || p->second->isCandidateOfCollecting() )
         // {
         //    THROW_LOGICAL_ERROR3("Rank = ", rank, " should not be in colleting mode and shuould not be a candidate of collecting.");
         // }
         return p->second->deleteCurrentNode();
      }
      else
      {
         THROW_LOGICAL_ERROR2("Rank = ", rank);
      }
   }
   else
   {
      if( pool[SOLVER_POOL_INDEX( rank )]->getStatus() != InterruptRequested &&
          pool[SOLVER_POOL_INDEX( rank )]->getStatus() != TerminateRequested && 
          pool[SOLVER_POOL_INDEX( rank )]->getStatus() != Terminated )
      {
	 std::cout << "Status = " << static_cast<int>(pool[SOLVER_POOL_INDEX( rank )]->getStatus()) << std::endl;
         THROW_LOGICAL_ERROR2("Rank = ", rank);
      }
   }
}

/** inactivate the Solver specified by rank */
void
BbParaSolverPool::inactivateSolver(
      int rank,                                 ///< rank of the solver to be inactivated
      long long numOfNodesSolved,               ///< number of nodes solved
      BbParaNodePool *paraNodePool              ///< pointer to ParaNodePool to change into collecting mode
      )
{
   if( rank < originRank || rank >= (signed)(originRank + nSolvers)  )
   {
      THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
   }
   // if( breakingFirstSubtree && rank == 1 && pool[SOLVER_POOL_INDEX(rank)]->isInCollectingMode() )
   if( breakingFirstSubtree && rank == 1 )
   {
      breakingFirstSubtree = false;
   }
   sendSwitchOutCollectingModeIfNecessary( rank );
   map<int, BbParaSolverPoolElementPtr>::iterator p;
   if ( pool[SOLVER_POOL_INDEX( rank )]->getStatus() == Active ||
        pool[SOLVER_POOL_INDEX( rank )]->getStatus() == InterruptRequested ||
        pool[SOLVER_POOL_INDEX( rank )]->getStatus() == TerminateRequested )
   {
      p = activeSolvers.find(rank);
      if( p != activeSolvers.end() )
      {
         if( p->second->getRank() != rank ||
               pool[SOLVER_POOL_INDEX( rank )]->getRank() != rank )
         {
            THROW_LOGICAL_ERROR2("Rank = ", rank);
         }
         if( numOfNodesSolved >= 0 )
         {
            nNodesSolvedInSolvers += numOfNodesSolved - p->second->getNumOfNodesSolved();
            // if(nNodesSolvedInSolvers > 999) abort();
         }
         nNodesInSolvers -= p->second->getNumOfNodesLeft();
         if( collectingMode && !pool[SOLVER_POOL_INDEX(rank)]->isOutCollectingMode() )
         {
            // Don't have to send this message. Solver should be initialized with out of collecting mode
            //
            // PARA_COMM_CALL(
            //      paraComm->send(NULL, 0, ParaBYTE, p->second->getRank(), TagOutCollectingMode)
            //      );
            //
            // Should be initialized
            // pool[SOLVER_POOL_INDEX(rank)]->setCollectingMode(false);
            nCollectingModeSolvers--;
            if( paraNodePool && 
                  selectionHeap->top()->getNumOfNodesLeft() > 1 &&
                  selectionHeap->top()->isGenerator() &&
                  (!selectionHeap->top()->isCollectingProhibited()) &&
                  selectionHeap->top()->isOutCollectingMode() )
            {
               //
               // the solver with the best dual bound should always in collecting mode first!
               //
               switchInCollectingToSolver(selectionHeap->top()->getRank(), paraNodePool);
            }
         }
         if( p->second->isCandidateOfCollecting() )
         {
            std::multimap< double, BbParaSolverPoolElementPtr >::iterator pCandidate;
            for( pCandidate = candidatesOfCollectingModeSolvers.begin();
                  pCandidate != candidatesOfCollectingModeSolvers.end(); ++pCandidate )
            {
               if( pCandidate->second->getRank() == rank )
                  break;
            }
            assert( pCandidate != candidatesOfCollectingModeSolvers.end() );
            pCandidate->second->setCandidateOfCollecting(false);
            candidatesOfCollectingModeSolvers.erase(pCandidate);
         }
         activeSolvers.erase(p);
      }
      else
      {
         THROW_LOGICAL_ERROR2("Rank = ", rank);
      }
   }
   else
   {
      THROW_LOGICAL_ERROR2("Rank = ", rank);
   }
   selectionHeap->deleteElement(pool[SOLVER_POOL_INDEX( rank )]);    /** need to have dual bound, so should be before inactivate() */
   if( collectingModeSolverHeap && pool[SOLVER_POOL_INDEX(rank)]->isInCollectingMode() )
   {
      collectingModeSolverHeap->deleteElement(pool[SOLVER_POOL_INDEX(rank)]);
   }
   pool[SOLVER_POOL_INDEX( rank )]->inactivate();
   if( paraParams->getBoolParamValue(ControlCollectingModeOnSolverSide) )
   {
      pool[SOLVER_POOL_INDEX( rank )]->setCollectingIsAllowed();
   }
   inactiveSolvers.insert(make_pair(rank,pool[SOLVER_POOL_INDEX( rank )]));
}

/** reset counters in the Solver specified by rank */
void
BbParaSolverPool::resetCountersInSolver(
      int rank,                                 ///< rank of the solver to be inactivated
      long long numOfNodesSolved,               ///< number of nodes solved
      int numOfSelfSplitNodesLeft,              ///< number of self-split nodes left
      BbParaNodePool *paraNodePool              ///< pointer to ParaNodePool to change into collecting mode
      )
{
   if( rank < originRank || rank >= (signed)(originRank + nSolvers)  )
   {
      THROW_LOGICAL_ERROR2("Invalid rank. Rank = ", rank);
   }
   // if( breakingFirstSubtree && rank == 1 && pool[SOLVER_POOL_INDEX(rank)]->isInCollectingMode() )
   if( breakingFirstSubtree && rank == 1 )
   {
      breakingFirstSubtree = false;
   }
   sendSwitchOutCollectingModeIfNecessary( rank );
   map<int, BbParaSolverPoolElementPtr>::iterator p;
   if ( pool[SOLVER_POOL_INDEX( rank )]->getStatus() == Active )
   {
      p = activeSolvers.find(rank);
      if( p != activeSolvers.end() )
      {
         if( p->second->getRank() != rank ||
               pool[SOLVER_POOL_INDEX( rank )]->getRank() != rank )
         {
            THROW_LOGICAL_ERROR2("Rank = ", rank);
         }
         nNodesSolvedInSolvers += numOfNodesSolved - p->second->getNumOfNodesSolved();
         // if(nNodesSolvedInSolvers > 999) abort();
         nNodesInSolvers -= p->second->getNumOfNodesLeft();
         if( collectingMode && !pool[SOLVER_POOL_INDEX(rank)]->isOutCollectingMode() )
         {
            // Don't have to send this message. Solver should be initialized with out of collecting mode
            //
            // PARA_COMM_CALL(
            //      paraComm->send(NULL, 0, ParaBYTE, p->second->getRank(), TagOutCollectingMode)
            //      );
            //
            // Should be initialized
            // pool[SOLVER_POOL_INDEX(rank)]->setCollectingMode(false);
            nCollectingModeSolvers--;
            if( selectionHeap->top()->getNumOfNodesLeft() > 1 &&
                  selectionHeap->top()->isGenerator() &&
                  (!selectionHeap->top()->isCollectingProhibited()) &&
                  selectionHeap->top()->isOutCollectingMode() )
            {
               //
               // the solver with the best dual bound should always in collecting mode first!
               //
               switchInCollectingToSolver(selectionHeap->top()->getRank(), paraNodePool);
            }
         }
         if( p->second->isCandidateOfCollecting() )
         {
            std::multimap< double, BbParaSolverPoolElementPtr >::iterator pCandidate;
            for( pCandidate = candidatesOfCollectingModeSolvers.begin();
                  pCandidate != candidatesOfCollectingModeSolvers.end(); ++pCandidate )
            {
               if( pCandidate->second->getRank() == rank )
                  break;
            }
            assert( pCandidate != candidatesOfCollectingModeSolvers.end() );
            pCandidate->second->setCandidateOfCollecting(false);
            candidatesOfCollectingModeSolvers.erase(pCandidate);
         }
         activeSolvers.erase(p);
      }
      else
      {
         THROW_LOGICAL_ERROR2("Rank = ", rank);
      }
   }
   else
   {
      THROW_LOGICAL_ERROR2("Rank = ", rank);
   }
   // selectionHeap->deleteElement(pool[SOLVER_POOL_INDEX( rank )]);    /** need to have dual bound, so should be before inactivate() */
   if( collectingModeSolverHeap && pool[SOLVER_POOL_INDEX(rank)]->isInCollectingMode() )
   {
      collectingModeSolverHeap->deleteElement(pool[SOLVER_POOL_INDEX(rank)]);
   }
   BbParaNode *node = pool[SOLVER_POOL_INDEX( rank )]->extractCurrentNode();
   // assert( node );
   selectionHeap->deleteElement(pool[SOLVER_POOL_INDEX( rank )]);    /** need to have dual bound, so should be before inactivate() */
   if( collectingModeSolverHeap && pool[SOLVER_POOL_INDEX(rank)]->isInCollectingMode() )
   {
      collectingModeSolverHeap->deleteElement(pool[SOLVER_POOL_INDEX(rank)]);
   }
   pool[SOLVER_POOL_INDEX( rank )]->inactivate();
   // pool[SOLVER_POOL_INDEX( rank )]->activate(node);
   if( paraParams->getBoolParamValue(ControlCollectingModeOnSolverSide) )
   {
      pool[SOLVER_POOL_INDEX( rank )]->setCollectingIsAllowed();
   }
   inactiveSolvers.insert(make_pair(rank,pool[SOLVER_POOL_INDEX( rank )]));
   if( node )
   {
      activateSolver(rank, node, numOfSelfSplitNodesLeft);
   }
}

/** solver specified by rank died */
BbParaNode *
BbParaSolverPool::solverDied(
      int rank                                 /**< rank of the dead solver */
      )
{
   if( rank < originRank || rank >= (signed)(originRank + nSolvers)  )
   {
      THROW_LOGICAL_ERROR2("Rank = ", rank);
   }
   map<int, BbParaSolverPoolElementPtr>::iterator p;
   if ( pool[SOLVER_POOL_INDEX(rank)]->getStatus() == Active )
   {
      p = activeSolvers.find(rank);
      if( p != activeSolvers.end() )
      {
         if( p->second->getRank() != rank ||
               pool[SOLVER_POOL_INDEX(rank)]->getRank() != rank )
         {
            THROW_LOGICAL_ERROR2("Rank = ", rank);
         }
         nNodesSolvedInSolvers -= p->second->getNumOfNodesSolved();
         nNodesInSolvers -= p->second->getNumOfNodesLeft();
         activeSolvers.erase(p);
      }
      else
      {
         THROW_LOGICAL_ERROR2("Rank = ", rank);
      }
   }
   else
   {
      THROW_LOGICAL_ERROR2("Rank = ", rank);
   }
   selectionHeap->deleteElement(pool[SOLVER_POOL_INDEX(rank)]);       /** need to have dual bound, so should be before died() */
   if( collectingModeSolverHeap && pool[SOLVER_POOL_INDEX(rank)]->isInCollectingMode() )
   {
      collectingModeSolverHeap->deleteElement(pool[SOLVER_POOL_INDEX(rank)]);
   }
   deadSolvers.insert(make_pair(rank,pool[SOLVER_POOL_INDEX(rank)]));
   return  pool[SOLVER_POOL_INDEX(rank)]->died();
}

/** switch out collecting mode */
void
BbParaSolverPool::switchOutCollectingMode(
      )
{
   if( !collectingMode )
   {
      return;
   }
   if( activeSolvers.size() > 0 )
   {
      map<int, BbParaSolverPoolElementPtr>::iterator p;
      for( p = activeSolvers.begin(); p != activeSolvers.end(); ++p)
      {
         int rank = p->second->getRank();
         sendSwitchOutCollectingModeIfNecessary(rank);
         if ( pool[SOLVER_POOL_INDEX(rank)]->isCandidateOfCollecting() )
         {
            pool[SOLVER_POOL_INDEX(rank)]->setCandidateOfCollecting(false);
            /** clear candidatesOfCollectingModeSolvers in below */
         }
         if( paraParams->getBoolParamValue(DualBoundGainTest) &&
            pool[SOLVER_POOL_INDEX(rank)]->isDualBoundGainTesting() )
         {
            PARA_COMM_CALL(
                  paraComm->send(NULL, 0, ParaBYTE, rank, TagNoTestDualBoundGain )
            );
            pool[SOLVER_POOL_INDEX(rank)]->setDualBoundGainTesting(false);
            nDualBoundGainTesting--;
         } 
      }
   }
   assert(nCollectingModeSolvers == 0);
   nCollectingModeSolvers = 0;
   candidatesOfCollectingModeSolvers.clear();
   collectingMode = false;
   switchOutTime = paraTimer->getElapsedTime();
}

/** switch out collecting mode to the specified rank if it is necessary */
void
BbParaSolverPool::enforcedSwitchOutCollectingMode(
      int rank             /**< rank of the solver */
      )
{
   /** sending TagOutCollectingMode is harmless */
   PARA_COMM_CALL(
         paraComm->send( NULL, 0, ParaBYTE, rank, TagOutCollectingMode )
         );
   /** pool[SOLVER_POOL_INDEX(rank)]->isOutCollectingMode() should be true */
   if ( !pool[SOLVER_POOL_INDEX(rank)]->isOutCollectingMode() )
   {
      pool[SOLVER_POOL_INDEX(rank)]->setCollectingMode(false);
      if( collectingModeSolverHeap )
      {
         collectingModeSolverHeap->deleteElement(pool[SOLVER_POOL_INDEX(rank)]);
      }
      nCollectingModeSolvers--;
   }
}

/** switch out collecting mode to the specified rank */
void
BbParaSolverPool::sendSwitchOutCollectingModeIfNecessary(
      int rank
      )
{
   if ( !pool[SOLVER_POOL_INDEX(rank)]->isOutCollectingMode() )
   {
      PARA_COMM_CALL(
            paraComm->send( NULL, 0, ParaBYTE, rank, TagOutCollectingMode )
            );
      pool[SOLVER_POOL_INDEX(rank)]->setCollectingMode(false);
      if( collectingModeSolverHeap )
      {
         collectingModeSolverHeap->deleteElement(pool[SOLVER_POOL_INDEX(rank)]);
      }
      nCollectingModeSolvers--;
   }
}

/** send switch in-collecting mode to a specific solver */
void
BbParaSolverPool::switchInCollectingToSolver(
      int rank,
      BbParaNodePool *paraNodePool
      )
{
   // assert(nLimitCollectingModeSolvers > nCollectingModeSolvers);   ** this is not hold anymore **
   int nCollect = std::min(pool[SOLVER_POOL_INDEX(rank)]->getNumOfNodesLeft()/4,
         static_cast<int>(
         getNumInactiveSolvers() +
         paraParams->getIntParamValue(NChangeIntoCollectingMode)*mCollectingNodes*
           paraParams->getRealParamValue(MultiplierForCollectingMode)
           - (signed)paraNodePool->getNumOfGoodNodes(getGlobalBestDualBoundValue())));
   if( (signed)paraNodePool->getNumOfGoodNodes( getGlobalBestDualBoundValue()) <
         (paraParams->getIntParamValue(NChangeIntoCollectingMode)*mCollectingNodes)/4 )
   {
      nCollect = 0 - (nCollect + 1);    // indicate aggressive collecting
   }
   assert( pool[SOLVER_POOL_INDEX(rank)]->isGenerator() );
   assert( !pool[SOLVER_POOL_INDEX(rank)]->isCollectingProhibited() );
   if( paraNodePool->isEmpty() && nLimitCollectingModeSolvers > 10 )
   {
      PARA_COMM_CALL(
            paraComm->send(NULL, 0, ParaBYTE, rank, TagNoWaitModeSend )
      );
   }
   PARA_COMM_CALL(
         paraComm->send(&nCollect, 1, ParaINT, rank, TagInCollectingMode)
   );
   if( pool[SOLVER_POOL_INDEX(rank)]->isCandidateOfCollecting() )
   {
      std::multimap< double, BbParaSolverPoolElementPtr >::iterator pCandidate;
      for( pCandidate = candidatesOfCollectingModeSolvers.begin();
            pCandidate != candidatesOfCollectingModeSolvers.end(); ++pCandidate )
      {
         if( pCandidate->second->getRank() == rank )
            break;
      }
      assert( pCandidate != candidatesOfCollectingModeSolvers.end() );
      pCandidate->second->setCandidateOfCollecting(false);
      candidatesOfCollectingModeSolvers.erase(pCandidate);
   }
   pool[SOLVER_POOL_INDEX(rank)]->setCollectingMode(true);
   if( collectingModeSolverHeap )
   {
      collectingModeSolverHeap->insert( pool[SOLVER_POOL_INDEX(rank)] );
   }
   nCollectingModeSolvers++;
}

/** send switch in-collecting mode */
void
BbParaSolverPoolForMinimization::switchInCollectingMode(
      BbParaNodePool *paraNodePool
      )
{
   if( paraParams->getBoolParamValue(DualBoundGainTest) &&
         ( beforeInitialCollect || (!beforeInitialCollect && beforeFinishingFirstCollect) ) &&
         (signed)paraNodePool->getNumOfGoodNodes(getGlobalBestDualBoundValue()) >
         paraParams->getIntParamValue(NChangeIntoCollectingMode) * 0.1
         )
   {
      return;
   }
   if( paraParams->getRealParamValue(CollectingModeInterval) > 0.0 &&
         switchOutTime > 0 &&
         (paraTimer->getElapsedTime() - switchOutTime) <
         paraParams->getRealParamValue(CollectingModeInterval) &&
         mCollectingNodes < 100 )
   {
      if( candidatesOfCollectingModeSolvers.size() > 0 &&
            candidatesOfCollectingModeSolvers.begin()->second->getNumOfDiffNodesLeft() < 0 )
      {
         mCollectingNodes--;
         if( mCollectingNodes <= 0 ) mCollectingNodes = 1;
      }
      else
      {
         mCollectingNodes++;
         if( mCollectingNodes > mMaxCollectingNodes ) 
         {
            mMaxCollectingNodes = mCollectingNodes;
         }
      }
      switchOutTime = -1.0;
      // std::cout << paraTimer->getElapsedTime() << " mCollectingNodes = " << mCollectingNodes << std::endl;
   }
   int nCollect = static_cast<int>(
         getNumInactiveSolvers() +
         paraParams->getIntParamValue(NChangeIntoCollectingMode)*mCollectingNodes*
        paraParams->getRealParamValue(MultiplierForCollectingMode)
        - (signed)paraNodePool->getNumOfGoodNodes(getGlobalBestDualBoundValue()));
   map<int, BbParaSolverPoolElementPtr>::iterator p;
   if( activeSolvers.size() > 0 )
   {
      if( breakingFirstSubtree )
      {
         for( p = activeSolvers.begin();
               p != activeSolvers.end();
               ++p)
         {
            if( p->second->getRank() == 1 )
            {
               int rank = p->second->getRank();
               assert(pool[SOLVER_POOL_INDEX(rank)]->isGenerator());
               if( paraNodePool->isEmpty() && nLimitCollectingModeSolvers > 10 )
               {
                  PARA_COMM_CALL(
                        paraComm->send(NULL, 0, ParaBYTE, rank, TagNoWaitModeSend )
                  );
               }
               PARA_COMM_CALL(
                     paraComm->send(&nCollect, 1, ParaINT, rank, TagInCollectingMode )
               );
               nCollect = 0;
               pool[SOLVER_POOL_INDEX(rank)]->setCollectingMode(true);
               if( collectingModeSolverHeap )
               {
                  collectingModeSolverHeap->insert( pool[SOLVER_POOL_INDEX(rank)] );
               }
               nCollectingModeSolvers++;
               break;
            }
         }
      }
      if( nCollect > 0 )
      {
         double globalBestDualBoundValue = selectionHeap->top()->getBestDualBoundValue();
         switch( paraParams->getIntParamValue(SolverOrderInCollectingMode) )
         {
         case -1: // no ordering
         {
            for( p = activeSolvers.begin();
                  p != activeSolvers.end() && nLimitCollectingModeSolvers > nCollectingModeSolvers;
                  ++p)
            {
               if( p->second->getNumOfNodesLeft() > 1 &&             /* Solver should have at least two nodes left */
                     ( (p->second->getBestDualBoundValue() - globalBestDualBoundValue) /
                     max( fabs(globalBestDualBoundValue), 1.0 ) ) < bgap )
               {
                  int rank = p->second->getRank();
                  if ( pool[SOLVER_POOL_INDEX(rank)]->isGenerator() &&
                        pool[SOLVER_POOL_INDEX(rank)]->isOutCollectingMode() )
                  {
                     int nCollectPerSolver = std::min(
                           pool[SOLVER_POOL_INDEX(rank)]->getNumOfNodesLeft()/4, nCollect );
                     if( (signed)paraNodePool->getNumOfGoodNodes( getGlobalBestDualBoundValue()) <
                           (paraParams->getIntParamValue(NChangeIntoCollectingMode)*mCollectingNodes)/4 )
                     {
                        nCollectPerSolver = 0 - (nCollectPerSolver + 1);    // indicate aggressive collecting
                     }
                     assert(pool[SOLVER_POOL_INDEX(rank)]->isGenerator());
                     if( paraNodePool->isEmpty() && nLimitCollectingModeSolvers > 10 )
                     {
                        PARA_COMM_CALL(
                              paraComm->send(NULL, 0, ParaBYTE, rank, TagNoWaitModeSend )
                        );
                     }
                     PARA_COMM_CALL(
                           paraComm->send(&nCollectPerSolver, 1, ParaINT, rank, TagInCollectingMode )
                     );
                     nCollect -= nCollectPerSolver;
                     pool[SOLVER_POOL_INDEX(rank)]->setCollectingMode(true);
                     nCollectingModeSolvers++;
                  }
               }
               else
               {
                  // The following code works, when the number of collecting mode solvers is increased
                  if( !paraNodePool->isEmpty() &&
                        ( ( (p->second->getBestDualBoundValue() - globalBestDualBoundValue) /
                        max( fabs(globalBestDualBoundValue), 1.0 ) ) > ( bgap * mBgap ) ) )
                  {
                     int rank = p->second->getRank();
                     sendSwitchOutCollectingModeIfNecessary( rank );
                  }
               }
            }
            break;
         }
         case 0:  // ordering by dual bound value
         {

            std::multimap< double, int > boundOrderMap;
            for( p = activeSolvers.begin(); p != activeSolvers.end(); ++p )
            {
               boundOrderMap.insert(std::make_pair(p->second->getBestDualBoundValue(),p->second->getRank()));
            }
            std::multimap< double, int >::iterator pbo;
            for( pbo = boundOrderMap.begin();
                  pbo != boundOrderMap.end() && nLimitCollectingModeSolvers > nCollectingModeSolvers;
                  ++pbo )
            {
               int rank = pbo->second;
               if( pool[SOLVER_POOL_INDEX(rank)]->getNumOfNodesLeft() > 1 ) // Solver should have at least two nodes left
               {
                  if ( pool[SOLVER_POOL_INDEX(rank)]->isGenerator() &&
                        pool[SOLVER_POOL_INDEX(rank)]->isOutCollectingMode() )
                  {
                     int nCollectPerSolver = std::min(
                           pool[SOLVER_POOL_INDEX(rank)]->getNumOfNodesLeft()/4, nCollect );
                     if( (signed)paraNodePool->getNumOfGoodNodes( getGlobalBestDualBoundValue()) <
                           (paraParams->getIntParamValue(NChangeIntoCollectingMode)*mCollectingNodes)/4 )
                     {
                        nCollectPerSolver = 0 - (nCollectPerSolver + 1);    // indicate aggressive collecting
                     }
                     assert(pool[SOLVER_POOL_INDEX(rank)]->isGenerator());
                     if( paraNodePool->isEmpty() && nLimitCollectingModeSolvers > 10 )
                     {
                        PARA_COMM_CALL(
                              paraComm->send(NULL, 0, ParaBYTE, rank, TagNoWaitModeSend )
                        );
                     }
                     PARA_COMM_CALL(
                           paraComm->send(&nCollectPerSolver, 1, ParaINT, rank, TagInCollectingMode )
                     );
                     nCollect -= nCollectPerSolver;
                     if( pool[SOLVER_POOL_INDEX(rank)]->isCandidateOfCollecting() )
                     {
                        std::multimap< double, BbParaSolverPoolElementPtr >::iterator pCandidate;
                        for( pCandidate = candidatesOfCollectingModeSolvers.begin();
                              pCandidate != candidatesOfCollectingModeSolvers.end(); ++pCandidate )
                        {
                           if( pCandidate->second->getRank() == rank )
                              break;
                        }
                        assert( pCandidate != candidatesOfCollectingModeSolvers.end() );
                        pCandidate->second->setCandidateOfCollecting(false);
                        candidatesOfCollectingModeSolvers.erase(pCandidate);
                     }
                     pool[SOLVER_POOL_INDEX(rank)]->setCollectingMode(true);
                     assert( collectingModeSolverHeap );
                     collectingModeSolverHeap->insert( pool[SOLVER_POOL_INDEX(rank)] );
                     nCollectingModeSolvers++;
                  }
                  else  //  pool[SOLVER_POOL_INDEX(rank)] is in collecting mode
                  {
                     if( pool[SOLVER_POOL_INDEX(rank)]->isGenerator() )
                     {
                        assert( pool[SOLVER_POOL_INDEX(rank)]->isOutCollectingMode() ); // No collecting mode soler exists. Never comes here.
                        THROW_LOGICAL_ERROR2("Logical Error. All solvers are not in collecting mode. rank = ", rank);
                        // The following code works, when the number of collecting mode solvers is increased
                        if( !paraNodePool->isEmpty() &&
                              ( ( (pool[SOLVER_POOL_INDEX(rank)]->getBestDualBoundValue() - globalBestDualBoundValue) /
                              max( fabs(globalBestDualBoundValue), 1.0 ) ) > ( bgap * mBgap ) ) )
                        {
                           sendSwitchOutCollectingModeIfNecessary( rank );
                        }
                        else  // re-boost the collecting mode solver
                        {
                           sendSwitchOutCollectingModeIfNecessary( rank );
                           int nCollectPerSolver = std::min(
                                 pool[SOLVER_POOL_INDEX(rank)]->getNumOfNodesLeft()/4, nCollect );
                           if( (signed)paraNodePool->getNumOfGoodNodes( getGlobalBestDualBoundValue()) <
                                 (paraParams->getIntParamValue(NChangeIntoCollectingMode)*mCollectingNodes)/4 )
                           {
                              nCollectPerSolver = 0 - (nCollectPerSolver + 1);    // indicate aggressive collecting
                           }
                           assert(pool[SOLVER_POOL_INDEX(rank)]->isGenerator());
                           if( paraNodePool->isEmpty() && nLimitCollectingModeSolvers > 10 )
                           {
                              PARA_COMM_CALL(
                                    paraComm->send(NULL, 0, ParaBYTE, rank, TagNoWaitModeSend )
                              );
                           }
                           PARA_COMM_CALL(
                                 paraComm->send(&nCollectPerSolver, 1, ParaINT, rank, TagInCollectingMode )
                           );
                           nCollect -= nCollectPerSolver;
                           pool[SOLVER_POOL_INDEX(rank)]->setCollectingMode(true);
                           assert( collectingModeSolverHeap );
                           collectingModeSolverHeap->insert( pool[SOLVER_POOL_INDEX(rank)] );
                           // the solvers was already in collecting mode, so do not have to nCollectingModeSolvers++;
                        }
                     }
                  }
               }
            }
            for(; pbo != boundOrderMap.end() &&
                        static_cast<int>( candidatesOfCollectingModeSolvers.size() )
                                             <  std::min((int)nLimitCollectingModeSolvers,
                                                   paraParams->getIntParamValue(NMaxCanditatesForCollecting) );
                  ++pbo )
            {
               int rank = pbo->second;
               if( pool[SOLVER_POOL_INDEX(rank)]->getNumOfNodesLeft() > 1 &&
                     pool[SOLVER_POOL_INDEX(rank)]->isGenerator() &&
                     ( !pool[SOLVER_POOL_INDEX(rank)]->isCandidateOfCollecting() ) )
               {
                  candidatesOfCollectingModeSolvers.insert(
                        make_pair(pool[SOLVER_POOL_INDEX(rank)]->getBestDualBoundValue(),pool[SOLVER_POOL_INDEX(rank)])
                        );
                  pool[SOLVER_POOL_INDEX(rank)]->setCandidateOfCollecting(true);
               }
            }
            break;
         }
         case 1:  // ordering by number of nodes left
         {
            std::multimap< int, int > nNodesOrderMap;
            for( p = activeSolvers.begin(); p != activeSolvers.end(); ++p )
            {
               nNodesOrderMap.insert(std::make_pair(p->second->getNumOfNodesLeft(),p->second->getRank()));
            }
            std::multimap< int, int >::iterator pno;
            for( pno = nNodesOrderMap.begin();
                  pno != nNodesOrderMap.end() && nLimitCollectingModeSolvers > nCollectingModeSolvers;
                  ++pno )
            {
               int rank = pno->second;
               if( pool[SOLVER_POOL_INDEX(rank)]->getNumOfNodesLeft() > 1 &&     /* Solver should have at least two nodes left */
                     pool[SOLVER_POOL_INDEX(rank)]->isGenerator() &&
                     ( (pool[SOLVER_POOL_INDEX(rank)]->getBestDualBoundValue() - globalBestDualBoundValue) /
                     max( fabs(globalBestDualBoundValue), 1.0 ) ) < bgap )
               {
                  if ( pool[SOLVER_POOL_INDEX(rank)]->isOutCollectingMode() )
                  {
                     int nCollectPerSolver = std::min(
                           pool[SOLVER_POOL_INDEX(rank)]->getNumOfNodesLeft()/4, nCollect );
                     if( (signed)paraNodePool->getNumOfGoodNodes( getGlobalBestDualBoundValue()) <
                           (paraParams->getIntParamValue(NChangeIntoCollectingMode)*mCollectingNodes)/4 )
                     {
                        nCollectPerSolver = 0 - (nCollectPerSolver + 1);    // indicate aggressive collecting
                     }
                     assert(pool[SOLVER_POOL_INDEX(rank)]->isGenerator());
                     if( paraNodePool->isEmpty() && nLimitCollectingModeSolvers > 10 )
                     {
                        PARA_COMM_CALL(
                              paraComm->send(NULL, 0, ParaBYTE, rank, TagNoWaitModeSend )
                        );
                     }
                     PARA_COMM_CALL(
                           paraComm->send(&nCollectPerSolver, 1, ParaINT, rank, TagInCollectingMode )
                     );
                     nCollect -= nCollectPerSolver;
                     pool[SOLVER_POOL_INDEX(rank)]->setCollectingMode(true);
                     nCollectingModeSolvers++;
                  }
               }
               else
               {
                  // The following code works, when the number of collecting mode solvers is increased
                  if( !paraNodePool->isEmpty() &&
                        ( ( (pool[SOLVER_POOL_INDEX(rank)]->getBestDualBoundValue() - globalBestDualBoundValue) /
                        max( fabs(globalBestDualBoundValue), 1.0 ) ) > ( bgap * mBgap ) ) )
                  {
                     sendSwitchOutCollectingModeIfNecessary( rank );
                  }
               }
            }
            break;
         }
         case 2:  // choose alternatively the best bound and the number of nodes orders
         {
            std::multimap< double, int > boundOrderMap;
            for( p = activeSolvers.begin(); p != activeSolvers.end(); ++p )
            {
               boundOrderMap.insert(std::make_pair(p->second->getBestDualBoundValue(),p->second->getRank()));
            }
            std::multimap< int, int > nNodesOrderMap;
            for( p = activeSolvers.begin(); p != activeSolvers.end(); ++p )
            {
               nNodesOrderMap.insert(std::make_pair(p->second->getNumOfNodesLeft(),p->second->getRank()));
            }
            std::multimap< double, int >::iterator pbo;
            std::multimap< int, int >::iterator pno;
            pbo = boundOrderMap.begin();
            pno = nNodesOrderMap.begin();
            bool boundOrderSection = true;
            while( pbo != boundOrderMap.end()
                  && pno != nNodesOrderMap.end()
                  && nLimitCollectingModeSolvers > nCollectingModeSolvers )
            {
               int rank;
               if( boundOrderSection )
               {
                  if( pbo !=  boundOrderMap.end() )
                  {
                     rank = pbo->second;
                     pbo++;
                     boundOrderSection = false;
                  }
                  else
                  {
                     boundOrderSection = false;
                     continue;
                  }
               }
               else
               {
                  if( pno != nNodesOrderMap.end() )
                  {
                     rank = pno->second;
                     pno++;
                     boundOrderSection = true;
                  } else {
                     boundOrderSection = true;
                     continue;
                  }
               }
               if( pool[SOLVER_POOL_INDEX(rank)]->getNumOfNodesLeft() > 1  &&        /* Solver should have at least two nodes left */
                     pool[SOLVER_POOL_INDEX(rank)]->isGenerator() &&
                     ( (pool[SOLVER_POOL_INDEX(rank)]->getBestDualBoundValue() - globalBestDualBoundValue) /
                     max( fabs(globalBestDualBoundValue), 1.0 ) ) < bgap )
               {
                  if ( pool[SOLVER_POOL_INDEX(rank)]->isOutCollectingMode() )
                  {
                     int nCollectPerSolver = std::min(
                           pool[SOLVER_POOL_INDEX(rank)]->getNumOfNodesLeft()/4, nCollect );
                     if( (signed)paraNodePool->getNumOfGoodNodes( getGlobalBestDualBoundValue()) <
                           (paraParams->getIntParamValue(NChangeIntoCollectingMode)*mCollectingNodes)/4 )
                     {
                        nCollectPerSolver = 0 - (nCollectPerSolver + 1);    // indicate aggressive collecting
                     }
                     assert(pool[SOLVER_POOL_INDEX(rank)]->isGenerator());
                     if( paraNodePool->isEmpty() && nLimitCollectingModeSolvers > 10 )
                     {
                        PARA_COMM_CALL(
                              paraComm->send(NULL, 0, ParaBYTE, rank, TagNoWaitModeSend )
                        );
                     }
                     PARA_COMM_CALL(
                           paraComm->send(&nCollectPerSolver, 1, ParaINT, rank, TagInCollectingMode )
                     );
                     nCollect -= nCollectPerSolver;
                     pool[SOLVER_POOL_INDEX(rank)]->setCollectingMode(true);
                     nCollectingModeSolvers++;
                  }
               }
               else
               {
                  // The following code works, when the number of collecting mode solvers is increased
                  if( !paraNodePool->isEmpty() &&
                        ( ( (pool[SOLVER_POOL_INDEX(rank)]->getBestDualBoundValue() - globalBestDualBoundValue) /
                        max( fabs(globalBestDualBoundValue), 1.0 ) ) > ( bgap * mBgap ) ) )
                  {
                     sendSwitchOutCollectingModeIfNecessary( rank );
                  }
               }
            }
            break;
         }
         default:
            THROW_LOGICAL_ERROR2("paraParams->getIntParamValue(SolverOrderInCollectingMode) = ", paraParams->getIntParamValue(SolverOrderInCollectingMode));
         }
      }
   }
   collectingMode = true;
   if( !beforeInitialCollect ) beforeFinishingFirstCollect = false;
   beforeInitialCollect = false;
}

void
BbParaSolverPoolForMinimization::updateSolverStatus(
      int rank,
      long long numOfNodesSolved,
      int numOfNodesLeft,
      double solverLocalBestDualBound,
      BbParaNodePool *paraNodePool
      )
{
   map<int, BbParaSolverPoolElementPtr>::iterator p;
   p = activeSolvers.find(rank);
   if( p != activeSolvers.end() )
   {
      if( (numOfNodesSolved - p->second->getNumOfNodesSolved() ) < 0 )
      {
         std::cout << "numOfNodesSolved = " << numOfNodesSolved << ", p->second->getNumOfNodesSolved() = " << p->second->getNumOfNodesSolved() << std::endl;
         THROW_LOGICAL_ERROR2("Rank = ", rank);
      }
      // assert( (numOfNodesSolved - p->second->getNumOfNodesSolved() ) >= 0 ); // if solver restart, this is not always true
      nNodesSolvedInSolvers += numOfNodesSolved - p->second->getNumOfNodesSolved();
      // if(nNodesSolvedInSolvers > 999) abort();
      nNodesInSolvers += numOfNodesLeft - p->second->getNumOfNodesLeft();
      p->second->setNumOfDiffNodesSolved( numOfNodesSolved - p->second->getNumOfNodesSolved() );
      p->second->setNumOfNodesSolved(numOfNodesSolved);
      p->second->setNumOfDiffNodesLeft( numOfNodesLeft - p->second->getNumOfNodesLeft() );
      p->second->setNumOfNodesLeft(numOfNodesLeft);
      if( p->second->getBestDualBoundValue() < solverLocalBestDualBound ) // even in LP solving, solver may notify its status
      {
         p->second->setDualBoundValue(solverLocalBestDualBound);
         selectionHeap->updateDualBoundValue(*(p->second->getSelectionHeapElement()), solverLocalBestDualBound);
         if( p->second->isInCollectingMode() )
         {
            if( collectingModeSolverHeap )
            {
               collectingModeSolverHeap->updateDualBoundValue(*(p->second->getCollectingModeSolverHeapElement()), solverLocalBestDualBound);
            }
         }
      }
   }
   else
   {
      THROW_LOGICAL_ERROR2("Rank = ", rank);
   }
   double globalBestDualBoundValue = selectionHeap->top()->getBestDualBoundValue();
   if( globalBestDualBoundValue > p->second->getBestDualBoundValue() )
   {
      std::cout << "Top rank = " << selectionHeap->top()->getRank() << std::endl;
      std::cout << "Top bound = " << selectionHeap->top()->getBestDualBoundValue() << std::endl;
      std::cout << "This bound = " << p->second->getBestDualBoundValue() << std::endl;
      std::cout << "myRank = " << rank << std::endl;
      std::cout << "solverLocalBestDualBound = " << p->second->getBestDualBoundValue() << std::endl;
      std::cout << selectionHeap->toString();
      if( collectingModeSolverHeap )
      {
         std::cout << collectingModeSolverHeap->toString();
      }
      abort();
   }

   if( pool[SOLVER_POOL_INDEX(rank)]->isDualBoundGainTesting()
         && numOfNodesSolved > 1 )
   {
      pool[SOLVER_POOL_INDEX(rank)]->setDualBoundGainTesting(false);
      nDualBoundGainTesting--;
   }

   if( collectingMode )
   {
      int nCollect = 0;  // assume zero
      if( pool[SOLVER_POOL_INDEX(rank)]->isInCollectingMode() )
      {
         if( !selectionHeap->top()->isInCollectingMode() &&
               selectionHeap->top()->isGenerator() &&
               (!selectionHeap->top()->isCollectingProhibited()) &&
               selectionHeap->top()->getNumOfNodesLeft() > 1 )
         {
            if( nLimitCollectingModeSolvers > nCollectingModeSolvers )
            {
               switchInCollectingToSolver(selectionHeap->top()->getRank(), paraNodePool);
            }
            else   // nLimitCollectingModeSolvers == nCollectingModeSolvers
            {
               if( pool[SOLVER_POOL_INDEX(rank)]->getBestDualBoundValue() - absoluteGap
                     < selectionHeap->top()->getBestDualBoundValue() )
               {
                  if( !( breakingFirstSubtree && rank == 1 ) )
                  {
                     if( !paraNodePool->isEmpty() )
                     {
                        sendSwitchOutCollectingModeIfNecessary( rank );
                     }
                     switchInCollectingToSolver(selectionHeap->top()->getRank(), paraNodePool);
                  }
               }
            }
         }
         else
         {
            if( collectingModeSolverHeap &&
                  !( breakingFirstSubtree && collectingModeSolverHeap->getHeapSize() <= 1 ) &&
                  ( p->second->getBestDualBoundValue() - collectingModeSolverHeap->top()->getBestDualBoundValue() ) > absoluteGap && // the notified solver is no so good
                 ( (!candidatesOfCollectingModeSolvers.empty()) &&
                       ( candidatesOfCollectingModeSolvers.begin()->second->getBestDualBoundValue() + absoluteGap ) // to avoid flip
                       < p->second->getBestDualBoundValue() )  //promising candidates exists
                    )
            {
               assert( p->second->getRank() == rank );
               assert( rank != selectionHeap->top()->getRank() );
               if( !paraNodePool->isEmpty() )
               {
                  sendSwitchOutCollectingModeIfNecessary( rank );
               }

               std::multimap< double, BbParaSolverPoolElementPtr >::iterator pCandidate;
               pCandidate = candidatesOfCollectingModeSolvers.begin();
               assert( pCandidate->second->isOutCollectingMode() );
               // nCollect = std::min(selectionHeap->top()->getNumOfNodesLeft()/4,
               nCollect = std::min(pCandidate->second->getNumOfNodesLeft()/4,
                     static_cast<int>(
                     getNumInactiveSolvers() +
                     paraParams->getIntParamValue(NChangeIntoCollectingMode)*mCollectingNodes*
                       paraParams->getRealParamValue(MultiplierForCollectingMode)
                       - (signed)paraNodePool->getNumOfGoodNodes(getGlobalBestDualBoundValue())));
               if( (signed)paraNodePool->getNumOfGoodNodes( getGlobalBestDualBoundValue()) <
                     (paraParams->getIntParamValue(NChangeIntoCollectingMode)*mCollectingNodes)/4 )
               {
                  nCollect = 0 - (nCollect + 1);    // indicate aggressive collecting
               }
               assert(pool[SOLVER_POOL_INDEX(pCandidate->second->getRank())]->isGenerator());
               if( paraNodePool->isEmpty() && nLimitCollectingModeSolvers > 10 )
               {
                  PARA_COMM_CALL(
                        paraComm->send(NULL, 0, ParaBYTE, rank, TagNoWaitModeSend )
                  );
               }
               PARA_COMM_CALL(
                     paraComm->send(&nCollect, 1, ParaINT, pCandidate->second->getRank(), TagInCollectingMode)
               );
               pCandidate->second->setCollectingMode(true);
               collectingModeSolverHeap->insert( pCandidate->second );
               nCollectingModeSolvers++;
               pCandidate->second->setCandidateOfCollecting(false);
               candidatesOfCollectingModeSolvers.erase(pCandidate);
            }
            else
            {
               if( !paraNodePool->isEmpty() &&
                     ( ( ( ( p->second->getBestDualBoundValue() - globalBestDualBoundValue ) /
                       max( fabs(globalBestDualBoundValue), 1.0 ) )  > ( bgap * mBgap ) ) ) )
               {
                  sendSwitchOutCollectingModeIfNecessary( p->second->getRank() );
               }
            }
         }
      }
      else  // current notification solver is out collecting mode
      {
         if( nLimitCollectingModeSolvers > nCollectingModeSolvers )
         {
            if( pool[SOLVER_POOL_INDEX(rank)]->getRank() == selectionHeap->top()->getRank() &&
                  pool[SOLVER_POOL_INDEX(rank)]->isGenerator() &&
                  (!pool[SOLVER_POOL_INDEX(rank)]->isCollectingProhibited()) &&
                  selectionHeap->top()->getNumOfNodesLeft() > 1 )
            {
                switchInCollectingToSolver(rank, paraNodePool);
            }
         }
         else  // nLimitCollectingModeSolvers == nCollectingModeSolvers
         {
            if( !paraNodePool->isEmpty() &&
                  pool[SOLVER_POOL_INDEX(rank)]->getNumOfDiffNodesLeft() > 1 &&
                  collectingModeSolverHeap &&
                  !( breakingFirstSubtree && collectingModeSolverHeap->getHeapSize() <= 1 ) &&
                  ( pool[SOLVER_POOL_INDEX(rank)]->getBestDualBoundValue() + absoluteGap  )
                  < collectingModeSolverHeap->top()->getBestDualBoundValue()  )
            {
               sendSwitchOutCollectingModeIfNecessary( collectingModeSolverHeap->top()->getRank() );
            }
         }
      }

      bool manyIdleSolversExist = false;
      if( getNumInactiveSolvers() > nSolvers/4 ) manyIdleSolversExist = true;
      if( nLimitCollectingModeSolvers > nCollectingModeSolvers )
      {
         if( manyIdleSolversExist )
         {
            double temp = switchOutTime;
            switchOutCollectingMode();
            switchOutTime = temp;
            switchInCollectingMode(paraNodePool);
         }
         else
         {
            if( selectionHeap->top()->getNumOfNodesLeft() > 1 &&
                  selectionHeap->top()->isGenerator() &&
                  (!selectionHeap->top()->isCollectingProhibited()) &&
                  selectionHeap->top()->isOutCollectingMode() )
            {
               switchInCollectingToSolver(selectionHeap->top()->getRank(), paraNodePool);
            }
            else
            {
               if( !candidatesOfCollectingModeSolvers.empty() )
               {
                  std::multimap< double, BbParaSolverPoolElementPtr >::iterator pCandidate;
                  pCandidate = candidatesOfCollectingModeSolvers.begin();
                  assert( pCandidate->second->isOutCollectingMode() );
                  nCollect = std::min(pCandidate->second->getNumOfNodesLeft()/4,
                        static_cast<int>(
                        getNumInactiveSolvers() +
                        paraParams->getIntParamValue(NChangeIntoCollectingMode)*mCollectingNodes*
                          paraParams->getRealParamValue(MultiplierForCollectingMode)
                          - (signed)paraNodePool->getNumOfGoodNodes(getGlobalBestDualBoundValue())));
                  if( (signed)paraNodePool->getNumOfGoodNodes( getGlobalBestDualBoundValue()) <
                        (paraParams->getIntParamValue(NChangeIntoCollectingMode)*mCollectingNodes)/4 )
                  {
                     nCollect = 0 - (nCollect + 1);    // indicate aggressive collecting
                  }
                  assert(pool[SOLVER_POOL_INDEX(pCandidate->second->getRank())]->isGenerator());
                  if( paraNodePool->isEmpty() && nLimitCollectingModeSolvers > 10 )
                  {
                     PARA_COMM_CALL(
                           paraComm->send(NULL, 0, ParaBYTE, rank, TagNoWaitModeSend )
                     );
                  }
                  PARA_COMM_CALL(
                        paraComm->send(&nCollect, 1, ParaINT, pCandidate->second->getRank(), TagInCollectingMode)
                  );
                  pCandidate->second->setCollectingMode(true);
                  if( collectingModeSolverHeap )
                  {
                     collectingModeSolverHeap->insert( pCandidate->second );
                  }
                  nCollectingModeSolvers++;
                  pCandidate->second->setCandidateOfCollecting(false);
                  candidatesOfCollectingModeSolvers.erase(pCandidate);
               }
               else
               {
                  assert( rank == p->second->getRank() );
                  if( p->second->getNumOfNodesLeft() > 1 &&                // Solver should have at least two nodes left //
                        p->second->isGenerator() &&
                        ( !p->second->isCollectingProhibited() ) &&
                        p->second->isOutCollectingMode() &&
                           ( paraNodePool->isEmpty() ||
                                 ( ( ( p->second->getBestDualBoundValue() - globalBestDualBoundValue ) /
                                       max( fabs(globalBestDualBoundValue), 1.0 ) )  < bgap ) ) ) //  &&
                  {
                     switchInCollectingToSolver(rank, paraNodePool);
                  }
               }
            }
         }
      }
      else   // nLimitCollectingModeSolvers == nCollectingModeSolvers
      {
         if ( pool[SOLVER_POOL_INDEX(rank)]->isOutCollectingMode() )
         {
            if( !candidatesOfCollectingModeSolvers.empty() )
            {
               std::multimap< double, BbParaSolverPoolElementPtr >::iterator pCandidate;
               if( p->second->isCandidateOfCollecting() )
               {
                  for( pCandidate = candidatesOfCollectingModeSolvers.begin();
                        pCandidate != candidatesOfCollectingModeSolvers.end(); ++pCandidate )
                  {
                     if( pCandidate->second->getRank() == rank )
                        break;
                  }
                  if( pCandidate == candidatesOfCollectingModeSolvers.end() )
                  {
                     THROW_LOGICAL_ERROR3("rank = ", rank , " is not in candidatesOfCollectingModeSolvers.");
                  }
                  /* anyway remove to update dual bound value */
                  pCandidate->second->setCandidateOfCollecting(false);
                  candidatesOfCollectingModeSolvers.erase(pCandidate);
               }
               if( static_cast<int>( candidatesOfCollectingModeSolvers.size() )
                     == std::min((signed)nLimitCollectingModeSolvers,
                           paraParams->getIntParamValue(NMaxCanditatesForCollecting) ) )
               {
                  std::multimap< double, BbParaSolverPoolElementPtr >::reverse_iterator prCandidate;
                  prCandidate = candidatesOfCollectingModeSolvers.rbegin();
                  if( p->second->getNumOfNodesLeft() > 1 &&
                        p->second->isGenerator() &&
                        prCandidate->second->getBestDualBoundValue() > p->second->getBestDualBoundValue() )
                  {
                     BbParaSolverPoolElementPtr tCandidate = prCandidate->second;
                     // prCandidate->second->setCandidateOfCollecting(false);
                     ++prCandidate;
                     assert(prCandidate.base()->second == tCandidate);
                     if( prCandidate.base()->second != tCandidate )
                     {
                        THROW_LOGICAL_ERROR4("prCandidate.base()->second != tCandidate, prCandidate.base()->second = ", prCandidate.base()->second, ", tCandidate = ", tCandidate );
                     }
                     prCandidate.base()->second->setCandidateOfCollecting(false);
                     candidatesOfCollectingModeSolvers.erase(prCandidate.base());
                     candidatesOfCollectingModeSolvers.insert(
                           make_pair(p->second->getBestDualBoundValue(),p->second)
                           );
                     p->second->setCandidateOfCollecting(true);
                  }
               }
               else  // candidatesOfCollectingModeSolvers.size()
                     //  < std::min(nLimitCollectingModeSolvers,
                     //               paraParams->getIntParamValue(NMaxCanditatesForCollecting)
               {
                  if( collectingModeSolverHeap &&
                        !( breakingFirstSubtree && collectingModeSolverHeap->getHeapSize() <= 1 ) &&
                        p->second->getNumOfNodesLeft() > 1 &&
                        p->second->isGenerator() &&
                        ( p->second->getBestDualBoundValue() - collectingModeSolverHeap->top()->getBestDualBoundValue() ) < absoluteGap )
                  {
                     candidatesOfCollectingModeSolvers.insert(
                           make_pair(p->second->getBestDualBoundValue(),p->second)
                           );
                     p->second->setCandidateOfCollecting(true);
                     // p->second solver is in collecting mode, when the current collecting mode solver notifies.
                  }
               }
            }
            else  // candidatesOfCollectingModeSolvers is empty
            {
               if( collectingModeSolverHeap &&
                     !( breakingFirstSubtree && collectingModeSolverHeap->getHeapSize() <= 1 ) &&
                     p->second->getNumOfNodesLeft() > 1 &&
                     p->second->isGenerator() &&
                     ( p->second->getBestDualBoundValue() - collectingModeSolverHeap->top()->getBestDualBoundValue() ) < absoluteGap )
               {
                  candidatesOfCollectingModeSolvers.insert(
                        make_pair(p->second->getBestDualBoundValue(),p->second)
                        );
                  p->second->setCandidateOfCollecting(true);
                  // p->second solver is in collecting mode, when the current collecting mode solver notifies.
               }
            }
         }
      }
#ifdef _DEBUG_LB
      std::cout << "ASBD = " << selectionHeap->top()->getBestDualBoundValue() << "(" << selectionHeap->top()->getRank() << ")";
      if( collectingModeSolverHeap && collectingModeSolverHeap->getHeapSize() > 0 )
      {
         std::cout << ", CMSBD = " << collectingModeSolverHeap->top()->getBestDualBoundValue() << "(" << collectingModeSolverHeap->top()->getRank() << ")";
      }
      if( !candidatesOfCollectingModeSolvers.empty() )
      {
         std::cout << ", CCSBD = ";
         std::multimap< double, BbParaSolverPoolElementPtr >::iterator pCandidate;
         for( pCandidate = candidatesOfCollectingModeSolvers.begin();
               pCandidate != candidatesOfCollectingModeSolvers.end(); ++pCandidate )
         {
            std::cout << ", " << pCandidate->second->getBestDualBoundValue() << "(" <<  pCandidate->second->getRank() << ")";
         }
      }
      std::cout << std::endl;
#endif
   }
   else
   {
      if( !pool[SOLVER_POOL_INDEX(rank)]->isOutCollectingMode() )
      {
         THROW_LOGICAL_ERROR2("Not Out collecting mode: Rank = ", rank);
      }
   }
}

void
BbParaRacingSolverPool::updateSolverStatus(
      int rank,
      long long numOfNodesSolved,
      int numOfNodesLeft,
      double solverLocalBestDualBound
      )
{
   if( pool[SOLVER_POOL_INDEX(rank)]->getStatus() != Racing &&
         pool[SOLVER_POOL_INDEX(rank)]->getStatus() != RacingEvaluation )
   {
      std::cout << "Tried to update no racing status solver's info." << std::endl;
      std::cout << "Update solver rank = " << rank << std::endl;
      abort();
   }


   if( paraParams->getIntParamDefaultValue(RacingRampUpTerminationCriteria) == 0 ) // Criterion is the number of nodes
   {
      if(pool[SOLVER_POOL_INDEX(rank)]->getNumOfNodesLeft() < paraParams->getIntParamDefaultValue(StopRacingNumberOfNodesLeft) &&
            numOfNodesLeft >= paraParams->getIntParamDefaultValue(StopRacingNumberOfNodesLeft) )
      {
         pool[SOLVER_POOL_INDEX(rank)]->switchIntoEvaluation();
         nEvaluationStage++;
      }
      if(pool[SOLVER_POOL_INDEX(rank)]->getNumOfNodesLeft() >= paraParams->getIntParamDefaultValue(StopRacingNumberOfNodesLeft) &&
            numOfNodesLeft < paraParams->getIntParamDefaultValue(StopRacingNumberOfNodesLeft) )
      {
         pool[SOLVER_POOL_INDEX(rank)]->switchOutEvaluation();
         nEvaluationStage--;
      }

   } else {
      if( !pool[SOLVER_POOL_INDEX(rank)]->isEvaluationStage() )
      {
         pool[SOLVER_POOL_INDEX(rank)]->switchIntoEvaluation();
         nEvaluationStage++;
      }
   }
   pool[SOLVER_POOL_INDEX(rank)]->setNumOfDiffNodesSolved( numOfNodesSolved - pool[SOLVER_POOL_INDEX(rank)]->getNumOfNodesSolved() );
   pool[SOLVER_POOL_INDEX(rank)]->setNumOfNodesSolved(numOfNodesSolved);
   pool[SOLVER_POOL_INDEX(rank)]->setNumOfDiffNodesLeft( numOfNodesLeft - pool[SOLVER_POOL_INDEX(rank)]->getNumOfNodesLeft() );
   pool[SOLVER_POOL_INDEX(rank)]->setNumOfNodesLeft(numOfNodesLeft);
//   if( rank != originRank ||
//         ( paraParams->getIntParamValue(UgCplexRunningMode) != 1 ) )
//   if( rank != originRank )
//   {
//      int prviousTop = selectionHeap->top()->getRank();
      selectionHeap->updateDualBoundValue(*(pool[SOLVER_POOL_INDEX(rank)]->getSelectionHeapElement()), solverLocalBestDualBound);
//      if( selectionHeap->top()->getRank() == rank || selectionHeap->top()->getRank() != prviousTop )
//      {
         nNodesInBestSolver = selectionHeap->top()->getNumOfNodesLeft();      // in racing ramp-up, winner nodes should be the number of nodes left
         nNodesSolvedInBestSolver = selectionHeap->top()->getNumOfNodesSolved();
//      }
      if( selectionHeap->getHeapSize() > 0 )
      {
         if( selectionHeap->top()->getBestDualBoundValue() > bestDualBound )
         {
            bestDualBound = selectionHeap->top()->getBestDualBoundValue();
         }
      }
//  }

   if( bestDualBoundInSolvers < solverLocalBestDualBound )
   {
      bestDualBoundInSolvers = solverLocalBestDualBound;
   }

}

bool
BbParaRacingSolverPool::isWinnerDecided(
      bool feasibleSol
      )
{
    double solverRatio = paraParams->getRealParamValue(CountingSolverRatioInRacing);
    switch( paraParams->getIntParamValue(RacingRampUpTerminationCriteria) )
    {
    case 0:  /** stop racing at the number of nodes left in a solver reached to the value */
    {
       if( ( paraParams->getIntParamValue(NEvaluationSolversToStopRacing) == -1 && nEvaluationStage == nSolvers  ) ||
             ( paraParams->getIntParamValue(NEvaluationSolversToStopRacing) == 0 && nEvaluationStage >= ( nSolvers/2 )  ) ||
             ( paraParams->getIntParamValue(NEvaluationSolversToStopRacing) > 0 &&
                    nEvaluationStage >= paraParams->getIntParamValue(NEvaluationSolversToStopRacing) ) )
       {

          if( selectionHeap->top()->getNumOfNodesLeft() > paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) )
          {
             winnerRank = selectionHeap->top()->getRank();
             return true;
          }
       }
       break;
    }
    case 1:  /** stop racing at time limit */
    {
       if( nEvaluationStage >= static_cast<int>(nSolvers*solverRatio) )  // some solver may not reply at racing
       {
          if( paraParams->getBoolParamValue(Deterministic) )
          {
             if( paraDetTimer->getElapsedTime() > paraParams->getRealParamValue(StopRacingTimeLimit) )
             {
                winnerRank = selectionHeap->top()->getRank();
                return true;
             }
          }
          else
          {
             if( paraTimer->getElapsedTime() > paraParams->getRealParamValue(StopRacingTimeLimit) )
             {
                winnerRank = selectionHeap->top()->getRank();
                return true;
             }
          }
       }
       break;
    }
    case 2:  /** stop racing at the Solver with the best bound value has a certain number of nodes left */
    {
       if( nEvaluationStage >= static_cast<int>(nSolvers*solverRatio) )  // some solver may not reply at racing
       {
          if( selectionHeap->top()->getNumOfNodesLeft() > paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) )
          {
             winnerRank = selectionHeap->top()->getRank();
             return true;
          }
       }
       break;
    }
    case 3:  /** node left first adaptive */
    {
       if( selectionHeap->top()->getNumOfNodesLeft() <= nSolvers
             && selectionHeap->top()->getNumOfNodesLeft() <= paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) )
       {
          return false;
       }
       if( nEvaluationStage >= static_cast<int>(nSolvers*solverRatio) )  // some solver may not reply at racing
       {
          if( selectionHeap->top()->getNumOfNodesLeft() > paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) )
          {
             winnerRank = selectionHeap->top()->getRank();
             return true;
          }

          if( paraParams->getBoolParamValue(Deterministic) )
          {
             if( paraDetTimer->getElapsedTime() > paraParams->getRealParamValue(StopRacingTimeLimit) )
             {
                winnerRank = selectionHeap->top()->getRank();
                return true;
             }
          }
          else
          {
             if( paraTimer->getElapsedTime() > paraParams->getRealParamValue(StopRacingTimeLimit) )
             {
                winnerRank = selectionHeap->top()->getRank();
                return true;
             }
          }
       }
       break;
    }
    case 4:  /** time limit first adaptive */
    {
       if( nEvaluationStage >= static_cast<int>(nSolvers*solverRatio) )  // some solver may not reply at racing
       {
          if( paraParams->getBoolParamValue(Deterministic) )
          {
             if( paraDetTimer->getElapsedTime() > paraParams->getRealParamValue(StopRacingTimeLimit) )
             {
                winnerRank = selectionHeap->top()->getRank();
                return true;
             }

             if( selectionHeap->top()->getNumOfNodesLeft() <= nSolvers
                   && selectionHeap->top()->getNumOfNodesLeft() <= paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) )
             {
                return false;
             }
             else
             {
                if( selectionHeap->top()->getNumOfNodesLeft() > paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) )
                {
                   winnerRank = selectionHeap->top()->getRank();
                   return true;
                }
             }
          }
          else
          {
             if( paraTimer->getElapsedTime() > paraParams->getRealParamValue(StopRacingTimeLimit) )
             {
                winnerRank = selectionHeap->top()->getRank();
                return true;
             }
             if( selectionHeap->top()->getNumOfNodesLeft() <= nSolvers
                   && selectionHeap->top()->getNumOfNodesLeft() <= paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) )
             {
                return false;
             }
             else
             {
                if( selectionHeap->top()->getNumOfNodesLeft() > paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) )
                {
                   winnerRank = selectionHeap->top()->getRank();
                   return true;
                }
             }
          }
       }
       break;
    }
    case 5:  /** node left first adaptive - adaptive node left (Default settings): this is for FiberSCIP */
    {
       if( nEvaluationStage >= static_cast<int>(nSolvers*solverRatio) )  // some solver may not reply at racing
       {
          if( paraParams->getBoolParamValue(KeepRacingUntilToFindFirstSolution) && (!feasibleSol) ) return false;   // no feasible solution keeps racing
          if( selectionHeap->top()->getNumOfNodesLeft() <= nSolvers )
          {
             return false;
          }

          if( selectionHeap->top()->getNumOfNodesLeft() >
              paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) )
          {
             if( !paraParams->getBoolParamValue(KeepRacingUntilToFindFirstSolution) || feasibleSol )
             {
                winnerRank = selectionHeap->top()->getRank();
                return true;
             }
             else
             {
                if( selectionHeap->top()->getNumOfNodesLeft() >
                   ( paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) * paraParams->getRealParamValue(StopRacingNumberOfNodesLeftMultiplier) ) )
                {
                   winnerRank = selectionHeap->top()->getRank();
                   return true;
                }
                if( paraParams->getBoolParamValue(Deterministic) )
                {
                   if( paraDetTimer->getElapsedTime() >
                        ( paraParams->getRealParamValue(StopRacingTimeLimit) * paraParams->getRealParamValue(StopRacingTimeLimitMultiplier) ) )
                   {
                      winnerRank = selectionHeap->top()->getRank();
                      return true;
                   }
                }
                else
                {
                   if( paraTimer->getElapsedTime() >
                        ( paraParams->getRealParamValue(StopRacingTimeLimit) * paraParams->getRealParamValue(StopRacingTimeLimitMultiplier) ) )
                   {
                      winnerRank = selectionHeap->top()->getRank();
                      return true;
                   }
                }
             }
          }
          else
          {
             if( paraParams->getBoolParamValue(Deterministic) )
             {
                if( paraDetTimer->getElapsedTime() > paraParams->getRealParamValue(StopRacingTimeLimit) )
                {
                   if( selectionHeap->top()->getNumOfNodesLeft() > nSolvers )
                   {
                      if( !paraParams->getBoolParamValue(KeepRacingUntilToFindFirstSolution) || feasibleSol )
                      {
                            winnerRank = selectionHeap->top()->getRank();
                            return true;
                      }
                      else
                      {
                         if( paraDetTimer->getElapsedTime() >
                             ( paraParams->getRealParamValue(StopRacingTimeLimit) * paraParams->getRealParamValue(StopRacingTimeLimitMultiplier) ) )
                         {
                            winnerRank = selectionHeap->top()->getRank();
                            return true;
                         }
                      }
                   }
                }
             }
             else
             {
                if( paraTimer->getElapsedTime() > paraParams->getRealParamValue(StopRacingTimeLimit) )
                {
                   if( selectionHeap->top()->getNumOfNodesLeft() > nSolvers )
                   {
                      if( !paraParams->getBoolParamValue(KeepRacingUntilToFindFirstSolution) || feasibleSol )
                      {
                            winnerRank = selectionHeap->top()->getRank();
                            return true;
                      }
                      else
                      {
                         if( paraTimer->getElapsedTime() >
                             ( paraParams->getRealParamValue(StopRacingTimeLimit) * paraParams->getRealParamValue(StopRacingTimeLimitMultiplier) ) )
                         {
                            winnerRank = selectionHeap->top()->getRank();
                            return true;
                         }
                      }
                   }
                }
             }
          }
       }
       break;
    }
    case 6:  /** time limit first adaptive - adaptive node left */
    {
       if( nEvaluationStage >= static_cast<int>(nSolvers*solverRatio) )  // some solver may not reply at racing
       {
          if( paraParams->getBoolParamValue(KeepRacingUntilToFindFirstSolution) && (!feasibleSol) ) return false;   // no feasible solution keeps racing
          if( selectionHeap->top()->getNumOfNodesLeft() <= nSolvers )
          {
             return false;
          }

          if( paraParams->getBoolParamValue(Deterministic) )
          {
             if( paraDetTimer->getElapsedTime() > paraParams->getRealParamValue(StopRacingTimeLimit) )
             {
                 if( selectionHeap->top()->getNumOfNodesLeft() > paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) )
                 {
                    winnerRank = selectionHeap->top()->getRank();
                    return true;
                 }
                 else
                 {
                    if( paraDetTimer->getElapsedTime() >
                         ( paraParams->getRealParamValue(StopRacingTimeLimit) * paraParams->getRealParamValue(StopRacingTimeLimitMultiplier) )  )
                    {
                       winnerRank = selectionHeap->top()->getRank();
                       return true;
                    }
                 }
             }
          }
          else
          {
             if( paraTimer->getElapsedTime() > paraParams->getRealParamValue(StopRacingTimeLimit) )
             {
                if( selectionHeap->top()->getNumOfNodesLeft() > paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) )
                {
                   winnerRank = selectionHeap->top()->getRank();
                   return true;
                }
                else
                {
                   if( paraTimer->getElapsedTime() >
                        ( paraParams->getRealParamValue(StopRacingTimeLimit) * paraParams->getRealParamValue(StopRacingTimeLimitMultiplier) ) )
                   {
                      winnerRank = selectionHeap->top()->getRank();
                      return true;
                   }
                }
             }
          }
       }
       break;
    }
    case 7:  /** time limit first adaptive and node limit also checked - adaptive node left */
    {
       if( nEvaluationStage >= static_cast<int>(nSolvers*solverRatio) )  // some solver may not reply at racing
       {
          if( paraParams->getBoolParamValue(KeepRacingUntilToFindFirstSolution) && (!feasibleSol) ) return false;   // no feasible solution keeps racing
          if( selectionHeap->top()->getNumOfNodesLeft() <= nSolvers )
          {
             return false;
          }

          if( paraParams->getBoolParamValue(Deterministic) )
          {
             if( paraDetTimer->getElapsedTime() > paraParams->getRealParamValue(StopRacingTimeLimit) )
             {
                 if( selectionHeap->top()->getNumOfNodesLeft() > paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) )
                 {
                    winnerRank = selectionHeap->top()->getRank();
                    return true;
                 }
                 else
                 {
                    if( paraDetTimer->getElapsedTime() >
                         ( paraParams->getRealParamValue(StopRacingTimeLimit) * paraParams->getRealParamValue(StopRacingTimeLimitMultiplier) )  )
                    {
                       winnerRank = selectionHeap->top()->getRank();
                       return true;
                    }
                 }
             }
             else
             {
                if( selectionHeap->top()->getNumOfNodesLeft() >
                   ( paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) * paraParams->getRealParamValue(StopRacingNumberOfNodesLeftMultiplier) ) )
                {
                   winnerRank = selectionHeap->top()->getRank();
                   return true;
                }
             }
          }
          else
          {
             if( paraTimer->getElapsedTime() > paraParams->getRealParamValue(StopRacingTimeLimit) )
             {
                if( selectionHeap->top()->getNumOfNodesLeft() > paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) )
                {
                   winnerRank = selectionHeap->top()->getRank();
                   return true;
                }
                else
                {
                   if( paraTimer->getElapsedTime() >
                        ( paraParams->getRealParamValue(StopRacingTimeLimit) * paraParams->getRealParamValue(StopRacingTimeLimitMultiplier) ) )
                   {
                      winnerRank = selectionHeap->top()->getRank();
                      return true;
                   }
                }
             }
             else
             {
                if( selectionHeap->top()->getNumOfNodesLeft() >
                   ( paraParams->getIntParamValue(StopRacingNumberOfNodesLeft) * paraParams->getRealParamValue(StopRacingNumberOfNodesLeftMultiplier) ) )
                {
                   winnerRank = selectionHeap->top()->getRank();
                   return true;
                }
             }
          }
       }
       break;
    }
    default:
       THROW_LOGICAL_ERROR1("invalid racing ramp-up termination criteria");
   }
   return false;
}

