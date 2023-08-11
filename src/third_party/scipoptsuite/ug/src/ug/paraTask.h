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

/**@file    paraTask.h
 * @brief   Base class for ParaTask.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_TASK_H__
#define __PARA_TASK_H__

#include <cassert>
#include <iostream>
#include <fstream>
#include <map>
#include "paraDef.h"
#include "paraComm.h"
#ifdef UG_WITH_ZLIB
#include "gzstream.h"
#endif
#include "paraDiffSubproblem.h"

namespace UG
{

static const int ParaTaskLocalPtr  = 0;
static const int ParaTaskRemotePtr = 1;

///
///  SubtaskId class
///
class SubtaskId
{
public:
   int lcId;                     ///< LoadCoordinator ID
   int globalSubtaskIdInLc;      ///< Global Subtask ID in Solvers managed by LoadCoordinator
   int solverId;                 ///< Solver ID

   ///
   ///  default constructor
   ///
   SubtaskId(
         )
         : lcId(-1),
           globalSubtaskIdInLc(-1),
           solverId(-1)
   {
   }

   ///
   ///  copy constructor
   ///
   SubtaskId(
         const SubtaskId& subtreeId
         )
   {
      lcId = subtreeId.lcId;
      globalSubtaskIdInLc = subtreeId.globalSubtaskIdInLc;
      solverId = subtreeId.solverId;
   }

   ///
   /// constructor
   ///
   SubtaskId(
         int inLcId,
         int inGlobalSubtreeIdInLc,
         int inSolverId
         )
        : lcId(inLcId),
          globalSubtaskIdInLc(inGlobalSubtreeIdInLc),
          solverId(inSolverId)
   {
   }

   ///
   ///  destructor
   ///
   ~SubtaskId(
         )
   {
   }

   ///
   /// = operator definition
   /// @return reference to SubtreeId
   ///
   SubtaskId& operator=(
         const UG::SubtaskId& subTreeId
         )
   {
      lcId = subTreeId.lcId;
      globalSubtaskIdInLc = subTreeId.globalSubtaskIdInLc;
      solverId = subTreeId.solverId;
      return *this;
   }

   ///
   /// == operator definition
   /// @return true if this task subtree id is equla to that given by argument
   ///
   bool operator == (
         const SubtaskId& inSid       ///< subtree id
         ) const
   {
      if( lcId == inSid.lcId &&
            globalSubtaskIdInLc == inSid.globalSubtaskIdInLc &&
            solverId == inSid.solverId )
         return true;
      else
         return false;
   }

   ///
   /// != operator definition
   /// @return true if this task subtree id is not that given by argument
   ///
   bool operator != (
         const SubtaskId& inSid       ///< subtree id
         ) const
   {
      if( lcId != inSid.lcId ||
            globalSubtaskIdInLc != inSid.globalSubtaskIdInLc ||
            solverId != inSid.solverId )
         return true;
      else
         return false;
   }

   ///
   /// < operator definition
   /// @return true if this task subtree id is less than that given by argument
   ///
   bool operator < (
         const SubtaskId& inSid        ///< subtree id
         ) const
   {
      if( lcId < inSid.lcId )
         return false;
      if( lcId == inSid.lcId && globalSubtaskIdInLc < inSid.globalSubtaskIdInLc )
         return true;
      if( lcId == inSid.lcId && globalSubtaskIdInLc == inSid.globalSubtaskIdInLc && solverId < inSid.solverId )
         return true;
      else
         return false;
   }

   ///
   /// getter of LoadCoordinator id
   /// @return LoadCoordinator id
   ///
   int getLcId(
         ) const
   {
      return lcId;
   }

   ///
   /// getter of global subtree id in Solvers managed by the LoadCoordinator
   /// @return global subtree id in LC
   ///
   int getGlobalSubtreeIdInLc(
         ) const
   {
      return globalSubtaskIdInLc;
   }

   ///
   /// getter of Solver id
   /// @return Solver id
   ///
   int getSolverId(
         ) const
   {
      return solverId;
   }

   ///
   /// Stringfy SubtreeId
   /// @return sting to show inside of this object
   ///
   std::string toString(
         )
   {
      std::ostringstream s;
      s << "(" << lcId << "," << globalSubtaskIdInLc << "," << solverId << ")";
      return s.str();
   }
};

///
///  TaskId class
///
class TaskId
{

public:

    SubtaskId subtaskId;            ///< subtree id
    long long seqNum;               ///< sequential number in the subtree

    ///
    ///  default constructor
    ///
    TaskId(
          )
          : subtaskId(SubtaskId()),
            seqNum(-1)
    {
    }

    ///
    ///  copy constructor
    ///
    TaskId(
          const TaskId& taskId
          )
    {
       subtaskId = taskId.subtaskId;
       seqNum = taskId.seqNum;
    }

    ///
    /// constructor
    ///
    TaskId(
          SubtaskId inSubtreeId,    ///< subtree id
          int inSeqNum              ///< sequential number in the subtree
          )
          : subtaskId(inSubtreeId),
            seqNum(inSeqNum)
    {
    }

    ///
    /// destructor
    ///
    ~TaskId(
          )
    {
    }

    ///
    /// = operator definition
    /// @return reference to TaskId
    ///
    TaskId& operator=(
          const UG::TaskId& taskId
          )
    {
       subtaskId = taskId.subtaskId;
       seqNum = taskId.seqNum;
       return *this;
    }

    ///
    /// == operator definition
    /// @return true if task id is equal to that given by argument
    ///
    bool operator == (
          const TaskId& inNid               ///< task id
          ) const
   {
      if( subtaskId == inNid.subtaskId &&
             seqNum == inNid.seqNum )
          return true;
      else
           return false;
   }

   ///
   /// != operator definition
   /// @return true if task id is not equal to that given by argument
   ///
   bool operator != (
         const TaskId& inNid
         ) const
   {
      if( subtaskId != inNid.subtaskId ||
            seqNum != inNid.seqNum )
         return true;
      else
         return false;
   }

   ///
   /// < operator definition
   /// @return true if task is less than that given by argument
   ///
   bool operator < (
         const TaskId& inNid
         ) const
   {
      if( subtaskId < inNid.subtaskId ) return true;
      if( subtaskId == inNid.subtaskId && seqNum < inNid.seqNum )
         return true;
      else
         return false;
   }

   ///
   /// getter of subtask id
   /// @return the subtask id
   ///
   SubtaskId getSubtaskId(
         ) const
   {
      return subtaskId;
   }

   ///
   /// getter of sequence number
   /// @return the sequence number
   ///
   long long getSeqNum(
         ) const
   {
      return seqNum;
   }

   ///
   /// stringfy task id
   /// @return string to show the task id
   ///
   std::string toString(
         )
   {
      std::ostringstream s;
      // for debug
      s << "[" << (subtaskId.toString()) << ":" <<  seqNum << "]";
      return s.str();
   }
};


class ParaTask;

///
///  class of pointer to indicate a ParaTask genealogical relation
///
class ParaTaskGenealogicalPtr
{
   TaskId   genealogicalTaskId;             ///< descendant TaskId or ascendant TaskId

public:

   ///
   /// constructor
   ///
   ParaTaskGenealogicalPtr(
         TaskId taskId                     ///< task id
         )
         : genealogicalTaskId(taskId)
   {
   }

   ///
   /// destructor
   ///
   virtual ~ParaTaskGenealogicalPtr(
         )
   {
   }

   ///
   /// getter type which indicate the pointer is local or remote
   /// @return type, 0: local, 1: remote
   ///
   virtual int getType() = 0;

   ///
   /// getter of genealogicaltaskId
   /// @return the genealogicalTaskId
   ///
   TaskId getTaskId(
         )
   {
      return genealogicalTaskId;
   }

};

///
/// class ParaTaskGenealogicalLocalPtr
///
class ParaTaskGenealogicalLocalPtr : public ParaTaskGenealogicalPtr
{

   ParaTask     *paraTaskPtr;        ///< pointer to ParaTask

public:

   ///
   /// default constructor
   ///
   ParaTaskGenealogicalLocalPtr(
         )
         : ParaTaskGenealogicalPtr(TaskId()),
           paraTaskPtr(0)
   {
   }

   ///
   /// constructor
   ///
   ParaTaskGenealogicalLocalPtr(
         TaskId taskId,                    ///< task id
         ParaTask *ptr                     ///< pointer to ParaTask
         )
         : ParaTaskGenealogicalPtr(taskId),
           paraTaskPtr(ptr)
   {
   }

   ///
   /// destructor
   ///
   ~ParaTaskGenealogicalLocalPtr(
         )
   {
   }

   ///
   /// getter of pointer type
   /// @return 0: local task pointer
   ///
   int getType(
         )
   {
      return ParaTaskLocalPtr;
   }

   ///
   /// getter for ParaTask pointer
   /// @return the task pointer
   ///
   ParaTask *getPointerValue(
         )
   {
      return paraTaskPtr;
   }

};

///
/// class ParaTaskGenealogicalRemotePtr
/// @note this class would not be used currently. This class is for future extension
///
class ParaTaskGenealogicalRemotePtr : public ParaTaskGenealogicalPtr
{

   int       transferringLcId;          ///< LoadCoordinator id that transfers to or is transferred from

public:

   ///
   /// default constructor
   ///
   ParaTaskGenealogicalRemotePtr(
         )
         : ParaTaskGenealogicalPtr(TaskId()),
           transferringLcId(-1)
   {
   }

   ///
   /// constructor
   ///
   ParaTaskGenealogicalRemotePtr(
         TaskId taskId,
         int lcId
         )
         : ParaTaskGenealogicalPtr(taskId),
           transferringLcId(lcId)
   {
   }

   ///
   /// destructor
   ///
   ~ParaTaskGenealogicalRemotePtr(
         )
   {
   }

   ///
   /// getter of pointer type
   /// @return 1: remote
   ///
   int getType(
         )
   {
      return ParaTaskRemotePtr;
   }

   ///
   /// getter of the pointer value
   /// @return LC id
   ///
   int getPointerValue(
         )
   {
      return transferringLcId;
   }

};

typedef ParaTaskGenealogicalPtr *ParaTaskGenealogicalPtrPtr;

///
/// class ParaTask
///
class ParaTask
{

public:

   ///
   /// solving task information
   ///
   TaskId          taskId;                                     ///< solving task id
   TaskId          generatorTaskId;                            ///< subtree root task id of generator
   ParaTaskGenealogicalPtr *ancestor;                          ///< pointer to ancestor ParaTask : This field is not transferred
   std::map< TaskId, ParaTaskGenealogicalPtrPtr > descendants; ///< collection of pointers to descendants : This filed is not transferred

protected:

   double          estimatedValue;         ///< estimate value
   int             diffSubproblemInfo;     ///< 1: with diffSubproblem, 0: no diffSubproblem
   ParaDiffSubproblem *diffSubproblem;     ///< difference between solving instance data and subproblem data

public:

   ///
   /// default constructor
   ///
   ParaTask(
         )
         : taskId(TaskId()),
           generatorTaskId(TaskId()),
           ancestor(0),
           estimatedValue(0.0),
           diffSubproblemInfo(0),
           diffSubproblem(0)
   {
   }

   ///
   ///  copy constructor
   ///
   ParaTask(
         const ParaTask& paraTask
         )
   {
        taskId = paraTask.taskId;
        generatorTaskId = paraTask.generatorTaskId;
        ancestor = paraTask.ancestor;
        estimatedValue = paraTask.estimatedValue;
        diffSubproblemInfo = paraTask.diffSubproblemInfo;
        diffSubproblem = paraTask.diffSubproblem;
   }

   ///
   ///  constructor
   ///
   ParaTask(
         TaskId inTaskId,                         ///< task id
         TaskId inGeneratorTaskId,                ///< generator task id
         double inEstimatedValue,                 ///< estimated value
         ParaDiffSubproblem *inDiffSubproblem     ///< pointer to ParaDiffSubproblem object
         )
         : taskId(inTaskId),
           generatorTaskId(inGeneratorTaskId),
           ancestor(0),
           estimatedValue(inEstimatedValue),
           diffSubproblem(inDiffSubproblem)

   {
      if( diffSubproblem ) diffSubproblemInfo = 1;
      else diffSubproblemInfo = 0;
   }

   ///
   ///  destructor
   ///
   virtual ~ParaTask(
         )
   {
      if( diffSubproblem ) delete diffSubproblem;
   }

   ///
   /// check if root task or not
   /// @return true if this task is the root task
   ///
   bool isRootTask(
         )
   {
      // we want to know on which solver on which LC is managed is to generate the root
      if( taskId.subtaskId.lcId == -1 &&
            taskId.subtaskId.globalSubtaskIdInLc == -1 &&
            taskId.subtaskId.solverId == -1 &&
            taskId.seqNum == -1 ) return true;
      else return false;
   }

   ///
   /// check if this task id is the same as argument ParaTask's task id
   /// @return true if the both are the same
   ///
   bool isSameTaskIdAs(
         const ParaTask& inTask       ///< ParaTask
         )
   {
      if( taskId == inTask.taskId ) return true;
      else return false;
   }

   ///
   /// check if this task's parent id is the same as that of argument ParaTask's task id
   /// @return true if the both are the same
   ///
   bool isSameParetntTaskIdAs(
         const ParaTask& inTask      ///< ParaTask
         )
   {
      if( generatorTaskId == inTask.generatorTaskId ) return true;
      else return false;
   }

   ///
   /// check if this task's parent subtree id is the same as that of argument ParaTask's task id
   /// @return true if the both are the same
   ///
   bool isSameParetntTaskSubtaskIdAs(
         const TaskId& inTaskId      ///< ParaTask id
         )
   {
      if( generatorTaskId.subtaskId == inTaskId.subtaskId ) return true;
      else return false;
   }

   ///
   /// check if this task's subtask id is the same as that of argument ParaTask's task id
   /// @return true if the both are the same
   ///
   bool isSameSubtaskIdAs(
         const ParaTask& inTask      ///< ParaTask
         )
   {
      if( taskId.subtaskId == inTask.taskId.subtaskId )
         return true;
      else return false;
   }

   ///
   /// check if this task's global subtask id in LC is the same as that of argument ParaTask's task id
   /// @return true if the both are the same
   ///
   bool isSameLcIdAs(
         const ParaTask& inTask       ///< ParaTask
         )
   {
      if( taskId.subtaskId.lcId
            == inTask.taskId.subtaskId.lcId )
         return true;
         else return false;
   }

   ///
   /// check if this task's global subtask LC id is the same as LC id of argument
   /// @return true if the both are the same
   ///
   bool isSameLcIdAs(
         const int lcId                ///< LC id
         )
   {
      if( taskId.subtaskId.lcId == lcId )
         return true;
      else return false;
   }

   ///
   /// check if this task's global subtask id in LC is the same as that of argument ParaTask's task id
   /// @return true if the both are the same
   ///
   bool isSameGlobalSubtaskIdInLcAs(
         const ParaTask& inTask        ///< ParaTask
         )
   {
      if( taskId.subtaskId.globalSubtaskIdInLc
            == inTask.taskId.subtaskId.globalSubtaskIdInLc )
         return true;
      else return false;
   }

   ///
   /// check if this task's global subtask id in LC is the same as that of argument
   /// @return true if the both are the same
   ///
   bool isSameGlobalSubtaskIdInLcAs(
         const int globalSubtaskIdInLc  ///< global subtask id in LC
         )
   {
      if( taskId.subtaskId.globalSubtaskIdInLc == globalSubtaskIdInLc )
         return true;
      else return false;
   }

   ///
   /// getter of  LoadCoordinator id
   /// @return LoadCoordinator id
   ///
   int getLcId(
         )
   {
      return taskId.subtaskId.lcId;
   }

   ///
   /// getter of global subtask id in Solvers managed by LoadCoordinator
   /// @return global subtask id
   ///
   int getGlobalSubtaskIdInLc(
         )
   {
      return taskId.subtaskId.globalSubtaskIdInLc;
   }

   ///
   /// setter of global subtask id
   ///
   void setGlobalSubtaskId(
         int lcId,                 ///< LoadCorrdinaor id
         int subtaskId             ///< subtask id
         )
   {
      taskId.subtaskId.lcId = lcId;
      taskId.subtaskId.globalSubtaskIdInLc = subtaskId;
   }

   ///
   /// getter of Solver id
   /// @return Solver id
   ///
   int getSolverId(
         )
   {
      return taskId.subtaskId.solverId;
   }

   ///
   /// setter of Solver id
   ///
   void setSolverId(
         int id                  ///< solver id
         )
   {
      taskId.subtaskId.solverId = id;
   }

   ///
   /// getter of task id
   /// @return task id
   ///
   TaskId getTaskId(
         )
   {
      return taskId;
   }

   ///
   /// setter of task id
   ///
   void setTaskId(
         TaskId inTaskId        ///< task id
         )
   {
      taskId = inTaskId;
   }

   ///
   /// getter of generator task id
   /// @return generator task id
   ///
   TaskId getGeneratorTaskId(
         )
   {
      return generatorTaskId;
   }

   ///
   /// setter of generator task id
   ///
   void setGeneratorTaskId(
         TaskId inGeneratorTaskId   ///< generator task id
         )
   {
      generatorTaskId = inGeneratorTaskId;
   }

   ///
   /// getter of estimated value
   /// @return estimated value
   ///
   double getEstimatedValue(
         )
   {
      return estimatedValue;
   }

   ///
   /// setter of estimated value
   ///
   void setEstimatedValue(
         double inEstimatedValue      ///< estimated value
         )
   {
      estimatedValue = inEstimatedValue;
   }

   ///
   /// getter of diffSubproblem
   /// @return pointer to ParaDiffSubproblem object
   ///
   ParaDiffSubproblem *getDiffSubproblem(
         )
   {
      assert( ((!diffSubproblem) && (!diffSubproblemInfo)) || (diffSubproblem && diffSubproblemInfo) );
      return diffSubproblem;
   }

   ///
   /// setter of diffSubproblem */
   ///
   void setDiffSubproblem(
         ParaDiffSubproblem *inDiffSubproblem    ///< pointer to ParaDiffSubproblem object
         )
   {
      diffSubproblem = inDiffSubproblem;
      diffSubproblemInfo = 1;
   }

   ///
   /// getter of ancestor
   /// @return ancestor ParaTaskGenealogicalPtr
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
         ParaTaskGenealogicalPtr *inAncestor   ///< ancestor ParaTaskGenealogicalPtr
         )
   {
      if( ancestor ) delete ancestor;
      ancestor = inAncestor;
   }

   ///
   /// remove a descendant
   ///
   void removeDescendant(
         TaskId removeTaskId                  ///<  task id to remove
         )
   {
      std::map< TaskId, ParaTaskGenealogicalPtrPtr >::iterator pos;
      pos = descendants.find(removeTaskId);
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
               std::cout << "Descendant TaskId = " << localPtrDescendant->getTaskId().toString() << std::endl;
            }
            else
            {
               /** not implemented yet */
            }
            pos++;
         }
         THROW_LOGICAL_ERROR1("invalid TaskId removed!");
      }
   }

   ///
   /// check if this task has descendant or not
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
         ParaTaskGenealogicalPtr *inDescendant   ///< descendant ParaTaskGenealogicalPtr
         )
   {
      descendants.insert(std::make_pair(inDescendant->getTaskId(),inDescendant));
   }

   ///
   /// clone this ParaTask
   /// @return pointer to cloned ParaTask object
   ///
   virtual ParaTask* clone(
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

#ifdef UG_WITH_ZLIB

   ///
   /// write to checkpoint file
   ///
   virtual void write(
         gzstream::ogzstream &out   ///< gzstream for output
         ) = 0;

#endif

   ///
   /// stringfy ParaTask
   /// @return string to show inside of this object
   ///
   virtual const std::string toString(
         )
   {
      std::ostringstream s;
      s << "ParaTaskId = " << (taskId.toString()) << ", GeneratorTaskId = " << (generatorTaskId.toString())
      // << ", depth = " << depth << ", dual bound value = " << dualBoundValue
      // << ", initialDualBoundValue = " << initialDualBoundValue
      << ", estimated value = " << estimatedValue << std::endl;
      if( diffSubproblem )
      {
         s << diffSubproblem->toString();
      }
      return s.str();
   }

   ///
   /// stringfy ParaTask as simple string
   /// @return string to show inside of this object
   ///
   virtual const std::string toSimpleString(
         )
   {
      std::ostringstream s;
      s << taskId.toString()
            << ", "
            << generatorTaskId.toString();
      return s.str();
   }

};

typedef ParaTask *ParaTaskPtr;

}

#endif // __PARA_TASK_H__

