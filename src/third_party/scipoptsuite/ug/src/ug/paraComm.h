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

/**@file    paraComm.h
 * @brief   Base class of communicator for UG Framework
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_COMM_H__
#define __PARA_COMM_H__
#include <sstream>
#include "paraDef.h"
#include "paraTagDef.h"
#include "paraParamSet.h"

namespace UG
{

#define PARA_COMM_CALL( paracommcall ) \
  { \
          int _status = paracommcall; \
          if( _status )  \
          { \
             std::ostringstream s_; \
             s_ << "[PARA_COMM_CALL ERROR: " << __FILE__  << "] func = " \
             << __func__ << ", line = " << __LINE__ << ": " \
             << "error_code = " << _status << std::endl; \
             throw std::logic_error( s_.str() ); \
          }\
  }

///
/// standard transfer data types
///
static const int TYPE_FIRST             =               0;
static const int ParaCHAR               = TYPE_FIRST +  0;
static const int ParaSHORT              = TYPE_FIRST +  1;
static const int ParaINT                = TYPE_FIRST +  2;
static const int ParaLONG               = TYPE_FIRST +  3;
static const int ParaLONG_LONG          = TYPE_FIRST +  4;
static const int ParaSIGNED_CHAR        = TYPE_FIRST +  5;
static const int ParaUNSIGNED_CHAR      = TYPE_FIRST +  6;
static const int ParaUNSIGNED_SHORT     = TYPE_FIRST +  7;
static const int ParaUNSIGNED           = TYPE_FIRST +  8;
static const int ParaUNSIGNED_LONG      = TYPE_FIRST +  9;
static const int ParaUNSIGNED_LONG_LONG = TYPE_FIRST + 10;
static const int ParaFLOAT              = TYPE_FIRST + 11;
static const int ParaDOUBLE             = TYPE_FIRST + 12;
static const int ParaLONG_DOUBLE        = TYPE_FIRST + 13;
static const int ParaBOOL               = TYPE_FIRST + 14;
static const int ParaBYTE               = TYPE_FIRST + 15;
static const int TYPE_LAST              = TYPE_FIRST + 15;
static const int TYPE_LIST_SIZE         = TYPE_LAST - TYPE_FIRST + 1;

static const int NumMaxWorkers          = 20000;

class ParaCalculationState;
class ParaParamSet;
class ParaSolverState;
class ParaSolverTerminationState;
class ParaInstance;
class ParaDiffSubproblem;
class ParaSolution;
class ParaInitialStat;
class ParaRacingRampUpParamSet;
class ParaTask;
class ParaTimer;
class TaskId;

///
/// Base class of communicator object
///
class ParaComm
{
public:

   ///
   /// default constructor of ParaComm
   ///
   ParaComm(
         )
   {
   }

   ///
   /// destructor of ParaComm
   ///
   virtual ~ParaComm(
         )
   {
   }

   ///
   /// initializer of this object
   ///
   virtual void init(
         int argc,        ///< number of arguments
         char **argv      ///< pointers to the arguments
         ) = 0;

   ///
   /// get rank of this process or this thread depending on run-time environment
   /// @return rank
   ///
   virtual int getRank() = 0;

   ///
   /// get number of UG processes or UG threads depending on run-time environment
   /// @return the number of UG processes or UG threads
   ///
   virtual int getSize() = 0;

   ///
   /// get size of the messageQueueTable
   /// @return the size of the messageQueueTable
   ///
   virtual int getNumOfMessagesWaitingToSend(
         int dest                         ///< destination of message
         ) = 0;

   ///
   /// special initializer when this object is used in LoadCoordinator
   ///
   virtual void lcInit(
         ParaParamSet *paraParamSet       ///< UG parameter set
         ) = 0;

   ///
   /// special initializer when this object is used in Solver
   ///
   virtual void solverInit(
         ParaParamSet *paraParamSet       ///< UG parameter set
         ) = 0;

   ///
   // set local rank in case of using a communicator for shared memory
   //
   virtual void setLocalRank(
         int inRank
         )
   {
   }

   ///
   /// abort function for this communicator
   /// (how to abort depends on which library used for communication)
   ///
   virtual void abort(
         ) = 0;

   ///
   /// function to wait Terminated message
   /// (This function is not used currently)
   /// @return true when MPI communication is used, false when thread communication used
   ///
   virtual bool waitTerminatedMessage(
         ) = 0;

   ///
   /// wait token when UG runs with deterministic mode
   /// @return true, when token is arrived to the rank
   ///
   virtual bool waitToken(
         int rank        ///< rank to check if token is arrived
         )
   {
      return true;
   }

   ///
   /// pass token to from the rank to the next
   ///
   virtual void passToken(
         int rank       ///< from this rank, the token is passed
         )
   {
   }

   ///
   /// pass termination token from the rank to the next
   /// @return true, when the termination token is passed from this rank, false otherwise
   ///
   virtual bool passTermToken(
         int rank       ///< from this rank, the termination token is passed
         )
   {
      return true;
   }

   ///
   /// set received token to this communicator
   ///
   virtual void setToken(
         int rank,     ///< rank to set the token
         int *token    ///< token to be set
         )
   {
   }

   ///
   /// lock UG application to synchronize with other threads
   ///
   virtual void lockApp(
         ) = 0;

   ///
   /// unlock UG application to synchronize with other threads
   ///
   virtual void unlockApp(
         ) = 0;

   /////////////////////////////////
   ///
   /// transfer object factory
   ///
   /// /////////////////////////////

   ///
   /// create ParaCalculationState object by default constructor
   /// @return pointer to ParaCalculationState object
   ///
   virtual ParaCalculationState *createParaCalculationState(
         ) = 0;


   ///
   /// create ParaRacingRampUpParamSet object
   /// @return pointer to ParaRacingRampUpParamSet object
   ///
   virtual ParaRacingRampUpParamSet* createParaRacingRampUpParamSet(
         ) = 0;

   ///
   /// create ParaTask object by default constructor
   /// @return pointer to ParaTask object
   ///
   virtual ParaTask *createParaTask(
         ) = 0;

   ///
   /// create ParaParamSet object
   /// @return pointer to ParaParamSet object
   ///
   virtual ParaParamSet *createParaParamSet(
         ) = 0;

   ///
   /// create ParaSolverState object by default constructor
   /// @return pointer to ParaSolverState object
   ///
   virtual ParaSolverState *createParaSolverState(
         ) = 0;

   ///
   /// create ParaSolverTerminationState object by default constructor
   /// @return pointer to ParaSolverTerminationState object
   ///
   virtual ParaSolverTerminationState *createParaSolverTerminationState(
         ) = 0;

   ///
   /// create ParaTimer object
   /// @return pointer to ParaTimer object
   ///
   virtual ParaTimer *createParaTimer(
         ) = 0;

   ///
   /// create ParaInstance object by default constructor
   /// @return pointer to ParaInstance object
   ///
   virtual ParaInstance *createParaInstance(
         ) = 0;

   ///
   /// create ParaSolution object by default constructor
   /// @return pointer to ParaSolution object
   ///
   virtual ParaSolution *createParaSolution(
         ) = 0;

   ///
   /// create ParaDiffSubproblem object by default constructor
   /// @return pointer to ParaDiffSubproblem object
   ///
   virtual ParaDiffSubproblem *createParaDiffSubproblem(
         )
   {
      THROW_LOGICAL_ERROR1("*** createParaDiffSubproblem() is called in ParaComm class ***");
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Some action need to be taken for fault tolerant, when the functions return.
   /// So, they rerun status value
   ////////////////////////////////////////////////////////////////////////////////

   ///
   /// broadcast function for standard ParaData types
   /// @return always 0 (for future extensions)
   ///
   virtual int bcast(
         void* buffer,              ///< point to the head of sending message
         int count,                 ///< the number of data in the message
         const int datatypeId,      ///< data type in the message
         int root                   ///< root rank for broadcasting
         ) = 0;

   ///
   /// send function for standard ParaData types
   /// @return always 0 (for future extensions)
   ///
   virtual int send(
         void* bufer,               ///< point to the head of sending message
         int count,                 ///< the number of data in the message
         const int datatypeId,      ///< data type in the message
         int dest,                  ///< destination to send the message
         const int tag              ///< tag of this message
         ) = 0;

   ///
   /// receive function for standard ParaData types
   /// @return always 0 (for future extensions)
   ///
   virtual int receive(
         void* bufer,               ///< point to the head of receiving message
         int count,                 ///< the number of data in the message
         const int datatypeId,      ///< data type in the message
         int source,                ///< source of the message coming from
         const int tag              ///< tag of the message
         ) = 0;

   ///
   /// wait function for a specific tag from a specific source coming from
   /// @return always 0 (for future extensions)
   ///
   virtual void waitSpecTagFromSpecSource(
         const int source,         ///< source rank which the message should come from
         const int tag,            ///< tag which the message should wait
         int *receivedTag          ///< tag of the message which is arrived
         ) = 0;

   //////////////////////////////////////////////////////////////////////////
   /// No need to take action for fault tolerant, when the functions return.
   /// So, they do not rerun status value
   //////////////////////////////////////////////////////////////////////////

   ///
   /// probe function which waits a new message
   /// @return always true
   ///
   virtual bool probe(
         int *source,             ///< source rank of the message arrived
         int *tag                 ///< tag of the message arrived
         ) = 0;

   ///
   /// iProbe function which checks if a new message is arrived or not
   /// @return true when a new message exists
   ///
   virtual bool iProbe(
         int *source,             ///< source rank of the message arrived
         int *tag                 ///< tag of the message arrived
         ) = 0;

   ///
   /// get Tag string for debugging
   /// @return string which shows Tag
   ///
   virtual const char *getTagString(
         int tag                          /// tag to be converted to string
         ) = 0;
};

}

#endif // __PARA_COMM_H__
