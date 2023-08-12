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

/**@file    paraCommMpi.h
 * @brief   ParaComm extension for MPI communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_COMM_MPI_H__
#define __PARA_COMM_MPI_H__

#include <thread>
#include <mutex>
#include <mpi.h>
#include <stdexcept>
#include <iostream>
#include <ostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <deque>
#include "paraDef.h"
#include "paraComm.h"
#include "paraInstance.h"
#include "paraDiffSubproblem.h"
#include "paraSolution.h"
// #include "paraCalculationStateMpi.h"
#include "paraParamSetMpi.h"
// #include "paraSolverStateMpi.h"
// #include "paraSolverTerminationStateMpi.h"
#include "paraTimerMpi.h"
#ifdef UG_WITH_UGS
#include "ugs/ugsParaCommMpi.h"
#endif 

namespace UG
{

#define MPI_CALL( mpicall ) do \
   { \
      int _error_value_; \
      _error_value_ = ( mpicall ); \
      if( _error_value_ != MPI_SUCCESS ) \
      { \
         std::cout << "[MPI ERROR: " << __FILE__  << "] func = " \
                   << __func__ << ", line = " << __LINE__ << ": " \
                   << "error_code = " << _error_value_ << std::endl; \
         MPI_Abort(MPI_COMM_WORLD, 1); \
      } \
   } while (0)

#define TAG_TRACE( call, fromTo, sourceDest, tag ) \
   if( tagTraceFlag )  \
   {  \
      *tos << (MPI_Wtime() - startTime) << " [Rank = " << myRank << "] " << #call << " " << #fromTo  \
      << " " << sourceDest << " with Tag = " << getTagString(tag) << std::endl; \
   }

class ParaIsendRequest;

///
/// Communicator object for MPI communications
///
class ParaCommMpi : public ParaComm
{
protected:
   MPI_Comm      myComm;                                 ///< MPI communicator
   int           myCommSize;                             ///< communicator size : number of processes joined in this system
   int           myRank;                                 ///< rank of this process
   int           namelen;                                ///< length of this process name
   char          procName[MPI_MAX_PROCESSOR_NAME];       ///< process name
   bool          tagTraceFlag;                           ///< indicate if tags are traced or not
   std::ofstream ofs;                                    ///< output file stream for tag trace
   std::ostream  *tos;                                   ///< output file stream for tag trace to change file name
   double        startTime;                              ///< start time of this communicator
   int           token[2];                               ///< index 0: token
                                                         ///< index 1: token color
                                                         ///<           -1: green
                                                         ///<           > 0: yellow ( termination origin solver number )
                                                         ///<           -2: red ( means the solver can terminate )
   static        MPI_Datatype datatypes[TYPE_LIST_SIZE]; ///< data type mapping table to MPI data type
   static const char *tagStringTable[];                  ///< table for tag name string

#ifdef _MUTEX_CPP11
   std::mutex                 tokenAccessLock;           ///< mutex for c++11 thread
#else
   pthread_mutex_t           tokenAccessLock;            ///< mutex for pthread thread
#endif
   std::mutex                applicationLockMutex;       ///< mutex for applications

#ifdef UG_WITH_UGS
   UGS::UgsParaCommMpi *commUgs;                         ///< communicator for UGS
#endif

   ///
   /// check if tag string table (for debugging) set up correctly
   /// @return true if tag string table is set up correctly, false otherwise
   ///
   virtual bool tagStringTableIsSetUpCoorectly(
         );

public:

   std::deque<ParaIsendRequest *> iSendRequestDeque;

   ///
   /// default constructor of ParaCommMpi
   ///
   ParaCommMpi(
         )
         : myComm(MPI_COMM_NULL),
           myCommSize(-1),
           myRank(-1),
           namelen(-1),
           tagTraceFlag(false),
           tos(0),
           startTime(0.0)
#ifdef UG_WITH_UGS
         , commUgs(0)
#endif
   {
#ifndef _MUTEX_CPP11
      pthread_mutex_init(&tokenAccessLock, NULL);
#endif
      token[0]=-1;
      token[1]=-1;
   }

   ///
   /// constructor of ParaCommMpi with MPI communicator
   ///
   ParaCommMpi(
         MPI_Comm comm             ///< my communicator
         )
         : myComm(comm),
           myCommSize(-1),
           myRank(-1),
           namelen(-1),
           tagTraceFlag(false),
           tos(0),
           startTime(0.0)
#ifdef UG_WITH_UGS
         , commUgs(0)
#endif
   {
   }

   ///
   /// destructor of this communicator
   ///
   virtual ~ParaCommMpi();

   ///
   /// getter of MPI_Comm
   ///
   MPI_Comm &getMpiComm(
         )
   {
      return myComm;
   }

   ///
   /// initializer of this communicator
   ///
   virtual void init(
         int argc,                ///< the number of arguments
         char **argv              ///< pointers to the arguments
         );

   ///
   /// get start time of this communicator
   /// @return start time
   ///
   double getStartTime(
         )
   {
      return startTime;
   }

   ///
   /// get rank of caller's thread
   /// @return rank of caller's thread
   ///
   int getRank(
         )
   {
      return myRank;
   }

   ///
   /// get size of this communicator, which indicates how many threads in a UG process
   /// @return the number of threads
   ///
   int getSize(
         )
   {
      return myCommSize;
   }

   ///
   /// get size of the messageQueueTable
   /// @return the size of the messageQueueTable
   ///
   int getNumOfMessagesWaitingToSend(
         int dest=-1
         )
   {
      return iSendRequestDeque.size();
   }

   ///
   /// initializer for LoadCoordinator
   ///
   void lcInit(
         ParaParamSet *paraParamSet    ///< UG parameter set
         );

   ///
   /// initializer for Solvers
   ///
   void solverInit(
         ParaParamSet *paraParamSet    ///< UG parameter set
         );

   ///
   /// abort. How it works sometimes depends on communicator used
   ///
   void abort(
         );

   ///
   /// function to wait Terminated message
   /// (This function is not used currently)
   /// @return true when MPI communication is used, false when thread communication used
   ///
   virtual bool waitTerminatedMessage(
         )
   {
      return true;
   }

   ///
   /// wait token when UG runs with deterministic mode
   /// @return true, when token is arrived to the rank
   ///
   virtual bool waitToken(
         int rank                     ///< rank to check if token is arrived
         );

   ///
   /// pass token to from the rank to the next
   ///
   virtual void passToken(
         int rank                     ///< from this rank, the token is passed
         );

   ///
   /// pass termination token from the rank to the next
   /// @return true, when the termination token is passed from this rank, false otherwise
   ///
   virtual bool passTermToken(
         int rank
         );

   ///
   /// set received token to this communicator
   ///
   virtual void setToken(
         int rank,                   ///< rank to set the token
         int *inToken                ///< token to be set
         )
   {
#ifdef _MUTEX_CPP11
      std::lock_guard<std::mutex> lock(tokenAccessLock);
#else
      pthread_mutex_lock(&tokenAccessLock);
#endif
      token[0] = inToken[0]; token[1] = inToken[1];
#ifndef _MUTEX_CPP11
      pthread_mutex_unlock(&tokenAccessLock);
#endif
   }

   ///
   /// lock UG application to synchronize with other threads
   ///
   void lockApp(
         )
   {
      applicationLockMutex.lock();
   }

   ///
   /// unlock UG application to synchronize with other threads
   ///
   void unlockApp(
         )
   {
      applicationLockMutex.unlock();
   }

   ///
   /// create ParaTimer object
   /// @return pointer to ParaTimer object
   ///
   ParaTimer *createParaTimer(
         )
   {
      return new ParaTimerMpi();
   }

   ///
   /// broadcast function for standard ParaData types
   /// @return always 0 (for future extensions)
   ///
   int bcast(
         void* buffer,           ///< point to the head of sending message
         int count,              ///< the number of data in the message
         const int datatypeId,   ///< data type in the message
         int root                ///< root rank for broadcasting
         );

   ///
   /// send function for standard ParaData types
   /// @return always 0 (for future extensions)
   ///
   virtual int send(
         void* bufer,            ///< point to the head of sending message
         int count,              ///< the number of data in the message
         const int datatypeId,   ///< data type in the message
         int dest,               ///< destination to send the message
         const int tag           ///< tag of this message
         );

   ///
   /// send function for standard ParaData types
   /// @return always 0 (for future extensions)
   ///
   int iSend(
         void* bufer,            ///< point to the head of sending message
         int count,              ///< the number of data in the message
         const int datatypeId,   ///< data type in the message
         int dest,               ///< destination to send the message
         const int tag,          ///< tag of this message
		   MPI_Request *req        ///< point to MPI_Request
         );

   ///
   /// receive function for standard ParaData types
   /// @return always 0 (for future extensions)
   ///
   int receive(
         void* bufer,            ///< point to the head of receiving message
         int count,              ///< the number of data in the message
         const int datatypeId,   ///< data type in the message
         int source,             ///< source of the message coming from
         const int tag           ///< tag of the message
         );

   ///
   /// wait function for a specific tag from a specific source coming from
   /// @return always 0 (for future extensions)
   ///
   void waitSpecTagFromSpecSource(
         const int source,       ///< source rank which the message should come from
         const int tag,          ///< tag which the message should wait
         int *receivedTag        ///< tag of the message which is arrived
         );

   ///
   /// probe function which waits a new message
   /// @return always true
   ///
   bool probe(
         int *source,            ///< source rank of the message arrived
         int *tag                ///< tag of the message arrived
         );

   ///
   /// iProbe function which checks if a new message is arrived or not
   /// @return true when a new message exists
   ///
   bool iProbe(
         int *source,           ///< source rank of the message arrived
         int *tag               ///< tag of the message arrived
         );

   ///
   /// User type bcast for created data type
   /// @return always 0 (for future extensions)
   ///
   int ubcast(
         void* buffer,             ///< point to the head of sending message
         int count,                ///< the number of created data type
         MPI_Datatype datatype,    ///< MPI data type
         int root                  ///< root rank for brodcasting
         );

   ///
   /// User type send for created data type
   /// @return always 0 (for future extensions)
   ///
   int usend(
         void* bufer,               ///< point to the head of sending message
         int count,                 ///< the number of created data type
         MPI_Datatype datatype,     ///< created data type
         int dest,                  ///< destination rank
         int tag                    ///< tag of the message
         );

   ///
   /// User type send for created data type
   /// @return always 0 (for future extensions)
   ///
   int iUsend(
         void* bufer,               ///< point to the head of sending message
         int count,                 ///< the number of created data type
         MPI_Datatype datatype,     ///< created data type
         int dest,                  ///< destination rank
         int tag,                   ///< tag of the message
		 MPI_Request *req           ///< point to MPI_Request
         );

   ///
   /// User type receive for created data type
   /// @return always 0 (for future extensions)
   ///
   int ureceive(
         void* bufer,               ///< point to the head of receiving message
         int count,                 ///< the number of created data type
         MPI_Datatype datatype,     ///< created data type
         int source,                ///< source rank
         int tag                    ///< tag of the message
         );

   int testAllIsends(
         );

   void waitAllIsends();

   ///
   /// get Tag string for debugging
   /// @return string which shows Tag
   ///
   virtual const char *getTagString(
         int tag                          /// tag to be converted to string
         );

};

#define DEF_PARA_COMM( para_comm, comm ) ParaCommMpi *para_comm = dynamic_cast< ParaCommMpi* >(comm)

}

#endif  // __PARA_COMM_MPI_H__
