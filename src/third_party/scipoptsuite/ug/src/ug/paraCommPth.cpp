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

/**@file    paraCommPth.cpp
 * @brief   ParaComm extension for Pthreads communication
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <cstring>
#include <cstdlib>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include "paraCommPth.h"
#include "paraTask.h"
#include "paraSolution.h"
#include "paraSolverTerminationState.h"
#include "paraCalculationState.h"
#include "paraSolverState.h"
#include "paraRacingRampUpParamSet.h"
#include "paraInitialStat.h"

using namespace UG;

Lock rankGenLock;
ConditionVariable solverRanksGenerated(&rankGenLock);
ThreadsTableElement *
ParaCommPth::threadsTable[HashTableSize];

#ifndef _UG_NO_THREAD_LOCAL_STATIC
__thread int ParaCommPth::localRank = -1;     /*< local thread rank */
#endif

const char *
ParaCommPth::tagStringTable[] = {
  TAG_STR(TagTask),
  TAG_STR(TagTaskReceived),
  TAG_STR(TagDiffSubproblem),
  TAG_STR(TagRampUp),
  TAG_STR(TagSolution),
  TAG_STR(TagIncumbentValue),
  TAG_STR(TagSolverState),
  TAG_STR(TagCompletionOfCalculation),
  TAG_STR(TagNotificationId),
  TAG_STR(TagTerminateRequest),
  TAG_STR(TagInterruptRequest),
  TAG_STR(TagTerminated),
  TAG_STR(TagRacingRampUpParamSets),
  TAG_STR(TagWinner),
  TAG_STR(TagHardTimeLimit),
  TAG_STR(TagAckCompletion),
  TAG_STR(TagToken),
  TAG_STR(TagParaInstance)
};

void
ParaCommPth::init( int argc, char **argv )
{
   // don't have to take any lock, because only LoadCoordinator call this function

   timer.start();
   comSize = 0;

   for( int i = 1; i < argc; i++ )
   {
      if( strcmp(argv[i], "-sth") == 0 )
      {
         i++;
         if( i < argc )
            comSize = atoi(argv[i]);   // if -sth 0, then it is considered as use the number of cores system has
         else
         {
            std::cerr << "missing the number of solver threads after parameter '-sth'" << std::endl;
            exit(1);
         }
      }
   }

   if( comSize > 0 )
   {
      comSize++;
   }
   else
   {
      comSize = sysconf(_SC_NPROCESSORS_CONF) + 1;
   }

   tokenAccessLock = new Lock[comSize];
   token = new int*[comSize];
   for( int i = 0; i < comSize; i++ )
   {
      token[i] = new int[2];
      token[i][0] = 0;
      token[i][1] = -1;
   }

   /** if you add tag, you should add tagStringTale too */
   // assert( sizeof(tagStringTable)/sizeof(char*) == N_TH_TAGS );
   assert( tagStringTableIsSetUpCoorectly() );

   /** initialize hashtable */
   for(int i = 0; i < HashTableSize; i++ )
   {
      threadsTable[i] = 0;
   }

   messageQueueTable = new MessageQueueTableElement *[comSize + 1];  // +1 for TimeLimitMonitor
   sentMessage = new bool[comSize + 1];
   queueLock = new Lock[comSize + 1];
   sentMsg = new ConditionVariable[comSize + 1];
   for( int i = 0; i < ( comSize + 1 ); i++ )
   {
      messageQueueTable[i] = new MessageQueueTableElement;
      sentMsg[i].setLock(&queueLock[i]);
      sentMessage[i] = false;
   }

}

void
ParaCommPth::lcInit(
      ParaParamSet *paraParamSet
      )
{

   // pthread_t tid = pthread_self();

   // don't have to take any lock, because only LoadCoordinator call this function
   LOCKED (&rankGenLock ) {
#ifndef _UG_NO_THREAD_LOCAL_STATIC
      assert( localRank == -1 );
      assert( threadsTable[0] == 0 );
      localRank = 0;
      threadsTable[0] = new ThreadsTableElement(0, paraParamSet);
#else
      pthread_t tid = pthread_self();
      threadsTable[HashEntry(tid)] = new ThreadsTableElement(tid, 0, paraParamSet);
#endif
   }
   tagTraceFlag = paraParamSet->getBoolParamValue(TagTrace);
}

void
ParaCommPth::solverInit(
      int rank,
      ParaParamSet *paraParamSet
      )
{
   // don't have to take any lock, because only LoadCoordinator call this function
   // CHANGED in multi-threaded solver case
   LOCKED (&rankGenLock ) {
#ifndef _UG_NO_THREAD_LOCAL_STATIC
      assert( localRank == -1 );
      assert( threadsTable[rank] == 0 );
      localRank = rank;
      threadsTable[localRank] = new ThreadsTableElement(localRank, paraParamSet);
      // std::cout << "tid = " << pthread_self() << " is initialized as Rank = " << localRank << std::endl;
#else
      pthread_t tid = pthread_self();
      int index = HashEntry(tid);
      if( threadsTable[index] == 0 )
      {
         threadsTable[index] = new ThreadsTableElement(tid, rank, paraParamSet);
      }
      else
      {
         ThreadsTableElement *elem = threadsTable[index];
         while( elem->getNext() != 0 )
         {
            if( pthread_equal( elem->getTid(), tid ) )
            {
               THROW_LOGICAL_ERROR4("Invalid solver tid is registered. Rank = ", rank, ", tid = ", tid );
            }
            elem = elem->getNext();
         }
         elem->link(new ThreadsTableElement(tid, rank, paraParamSet));
      }
      // std::cout << "tid = " << tid << " for Rank = " << rank << " is added to table" << std::endl;
#endif
   }
}

void
ParaCommPth::solverReInit(
      int rank,
      ParaParamSet *paraParamSet
      )
{
   // don't have to take any lock, because only LoadCoordinator call this function
   // CHANGED in multi-threaded solver case
   LOCKED (&rankGenLock ) {
#ifndef _UG_NO_THREAD_LOCAL_STATIC
      assert( localRank == -1 );
      assert( threadsTable[rank] != 0 );
      localRank = rank;
      // threadsTable[localRank] = new ThreadsTableElement(localRank, paraParamSet);
#else
      // ****** CAUTION *******
      // Should not use this, since no chance to release ThreadsTableElement entry.
      // This means that new entries are always added and shuld not use.
      // ***********************
      THROW_LOGICAL_ERROR1("solverReInit only can work with thread local variable. The following routine should not be used!");
      pthread_t tid = pthread_self();
      int index = HashEntry(tid);
      if( threadsTable[index] == 0 )
      {
         threadsTable[index] = new ThreadsTableElement(tid, rank, paraParamSet);
      }
      else
      {
         ThreadsTableElement *elem = threadsTable[index];
         while( elem != 0 )
         {
            if( pthread_equal( elem->getTid(), tid ) )
            {
               elem->setRank(rank);
               break;
            }
            elem = elem->getNext();
         }
         if( elem == 0 )
         {
            THROW_LOGICAL_ERROR4("Invalid solver tid is reInit. Rank = ", rank, ", tid = ", tid );
         }
      }
      // std::cout << "tid = " << tid << " for Rank = " << rank << " is added to table" << std::endl;
#endif
   }

}

void
ParaCommPth::solverDel(
      int rank
      )
{
   LOCKED (&rankGenLock ) {
#ifndef _UG_NO_THREAD_LOCAL_STATIC
      assert(rank == localRank);
      if( threadsTable[rank] == 0 )
      {
         THROW_LOGICAL_ERROR2("Invalid remove thread. Rank = ", rank);
      }
      else
      {
         ThreadsTableElement *elem = threadsTable[rank];
         delete elem;
         threadsTable[rank] = 0;
         localRank = -1;
      }
#else
      pthread_t tid = pthread_self();
      int index = HashEntry(tid);
      if( threadsTable[index] == 0 )
      {
         THROW_LOGICAL_ERROR4("Invalid remove thread. Rank = ", rank, ", tid = ", tid );
      }
      else
      {
         ThreadsTableElement *elem = threadsTable[index];
         ThreadsTableElement *pre = elem;
         while( elem && !pthread_equal( elem->getTid(), tid ) )
         {
            pre = elem;
            elem = elem->getNext();
         }
         if( !elem || !pthread_equal( elem->getTid(), tid ) )
         {
            THROW_LOGICAL_ERROR4("Invalid remove thread. Rank = ", rank, ", tid = ", tid );
         }
         if( elem == threadsTable[index] )
         {
            threadsTable[index] = elem->getNext();
         }
         else
         {
            pre->link(elem->getNext());
         }
         delete elem;
      }
      // std::cout << "tid = " << tid << " is deleleted  from table" << std::endl;
#endif
   }
}

void
ParaCommPth::waitUntilRegistered(
      )
{
#ifdef _UG_NO_THREAD_LOCAL_STATIC
   pthread_t tid = pthread_self();
#endif

   bool registered = false;
   LOCKED(&rankGenLock){
      while( !registered )
      {
#ifndef _UG_NO_THREAD_LOCAL_STATIC
         if( threadsTable[localRank] != 0 )
         {
            registered = true;
            break;
         }
#else
         ThreadsTableElement *elem = threadsTable[HashEntry(tid)];
         while( elem && !pthread_equal( elem->getTid(), tid ) )
         {
            elem = elem->getNext();
         }
         if( elem )
         {
            assert( pthread_equal( elem->getTid(), tid ) );
            // std::cout << "tid = " << tid << ", hash = " << HashEntry(tid) << std::endl;
            registered = true;
            break;
         }
#endif
         solverRanksGenerated.wait();
      }
   }
}

void
ParaCommPth::registedAllSolvers(
      )
{
   solverRanksGenerated.broadcast();
}

bool
ParaCommPth::waitToken(
      int rank
      )
{
   // int rank = getRank();   // multi-thread solver may change rank here
   LOCK_RAII(&tokenAccessLock[rank]);
   if( token[rank][0] == rank )
   {
      return true;
   }
   else
   {
      int receivedTag;
      int source;
      probe(&source, &receivedTag);
      TAG_TRACE (Probe, From, source, receivedTag);
      if( source == 0 && receivedTag == TagToken )
      {
         receive(token[rank], 2, ParaINT, 0, TagToken);
         assert( token[rank][0] == rank );
         return true;
      }
      else
      {
         return false;
      }
   }
}

void
ParaCommPth::passToken(
      int rank
      )
{
   // int rank = getRank();   // multi-thread solver may change rank here
   LOCK_RAII(&tokenAccessLock[rank]);
   assert( token[rank][0] == rank && rank != 0 );
   token[rank][0] = ( token[rank][0]  % (comSize - 1) ) + 1;
   token[rank][1] = -1;
   send(token[rank], 2, ParaINT, 0, TagToken);
}

bool
ParaCommPth::passTermToken(
      int rank
      )
{
   // int rank = getRank();   // multi-thread solver may change rank here
   LOCK_RAII(&tokenAccessLock[rank]);
   if( rank == token[rank][0] )
   {
      if( token[rank][1] == token[rank][0] ) token[rank][1] = -2;
      else if( token[rank][1] == -1 ) token[rank][1] = token[rank][0];
      token[rank][0] = ( token[rank][0]  % (comSize - 1) ) + 1;
   }
   else
   {
      THROW_LOGICAL_ERROR4("Invalid token update. Rank = ", getRank(), ", token = ", token[0] );
   }
   send(token[rank], 2, ParaINT, 0, TagToken);
   if( token[rank][1] == -2 )
   {
      return true;
   }
   else
   {
      return false;
   }
}

void
ParaCommPth::setToken(
      int rank,
      int *inToken
      )
{
   // int rank = getRank();
   LOCK_RAII(&tokenAccessLock[rank]);
   assert( rank == 0 || ( rank != 0 && inToken[0] == rank ) );
   token[rank][0] = inToken[0];
   token[rank][1] = inToken[1];
}

ParaCommPth::~ParaCommPth()
{
   LOCK_RAII(&rankGenLock); // rankGenLock is not good name
   for(int i = 0; i < HashTableSize; i++ )
   {
      if( threadsTable[i] )
      {
         while( threadsTable[i] )
         {
            ThreadsTableElement  *next = threadsTable[i]->getNext();
            delete threadsTable[i];
            threadsTable[i] = next;
         }
      }
   }

   for( int i = 0; i < comSize; i++ )
   {
      delete [] token[i];
   }
   delete [] token;
   delete [] tokenAccessLock;

   for(int i = 0; i < (comSize + 1); i++)
   {
      MessageQueueElement *elem = messageQueueTable[i]->extarctElement(&sentMessage[i]);
      while( elem )
      {
         if( elem->getData() )
         {
            if( !freeStandardTypes(elem) )
            {
               ABORT_LOGICAL_ERROR2("Requested type is not implemented. Type = ", elem->getDataTypeId() );
            }
         }
         delete elem;
         elem = messageQueueTable[i]->extarctElement(&sentMessage[i]);
      }
      delete messageQueueTable[i];
   }
   delete [] messageQueueTable;

   if( sentMessage ) delete [] sentMessage;
   if( queueLock ) delete [] queueLock;
   if( sentMsg ) delete [] sentMsg;

}

unsigned int
ParaCommPth::hashCode(
      pthread_t tid
      )
{
   union {
      pthread_t tid;
      unsigned char      cTid[sizeof(pthread_t)];
   } reinterpret;

   reinterpret.tid = tid;
   unsigned int h = 0;
   for (unsigned int i = 0; i < sizeof(pthread_t); i++) {
       h = 31*h + reinterpret.cTid[i];
   }
   return h;
}

int
ParaCommPth::getRank(
      )
{
   LOCK_RAII(&rankGenLock);
#ifndef _UG_NO_THREAD_LOCAL_STATIC
   if( localRank >= 0 ) return localRank;
   else return -1;
#else
   pthread_t tid = pthread_self();
   // std::cout << "gerRank tid = " << tid << std::endl;
   ThreadsTableElement *elem = threadsTable[HashEntry(tid)];
   while( elem && !pthread_equal( elem->getTid(), tid ) )
   {
      elem = elem->getNext();
   }
   if( elem )
   {
      return elem->getRank();
   }
   else
   {
      return -1; // No ug threads
   }
#endif
}

std::ostream *
ParaCommPth::getOstream(
      )
{
   LOCK_RAII(&rankGenLock);
#ifdef _UG_NO_THREAD_LOCAL_STATIC
   pthread_t tid = pthread_self();
   ThreadsTableElement *elem = threadsTable[HashEntry(tid)];
   assert(elem);
   while( !pthread_equal( elem->getTid(), tid ) )
   {
      elem = elem->getNext();
      assert(elem);
   }
   return elem->getOstream();
#else
   return threadsTable[localRank]->getOstream();
#endif
}

void *
ParaCommPth::allocateMemAndCopy(
      const void* buffer,
      int count,
      const int datatypeId
      )
{
   void *newBuf = 0;
   if( count == 0 ) return newBuf;

   switch(datatypeId)
   {
   case ParaCHAR :
   {
      newBuf = new char[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(char)*count);
      break;
   }
   case ParaSHORT :
   {
      newBuf = new short[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(short)*count);
      break;
   }
   case ParaINT :
   {
      newBuf = new int[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(int)*count);
      break;
   }
   case ParaLONG :
   {
      newBuf = new long[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(long)*count);
      break;
   }
   case ParaUNSIGNED_CHAR :
   {
      newBuf = new unsigned char[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(unsigned char)*count);
      break;
   }
   case ParaUNSIGNED_SHORT :
   {
      newBuf = new unsigned short[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(unsigned short)*count);
      break;
   }
   case ParaUNSIGNED :
   {
      newBuf = new unsigned int[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(unsigned int)*count);
      break;
   }
   case ParaUNSIGNED_LONG :
   {
      newBuf = new unsigned long[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(unsigned long)*count);
      break;
   }
   case ParaFLOAT :
   {
      newBuf = new float[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(float)*count);
      break;
   }
   case ParaDOUBLE :
   {
      newBuf = new double[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(double)*count);
      break;
   }
   case ParaLONG_DOUBLE :
   {
      newBuf = new long double[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(long double)*count);
      break;
   }
   case ParaBYTE :
   {
      newBuf = new char[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(char)*count);
      break;
   }
   case ParaSIGNED_CHAR :
   {
      newBuf = new char[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(char)*count);
      break;
   }
   case ParaLONG_LONG :
   {
      newBuf = new long long[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(long long)*count);
      break;
   }
   case ParaUNSIGNED_LONG_LONG :
   {
      newBuf = new unsigned long long[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(unsigned long long)*count);
      break;
   }
   case ParaBOOL :
   {
      newBuf = new bool[count];
      memcpy(newBuf, buffer, (unsigned long int)sizeof(bool)*count);
      break;
   }
   default :
      THROW_LOGICAL_ERROR2("This type is not implemented. Type = ", datatypeId);
   }

   return newBuf;
}

void
ParaCommPth::copy(
      void *dest, const void *src, int count, int datatypeId
      )
{

   if( count == 0 ) return;

   switch(datatypeId)
   {
   case ParaCHAR :
   {
      memcpy(dest, src, (unsigned long int)sizeof(char)*count);
      break;
   }
   case ParaSHORT :
   {
      memcpy(dest, src, (unsigned long int)sizeof(short)*count);
      break;
   }
   case ParaINT :
   {
      memcpy(dest, src, (unsigned long int)sizeof(int)*count);
      break;
   }
   case ParaLONG :
   {
      memcpy(dest, src, (unsigned long int)sizeof(long)*count);
      break;
   }
   case ParaUNSIGNED_CHAR :
   {
      memcpy(dest, src, (unsigned long int)sizeof(unsigned char)*count);
      break;
   }
   case ParaUNSIGNED_SHORT :
   {
      memcpy(dest, src, (unsigned long int)sizeof(unsigned short)*count);
      break;
   }
   case ParaUNSIGNED :
   {
      memcpy(dest, src, (unsigned long int)sizeof(unsigned int)*count);
      break;
   }
   case ParaUNSIGNED_LONG :
   {
      memcpy(dest, src, (unsigned long int)sizeof(unsigned long)*count);
      break;
   }
   case ParaFLOAT :
   {
      memcpy(dest, src, (unsigned long int)sizeof(float)*count);
      break;
   }
   case ParaDOUBLE :
   {
      memcpy(dest, src, (unsigned long int)sizeof(double)*count);
      break;
   }
   case ParaLONG_DOUBLE :
   {
      memcpy(dest, src, (unsigned long int)sizeof(long double)*count);
      break;
   }
   case ParaBYTE :
   {
      memcpy(dest, src, (unsigned long int)sizeof(char)*count);
      break;
   }
   case ParaSIGNED_CHAR :
   {
      memcpy(dest, src, (unsigned long int)sizeof(char)*count);
      break;
   }
   case ParaLONG_LONG :
   {
      memcpy(dest, src, (unsigned long int)sizeof(long long)*count);
      break;
   }
   case ParaUNSIGNED_LONG_LONG :
   {
      memcpy(dest, src, (unsigned long int)sizeof(unsigned long long)*count);
      break;
   }
   case ParaBOOL :
   {
      memcpy(dest, src, (unsigned long int)sizeof(bool)*count);
      break;
   }
   default :
      THROW_LOGICAL_ERROR2("This type is not implemented. Type = ", datatypeId);
   }

}

void
ParaCommPth::freeMem(
      void* buffer,
      int count,
      const int datatypeId
      )
{

   if( count == 0 ) return;

   switch(datatypeId)
   {
   case ParaCHAR :
   {
      delete [] static_cast<char *>(buffer);
      break;
   }
   case ParaSHORT :
   {
      delete [] static_cast<short *>(buffer);
      break;
   }
   case ParaINT :
   {
      delete [] static_cast<int *>(buffer);
      break;
   }
   case ParaLONG :
   {
      delete [] static_cast<long *>(buffer);
      break;
   }
   case ParaUNSIGNED_CHAR :
   {
      delete [] static_cast<unsigned char *>(buffer);
      break;
   }
   case ParaUNSIGNED_SHORT :
   {
      delete [] static_cast<unsigned short *>(buffer);
      break;
   }
   case ParaUNSIGNED :
   {
      delete [] static_cast<unsigned int *>(buffer);
      break;
   }
   case ParaUNSIGNED_LONG :
   {
      delete [] static_cast<unsigned long *>(buffer);
      break;
   }
   case ParaFLOAT :
   {
      delete [] static_cast<float *>(buffer);
      break;
   }
   case ParaDOUBLE :
   {
      delete [] static_cast<double *>(buffer);
      break;
   }
   case ParaLONG_DOUBLE :
   {
      delete [] static_cast<long double *>(buffer);
      break;
   }
   case ParaBYTE :
   {
      delete [] static_cast<char *>(buffer);
      break;
   }
   case ParaSIGNED_CHAR :
   {
      delete [] static_cast<char *>(buffer);
      break;
   }
   case ParaLONG_LONG :
   {
      delete [] static_cast<long long *>(buffer);
      break;
   }
   case ParaUNSIGNED_LONG_LONG :
   {
      delete [] static_cast<unsigned long long *>(buffer);
      break;
   }
   case ParaBOOL :
   {
      delete [] static_cast<bool *>(buffer);;
      break;
   }
   default :
      THROW_LOGICAL_ERROR2("This type is not implemented. Type = ", datatypeId);
   }

}

bool
ParaCommPth::freeStandardTypes(
      MessageQueueElement *elem   ///< pointer to a message queue element
      )
{
   if( elem->getDataTypeId() < UG_USER_TYPE_FIRST )
   {
      freeMem(elem->getData(), elem->getCount(), elem->getDataTypeId() );
   }
   else
   {
      switch( elem->getDataTypeId())
      {
      case ParaInstanceType:
      {
         delete reinterpret_cast<ParaInstance *>(elem->getData());
         break;
      }
      case ParaSolutionType:
      {
         delete reinterpret_cast<ParaSolution *>(elem->getData());
         break;
      }
      case ParaParamSetType:
      {
         delete reinterpret_cast<ParaParamSet *>(elem->getData());
         break;
      }
      case ParaTaskType:
      {
         delete reinterpret_cast<ParaTask *>(elem->getData());
         break;
      }
      case ParaSolverStateType:
      {
         delete reinterpret_cast<ParaSolverState *>(elem->getData());
         break;
      }
      case ParaCalculationStateType:
      {
         delete reinterpret_cast<ParaCalculationState *>(elem->getData());
         break;
      }
      case ParaSolverTerminationStateType:
      {
         delete reinterpret_cast<ParaSolverTerminationState *>(elem->getData());
         break;
      }
      case ParaRacingRampUpParamType:
      {
         delete reinterpret_cast<ParaRacingRampUpParamSet *>(elem->getData());
         break;
      }
      default:
      {
         return false;
      }
      }
   }
   return true;
}

bool
ParaCommPth::tagStringTableIsSetUpCoorectly(
      )
{
   // std::cout << "size = " << sizeof(tagStringTable)/sizeof(char*) << ", N_TH_TAGS = " <<  N_TH_TAGS << std::endl;
   return ( sizeof(tagStringTable)/sizeof(char*) == N_TH_TAGS );
}

const char *
ParaCommPth::getTagString(
      int tag                 /// tag to be converted to string
      )
{
   assert( tag >= 0 && tag < N_TH_TAGS );
   return tagStringTable[tag];
}

int
ParaCommPth::bcast(
   void* buffer,
   int count,
   const int datatypeId,
   int root
   )
{
   if( getRank() == root )
   {
      for(int i=0; i < comSize; i++)
      {
         if( i != root )
         {
            send(buffer, count, datatypeId, i, -1);
         }
      }
   }
   else
   {
      receive(buffer, count, datatypeId, root, -1);
   }
   return 0;
}

int
ParaCommPth::send(
   void* buffer,
   int count,
   const int datatypeId,
   int dest,
   const int tag
   )
{
   LOCKED ( &queueLock[dest] )
   {
      messageQueueTable[dest]->enqueue(&sentMsg[dest],&sentMessage[dest],
            new MessageQueueElement(getRank(), count, datatypeId, tag,
                  allocateMemAndCopy(buffer, count, datatypeId) ) );
   }
   TAG_TRACE (Send, To, dest, tag);
   return 0;
}

int
ParaCommPth::receive(
   void* buffer,
   int count,
   const int datatypeId,
   int source,
   const int tag
   )
{
   int qRank = getRank();
   MessageQueueElement *elem = 0;
   if( !messageQueueTable[qRank]->checkElement(source, datatypeId, tag) )
   {
      messageQueueTable[qRank]->waitMessage(&sentMsg[qRank], &sentMessage[qRank], source, datatypeId, tag);
   }
   LOCKED ( &queueLock[qRank] )
   {
      elem = messageQueueTable[qRank]->extarctElement(&sentMessage[qRank],source, datatypeId, tag);
   }
   assert(elem);
   copy( buffer, elem->getData(), count, datatypeId );
   freeMem(elem->getData(), count, datatypeId );
   delete elem;
   TAG_TRACE (Recv, From, source, tag);
   return 0;
}

void
ParaCommPth::waitSpecTagFromSpecSource(
      const int source,
      const int tag,
      int *receivedTag
      )
{
   /*
   // Just wait, iProbe and receive will be performed after this call
   messageQueueTable[getRank()]->waitMessage(source, datatypeId, tag);
   TAG_TRACE (Probe, From, source, tag);
   return 0;
   */
   int qRank = getRank();
   // LOCKED ( &queueLock[getRank()] )
   // {
   (*receivedTag) = tag;
   messageQueueTable[qRank]->waitMessage(&sentMsg[qRank], &sentMessage[qRank], source, receivedTag);
   // }
   TAG_TRACE (Probe, From, source, *receivedTag);
   return;
}

bool
ParaCommPth::probe(
   int* source,
   int* tag
   )
{
   int qRank = getRank();
   messageQueueTable[qRank]->waitMessage(&sentMsg[qRank], &sentMessage[qRank]);
   MessageQueueElement *elem = messageQueueTable[qRank]->getHead();
   *source = elem->getSource();
   *tag = elem->getTag();
   TAG_TRACE (Probe, From, *source, *tag);
   return true;
}

bool
ParaCommPth::iProbe(
   int* source,
   int* tag
   )
{
   bool flag = false;
   int qRank = getRank();
   LOCKED ( &queueLock[qRank] )
   {
      if( *tag == TagAny )
      {
         flag = !(messageQueueTable[qRank]->isEmpty());
         if( flag )
         {
            MessageQueueElement *elem = messageQueueTable[qRank]->getHead();
            *source = elem->getSource();
            *tag = elem->getTag();
            TAG_TRACE (Iprobe, From, *source, *tag);
         }
      }
      else
      {
         MessageQueueElement *elem = messageQueueTable[qRank]->checkElementWithTag(*tag);
         if( elem )
         {
            *source = elem->getSource();
            *tag = elem->getTag();
            TAG_TRACE (Iprobe, From, *source, *tag);
            flag = true;
         }
      }
   }
   return flag;
}

int
ParaCommPth::uTypeSend(
   void* buffer,
   const int datatypeId,
   int dest,
   const int tag
   )
{
   LOCKED ( &queueLock[dest] )
   {
      messageQueueTable[dest]->enqueue(&sentMsg[dest],&sentMessage[dest],
            new MessageQueueElement(getRank(), 1, datatypeId, tag, buffer ) );
   }
   TAG_TRACE (Send, To, dest, tag);
   return 0;
}

int
ParaCommPth::uTypeReceive(
   void** buffer,
   const int datatypeId,
   int source,
   const int tag
   )
{
   int qRank = getRank();
   if( !messageQueueTable[qRank]->checkElement(source, datatypeId, tag) )
   {
      messageQueueTable[qRank]->waitMessage(&sentMsg[qRank], &sentMessage[qRank], source, datatypeId, tag);
   }
   MessageQueueElement *elem = 0;
   LOCKED ( &queueLock[qRank] )
   {

      elem = messageQueueTable[qRank]->extarctElement(&sentMessage[qRank], source, datatypeId, tag);
   }
   assert(elem);
   *buffer = elem->getData();
   delete elem;
   TAG_TRACE (Recv, From, source, tag);
   return 0;
}
