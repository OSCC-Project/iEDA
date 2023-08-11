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

/**@file    paraCommMpi.cpp
 * @brief   ParaComm extension for MPI communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <cassert>
#include <cstring>
#include "paraTagDef.h"
#include "paraCommMpi.h"
#include "paraIsendRequest.h"

using namespace UG;

MPI_Datatype
ParaCommMpi::datatypes[TYPE_LIST_SIZE];

const char *
ParaCommMpi::tagStringTable[] = {
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
  TAG_STR(TagToken)
};

void
ParaCommMpi::lcInit(
      ParaParamSet *paraParamSet
      )
{
   tagTraceFlag = paraParamSet->getBoolParamValue(TagTrace);
   if( tagTraceFlag )
   {
      if( paraParamSet->isStringParamDefaultValue(TagTraceFileName) )
      {
         tos = &std::cout;
      }
      else
      {
         std::ostringstream s;
         s << paraParamSet->getStringParamValue(TagTraceFileName) << myRank;
         ofs.open(s.str().c_str());
         tos = &ofs;
      }
   }
   if( paraParamSet->getBoolParamValue(Deterministic) )
   {
      token[0] = 0;
      token[1] = -1;
   }
}

void
ParaCommMpi::solverInit(
      ParaParamSet *paraParamSet
      )
{
   tagTraceFlag = paraParamSet->getBoolParamValue(TagTrace);
   if( tagTraceFlag )
   {
      if( paraParamSet->isStringParamDefaultValue(TagTraceFileName) )
      {
         tos = &std::cout;
      }
      else
      {
         std::ostringstream s;
         s << paraParamSet->getStringParamValue(TagTraceFileName) << myRank;
         ofs.open(s.str().c_str());
         tos = &ofs;
      }
   }
}

void
ParaCommMpi::abort(
      )
{
   MPI_Abort(MPI_COMM_WORLD, 0);
}

bool
ParaCommMpi::waitToken(
      int tempRank
      )
{
#ifdef _MUTEX_CPP11
   std::lock_guard<std::mutex> lock(tokenAccessLock);
#else
   pthread_mutex_lock(&tokenAccessLock);
#endif
   if( token[0] == myRank )
   {
#ifndef _MUTEX_CPP11
      pthread_mutex_unlock(&tokenAccessLock);
#endif
      return true;
   }
   else
   {
      int previousRank = myRank - 1;
      if( previousRank == 0 )
      {
         if( token[0] != -1 )
         {
            previousRank = myCommSize - 1;
         }
      }
      int receivedTag;
      MPI_Status mpiStatus;
      MPI_CALL (
         MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, myComm, &mpiStatus)
      );
      receivedTag = mpiStatus.MPI_TAG;
      TAG_TRACE (Probe, From, mpiStatus.MPI_SOURCE, receivedTag);
      if( receivedTag == TagToken )
      {
         receive(token, 2, ParaINT, 0, TagToken);
         assert(token[0] == myRank);
#ifndef _MUTEX_CPP11
         pthread_mutex_unlock(&tokenAccessLock);
#endif
         return true;
      }
      else
      {
#ifndef _MUTEX_CPP11
         pthread_mutex_unlock(&tokenAccessLock);
#endif
         return false;
      }
   }
}

void
ParaCommMpi::passToken(
      int tempRank
      )
{
#ifdef _MUTEX_CPP11
   std::lock_guard<std::mutex> lock(tokenAccessLock);
#else
   pthread_mutex_lock(&tokenAccessLock);
#endif
   assert( token[0] == myRank );
   token[0] = ( token[0]  % (myCommSize - 1) ) + 1;
   token[1] = -1;
   send(token, 2, ParaINT, 0, TagToken);
#ifndef _MUTEX_CPP11
   pthread_mutex_unlock(&tokenAccessLock);
#endif
}

bool
ParaCommMpi::passTermToken(
      int tempRank
      )
{
#ifdef _MUTEX_CPP11
   std::lock_guard<std::mutex> lock(tokenAccessLock);
#else
   pthread_mutex_lock(&tokenAccessLock);
#endif
   if( myRank == token[0] )
   {
      if( token[1] == token[0] ) token[1] = -2;
      else if( token[1] == -1 ) token[1] = token[0];
      token[0] = ( token[0]  % (myCommSize - 1) ) + 1;
   }
   else
   {
      THROW_LOGICAL_ERROR4("Invalid token update. Rank = ", getRank(), ", token = ", token[0] );
   }
   send(token, 2, ParaINT, 0, TagToken);
   if( token[1] == -2 )
   {
#ifndef _MUTEX_CPP11
      pthread_mutex_unlock(&tokenAccessLock);
#endif
      return true;
   }
   else
   {
#ifndef _MUTEX_CPP11
      pthread_mutex_unlock(&tokenAccessLock);
#endif
      return false;
   }
}

/// MPI call wrappers */
void
ParaCommMpi::init( int argc, char **argv )
{

#ifdef UG_WITH_UGS
   if( !commUgs )
   {
      MPI_Init( &argc, &argv );

//
// To test if MPI support MPI_THREAD_MULTIPLE
//
//      int provided;
//      MPI_CALL(
//         MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided)
//      );
//      if (provided < MPI_THREAD_MULTIPLE)
//      {
//         std::cerr << "Error: the MPI library doesn't provide the required thread level" << std::endl;
//         MPI_Abort(MPI_COMM_WORLD, 0);
//      }

   }
#else
//   MPI_Init( &argc, &argv );

//
// To test if MPI support MPI_THREAD_MULTIPLE
//
   int provided, claimed;
   MPI_CALL(
      MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided)
   );
   MPI_Query_thread( &claimed );
   // printf( "Query thread level= %d  Init_thread level= %d\n", claimed, provided );
   assert(provided ==  MPI_THREAD_MULTIPLE);
   if (provided < MPI_THREAD_MULTIPLE)
   {
      std::cerr << "Error: the MPI library doesn't provide the required thread level" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 0);
   }
   // std::cout << "***** MPI multiple! *****" << std::endl;

#endif
   startTime = MPI_Wtime();
   char *pprocName = procName;
   MPI_CALL(
      MPI_Get_processor_name(pprocName, &namelen)
   );

   /// if you add tag, you should add tagStringTale too */
   // assert( sizeof(tagStringTable)/sizeof(char*) == N_MPI_TAGS );
   assert( tagStringTableIsSetUpCoorectly() );

   /// Data Types */
   datatypes[ParaCHAR] = MPI_CHAR;
   datatypes[ParaSHORT] = MPI_SHORT;
   datatypes[ParaINT] = MPI_INT;
   datatypes[ParaLONG] = MPI_LONG;
   datatypes[ParaUNSIGNED_CHAR] = MPI_UNSIGNED_CHAR;
   datatypes[ParaUNSIGNED_SHORT] = MPI_UNSIGNED_SHORT;
   datatypes[ParaUNSIGNED] = MPI_UNSIGNED;
   datatypes[ParaUNSIGNED_LONG] = MPI_UNSIGNED_LONG;
   datatypes[ParaFLOAT] = MPI_FLOAT;
   datatypes[ParaDOUBLE] = MPI_DOUBLE;
   datatypes[ParaLONG_DOUBLE] = MPI_LONG_DOUBLE;
   datatypes[ParaBYTE] = MPI_BYTE;

#ifdef _ALIBABA
   datatypes[ParaSIGNED_CHAR] = MPI_CHAR;
   datatypes[ParaLONG_LONG] = MPI_LONG;
   datatypes[ParaUNSIGNED_LONG_LONG] = MPI_UNSIGNED_LONG;
   datatypes[ParaBOOL] = MPI_INT;
#else
   datatypes[ParaSIGNED_CHAR] = MPI_SIGNED_CHAR;
   datatypes[ParaLONG_LONG] = MPI_LONG_LONG;
   datatypes[ParaUNSIGNED_LONG_LONG] = MPI_UNSIGNED_LONG_LONG;
   datatypes[ParaBOOL] = MPI_INT;
#endif

}

ParaCommMpi::~ParaCommMpi()
{
   MPI_Finalize();
}

bool
ParaCommMpi::tagStringTableIsSetUpCoorectly(
      )
{
   return ( sizeof(tagStringTable)/sizeof(char*) == N_MPI_TAGS );
}

const char *
ParaCommMpi::getTagString(
      int tag                 /// tag to be converted to string
      )
{
   assert( tag >= 0 && tag < N_MPI_TAGS );
   return tagStringTable[tag];
}

int
ParaCommMpi::bcast(
   void* buffer,
   int count,
   const int datatypeId,
   int root
   )
{
   MPI_CALL(
      MPI_Bcast( buffer, count, datatypes[datatypeId], root, myComm )
   );
   return 0;
}

int
ParaCommMpi::send(
   void* buffer,
   int count,
   const int datatypeId,
   int dest,
   const int tag
   )
{
   MPI_CALL(
      MPI_Send( buffer, count, datatypes[datatypeId], dest, tag, myComm )
   );
   TAG_TRACE (Send, To, dest, tag);
   return 0;
}

int
ParaCommMpi::iSend(
   void* buffer,
   int count,
   const int datatypeId,
   int dest,
   const int tag,
   MPI_Request *req
   )
{
   MPI_CALL(
      MPI_Isend( buffer, count, datatypes[datatypeId], dest, tag, myComm, req )
   );
   TAG_TRACE (iSend, To, dest, tag);
   return 0;
}

int
ParaCommMpi::receive(
   void* buffer,
   int count,
   const int datatypeId,
   int source,
   const int tag
   )
{
   MPI_Status mpiStatus;
   MPI_CALL (
      MPI_Recv( buffer, count, datatypes[datatypeId], source, tag, myComm, &mpiStatus )
   );
   TAG_TRACE (Recv, From, source, tag);
   return 0;
}

void
ParaCommMpi::waitSpecTagFromSpecSource(
      const int source,
      const int tag,
      int *receivedTag
      )
{
   MPI_Status mpiStatus;
   if( tag == TagAny )
   {
      MPI_CALL (
         MPI_Probe(source, MPI_ANY_TAG, myComm, &mpiStatus)
      );
   }
   else
   {
      MPI_CALL (
         MPI_Probe(source, tag, myComm, &mpiStatus)
      );
   }
   if( tag == TagAny )
   {
      (*receivedTag) = mpiStatus.MPI_TAG;
      TAG_TRACE (Probe, From, source, (*receivedTag));
      return;
   }
   else
   {
      assert( tag == mpiStatus.MPI_TAG );
      (*receivedTag) = mpiStatus.MPI_TAG;
      TAG_TRACE (Probe, From, source, (*receivedTag));
      return;
   }
}

bool
ParaCommMpi::probe(
   int* source,
   int* tag
   )
{
   MPI_Status mpiStatus;
   MPI_CALL (
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, myComm, &mpiStatus)
   );
   *source = mpiStatus.MPI_SOURCE;
   *tag = mpiStatus.MPI_TAG;
   TAG_TRACE (Probe, From, *source, *tag);
   return true;
}

bool
ParaCommMpi::iProbe(
   int* source,
   int* tag
   )
{
   int flag;
   MPI_Status mpiStatus;
   if( *tag == TagAny )
   {
      MPI_CALL (
         MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, myComm, &flag, &mpiStatus)
      );
   }
   else
   {
      MPI_CALL (
         MPI_Iprobe(MPI_ANY_SOURCE, *tag, myComm, &flag, &mpiStatus)
      );
   }
   if( flag )
   {
      *source = mpiStatus.MPI_SOURCE;
      *tag = mpiStatus.MPI_TAG;
      TAG_TRACE (Iprobe, From, *source, *tag);
   }
   return flag;
}

int
ParaCommMpi::ubcast(
   void* buffer,
   int count,
   MPI_Datatype datatype,
   int root
   )
{
   MPI_CALL(
      MPI_Bcast( buffer, count, datatype, root, myComm )
   );
   return 0;
}

int
ParaCommMpi::usend(
   void* buffer,
   int count,
   MPI_Datatype datatype,
   int dest,
   const int tag
   )
{
   MPI_CALL (
      MPI_Send( buffer, count, datatype, dest, tag, myComm )
      // MPI_Ssend( buffer, count, datatype, dest, tag, myComm )  // after racing, program hang
   );
   TAG_TRACE (Send, To, dest, tag);
   return 0;
}

int
ParaCommMpi::iUsend(
   void* buffer,
   int count,
   MPI_Datatype datatype,
   int dest,
   const int tag,
   MPI_Request *req
   )
{
   MPI_CALL (
      MPI_Isend( buffer, count, datatype, dest, tag, myComm, req )
      // MPI_Ssend( buffer, count, datatype, dest, tag, myComm )  // after racing, program hang
   );
   TAG_TRACE (iSend, To, dest, tag);
   return 0;
}

int
ParaCommMpi::ureceive(
   void* buffer,
   int count,
   MPI_Datatype datatype,
   int source,
   const int tag
   )
{
   MPI_Status mpiStatus;
   MPI_CALL (
      MPI_Recv( buffer, count, datatype, source, tag, myComm, &mpiStatus )
   );
   TAG_TRACE (Recv, From, source, tag);
   return 0;
}

int
ParaCommMpi::testAllIsends(
      )
{
   if( !iSendRequestDeque.empty() )
   {
      std::deque<ParaIsendRequest *>::iterator it = iSendRequestDeque.begin();
      while( it != iSendRequestDeque.end() )
      {
         ParaIsendRequest *temp = *it;
         if( temp->test() )
         {
            it = iSendRequestDeque.erase(it);
            delete temp;
         }
         else
         {
            it++;
         }
      }
   }
   return iSendRequestDeque.size();
}

void
ParaCommMpi::waitAllIsends(
         )
{
   if( !iSendRequestDeque.empty() )
   {
      std::deque<ParaIsendRequest *>::iterator it = iSendRequestDeque.begin();
      while( it != iSendRequestDeque.end() )
      {
         ParaIsendRequest *temp = *it;
         temp->wait();
         it = iSendRequestDeque.erase(it);
         delete temp;
      }
   }
}
