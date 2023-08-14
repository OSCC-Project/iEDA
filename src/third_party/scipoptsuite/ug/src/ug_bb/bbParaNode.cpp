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

/**@file    paraNode.cpp
 * @brief   Base class for BbParaNode.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "ug/paraComm.h"
#include "bbParaNode.h"

using namespace UG;

#ifdef UG_WITH_ZLIB
void
BbParaNode::write(gzstream::ogzstream &out){
   out.write((char *)&taskId.subtaskId.lcId, sizeof(int));
   out.write((char *)&taskId.subtaskId.globalSubtaskIdInLc, sizeof(int));
   out.write((char *)&taskId.subtaskId.solverId, sizeof(int));
   out.write((char *)&taskId.seqNum, sizeof(long long));
   out.write((char *)&generatorTaskId.subtaskId.lcId, sizeof(int));
   out.write((char *)&generatorTaskId.subtaskId.globalSubtaskIdInLc, sizeof(int));
   out.write((char *)&generatorTaskId.subtaskId.solverId, sizeof(int));
   out.write((char *)&generatorTaskId.seqNum, sizeof(long long));
   out.write((char *)&depth, sizeof(int));
   out.write((char *)&dualBoundValue, sizeof(double));
   out.write((char *)&initialDualBoundValue, sizeof(double));
   out.write((char *)&estimatedValue, sizeof(double));
   out.write((char *)&diffSubproblemInfo, sizeof(int));
   if( !mergeNodeInfo )
   {
      if( diffSubproblemInfo ) diffSubproblem->write(out);
   }
   else
   {
      if( mergeNodeInfo->origDiffSubproblem )
      {
         mergeNodeInfo->origDiffSubproblem->write(out);
      }
      else
      {
         if( diffSubproblemInfo ) diffSubproblem->write(out);
      }
   }
   out.write((char *)&basisInfo, sizeof(int));
   // out.write((char *)&mergingStatus, sizeof(int));
}

bool
BbParaNode::read(ParaComm *comm, gzstream::igzstream &in, bool onlyBoundChanges){ //  bool hasMergingStatus){
   in.read((char *)&taskId.subtaskId.lcId, sizeof(int));
   if( in.eof() ) return false;
   in.read((char *)&taskId.subtaskId.globalSubtaskIdInLc, sizeof(int));
   in.read((char *)&taskId.subtaskId.solverId, sizeof(int));
   in.read((char *)&taskId.seqNum, sizeof(long long));
   in.read((char *)&generatorTaskId.subtaskId.lcId, sizeof(int));
   in.read((char *)&generatorTaskId.subtaskId.globalSubtaskIdInLc, sizeof(int));
   in.read((char *)&generatorTaskId.subtaskId.solverId, sizeof(int));
   in.read((char *)&generatorTaskId.seqNum, sizeof(long long));
   in.read((char *)&depth, sizeof(int));
   in.read((char *)&dualBoundValue, sizeof(double));
   in.read((char *)&initialDualBoundValue, sizeof(double));
   in.read((char *)&estimatedValue, sizeof(double));
   in.read((char *)&diffSubproblemInfo, sizeof(int));
   if( diffSubproblemInfo ){
      diffSubproblem = comm->createParaDiffSubproblem();
      dynamic_cast<BbParaDiffSubproblem *>(diffSubproblem)->read(comm, in, onlyBoundChanges);
   }
   in.read((char *)&basisInfo, sizeof(int));
   /*
   if( hasMergingStatus )
   {
      in.read((char *)&mergingStatus, sizeof(int));
      if( mergingStatus != 1 )
      {
         mergingStatus = -1;
      }
   }
   */
   return true;
}

#endif
