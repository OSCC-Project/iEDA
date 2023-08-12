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

/**@file    scipParaDiffSubproblemMpi.cpp
 * @brief   ScipParaDiffSubproblem extension for MPI communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include <mpi.h>
#include "scipParaCommMpi.h"
#include "scipParaDiffSubproblemMpi.h"

using namespace UG;
using namespace ParaSCIP;

/** create ScipDiffSubproblemDatatype1 */
/************************************************
 * Currently, Datatype1 is not necessary.       *
 * I create this code for the future extension. *
 ************************************************/
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeCounters(
      )
{

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

#ifndef UG_DEBUG_SOLUTION
   int blockLengths[9];
   MPI_Aint displacements[9];
   MPI_Datatype types[9];
#else
   int blockLengths[10];
   MPI_Aint displacements[10];
   MPI_Datatype types[10];
#endif

   MPI_CALL(
      MPI_Get_address( &localInfoIncluded, &startAddress )
   );
   blockLengths[nBlocks] = 1;
   displacements[nBlocks] = 0;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( &nBoundChanges, &address )
   );
   blockLengths[nBlocks] = 1;
   displacements[nBlocks] = address - startAddress;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( &nBranchLinearConss, &address )
   );
   blockLengths[nBlocks] = 1;
   displacements[nBlocks] = address - startAddress;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( &nBranchSetppcConss, &address )
   );
   blockLengths[nBlocks] = 1;
   displacements[nBlocks] = address - startAddress;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( &nLinearConss, &address )
   );
   blockLengths[nBlocks] = 1;
   displacements[nBlocks] = address - startAddress;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( &nBendersLinearConss, &address )
   );
   blockLengths[nBlocks] = 1;
   displacements[nBlocks] = address - startAddress;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( &nBoundDisjunctions, &address )
   );
   blockLengths[nBlocks] = 1;
   displacements[nBlocks] = address - startAddress;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( &nVarBranchStats, &address )
   );
   blockLengths[nBlocks] = 1;
   displacements[nBlocks] = address - startAddress;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( &nVarValueVars, &address )
   );
   blockLengths[nBlocks] = 1;
   displacements[nBlocks] = address - startAddress;
   types[nBlocks] = MPI_INT;
   nBlocks++;

#ifdef UG_DEBUG_SOLUTION
   MPI_CALL(
      MPI_Get_address( &includeOptimalSol, &address )
   );
   blockLengths[nBlocks] = 1;
   displacements[nBlocks] = address - startAddress;
   types[nBlocks] = MPI_INT;
   nBlocks++;
   assert( nBlocks == 10 );
#else
   assert( nBlocks == 9 );
#endif

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
         );

   return datatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeBoundChnages(
      bool memAllocNecessary
      )
{
   assert( nBoundChanges > 0 );

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int blockLengths[3];           // reserve maximum number of elements
   MPI_Aint displacements[3];     // reserve maximum number of elements
   MPI_Datatype types[3];         // reserve maximum number of elements


   if( memAllocNecessary )
   {
      indicesAmongSolvers = new int[nBoundChanges];
      branchBounds = new SCIP_Real[nBoundChanges];
      boundTypes = new SCIP_BOUNDTYPE[nBoundChanges];
   }

   MPI_CALL(
      MPI_Get_address( indicesAmongSolvers, &startAddress )
   );
   displacements[nBlocks] = 0;
   blockLengths[nBlocks] = nBoundChanges;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( branchBounds, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = nBoundChanges;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( boundTypes, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = nBoundChanges;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
   );

   return datatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeBranchLinearConss1(
      bool memAllocNecessary
      )
{
   assert(nBranchLinearConss > 0);

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int blockLengths[4];           // reserve maximum number of elements
   MPI_Aint displacements[4];     // reserve maximum number of elements
   MPI_Datatype types[4];         // reserve maximum number of elements

   if( memAllocNecessary )
   {
      branchLinearConss = new ScipParaDiffSubproblemBranchLinearCons();
      branchLinearConss->nLinearConss = nBranchLinearConss;
      branchLinearConss->linearLhss = new SCIP_Real[nBranchLinearConss];
      branchLinearConss->linearRhss = new SCIP_Real[nBranchLinearConss];
      branchLinearConss->nLinearCoefs = new int[nBranchLinearConss];
   }
   MPI_CALL(
      MPI_Get_address( branchLinearConss->linearLhss, &startAddress )
   );
   displacements[nBlocks] = 0;
   blockLengths[nBlocks] = branchLinearConss->nLinearConss;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( branchLinearConss->linearRhss, &address )
   );
   displacements[nBlocks] = address - startAddress;;
   blockLengths[nBlocks] = branchLinearConss->nLinearConss;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( branchLinearConss->nLinearCoefs, &address )
   );
   displacements[nBlocks] = address - startAddress;;
   blockLengths[nBlocks] = branchLinearConss->nLinearConss;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( &(branchLinearConss->lConsNames), &address )
   );
   displacements[nBlocks] = address - startAddress;;
   blockLengths[nBlocks] = 1;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
   );

   return datatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeBranchLinearConss2(
      bool memAllocNecessary
      )
{
   assert(nBranchLinearConss > 0 && branchLinearConss);

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int nTotalBlocks = (nBranchLinearConss*2) + 1;
   int *blockLengths = new int[nTotalBlocks];
   MPI_Aint *displacements = new MPI_Aint[nTotalBlocks];
   MPI_Datatype *types = new MPI_Datatype[nTotalBlocks];

   if( memAllocNecessary )
   {
      branchLinearConss->linearCoefs = new SCIP_Real*[nBranchLinearConss];
      branchLinearConss->idxLinearCoefsVars = new int*[nBranchLinearConss];
      branchLinearConss->consNames = new char[branchLinearConss->lConsNames];
   }

   for(int i = 0; i < nBranchLinearConss; i++ )
   {
      if( memAllocNecessary )
      {
         branchLinearConss->linearCoefs[i] = new SCIP_Real[branchLinearConss->nLinearCoefs[i]];
         branchLinearConss->idxLinearCoefsVars[i] = new int[branchLinearConss->nLinearCoefs[i]];
      }
      if( i == 0 )
      {
         MPI_CALL(
            MPI_Get_address( branchLinearConss->linearCoefs[i], &startAddress )
         );
         displacements[nBlocks] = 0;
         blockLengths[nBlocks] = branchLinearConss->nLinearCoefs[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;
      }
      else
      {
         MPI_CALL(
            MPI_Get_address( branchLinearConss->linearCoefs[i], &address )
         );
         displacements[nBlocks] = address - startAddress;
         blockLengths[nBlocks] = branchLinearConss->nLinearCoefs[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;
      }

      MPI_CALL(
         MPI_Get_address( branchLinearConss->idxLinearCoefsVars[i], &address )
      );
      displacements[nBlocks] = address - startAddress;
      blockLengths[nBlocks] = branchLinearConss->nLinearCoefs[i];
      types[nBlocks] = MPI_INT;
      nBlocks++;
   }
   MPI_CALL(
      MPI_Get_address( branchLinearConss->consNames, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = branchLinearConss->lConsNames;
   types[nBlocks] = MPI_CHAR;
   nBlocks++;

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
   );

   delete [] blockLengths;
   delete [] displacements;
   delete [] types;

   return datatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeBranchSetppcConss1(
      bool memAllocNecessary
      )
{
   assert(nBranchSetppcConss > 0);

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int blockLengths[3];           // reserve maximum number of elements
   MPI_Aint displacements[3];     // reserve maximum number of elements
   MPI_Datatype types[3];         // reserve maximum number of elements

   if( memAllocNecessary )
   {
      branchSetppcConss = new ScipParaDiffSubproblemBranchSetppcCons();
      branchSetppcConss->nSetppcConss = nBranchSetppcConss;
      branchSetppcConss->nSetppcVars = new int[nBranchSetppcConss];
      branchSetppcConss->setppcTypes = new int[nBranchSetppcConss];
   }
   MPI_CALL(
      MPI_Get_address( branchSetppcConss->nSetppcVars, &startAddress )
   );
   displacements[nBlocks] = 0;
   blockLengths[nBlocks] = branchSetppcConss->nSetppcConss;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( branchSetppcConss->setppcTypes, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = branchSetppcConss->nSetppcConss;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( &(branchSetppcConss->lConsNames), &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = 1;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
   );

   return datatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeBranchSetppcConss2(
      bool memAllocNecessary
      )
{
   assert(nBranchSetppcConss > 0 && branchSetppcConss);

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int nTotalBlocks = nBranchSetppcConss + 1;
   int *blockLengths = new int[nTotalBlocks];
   MPI_Aint *displacements = new MPI_Aint[nTotalBlocks];
   MPI_Datatype *types = new MPI_Datatype[nTotalBlocks];

   if( memAllocNecessary )
    {
       assert(branchSetppcConss);
       branchSetppcConss->idxSetppcVars = new int*[nBranchSetppcConss];
       branchSetppcConss->consNames = new char[branchSetppcConss->lConsNames];
    }

    for(int i = 0; i < nBranchSetppcConss; i++ )
    {
       if( memAllocNecessary )
       {
          branchSetppcConss->idxSetppcVars[i] = new int[branchSetppcConss->nSetppcVars[i]];
       }
       if( i == 0 )
       {
          MPI_CALL(
             MPI_Get_address( branchSetppcConss->idxSetppcVars[i], &startAddress )
          );
          displacements[nBlocks] = 0;
          blockLengths[nBlocks] = branchSetppcConss->nSetppcVars[i];
          types[nBlocks] = MPI_INT;
          nBlocks++;
       }
       else
       {
          MPI_CALL(
             MPI_Get_address( branchSetppcConss->idxSetppcVars[i], &address )
          );
          displacements[nBlocks] = address - startAddress;
          blockLengths[nBlocks] = branchSetppcConss->nSetppcVars[i];
          types[nBlocks] = MPI_INT;
          nBlocks++;
       }
    }
    MPI_CALL(
       MPI_Get_address( branchSetppcConss->consNames, &address )
    );
    displacements[nBlocks] = address - startAddress;
    blockLengths[nBlocks] = branchSetppcConss->lConsNames;
    types[nBlocks] = MPI_CHAR;
    nBlocks++;

    MPI_CALL(
          MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
    );

    delete [] blockLengths;
    delete [] displacements;
    delete [] types;

    return datatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeLinearConss1(
      bool memAllocNecessary
      )
{
   assert(nLinearConss > 0);

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int blockLengths[3];           // reserve maximum number of elements
   MPI_Aint displacements[3];     // reserve maximum number of elements
   MPI_Datatype types[3];         // reserve maximum number of elements

   if( memAllocNecessary )
   {
      linearConss = new ScipParaDiffSubproblemLinearCons();
      linearConss->nLinearConss = nLinearConss;
      linearConss->linearLhss = new SCIP_Real[nLinearConss];
      linearConss->linearRhss = new SCIP_Real[nLinearConss];
      linearConss->nLinearCoefs = new int[nLinearConss];
   }
   MPI_CALL(
      MPI_Get_address( linearConss->linearLhss, &startAddress )
   );
   displacements[nBlocks] = 0;
   blockLengths[nBlocks] = linearConss->nLinearConss;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( linearConss->linearRhss, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = linearConss->nLinearConss;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( linearConss->nLinearCoefs, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = linearConss->nLinearConss;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
   );

   return datatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeLinearConss2(
      bool memAllocNecessary
      )
{
   assert(nLinearConss > 0 && linearConss);

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int nTotalBlocks = nLinearConss*2;
   int *blockLengths = new int[nTotalBlocks];
   MPI_Aint *displacements = new MPI_Aint[nTotalBlocks];
   MPI_Datatype *types = new MPI_Datatype[nTotalBlocks];

   if( memAllocNecessary )
   {
      linearConss->linearCoefs = new SCIP_Real*[nLinearConss];
      linearConss->idxLinearCoefsVars = new int*[nLinearConss];
   }

   for(int i = 0; i < nLinearConss; i++ )
   {
      if( memAllocNecessary )
      {
         linearConss->linearCoefs[i] = new SCIP_Real[linearConss->nLinearCoefs[i]];
         linearConss->idxLinearCoefsVars[i] = new int[linearConss->nLinearCoefs[i]];
      }
      if( i == 0 )
      {
         MPI_CALL(
            MPI_Get_address( linearConss->linearCoefs[i], &startAddress )
         );
         displacements[nBlocks] = 0;
         blockLengths[nBlocks] = linearConss->nLinearCoefs[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;
      }
      else
      {
         MPI_CALL(
            MPI_Get_address( linearConss->linearCoefs[i], &address )
         );
         displacements[nBlocks] = address - startAddress;
         blockLengths[nBlocks] = linearConss->nLinearCoefs[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;
      }


      MPI_CALL(
         MPI_Get_address( linearConss->idxLinearCoefsVars[i], &address )
      );
      displacements[nBlocks] = address - startAddress;
      blockLengths[nBlocks] = linearConss->nLinearCoefs[i];
      types[nBlocks] = MPI_INT;
      nBlocks++;
   }

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
   );

   delete [] blockLengths;
   delete [] displacements;
   delete [] types;

   return datatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeBendersLinearConss1(
      bool memAllocNecessary
      )
{
   assert(nBendersLinearConss > 0);

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int blockLengths[3];           // reserve maximum number of elements
   MPI_Aint displacements[3];     // reserve maximum number of elements
   MPI_Datatype types[3];         // reserve maximum number of elements

   if( memAllocNecessary )
   {
      bendersLinearConss = new ScipParaDiffSubproblemLinearCons();
      bendersLinearConss->nLinearConss = nBendersLinearConss;
      bendersLinearConss->linearLhss = new SCIP_Real[nBendersLinearConss];
      bendersLinearConss->linearRhss = new SCIP_Real[nBendersLinearConss];
      bendersLinearConss->nLinearCoefs = new int[nBendersLinearConss];
   }
   MPI_CALL(
      MPI_Get_address( bendersLinearConss->linearLhss, &startAddress )
   );
   displacements[nBlocks] = 0;
   blockLengths[nBlocks] = bendersLinearConss->nLinearConss;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( bendersLinearConss->linearRhss, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = bendersLinearConss->nLinearConss;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( bendersLinearConss->nLinearCoefs, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = bendersLinearConss->nLinearConss;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
   );

   return datatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeBendersLinearConss2(
      bool memAllocNecessary
      )
{
   assert(nBendersLinearConss > 0 && bendersLinearConss);

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int nTotalBlocks = nBendersLinearConss*2;
   int *blockLengths = new int[nTotalBlocks];
   MPI_Aint *displacements = new MPI_Aint[nTotalBlocks];
   MPI_Datatype *types = new MPI_Datatype[nTotalBlocks];

   if( memAllocNecessary )
   {
      bendersLinearConss->linearCoefs = new SCIP_Real*[nBendersLinearConss];
      bendersLinearConss->idxLinearCoefsVars = new int*[nBendersLinearConss];
   }

   for(int i = 0; i < nBendersLinearConss; i++ )
   {
      if( memAllocNecessary )
      {
         bendersLinearConss->linearCoefs[i] = new SCIP_Real[bendersLinearConss->nLinearCoefs[i]];
         bendersLinearConss->idxLinearCoefsVars[i] = new int[bendersLinearConss->nLinearCoefs[i]];
      }
      if( i == 0 )
      {
         MPI_CALL(
            MPI_Get_address( bendersLinearConss->linearCoefs[i], &startAddress )
         );
         displacements[nBlocks] = 0;
         blockLengths[nBlocks] = bendersLinearConss->nLinearCoefs[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;
      }
      else
      {
         MPI_CALL(
            MPI_Get_address( bendersLinearConss->linearCoefs[i], &address )
         );
         displacements[nBlocks] = address - startAddress;
         blockLengths[nBlocks] = bendersLinearConss->nLinearCoefs[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;
      }


      MPI_CALL(
         MPI_Get_address( bendersLinearConss->idxLinearCoefsVars[i], &address )
      );
      displacements[nBlocks] = address - startAddress;
      blockLengths[nBlocks] = bendersLinearConss->nLinearCoefs[i];
      types[nBlocks] = MPI_INT;
      nBlocks++;
   }

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
   );

   delete [] blockLengths;
   delete [] displacements;
   delete [] types;

   return datatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeBoundDisjunctions1(
      bool memAllocNecessary
      )
{
   assert( nBoundDisjunctions > 0 );

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int blockLengths[12];           // reserve maximum number of elements
   MPI_Aint displacements[12];     // reserve maximum number of elements
   MPI_Datatype types[12];         // reserve maximum number of elements

   if( memAllocNecessary )
   {
      boundDisjunctions = new ScipParaDiffSubproblemBoundDisjunctions();
      boundDisjunctions->nBoundDisjunctions = nBoundDisjunctions;
      boundDisjunctions->nVarsBoundDisjunction = new int[nBoundDisjunctions];
      boundDisjunctions->flagBoundDisjunctionInitial = new SCIP_Bool[nBoundDisjunctions];
      boundDisjunctions->flagBoundDisjunctionSeparate = new SCIP_Bool[nBoundDisjunctions];
      boundDisjunctions->flagBoundDisjunctionEnforce = new SCIP_Bool[nBoundDisjunctions];
      boundDisjunctions->flagBoundDisjunctionCheck = new SCIP_Bool[nBoundDisjunctions];
      boundDisjunctions->flagBoundDisjunctionPropagate = new SCIP_Bool[nBoundDisjunctions];
      boundDisjunctions->flagBoundDisjunctionLocal = new SCIP_Bool[nBoundDisjunctions];
      boundDisjunctions->flagBoundDisjunctionModifiable = new SCIP_Bool[nBoundDisjunctions];
      boundDisjunctions->flagBoundDisjunctionDynamic = new SCIP_Bool[nBoundDisjunctions];
      boundDisjunctions->flagBoundDisjunctionRemovable = new SCIP_Bool[nBoundDisjunctions];
      boundDisjunctions->flagBoundDisjunctionStickingatnode = new SCIP_Bool[nBoundDisjunctions];
   }
   MPI_CALL(
      MPI_Get_address( &(boundDisjunctions->nTotalVarsBoundDisjunctions), &startAddress )
   );
   displacements[nBlocks] = 0;
   blockLengths[nBlocks] = 1;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( boundDisjunctions->nVarsBoundDisjunction, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = boundDisjunctions->nBoundDisjunctions;
   types[nBlocks] = MPI_UNSIGNED;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( boundDisjunctions->flagBoundDisjunctionInitial, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = boundDisjunctions->nBoundDisjunctions;
   types[nBlocks] = MPI_UNSIGNED;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( boundDisjunctions->flagBoundDisjunctionSeparate, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = boundDisjunctions->nBoundDisjunctions;
   types[nBlocks] = MPI_UNSIGNED;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( boundDisjunctions->flagBoundDisjunctionEnforce, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = boundDisjunctions->nBoundDisjunctions;
   types[nBlocks] = MPI_UNSIGNED;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( boundDisjunctions->flagBoundDisjunctionCheck, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = boundDisjunctions->nBoundDisjunctions;
   types[nBlocks] = MPI_UNSIGNED;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( boundDisjunctions->flagBoundDisjunctionPropagate, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = boundDisjunctions->nBoundDisjunctions;
   types[nBlocks] = MPI_UNSIGNED;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( boundDisjunctions->flagBoundDisjunctionLocal, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = boundDisjunctions->nBoundDisjunctions;
   types[nBlocks] = MPI_UNSIGNED;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( boundDisjunctions->flagBoundDisjunctionModifiable, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = boundDisjunctions->nBoundDisjunctions;
   types[nBlocks] = MPI_UNSIGNED;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( boundDisjunctions->flagBoundDisjunctionDynamic, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = boundDisjunctions->nBoundDisjunctions;
   types[nBlocks] = MPI_UNSIGNED;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( boundDisjunctions->flagBoundDisjunctionRemovable, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = boundDisjunctions->nBoundDisjunctions;
   types[nBlocks] = MPI_UNSIGNED;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( boundDisjunctions->flagBoundDisjunctionStickingatnode, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = boundDisjunctions->nBoundDisjunctions;
   types[nBlocks] = MPI_UNSIGNED;
   nBlocks++;
 
   assert(nBlocks == 12);

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
   );

   return datatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeBoundDisjunctions2(
      bool memAllocNecessary
      )
{
   assert( nBoundDisjunctions > 0 && boundDisjunctions );

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int nTotalBlocks = nBoundDisjunctions*3;
   int *blockLengths = new int[nTotalBlocks];
   MPI_Aint *displacements = new MPI_Aint[nTotalBlocks];
   MPI_Datatype *types = new MPI_Datatype[nTotalBlocks];

   if( memAllocNecessary )
   {
      boundDisjunctions->idxBoundDisjunctionVars = new int*[nBoundDisjunctions];
      boundDisjunctions->boundTypesBoundDisjunction = new SCIP_BOUNDTYPE*[nBoundDisjunctions];
      boundDisjunctions->boundsBoundDisjunction = new SCIP_Real*[nBoundDisjunctions];
   }

   for( int i = 0; i < nBoundDisjunctions; i++ )
   {
      if( memAllocNecessary )
      {
         boundDisjunctions->idxBoundDisjunctionVars[i] = new int[boundDisjunctions->nVarsBoundDisjunction[i]];
         boundDisjunctions->boundTypesBoundDisjunction[i] = new SCIP_BOUNDTYPE[boundDisjunctions->nVarsBoundDisjunction[i]];
         boundDisjunctions->boundsBoundDisjunction[i] = new SCIP_Real[boundDisjunctions->nVarsBoundDisjunction[i]];
      }
      if( i == 0 )
      {
         MPI_CALL(
            MPI_Get_address( boundDisjunctions->idxBoundDisjunctionVars[i], &startAddress )
         );
         displacements[nBlocks] = 0;
         blockLengths[nBlocks] = boundDisjunctions->nVarsBoundDisjunction[i];
         assert( blockLengths[nBlocks] > 0 );
         types[nBlocks] = MPI_INT;
         nBlocks++;
      }
      else
      {
         MPI_CALL(
            MPI_Get_address( boundDisjunctions->idxBoundDisjunctionVars[i], &address )
         );
         displacements[nBlocks] = address - startAddress;
         blockLengths[nBlocks] = boundDisjunctions->nVarsBoundDisjunction[i];
         assert( blockLengths[nBlocks] > 0 );
         types[nBlocks] = MPI_INT;
         nBlocks++;
      }


      MPI_CALL(
         MPI_Get_address( boundDisjunctions->boundTypesBoundDisjunction[i], &address )
      );
      displacements[nBlocks] = address - startAddress;
      blockLengths[nBlocks] = boundDisjunctions->nVarsBoundDisjunction[i];
      assert( blockLengths[nBlocks] > 0 );
      assert(sizeof(SCIP_BOUNDTYPE) == sizeof(unsigned int));
      types[nBlocks] = MPI_UNSIGNED;   // actual SCIP_BoundType is enum
      nBlocks++;

      MPI_CALL(
         MPI_Get_address( boundDisjunctions->boundsBoundDisjunction[i], &address )
      );
      displacements[nBlocks] = address - startAddress;
      blockLengths[nBlocks] = boundDisjunctions->nVarsBoundDisjunction[i];
      assert( blockLengths[nBlocks] > 0 );
      types[nBlocks] = MPI_DOUBLE;
      nBlocks++;
   }

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
   );

   delete [] blockLengths;
   delete [] displacements;
   delete [] types;

   return datatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeVarBranchStats(
      bool memAllocNecessary
      )
{
   assert( nVarBranchStats > 0 );

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int blockLengths[12];           // reserve maximum number of elements
   MPI_Aint displacements[12];     // reserve maximum number of elements
   MPI_Datatype types[12];         // reserve maximum number of elements

   if( memAllocNecessary )
   {
      varBranchStats = new ScipParaDiffSubproblemVarBranchStats();
      varBranchStats->nVarBranchStats = nVarBranchStats;
      varBranchStats->idxBranchStatsVars = new int[nVarBranchStats];
      varBranchStats->downpscost = new SCIP_Real[nVarBranchStats];
      varBranchStats->uppscost = new SCIP_Real[nVarBranchStats];
      varBranchStats->downvsids = new SCIP_Real[nVarBranchStats];
      varBranchStats->upvsids = new SCIP_Real[nVarBranchStats];
      varBranchStats->downconflen = new SCIP_Real[nVarBranchStats];
      varBranchStats->upconflen = new SCIP_Real[nVarBranchStats];
      varBranchStats->downinfer = new SCIP_Real[nVarBranchStats];
      varBranchStats->upinfer = new SCIP_Real[nVarBranchStats];
      varBranchStats->downcutoff = new SCIP_Real[nVarBranchStats];
      varBranchStats->upcutoff = new SCIP_Real[nVarBranchStats];
   }

   MPI_CALL(
      MPI_Get_address( &(varBranchStats->offset), &startAddress )
   );
   displacements[nBlocks] = 0;
   blockLengths[nBlocks] = 1;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( varBranchStats->idxBranchStatsVars, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = varBranchStats->nVarBranchStats;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( varBranchStats->downpscost, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = varBranchStats->nVarBranchStats;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( varBranchStats->uppscost, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = varBranchStats->nVarBranchStats;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( varBranchStats->downvsids, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = varBranchStats->nVarBranchStats;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( varBranchStats->upvsids, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = varBranchStats->nVarBranchStats;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( varBranchStats->downconflen, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = varBranchStats->nVarBranchStats;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( varBranchStats->upconflen, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = varBranchStats->nVarBranchStats;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( varBranchStats->downinfer, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = varBranchStats->nVarBranchStats;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( varBranchStats->upinfer, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = varBranchStats->nVarBranchStats;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( varBranchStats->downcutoff, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = varBranchStats->nVarBranchStats;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( varBranchStats->upcutoff, &address )
   );
   displacements[nBlocks] = address - startAddress;
   blockLengths[nBlocks] = varBranchStats->nVarBranchStats;
   types[nBlocks] = MPI_DOUBLE;
   nBlocks++;

   assert( nBlocks == 12 );

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
   );

   return datatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeVarValueVars1(
      bool memAllocNecessary
      )
{
   assert( nVarValueVars > 0 );

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int blockLengths[3];           // reserve maximum number of elements
   MPI_Aint displacements[3];     // reserve maximum number of elements
   MPI_Datatype types[3];         // reserve maximum number of elements

   if( memAllocNecessary )
   {
      varValues = new ScipParaDiffSubproblemVarValues();
      varValues->nVarValueVars = nVarValueVars;
      varValues->idxVarValueVars = new int[nVarValueVars];
      varValues->nVarValueValues = new int[nVarValueVars];
   }

   MPI_CALL(
      MPI_Get_address( &(varValues->nVarValues), &startAddress )
   );
   displacements[nBlocks] = 0;
   blockLengths[nBlocks] = 1;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( varValues->idxVarValueVars, &address )
   );
   displacements[nBlocks] = address - startAddress;;
   blockLengths[nBlocks] = varValues->nVarValueVars;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   MPI_CALL(
      MPI_Get_address( varValues->nVarValueValues, &address )
   );
   displacements[nBlocks] = address - startAddress;;
   blockLengths[nBlocks] = varValues->nVarValueVars;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   assert( nBlocks == 3);

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
   );

   return datatype;
}

/** create ScipDiffSubproblemDatatype */
MPI_Datatype
ScipParaDiffSubproblemMpi::createDatatypeVarValueVars2(
      bool memAllocNecessary
      )
{
   assert( nVarValueVars > 0 && varValues);

   int nBlocks = 0;

   MPI_Datatype datatype;

   MPI_Aint startAddress = 0;
   MPI_Aint address = 0;

   int nVarValueBlock = 0;
   if( nVarValueVars > 0 )
   {
      for(int i = 0; i <  varValues->nVarValueVars; i++ )
      {
         if(  varValues->nVarValueValues[i] > 0 )
         {
            nVarValueBlock += 9;
         }
      }
   }

   int *blockLengths = new int[nVarValueBlock + 1];
   MPI_Aint *displacements = new MPI_Aint[nVarValueBlock + 1];
   MPI_Datatype *types = new MPI_Datatype[nVarValueBlock + 1];

   /* this duplicate send of &nVarValueVars is a dummy to get startAddress */
   MPI_CALL(
      MPI_Get_address( &nVarValueVars, &startAddress )
   );
   blockLengths[nBlocks] = 1;
   displacements[nBlocks] = 0;
   types[nBlocks] = MPI_INT;
   nBlocks++;

   if( memAllocNecessary )
   {
      assert(varValues);
      varValues->varValue            = new SCIP_Real*[nVarValueVars];
      varValues->varValueDownvsids   = new SCIP_Real*[nVarValueVars];
      varValues->varVlaueUpvsids     = new SCIP_Real*[nVarValueVars];
      varValues->varValueDownconflen = new SCIP_Real*[nVarValueVars];
      varValues->varValueUpconflen   = new SCIP_Real*[nVarValueVars];
      varValues->varValueDowninfer   = new SCIP_Real*[nVarValueVars];
      varValues->varValueUpinfer     = new SCIP_Real*[nVarValueVars];
      varValues->varValueDowncutoff  = new SCIP_Real*[nVarValueVars];
      varValues->varValueUpcutoff    = new SCIP_Real*[nVarValueVars];
   }

   for(int i = 0; i < nVarValueVars; i++ )
   {
      if(varValues-> nVarValueValues[i] > 0 )
      {
         if( memAllocNecessary )
         {
            varValues->varValue[i]            = new SCIP_Real[varValues->nVarValueValues[i]];
            varValues->varValueDownvsids[i]   = new SCIP_Real[varValues->nVarValueValues[i]];
            varValues->varVlaueUpvsids[i]     = new SCIP_Real[varValues->nVarValueValues[i]];
            varValues->varValueDownconflen[i] = new SCIP_Real[varValues->nVarValueValues[i]];
            varValues->varValueUpconflen[i]   = new SCIP_Real[varValues->nVarValueValues[i]];
            varValues->varValueDowninfer[i]   = new SCIP_Real[varValues->nVarValueValues[i]];
            varValues->varValueUpinfer[i]     = new SCIP_Real[varValues->nVarValueValues[i]];
            varValues->varValueDowncutoff[i]  = new SCIP_Real[varValues->nVarValueValues[i]];
            varValues->varValueUpcutoff[i]    = new SCIP_Real[varValues->nVarValueValues[i]];
         }

         MPI_CALL(
            MPI_Get_address( varValues->varValue[i], &address )
         );
         displacements[nBlocks] = address - startAddress;
         blockLengths[nBlocks] = varValues->nVarValueValues[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;

         MPI_CALL(
            MPI_Get_address( varValues->varValueDownvsids[i], &address )
         );
         displacements[nBlocks] = address - startAddress;
         blockLengths[nBlocks] = varValues->nVarValueValues[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;

         MPI_CALL(
            MPI_Get_address( varValues->varVlaueUpvsids[i], &address )
         );
         displacements[nBlocks] = address - startAddress;
         blockLengths[nBlocks] = varValues->nVarValueValues[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;

         MPI_CALL(
            MPI_Get_address( varValues->varValueDownconflen[i], &address )
         );
         displacements[nBlocks] = address - startAddress;
         blockLengths[nBlocks] = varValues->nVarValueValues[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;

         MPI_CALL(
            MPI_Get_address( varValues->varValueUpconflen[i], &address )
         );
         displacements[nBlocks] = address - startAddress;
         blockLengths[nBlocks] = varValues->nVarValueValues[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;

         MPI_CALL(
            MPI_Get_address( varValues->varValueDowninfer[i], &address )
         );
         displacements[nBlocks] = address - startAddress;
         blockLengths[nBlocks] = varValues->nVarValueValues[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;

         MPI_CALL(
            MPI_Get_address( varValues->varValueUpinfer[i], &address )
         );
         displacements[nBlocks] = address - startAddress;
         blockLengths[nBlocks] = varValues->nVarValueValues[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;

         MPI_CALL(
            MPI_Get_address( varValues->varValueDowncutoff[i], &address )
         );
         displacements[nBlocks] = address - startAddress;
         blockLengths[nBlocks] = varValues->nVarValueValues[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;

         MPI_CALL(
            MPI_Get_address( varValues->varValueUpcutoff[i], &address )
         );
         displacements[nBlocks] = address - startAddress;
         blockLengths[nBlocks] = varValues->nVarValueValues[i];
         types[nBlocks] = MPI_DOUBLE;
         nBlocks++;

      }
   }

   MPI_CALL(
         MPI_Type_create_struct(nBlocks, blockLengths, displacements, types, &datatype)
   );

   delete [] blockLengths;
   delete [] displacements;
   delete [] types;

   return datatype;
}

int
ScipParaDiffSubproblemMpi::bcast(ParaComm *comm, int root)
{
   DEF_PARA_COMM( commMpi, comm);

   if( branchLinearConss )
   {
      nBranchLinearConss = branchLinearConss->nLinearConss;
   }
   if( branchSetppcConss )
   {
      nBranchSetppcConss = branchSetppcConss->nSetppcConss;
   }
   if( linearConss )
   {
      nLinearConss = linearConss->nLinearConss;
   }
   if( bendersLinearConss )
   {
      nBendersLinearConss = bendersLinearConss->nLinearConss;
   }
   if( boundDisjunctions )
   {
      nBoundDisjunctions = boundDisjunctions->nBoundDisjunctions;
   }
   if( varBranchStats )
   {
      nVarBranchStats = varBranchStats->nVarBranchStats;
   }
   if( varValues )
   {
      nVarValueVars = varValues->nVarValueVars;
   }

   MPI_Datatype datatypeCounters;
   datatypeCounters = createDatatypeCounters();
   MPI_CALL(
      MPI_Type_commit( &datatypeCounters )
   );
   PARA_COMM_CALL(
      commMpi->ubcast(&localInfoIncluded, 1, datatypeCounters, root)
   );
   MPI_CALL(
      MPI_Type_free( &datatypeCounters )
   );

   if( nBoundChanges > 0 )
   {
      MPI_Datatype datatypeBoundChanges;
      if( comm->getRank() == root )
      {
         datatypeBoundChanges = createDatatypeBoundChnages(false);
      }
      else
      {
         datatypeBoundChanges = createDatatypeBoundChnages(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatypeBoundChanges )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(indicesAmongSolvers, 1, datatypeBoundChanges, root)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBoundChanges )
      );
   }

   if( nBranchLinearConss > 0  )
   {
      MPI_Datatype datatypeBranchLinearConss1;
      if( comm->getRank() == root )
      {
         datatypeBranchLinearConss1 = createDatatypeBranchLinearConss1(false);
      }
      else
      {
         datatypeBranchLinearConss1 = createDatatypeBranchLinearConss1(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatypeBranchLinearConss1 )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(branchLinearConss->linearLhss, 1, datatypeBranchLinearConss1, root)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBranchLinearConss1 )
      );

      MPI_Datatype datatypeBranchLinearConss2;
      if( comm->getRank() == root )
      {
         datatypeBranchLinearConss2 = createDatatypeBranchLinearConss2(false);
      }
      else
      {
         datatypeBranchLinearConss2 = createDatatypeBranchLinearConss2(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatypeBranchLinearConss2 )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(branchLinearConss->linearCoefs[0], 1, datatypeBranchLinearConss2, root)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBranchLinearConss2 )
      );
   }

   if(  nBranchSetppcConss > 0 )
   {
      MPI_Datatype datatypeBranchSetppcConss1;
      if( comm->getRank() == root )
      {
         datatypeBranchSetppcConss1 = createDatatypeBranchSetppcConss1(false);
      }
      else
      {
         datatypeBranchSetppcConss1 = createDatatypeBranchSetppcConss1(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatypeBranchSetppcConss1 )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(branchSetppcConss->nSetppcVars, 1, datatypeBranchSetppcConss1, root)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBranchSetppcConss1 )
      );

      MPI_Datatype datatypeBranchSetppcConss2;
      if( comm->getRank() == root )
      {
         datatypeBranchSetppcConss2 = createDatatypeBranchSetppcConss2(false);
      }
      else
      {
         datatypeBranchSetppcConss2 = createDatatypeBranchSetppcConss2(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatypeBranchSetppcConss2 )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(branchSetppcConss->idxSetppcVars[0], 1, datatypeBranchSetppcConss2, root)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBranchSetppcConss2 )
      );
   }

   if( nLinearConss > 0 )
   {
      MPI_Datatype datatypeLinearConss1;
      if( comm->getRank() == root )
      {
         datatypeLinearConss1 = createDatatypeLinearConss1(false);
      }
      else
      {
         datatypeLinearConss1 = createDatatypeLinearConss1(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatypeLinearConss1 )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(linearConss->linearLhss, 1, datatypeLinearConss1, root)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeLinearConss1 )
      );

      MPI_Datatype datatypeLinearConss2;
      if( comm->getRank() == root )
      {
         datatypeLinearConss2 = createDatatypeLinearConss2(false);
      }
      else
      {
         datatypeLinearConss2 = createDatatypeLinearConss2(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatypeLinearConss2 )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(linearConss->linearCoefs[0], 1, datatypeLinearConss2, root)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeLinearConss2 )
      );
   }

   if( nBendersLinearConss > 0 )
   {
      MPI_Datatype datatypeBendersLinearConss1;
      if( comm->getRank() == root )
      {
         datatypeBendersLinearConss1 = createDatatypeBendersLinearConss1(false);
      }
      else
      {
         datatypeBendersLinearConss1 = createDatatypeBendersLinearConss1(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatypeBendersLinearConss1 )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(bendersLinearConss->linearLhss, 1, datatypeBendersLinearConss1, root)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBendersLinearConss1 )
      );

      MPI_Datatype datatypeBendersLinearConss2;
      if( comm->getRank() == root )
      {
         datatypeBendersLinearConss2 = createDatatypeBendersLinearConss2(false);
      }
      else
      {
         datatypeBendersLinearConss2 = createDatatypeBendersLinearConss2(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatypeBendersLinearConss2 )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(bendersLinearConss->linearCoefs[0], 1, datatypeBendersLinearConss2, root)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBendersLinearConss2 )
      );
   }

   if( nBoundDisjunctions > 0 )
   {
      MPI_Datatype datatypeBoundDisjunctions1;
      if( comm->getRank() == root )
      {
         datatypeBoundDisjunctions1 = createDatatypeBoundDisjunctions1(false);
      }
      else
      {
         datatypeBoundDisjunctions1 = createDatatypeBoundDisjunctions1(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatypeBoundDisjunctions1 )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(&(boundDisjunctions->nTotalVarsBoundDisjunctions), 1, datatypeBoundDisjunctions1, root)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBoundDisjunctions1 )
      );

      MPI_Datatype datatypeBoundDisjunctions2;
      if( comm->getRank() == root )
      {
         datatypeBoundDisjunctions2 = createDatatypeBoundDisjunctions2(false);
      }
      else
      {
         datatypeBoundDisjunctions2 = createDatatypeBoundDisjunctions2(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatypeBoundDisjunctions2 )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(boundDisjunctions->idxBoundDisjunctionVars[0], 1, datatypeBoundDisjunctions2, root)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBoundDisjunctions2 )
      );
   }

   if( nVarBranchStats > 0 )
   {
      MPI_Datatype datatypeVarBranchStats;
      if( comm->getRank() == root )
      {
         datatypeVarBranchStats = createDatatypeVarBranchStats(false);
      }
      else
      {
         datatypeVarBranchStats = createDatatypeVarBranchStats(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatypeVarBranchStats )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(&(varBranchStats->offset), 1, datatypeVarBranchStats, root)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeVarBranchStats )
      );
   }

   if( nVarValueVars > 0 )
   {
      MPI_Datatype datatypeVarValueVars1;
      if( comm->getRank() == root )
      {
         datatypeVarValueVars1 = createDatatypeVarValueVars1(false);
      }
      else
      {
         datatypeVarValueVars1 = createDatatypeVarValueVars1(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatypeVarValueVars1 )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(&(varValues->nVarValues), 1, datatypeVarValueVars1, root)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeVarValueVars1 )
      );

      MPI_Datatype datatypeBoundDisjunctions2;
      if( comm->getRank() == root )
      {
         datatypeBoundDisjunctions2 = createDatatypeBoundDisjunctions2(false);
      }
      else
      {
         datatypeBoundDisjunctions2 = createDatatypeBoundDisjunctions2(true);
      }
      MPI_CALL(
         MPI_Type_commit( &datatypeBoundDisjunctions2 )
      );
      PARA_COMM_CALL(
            commMpi->ubcast(&nVarValueVars, 1, datatypeBoundDisjunctions2, root)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBoundDisjunctions2 )
      );
   }

   return 0;
}

int
ScipParaDiffSubproblemMpi::send(ParaComm *comm, int dest)
{
   DEF_PARA_COMM( commMpi, comm);

   if( branchLinearConss )
   {
      nBranchLinearConss = branchLinearConss->nLinearConss;
   }
   if( branchSetppcConss )
   {
      nBranchSetppcConss = branchSetppcConss->nSetppcConss;
   }
   if( linearConss )
   {
      nLinearConss = linearConss->nLinearConss;
   }
   if( bendersLinearConss )
   {
      nBendersLinearConss = bendersLinearConss->nLinearConss;
   }
   if( boundDisjunctions )
   {
      nBoundDisjunctions = boundDisjunctions->nBoundDisjunctions;
   }
   if( varBranchStats )
   {
      nVarBranchStats = varBranchStats->nVarBranchStats;
   }
   if( varValues )
   {
      nVarValueVars = varValues->nVarValueVars;
   }


   MPI_Datatype datatypeCounters;
   datatypeCounters = createDatatypeCounters();
   MPI_CALL(
      MPI_Type_commit( &datatypeCounters )
   );
   PARA_COMM_CALL(
      commMpi->usend(&localInfoIncluded, 1, datatypeCounters, dest, TagDiffSubproblem)
   );
   MPI_CALL(
      MPI_Type_free( &datatypeCounters )
   );

   if( nBoundChanges > 0 )
   {
      MPI_Datatype datatypeBoundChanges;
      datatypeBoundChanges = createDatatypeBoundChnages(false);
      MPI_CALL(
         MPI_Type_commit( &datatypeBoundChanges )
      );
      PARA_COMM_CALL(
            commMpi->usend(indicesAmongSolvers, 1, datatypeBoundChanges, dest, TagDiffSubproblem1)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBoundChanges )
      );
   }

   if( nBranchLinearConss > 0  )
   {
      MPI_Datatype datatypeBranchLinearConss1;
      datatypeBranchLinearConss1 = createDatatypeBranchLinearConss1(false);
      MPI_CALL(
         MPI_Type_commit( &datatypeBranchLinearConss1 )
      );
      PARA_COMM_CALL(
            commMpi->usend(branchLinearConss->linearLhss, 1, datatypeBranchLinearConss1, dest, TagDiffSubproblem2)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBranchLinearConss1 )
      );

      MPI_Datatype datatypeBranchLinearConss2;
      datatypeBranchLinearConss2 = createDatatypeBranchLinearConss2(false);
      MPI_CALL(
         MPI_Type_commit( &datatypeBranchLinearConss2 )
      );
      PARA_COMM_CALL(
            commMpi->usend(branchLinearConss->linearCoefs[0], 1, datatypeBranchLinearConss2, dest, TagDiffSubproblem3)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBranchLinearConss2 )
      );
   }

   if(  nBranchSetppcConss > 0 )
   {
      MPI_Datatype datatypeBranchSetppcConss1;
      datatypeBranchSetppcConss1 = createDatatypeBranchSetppcConss1(false);
      MPI_CALL(
         MPI_Type_commit( &datatypeBranchSetppcConss1 )
      );
      PARA_COMM_CALL(
            commMpi->usend(branchSetppcConss->nSetppcVars, 1, datatypeBranchSetppcConss1, dest, TagDiffSubproblem4)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBranchSetppcConss1 )
      );

      MPI_Datatype datatypeBranchSetppcConss2;
      datatypeBranchSetppcConss2 = createDatatypeBranchSetppcConss2(false);
      MPI_CALL(
         MPI_Type_commit( &datatypeBranchSetppcConss2 )
      );
      PARA_COMM_CALL(
            commMpi->usend(branchSetppcConss->idxSetppcVars[0], 1, datatypeBranchSetppcConss2, dest, TagDiffSubproblem5)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBranchSetppcConss2 )
      );
   }

   if( nLinearConss > 0 )
   {
      MPI_Datatype datatypeLinearConss1;
      datatypeLinearConss1 = createDatatypeLinearConss1(false);
      MPI_CALL(
         MPI_Type_commit( &datatypeLinearConss1 )
      );
      PARA_COMM_CALL(
            commMpi->usend(linearConss->linearLhss, 1, datatypeLinearConss1, dest, TagDiffSubproblem6)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeLinearConss1 )
      );

      MPI_Datatype datatypeLinearConss2;
      datatypeLinearConss2 = createDatatypeLinearConss2(false);
      MPI_CALL(
         MPI_Type_commit( &datatypeLinearConss2 )
      );
      PARA_COMM_CALL(
            commMpi->usend(linearConss->linearCoefs[0], 1, datatypeLinearConss2, dest, TagDiffSubproblem7)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeLinearConss2 )
      );
   }

   if( nBendersLinearConss > 0 )
   {
      MPI_Datatype datatypeBendersLinearConss1;
      datatypeBendersLinearConss1 = createDatatypeBendersLinearConss1(false);
      MPI_CALL(
         MPI_Type_commit( &datatypeBendersLinearConss1 )
      );
      PARA_COMM_CALL(
            commMpi->usend(bendersLinearConss->linearLhss, 1, datatypeBendersLinearConss1, dest, TagDiffSubproblem8)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBendersLinearConss1 )
      );

      MPI_Datatype datatypeBendersLinearConss2;
      datatypeBendersLinearConss2 = createDatatypeBendersLinearConss2(false);
      MPI_CALL(
         MPI_Type_commit( &datatypeBendersLinearConss2 )
      );
      PARA_COMM_CALL(
            commMpi->usend(bendersLinearConss->linearCoefs[0], 1, datatypeBendersLinearConss2, dest, TagDiffSubproblem9)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBendersLinearConss2 )
      );
   }

   if( nBoundDisjunctions > 0 )
   {
      MPI_Datatype datatypeBoundDisjunctions1;
      datatypeBoundDisjunctions1 = createDatatypeBoundDisjunctions1(false);
      MPI_CALL(
         MPI_Type_commit( &datatypeBoundDisjunctions1 )
      );
      PARA_COMM_CALL(
            commMpi->usend(&(boundDisjunctions->nTotalVarsBoundDisjunctions), 1, datatypeBoundDisjunctions1, dest, TagDiffSubproblem10)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBoundDisjunctions1 )
      );

      MPI_Datatype datatypeBoundDisjunctions2;
      datatypeBoundDisjunctions2 = createDatatypeBoundDisjunctions2(false);
      MPI_CALL(
         MPI_Type_commit( &datatypeBoundDisjunctions2 )
      );
      PARA_COMM_CALL(
            commMpi->usend(boundDisjunctions->idxBoundDisjunctionVars[0], 1, datatypeBoundDisjunctions2, dest, TagDiffSubproblem11)
            // commMpi->usend(&(boundDisjunctions->idxBoundDisjunctionVars[0]), 1, datatypeBoundDisjunctions2, dest, TagDiffSubproblem11)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBoundDisjunctions2 )
      );
   }

   if( nVarBranchStats > 0 )
   {
      MPI_Datatype datatypeVarBranchStats;
      datatypeVarBranchStats = createDatatypeVarBranchStats(false);
      MPI_CALL(
         MPI_Type_commit( &datatypeVarBranchStats )
      );
      PARA_COMM_CALL(
            commMpi->usend(&(varBranchStats->offset), 1, datatypeVarBranchStats, dest, TagDiffSubproblem12)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeVarBranchStats )
      );
   }

   if( nVarValueVars > 0 )
   {
      MPI_Datatype datatypeVarValueVars1;
      datatypeVarValueVars1 = createDatatypeVarValueVars1(false);
      MPI_CALL(
         MPI_Type_commit( &datatypeVarValueVars1 )
      );
      PARA_COMM_CALL(
            commMpi->usend(&(varValues->nVarValues), 1, datatypeVarValueVars1, dest, TagDiffSubproblem13)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeVarValueVars1 )
      );

      MPI_Datatype datatypeBoundDisjunctions2;
      datatypeBoundDisjunctions2 = createDatatypeBoundDisjunctions2(false);
      MPI_CALL(
         MPI_Type_commit( &datatypeBoundDisjunctions2 )
      );
      PARA_COMM_CALL(
            commMpi->usend(&nVarValueVars, 1, datatypeBoundDisjunctions2, dest, TagDiffSubproblem14)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBoundDisjunctions2 )
      );
   }
   return 0;
}

int
ScipParaDiffSubproblemMpi::receive(ParaComm *comm, int source)
{
   DEF_PARA_COMM( commMpi, comm);

   MPI_Datatype datatypeCounters;
   datatypeCounters = createDatatypeCounters();
   MPI_CALL(
      MPI_Type_commit( &datatypeCounters )
   );
   PARA_COMM_CALL(
      commMpi->ureceive(&localInfoIncluded, 1, datatypeCounters, source, TagDiffSubproblem)
   );
   MPI_CALL(
      MPI_Type_free( &datatypeCounters )
   );

   if( nBoundChanges > 0 )
   {
      MPI_Datatype datatypeBoundChanges;
      datatypeBoundChanges = createDatatypeBoundChnages(true);
      MPI_CALL(
         MPI_Type_commit( &datatypeBoundChanges )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(indicesAmongSolvers, 1, datatypeBoundChanges, source, TagDiffSubproblem1)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBoundChanges )
      );
   }

   if( nBranchLinearConss > 0  )
   {
      MPI_Datatype datatypeBranchLinearConss1;
      datatypeBranchLinearConss1 = createDatatypeBranchLinearConss1(true);
      MPI_CALL(
         MPI_Type_commit( &datatypeBranchLinearConss1 )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(branchLinearConss->linearLhss, 1, datatypeBranchLinearConss1, source, TagDiffSubproblem2)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBranchLinearConss1 )
      );

      MPI_Datatype datatypeBranchLinearConss2;
      datatypeBranchLinearConss2 = createDatatypeBranchLinearConss2(true);
      MPI_CALL(
         MPI_Type_commit( &datatypeBranchLinearConss2 )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(branchLinearConss->linearCoefs[0], 1, datatypeBranchLinearConss2, source, TagDiffSubproblem3)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBranchLinearConss2 )
      );
   }

   if(  nBranchSetppcConss > 0 )
   {
      MPI_Datatype datatypeBranchSetppcConss1;
      datatypeBranchSetppcConss1 = createDatatypeBranchSetppcConss1(true);
      MPI_CALL(
         MPI_Type_commit( &datatypeBranchSetppcConss1 )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(branchSetppcConss->nSetppcVars, 1, datatypeBranchSetppcConss1, source, TagDiffSubproblem4)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBranchSetppcConss1 )
      );

      MPI_Datatype datatypeBranchSetppcConss2;
      datatypeBranchSetppcConss2 = createDatatypeBranchSetppcConss2(true);
      MPI_CALL(
         MPI_Type_commit( &datatypeBranchSetppcConss2 )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(branchSetppcConss->idxSetppcVars[0], 1, datatypeBranchSetppcConss2, source, TagDiffSubproblem5)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBranchSetppcConss2 )
      );
   }

   if( nLinearConss > 0 )
   {
      MPI_Datatype datatypeLinearConss1;
      datatypeLinearConss1 = createDatatypeLinearConss1(true);
      MPI_CALL(
         MPI_Type_commit( &datatypeLinearConss1 )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(linearConss->linearLhss, 1, datatypeLinearConss1, source, TagDiffSubproblem6)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeLinearConss1 )
      );

      MPI_Datatype datatypeLinearConss2;
      datatypeLinearConss2 = createDatatypeLinearConss2(true);
      MPI_CALL(
         MPI_Type_commit( &datatypeLinearConss2 )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(linearConss->linearCoefs[0], 1, datatypeLinearConss2, source, TagDiffSubproblem7)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeLinearConss2 )
      );
   }

   if( nBendersLinearConss > 0 )
   {
      MPI_Datatype datatypeBendersLinearConss1;
      datatypeBendersLinearConss1 = createDatatypeBendersLinearConss1(true);
      MPI_CALL(
         MPI_Type_commit( &datatypeBendersLinearConss1 )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(bendersLinearConss->linearLhss, 1, datatypeBendersLinearConss1, source, TagDiffSubproblem8)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBendersLinearConss1 )
      );

      MPI_Datatype datatypeBendersLinearConss2;
      datatypeBendersLinearConss2 = createDatatypeBendersLinearConss2(true);
      MPI_CALL(
         MPI_Type_commit( &datatypeBendersLinearConss2 )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(bendersLinearConss->linearCoefs[0], 1, datatypeBendersLinearConss2, source, TagDiffSubproblem9)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBendersLinearConss2 )
      );
   }

   if( nBoundDisjunctions > 0 )
   {
      MPI_Datatype datatypeBoundDisjunctions1;
      datatypeBoundDisjunctions1 = createDatatypeBoundDisjunctions1(true);
      MPI_CALL(
         MPI_Type_commit( &datatypeBoundDisjunctions1 )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(&(boundDisjunctions->nTotalVarsBoundDisjunctions), 1, datatypeBoundDisjunctions1, source, TagDiffSubproblem10)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBoundDisjunctions1 )
      );

      MPI_Datatype datatypeBoundDisjunctions2;
      datatypeBoundDisjunctions2 = createDatatypeBoundDisjunctions2(true);
      MPI_CALL(
         MPI_Type_commit( &datatypeBoundDisjunctions2 )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(boundDisjunctions->idxBoundDisjunctionVars[0], 1, datatypeBoundDisjunctions2, source, TagDiffSubproblem11)
            // commMpi->ureceive(&(boundDisjunctions->idxBoundDisjunctionVars[0]), 1, datatypeBoundDisjunctions2, source, TagDiffSubproblem11)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBoundDisjunctions2 )
      );
   }

   if( nVarBranchStats > 0 )
   {
      MPI_Datatype datatypeVarBranchStats;
      datatypeVarBranchStats = createDatatypeVarBranchStats(true);
      MPI_CALL(
         MPI_Type_commit( &datatypeVarBranchStats )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(&(varBranchStats->offset), 1, datatypeVarBranchStats, source, TagDiffSubproblem12)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeVarBranchStats )
      );
   }

   if( nVarValueVars > 0 )
   {
      MPI_Datatype datatypeVarValueVars1;
      datatypeVarValueVars1 = createDatatypeVarValueVars1(true);
      MPI_CALL(
         MPI_Type_commit( &datatypeVarValueVars1 )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(&(varValues->nVarValues), 1, datatypeVarValueVars1, source, TagDiffSubproblem13)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeVarValueVars1 )
      );

      MPI_Datatype datatypeBoundDisjunctions2;
      datatypeBoundDisjunctions2 = createDatatypeBoundDisjunctions2(true);
      MPI_CALL(
         MPI_Type_commit( &datatypeBoundDisjunctions2 )
      );
      PARA_COMM_CALL(
            commMpi->ureceive(&nVarValueVars, 1, datatypeBoundDisjunctions2, source, TagDiffSubproblem14)
      );

      MPI_CALL(
         MPI_Type_free( &datatypeBoundDisjunctions2 )
      );
   }

   return 0;
}

/** create clone of this object */
ScipParaDiffSubproblemMpi *
ScipParaDiffSubproblemMpi::clone(ParaComm *comm)
{
   return( new ScipParaDiffSubproblemMpi(this) );

}
