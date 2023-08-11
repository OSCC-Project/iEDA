/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program                         */
/*          GCG --- Generic Column Generation                                */
/*                  a Dantzig-Wolfe decomposition based extension            */
/*                  of the branch-cut-and-price framework                    */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/* Copyright (C) 2010-2022 Operations Research, RWTH Aachen University       */
/*                         Zuse Institute Berlin (ZIB)                       */
/*                                                                           */
/* This program is free software; you can redistribute it and/or             */
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
/* along with this program; if not, write to the Free Software               */
/* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.*/
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   class_detprobdata.cpp
 * @brief  class storing partialdecomps and the problem matrix
 * @author Michael Bastubbe
 * @author Julius Hense
 * @author Hanna Franzen
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

/*@todo don't disable lint */
/*lint -e64 disable useless and wrong lint warning */

/*@todo this looks like a workaround, is disabling warnings a good idea? */
#ifdef __INTEL_COMPILER
#ifndef _OPENMP
#pragma warning disable 3180  /* disable wrong and useless omp warnings */
#endif
#endif

//#define WRITE_ORIG_CONSTYPES
// #define SCIP_DEBUG

#include "scip/scipdefplugins.h"
#include "gcg.h"
#include "objscip/objscip.h"
#include "scip/scip.h"
#include "class_detprobdata.h"
#include "struct_detector.h"
#include "pub_decomp.h"
#include "struct_decomp.h"
#include "cons_decomp.h"
#include "decomp.h"
#include "miscvisualization.h"
#include "scip_misc.h"
#include "scip/clock.h"
#include "scip/cons.h"
#include "scip/scip.h"
#include "scip/var.h"
#include <algorithm>
#include <list>
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <exception>
#include <random> /* needed for exponential distributed random dual variables */
#include <set>

#include "reader_gp.h"


#ifdef _OPENMP
#include <omp.h>
#endif

#define SCIP_CALL_EXC( x ) do                                                                                 \
                       {                                                                                      \
                          SCIP_RETCODE _restat_;                                                              \
                          if( ( _restat_ = ( x ) ) != SCIP_OKAY )                                             \
                          {                                                                                   \
                             SCIPerrorMessage( "Error <%d> in function call\n", _restat_);                    \
                             throw std::exception();                                                          \
                          }                                                                                   \
                       }                                                                                      \
                       while( false )

#define ENUM_TO_STRING( x ) # x
#define DEFAULT_THREADS    0     /**< number of threads (0 is OpenMP default) */


namespace gcg{

/* local methods */

struct sort_decr
{
   bool operator()(
      const std::pair<int, int> &left,
      const std::pair<int, int> &right)
   {
      if( left.second != right.second )
         return left.second > right.second;
      else return left.first < right.first;
   }
};


/** returns the relevant representative of a var */
SCIP_Bool varIsFixedToZero(
   SCIP* scip,
   SCIP_VAR* var
   )
{
   return ( SCIPisEQ(scip, SCIPvarGetLbGlobal(var), SCIPvarGetUbGlobal(var) ) && SCIPisEQ(scip, SCIPvarGetUbGlobal(var), 0. ) );
}


void DETPROBDATA::calcTranslationMapping(
   DETPROBDATA* origdata,
   std::vector<int>& rowothertothis,
   std::vector<int>& rowthistoother,
   std::vector<int>& colothertothis,
   std::vector<int>& colthistoother,
   std::vector<int>& missingrowinthis
   )
{
   SCIP_CONS* transcons;
   int nrowsother = origdata->nconss;
   int nrowsthis = nconss;
   int ncolsother = origdata->nvars;
   int ncolsthis = nvars;

   std::vector<SCIP_CONS*> origscipconss = origdata->relevantconss;
   std::vector<SCIP_CONS*> thisscipconss = relevantconss;
   std::vector<SCIP_VAR*> origscipvars = origdata->relevantvars;
   std::vector<SCIP_VAR*> thisscipvars = relevantvars;

   assert(nrowsother == (int) origscipconss.size() );
   assert(nrowsthis == (int) thisscipconss.size() );

   assert(ncolsother == (int) origscipvars.size() );
   assert(ncolsthis == (int) thisscipvars.size() );

   rowothertothis.assign( nrowsother, - 1 );
   rowthistoother.assign( nrowsthis, - 1 );
   colothertothis.assign( ncolsother, - 1 );
   colthistoother.assign( ncolsthis, - 1 );

   missingrowinthis.clear();

   /* identify new and deleted rows and cols; and identify bijection between maintained variables */
   for( int i = 0; i < nrowsother ; ++i )
   {
      SCIP_CONS* otherrow = origscipconss[i];
      assert( otherrow != NULL );
      SCIP_Bool foundmaintained = false;
      for( int j2 = i; j2 < nrowsthis + i; ++j2 )
      {
         int j = j2 % nrowsthis;
         SCIP_CONS* thisrow = thisscipconss[j];
         assert( SCIPconsIsTransformed( thisrow ) );

         SCIPgetTransformedCons(scip, origscipconss[i], &transcons);
         if( transcons == thisrow)
         {
            rowothertothis[i] = j;
            rowthistoother[j] = i;
            foundmaintained = true;
            break;
         }

         char buffer[SCIP_MAXSTRLEN];
         assert( this->scip != NULL );
         strcpy( buffer, SCIPconsGetName( thisrow ) + 2 );
         assert( this->scip != NULL );
         if( strcmp( SCIPconsGetName( otherrow ), SCIPconsGetName( thisrow ) ) == 0 )
         {
            rowothertothis[i] = j;
            rowthistoother[j] = i;
            foundmaintained = true;
            break;
         }
      }
      if( ! foundmaintained )
      {
         missingrowinthis.push_back( i );
      }
   }

   for( int i = 0; i < ncolsother; ++i )
   {
      SCIP_VAR* othervar;
      SCIP_VAR* probvar;
      SCIP_CALL_ABORT( SCIPgetTransformedVar(scip, origscipvars[i], &othervar ) );
      if (othervar == NULL)
         continue;

      probvar = SCIPvarGetProbvar(othervar);
      if ( probvar == NULL )
         continue;

      for( int j2 = i; j2 < ncolsthis + i; ++j2 )
      {
         int j = j2 % ncolsthis;
         if( probvar == thisscipvars[j] )
         {
            colothertothis[i] = j;
            colthistoother[j] = i;
            break;
         }
      }
   }
}


void DETPROBDATA::getTranslatedPartialdecs(
   std::vector<PARTIALDECOMP*>& origpartialdecs,
   std::vector<int>& rowothertothis,
   std::vector<int>& rowthistoother,
   std::vector<int>& colothertothis,
   std::vector<int>& colthistoother,
   std::vector<PARTIALDECOMP*>& translatedpartialdecs
   )
{
   for(auto otherpartialdec : origpartialdecs)
   {
      PARTIALDECOMP* newpartialdec;

      SCIPverbMessage(this->scip, SCIP_VERBLEVEL_FULL, NULL, " transform partialdec %d \n", otherpartialdec->getID());

      newpartialdec = new PARTIALDECOMP(scip, original);

      /* prepare new partialdec */
      newpartialdec->setNBlocks(otherpartialdec->getNBlocks());

      newpartialdec->setUsergiven(otherpartialdec->getUsergiven());

      /* set all (which have representative in the orig partialdec) constraints according to their representatives in the orig partialdec */
      for( int b = 0; b < otherpartialdec->getNBlocks(); ++ b )
      {
         for( int i = 0; i < otherpartialdec->getNConssForBlock( b ); i ++ )
         {
            int thiscons = rowothertothis[otherpartialdec->getConssForBlock( b )[i]];
            if( thiscons != - 1 )
            {
               newpartialdec->fixConsToBlock( thiscons, b );
            }
         }
      }

      for( int i = 0; i < otherpartialdec->getNMasterconss(); i ++ )
      {
         int thiscons = rowothertothis[otherpartialdec->getMasterconss()[i]];
         if( thiscons != - 1 )
         {
            newpartialdec->fixConsToMaster(thiscons);
         }
      }

      // we do not assign variables as the previous assignment might be invalid due to presolving

      newpartialdec->setDetectorchain(otherpartialdec->getDetectorchain());
      newpartialdec->setAncestorList(otherpartialdec->getAncestorList());

      newpartialdec->addAncestorID(otherpartialdec->getID());

      newpartialdec->copyPartitionStatistics(otherpartialdec);

      for( int i = 0; i < otherpartialdec->getNDetectors(); ++i )
      {
         newpartialdec->addClockTime(otherpartialdec->getDetectorClockTime(i));
         newpartialdec->addPctConssFromFree(otherpartialdec->getPctConssFromFree(i));
         newpartialdec->addPctConssToBlock(otherpartialdec->getPctConssToBlock(i));
         newpartialdec->addPctConssToBorder(otherpartialdec->getPctConssToBorder(i));
         newpartialdec->addPctVarsFromFree(otherpartialdec->getPctVarsFromFree(i));
         newpartialdec->addPctVarsToBlock(otherpartialdec->getPctVarsToBlock(i));
         newpartialdec->addPctVarsToBorder(otherpartialdec->getPctVarsToBorder(i));
         newpartialdec->addNNewBlocks(otherpartialdec->getNNewBlocks(i));
         newpartialdec->addDetectorChainInfo(otherpartialdec->getDetectorchainInfo()[i].c_str());
      }

      newpartialdec->setStemsFromOrig(otherpartialdec->isAssignedToOrigProb());
      newpartialdec->setFinishedByFinisherOrig(otherpartialdec->getFinishedByFinisher());
      otherpartialdec->setTranslatedpartialdecid(newpartialdec->getID());

      if( otherpartialdec->getFinishedByFinisher() )
         newpartialdec->setDetectorFinishedOrig(otherpartialdec->getDetectorchain()[otherpartialdec->getNDetectors() - 1]);

      newpartialdec->setFinishedByFinisher(otherpartialdec->getFinishedByFinisher());
      newpartialdec->prepare();

      newpartialdec->getScore(GCGconshdlrDecompGetScoretype(scip)) ;

      translatedpartialdecs.push_back(newpartialdec);
   }
}


DETPROBDATA::DETPROBDATA(
   SCIP* givenScip,
   SCIP_Bool _originalProblem
   ) :
      scip(givenScip), openpartialdecs(0), ancestorpartialdecs( 0 ), origfixedtozerovars(0),
      nvars(SCIPgetNVars(givenScip)), nconss(SCIPgetNConss(givenScip)), nnonzeros(0), original(_originalProblem), candidatesNBlocks(0),
      classificationtime(0.), nblockscandidatescalctime(0.), postprocessingtime(0.), translatingtime(0.)
{
   SCIP_CONS** conss;
   SCIP_VAR** vars;
   SCIP_Bool createconssadj;

   if( original )
   {
      nvars = SCIPgetNOrigVars(scip);
      nconss = SCIPgetNOrigConss(scip);
   }

   int relevantVarCounter = 0;
   int relevantConsCounter = 0;

   /* initilize matrix datastructures */
   if( original )
   {
      conss = SCIPgetOrigConss(scip);
      vars = SCIPgetOrigVars(scip);
   }
   else
   {
      conss = SCIPgetConss(scip);
      vars = SCIPgetVars(scip);
   }

   /* assign an index to every cons and var
    * @TODO: are all constraints/variables relevant? (probvars etc)  */
   for( int i = 0; i < nconss; ++ i )
   {
      if( SCIPconsIsDeleted( conss[i] ) || SCIPconsIsObsolete(conss[i]) )
      {
         continue;
      }

      if( conss[i] != NULL )
      {
         constoindex[conss[i]] = relevantConsCounter;
         relevantconss.push_back(conss[i]);
         SCIPcaptureCons(scip, conss[i]);

         ++relevantConsCounter;
      }
      else
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL, "relevant cons is NULL\n");
      }
   }

   for( int i = 0; i < nvars; ++ i )
   {
      SCIP_VAR* relevantVar = original ? vars[i] : SCIPvarGetProbvar(vars[i]);

      if( varIsFixedToZero(scip, vars[i]) )
      {
         origfixedtozerovars.push_back(relevantVar);
      } 
      else if( relevantVar != NULL )
      {
         vartoindex[relevantVar] = relevantVarCounter;
         relevantvars.push_back(relevantVar);
         ++ relevantVarCounter;
      }
   }

    /* from here on nvars and nconss represents the relevant numbers */
   nvars = relevantvars.size();
   nconss = relevantconss.size();
   SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL, " nvars: %d / nconss: %d \n", nvars, nconss  );
   varsforconss.resize(nconss);
   valsforconss.resize(nconss);
   conssforvars.resize(nvars);

   /* assumption: now every relevant constraint and variable has its index
    * and is stored in the corresponding unordered_map */
   /* find constraint <-> variable relationships and store them in both directions */
   for( int i = 0; i < (int) relevantconss.size(); ++ i )
   {
      SCIP_CONS* cons;
      SCIP_VAR** currVars = NULL;
      SCIP_Real* currVals = NULL;
      int nCurrVars;

      cons = relevantconss[i];

      nCurrVars = GCGconsGetNVars(scip, cons);
      if( nCurrVars == 0 )
         continue;

      assert(SCIPconsGetName( cons) != NULL);

      SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &currVars, nCurrVars) );
      SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &currVals, nCurrVars) );
      SCIP_CALL_ABORT( GCGconsGetVars(scip, cons, currVars, nCurrVars) );
      SCIP_CALL_ABORT( GCGconsGetVals(scip, cons, currVals, nCurrVars) );

      for( int currVar = 0; currVar < nCurrVars; ++ currVar )
      {
         int varIndex;
         unordered_map<SCIP_VAR*, int>::const_iterator iterVar;

         if( varIsFixedToZero(scip, currVars[currVar]) )
            continue;

         /*@todo remove this after the bug is fixed */
         /* because of the bug of GCGconsGet*()-methods some variables have to be negated */
         if( !SCIPvarIsNegated(currVars[currVar]) )
            iterVar = vartoindex.find(currVars[currVar]);
         else
            iterVar = vartoindex.find(SCIPvarGetNegatedVar(currVars[currVar]));

         if( iterVar == vartoindex.end() )
            continue;

         varIndex = iterVar->second;

         varsforconss[i].push_back(varIndex);
         conssforvars[varIndex].push_back(i);
         valsforconss[i].push_back(currVals[currVar]);
         valsMap[std::pair<int, int>(i, varIndex)] = currVals[currVar];
         ++ nnonzeros;
      }
      SCIPfreeBufferArrayNull( scip, & currVals );
      SCIPfreeBufferArrayNull( scip, & currVars );
   }

   createconssadj = (getNConss() < 1000);

   if( createconssadj )
   {
      createConssAdjacency();
   }
} //end constructor


DETPROBDATA::~DETPROBDATA()
{
   for( int c = 0; c < nconss; ++c )
   {
      SCIP_CONS* cons;

      cons = getCons(c);
      SCIPreleaseCons(scip, &cons);
   }

   // Delete all partialdecs
   GCGconshdlrDecompDeregisterPartialdecs(scip, original);

   for( size_t i = 0; i < conspartitioncollection.size(); ++ i )
   {
      size_t help = conspartitioncollection.size() - i - 1;
      if( conspartitioncollection[help] != NULL )
         delete conspartitioncollection[help];
   }

   for( size_t i = 0; i < varpartitioncollection.size(); ++ i )
   {
      size_t help = varpartitioncollection.size() - i - 1;
      if( varpartitioncollection[help] != NULL )
         delete varpartitioncollection[help];
   }
}


/* @brief adds a candidate for block number and counts how often a candidate is added */
void DETPROBDATA::addCandidatesNBlocksNVotes(
   int candidate, /*< candidate for block size */
   int nvotes     /*< number of votes this candidates will get */
   )
{
   if( candidate > 1 )
   {
      bool alreadyin = false;
      for(auto& candidatesNBlock : candidatesNBlocks)
      {
         if( candidatesNBlock.first == candidate )
         {
            alreadyin = true;
            candidatesNBlock.second = MAX(INT_MAX, candidatesNBlock.second + nvotes);
            break;
         }
      }
      if( !alreadyin )
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL, "added block number candidate: %d \n", candidate);
         candidatesNBlocks.emplace_back(candidate, nvotes);
      }
   }
}


void DETPROBDATA::addConsPartition(
   ConsPartition* partition
   )
{
   SCIP_Bool allowduplicates;

   SCIPgetBoolParam(scip, "detection/classification/allowduplicates", &allowduplicates);

   if( partition != NULL )
   {
      /* check whether there already exists an equivalent conspartition */
      ConsPartition* equiv = NULL;

      for( size_t i = 0; !allowduplicates && i < conspartitioncollection.size(); ++ i )
      {
         if( partition->isDuplicateOf(conspartitioncollection[i]) )
         {
            equiv = conspartitioncollection[i];
            break;
         }
      }

      if( equiv == NULL )
         conspartitioncollection.push_back(partition);
      else
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " Conspartition \"%s\" is not considered since it offers the same structure as \"%s\" conspartition\n", partition->getName(), equiv->getName() );
         delete partition;
      }
   }
}


void DETPROBDATA::addPartialdecToAncestor(
   PARTIALDECOMP* partialdec
   )
{
   ancestorpartialdecs.push_back(partialdec);
}


bool DETPROBDATA::addPartialdecToOpen(
   PARTIALDECOMP* partialdec
   )
{
   assert(partialdec->checkConsistency());
   if( partialdecIsNoDuplicateOfPartialdecs(partialdec, openpartialdecs, true) )
   {
      openpartialdecs.push_back(partialdec);
      return true;
   }
   else
   {
      return false;
   }
}


bool DETPROBDATA::addPartialdecToFinished(
   PARTIALDECOMP* partialdec
   )
{
   assert(partialdec->checkConsistency());
   if( partialdec->isComplete() && partialdecIsNoDuplicateOfPartialdecs(partialdec, finishedpartialdecs, false) )
   {
      finishedpartialdecs.push_back(partialdec);
      return true;
   }
   else
   {
      return false;
   }
}


void DETPROBDATA::addPartialdecToFinishedUnchecked(
   PARTIALDECOMP* partialdec
   )
{
   assert(partialdec->checkConsistency());
   finishedpartialdecs.push_back(partialdec);
}


void DETPROBDATA::addVarPartition(
   VarPartition* partition
   )
{
   SCIP_Bool allowduplicates;

   SCIPgetBoolParam(scip, "detection/classification/allowduplicates", &allowduplicates);

   if( partition != NULL )
   {
      /* check whether there already exists an equivalent varpartition */
      VarPartition* equiv = NULL;

      for( size_t i = 0; !allowduplicates && i < varpartitioncollection.size(); ++ i )
      {
         if( partition->isDuplicateOf(varpartitioncollection[i]) )
         {
            equiv = varpartitioncollection[i];
            break;
         }
      }

      if( equiv == NULL )
         varpartitioncollection.push_back(partition);
      else
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " Varpartition \"%s\" is not considered since it offers the same structure as \"%s\"\n", partition->getName(), equiv->getName() );
         delete partition;
      }

   }
}


void DETPROBDATA::clearAncestorPartialdecs()
{
   ancestorpartialdecs.clear();
}


void DETPROBDATA::clearCurrentPartialdecs()
{
   openpartialdecs.clear();
}


void DETPROBDATA::clearFinishedPartialdecs()
{
   finishedpartialdecs.clear();
}


void DETPROBDATA::createConssAdjacency()
{
   std::set<int> conssadjacenciestemp;
   conssadjacencies.reserve(relevantconss.size());

   /* find constraint <-> constraint relationships and store them in both directions */
   for( size_t i = 0; i < relevantconss.size(); ++i )
   {
      for( size_t varid = 0; varid < varsforconss[i].size(); ++varid )
      {
         int var = varsforconss[i][varid];

         for( int othercons : conssforvars[var] )
         {
            if( othercons == (int) i )
               continue;

            conssadjacenciestemp.insert(othercons);
         }
      }

      conssadjacencies.emplace_back();
      for( int cons : conssadjacenciestemp )
      {
         conssadjacencies[i].push_back(cons);
      }
      conssadjacenciestemp.clear();
   }
}


void DETPROBDATA::freeTemporaryData()
{
   conssadjacencies = std::vector<std::vector<int>>();
}


PARTIALDECOMP* DETPROBDATA::getAncestorPartialdec(
   int partialdecindex
   )
{
   assert( 0 <= partialdecindex && partialdecindex < (int) ancestorpartialdecs.size() );

   return ancestorpartialdecs[partialdecindex];
}


ConsPartition* DETPROBDATA::getConsPartition(
   int partitionIndex
   )
{
   assert(0 <= partitionIndex && partitionIndex < (int) conspartitioncollection.size() );

   return conspartitioncollection[partitionIndex];
}


SCIP_CONS* DETPROBDATA::getCons(
   int consIndex
   )
{
   return relevantconss[consIndex];
}


std::vector<int>& DETPROBDATA::getConssForCons(
   int cons
   )
{
   return conssadjacencies[cons];
}


std::vector<int>& DETPROBDATA::getConssForVar(
   int var
   )
{
   return conssforvars[var];
}


std::vector<PARTIALDECOMP*>& DETPROBDATA::getOpenPartialdecs()
{
   return openpartialdecs;
}


PARTIALDECOMP* DETPROBDATA::getFinishedPartialdec(
   int partialdecindex
   )
{
   assert( 0 <= partialdecindex && partialdecindex < (int) finishedpartialdecs.size() );

   return finishedpartialdecs[partialdecindex];
}


std::vector<PARTIALDECOMP*>& DETPROBDATA::getFinishedPartialdecs()
{
   return finishedpartialdecs;
}


int DETPROBDATA::getIndexForCons(
   SCIP_CONS* cons
   )
{
   assert(constoindex.find(cons) != constoindex.cend());
   return constoindex[cons];
}


int DETPROBDATA::getIndexForCons(
   const char* consname
   )
{
   SCIP_CONS* cons = original ?
      (SCIPfindOrigCons(scip, consname) == NULL ?
      SCIPfindCons(scip, consname): SCIPfindOrigCons(scip, consname)) : SCIPfindCons(scip, consname);
   if( cons == NULL )
   {
      return -1;
   }
   return getIndexForCons(cons);
}


int DETPROBDATA::getIndexForVar(
   const char* varname
   )
{
   return getIndexForVar(SCIPfindVar(scip, varname));
}


int DETPROBDATA::getIndexForVar(
   SCIP_VAR* var
   )
{
   return vartoindex[var];
}


int DETPROBDATA::getNAncestorPartialdecs()
{
   return ancestorpartialdecs.size();
}


int DETPROBDATA::getNConsPartitions()
{
   return (int) conspartitioncollection.size();
}


int DETPROBDATA::getNConss()
{
   return nconss;
}


int DETPROBDATA::getNConssForCons(
   int cons
   )
{
   return conssadjacencies[cons].size();
}


int DETPROBDATA::getNConssForVar(
   int var
   )
{
   return conssforvars[var].size();
}


int DETPROBDATA::getNOpenPartialdecs()
{
   return openpartialdecs.size();
}


int DETPROBDATA::getNFinishedPartialdecs()
{
   return finishedpartialdecs.size();
}


int DETPROBDATA::getNPartialdecs()
{
   return (int) (finishedpartialdecs.size() + openpartialdecs.size());
}


int DETPROBDATA::getNNonzeros()
{
   return nnonzeros;
}


int DETPROBDATA::getNVarPartitions()
{
   return (int) varpartitioncollection.size();
}


int DETPROBDATA::getNVars()
{
   return nvars;
}


int DETPROBDATA::getNVarsForCons(
   int cons
   )
{
   return varsforconss[cons].size();
}


std::vector<SCIP_VAR*> DETPROBDATA::getOrigVarsFixedZero()
{
   return origfixedtozerovars;
}


std::vector<SCIP_CONS*> DETPROBDATA::getRelevantConss()
{
   return relevantconss;
}


std::vector<SCIP_VAR*> DETPROBDATA::getRelevantVars()
{
   return relevantvars;
}


SCIP* DETPROBDATA::getScip()
{
   return scip;
}


void DETPROBDATA::getSortedCandidatesNBlocks(
   std::vector<int>& candidates
   )
{
   int nusercandidates = GCGconshdlrDecompGetNBlockNumberCandidates(scip);
   /* get the block number candidates directly given by the user */
   SCIPdebugMessage("number of user block number candidates: %d\n", nusercandidates);
   for( int i = 0; i < nusercandidates; ++i )
   {
      int candidate = GCGconshdlrDecompGetBlockNumberCandidate(scip, i);
      candidates.push_back(candidate);
      SCIPdebugMessage("  %d\n", candidate);
   }

   /* sort the current candidates */
   std::sort(candidatesNBlocks.begin(), candidatesNBlocks.end(), sort_decr());

   /* add sorted candidates and also */
   SCIPdebugMessage("Sorted Candidates:\n");

   for( auto& candidatesNBlock : candidatesNBlocks )
   {
      SCIPdebugMessage("  %d, %d\n", candidatesNBlock.first, candidatesNBlock.second);

      /* push candidates to output vector */
      candidates.push_back(candidatesNBlock.first);
   }
}


SCIP_Real DETPROBDATA::getVal(
   int row,
   int col
   )
{
   unordered_map<std::pair<int, int>, SCIP_Real, pair_hash>::const_iterator iter = valsMap.find(
      std::pair<int, int>( row, col ) );

   if( iter == valsMap.end() )
      return 0.;

   return iter->second;
}


std::vector<SCIP_Real>& DETPROBDATA::getValsForCons(
   int cons
   )
{
   return valsforconss[cons];
}


VarPartition* DETPROBDATA::getVarPartition(
   int partitionIndex
   )
{
   assert(0 <= partitionIndex && partitionIndex < (int) varpartitioncollection.size() );

   return varpartitioncollection[partitionIndex];
}


std::vector<VarPartition*> DETPROBDATA::getVarPartitions()
{
   return varpartitioncollection;
}


SCIP_VAR* DETPROBDATA::getVar(
   int varIndex
   )
{
   return relevantvars[varIndex];
}


std::vector<int>& DETPROBDATA::getVarsForCons(
   int cons
   )
{
   return varsforconss[cons];
}


bool DETPROBDATA::isConsCardinalityCons(
   int  consindexd
   )
{
   SCIP_CONS* cons;

   cons = relevantconss[consindexd];

   assert(cons != NULL);

   return GCGgetConsIsCardinalityCons(scip, cons);
}


SCIP_Bool DETPROBDATA::isConssAdjInitialized()
{
   return !conssadjacencies.empty();
}


bool DETPROBDATA::isConsSetppc(
   int  consindexd
   )
{
   SCIP_CONS* cons;

   cons = relevantconss[consindexd];

   assert(cons != NULL);

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), "setppc") == 0 )
   {
      switch( SCIPgetTypeSetppc(scip, cons) )
      {
      case SCIP_SETPPCTYPE_COVERING:
      case SCIP_SETPPCTYPE_PARTITIONING:
      case SCIP_SETPPCTYPE_PACKING:
         return true;
      }
   }
   else if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), "logicor") == 0 )
   {
      return true;
   }
   else if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), "linear") == 0 )
   {
      SCIP_SETPPCTYPE type;

      if( GCGgetConsIsSetppc(scip, cons, &type) )
      {
         switch( type )
         {
         case SCIP_SETPPCTYPE_COVERING:
         case SCIP_SETPPCTYPE_PARTITIONING:
         case SCIP_SETPPCTYPE_PACKING:
            return true;
         }
      }
   }

   return false;
}


bool DETPROBDATA::isConsSetpp(
      int  consindexd
      )
{
   SCIP_CONS* cons;

   cons = relevantconss[consindexd];

   assert(cons != NULL);

   if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), "setppc") == 0 )
   {
      switch( SCIPgetTypeSetppc(scip, cons) )
      {
      case SCIP_SETPPCTYPE_PARTITIONING:
      case SCIP_SETPPCTYPE_PACKING:
         return true;
      case SCIP_SETPPCTYPE_COVERING:
         return false;

      }
   }
   else if( strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), "linear") == 0 )
   {
      SCIP_SETPPCTYPE type;

      if( GCGgetConsIsSetppc(scip, cons, &type) )
      {
         switch( type )
         {
         case SCIP_SETPPCTYPE_PARTITIONING:
         case SCIP_SETPPCTYPE_PACKING:
         return true;
         case SCIP_SETPPCTYPE_COVERING:
            return false;

         }
      }
   }

   return false;
}


SCIP_Bool DETPROBDATA::isFiniteNonnegativeIntegral(
   SCIP*                 givenScip,          /**< SCIP data structure */
   SCIP_Real             x                   /**< value */
   )
{
   assert(scip != NULL);

   return (!SCIPisInfinity(givenScip, x) && !SCIPisNegative(givenScip, x) && SCIPisIntegral(givenScip, x));
}


SCIP_Bool DETPROBDATA::isPartialdecDuplicateofFinished(
   PARTIALDECOMP* partialdec
   )
{
   return !(partialdecIsNoDuplicateOfPartialdecs(partialdec, finishedpartialdecs, false));
}


SCIP_Bool DETPROBDATA::isAssignedToOrigProb()
{
   return original;
}


SCIP_Bool DETPROBDATA::isRangedRow(
   SCIP*                 givenScip,               /**< SCIP data structure */
   SCIP_Real             lhs,
   SCIP_Real             rhs
   )
{
   assert(scip != NULL);

   return !(SCIPisEQ(givenScip, lhs, rhs)
      || SCIPisInfinity(givenScip, -lhs) || SCIPisInfinity(givenScip, rhs) );
}


SCIP_Bool DETPROBDATA::partialdecIsNoDuplicateOfPartialdecs(
   PARTIALDECOMP* comppartialdec,
   std::vector<PARTIALDECOMP*> const & partialdecs,
   bool sort
)
{
   assert( comppartialdec != NULL );
   SCIP_Bool isduplicate;

   for( size_t i = 0; i < partialdecs.size(); ++ i )
   {
      assert( partialdecs[i] != NULL );

      comppartialdec->isEqual( partialdecs[i], & isduplicate, sort );
      if( isduplicate )
         return false;
   }
   return true;
}


void DETPROBDATA::printBlockcandidateInformation(
 SCIP*                 givenscip,               /**< SCIP data structure */
 FILE*                 file                /**< output file or NULL for standard output */
   )
{

   std::sort( candidatesNBlocks.begin(), candidatesNBlocks.end(), sort_decr() );
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(givenscip), file, "NBLOCKCANDIDATES   \n" );
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(givenscip), file, "The following %d candidates for the number of blocks are known: (candidate : number of votes)   \n", (int) candidatesNBlocks.size() );
   for( size_t i  = 0; i  < candidatesNBlocks.size(); ++i )
   {
      if( candidatesNBlocks[i].second != INT_MAX )
         SCIPmessageFPrintInfo(SCIPgetMessagehdlr(givenscip), file, "%d : %d  \n", candidatesNBlocks[i].first, candidatesNBlocks[i].second );
      else
         SCIPmessageFPrintInfo(SCIPgetMessagehdlr(givenscip), file, "%d : %s  \n", candidatesNBlocks[i].first, "user given" );
   }
}


void DETPROBDATA::printPartitionInformation(
   FILE*                 file                /**< output file or NULL for standard output */
   )
{
   /* NPARTITION */
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "CONSPARTITION  \n" );
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%d  \n", (int) conspartitioncollection.size()  );

   for(auto partition : conspartitioncollection)
   {
      std::vector<std::vector<int> > conssofclasses = std::vector<std::vector<int> >(partition->getNClasses()) ;
      for( int cons = 0; cons < getNConss(); ++cons )
         conssofclasses[partition->getClassOfCons(cons)].push_back(cons);

      /* PARTITIONNAME */
      SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%s  \n", partition->getName() );


      /* NCLASSES */
      SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%d  \n", partition->getNClasses() );

      for( int cl = 0; cl < partition->getNClasses(); ++cl )
      {
         /* CLASSNAME */
         SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%s: %s\n", partition->getClassName(cl), partition->getClassDescription(cl) );
         /* NMEMBERS */
         SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%ld\n", conssofclasses[cl].size() );
      }
   }

   /* NPARTITION */
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "VARPARTITION  \n" );
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%d  \n", (int) varpartitioncollection.size()  );

   for(auto partition : varpartitioncollection)
   {
      std::vector<std::vector<int> > varsofclasses = std::vector<std::vector<int> >(partition->getNClasses()) ;
      for( int var = 0; var < getNVars(); ++var )
         varsofclasses[partition->getClassOfVar(var)].push_back(var);

      /* PARTITIONNAME */
      SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%s  \n", partition->getName() );


      /* NCLASSES */
      SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%d  \n", partition->getNClasses() );

      for( int cl = 0; cl < partition->getNClasses(); ++cl )
      {
         /* CLASSNAME */
         SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%s: %s\n", partition->getClassName(cl), partition->getClassDescription(cl) );
         /* NMEMBERS */
         SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%ld\n", varsofclasses[cl].size() );
      }
   }
}


 void DETPROBDATA::sortFinishedForScore()
{
   /* get scoretype once, no need to call it twice for every comparison */
   SCORETYPE sctype = GCGconshdlrDecompGetScoretype(scip);
   /* sort by score in descending order */
   std::sort(finishedpartialdecs.begin(), finishedpartialdecs.end(), [&](PARTIALDECOMP* a, PARTIALDECOMP* b) {return (a->getScore(sctype) > b->getScore(sctype)); });
}


std::vector<PARTIALDECOMP*> DETPROBDATA::translatePartialdecs(
   DETPROBDATA* origdata,
   std::vector<PARTIALDECOMP*> origpartialdecs
   )
{
   std::vector<int> rowothertothis;
   std::vector<int> rowthistoother;
   std::vector<int> colothertothis;
   std::vector<int> colthistoother;
   std::vector<int> missingrowinthis;
   std::vector<PARTIALDECOMP*> newpartialdecs;

   calcTranslationMapping( origdata, rowothertothis, rowthistoother, colothertothis, colthistoother, missingrowinthis );

   SCIPverbMessage( this->scip, SCIP_VERBLEVEL_HIGH, NULL,
      " calculated translation; number of missing constraints: %ld; number of other partialdecs: %ld \n", missingrowinthis.size(),
      origpartialdecs.size() );

   getTranslatedPartialdecs( origpartialdecs, rowothertothis, rowthistoother, colothertothis, colthistoother, newpartialdecs );
   return newpartialdecs;
}

std::vector<PARTIALDECOMP*> DETPROBDATA::translatePartialdecs(
   DETPROBDATA* origdata
)
{
   std::vector<int> rowothertothis;
   std::vector<int> rowthistoother;
   std::vector<int> colothertothis;
   std::vector<int> colthistoother;
   std::vector<int> missingrowinthis;
   std::vector<PARTIALDECOMP*> newpartialdecs;

   calcTranslationMapping( origdata, rowothertothis, rowthistoother, colothertothis, colthistoother, missingrowinthis );

   SCIPverbMessage( this->scip, SCIP_VERBLEVEL_HIGH, NULL,
      " calculated translation; number of missing constraints: %ld; number of other partialdecs: %ld \n", missingrowinthis.size(),
      (origdata->getOpenPartialdecs().size() + origdata->getFinishedPartialdecs().size()) );

   getTranslatedPartialdecs( origdata->getOpenPartialdecs(), rowothertothis, rowthistoother, colothertothis, colthistoother, newpartialdecs );
   getTranslatedPartialdecs( origdata->getFinishedPartialdecs(), rowothertothis, rowthistoother, colothertothis, colthistoother, newpartialdecs );
   return newpartialdecs;
}

} /* namespace gcg */
