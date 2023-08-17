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

/**@file   class_partialdecomp.cpp
 * @brief  class storing incomplete decompositions
 * @author Michael Bastubbe
 * @author Hanna Franzen
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

/*#define SCIP_DEBUG*/

#include "class_partialdecomp.h"
#include "class_detprobdata.h"
#include "scip/cons_setppc.h"
#include "scip/scip.h"
#include "scip_misc.h"
#include "struct_detector.h"
#include "struct_decomp.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include "params_visu.h"
#include "miscvisualization.h"
#include "reader_gp.h"
#include "bliss_automorph.hpp"

#include <sstream>
#include <iostream>
#include <exception>
#include <algorithm>
#include <queue>
#include <utility>
#include <stdlib.h>

#ifdef WITH_BLISS
#include "bliss_automorph.h"
#endif


/** macro to throw error if SCIP return status of the called function is not SCIP_OKAY */
#define SCIP_CALL_EXC( x ) do                                                                                 \
                       {                                                                                      \
                          SCIP_RETCODE _restat_;                                                              \
                          if( ( _restat_ = ( x ) ) !=  SCIP_OKAY )                                            \
                          {                                                                                   \
                             SCIPerrorMessage( "Error <%d> in function call\n", _restat_ );                   \
                             throw std::exception();                                                          \
                          }                                                                                   \
                       }                                                                                      \
                       while( FALSE )

namespace gcg {

/** array of prime numbers */
const int PARTIALDECOMP::primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
   101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227,
   229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349};

/** length of primes array */
const int PARTIALDECOMP::nprimes = 70;


PARTIALDECOMP::PARTIALDECOMP(
   SCIP* _scip,
   bool originalProblem
   ) :
   scip( _scip ), nblocks( 0 ), masterconss( 0 ),
   mastervars( 0 ), conssforblocks( 0 ), varsforblocks( 0 ), linkingvars( 0 ), stairlinkingvars( 0 ),
   ncoeffsforblock(std::vector<int>(0)), calculatedncoeffsforblock(FALSE), ncoeffsforblockformastercons(0),
   varsforblocksorted(true), stairlinkingvarsforblocksorted(true),
   conssforblocksorted(true), linkingvarssorted(true), mastervarssorted(true),
   masterconsssorted(true), hashvalue( 0 ), hvoutdated(true), isselected( false ), isagginfoalreadytoexpensive(false), isfinishedbyfinisher( false ),
   nrepblocks(0), reptoblocks(std::vector<std::vector<int>>(0)), blockstorep(std::vector<int>(0) ), pidtopidvarmaptofirst(std::vector<std::vector<std::vector<int> > >(0)),
   detectorchain( 0 ), detectorclocktimes( 0 ), pctvarstoborder( 0 ),
   pctvarstoblock( 0 ), pctvarsfromfree( 0 ), pctconsstoborder( 0 ), pctconsstoblock( 0 ), pctconssfromfree( 0 ),
   nnewblocks( 0 ), usedpartition(0 ), classestomaster(0 ), classestolinking(0 ), listofancestorids(0 ),
   usergiven( USERGIVEN::NOT ), maxwhitescore( -1. ), borderareascore( -1. ), classicscore( -1. ),
   maxforeseeingwhitescore(-1.),
   setpartfwhitescore(-1.), maxforeseeingwhitescoreagg(-1.), setpartfwhitescoreagg(-1.), bendersscore(-1.), strongdecompositionscore(-1.),
   stemsfromorig( false ), original( originalProblem ), translatedpartialdecid( -1 )
{
   // unique id
   id = GCGconshdlrDecompGetNextPartialdecID(scip);

   DETPROBDATA* detprobdata = getDetprobdata();

   nvars = detprobdata->getNVars();
   nconss = detprobdata->getNConss();

   // vector of bools are a special case, no "(n fields, default value)" constructor available
   for(int i = 0; i < nvars; i++)
   {
      isvaropen.push_back(true);
      isvarmaster.push_back(false);
      openvars.push_back(i);
   }

   for(int i = 0; i < nconss; i++)
   {
      isconsopen.push_back(true);
      isconsmaster.push_back(false);
      openconss.push_back(i);
   }

   GCGconshdlrDecompRegisterPartialdec(scip, this);
}


PARTIALDECOMP::PARTIALDECOMP(
   const PARTIALDECOMP *partialdectocopy
   )
{
   scip = ( partialdectocopy->scip );

   // unique id
   id = GCGconshdlrDecompGetNextPartialdecID(scip);

   // rest is copied
   nblocks = partialdectocopy->nblocks;
   nvars = partialdectocopy->nvars;
   nconss = partialdectocopy->nconss;
   masterconss = partialdectocopy->masterconss;
   mastervars = partialdectocopy->mastervars;
   conssforblocks = partialdectocopy->conssforblocks;
   varsforblocks = partialdectocopy->varsforblocks;
   linkingvars = partialdectocopy->linkingvars;
   stairlinkingvars = partialdectocopy->stairlinkingvars;
   openvars = partialdectocopy->openvars;
   openconss = partialdectocopy->openconss;

   isvaropen = partialdectocopy->isvaropen;
   masterconsssorted = partialdectocopy->masterconsssorted;

   isconsopen = partialdectocopy->isconsopen;

   isvarmaster = partialdectocopy->isvarmaster;
   isconsmaster = partialdectocopy->isconsmaster;

   detectorchain = partialdectocopy->detectorchain;
   detectorchaininfo = partialdectocopy->detectorchaininfo;
   hashvalue = partialdectocopy->hashvalue;
   usergiven = partialdectocopy->usergiven;
   classicscore = partialdectocopy->classicscore;
   borderareascore = partialdectocopy->borderareascore;
   maxwhitescore = partialdectocopy->maxwhitescore;
   bendersscore = -1.;
   detectorclocktimes = partialdectocopy->detectorclocktimes;
   pctvarstoborder = partialdectocopy->pctvarstoborder;
   pctvarstoblock = partialdectocopy->pctvarstoblock;
   pctvarsfromfree = partialdectocopy->pctvarsfromfree;
   pctconsstoborder = partialdectocopy->pctconsstoborder;
   pctconsstoblock = partialdectocopy->pctconsstoblock;
   pctconssfromfree = partialdectocopy->pctconssfromfree;
   usedpartition = partialdectocopy->usedpartition;
   classestomaster = partialdectocopy->classestomaster;
   classestolinking = partialdectocopy->classestolinking;
   isfinishedbyfinisher = partialdectocopy->isfinishedbyfinisher;
   ncoeffsforblockformastercons = partialdectocopy->ncoeffsforblockformastercons;
   nnewblocks = partialdectocopy->nnewblocks;
   stemsfromorig = partialdectocopy->stemsfromorig;
   isselected = false;
   original = partialdectocopy->original;
   listofancestorids = partialdectocopy->listofancestorids;
   listofancestorids.push_back(partialdectocopy->id);

   varsforblocksorted = partialdectocopy->varsforblocksorted;
   stairlinkingvarsforblocksorted = partialdectocopy->stairlinkingvarsforblocksorted;
   conssforblocksorted = partialdectocopy->conssforblocksorted;
   linkingvarssorted = partialdectocopy->linkingvarssorted;
   mastervarssorted = partialdectocopy->mastervarssorted;
   hvoutdated = partialdectocopy->hvoutdated;

   nrepblocks  = partialdectocopy->nrepblocks;
   reptoblocks = partialdectocopy->reptoblocks;
   blockstorep = partialdectocopy->blockstorep;
   pidtopidvarmaptofirst = partialdectocopy->pidtopidvarmaptofirst;
   ncoeffsforblock = partialdectocopy->ncoeffsforblock;
   calculatedncoeffsforblock = FALSE;
   translatedpartialdecid = partialdectocopy->getTranslatedpartialdecid();

   maxforeseeingwhitescore = -1.;
   maxforeseeingwhitescoreagg = -1.;
   setpartfwhitescore = -1.;
   setpartfwhitescoreagg = -1.;
   strongdecompositionscore = -1;

   isagginfoalreadytoexpensive = partialdectocopy->isagginfoalreadytoexpensive;

   GCGconshdlrDecompRegisterPartialdec(scip, this);
}


PARTIALDECOMP::~PARTIALDECOMP()
{
   GCGconshdlrDecompDeregisterPartialdec(scip, this);
}


/** @brief checks whether two arrays of SCIP_Real's are identical
 * @returns true iff the given arrays are identical */
static
SCIP_Bool realArraysAreEqual(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real*            array1,             /**< first array */
   int                   array1length,       /**< length of first array */
   SCIP_Real*            array2,             /**< second array */
   int                   array2length        /**< length of second array */
   )
{
   int i;

   if( array1length != array2length )
      return FALSE;

   if( array1length == 0 )
      return TRUE;

   assert(array1 != NULL);
   assert(array2 != NULL);

   for( i = 0; i < array1length; i++ )
   {
      if( !SCIPisEQ(scip, array1[i], array2[i]) )
         return FALSE;
   }

   return TRUE;
}


/** Checks whether the second value of a is lower than the second value of b
 * @returns true iff the second value of a is lower than the second value of b */
static
bool compare_blocks(
   std::pair<int, int> const & a,
   std::pair<int, int> const & b
   )
{
   return ( a.second < b.second );
}


int PARTIALDECOMP::addBlock()
{
   std::vector<int> vector = std::vector<int>( 0 );

   assert( (int) conssforblocks.size() == nblocks );
   assert( (int) varsforblocks.size() == nblocks );
   assert( (int) stairlinkingvars.size() == nblocks );

   conssforblocks.push_back( vector );
   varsforblocks.push_back( vector );
   stairlinkingvars.push_back( vector );
   nblocks ++;
   return nblocks - 1;
}


void PARTIALDECOMP::addClockTime(
   SCIP_Real clocktime
   )
{
   detectorclocktimes.push_back( clocktime );
}


void PARTIALDECOMP::addDecChangesFromAncestor(
   PARTIALDECOMP* ancestor
   )
{
   /* add number of new blocks */
   assert( ancestor != NULL );

   nnewblocks.push_back( getNBlocks() - ancestor->getNBlocks() );
   pctconssfromfree.push_back( getNConss() != 0 ? ( ancestor->getNOpenconss() - getNOpenconss() ) / (SCIP_Real) getNConss() : 0. );
   pctvarsfromfree.push_back( getNVars() != 0 ? ( ancestor->getNOpenvars() - getNOpenvars() ) / (SCIP_Real) getNVars() : 0. );
   pctconsstoblock.push_back( getNConss() != 0 ?
      ( - getNOpenconss() - getNMasterconss() + ancestor->getNOpenconss() + ancestor->getNMasterconss() ) / getNConss() : 0. );
   pctvarstoblock.push_back( getNVars() != 0 ? ( - getNOpenvars() - getNMastervars() - getNLinkingvars() - getNTotalStairlinkingvars() + ancestor->getNOpenvars()
         + ancestor->getNMastervars() + ancestor->getNLinkingvars() + ancestor->getNTotalStairlinkingvars() ) / getNVars() : 0. );
   pctconsstoborder.push_back( getNConss() != 0 ? ( getNMasterconss() - ancestor->getNMasterconss() ) / (SCIP_Real) getNConss() : 0. );
   pctvarstoborder.push_back( getNVars() != 0 ? ( getNMastervars() + getNLinkingvars() + getNTotalStairlinkingvars() - ancestor->getNMastervars()
         - ancestor->getNLinkingvars() - ancestor->getNTotalStairlinkingvars() ) / (SCIP_Real) getNVars() : 0. );
   listofancestorids.push_back( ancestor->getID() );
}


void PARTIALDECOMP::addDetectorChainInfo(
   const char* decinfo
   )
{
   std::stringstream help;
   help << decinfo;
   detectorchaininfo.push_back( help.str() );
}


void PARTIALDECOMP::addEmptyPartitionStatistics()
{
   std::vector<int> emptyVector( 0 );
   usedpartition.push_back(NULL );
   classestomaster.push_back( emptyVector );
   classestolinking.push_back( emptyVector );
}


 void PARTIALDECOMP::addNNewBlocks(
    int newblocks
    )
 {
    nnewblocks.push_back( newblocks );
 }


 void PARTIALDECOMP::addPctConssFromFree(
    SCIP_Real pct
    )
 {
    pctconssfromfree.push_back( pct );
 }


 void PARTIALDECOMP::addPctConssToBlock(
    SCIP_Real pct
    )
 {
    pctconsstoblock.push_back( pct );
 }


 void PARTIALDECOMP::addPctConssToBorder(
    SCIP_Real pct
    )
 {
    pctconsstoborder.push_back( pct );
 }


 void PARTIALDECOMP::addPctVarsFromFree(
    SCIP_Real pct
    )
 {
    pctvarsfromfree.push_back( pct );
 }


 void PARTIALDECOMP::addPctVarsToBlock(
    SCIP_Real pct
    )
 {
    pctvarstoblock.push_back( pct );
 }


 void PARTIALDECOMP::addPctVarsToBorder(
    SCIP_Real pct
    )
 {
    pctvarstoborder.push_back( pct );
 }


bool PARTIALDECOMP::alreadyAssignedConssToBlocks()
{
   for( int b = 0; b < this->nblocks; ++ b )
      if( conssforblocks[b].size() != 0 )
         return true;
   return false;
}


SCIP_RETCODE PARTIALDECOMP::assignBorderFromConstoblock(
   SCIP_HASHMAP* constoblock,
   int givenNBlocks
   )
{
   int cons;
   std::vector<int> del;

   for( int i = 0; i < getNOpenconss(); ++ i )
   {
      cons = openconss[i];
      if( ! SCIPhashmapExists( constoblock, (void*) (size_t) cons ) )
         continue;
      if( (int) (size_t) SCIPhashmapGetImage( constoblock, (void*) (size_t) cons ) - 1 == givenNBlocks )
      {
         setConsToMaster( cons );
         del.push_back(cons);
      }
   }

   /* remove assigned conss from list of open conss */
   for(auto c : del)
      deleteOpencons(c);

   sort();
   assert( checkConsistency() );
   return SCIP_OKAY;
}


bool PARTIALDECOMP::assignCurrentStairlinking(
   )
{
   std::vector<int> blocksOfOpenvar;
   bool assigned = false;
   int var;
   int cons;
   std::vector<int> del;

   DETPROBDATA* detprobdata = this->getDetprobdata();

   /* look for blocks in which the var appears */
   for( int i = 0; i < getNOpenvars(); ++ i )
   {
      blocksOfOpenvar.clear();
      var = openvars.at(i);
      for( int b = 0; b < nblocks; ++ b )
      {
         for( int c = 0; c < getNConssForBlock( b ); ++ c )
         {
            cons = conssforblocks.at(b).at(c);
            if( detprobdata->getVal( cons, var ) != 0 )
            {
               blocksOfOpenvar.push_back( b );
               break;
            }
         }
      }
      /* assign all vars included in two consecutive blocks to stairlinking */
      if( blocksOfOpenvar.size() == 2 && blocksOfOpenvar.at(0) + 1 == blocksOfOpenvar.at(1) )
      {
         setVarToStairlinking(var, blocksOfOpenvar.at(0), blocksOfOpenvar.at(1));
         del.push_back(var);
         assigned = true;
      }
   }

   /* remove assigned vars from open vars list */
   for(auto v : del)
      deleteOpenvar(v);
   
   sort();

   return assigned;
}


bool PARTIALDECOMP::assignHittingOpenconss(
   )
{
   int cons;
   int var;
   int block;
   bool stairlinking; /* true if the cons includes stairlinkingvars */
   bool assigned = false; /* true if open conss get assigned in this function */
   std::vector<int>::iterator it;
   std::vector<int> blocksOfStairlinkingvars; /* first block of the stairlinkingvars which can be found in the conss */
   std::vector<int> blocksOfVars; /* blocks of the vars which can be found in the conss */
   std::vector<int> blocks; /* cons can be assigned to the blocks stored in this vector */
   std::vector<int> eraseBlock;
   std::vector<int> del;

   DETPROBDATA* detprobdata = this->getDetprobdata();

   for( size_t c = 0; c < openconss.size(); ++ c )
   {
      cons = openconss[c];
      stairlinking = false;

      blocksOfVars.clear();
      blocks.clear();
      blocksOfStairlinkingvars.clear();
      eraseBlock.clear();

      /* fill out blocksOfStairlinkingvars and blocksOfBlockvars */
      for( int b = 0; b < nblocks; ++ b )
      {
         for( int v = 0; v < detprobdata->getNVarsForCons( cons ); ++ v )
         {
            var = detprobdata->getVarsForCons( cons )[v];
            if( isVarBlockvarOfBlock( var, b ) )
            {
               blocksOfVars.push_back( b );
               break;
            }
         }
      }

      for( int b = 0; b < nblocks; ++ b )
      {
         for( int v = 0; v < detprobdata->getNVarsForCons( cons ); ++ v )
         {
            int var2 = detprobdata->getVarsForCons(cons)[v];
            std::vector<int>::iterator lb = lower_bound( stairlinkingvars[b].begin(), stairlinkingvars[b].end(), var2 );
            if( lb != stairlinkingvars[b].end() &&  *lb == var2 )
            {
               stairlinking = true;
               blocksOfStairlinkingvars.push_back( b );
               break;
            }
         }
      }

      /* fill out blocks */
      if( stairlinking && blocksOfVars.size() < 2 )
      {
         if( blocksOfVars.size() == 0 )
         {
            blocks.push_back( blocksOfStairlinkingvars[0] );
            blocks.push_back( blocksOfStairlinkingvars[0] + 1 );
            for( size_t i = 1; i < blocksOfStairlinkingvars.size(); ++ i )
            {
               for( it = blocks.begin(); it != blocks.end(); ++ it )
               {
                  if( * it != blocksOfStairlinkingvars[i] && * it != blocksOfStairlinkingvars[i] + 1 )
                     eraseBlock.push_back( * it );
               }
               for( size_t j = 0; j < eraseBlock.size(); ++ j )
               {
                  it = find( blocks.begin(), blocks.end(), eraseBlock[j] );
                  assert( it != blocks.end() );
                  blocks.erase( it );
               }
            }
         }
         else
         {
            blocks.push_back( blocksOfVars[0] );
            for( size_t i = 0; i < blocksOfStairlinkingvars.size(); ++ i )
            {
               if( blocks[0] != blocksOfStairlinkingvars[i] && blocks[0] != blocksOfStairlinkingvars[i] + 1 )
               {
                  blocks.clear();
                  break;
               }
            }
         }
      }

      if( blocksOfVars.size() > 1 )
      {
         setConsToMaster( cons );
         del.push_back(cons);
         assigned = true;
      }
      else if( ! stairlinking && blocksOfVars.size() == 1 )
      {
         setConsToBlock( cons, blocksOfVars[0] );
         del.push_back(cons);
         assigned = true;
      }
      else if( stairlinking && blocks.size() == 0 )
      {
         setConsToMaster( cons );
         del.push_back(cons);
         assigned = true;
      }
      else if( stairlinking && blocks.size() == 1 )
      {
         setConsToBlock( cons, blocks[0] );
         del.push_back(cons);
         assigned = true;
      }
      else if( stairlinking && blocks.size() > 1 )
      {
         block = blocks[0];
         for( size_t i = 1; i < blocks.size(); ++ i )
         {
            if( getNConssForBlock( i ) < getNConssForBlock( block ) )
               block = i;
         }
         setConsToBlock( cons, block );
         del.push_back(cons);
         assigned = true;
      }
   }

   /* remove assigned conss from list of open conss */
   if( assigned )
   {
      for(auto c : del)
         deleteOpencons(c);
      sort();
   }

   return assigned;
}


bool PARTIALDECOMP::assignHittingOpenvars(
   )
{
   int cons;
   int var;
   std::vector<int> blocksOfOpenvar;
   bool found;
   bool assigned = false;
   std::vector<int> del;

   DETPROBDATA* detprobdata = this->getDetprobdata();

   /* set vars to linking, if they can be found in more than one block;
    * set vars to block if they can be found in only one block */
   for( size_t i = 0; i < openvars.size(); ++ i )
   {
      blocksOfOpenvar.clear();
      var = openvars.at(i);
      assert( var >= 0 && var < nvars );
      for( int b = 0; b < nblocks; ++ b )
      {
         found = false;
         for( int c = 0; c < getNConssForBlock( b ) && ! found; ++ c )
         {
            cons = conssforblocks[b][c];
            for( int v = 0; v < detprobdata->getNVarsForCons( cons ) && ! found; ++ v )
            {
               if( detprobdata->getVarsForCons( cons )[v] == var )
               {
                  blocksOfOpenvar.push_back( b );
                  found = true;
               }
            }
         }
      }
      if( blocksOfOpenvar.size() == 1 )
      {
         setVarToBlock( var, blocksOfOpenvar.at(0) );
         del.push_back(var);
         assigned = true;
      }
      else if( blocksOfOpenvar.size() > 1 )
      {
         setVarToLinking(var);
         del.push_back( var );
         assigned = true;
      }
   }

   /* remove the stored vars from open vars list */
   if( assigned )
   {
      for(auto v : del)
         deleteOpenvar(v);

      sort();
   }

   return assigned;
}


void PARTIALDECOMP::assignOpenConssToMaster(
   )
{
   for( auto cons : openconss )
   {
      setConsToMaster(cons);
      isconsopen[cons] = false;
   }
   openconss.clear();
}


void PARTIALDECOMP::assignOpenPartialHittingConsToMaster(
   )
{
   int cons;
   int var;
   std::vector<int> blocksOfBlockvars; /* blocks with blockvars which can be found in the cons */
   std::vector<int> blocksOfOpenvar; /* blocks in which the open var can be found */
   bool master;
   bool hitsOpenVar;
   std::vector<bool> isblockhit;
   std::vector<int> del;

   DETPROBDATA* detprobdata = this->getDetprobdata();

   /* set openconss with more than two blockvars to master */
   for( size_t c = 0; c < openconss.size(); ++ c )
   {
      isblockhit= std::vector<bool>(getNBlocks(), false );
      blocksOfBlockvars.clear();
      master = false;
      hitsOpenVar = false;
      cons = openconss[c];

      for( int v = 0; v < detprobdata->getNVarsForCons( cons ) && ! master; ++ v )
      {
         var = detprobdata->getVarsForCons( cons )[v];

         if( isVarOpenvar( var ) )
         {
            hitsOpenVar = true;
            continue;
         }

         if( isVarMastervar( var ) )
         {
            master = true;
            setConsToMaster( cons );
            del.push_back(cons);
            continue;
         }

         for( int b = 0; b < nblocks; ++ b )
         {
            if( isblockhit[b] )
               continue;

            if( isVarBlockvarOfBlock( var, b ) )
            {
               blocksOfBlockvars.push_back( b );
               isblockhit[b] = true;
               break;
            }
         }
      }
      if( blocksOfBlockvars.size() == 1 && hitsOpenVar )
      {
         setConsToMaster( cons );
         del.push_back(cons);
      }
   }

   /* remove assigned conss from list of open conss */
   for(auto c : del)
      deleteOpencons(c);
   sort();
}


void PARTIALDECOMP::assignOpenPartialHittingToMaster(
   )
{
   assignOpenPartialHittingConsToMaster( );
   assignOpenPartialHittingVarsToMaster( );
}


void PARTIALDECOMP::assignOpenPartialHittingVarsToMaster(
   )
{
   int cons;
   int var;
   std::vector<int> blocksOfBlockvars; /* blocks with blockvars which can be found in the cons */
   std::vector<int> blocksOfOpenvar; /* blocks in which the open var can be found */
   bool hitsOpenCons;
   std::vector<bool> isblockhit;
   SCIP_Bool benders;
   std::vector<int> del;

   DETPROBDATA* detprobdata = this->getDetprobdata();

   SCIPgetBoolParam(scip, "detection/benders/enabled", &benders);

   /* set open var to linking if it can be found in one block and open constraint */
   for( size_t i = 0; i < openvars.size(); ++ i )
   {
      isblockhit= std::vector<bool>(getNBlocks(), false );
      blocksOfOpenvar.clear();
      var = openvars[i];
      hitsOpenCons = false;

      for( int c = 0; c < detprobdata->getNConssForVar( var ); ++ c )
      {
         cons = detprobdata->getConssForVar( var )[c];

         if( benders && isConsMastercons( cons ) )
         {
            continue;
         }

         if( isConsOpencons( cons ) )
         {
            hitsOpenCons = true;
            continue;
         }
         for( int b = 0; b < nblocks; ++ b )
         {
            if ( isblockhit[b] )
               continue;

            if( isConsBlockconsOfBlock( cons, b ) )
            {
               blocksOfOpenvar.push_back( b );
               isblockhit[b] = true;
               break;
            }
         }
      }

      if(  blocksOfOpenvar.size() == 1 && hitsOpenCons )
      {
         setVarToLinking(var);
         del.push_back( var );
      }
   }

   /* revome assigned vars from list of open vars */
   for(auto v : del)
      deleteOpenvar(v);

   sort();
}


SCIP_RETCODE PARTIALDECOMP::assignPartialdecFromConstoblock(
   SCIP_HASHMAP* constoblock,
   int additionalNBlocks
)
{
   int oldNBlocks = nblocks;
   int consblock;
   int cons;
   std::vector<int> del;

   assert( additionalNBlocks >= 0 );

   for( int b = 0; b < additionalNBlocks; ++ b )
      addBlock();

   for( int i = 0; i < getNOpenconss(); ++ i )
   {
      cons = openconss[i];

      if( ! SCIPhashmapExists( constoblock, (void*) (size_t) cons ) )
         continue;
      consblock = oldNBlocks + ( (int) (size_t) SCIPhashmapGetImage( constoblock, (void*) (size_t) cons ) - 1 );
      assert( consblock >= oldNBlocks && consblock <= nblocks );
      if( consblock == nblocks )
      {
         setConsToMaster( cons );
         del.push_back(cons);
      }
      else
      {
         setConsToBlock( cons, consblock );
         del.push_back(cons);
      }
   }

   /* remove assigned conss from list of open conss */
   for(auto c : del)
      deleteOpencons(c);

   deleteEmptyBlocks(false);
   sort();
   assert( checkConsistency( ) );
   return SCIP_OKAY;
}


SCIP_RETCODE PARTIALDECOMP::assignPartialdecFromConstoblockVector(
   std::vector<int> constoblock,
   int additionalNBlocks
      )
{
   int oldNBlocks = nblocks;
   int consblock;
   int cons;
   std::vector<int> del;

   assert( additionalNBlocks >= 0 );

   for( int b = 0; b < additionalNBlocks; ++ b )
      addBlock();

   for( int i = 0; i < getNOpenconss(); ++ i )
   {
      cons = openconss[i];

      if( constoblock[cons] == - 1 )
         continue;

      consblock = oldNBlocks + ( constoblock[cons] - 1 );
      assert( consblock >= oldNBlocks && consblock <= nblocks );
      if( consblock == nblocks )
      {
         setConsToMaster( cons );
         del.push_back(cons);
      }
      else
      {
         setConsToBlock( cons, consblock );
         del.push_back(cons);
      }
   }

   /* remove assigned conss from list of open conss */
   for(auto c : del)
      deleteOpencons(c);

   deleteEmptyBlocks(false);
   sort();
   assert( checkConsistency( ) );
   return SCIP_OKAY;
}


void PARTIALDECOMP::assignSmallestComponentsButOneConssAdjacency()
{
   /* tools to check if the openvars can still be found in a constraint yet */
   std::vector<int> varinblocks(nvars, -1); /* stores, in which block the variable can be found */

   /* tools to update openvars */
   std::vector<int> oldOpenconss;
   std::vector<int> openvarsToDelete;
   gcg::DETPROBDATA* detprobdata = getDetprobdata();

   if( getNLinkingvars() != 0 )
   {
      complete();
      return;
   }

   if ( !detprobdata->isConssAdjInitialized() )
      detprobdata->createConssAdjacency();

   std::vector<bool> isConsOpen(nconss, false);
   std::vector<bool> isConsVisited(nconss, false);

   std::vector<std::vector<int>> conssfornewblocks;
   std::vector<std::vector<int>> varsfornewblocks;

   int newblocks;
   int largestcomponent;
   int sizelargestcomponent;

   newblocks = 0;
   largestcomponent = -1;
   sizelargestcomponent = 0;

   std::queue<int> helpqueue;
   std::vector<int> neighborConss;

   assert( (int) conssforblocks.size() == getNBlocks() );
   assert( (int)varsforblocks.size() == getNBlocks() );
   assert( (int)stairlinkingvars.size() == getNBlocks() );

   if( getNBlocks() < 0 )
      setNBlocks(0);

   /* do breadth first search to find connected conss */
   auto constoconsider = getOpenconssVec();
   while( !constoconsider.empty() )
   {
      std::vector<int> newconss;
      std::vector<int> newvars;

      assert( helpqueue.empty() );
      helpqueue.push(constoconsider[0]);
      neighborConss.clear();
      neighborConss.push_back(constoconsider[0]);
      isConsVisited[constoconsider[0]] = true;

      while( !helpqueue.empty() )
      {
         int nodeCons = helpqueue.front();
         assert( isConsOpencons(nodeCons) );
         helpqueue.pop();
         for( int cons :  detprobdata->getConssForCons(nodeCons) )
         {
            if( isConsVisited[cons] || isConsMastercons(cons) || !isConsOpen[cons] )
               continue;

            assert( isConsOpencons(cons) );
            isConsVisited[cons] = true;
            neighborConss.push_back(cons);
            helpqueue.push(cons);
         }
      }

      /* assign found conss and vars to a new block */
      ++newblocks;
      for( int cons : neighborConss )
      {
         std::vector<int>::iterator consiter = std::lower_bound(constoconsider.begin(), constoconsider.end(), cons);
         assert(consiter != constoconsider.end() );
         constoconsider.erase(consiter);
         assert( isConsOpencons(cons) );
         newconss.push_back(cons);

         for( int var : detprobdata->getVarsForCons(cons) )
         {
            if( isVarLinkingvar(var) || varinblocks[var] != -1 )
               continue;

            assert( !isVarMastervar(var) );
            newvars.push_back(var);
            varinblocks[var] = newblocks;
         }
      }
      conssfornewblocks.push_back(newconss);
      varsfornewblocks.push_back(newvars);
   }

   for( int i = 0; i < newblocks; ++i )
   {
      if( (int)conssfornewblocks[i].size() > sizelargestcomponent )
      {
         sizelargestcomponent = (int)conssfornewblocks[i].size();
         largestcomponent = i;
      }
   }

   if( newblocks > 1 )
   {
      int oldnblocks = getNBlocks();;
      bool largestdone = false;

      setNBlocks(newblocks - 1 + getNBlocks());

      for( int i = 0; i < newblocks; ++i)
      {
         if( i == largestcomponent )
         {
            largestdone = true;
            continue;
         }
         for( int c = 0; c < (int) conssfornewblocks[i].size() ; ++c)
         {
            fixConsToBlock(conssfornewblocks[i][c], oldnblocks + i - (largestdone ? 1 : 0) );
         }

         for( int v = 0; v < (int) varsfornewblocks[i].size() ; ++v )
         {
            fixVarToBlock(varsfornewblocks[i][v], oldnblocks + i - (largestdone ? 1 : 0) );
         }
      }
      prepare();
   }

   assert( checkConsistency() );
}


SCIP_Bool PARTIALDECOMP::isAgginfoTooExpensive()
{

   int limitfornconss;
   int limitfornvars;

   if( isagginfoalreadytoexpensive )
      return TRUE;

   SCIPgetIntParam(scip, "detection/aggregation/limitnconssperblock", &limitfornconss);
   SCIPgetIntParam(scip, "detection/aggregation/limitnvarsperblock", &limitfornvars);

   /* check if calculating aggregation information is too expensive */
   for( int b1 = 0; b1 < getNBlocks() ; ++b1 )
   {
      for( int b2 = b1+1; b2 < getNBlocks(); ++b2 )
      {
         if( getNVarsForBlock(b1) != getNVarsForBlock(b2) )
            continue;

         if( getNConssForBlock(b1) != getNConssForBlock(b2) )
            continue;

         SCIPdebugMessage("Checking  if agg info is too expensive for blocks %d and %d, nconss: %d, nvars: %d . \n", b1, b2, getNConssForBlock(b2), getNVarsForBlock(b2) );
         if( getNConssForBlock(b2) >= limitfornconss || getNVarsForBlock(b2) >= limitfornvars )
         {
            SCIPdebugMessage("Calculating agg info is too expensive, nconss: %d, nvars: %d . \n", getNConssForBlock(b2), getNVarsForBlock(b2) );
            isagginfoalreadytoexpensive = true;
            return TRUE;
         }
      }

   }

   /* check if there are too many master coeffs */
   SCIPdebugMessage("Calculated: agg info is NOT too expensive.\n");
   return FALSE;
}


void PARTIALDECOMP::calcAggregationInformation(
   bool ignoreDetectionLimits
   )
{
#ifdef WITH_BLISS
   SCIP_Bool tooexpensive;
   SCIP_Bool usebliss;
   int searchnodelimit;
   int generatorlimit;
#endif
   SCIP_Bool aggisnotactive;
   SCIP_Bool discretization;
   SCIP_Bool aggregation;

   int nreps = 1;

   if( aggInfoCalculated() )
      return;

   if( !isComplete() )
      return;

#ifdef WITH_BLISS
   if(
#if defined(BLISS_PATCH_PRESENT) || BLISS_VERSION_MAJOR >= 1 || BLISS_VERSION_MINOR >= 76
         !ignoreDetectionLimits &&
#endif
         isAgginfoTooExpensive()
         )
      tooexpensive = TRUE;
   else
      tooexpensive = FALSE;
   SCIPgetBoolParam(scip, "relaxing/gcg/bliss/enabled", &usebliss);
   SCIPgetIntParam(scip, "relaxing/gcg/bliss/searchnodelimit", &searchnodelimit);
   SCIPgetIntParam(scip, "relaxing/gcg/bliss/generatorlimit", &generatorlimit);
#endif

   SCIPgetBoolParam(scip, "relaxing/gcg/aggregation", &aggregation);
   SCIPgetBoolParam(scip, "relaxing/gcg/discretization", &discretization);

   if( discretization && aggregation )
      aggisnotactive = FALSE;
   else
      aggisnotactive = TRUE;

   std::vector<std::vector<int>> identblocksforblock( getNBlocks(), std::vector<int>(0) );

   blockstorep = std::vector<int>(getNBlocks(), -1);

   for( int b1 = 0; b1 < getNBlocks() ; ++b1 )
   {
      std::vector<int> currrep = std::vector<int>(0);
      std::vector< std::vector<int> > currrepvarmapforthisrep =std::vector<std::vector<int>>(0);
      std::vector<int> identityvec = std::vector<int>(0);


      if( !identblocksforblock[b1].empty() )
         continue;

      for( int i = 0; i  < getNVarsForBlock(b1); ++i )
         identityvec.push_back(i);

      currrep.push_back(b1);
      currrepvarmapforthisrep.push_back(identityvec);


      for( int b2 = b1+1; b2 < getNBlocks(); ++b2 )
      {
         SCIP_Bool identical;
         SCIP_Bool notidentical;
         std::vector<int> varmap;
         SCIP_HASHMAP* varmap2;

         notidentical = FALSE;
         identical = FALSE;

         if( !identblocksforblock[b2].empty() )
            continue;

         if( aggisnotactive )
            continue;


         SCIP_CALL_ABORT( SCIPhashmapCreate(  &varmap2,
               SCIPblkmem(scip),
               5 * getNVarsForBlock(b1)+1) ); /* +1 to deal with empty subproblems */

         SCIPdebugMessage("Check identity for block %d and block %d!\n", b1, b2);

         checkIdenticalBlocksTrivial( b1, b2, &notidentical);

         if( !notidentical )
         {
            checkIdenticalBlocksBrute( b1, b2, varmap, varmap2, &identical);

#ifdef WITH_BLISS
            if( usebliss && !tooexpensive && !identical )
               checkIdenticalBlocksBliss(b1, b2, varmap, varmap2, &identical,
                     searchnodelimit >= 0 ? searchnodelimit : 0u, generatorlimit >= 0 ? generatorlimit : 0u);
#endif
         }
         else
            identical = FALSE;

         if( identical )
         {
            SCIPdebugMessage("Block %d is identical to block %d!\n", b1, b2);
            identblocksforblock[b1].push_back(b2);
            identblocksforblock[b2].push_back(b1);
            currrep.push_back(b2);
            /* handle varmap */
            currrepvarmapforthisrep.push_back(varmap);

         }
         else
         {
            SCIPdebugMessage("Block %d is not identical to block %d!\n", b1, b2);
         }
         SCIPhashmapFree(&varmap2);
      }

      reptoblocks.push_back( currrep );
      pidtopidvarmaptofirst.push_back(currrepvarmapforthisrep);
      for( size_t i = 0; i < currrep.size(); ++i )
         blockstorep[currrep[i]] = nreps-1;
      ++nreps;

   }
   nrepblocks = nreps-1;
}


void PARTIALDECOMP::calcHashvalue()
{
   if( !hvoutdated )
      return;

   std::vector<std::pair<int, int>> blockorder = std::vector < std::pair<int, int> > ( 0 );
   unsigned long hashval = 0;
   unsigned long borderval = 0;

   sort();

   /* find sorting for blocks (non decreasing according smallest row index) */
   for( int i = 0; i < this->nblocks; ++ i )
   {
      if( !this->conssforblocks[i].empty() )
         blockorder.emplace_back( i, this->conssforblocks[i][0] );
      else
      {
         assert( this->varsforblocks[i].size() > 0 );
         blockorder.emplace_back( i, this->getNConss() + this->varsforblocks[i][0] );
      }
   }

   std::sort( blockorder.begin(), blockorder.end(), compare_blocks );

   for( int i = 0; i < nblocks; ++ i )
   {
      unsigned long blockval = 0;
      int blockid = blockorder[i].first;

      for( size_t tau = 0; tau < conssforblocks[blockid].size(); ++ tau )
      {
         blockval += ( 2 * conssforblocks[blockid][tau] + 1 ) * (unsigned long) pow( 2, tau % 16 );
      }

      hashval += primes[i % ( nprimes - 1 )] * blockval;
   }

   for( size_t tau = 0; tau < masterconss.size(); ++ tau )
   {
      borderval += ( 2 * masterconss[tau] + 1 ) * (unsigned long) pow( 2, tau % 16 );
   }

   hashval += primes[nblocks % nprimes] * borderval;
   hashval += primes[( nblocks + 1 ) % nprimes] * openvars.size();

   hashvalue = hashval;
   hvoutdated = false;
}


void PARTIALDECOMP::calcNCoeffsForBlocks()
{
   if( calculatedncoeffsforblock )
      return;

   ncoeffsforblock = std::vector<int>(getNBlocks(), 0);
   int counter;
   DETPROBDATA* detprobdata = this->getDetprobdata();

   for( int b = 0; b < getNBlocks(); ++b )
   {
      counter = 0;
      for( int blco = 0; blco < getNConssForBlock(b); ++blco )
      {
            int consid = getConssForBlock(b)[blco];
            for( int cva = 0; cva < detprobdata->getNVarsForCons(consid) ;++cva )
               if( isVarBlockvarOfBlock(detprobdata->getVarsForCons(consid)[cva], b ) )
                  ++counter;
      }
      ncoeffsforblock[b] = counter;
   }

   counter = 0;
   for( int mco = 0; mco < getNMasterconss(); ++mco )
   {
         int consid = getMasterconss()[mco];
         counter += detprobdata->getNVarsForCons(consid);
   }
   ncoeffsformaster = counter;

   calculatedncoeffsforblock = TRUE;
}


void PARTIALDECOMP::calcStairlinkingVars(
   )
{
   assert( getNTotalStairlinkingvars() == 0 );

   /* data structure containing pairs of varindices and blocknumbers */
   std::vector< std::pair< int, std::vector< int > > > blocksOfVars = findLinkingVarsPotentiallyStairlinking( );

   /* if there are no vars that are potentially stairlinking, return without further calculations */
   if( blocksOfVars.size() == 0 )
      return;

   GraphGCG* g = new GraphGCG( getNBlocks(), true );

   /* create block graph: every block is represented by a node and two nodes are adjacent if there exists a
    * var that potentially links these blocks, the edge weight is the number of such variables */
   for( int i = 0; i < (int) blocksOfVars.size(); ++i )
   {
      assert( blocksOfVars[i].second.size() == 2 );
      int v = blocksOfVars[i].second[0];
      int w = blocksOfVars[i].second[1];

      if ( g->isEdge( v, w ) )
      {
         g->setEdge( v, w, g->getEdgeWeight( v, w ) + 1 );
      }
      else
      {
         g->setEdge( v, w, 1 );
      }
   }

   bool isstaircase = true; /* maintains information whether staircase structure is still possible */
   std::vector< int > sources( 0 ); /* all nodes with degree one */
   std::vector< bool > marks( getNBlocks() ); /* a node is marked if its degree is zero or it is reachable from a source  */

   /* firstly, check whether every node has an degree of at most 2 */
   for( int b = 0; b < getNBlocks(); ++b )
   {
      if( g->getNNeighbors( b ) > 2 )
      {
         isstaircase = false;
         break;
      }
      else if( g->getNNeighbors( b ) == 1 )
      {
         sources.push_back( b );
      }
      else if ( g->getNNeighbors( b ) == 0 )
      {
         marks[b] = true;
      }
   }

   /* secondly, check whether there exists a circle in the graph by moving along all paths starting from a source */
   for( int s = 0; s < (int) sources.size() && isstaircase; ++s )
   {
      int curBlock = sources[s];
      if( marks[curBlock] )
         continue;

      marks[curBlock] = true;

      /* check whether there is an unmarked neighbor
       * if there is none, a circle is detected */
      do
      {
         std::vector< int > neighbors = g->getNeighbors( curBlock );
         if( !marks[neighbors[0]] )
         {
            marks[neighbors[0]] = true;
            curBlock = neighbors[0];
         }
         else if ( !marks[neighbors[1]] )
         {
            marks[neighbors[1]] = true;
            curBlock = neighbors[1];
         }
         else
         {
            isstaircase = false;
            break;
         }
      }
      while( g->getNNeighbors( curBlock ) != 1 );
   }

   /* thirdly, check whether all nodes with neighbors are reachable from a source,
    * since there is a circle if this is not the case */
   for( int b = 0; b < getNBlocks() && isstaircase; ++b )
   {
      if( !marks[b] )
      {
         isstaircase = false;
         break;
      }
   }

   if( isstaircase )
   {
      changeBlockOrderStaircase( g );
   }
   else
   {
      return;
   }

   findVarsLinkingToStairlinking( );

   assert( checkConsistency( ) );
}


void PARTIALDECOMP::changeBlockOrderStaircase(
   GraphGCG* g
   )
{
   int blockcounter = 0; /* counts current new block to assign an old one to */
   std::vector< int > blockmapping( getNBlocks() ); /* stores new block order */
   for( int b = 0; b < getNBlocks(); ++b )
      blockmapping[b] = -1;

   for( int b = 0; b < getNBlocks(); ++b )
   {
      if( g->getNNeighbors( b ) == 0 )
      {
         /* if block does not have a neighbor, just assign it to current blockindex */
         assert( blockmapping[b] == -1 );
         blockmapping[b] = blockcounter;
         ++blockcounter;
      }
      else if( blockmapping[b] == -1 && g->getNNeighbors( b ) == 1 )
      {
         /* if the block is the source of an yet unconsidered path, assign whole path to ascending new block ids */
         int curBlock = b;
         blockmapping[b] = blockcounter;
         ++blockcounter;

         do
         {
            std::vector< int > neighbors = g->getNeighbors( curBlock );

            if( blockmapping[neighbors[0]] == -1 )
            {
               blockmapping[neighbors[0]] = blockcounter;
               curBlock = neighbors[0];
            }
            else if ( blockmapping[neighbors[1]] == -1 )
            {
               blockmapping[neighbors[1]] = blockcounter;
               curBlock = neighbors[1];
            }
            else
            {
               assert( false );
            }
            ++blockcounter;
         }
         while( g->getNNeighbors( curBlock ) != 1 );
      }
   }

   changeBlockOrder( blockmapping );
}


void PARTIALDECOMP::changeBlockOrder(
   std::vector<int> oldToNewBlockIndex
   )
{
   assert((int ) oldToNewBlockIndex.size() == getNBlocks() );
   assert( getNTotalStairlinkingvars() == 0 );

   std::vector< std::vector< int > > newconssforblocks( getNBlocks() );
   std::vector< std::vector< int > > newvarsforblocks( getNBlocks() );

   for( int b = 0; b < getNBlocks(); ++b )
   {
      assert( 0 <= oldToNewBlockIndex[b] && oldToNewBlockIndex[b] < getNBlocks() );

      newconssforblocks[oldToNewBlockIndex[b]] = conssforblocks[b];
      newvarsforblocks[oldToNewBlockIndex[b]] = varsforblocks[b];
   }

   conssforblocks = newconssforblocks;
   varsforblocks = newvarsforblocks;
}


bool PARTIALDECOMP::checkAllConssAssigned()
{
   for( size_t i = 0; i < openconss.size(); ++ i )
   {
      bool consfound = false;
      for( size_t k = 0; k < masterconss.size(); ++ k )
      {
         if( openconss[i] == masterconss[k] )
         {
            consfound = true;
            break;
         }
      }
      for( int b = 0; b < nblocks && ! consfound; ++ b )
      {
         for( size_t k = 0; k < conssforblocks[b].size(); ++ k )
         {
            if( openconss[i] == conssforblocks[b][k] )
            {
               consfound = true;
               break;
            }
         }
      }
      if( ! consfound )
      {
         return false;
      }
   }
   openconss.clear();
   isconsopen = std::vector<bool>(nconss, false);
   return true;
}


bool PARTIALDECOMP::checkConsistency(
   )
{
   std::vector<bool> openvarsBool( nvars, true );
   std::vector<int> stairlinkingvarsvec( 0 );
   std::vector<int>::const_iterator varIter = linkingvars.begin();
   std::vector<int>::const_iterator varIterEnd = linkingvars.end();

   int value;

   /* check if nblocks is set appropriately */
   if( nblocks != (int) conssforblocks.size() )
   {
      SCIPwarningMessage(scip, "In (partialdec %d) nblocks %d and size of conssforblocks %ld are not identical! \n" , id, nblocks, conssforblocks.size() );
      assert( false );
      return false;
   }

   if( nblocks != (int) varsforblocks.size() )
   {
      SCIPwarningMessage(scip, "In (partialdec %d) nblocks %d and size of varsforblocks %ld are not identical! \n" , id, nblocks, varsforblocks.size() );
      assert( false );
      return false;
   }

   /* check for empty (row- and col-wise) blocks */
   for( int b = 0; b < nblocks; ++ b )
   {
      if( conssforblocks[b].empty() && varsforblocks[b].empty() )
      {
         SCIPwarningMessage(scip, "In (partialdec %d) block %d is empty! \n" , id, b );
         assert( false );
         return false;
      }
   }

   /* check variables (every variable is assigned at most once) */
   for( ; varIter != varIterEnd; ++ varIter )
   {
      if( ! openvarsBool[ * varIter] )
      {
         SCIPwarningMessage(scip, "In (partialdec %d) linking variable with index %d is already assigned! \n" , id, * varIter );

         assert( false );
         return false;
      }
      openvarsBool[ * varIter] = false;
   }

   for( int b = 0; b < nblocks; ++ b )
   {
      varIterEnd = varsforblocks[b].end();
      for( varIter = varsforblocks[b].begin(); varIter != varIterEnd; ++ varIter )
      {
         if( ! openvarsBool[ * varIter] )
         {
            SCIPwarningMessage(scip, "In (partialdec %d) variable with index %d is already assigned but also assigned to block %d! \n" , id, * varIter, b );
            assert( false );
            return false;
         }
         openvarsBool[ * varIter] = false;
      }
   }

   varIterEnd = mastervars.end();
   for( varIter = mastervars.begin(); varIter != varIterEnd; ++ varIter )
   {
      if( ! openvarsBool[ * varIter] )
      {
         SCIPwarningMessage(scip, "In (partialdec %d) variable with index %d is already assigned but also assigned to master! \n" , id, * varIter);
         assert( false );
         return false;
      }
      openvarsBool[ * varIter] = false;
   }

   for( int b = 0; b < nblocks; ++ b )
   {
      varIter = stairlinkingvars[b].begin();
      varIterEnd = stairlinkingvars[b].end();
      for( ; varIter != varIterEnd; ++ varIter )
      {
         if( ! openvarsBool[ * varIter] )
         {
            SCIPwarningMessage(scip, "In (partialdec %d) variable with index %d is already assigned but also assigned to stairlinking block %d! \n" , id, * varIter, b );
            assert( false );
            return false;
         }
         openvarsBool[ * varIter] = false;
      }
      if( ( b == nblocks - 1 ) && ( (int) stairlinkingvars[b].size() != 0 ) )
      {
         SCIPwarningMessage(scip, "In (partialdec %d) variable with index %d is is as assigned as stairlinking var of last block! \n" , id, * varIter );
         assert( false );
         return false;
      }
   }

   /* check if all not assigned variables are open vars */
   for( int v = 0; v < nvars; ++ v )
   {
      if( openvarsBool[v] && !isVarOpenvar(v) )
      {
         SCIPwarningMessage(scip, "In (partialdec %d) variable with index %d is not assigned and not an open var! \n" , id, v );
         assert( false );
         return false;
      }
   }

   /* check if all open vars are not assigned */
   for( size_t i = 0; i < openvars.size(); ++ i )
   {
      if( !openvarsBool[openvars[i]] )
      {
         SCIPwarningMessage(scip, "In (partialdec %d) variable with index %d is an open var but assigned! \n" , id, openvars[i]  );
         assert( false );
         return false;
      }
   }

   for( size_t i = 0; i < openvarsBool.size(); ++ i )
   {
      if( openvarsBool[i] != isvaropen[i] )
      {
         SCIPwarningMessage(scip, "In (partialdec %d) variable with index %d is causes asynchronity with isvaropen array ! \n" , id, openvars[i]  );
         assert( false );
         return false;

      }
   }

   /* check constraints (every constraint is assigned at most once) */
   std::vector<bool> openconssBool( nconss, true );
   std::vector<int> openconssVec( 0 );
   std::vector<int>::const_iterator consIter = masterconss.begin();
   std::vector<int>::const_iterator consIterEnd = masterconss.end();

   for( ; consIter != consIterEnd; ++ consIter )
   {
      if( ! openconssBool[ * consIter] )
      {
         SCIPwarningMessage(scip, "In (partialdec %d) constraint with index %d is at least two times assigned as a master constraint! \n" , id, * consIter  );
         assert( false );
         return false;
      }
      openconssBool[ * consIter] = false;
   }

   for( int b = 0; b < nblocks; ++ b )
   {
      consIterEnd = conssforblocks[b].end();
      for( consIter = conssforblocks[b].begin(); consIter != consIterEnd; ++ consIter )
      {
         if( ! openconssBool[ * consIter] )
         {
            SCIPwarningMessage(scip, "In (partialdec %d) constraint with index %d is already assigned but also assigned to block %d! \n" , id, * consIter, b  );
            assert( false );
            return false;
         }
         openconssBool[ * consIter] = false;
      }
   }

   /* check if all not assigned constraints are open cons */
   for( int v = 0; v < nconss; ++ v )
   {
      if( openconssBool[v] && !isConsOpencons(v) )
      {
         SCIPwarningMessage(scip, "In (partialdec %d) constraint with index %d is not assigned and not an open cons! \n" , id, v  );
         assert( false );
         return false;
      }
   }

   /* check if all open conss are not assigned */
   for( size_t i = 0; i < openconss.size(); ++ i )
   {
      if( !openconssBool[openconss[i]] )
      {
         SCIPwarningMessage(scip, "In (partialdec %d) constraint with index %d is an open cons but assigned! \n" , id,  openconss[i] );
         assert( false );
         return false;
      }
   }

   /* check if the partialdec is sorted */
   for( int b = 0; b < nblocks; ++ b )
   {
      value = - 1;
      for( int v = 0; v < getNVarsForBlock( b ); ++ v )
      {
         if( value >= getVarsForBlock(b)[v] )
         {
            SCIPwarningMessage(scip, "In (partialdec %d) variables of block %d are not sorted! \n" , id,  b );
            assert( false );
            return false;
         }
         value = getVarsForBlock( b )[v];
      }
   }
   for( int b = 0; b < nblocks; ++ b )
   {
      value = - 1;
      for( int v = 0; v < getNStairlinkingvars( b ); ++ v )
      {
         if( value >= getStairlinkingvars(b)[v] )
         {
            SCIPwarningMessage(scip, "In (partialdec %d) stairlinking variables of block %d are not sorted! \n" , id,  b );
            assert( false );
            return false;
         }
         value = getStairlinkingvars( b )[v];
      }
   }
   value = - 1;
   for( int v = 0; v < getNLinkingvars(); ++ v )
   {
      if( value >= getLinkingvars()[v] )
      {
         SCIPwarningMessage(scip, "In (partialdec %d) linking variables are not sorted! \n" , id );
         assert( false );
         return false;
      }
      value = getLinkingvars()[v];
   }
   value = - 1;
   for( int v = 0; v < getNMastervars(); ++ v )
   {
      if( value >= getMastervars()[v] )
      {
         SCIPwarningMessage(scip, "In (partialdec %d) master variables are not sorted! \n" , id );
         assert( false );
         return false;
      }
      value = getMastervars()[v];
   }
   for( int b = 0; b < nblocks; ++ b )
   {
      value = - 1;
      for( int v = 0; v < getNConssForBlock(b); ++ v )
      {
         if( value >= getConssForBlock(b)[v] )
         {
            SCIPwarningMessage(scip, "In (partialdec %d) constraints of block %d are not sorted! \n" , id,  b );
            assert( false );
            return false;
         }
         value = getConssForBlock(b)[v];
      }
   }
   value = - 1;
   for( int v = 0; v < getNMasterconss(); ++ v )
   {
      if( value >= getMasterconss()[v] )
      {
         SCIPwarningMessage(scip, "In (partialdec %d) master constraints are not sorted! \n" , id);
         assert( false );
         return false;
      }
      value = getMasterconss()[v];
   }

   DETPROBDATA* detprobdata = this->getDetprobdata();
   /* check if variables hitting a cons are either in the cons's block or border or still open */
   for( int b = 0; b < nblocks; ++ b )
   {
      for( int c = 0; c < getNConssForBlock( b ); ++ c )
      {
         for( int v = 0; v < detprobdata->getNVarsForCons( getConssForBlock( b )[c] ); ++ v )
         {
            int varid = detprobdata->getVarsForCons( getConssForBlock( b )[c] )[v];

            if( !(isVarBlockvarOfBlock(varid, b) || isVarLinkingvar(varid) || isVarStairlinkingvarOfBlock(varid, b)
               || isVarOpenvar(varid)) )
            {
               SCIP_Bool partofblock;

               partofblock = FALSE;

               SCIPwarningMessage(scip,
                  "This should only happen during translation of (partial) decompositions from orginal to transformed problem, and means that translation has failed for this particaluar partial decomposition. Variable %d is not part of block %d or linking or open as constraint %d suggests! \n ", varid, b,
                  getConssForBlock(b)[c]);

               for( int b2 = 0; b2 < getNBlocks(); ++b2 )
               {
                  if ( isVarBlockvarOfBlock(varid, b2) )
                  {
                     partofblock = TRUE;
                     SCIPwarningMessage(scip,
                        "instead Variable %d is part of block %d  \n ", varid, b2);
                     break;
                  }
               }

               if( !partofblock )
               {
                  if( isvarmaster[varid] )
                     SCIPwarningMessage(scip, "instead Variable %d is part of master  \n ", varid);
                  else
                     SCIPwarningMessage(scip, "in fact Variable %d is completely unassigned  \n ", varid);
               }
               assert(false);
               return false;
            }
         }
      }
   }

   if( getDetectorchain().size() != getDetectorchainInfo().size() )
   {
      assert(false);
      return false;
   }

   if( getNDetectors() != (int) pctvarstoblock.size() || getNDetectors() != (int) pctvarstoborder.size()
      || getNDetectors() != (int) pctvarsfromfree.size() || getNDetectors() != (int) pctconsstoblock.size()
      || getNDetectors() != (int) pctconsstoborder.size() || getNDetectors() != (int) pctconssfromfree.size() )
   {
      assert(false);
      return false;
   }

   return true;
}


#ifdef WITH_BLISS
void PARTIALDECOMP::checkIdenticalBlocksBliss(
   int                  b1,
   int                  b2,
   std::vector<int>&    varmap,
   SCIP_HASHMAP*        varmap2,
   SCIP_Bool*           identical,
   unsigned int         searchnodelimit,
   unsigned int         generatorlimit
   )
{
   *identical = FALSE;
   SCIP_HASHMAP* consmap;
   SCIP_Result result;

   varmap = std::vector<int>(getNVarsForBlock(b1), -1);

   SCIP_CALL_ABORT( SCIPhashmapCreate(&consmap,
      SCIPblkmem(scip ),
      getNConssForBlock(b1)+1) ); /* +1 to deal with empty subproblems */


   SCIPdebugMessage("obvious test fails, start building graph \n");

   cmpGraphPair(scip, this, b1, b2, &result, varmap2, consmap, searchnodelimit, generatorlimit);
   if( result == SCIP_SUCCESS )
   {
      *identical = TRUE;
      /** TODO translate varmaps */
      for( int var2idinblock = 0; var2idinblock < getNVarsForBlock(b2) ; ++var2idinblock )
      {
         SCIP_VAR* var2;
         SCIP_VAR* var1;
         int var1idinblock;
         int var1id;
         auto detprobdata = this->getDetprobdata();

         var2 = detprobdata->getVar(getVarsForBlock(b2)[var2idinblock]);
         var1 = (SCIP_VAR*) SCIPhashmapGetImage(varmap2, (void*) var2);
         var1id = detprobdata->getIndexForVar(var1);
         var1idinblock = getVarProbindexForBlock(var1id, b1);
         varmap[var2idinblock] = var1idinblock;
      }

   }
   else
      *identical = FALSE;

   SCIPhashmapFree(&consmap);

   return;

}
#endif


void PARTIALDECOMP::checkIdenticalBlocksBrute(
   int                  b1,
   int                  b2,
   std::vector<int>&    varmap,
   SCIP_HASHMAP*        varmap2,
   SCIP_Bool*           identical
   )
{


   *identical = FALSE;
   SCIPdebugMessage("check block %d and block %d for identity...\n", b1, b2);
   varmap = std::vector<int>(getNVars(), -1);
   DETPROBDATA* detprobdata = this->getDetprobdata();

   /* check variables */
   for( int i = 0; i < getNVarsForBlock(b1); ++i )
   {
      SCIP_VAR* var1;
      SCIP_VAR* var2;

      var1 = detprobdata->getVar(getVarsForBlock(b1)[i]);
      var2 = detprobdata->getVar(getVarsForBlock(b2)[i]);

      if( !SCIPisEQ(scip, SCIPvarGetObj(var1), SCIPvarGetObj(var2) ) )
      {
         SCIPdebugMessage("--> obj differs for var %s and var %s!\n", SCIPvarGetName(var1), SCIPvarGetName(var2));
             return;
      }
      if( !SCIPisEQ(scip, SCIPvarGetLbGlobal(var1), SCIPvarGetLbGlobal(var2) ) )
      {
         SCIPdebugMessage("--> lb differs for var %s and var %s!\n", SCIPvarGetName(var1), SCIPvarGetName(var2));
             return;
      }
      if( !SCIPisEQ(scip, SCIPvarGetUbGlobal(var1), SCIPvarGetUbGlobal(var2) ) )
      {
         SCIPdebugMessage("--> ub differs for var %s and var %s!\n", SCIPvarGetName(var1), SCIPvarGetName(var2));
             return;
      }
      if( SCIPvarGetType(var1) != SCIPvarGetType(var2) )
      {
         SCIPdebugMessage("--> type differs for var %s and var %s!\n", SCIPvarGetName(var1), SCIPvarGetName(var2));
             return;
      }

      for( int mc = 0; mc < getNMasterconss(); ++mc )
      {

         if( !SCIPisEQ(scip, detprobdata->getVal(getMasterconss()[mc], getVarsForBlock(b1)[i]), detprobdata->getVal(getMasterconss()[mc], getVarsForBlock(b2)[i])  ))
         {
            SCIPdebugMessage("--> master coefficients differ for var %s (%f) and var %s  (%f) !\n", SCIPvarGetName(
               detprobdata->getVar(getVarsForBlock(b1)[i]) ), detprobdata->getVal(getMasterconss()[mc], getVarsForBlock(b1)[i]), SCIPvarGetName(
               detprobdata->getVar(getVarsForBlock(b2)[i])), detprobdata->getVal(getMasterconss()[mc], getVarsForBlock(b2)[i])  );
            return;
         }
      }

      /* variables seem to be identical so far */
      varmap[getVarsForBlock(b2)[i]] = getVarsForBlock(b1)[i];
   }

   for( int i = 0; i < getNConssForBlock(b1); ++i )
   {
      int cons1id;
      int cons2id;
      SCIP_CONS* cons1;
      SCIP_CONS* cons2;
      SCIP_Real* vals1;
      SCIP_Real* vals2;
      int nvals1;
      int nvals2;

      cons1id = getConssForBlock(b1)[i];
      cons2id = getConssForBlock(b2)[i];

      cons1 = detprobdata->getCons(cons1id);
      cons2 = detprobdata->getCons(cons2id);

      vals1 = NULL;
      vals2 = NULL;

      if( detprobdata->getNVarsForCons(cons1id) != detprobdata->getNVarsForCons(cons2id) )
      {
         SCIPdebugMessage("--> nvars differs for cons %s and cons %s!\n", SCIPconsGetName(cons1), SCIPconsGetName(cons2));
         return;
      }

      if( !SCIPisEQ(scip, GCGconsGetLhs(scip, cons1), GCGconsGetLhs(scip, cons2) ) )
      {
         SCIPdebugMessage("--> lhs differs for cons %s and cons %s!\n", SCIPconsGetName(cons1), SCIPconsGetName(cons2));
         return;
      }

      if( !SCIPisEQ(scip, GCGconsGetRhs(scip, cons1), GCGconsGetRhs(scip, cons2) ) )
      {
         SCIPdebugMessage("--> rhs differs for cons %s and cons %s!\n", SCIPconsGetName(cons1), SCIPconsGetName(cons2));
         return;
      }

      nvals1 = GCGconsGetNVars(scip, cons1);
      nvals2 = GCGconsGetNVars(scip, cons2);
      SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &vals1, nvals1) );
      SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &vals2, nvals2) );
      GCGconsGetVals(scip, cons1, vals1, nvals1);
      GCGconsGetVals(scip, cons2, vals2, nvals2);

      if( !realArraysAreEqual(scip, vals1, nvals1, vals2, nvals2) )
       {
          SCIPdebugMessage("--> coefs differ for cons %s and cons %s!\n", SCIPconsGetName(cons1), SCIPconsGetName(cons2));
          SCIPfreeBufferArray(scip, &vals1);
           SCIPfreeBufferArray(scip, &vals2);
          return;
       }

      for( int v = 0; v < detprobdata->getNVarsForCons(cons1id) ; ++v )
      {
         if( varmap[detprobdata->getVarsForCons(cons2id)[v]] != detprobdata->getVarsForCons(cons1id)[v])
         {
            SCIPfreeBufferArray(scip, &vals1);
             SCIPfreeBufferArray(scip, &vals2);
            SCIPdebugMessage("--> vars differ for cons %s and cons %s!\n", SCIPconsGetName(cons1), SCIPconsGetName(cons2));
            return;
         }
      }

      SCIPfreeBufferArray(scip, &vals1);
      SCIPfreeBufferArray(scip, &vals2);
   }

   varmap = std::vector<int>(getNVarsForBlock(b1), -1);
   for( int i = 0; i < getNVarsForBlock(b1); ++i )
      varmap[i] = i;

   *identical = TRUE;
   return;
}

void PARTIALDECOMP::calcNCoeffsForBlockForMastercons()
{
   ncoeffsforblockformastercons = std::vector<std::vector<int>>(getNBlocks());
   DETPROBDATA* detprobdata = this->getDetprobdata();

   for( int b = 0; b < getNBlocks(); ++b )
      ncoeffsforblockformastercons[b] = std::vector<int>(getNMasterconss(), 0);

   for( int mc = 0; mc < getNMasterconss(); ++mc )
   {
      int cons = getMasterconss()[mc];
      for ( int vmc = 0; vmc < detprobdata->getNVarsForCons(cons); ++vmc )
      {
         int var = detprobdata->getVarsForCons(cons)[vmc];
         for( int b = 0; b < getNBlocks(); ++b )
         {
            if( isVarBlockvarOfBlock(var, b) )
               ++ncoeffsforblockformastercons[b][mc];
         }
      }
   }
   return;
}


void PARTIALDECOMP::checkIdenticalBlocksTrivial(
   int                  b1,
   int                  b2,
   SCIP_Bool*           notidentical)
{
   if( getNConssForBlock(b1) != getNConssForBlock(b2) )
   {
      SCIPdebugMessage("--> number of constraints differs!\n");
      *notidentical = TRUE;
   }
   else if( getNVarsForBlock(b1) != getNVarsForBlock(b2) )
   {
      SCIPdebugMessage("--> number of variables differs!\n");
      *notidentical = TRUE;
   }
   else if( getNCoeffsForBlock(b1) != getNCoeffsForBlock( b2) )
   {
      SCIPdebugMessage("--> number of nonzero coeffs differs!\n");
      *notidentical = TRUE;
   }
   else
   {
      if( ncoeffsforblockformastercons.size() == 0 )
         calcNCoeffsForBlockForMastercons();

      for( int mc = 0; mc < getNMasterconss(); ++mc )
      {
         if ( ncoeffsforblockformastercons[b1][mc] != ncoeffsforblockformastercons[b2][mc] )
         {
            SCIPdebugMessage("--> number of nonzero coeffs in %d-th master cons differs!\n", mc);
            *notidentical = TRUE;
            break;
         }
      }
   }
}


void PARTIALDECOMP::complete()
{
   size_t nopenconss = openconss.size();
   size_t nopenvars = openvars.size();
   size_t i = 0;

   refineToBlocks();

   // assign the open components to the master (without removing them to avoid screwing with the vector indices)
   for(auto cons : openconss)
      setConsToMaster(cons);
   // remove them from the open lists
   openconss.clear();
   for(i = 0; i < nopenconss; i++)
   {
      isconsopen[i] = false;
   }

   for(auto var: openvars)
      setVarToMaster(var);

   openvars.clear();
   for(i = 0; i < nopenvars; i++)
   {
      isvaropen[i] = false;
   }

   // consider implicits, calc hash, and check whether the partialdec is still consistent
   prepare();
}


void PARTIALDECOMP::completeByConnected()
{
   /* tools to update openvars */
   std::vector<int> openvarsToDelete;
   std::vector<int> oldOpenconss;

   std::vector<bool> isConsVisited( nconss, false );
   std::vector<bool> isVarVisited( nvars, false );

   std::queue<int> helpqueue;
   std::vector<int> neighborConss;
   std::vector<int> neighborVars;

   assert( (int) conssforblocks.size() == getNBlocks() );
   assert( (int) varsforblocks.size() == getNBlocks() );
   assert( (int) stairlinkingvars.size() == getNBlocks() );

   refineToMaster();

   if( getNBlocks() < 0 )
   {
      setNBlocks(0);
   }

   gcg::DETPROBDATA* detprobdata = getDetprobdata();

   /* do breadth first search to find connected conss and vars */
   while( !openconss.empty() )
   {
      int newBlockNr;

      assert( helpqueue.empty() );
      helpqueue.push(openconss[0]);
      neighborConss.clear();
      neighborConss.push_back(openconss[0]);
      isConsVisited[openconss[0]] = true;
      neighborVars.clear();

      while( !helpqueue.empty() )
      {
         int nodeCons = helpqueue.front();
         assert( isConsOpencons(nodeCons) );
         helpqueue.pop();
         for( int var : detprobdata->getVarsForCons(nodeCons) )
         {
            assert( isVarOpenvar(var) || isVarLinkingvar(var) );

            if( isVarVisited[var] || isVarLinkingvar(var) )
               continue;

            for( int cons : detprobdata->getConssForVar(var) )
            {
               if( !isConsOpencons(cons) || isConsVisited[cons] )
               {
                  continue;
               }
               assert( isConsOpencons(cons) );
               isConsVisited[cons] = true;
               neighborConss.push_back(cons);
               helpqueue.push(cons);
            }
            isVarVisited[var] = true;
            neighborVars.push_back(var);
         }
      }

      /* assign found conss and vars to a new block */
      newBlockNr = getNBlocks() + 1;
      setNBlocks(newBlockNr);
      for( int cons : neighborConss )
      {
         setConsToBlock(cons, newBlockNr - 1);
         if( isConsOpencons(cons) )
            deleteOpencons(cons);
      }
      for( int var : neighborVars )
      {
         setVarToBlock(var, newBlockNr - 1);
         if( isVarOpenvar(var) )
            deleteOpenvar(var);
      }
   }

   /* assign left open vars to block 0, if it exists, and to master, otherwise */
   for( int var : openvars )
   {
      if( getNBlocks() != 0 )
         setVarToBlock(var, 0);
      else
         setVarToMaster(var);
      openvarsToDelete.push_back(var);
   }

   for( int var : openvarsToDelete )
   {
      if( isVarOpenvar(var) )
         deleteOpenvar(var);
   }

   assert( getNOpenconss() == 0 );
   assert( getNOpenvars() == 0 );

   prepare();

   assert( checkConsistency() );
}


/**
* @brief assigns all open constraints and open variables
*
*  strategy: assigns all conss and vars to the same block if they are connected
*  a cons and a var are adjacent if the var appears in the cons
*  \note this relies on the consadjacency structure of the detprobdata
*  hence it cannot be applied in presence of linking variables
*/
void PARTIALDECOMP::completeByConnectedConssAdjacency()
{
   /* tools to check if the openvars can still be found in a constraint yet */
   std::vector<int> varinblocks(nvars, -1); /* stores in which block the variable can be found */

   /* tools to update openvars */
   std::vector<int> oldOpenconss;
   std::vector<int> openvarsToDelete;

   // note: this should not happen
   if( getNLinkingvars() != 0 )
      completeByConnected();

   std::vector<bool> isConsVisited(nconss, false);

   std::queue<int> helpqueue;
   std::vector<int> neighborConss;

   assert( (int) conssforblocks.size() == getNBlocks() );
   assert( (int) varsforblocks.size() == getNBlocks() );
   assert( (int) stairlinkingvars.size() == getNBlocks() );

   refineToMaster();

   assert( checkConsistency() );
   gcg::DETPROBDATA* detprobdata = getDetprobdata();

   if( getNBlocks() < 0 )
   {
      setNBlocks(0);
   }

   /* do breadth first search to find connected conss */
   while( !openconss.empty() )
   {
      int newBlockNr;

      assert( helpqueue.empty() );
      helpqueue.push(openconss[0]);
      neighborConss.clear();
      neighborConss.push_back(openconss[0]);
      isConsVisited[openconss[0]] = true;

      while( !helpqueue.empty() )
      {
         int nodeCons = helpqueue.front();
         assert( isConsOpencons(nodeCons) );
         helpqueue.pop();
         for( int cons : detprobdata->getConssForCons(nodeCons) )
         {
            if( isConsVisited[cons] || isConsMastercons(cons) || !isConsOpencons(cons) )
               continue;

            assert( isConsOpencons(cons) );
            isConsVisited[cons] = true;
            neighborConss.push_back(cons);
            helpqueue.push(cons);
         }
      }

      /* assign found conss and vars to a new block */
      newBlockNr = getNBlocks() + 1;
      setNBlocks( newBlockNr );
      for( int cons : neighborConss )
      {
         setConsToBlock(cons, newBlockNr - 1);
         if(isConsOpencons(cons))
            deleteOpencons(cons);

         for( int var : detprobdata->getVarsForCons(cons) )
         {

            if( isVarLinkingvar(var) || varinblocks[var] != -1 )
               continue;

            assert( !isVarMastervar(var) );
            setVarToBlock(var, newBlockNr - 1);
            varinblocks[var] = newBlockNr - 1;
            if( isVarOpenvar(var) )
               deleteOpenvar(var);
         }
      }
   }

   /* assign left open vars to block 0, if it exists, and to master, otherwise */
   for( int var : openvars )
   {
      if( getNBlocks() != 0 )
         setVarToBlock(var, 0);
      else
         setVarToMaster(var);
      openvarsToDelete.push_back(var);
   }

   for( int var : openvarsToDelete )
   {
      if( isVarOpenvar(var) )
         deleteOpenvar(var);
   }

   assert( getNOpenconss() == 0 );
   assert( getNOpenvars() == 0 );

   prepare();

   assert( checkConsistency() );
}


void PARTIALDECOMP::completeGreedily(
   )
{
   bool checkvar;
   bool isvarinblock;
   bool notassigned;
   DETPROBDATA* detprobdata = getDetprobdata();

   /* tools to check if the openvars can still be found in a constraint yet*/
   std::vector<int> varinblocks; /* stores in which block the variable can be found */

   if( getNBlocks() == 0 && getNOpenconss() > 0 )
   {
      int block = addBlock();
      fixConsToBlock( openconss[0], block );
   }

   std::vector<int> del;

   /* check if the openvars can already be found in a constraint */
   for( int i = 0; i < getNOpenvars(); ++ i )
   {
      varinblocks.clear();

      /* test if the variable can be found in blocks */
      for( int b = 0; b < getNBlocks(); ++ b )
      {
         isvarinblock = false;
         std::vector<int>& conssforblock = getConssForBlock(b);
         for( int k = 0; k < getNConssForBlock(b) && !isvarinblock; ++ k )
         {
            for( int l = 0; l < detprobdata->getNVarsForCons( conssforblock[k] ); ++ l )
            {
               if( openvars[i] == detprobdata->getVarsForCons( conssforblock[k] )[l] )
               {
                  varinblocks.push_back( b );
                  isvarinblock = true;
                  break;
               }
            }
         }
      }
      if( varinblocks.size() == 1 ) /* if the variable can be found in one block set the variable to a variable of the block*/
      {
         setVarToBlock(openvars[i], varinblocks[0]);
         del.push_back(openvars[i]);
         continue; /* the variable doesn't need to be checked any more */
      }
      else if( varinblocks.size() == 2 ) /* if the variable can be found in two blocks check if it is a linking var or a stairlinking var*/
      {
         if( varinblocks[0] + 1 == varinblocks[1] )
         {
            setVarToStairlinking(openvars[i], varinblocks[0], varinblocks[1]);
            del.push_back(openvars[i]);
            continue; /* the variable doesn't need to be checked any more */
         }
         else
         {
            setVarToLinking(openvars[i]);
            del.push_back(openvars[i]);
            continue; /* the variable doesn't need to be checked any more */
         }
      }
      else if( varinblocks.size() > 2 ) /* if the variable can be found in more than two blocks it is a linking var */
      {
         setVarToLinking(openvars[i]);
         del.push_back(openvars[i]);
         continue; /* the variable doesn't need to be checked any more */
      }

      checkvar = true;

      /* if the variable can be found in an open constraint it is still an open var */
      for( int j = 0; j < getNOpenconss(); ++ j )
      {
         checkvar = true;
         for( int k = 0; k < detprobdata->getNVarsForCons( j ); ++ k )
         {
            if( openvars[i] == detprobdata->getVarsForCons( j )[k] )
            {
               checkvar = false;
               break;
            }
         }
         if( ! checkvar )
         {
            break;
         }
      }

      /* test if the variable can be found in a master constraint yet */
      for( int k = 0; k < detprobdata->getNConssForVar( openvars[i] ) && checkvar; ++ k )
      {
         if( isConsMastercons(detprobdata->getConssForVar(openvars[i])[k]) )
         {
            setVarToMaster(openvars[i]);
            del.push_back(openvars[i]);
            checkvar = false; /* the variable does'nt need to be checked any more */
            break;
         }
      }
   }

   /* remove assigned vars from list of open vars */
   for(auto v : del)
      deleteOpenvar(v);

   del.clear();
   sort();

   std::vector<int> delconss;

   /* assign open conss greedily */
   for( int i = 0; i < getNOpenconss(); ++ i )
   {
      std::vector<int> vecOpenvarsOfBlock; /* stores the open vars of the blocks */
      bool consGotBlockcons = false; /* if the constraint can be assigned to a block */

      /* check if the constraint can be assigned to a block */
      for( int j = 0; j < getNBlocks(); ++ j )
      {
         /* check if all vars of the constraint are a block var of the current block, an open var, a linkingvar or a mastervar*/
         consGotBlockcons = true;
         for( int k = 0; k < detprobdata->getNVarsForCons( openconss[i] ); ++ k )
         {
            if( isVarBlockvarOfBlock( detprobdata->getVarsForCons( openconss[i] )[k], j )
                || isVarOpenvar( detprobdata->getVarsForCons( openconss[i] )[k] )
                || isVarLinkingvar( detprobdata->getVarsForCons( openconss[i] )[k] )
                || isVarStairlinkingvarOfBlock( detprobdata->getVarsForCons( openconss[i] )[k], j )
                || ( j != 0 && isVarStairlinkingvarOfBlock( detprobdata->getVarsForCons( openconss[i] )[k], j - 1 ) ) )
            {
               if( isVarOpenvar( detprobdata->getVarsForCons( openconss[i] )[k] ) )
               {
                  vecOpenvarsOfBlock.push_back( detprobdata->getVarsForCons( openconss[i] )[k] );
               }
            }
            else
            {
               vecOpenvarsOfBlock.clear(); /* the open vars don't get vars of the block */
               consGotBlockcons = false; /* the constraint can't be constraint of the block, check the next block */
               break;
            }
         }
         if( consGotBlockcons ) /* the constraint can be assigned to the current block */
         {
            setConsToBlock( openconss[i], j );
            delconss.push_back(openconss[i]);
            for( size_t k = 0; k < vecOpenvarsOfBlock.size(); ++ k ) /* the openvars in the constraint get block vars */
            {
               setVarToBlock( vecOpenvarsOfBlock[k], j );
               deleteOpenvar( vecOpenvarsOfBlock[k] );
            }
            vecOpenvarsOfBlock.clear();

            break;
         }
      }

      if( !consGotBlockcons ) /* the constraint can not be assigned to a block, set it to master */
      {
         setConsToMaster( openconss[i] );
         delconss.push_back(openconss[i]);
      }
   }

   /* remove assigned conss from list of open conss */
   for(auto c : delconss)
      deleteOpencons(c);

   sort();

   /* assign open vars greedily */
   for( int i = 0; i < getNOpenvars(); ++ i )
   {
      notassigned = true;
      for( int j = 0; j < getNMasterconss() && notassigned; ++ j )
      {
         for( int k = 0; k < detprobdata->getNVarsForCons(masterconss[j]); ++ k )
         {
            if( openvars[i] == detprobdata->getVarsForCons(masterconss[j])[k] )
            {
               setVarToMaster(openvars[i]);
               del.push_back(openvars[i]);
               notassigned = false;
               break;
            }
         }
      }
   }

   /* remove assigned vars from list of open vars */
   for(auto v : del)
      deleteOpenvar(v);

   sort();

   /* check if the open conss are all assigned */
   assert( checkAllConssAssigned() );

   /* check if the open vars are all assigned */
   assert( getNOpenvars() == 0 );

   assert( checkConsistency() );
}


void PARTIALDECOMP::removeMastercons(
   int consid
   )
{
   std::vector<int>::iterator todelete = lower_bound( masterconss.begin(), masterconss.end(), consid );
   masterconss.erase(todelete);
}


bool PARTIALDECOMP::consPartitionUsed(
   int detectorchainindex
   )
{
   assert( 0 <= detectorchainindex && detectorchainindex < (int) usedpartition.size() );

   return (usedpartition[detectorchainindex] != NULL )
      && (dynamic_cast<ConsPartition*>( usedpartition[detectorchainindex] ) != NULL );
}


void PARTIALDECOMP::considerImplicits(
   )
{
   int cons;
   int var;
   std::vector<int> blocksOfBlockvars; /* blocks with blockvars which can be found in the cons */
   std::vector<int> blocksOfOpenvar; /* blocks in which the open var can be found */
   bool master;
   bool hitsOpenVar;
   bool hitsOpenCons;
   SCIP_Bool benders;
   std::vector<int> del;
   std::vector<int> delconss;

   DETPROBDATA* detprobdata = this->getDetprobdata();

   SCIPgetBoolParam(scip, "detection/benders/enabled", &benders);

   sort();

   /* set openconss with more than two blockvars to master */
   for( size_t c = 0; c < openconss.size(); ++ c )
   {
      std::vector<bool> hitsblock = std::vector<bool>(nblocks, false);
      blocksOfBlockvars.clear();
      master = false;
      hitsOpenVar = false;
      cons = openconss[c];

      for( int v = 0; v < detprobdata->getNVarsForCons( cons ) && ! master; ++ v )
      {
         var = detprobdata->getVarsForCons( cons )[v];

         if( isVarMastervar( var ) )
         {
            master = true;
            fixConsToMaster( cons );
            continue;
         }

         if( isVarOpenvar( var ) )
         {
            hitsOpenVar = true;
            if( !benders )
               continue;
         }

         for( int b = 0; b < nblocks && ! master; ++ b )
         {
            if( isVarBlockvarOfBlock( var, b ) && !hitsblock[b] )
            {
               hitsblock[b] = true;
               blocksOfBlockvars.push_back( b );
               break;
            }
         }
      }

      if ( benders && blocksOfBlockvars.size() == 1 && !master )
      {
         setConsToBlock( cons, blocksOfBlockvars[0] );
         delconss.push_back(cons);
      }

      if( !benders && blocksOfBlockvars.size() > 1 )
      {
         setConsToMaster( cons );
         delconss.push_back(cons);
      }

      /* also assign open constraints that only have vars assigned to one single block and no open vars*/
      if( blocksOfBlockvars.size() == 1 && ! hitsOpenVar && ! master && ! benders )
      {
         setConsToBlock( cons, blocksOfBlockvars[0] );
         delconss.push_back(cons);
      }
   }

   /* remove assigned conss from list of open conss */
   for(auto c : delconss)
      deleteOpencons(c);

   sort();

   /* set open var to linking, if it can be found in more than one block 
    * or set it to a block if it has only constraints in that block and no open constraints 
    * or set it to master if it only hits master constraints */
   for( size_t i = 0; i < openvars.size(); ++ i )
   {
      std::vector<bool> hitsblock = std::vector<bool>(nblocks, false);
      bool hitsmasterconss = false;
      bool hitonlymasterconss = true;
      bool hitonlyblockconss = true;
      blocksOfOpenvar.clear();
      var = openvars[i];
      hitsOpenCons = false;


      for( int c = 0; c < detprobdata->getNConssForVar( var ); ++ c )
      {
         cons = detprobdata->getConssForVar( var )[c];
         if ( isConsMastercons(cons) )
         {
            hitsmasterconss = true;
            hitonlyblockconss = false;
            continue;
         }

         if( isConsOpencons( cons ) )
         {
            hitsOpenCons = true;
            hitonlyblockconss = false;
            hitonlymasterconss = false;
            continue;
         }
      }
      for( int b = 0; b < nblocks; ++ b )
      {
         for( int c = 0; c < detprobdata->getNConssForVar( var ); ++ c )
         {
            cons = detprobdata->getConssForVar( var )[c];
            if( isConsBlockconsOfBlock( cons, b ) && !hitsblock[b] )
            {
               hitsblock[b] = true;
               hitonlymasterconss = false;
               blocksOfOpenvar.push_back( b );
               break;
            }
         }
      }

      if( blocksOfOpenvar.size() > 1 )
      {
         setVarToLinking( var );
         del.push_back(var);
         continue;
      }

      if( benders && blocksOfOpenvar.size() == 1 &&  hitsmasterconss )
      {
         setVarToLinking( var );
         del.push_back(var);
      }

      if( benders && hitonlyblockconss && blocksOfOpenvar.size() > 0 )
      {
         setVarToBlock( var, blocksOfOpenvar[0] );
         del.push_back(var);
      }

      if( benders && hitonlymasterconss)
      {
         setVarToMaster( var);
         del.push_back(var);
      }

      if( !benders && blocksOfOpenvar.size() == 1 && ! hitsOpenCons )
      {
         setVarToBlock( var, blocksOfOpenvar[0] );
         del.push_back(var);
      }

      if( !benders && blocksOfOpenvar.size() == 0 && ! hitsOpenCons )
      {
         setVarToMaster( var );
         del.push_back(var);
      }
   }

   /* remove assigned vars from list of open vars */
   for(auto v : del)
      deleteOpenvar(v);
   sort();
}


void PARTIALDECOMP::copyPartitionStatistics(
   const PARTIALDECOMP* otherpartialdec
   )
{
   usedpartition = otherpartialdec->usedpartition;
   classestomaster = otherpartialdec->classestomaster;
   classestolinking = otherpartialdec->classestolinking;
}


void PARTIALDECOMP::deleteEmptyBlocks(
   bool variables /* if TRUE a block is only considered to be empty if it contains neither constraints or variables */
   )
{
   bool emptyBlocks = true;
   SCIP_Bool benders = FALSE;
   int block = - 1;
   int b;

   assert( (int) conssforblocks.size() == nblocks );
   assert( (int) varsforblocks.size() == nblocks );
   assert( (int) stairlinkingvars.size() == nblocks );

   SCIPgetBoolParam(scip, "detection/benders/enabled", &benders);

   while( emptyBlocks )
   {
      emptyBlocks = false;
      for( b = nblocks - 1; b >= 0; --b )
      {
         if( conssforblocks[b].empty() &&  ( variables ? varsforblocks[b].empty() : true) )
         {
            emptyBlocks = true;
            block = b;
            break;
         }
         if( benders && ( conssforblocks[b].empty() || varsforblocks[b].empty()) )
         {
            emptyBlocks = true;
            block = b;
            break;
         }

      }
      if( emptyBlocks )
      {
         nblocks--;

         stairlinkingvars.erase(stairlinkingvars.begin() + block);

         for( int j : conssforblocks[block] )
         {
            masterconss.push_back(j);
            isconsmaster[j] = true;
         }
         std::sort(masterconss.begin(), masterconss.end());
         conssforblocks.erase(conssforblocks.begin() + block);

         for( int j : varsforblocks[block] )
         {
            mastervars.push_back(j);
            isvarmaster[j] = true;
         }
         varsforblocks.erase(varsforblocks.begin() + block);
         std::sort( mastervars.begin(), mastervars.end() );

         //set stairlinkingvars of the previous block to block vars
         if( block != 0 && !stairlinkingvars[block - 1].empty() )
         {
            for( int j : stairlinkingvars[block - 1] )
            {
               fixVarToBlock(j, block - 1);
            }
            stairlinkingvars[block - 1].clear();
            sort();
         }
      }
   }
}


void PARTIALDECOMP::deleteOpencons(
   int opencons
   )
{
   assert( opencons >= 0 && opencons < nconss );
   std::vector<int>::iterator it;
   it = lower_bound( openconss.begin(), openconss.end(), opencons );
   assert( it != openconss.end() && *it == opencons );
   openconss.erase(it);
   isconsopen[opencons] = false;
}


std::vector<int>::const_iterator PARTIALDECOMP::deleteOpencons(
   std::vector<int>::const_iterator itr
   )
{
   isconsopen[*itr] = false;
   return openconss.erase(itr);
}


void PARTIALDECOMP::deleteOpenvar(
   int openvar
   )
{
   assert( openvar >= 0 && openvar < nvars );
   std::vector<int>::iterator it;
   it = lower_bound( openvars.begin(), openvars.end(), openvar );
   assert( it != openvars.end() && *it == openvar );
   openvars.erase( it );
   isvaropen[openvar] = false;
}


std::vector<int>::const_iterator PARTIALDECOMP::deleteOpenvar(
   std::vector<int>::const_iterator itr
   )
{
   assert( itr != openvars.cend() );

   isvaropen[*itr] = false;
   return openvars.erase(itr);
}


void PARTIALDECOMP::displayAggregationInformation()
{
   if( !aggInfoCalculated() )
   {
      SCIPinfoMessage(scip, NULL, " Aggregation information is not calculated yet \n ");
   }
   else
   {
      SCIPinfoMessage(scip, NULL, " number of representative blocks: %d \n", nrepblocks);
      for( int i = 0; i < nrepblocks; ++i )
      {
         SCIPinfoMessage(scip, NULL, "representative block %d : ", i);

         for( size_t b = 0; b < reptoblocks[i].size(); ++b )
            SCIPinfoMessage(scip, NULL, "%d ", reptoblocks[i][b] );

         SCIPinfoMessage(scip, NULL, "\n");
      }
   }
}


void PARTIALDECOMP::displayInfo(
   int detailLevel
   )
{
   assert( 0 <= detailLevel );
   DETPROBDATA* detprobdata = this->getDetprobdata();

   std::cout << std::endl;

   /* general information */
   std::cout << "-- General information --" << std::endl;
   std::cout << " ID: " << id << std::endl;
   std::cout << " Hashvalue: " << hashvalue << std::endl;
   std::cout << " Score: " << classicscore << std::endl;
   if( getNOpenconss() + getNOpenconss() > 0 )
      std::cout << " Maxwhitescore >= " << maxwhitescore << std::endl;
   else
      std::cout << " Max white score: " << maxwhitescore << std::endl;
   if( getNOpenconss() + getNOpenconss() == 0 )
         std::cout << " Max-foreseeing-white-score: " << maxforeseeingwhitescore << std::endl;
   if( getNOpenconss() + getNOpenconss() == 0 )
         std::cout << " Max-foreseeing-white-aggregated-score: " << maxforeseeingwhitescoreagg << std::endl;
   if( getNOpenconss() + getNOpenconss() == 0 )
         std::cout << " PPC-max-foreseeing-white-score: " <<  setpartfwhitescore << std::endl;

   if( getNOpenconss() + getNOpenconss() == 0 )
          std::cout << " PPC-max-foreseeing-white-aggregated-score: " <<  setpartfwhitescoreagg << std::endl;

   if( getNOpenconss() + getNOpenconss() == 0 )
          std::cout << " Bendersscore: " << bendersscore << std::endl;

   if( getNOpenconss() + getNOpenconss() == 0 )
          std::cout << " borderareascore: " << borderareascore << std::endl;

   std::cout << " HassetppMaster: " << hasSetppMaster() << std::endl;
   std::cout << " HassetppcMaster: " << hasSetppcMaster() << std::endl;
   std::cout << " HassetppccardMaster: " << hasSetppccardMaster() << std::endl;
   std::cout << " PARTIALDECOMP is for the " << (original ? "original" : "presolved" ) << " problem and "
             << ( usergiven ? "usergiven" : "not usergiven" ) << "." << std::endl;
   std::cout << " Number of constraints: " << getNConss() << std::endl;
   std::cout << " Number of variables: " << getNVars() << std::endl;

   displayAggregationInformation();
   std::cout << std::endl;

   /* detection information */
   std::cout << "-- Detection and detectors --" << std::endl;
   std::cout << " PARTIALDECOMP stems from the " << ( stemsfromorig ? "original" : "presolved" ) << " problem." << std::endl;

   /* ancestor partialdecs' ids */
   std::cout << " IDs of ancestor partialdecs: ";
   if( !listofancestorids.empty() )
      std::cout << listofancestorids[0];
   for( int i = 1; i < (int) listofancestorids.size(); ++i )
      std::cout << ", " << listofancestorids[i];
   std::cout << std::endl;

   /* detector chain information */
   std::cout << " " << getNDetectors() << " detector" << ( getNDetectors() > 1 ? "s" : "" ) << " worked on this partialdec:";
   if( getNDetectors() != 0 )
   {
      std::string detectorrepres;

      if( detectorchain[0] == NULL )
         detectorrepres = "user";
      else
      {
         /* potentially add finisher label */
         detectorrepres = (
            getNDetectors() != 1 || !isfinishedbyfinisher ? DECdetectorGetName(detectorchain[0]) :
               "(finish) " + std::string(DECdetectorGetName(detectorchain[0])));
      }

      if( detailLevel > 0 )
      {
         std::cout << std::endl << " 1.: " << detectorrepres << std::endl;
         std::cout << getDetectorStatistics( 0 );
         std::cout << getDetectorPartitionInfo(0, detailLevel > 1 && (!stemsfromorig || original));
      }
      else
      {
         std::cout << " " << detectorrepres;
      }

      for( int d = 1; d < getNDetectors(); ++d )
      {
         /* potentially add finisher label */
         detectorrepres = (
            getNDetectors() != d + 1 || !isfinishedbyfinisher ? DECdetectorGetName(detectorchain[d]) :
               "(finish) " + std::string(DECdetectorGetName(detectorchain[d])));


         if( detailLevel > 0 )
         {
            std::cout << " " << ( d + 1 ) << ".: " << detectorrepres << std::endl;
            std::cout << getDetectorStatistics( d );
            std::cout << getDetectorPartitionInfo(d, detailLevel > 1 && (!stemsfromorig || original));
         }
         else
         {
            std::cout << ", " << detectorrepres;
         }
      }

      if( detailLevel <= 0 )
      {
         std::cout << std::endl;
      }
   }

   std::cout << std::endl;

   /* variable information */
   std::cout << "-- Border and unassigned --" << std::endl;
   std::cout << " Linkingvariables";
   if( detailLevel > 1 )
   {
      std::cout << " (" << getNLinkingvars() << ")";
      if( getNLinkingvars() > 0 )
         std::cout << ":  " << SCIPvarGetName(detprobdata->getVar(getLinkingvars()[0]));
      for( int v = 1; v < getNLinkingvars(); ++v )
      {
         std::cout << ", " << SCIPvarGetName(detprobdata->getVar(getLinkingvars()[v]));
      }
      std::cout << std::endl;
   }
   else
   {
      std::cout << ": " << getNLinkingvars() << std::endl;
   }
   std::cout << " Masterconstraints";
   if( detailLevel > 1 )
   {
      std::cout << " (" << getNMasterconss() << ")";
      if( getNMasterconss() > 0 )
         std::cout << ":  " << SCIPconsGetName(detprobdata->getCons(getMasterconss()[0]));
      for( int c = 1; c < getNMasterconss(); ++c )
      {
         std::cout << ", " << SCIPconsGetName(detprobdata->getCons(getMasterconss()[c]));
      }
      std::cout << std::endl;
   }
   else
   {
      std::cout << ": " << getNMasterconss() << std::endl;
   }
   std::cout << " Mastervariables";
   if( detailLevel > 1 )
   {
      std::cout << " (" << getNMastervars() << ")";
      if( getNMastervars() > 0 )
         std::cout << ":  " << SCIPvarGetName(detprobdata->getVar(getMastervars()[0]));
      for( int v = 1; v < getNMastervars(); ++v )
      {
         std::cout << ", " << SCIPvarGetName(detprobdata->getVar(getMastervars()[v]));
      }
      std::cout << std::endl;
   }
   else
   {
      std::cout << ": " << getNMastervars() << std::endl;
   }
   std::cout << " Open constraints";
   if( detailLevel > 1 )
   {
      std::cout << " (" << getNOpenconss() << ")";
      if( getNOpenconss() > 0 )
         std::cout << ":  " << SCIPconsGetName(detprobdata->getCons(getOpenconss()[0]));
      for( int c = 1; c < getNOpenconss(); ++c )
      {
         std::cout << ", " << SCIPconsGetName(detprobdata->getCons(getOpenconss()[c]));
      }
      std::cout << std::endl;
   }
   else
   {
      std::cout << ": " << getNOpenconss() << std::endl;
   }
   std::cout << " Open variables";
   if( detailLevel > 1 )
   {
      std::cout << " (" << getNOpenvars() << ")";
      if( getNOpenvars() > 0 )
         std::cout << ":  " << SCIPvarGetName(detprobdata->getVar(getOpenvars()[0]));
      for( int v = 1; v < getNOpenvars(); ++v )
      {
         std::cout << ", " << SCIPvarGetName(detprobdata->getVar(getOpenvars()[v]));
      }
      std::cout << std::endl;
   }
   else
   {
      std::cout << ": " << getNOpenvars() << std::endl;
   }

   std::cout << std::endl;

   /* block information */
   std::cout << "-- Blocks --" << std::endl;
   std::cout << " Number of blocks: " << nblocks << std::endl;

   if( detailLevel > 0 )
   {
      for( int b = 0; b < nblocks; ++b )
      {
         std::cout << " Block " << b << ":" << std::endl;

         std::cout << "  Constraints";
         if( detailLevel > 1 )
         {
            std::cout << " (" << getNConssForBlock(b) << ")";
            if( getNConssForBlock(b) > 0 )
               std::cout << ":  " << SCIPconsGetName(detprobdata->getCons(getConssForBlock(b)[0]));
            for( int c = 1; c < getNConssForBlock(b); ++c )
            {
               std::cout << ", " << SCIPconsGetName(detprobdata->getCons(getConssForBlock(b)[c]));
            }
            std::cout << std::endl;
         }
         else
         {
            std::cout << ": " << getNConssForBlock(b) << std::endl;
         }

         std::cout << "  Variables";
         if( detailLevel > 1 )
         {
            std::cout << " (" << getNVarsForBlock(b) << ")";
            if( getNVarsForBlock(b) > 0 )
               std::cout << ":  " << SCIPvarGetName(detprobdata->getVar(getVarsForBlock(b)[0]));
            for( int v = 1; v < getNVarsForBlock(b); ++v )
            {
               std::cout << ", " << SCIPvarGetName(detprobdata->getVar(getVarsForBlock(b)[v]));
            }
            std::cout << std::endl;
         }
         else
         {
            std::cout << ": " << getNVarsForBlock(b) << std::endl;
         }

         std::cout << "  Stairlinkingvariables";
         if( detailLevel > 1 )
         {
            std::cout << " (" << getNStairlinkingvars(b) << ")";
            if( getNStairlinkingvars(b) > 0 )
               std::cout << ":  " << SCIPvarGetName(detprobdata->getVar(getStairlinkingvars(b)[0]));
            for( int v = 1; v < getNStairlinkingvars(b); ++v )
            {
               std::cout << ", " << SCIPvarGetName(detprobdata->getVar(getStairlinkingvars(b)[v]));
            }
            std::cout << std::endl;
         }
         else
         {
            std::cout << ": " << getNStairlinkingvars(b) << std::endl;
         }
      }
   }

   std::cout << std::endl;
}


SCIP_Bool PARTIALDECOMP::hasSetppccardMaster(
)
{
   SCIP_Bool hassetpartmaster;
   SCIP_Bool verbose;
   hassetpartmaster = TRUE;
   verbose = FALSE;

   if( getNTotalStairlinkingvars() + getNLinkingvars() > 0 )
      return FALSE;

   DETPROBDATA* detprobdata = this->getDetprobdata();

   for( int l = 0; l < getNMasterconss(); ++l )
   {
      int consid = getMasterconss()[l];
      if( !detprobdata->isConsSetppc(consid) && !detprobdata->isConsCardinalityCons(consid) )
      {
         hassetpartmaster = FALSE;
         if( verbose )
            std::cout << " cons with name  " << SCIPconsGetName(detprobdata->getCons(consid)) << " is no setppccard constraint." << std::endl;
         break;
      }
   }

   return hassetpartmaster;
}


SCIP_Bool PARTIALDECOMP::hasSetppcMaster(
)
{
   SCIP_Bool hassetpartmaster;
   hassetpartmaster = TRUE;

    if( getNTotalStairlinkingvars() + getNLinkingvars() > 0 )
      return FALSE;

   DETPROBDATA* detprobdata = this->getDetprobdata();

   for( int l = 0; l < getNMasterconss(); ++l )
   {
      int consid = getMasterconss()[l];
      if( !detprobdata->isConsSetppc(consid)  )
      {
         hassetpartmaster = FALSE;
         break;
      }
   }
   return hassetpartmaster;
}


SCIP_Bool PARTIALDECOMP::hasSetppMaster(
)
{
   SCIP_Bool hassetpartmaster;
   hassetpartmaster = TRUE;

   if( getNTotalStairlinkingvars() + getNLinkingvars() > 0 )
      return FALSE;
   
   DETPROBDATA* detprobdata = this->getDetprobdata();

   for( int l = 0; l < getNMasterconss(); ++l )
   {
      int consid = getMasterconss()[l];
      if( !detprobdata->isConsSetpp(consid)  )
      {
         hassetpartmaster = FALSE;
         break;
      }
   }
   return hassetpartmaster;
}


SCIP_RETCODE PARTIALDECOMP::filloutBorderFromConstoblock(
   SCIP_HASHMAP* constoblock,
   int givenNBlocks
   )
{
   assert( givenNBlocks >= 0 );
   assert( nblocks == 0 );
   assert( (int) conssforblocks.size() == nblocks );
   assert( (int) varsforblocks.size() == nblocks );
   assert( (int) stairlinkingvars.size() == nblocks );
   assert( ! alreadyAssignedConssToBlocks() );
   nblocks = givenNBlocks;
   DETPROBDATA* detprobdata = this->getDetprobdata();
   nvars = detprobdata->getNVars();
   nconss = detprobdata->getNConss();
   int consnum;
   int consblock;

   for( int i = 0; i < nconss; ++ i )
   {
      consnum = i;
      consblock = ( (int) (size_t) SCIPhashmapGetImage( constoblock, (void*) (size_t) i ) ) - 1;
      assert( consblock >= 0 && consblock <= nblocks );
      if( consblock == nblocks )
      {
         setConsToMaster( consnum );
         deleteOpencons( consnum );
      }
   }

   nblocks = 0;
   sort();

   assert( checkConsistency() );

   return SCIP_OKAY;
}


SCIP_RETCODE PARTIALDECOMP::filloutPartialdecFromConstoblock(
   SCIP_HASHMAP* constoblock,
   int givenNBlocks
   )
{
   assert( givenNBlocks >= 0 );
   assert( nblocks == 0 );
   assert( (int) conssforblocks.size() == nblocks );
   assert( (int) varsforblocks.size() == nblocks );
   assert( (int) stairlinkingvars.size() == nblocks );
   assert( ! alreadyAssignedConssToBlocks() );
   nblocks = givenNBlocks;
   DETPROBDATA* detprobdata = this->getDetprobdata();
   nvars = detprobdata->getNVars();
   nconss = detprobdata->getNConss();
   int consnum;
   int consblock;
   int varnum;
   bool varInBlock;
   std::vector<int> varinblocks = std::vector<int>( 0 );
   std::vector<int> emptyVector = std::vector<int>( 0 );

   for( int c = 0; c < nconss; ++ c )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL, "%d\n", c);
      assert( SCIPhashmapExists( constoblock, (void*) (size_t) c ) );
      assert( (int) (size_t) SCIPhashmapGetImage( constoblock, (void*) (size_t) c ) - 1 <= nblocks );
      assert( (int) (size_t) SCIPhashmapGetImage( constoblock, (void*) (size_t) c ) - 1 >= 0 );
   }

   for( int b = (int) conssforblocks.size(); b < nblocks; b ++ )
      conssforblocks.push_back( emptyVector );

   for( int b = (int) varsforblocks.size(); b < nblocks; b ++ )
      varsforblocks.push_back( emptyVector );

   for( int b = (int) stairlinkingvars.size(); b < nblocks; b ++ )
      stairlinkingvars.push_back( emptyVector );

   for( int i = 0; i < nconss; ++ i )
   {
      consnum = i;
      consblock = ( (int) (size_t) SCIPhashmapGetImage( constoblock, (void*) (size_t) i ) ) - 1;
      assert( consblock >= 0 && consblock <= nblocks );
      if( consblock == nblocks )
         setConsToMaster( consnum );
      else
         setConsToBlock( consnum, consblock );
   }

   for( int i = 0; i < nvars; ++ i )
   {
      varinblocks.clear();
      varnum = i;

      /* test if the variable can be found in blocks */
      for( int b = 0; b < nblocks; ++ b )
      {
         varInBlock = false;
         for( size_t k = 0; k < conssforblocks[b].size() && ! varInBlock; ++ k )
         {
            for( int l = 0; l < detprobdata->getNVarsForCons( conssforblocks[b][k] ) && ! varInBlock; ++ l )
            {
               if( varnum == ( detprobdata->getVarsForCons( conssforblocks[b][k] ) )[l] )
               {
                  varinblocks.push_back( b );
                  varInBlock = true;
               }
            }
         }
      }
      if( varinblocks.size() == 1 ) /* if the var can be found in one block set the var to block var */
         setVarToBlock( varnum, varinblocks[0] );
      else if( varinblocks.size() == 2 ) /* if the variable can be found in two blocks check if it is a linking var or a stairlinking var*/
      {
         if( varinblocks[0] + 1 == varinblocks[1] )
            setVarToStairlinking( varnum, varinblocks[0], varinblocks[1] );
         else
            setVarToLinking( varnum );
      }
      else if( varinblocks.size() > 2 ) /* if the variable can be found in more than two blocks it is a linking var */
         setVarToLinking( varnum );
      else
         assert( varinblocks.size() == 0 );
      setVarToMaster( varnum );
   }
   sort();
   openvars = std::vector<int>( 0 );
   openconss = std::vector<int>( 0 );
   isvaropen = std::vector<bool>(nvars, false);
   isconsopen = std::vector<bool>(nconss, false);

   deleteEmptyBlocks(false);
   sort();
   assert( checkConsistency( ) );

   return SCIP_OKAY;
}


void PARTIALDECOMP::findVarsLinkingToMaster(
   )
{
   int i;
   int j;
   bool isMasterVar;
   std::vector<int> foundMasterVarIndices;
   std::vector<int>& lvars = getLinkingvars();

   DETPROBDATA* detprobdata = this->getDetprobdata();

   // sort Master constraints for binary search
   sort();

   for( i = 0; i < getNLinkingvars(); ++ i )
   {
      isMasterVar = true;
      std::vector<int>& varcons = detprobdata->getConssForVar( lvars[i] );
      for( j = 0; j < detprobdata->getNConssForVar( lvars[i] ); ++ j )
      {
         if( ! isconsmaster[varcons[j]]  )
         {
            isMasterVar = false;
            break;
         }
      }

      if( isMasterVar )
      {
         foundMasterVarIndices.push_back( i );
      }
   }

   for( auto it = foundMasterVarIndices.rbegin(); it != foundMasterVarIndices.rend(); ++ it )
   {
      mastervars.push_back( lvars[ * it] );
      mastervarssorted = false;
      hvoutdated = true;
      isvarmaster[lvars[ * it]] = true;
      linkingvars.erase( linkingvars.begin() + * it );
   }
}


void PARTIALDECOMP::findVarsLinkingToStairlinking(
   )
{
   int i;
   int j;
   int k;

   int consblock;
   int block1 = - 1;
   int block2 = - 1;

   std::vector<int>& lvars = getLinkingvars();

   std::vector<int> foundMasterVarIndices;
   DETPROBDATA* detprobdata = this->getDetprobdata();

   sort();

   for( i = 0; i < getNLinkingvars(); ++ i )
   {
      block1 = - 1;
      block2 = - 1;
      std::vector<int>& varcons = detprobdata->getConssForVar( lvars[i] );
      for( j = 0; j < detprobdata->getNConssForVar( lvars[i] ); ++ j )
      {
         consblock = - 1;
         for( k = 0; k < nblocks; ++ k )
         {
            if( std::binary_search( conssforblocks[k].begin(), conssforblocks[k].end(), varcons[j] ) )
            {
               consblock = k;
               break;
            }
         }

         if( consblock == - 1 )
         {
            block1 = - 1;
            block2 = - 1;
            break;
         }
         else if( block1 == consblock || block2 == consblock )
         {
            continue;
         }
         else if( block1 == - 1 )
         {
            block1 = consblock;
            continue;
         }
         else if( block2 == - 1 )
         {
            block2 = consblock;
            continue;
         }
         else
         {
            block1 = - 1;
            block2 = - 1;
            break;
         }
      }

      if( block1 != - 1 && block2 != - 1 && ( block1 == block2 + 1 || block1 + 1 == block2 ) )
      {

         setVarToStairlinking( lvars[i], block1, block2 );
         foundMasterVarIndices.push_back( i );
      }
   }

   for( auto it = foundMasterVarIndices.rbegin(); it != foundMasterVarIndices.rend(); ++ it )
   {
      linkingvars.erase( linkingvars.begin() + * it );
   }
}


std::vector< std::pair< int, std::vector< int > > > PARTIALDECOMP::findLinkingVarsPotentiallyStairlinking(
   )
{
	std::vector< std::pair< int, std::vector< int > > > blocksOfVars( 0 );
	int blockcounter;
   DETPROBDATA* detprobdata = this->getDetprobdata();

   /* if there is at least one linking variable, then the blocks of vars must be created. */
   if( getNLinkingvars() > 0 )
   {
      std::vector<int>& lvars = getLinkingvars();
      sort();

      /* check every linking var */
      for ( int v = 0; v < getNLinkingvars(); ++v )
      {
         std::vector< int > blocksOfVar( 0 );
         blockcounter = 0;

         std::vector<int>& varcons = detprobdata->getConssForVar( lvars[v] );

         /* find all blocks that are hit by this linking var */
         for ( int c = 0; c < detprobdata->getNConssForVar( lvars[v] ) && blockcounter <= 2; ++c )
         {
            for ( int b = 0; b < nblocks && blockcounter <= 2; ++b )
            {
               if ( std::binary_search( conssforblocks[b].begin(),
                     conssforblocks[b].end(), varcons[c] ) )
               {
                  /* if the hit block is new, add it to blockOfVar vector */
                  if ( std::find( blocksOfVar.begin(), blocksOfVar.end(), b ) == blocksOfVar.end() )
                  {
                     ++blockcounter;
                     blocksOfVar.push_back( b );
                  }
               }
            }
         }

         /* if the var hits exactly two blocks, it is potentially stairlinking */
         if ( blockcounter == 2 )
         {
            std::pair< int, std::vector< int > > pair( v, blocksOfVar );
            blocksOfVars.push_back( pair );
         }
      }
   }

	return blocksOfVars;
}


int PARTIALDECOMP::getAncestorID(
   int ancestorindex
   )
{
   assert( 0 <= ancestorindex && ancestorindex < (int) listofancestorids.size() );

   return listofancestorids[ancestorindex];
}


std::vector<int>& PARTIALDECOMP::getAncestorList()
{
   return listofancestorids;
}

const std::vector<int> & PARTIALDECOMP::getBlocksForRep(int repid)
{
   return reptoblocks[repid];
}


void PARTIALDECOMP::setAncestorList(
   std::vector<int>& newlist
   )
{
   listofancestorids = newlist;
}


void PARTIALDECOMP::addAncestorID(
   int ancestor
   )
{
   assert( 0 <= ancestor );
   listofancestorids.push_back(ancestor);
}


void PARTIALDECOMP::removeAncestorID(
   int ancestorid
   )
{
   int i;
   std::vector<int> indices;
   // look for id (including duplicates)
   for(i = 0; i < (int) listofancestorids.size(); i++)
   {
      // add indices, descending
      if(listofancestorids.at(i) == ancestorid)
      {
         indices.insert(indices.begin(), i);
      }
   }
   // as indices are descending this is consistent
   for(auto index : indices)
      listofancestorids.erase(listofancestorids.begin() + index);
}


SCIP_Real PARTIALDECOMP::getDetectorClockTime(
   int detectorchainindex
   )
{
   assert( 0 <= detectorchainindex && detectorchainindex < (int) detectorclocktimes.size() );

   return detectorclocktimes[ detectorchainindex ];
}


std::vector<SCIP_Real>& PARTIALDECOMP::getDetectorClockTimes()
{
   return detectorclocktimes;
}


void PARTIALDECOMP::getConsPartitionData(
   int detectorchainindex,
   ConsPartition** partition,
   std::vector<int>& consclassesmaster
   )
{
   assert(consPartitionUsed(detectorchainindex) );

   *partition = dynamic_cast<ConsPartition*>( usedpartition[detectorchainindex] );
   consclassesmaster = classestomaster[detectorchainindex];
}


void PARTIALDECOMP::setDetectorClockTimes(
   std::vector<SCIP_Real>& newvector)
{
   detectorclocktimes = newvector;
}


bool PARTIALDECOMP::varPartitionUsed(
   int detectorchainindex
   )
{
   assert( 0 <= detectorchainindex && detectorchainindex < (int) usedpartition.size() );

   return (usedpartition[detectorchainindex] != NULL )
      && (dynamic_cast<VarPartition*>( usedpartition[detectorchainindex] ) != NULL );
}


std::vector<int>& PARTIALDECOMP::getConssForBlock(
   int block
   )
{
   assert( block >= 0 && block < nblocks );
   return conssforblocks[block];
}


std::vector<DEC_DETECTOR*>& PARTIALDECOMP::getDetectorchain()
{
   return detectorchain;
}


std::string PARTIALDECOMP::getDetectorStatistics(
   int detectorchainindex
   )
{
   std::stringstream output;

   if( (int) getDetectorClockTimes().size() > detectorchainindex )
      output << "  Detection time: " << getDetectorClockTime( detectorchainindex ) << std::endl;
   if( (int) getPctConssFromFreeVector().size() > detectorchainindex )
      output << "  % newly assigned constraints: " << getPctConssFromFree( detectorchainindex ) << std::endl;
   if( (int) getPctConssToBorderVector().size() > detectorchainindex )
      output << "  % constraints the detector assigned to border: " << getPctConssToBorder( detectorchainindex ) << std::endl;
   if( (int) getPctConssToBlockVector().size() > detectorchainindex )
      output << "  % constraints the detector assigned to blocks: " << getPctConssToBlock( detectorchainindex ) << std::endl;
   if( (int) getPctVarsFromFreeVector().size() > detectorchainindex )
      output << "  % newly assigned variables: " << getPctVarsFromFree( detectorchainindex ) << std::endl;
   if( (int) getPctVarsToBorderVector().size() > detectorchainindex )
      output << "  % variables the detector assigned to border: " << getPctVarsToBorder( detectorchainindex ) << std::endl;
   if( (int) getPctVarsToBlockVector().size() > detectorchainindex )
      output << "  % variables the detector assigned to blocks: " << getPctVarsToBlock( detectorchainindex ) << std::endl;
   if( (int) getNNewBlocksVector().size() > detectorchainindex )
         output << "  New blocks: " << getNNewBlocks( detectorchainindex ) << std::endl;

   return output.str();
}


std::string PARTIALDECOMP::getDetectorPartitionInfo(
   int detectorchainindex,
   bool displayConssVars
   )
{
   std::stringstream output;
   DETPROBDATA* detprobdata = this->getDetprobdata();

   if( consPartitionUsed(detectorchainindex) )
   {
      ConsPartition* partition;
      std::vector<int> constomaster;

      getConsPartitionData(detectorchainindex, &partition, constomaster);

      output << "  Used conspartition: " << partition->getName() << std::endl;
      output << "   Pushed to master:";

      if( !constomaster.empty() )
      {
         if( displayConssVars )
         {
            output << std::endl << "    " << partition->getClassName(constomaster[0] ) << " ("
                   << partition->getClassDescription(constomaster[0] ) << "): ";
            bool first = true;
            for( int c = 0; c < partition->getNConss(); ++c )
            {
               if( partition->getClassOfCons(c ) == constomaster[0] )
               {
                  if( first )
                  {
                     output << SCIPconsGetName(detprobdata->getCons(c));
                     first = false;
                  }
                  else
                  {
                     output << ", " << SCIPconsGetName(detprobdata->getCons(c));
                  }
               }
            }
            output << std::endl;
         }
         else
         {
            output << " " << partition->getClassName(constomaster[0] );
         }
      }

      for( size_t i = 1; i < constomaster.size(); ++i )
      {
         if( displayConssVars )
         {
            output << "    " << partition->getClassName(constomaster[i] ) << " ("
                   << partition->getClassDescription(constomaster[i] ) << "): ";
            bool first = true;
            for( int c = 0; c < partition->getNConss(); ++c )
            {
               if( partition->getClassOfCons(c ) == constomaster[i] )
               {
                  if( first )
                  {
                     output << SCIPconsGetName(detprobdata->getCons(c));
                     first = false;
                  }
                  else
                  {
                     output << ", " << SCIPconsGetName(detprobdata->getCons(c));
                  }
               }
            }
            output << std::endl;
         }
         else
         {
            output << ", " << partition->getClassName(constomaster[i] );
         }
      }

      if ( !displayConssVars || constomaster.empty() )
      {
         output << std::endl;
      }
   }

   if( varPartitionUsed(detectorchainindex) )
   {
      VarPartition* partition;
      std::vector<int> vartolinking;
      std::vector<int> vartomaster;

      getVarPartitionData(detectorchainindex, &partition, vartolinking, vartomaster);

      output << "  Used varpartition: " << partition->getName() << std::endl;
      output << "   Pushed to linking:";

      if( !vartolinking.empty() )
      {
         if( displayConssVars )
         {
            output << std::endl << "    " << partition->getClassName(vartolinking[0] ) << " ("
                   << partition->getClassDescription(vartolinking[0] ) << "): ";
            bool first = true;
            for( int v = 0; v < partition->getNVars(); ++v )
            {
               if( partition->getClassOfVar(v ) == vartolinking[0] )
               {
                  if( first )
                  {
                     output << SCIPvarGetName(detprobdata->getVar(v));
                     first = false;
                  }
                  else
                  {
                     output << ", " << SCIPvarGetName(detprobdata->getVar(v));
                  }
               }
            }
            output << std::endl;
         }
         else
         {
            output << " " << partition->getClassName(vartolinking[0] );
         }
      }

      for( size_t i = 1; i < vartolinking.size(); ++i )
      {
         if( displayConssVars )
         {
            output << "    " << partition->getClassName(vartolinking[i] ) << " ("
                   << partition->getClassDescription(vartolinking[i] ) << "): ";
            bool first = true;
            for( int v = 0; v < partition->getNVars(); ++v )
            {
               if( partition->getClassOfVar(v ) == vartolinking[i] )
               {
                  if( first )
                  {
                     output << SCIPvarGetName(detprobdata->getVar(v));
                     first = false;
                  }
                  else
                  {
                     output << ", " << SCIPvarGetName(detprobdata->getVar(v));
                  }
               }
            }
            output << std::endl;
         }
         else
         {
            output << ", " << partition->getClassName(vartolinking[i] );
         }
      }

      if ( !displayConssVars || vartolinking.empty() )
      {
         output << std::endl;
      }

      output << "   Pushed to master:";

      if( !vartomaster.empty() )
      {
         if( displayConssVars )
         {
            output << std::endl << "    " << partition->getClassName(vartomaster[0] ) << " ("
                   << partition->getClassDescription(vartomaster[0] ) << "): ";
            bool first = true;
            for( int v = 0; v < partition->getNVars(); ++v )
            {
               if( partition->getClassOfVar(v ) == vartomaster[0] )
               {
                  if( first )
                  {
                     output << SCIPvarGetName(detprobdata->getVar(v));
                     first = false;
                  }
                  else
                  {
                     output << ", " << SCIPvarGetName(detprobdata->getVar(v));
                  }
               }
            }
            output << std::endl;
         }
         else
         {
            output << " " << partition->getClassName(vartomaster[0] );
         }
      }

      for( size_t i = 1; i < vartomaster.size(); ++i )
      {
         if( displayConssVars )
         {
            output << "    " << partition->getClassName(vartomaster[i] ) << " ("
                   << partition->getClassDescription(vartomaster[i] ) << "): ";
            bool first = true;
            for( int v = 0; v < partition->getNVars(); ++v )
            {
               if( partition->getClassOfVar(v ) == vartolinking[i] )
               {
                  if( first )
                  {
                     output << SCIPvarGetName(detprobdata->getVar(v));
                     first = false;
                  }
                  else
                  {
                     output << ", " << SCIPvarGetName(detprobdata->getVar(v));
                  }
               }
            }
            output << std::endl;
         }
         else
         {
            output << ", " << partition->getClassName(vartomaster[i] );
         }
      }

      if ( !displayConssVars || vartomaster.empty() )
      {
         output << std::endl;
      }
   }

   return output.str();
}


bool PARTIALDECOMP::getFinishedByFinisher()
{
   return isfinishedbyfinisher;
}


unsigned long PARTIALDECOMP::getHashValue()
{
   calcHashvalue();
   return hashvalue;
}


int PARTIALDECOMP::getID()
{
   return id;
}


std::vector<int>& PARTIALDECOMP::getLinkingvars()
{
   return linkingvars;
}


std::vector<int>& PARTIALDECOMP::getMasterconss()
{
   return masterconss;
}


std::vector<int>& PARTIALDECOMP::getMastervars()
{
   return mastervars;
}


int PARTIALDECOMP::getNCoeffsForBlock(
   int blockid
   ){

   if( !calculatedncoeffsforblock )
      calcNCoeffsForBlocks();

   return ncoeffsforblock[blockid];
}


int PARTIALDECOMP::getNCoeffsForMaster(
   ){

   if( !calculatedncoeffsforblock )
      calcNCoeffsForBlocks();

   return ncoeffsformaster;
}


SCIP_Real PARTIALDECOMP::getScore(
   SCORETYPE type
)
{
   /* if there are indicator constraints in the master we want to reject this decomposition */
   for( int mc = 0; mc < getNMasterconss(); ++mc )
   {
      SCIP_CONS* cons;
      cons = getDetprobdata()->getCons(getMasterconss()[mc]);
      if( GCGconsGetType(scip, cons) == consType::indicator )
         return 0.;
   }

   if( type == scoretype::MAX_WHITE )
   {
      if( maxwhitescore == -1. )
         GCGconshdlrDecompCalcMaxWhiteScore(scip, this->id, &maxwhitescore);
      return maxwhitescore;
   }
   else if( type == scoretype::CLASSIC )
   {
      if ( classicscore == -1. )
         GCGconshdlrDecompCalcClassicScore(scip, this->id, &classicscore);
      return classicscore;
   }
   else if( type == scoretype::BORDER_AREA )
   {
      if( borderareascore == -1. )
         GCGconshdlrDecompCalcBorderAreaScore(scip, this->id, &borderareascore);
      return borderareascore;
   }
   else if( type == scoretype::MAX_FORESSEEING_WHITE )
   {
      if( maxforeseeingwhitescore == -1. )
         GCGconshdlrDecompCalcMaxForseeingWhiteScore(scip, this->id, &maxforeseeingwhitescore);
      return maxforeseeingwhitescore;
   }
   else if( type == scoretype::MAX_FORESEEING_AGG_WHITE )
   {
      if( maxforeseeingwhitescoreagg == -1. )
         GCGconshdlrDecompCalcMaxForeseeingWhiteAggScore(scip, this->id, &maxforeseeingwhitescoreagg);
      return maxforeseeingwhitescoreagg;
   }
   else if( type == scoretype::SETPART_FWHITE )
   {
      if( setpartfwhitescore == -1. )
         GCGconshdlrDecompCalcSetPartForseeingWhiteScore(scip, this->id, &setpartfwhitescore);
      return setpartfwhitescore;
   }
   else if( type == scoretype::SETPART_AGG_FWHITE )
   {
      if( setpartfwhitescoreagg == -1. )
         GCGconshdlrDecompCalcSetPartForWhiteAggScore(scip, this->id, &setpartfwhitescoreagg);
      return setpartfwhitescoreagg;
   }
   else if( type == scoretype::BENDERS )
   {
      if( bendersscore == -1. )
         GCGconshdlrDecompCalcBendersScore(scip, this->id, &bendersscore);
      return bendersscore;
   }
   else if( type == scoretype::STRONG_DECOMP )
   {
      if( strongdecompositionscore == -1. )
         GCGconshdlrDecompCalcStrongDecompositionScore(scip, this->id, &strongdecompositionscore);
      return strongdecompositionscore;
   }
   
   return 0;
}


USERGIVEN PARTIALDECOMP::getUsergiven()
{
   return usergiven;
}


int PARTIALDECOMP::getNAncestors()
{
   return listofancestorids.size();
}


int PARTIALDECOMP::getNBlocks()
{
   return nblocks;
}


int PARTIALDECOMP::getNConss()
{
   return nconss;
}


int PARTIALDECOMP::getNConssForBlock(
   int block
   )
{
   assert( block >= 0 && block < nblocks );
   return (int) conssforblocks[block].size();
}


std::vector<std::string>& PARTIALDECOMP::getDetectorchainInfo()
{
   return detectorchaininfo;
}


int PARTIALDECOMP::getNDetectors()
{
   return (int) detectorchain.size();
}


int PARTIALDECOMP::getNUsedPartitions()
{
   return (int) usedpartition.size();
}


int PARTIALDECOMP::getNLinkingvars()
{
   return (int) linkingvars.size();
}


int PARTIALDECOMP::getNNewBlocks(
      int detectorchainindex
   )
{
   assert( 0 <= detectorchainindex && detectorchainindex < (int) detectorchain.size() );

   return nnewblocks[detectorchainindex];
}


std::vector<int> PARTIALDECOMP::getNNewBlocksVector()
{
   return nnewblocks;
}


int PARTIALDECOMP::getNMasterconss()
{
   return (int) masterconss.size();
}


int PARTIALDECOMP::getNMastervars()
{
   return (int) mastervars.size();
}


int PARTIALDECOMP::getNTotalStairlinkingvars()
{
   int nstairlinkingvars = 0;
   for( int b = 0; b < getNBlocks(); ++ b )
      nstairlinkingvars += getNStairlinkingvars( b );

   return nstairlinkingvars;
}


int PARTIALDECOMP::getNOpenconss()
{
   return (int) openconss.size();
}


int PARTIALDECOMP::getNOpenvars()
{
   return (int) openvars.size();
}


int PARTIALDECOMP::getNReps(){

   return nrepblocks;
}


int PARTIALDECOMP::getNStairlinkingvars(
   int block
   )
{
   assert( block >= 0 && block < nblocks );
   assert( (int) stairlinkingvars.size() > block );
   return (int) stairlinkingvars[block].size();
}


int PARTIALDECOMP::getNVars()
{
   return nvars;
}


int PARTIALDECOMP::getNVarsForBlock(
   int block
   )
{
   assert( block >= 0 && block < nblocks );
   return (int) varsforblocks[block].size();
}


int PARTIALDECOMP::getNVarsForBlocks()
{
   int count = 0;
   for( auto& block : varsforblocks )
   {
      count += (int) block.size();
   }

   return count;
}


const int* PARTIALDECOMP::getOpenconss()
{
   return openconss.data();
}


std::vector<int>& PARTIALDECOMP::getOpenconssVec()
{
   return openconss;
}


const int* PARTIALDECOMP::getOpenvars()
{
   return openvars.data();
}


std::vector<int>& PARTIALDECOMP::getOpenvarsVec()
{
   return openvars;
}


SCIP_Real PARTIALDECOMP::getPctVarsToBorder(
   int detectorchainindex
   )
{
   assert( 0 <= detectorchainindex && detectorchainindex < (int) detectorchain.size() );

   return pctvarstoborder[detectorchainindex];
}


std::vector<SCIP_Real>& PARTIALDECOMP::getPctVarsToBorderVector()
{
   return pctvarstoborder;
}


void PARTIALDECOMP::setPctVarsToBorderVector(
   std::vector<SCIP_Real>& newvector
   )
{
   pctvarstoborder = newvector;
}


SCIP_Real PARTIALDECOMP::getPctVarsToBlock(
   int detectorchainindex
   )
{
   assert( 0 <= detectorchainindex && detectorchainindex < (int) detectorchain.size() );

   return pctvarstoblock[detectorchainindex];
}


std::vector<SCIP_Real>& PARTIALDECOMP::getPctVarsToBlockVector()
{
   return pctvarstoblock;
}



void PARTIALDECOMP::setPctVarsToBlockVector(
   std::vector<SCIP_Real>& newvector
)
{
   pctvarstoblock = newvector;
}


SCIP_Real PARTIALDECOMP::getPctVarsFromFree(
   int detectorchainindex
   )
{
   assert( 0 <= detectorchainindex && detectorchainindex < (int) detectorchain.size() );

   return pctvarsfromfree[detectorchainindex];
}


std::vector<SCIP_Real>& PARTIALDECOMP::getPctVarsFromFreeVector()
{
   return pctvarsfromfree;
}


void PARTIALDECOMP::setPctVarsFromFreeVector(
   std::vector<SCIP_Real>& newvector
   )
{
   pctvarsfromfree = newvector;
}


SCIP_Real PARTIALDECOMP::getPctConssToBorder(
   int detectorchainindex
   )
{
   assert( 0 <= detectorchainindex && detectorchainindex < (int) detectorchain.size() );

   return pctconsstoborder[detectorchainindex];
}


std::vector<SCIP_Real>& PARTIALDECOMP::getPctConssToBorderVector()
{
   return pctconsstoborder;
}


void PARTIALDECOMP::setPctConssToBorderVector(
   std::vector<SCIP_Real>& newvector
   )
{
   pctconsstoborder = newvector;
}


SCIP_Real PARTIALDECOMP::getPctConssToBlock(
   int detectorchainindex
   )
{
   assert( 0 <= detectorchainindex && detectorchainindex < (int) detectorchain.size() );

   return pctconsstoblock[detectorchainindex];
}


std::vector<SCIP_Real>& PARTIALDECOMP::getPctConssToBlockVector()
{
   return pctconsstoblock;
}


void PARTIALDECOMP::setPctConssToBlockVector(
   std::vector<SCIP_Real>& newvector
   )
{
   pctconsstoblock = newvector;
}


SCIP_Real PARTIALDECOMP::getPctConssFromFree(
   int detectorchainindex
   )
{
   assert( 0 <= detectorchainindex && detectorchainindex < (int) detectorchain.size() );

   return pctconssfromfree[detectorchainindex];
}


std::vector<SCIP_Real>& PARTIALDECOMP::getPctConssFromFreeVector()
{
   return pctconssfromfree;
}


int PARTIALDECOMP::getRepForBlock(
   int blockid
   )
{
     return blockstorep[blockid];
}

std::vector<int>& PARTIALDECOMP::getRepVarmap(
   int repid,
   int blockrepid
   )
{
   return pidtopidvarmaptofirst[repid][blockrepid];
}


DETPROBDATA* PARTIALDECOMP::getDetprobdata()
{
   DETPROBDATA* detprobdata;
   if( original )
      detprobdata = GCGconshdlrDecompGetDetprobdataOrig(scip);
   else
      detprobdata = GCGconshdlrDecompGetDetprobdataPresolved(scip);

   assert(detprobdata != NULL);

   return detprobdata;
}


void PARTIALDECOMP::setPctConssFromFreeVector(
   std::vector<SCIP_Real>& newvector
   )
{
   pctconssfromfree = newvector;
}


const int* PARTIALDECOMP::getStairlinkingvars(
   int block
   )
{
   assert( block >= 0 && block < nblocks );
   return stairlinkingvars[block].data();
}


void PARTIALDECOMP::getVarPartitionData(
   int detectorchainindex,
   VarPartition** partition,
   std::vector<int>& varclasseslinking,
   std::vector<int>& varclassesmaster
   )
{
   assert(varPartitionUsed(detectorchainindex) );

   *partition = dynamic_cast<VarPartition*>( usedpartition[detectorchainindex] );
   varclasseslinking = classestolinking[detectorchainindex];
   varclassesmaster = classestomaster[detectorchainindex];
}


std::vector<int>& PARTIALDECOMP::getVarsForBlock(
   int block
   )
{
   assert( block >= 0 && block < nblocks );
   return varsforblocks[block];
}


int PARTIALDECOMP::getVarProbindexForBlock(
   int varid,
   int block
   )
{
   std::vector<int>::iterator lb = lower_bound( varsforblocks[block].begin(), varsforblocks[block].end(), varid );

   if( lb != varsforblocks[block].end() )
      return (int) ( lb - varsforblocks[block].begin() );
   else
      return -1;

}


bool PARTIALDECOMP::isComplete()
{
   return ( 0 == getNOpenconss() && 0 == getNOpenvars() );
}


bool PARTIALDECOMP::isConsBlockconsOfBlock(
   int cons,
   int block
   )
{
   assert( cons >= 0 && cons < nconss );
   assert( block >= 0 && block < nblocks );
   std::vector<int>::iterator lb = lower_bound( conssforblocks[block].begin(), conssforblocks[block].end(), cons );
   if( lb != conssforblocks[block].end() &&  *lb == cons )
      return true;
   else
      return false;
}


bool PARTIALDECOMP::isConsMastercons(
   int cons
   )
{
   assert( cons >= 0 && cons < nconss );
  return isconsmaster[cons];
}


bool PARTIALDECOMP::isConsOpencons(
   int cons
   )
{
   assert( cons >= 0 && cons < nconss );
   return isconsopen[cons];
}


bool PARTIALDECOMP::isAssignedToOrigProb()
{
   return original;
}


SCIP_RETCODE PARTIALDECOMP::isEqual(
   PARTIALDECOMP* otherpartialdec,
   SCIP_Bool* isequal,
   bool sortpartialdecs
   )
{
   if( sortpartialdecs )
   {
      sort();
      otherpartialdec->sort();
   }

   * isequal = isEqual( otherpartialdec );

   return SCIP_OKAY;
}


bool PARTIALDECOMP::isEqual(
   PARTIALDECOMP* other
   )
{
   if( getNMasterconss() != other->getNMasterconss() || getNMastervars() != other->getNMastervars()
      || getNBlocks() != other->getNBlocks() || getNLinkingvars() != other->getNLinkingvars() )
      return false;

   std::vector<std::pair<int, int>> blockorderthis;
   std::vector<std::pair<int, int>> blockorderother;

   /* find sorting for blocks (non decreasing according smallest row index) */
   for( int i = 0; i < this->nblocks; ++ i )
   {
      blockorderthis.emplace_back(i, conssforblocks[i][0]);
      blockorderother.emplace_back(i, other->conssforblocks[i][0]);
   }

   std::sort(blockorderthis.begin(), blockorderthis.end(), compare_blocks);
   std::sort(blockorderother.begin(), blockorderother.end(), compare_blocks);

   /* compares the number of stairlinking vars */
   for( int b = 0; b < getNBlocks(); ++ b )
   {
      int blockthis = blockorderthis[b].first;
      int blockother = blockorderother[b].first;

      if( getNStairlinkingvars(blockthis) != other->getNStairlinkingvars(blockother) )
         return false;
   }

   /* compares the number of constraints and variables in the blocks*/
   for( int b = 0; b < getNBlocks(); ++ b )
   {
      int blockthis = blockorderthis[b].first;
      int blockother = blockorderother[b].first;

      if( ( getNVarsForBlock( blockthis ) != other->getNVarsForBlock(blockother) )
         || ( getNConssForBlock( blockthis ) != other->getNConssForBlock(blockother) ) )
         return false;
   }

   /* compares the master cons */
   for( int j = 0; j < getNMasterconss(); ++ j )
   {
      if( getMasterconss()[j] != other->getMasterconss()[j] )
         return false;
   }

   /* compares the master vars */
   for( int j = 0; j < getNMastervars(); ++ j )
   {
      if( getMastervars()[j] != other->getMastervars()[j] )
         return false;
   }

   /* compares the constrains and variables in the blocks */
   for( int b = 0; b < getNBlocks(); ++ b )
   {
      int blockthis = blockorderthis[b].first;
      int blockother = blockorderother[b].first;

      for( int j = 0; j < getNConssForBlock( blockthis ); ++ j )
      {
         if( getConssForBlock( blockthis )[j] != other->getConssForBlock(blockother)[j] )
            return false;
      }

      for( int j = 0; j < getNVarsForBlock( blockthis ); ++ j )
      {
         if( getVarsForBlock(blockthis)[j] != other->getVarsForBlock(blockother)[j] )
            return false;
      }

      for( int j = 0; j < getNStairlinkingvars( blockthis ); ++ j )
      {
         if( getStairlinkingvars(blockthis)[j] != other->getStairlinkingvars(blockother)[j] )
            return false;
      }
   }

   /* compares the linking vars */
   for( int j = 0; j < getNLinkingvars(); ++ j )
   {
      if( getLinkingvars()[j] != other->getLinkingvars()[j] )
         return false;
   }

   return true;
}


bool PARTIALDECOMP::isPropagatedBy(
   DEC_DETECTOR* detector
   )
{
   std::vector<DEC_DETECTOR*>::const_iterator iter = std::find(detectorchain.begin(), detectorchain.end(), detector);

   return iter != detectorchain.end();
}


bool PARTIALDECOMP::isTrivial()
{
   if( getNBlocks() == 1 && (SCIP_Real) getNConssForBlock( 0 ) >= 0.95 * getNConss() )
      return true;

   if( getNConss() == getNMasterconss() )
      return true;

   if( getNConss() == getNOpenconss() && getNVars() == getNOpenvars() )
      return true;

   if( getNVars() == getNMastervars() + getNLinkingvars() )
      return true;

   return false;
}


bool PARTIALDECOMP::isSelected()
{
   return isselected;
}


bool PARTIALDECOMP::isVarBlockvarOfBlock(
   int var,
   int block
   )
{
   assert( var >= 0 && var < nvars );
   assert( block >= 0 && block < nconss );

   std::vector<int>::iterator lb = lower_bound(varsforblocks[block].begin(), varsforblocks[block].end(), var);
   if( lb != varsforblocks[block].end() &&  *lb == var )
      return true;
   else
      return false;
}


bool PARTIALDECOMP::isVarMastervar(
   int var
   )
{
   assert( var >= 0 && var < nvars );
  return isvarmaster[var];
}


bool PARTIALDECOMP::isVarLinkingvar(
   int var
   )
{
   assert( var >= 0 && var < nvars );
   std::vector<int>::iterator lb = lower_bound(linkingvars.begin(), linkingvars.end(), var);
   if( lb != linkingvars.end() &&  *lb == var )
      return true;
   else
      return false;
}


bool PARTIALDECOMP::isVarOpenvar(
   int var
   )
{
   assert( var >= 0 && var < nvars );
   return isvaropen[var];
}


bool PARTIALDECOMP::isVarStairlinkingvar(
   int var
   )
{
   for( int b = 0; b < nblocks; ++ b )
   {
      std::vector<int>::iterator lb = lower_bound(stairlinkingvars[b].begin(), stairlinkingvars[b].end(), var);
      if( lb != stairlinkingvars[b].end() &&  *lb == var )
         return true;
   }
   return false;
}


bool PARTIALDECOMP::isVarStairlinkingvarOfBlock(
   int var,
   int block
   )
{
   assert( var >= 0 && var < nvars );
   assert( block >= 0 && block < nblocks );
   std::vector<int>::iterator lb = lower_bound(stairlinkingvars[block].begin(), stairlinkingvars[block].end(), var);
   if( lb != stairlinkingvars[block].end() &&  *lb == var )
      return true;
   else
   {
      if( block == 0 )
         return false;
      else
      {
         lb = lower_bound(stairlinkingvars[block - 1].begin(), stairlinkingvars[block - 1].end(), var);
         return ( lb != stairlinkingvars[block-1].end() &&  *lb == var );
      }
   }
}


void PARTIALDECOMP::printPartitionInformation(
   SCIP*                givenscip,
   FILE*                file
   )
{

   int nusedpartitions = (int) getNUsedPartitions();
   int nconspartitions = 0;
   int nvarpartitions = 0;

   for( int i = 0; i < nusedpartitions; ++i)
   {
      if( usedpartition[i] == NULL )
         continue;

      if( dynamic_cast<ConsPartition*>( usedpartition[i] ) != NULL )
      {
         /* partition is cons partition */
         ++nconspartitions;
      }
      else
      {
         /* partition is var partition */
         ++nvarpartitions;
      }
   }

   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%d\n", nconspartitions);

   for( int i = 0; i < nusedpartitions; ++i)
   {
      if( dynamic_cast<ConsPartition*>( usedpartition[i] ) != NULL )
      {
         /* partition is cons partition */
         int nmasterclasses = (int) classestomaster[i].size();
         SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%s\n", usedpartition[i]->getName());
         SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%d\n", nmasterclasses);
         for ( int mclass = 0; mclass < (int) classestomaster[i].size(); ++mclass )
         {
            int classid = classestomaster[i][mclass];
            SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%s\n", usedpartition[i]->getClassName(classid));
         }
      }
   }

   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%d\n", nvarpartitions);

   for( int i = 0; i < nusedpartitions; ++i)
   {
      if( dynamic_cast<VarPartition*>( usedpartition[i] ) != NULL )
      {
         /* partition is var partition */
         int nmasterclasses = (int) classestomaster[i].size();
         int nlinkingclasses = (int) classestolinking[i].size();
         SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%s\n", usedpartition[i]->getName());
         SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%d\n", nmasterclasses);
         for ( int mclass = 0; mclass < (int) classestomaster[i].size();   ++mclass )
         {
            int classid = classestomaster[i][mclass];
            SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%s : %s\n", usedpartition[i]->getClassName(classid), usedpartition[i]->getClassDescription(classid));
         }

         SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%d\n", nlinkingclasses  );
         for ( int linkingclass = 0; linkingclass < nlinkingclasses;   ++linkingclass )
         {
            int classid = classestolinking[i][linkingclass];
            SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%s : %s\n", usedpartition[i]->getClassName(classid), usedpartition[i]->getClassDescription(classid));
         }

      }
   }
}


void PARTIALDECOMP::refineToBlocks(
   )
{
   bool success = true;

   while( success )
      success = assignHittingOpenconss() || assignHittingOpenvars();
   sort();
}


void PARTIALDECOMP::refineToMaster(
    )
{
   considerImplicits();
   assignOpenPartialHittingToMaster();
}


void PARTIALDECOMP::setConsPartitionStatistics(
   int detectorchainindex,
   ConsPartition* partition,
   std::vector<int>& consclassesmaster
   )
{
   assert( 0 <= detectorchainindex  );

   if( detectorchainindex >= (int) usedpartition.size() )
   {
      usedpartition.resize(detectorchainindex + 1);
      classestomaster.resize(detectorchainindex + 1);
   }

   usedpartition[detectorchainindex] = partition;
   classestomaster[detectorchainindex] = consclassesmaster;
}


void PARTIALDECOMP::setConsToBlock(
   int consToBlock,
   int block
   )
{
   assert( consToBlock >= 0 && consToBlock < nconss );
   assert( block >= 0 && block < nblocks );
   assert( (int) conssforblocks.size() > block );

   conssforblocks[block].push_back( consToBlock );
   conssforblocksorted = false;
   hvoutdated = true;
}


void PARTIALDECOMP::fixConsToBlock(
   int cons,
   int block
   )
{
   assert( cons >= 0 && cons < nconss );
   assert( isconsopen[cons] );

   if( block >= nblocks )
      setNBlocks(block+1);
   assert( block >= 0 && block < nblocks );

   setConsToBlock(cons, block);
   deleteOpencons(cons);
}

bool PARTIALDECOMP::fixConsToBlock(
   SCIP_CONS*            cons,                /**< pointer of the constraint */
   int                   block                /**< block index (counting from 0) */
   )
{
   int consindex = getDetprobdata()->getIndexForCons(cons);

   if( consindex >= 0 )
   {
      fixConsToBlock(consindex, block);
      return true;
   }
   return false;
}


void PARTIALDECOMP::setConsToMaster(
   int consToMaster
   )
{
   assert( consToMaster >= 0 && consToMaster < nconss );
   masterconss.push_back( consToMaster );
   isconsmaster[consToMaster] = true;
   masterconsssorted = false;
   hvoutdated = true;
}


std::vector<int>::const_iterator PARTIALDECOMP::fixConsToMaster(
   std::vector<int>::const_iterator itr
   )
{
   assert(itr != openconss.cend());
   assert(isconsopen[*itr]);
   setConsToMaster(*itr);
   return deleteOpencons(itr);
}


void PARTIALDECOMP::fixConsToMaster(
   int cons
   )
{
   assert( cons >= 0 && cons < nconss );
   assert(isconsopen[cons]);

   setConsToMaster(cons);
   deleteOpencons(cons);
}

bool PARTIALDECOMP::fixConsToMaster(
   SCIP_CONS* cons   /**< pointer of cons to fix as master cons */
   )
{
   int consindex = getDetprobdata()->getIndexForCons(cons);
   if( consindex >= 0 )
   {
      fixConsToMaster(consindex);
      return true;
   }
   return false;
}


void PARTIALDECOMP::setDetectorchain(
   std::vector<DEC_DETECTOR*>& givenDetectorChain
   )
{
   detectorchain = givenDetectorChain;
}


void PARTIALDECOMP::setDetectorPropagated(
   DEC_DETECTOR* detectorID
   )
{
   detectorchain.push_back(detectorID);
   addEmptyPartitionStatistics();
}


void PARTIALDECOMP::setDetectorFinished(
   DEC_DETECTOR* detectorID
   )
{
   isfinishedbyfinisher = true;
   detectorchain.push_back(detectorID);
   addEmptyPartitionStatistics();
}


void PARTIALDECOMP::setDetectorFinishedOrig(
   DEC_DETECTOR* detectorID
   )
{
   isfinishedbyfinisherorig = true;
}


void PARTIALDECOMP::setFinishedByFinisher(
   bool finished
   )
{
   isfinishedbyfinisher = finished;
}


void PARTIALDECOMP::setFinishedByFinisherOrig(
   bool finished
   )
{
   isfinishedbyfinisherorig = finished;
}


void PARTIALDECOMP::setNBlocks(
   int newNBlocks
   )
{
   assert( newNBlocks >= nblocks );

   assert( (int) conssforblocks.size() == nblocks );
   assert( (int) varsforblocks.size() == nblocks );
   assert( (int) stairlinkingvars.size() == nblocks );
   /* increase number of blocks in conssforblocks and varsforblocks */

   for( int b = nblocks; b < newNBlocks; ++ b )
   {
      conssforblocks.emplace_back(0);
      varsforblocks.emplace_back(0);
      stairlinkingvars.emplace_back(0);
   }

   nblocks = newNBlocks;
}


void PARTIALDECOMP::setSelected(
   bool selected
   )
{
   isselected = selected;
}


void PARTIALDECOMP::setStemsFromOrig(
   bool fromorig
   )
{
   stemsfromorig = fromorig;
}


void PARTIALDECOMP::setUsergiven(
   USERGIVEN givenusergiven
   )
{
   usergiven = givenusergiven;
}


void PARTIALDECOMP::setVarPartitionStatistics(
   int detectorchainindex,
   VarPartition* partition,
   std::vector<int>& varclasseslinking,
   std::vector<int>& varclassesmaster
   )
{
   assert( 0 <= detectorchainindex );

   if( detectorchainindex >= (int) usedpartition.size() )
    {
       usedpartition.resize(detectorchainindex + 1);
       classestomaster.resize(detectorchainindex + 1);
       classestolinking.resize(detectorchainindex + 1);
    }


   usedpartition[detectorchainindex] = partition;
   classestolinking[detectorchainindex] = varclasseslinking;
   classestomaster[detectorchainindex] = varclassesmaster;
}


void PARTIALDECOMP::setVarToBlock(
   int varToBlock,
   int block
   )
{
   assert( varToBlock >= 0 && varToBlock < nvars );
   assert( block >= 0 && block < nblocks );
   assert( (int) varsforblocks.size() > block );

   varsforblocks[block].push_back(varToBlock);
   varsforblocksorted = false;
   hvoutdated = true;
}


void PARTIALDECOMP::fixVarToBlock(
   int var,
   int block
   )
{
   assert( var >= 0 && var < nvars );
   assert( isvaropen[var] );
   assert( block >= 0 && block < nblocks );

   if( isVarLinkingvar(var) )
      return;
   
   setVarToBlock(var, block);
   deleteOpenvar(var);
}


std::vector<int>::const_iterator PARTIALDECOMP::fixVarToBlock(
   std::vector<int>::const_iterator itr,
   int block
)
{
   assert( itr != openvars.cend() );
   assert( isvaropen[*itr] );
   assert( block >= 0 && block < nblocks );

   if( isVarLinkingvar(*itr) )
      return ++itr;

   setVarToBlock(*itr, block);
   return deleteOpenvar(itr);
}


void PARTIALDECOMP::setVarToLinking(
   int varToLinking
   )
{
   assert( varToLinking >= 0 && varToLinking < nvars );
   linkingvars.push_back( varToLinking );
   linkingvarssorted = false;
   hvoutdated = true;
}


void PARTIALDECOMP::fixVarToLinking(
   int var
   )
{
   assert( var >= 0 && var < nvars );
   assert( isvaropen[var]);

   setVarToLinking(var);
   deleteOpenvar(var);

   // delete this var in blocks
   for(auto block : varsforblocks)
   {
      for(auto iter = block.begin(); iter != block.end(); iter++)
      {
         if((*iter) == var)
            block.erase(iter);
      }
   }
}


std::vector<int>::const_iterator PARTIALDECOMP::fixVarToLinking(
   std::vector<int>::const_iterator itr
)
{
   assert( itr != openvars.cend() );
   assert( isvaropen[*itr] );

   setVarToLinking(*itr);

   // delete this var in blocks
   for(auto block : varsforblocks)
   {
      for(auto iter = block.begin(); iter != block.end(); iter++)
      {
         if((*iter) == *itr)
            block.erase(iter);
      }
   }
   return deleteOpenvar(itr);;
}


void PARTIALDECOMP::setVarToMaster(
   int varToMaster
   )
{
   assert( varToMaster >= 0 && varToMaster < nvars );
   mastervars.push_back( varToMaster );
   isvarmaster[varToMaster] = true;
   mastervarssorted = false;
   hvoutdated = true;
}


void PARTIALDECOMP::fixVarToMaster(
   int var
   )
{
   assert( var >= 0 && var < nvars );
   assert( isvaropen[var]);

   setVarToMaster(var);
   deleteOpenvar(var);
}


std::vector<int>::const_iterator PARTIALDECOMP::fixVarToMaster(
   std::vector<int>::const_iterator itr
)
{
   assert( itr != openvars.cend() );
   assert( isvaropen[*itr] );

   setVarToMaster(*itr);
   return deleteOpenvar(itr);
}


void PARTIALDECOMP::setVarToStairlinking(
   int varToStairlinking,
   int block1,
   int block2
   )
{
   assert( varToStairlinking >= 0 && varToStairlinking < nvars );
   assert( block1 >= 0 && block1 <= nblocks );
   assert( block2 >= 0 && block2 <= nblocks );
   assert( ( block1 + 1 == block2 ) || ( block2 + 1 == block1 ) );

   if( block1 > block2 )
      stairlinkingvars[block2].push_back( varToStairlinking );
   else
      stairlinkingvars[block1].push_back( varToStairlinking );

   stairlinkingvarsforblocksorted = false;
   hvoutdated = true;
}


void PARTIALDECOMP::fixVarToStairlinking(
   int var,
   int firstblock
   )
{
   assert( isvaropen[var]);
   assert( var >= 0 && var < nvars );
   assert( firstblock >= 0 && firstblock < ( nblocks - 1 ) );

   setVarToStairlinking(var, firstblock, firstblock + 1);
   deleteOpenvar(var);
}


std::vector<int>::const_iterator PARTIALDECOMP::fixVarToStairlinking(
   std::vector<int>::const_iterator itr,
   int firstblock
)
{
   assert( isvaropen[*itr]);
   assert( itr != openvars.cend() );
   assert( firstblock >= 0 && firstblock < ( nblocks - 1 ) );

   setVarToStairlinking(*itr, firstblock, firstblock + 1);
   return openvars.erase(itr);
}


bool PARTIALDECOMP::fixConsToBlockByName(
   const char*           consname,            /**< name of the constraint */
   int                   blockid              /**< block index (counting from 0) */
   )
{
   int consindex = getDetprobdata()->getIndexForCons(consname);

   if( consindex >= 0 )
   {
      fixConsToBlock(consindex, blockid);
      return true;
   }
   return false;
}


bool PARTIALDECOMP::fixVarToBlockByName(
   const char*           varname,
   int                   blockid
   )
{
   int varindex = getDetprobdata()->getIndexForVar(varname);

   if( varindex >= 0 )
   {
      if( blockid >= nblocks )
         nblocks = blockid + 1;
      fixVarToBlock(varindex, blockid);
      return true;
   }
   return false;
}


bool PARTIALDECOMP::fixConsToMasterByName(
   const char*           consname   /**< name of cons to fix as master cons */
   )
{
   int consindex = getDetprobdata()->getIndexForCons(consname);
   if( consindex >= 0 )
   {
      fixConsToMaster(consindex);
      return true;
   }
   return false;
}


bool PARTIALDECOMP::fixVarToMasterByName(
   const char*           varname
   )
{
   int varindex = getDetprobdata()->getIndexForVar(varname);
   if( varindex >= 0 )
   {
      fixVarToMaster(varindex);
      return true;
   }
   return false;
}


bool PARTIALDECOMP::fixVarToLinkingByName(
   const char*           varname              /**< name of the variable */
   )
{
   int varindex = getDetprobdata()->getIndexForVar(varname);
   if( varindex >= 0 )
   {
      fixVarToLinking(varindex);
      return true;
   }
   return false;
}


void PARTIALDECOMP::showVisualization()
{
   int returnvalue;

   /* get names for gp file and output file */
   char filename[SCIP_MAXSTRLEN];
   char outname[SCIP_MAXSTRLEN];
   GCGgetVisualizationFilename(scip, this, ".gp", filename);
   GCGgetVisualizationFilename(scip, this, ".pdf", outname);

   this->generateVisualization(filename, outname);

   /* open outputfile */
   char command[SCIP_MAXSTRLEN];
   /* command: e.g. evince "outname" && rm "filename" */
   strcpy(command, GCGVisuGetPdfReader(scip));
   strcat(command, " \"");
   strcat(command, outname);
   strcat(command, "\" && rm \"");
   strcat(command, filename);
   strcat(command, "\"");
#ifdef SCIP_DEBUG
   SCIPinfoMessage(scip, NULL, "%s\n", command);
#endif
   returnvalue = system(command);
   if( returnvalue == -1 )
      SCIPwarningMessage(scip, "Unable to open gnuplot file\n");
   SCIPinfoMessage(scip, NULL, "Please note that the generated pdf file was not deleted automatically!  \n");
}

void PARTIALDECOMP::generateVisualization(
   char* filename,
   char* outname,
   GP_OUTPUT_FORMAT outputformat
   )
{
   this->writeVisualizationFile(filename, outname, outputformat);

   int returnvalue;

   /* compile gp file */
   char command[SCIP_MAXSTRLEN];
   /* command: gnuplot "filename" */
   strcpy(command, "gnuplot \"");
   strcat(command, filename);
   strcat(command, "\"");
#ifdef SCIP_DEBUG
   SCIPinfoMessage(scip, NULL, "%s\n", command);
#endif
   returnvalue = system(command);
   if( returnvalue == -1 )
   {
      SCIPwarningMessage(scip, "Unable to write gnuplot file\n");
      return;
   }
}

void PARTIALDECOMP::writeVisualizationFile(
   char* filename,
   char* outname,
   GP_OUTPUT_FORMAT outputformat
   )
{
   /* generate gp file */
   GCGwriteGpVisualizationFormat( scip, filename, outname, getID(), outputformat );
}


void PARTIALDECOMP::exportVisualization()
{
   /* get names for gp file and output file */
   char filename[SCIP_MAXSTRLEN];
   char outname[SCIP_MAXSTRLEN];
   GCGgetVisualizationFilename(scip, this, ".gp", filename);
   GCGgetVisualizationFilename(scip, this, ".pdf", outname);

   /* generate gp file */
   GCGwriteGpVisualization( scip, filename, outname, getID() );
}

SCIP_Bool PARTIALDECOMP::shouldCompletedByConsToMaster()
{
   return usergiven == USERGIVEN::COMPLETED_CONSTOMASTER;
}


bool PARTIALDECOMP::sort()
{
   if( varsforblocksorted && stairlinkingvarsforblocksorted && conssforblocksorted && linkingvarssorted
       && mastervarssorted && masterconsssorted )
   {
      return false;
   }

   for( int b = 0; b < nblocks; ++ b )
   {
      if( !varsforblocksorted )
         std::sort( varsforblocks[b].begin(), varsforblocks[b].end() );
      if( !stairlinkingvarsforblocksorted )
         std::sort( stairlinkingvars[b].begin(), stairlinkingvars[b].end() );
      if( !conssforblocksorted )
         std::sort( conssforblocks[b].begin(), conssforblocks[b].end() );
   }
   if( !linkingvarssorted )
      std::sort( linkingvars.begin(), linkingvars.end() );
   if( !mastervarssorted )
      std::sort( mastervars.begin(), mastervars.end() );
   if( !masterconsssorted )
      std::sort( masterconss.begin(), masterconss.end() );

   varsforblocksorted = true;
   stairlinkingvarsforblocksorted = true;
   conssforblocksorted = true;
   linkingvarssorted = true;
   mastervarssorted = true;
   masterconsssorted = true;

   return true;
}


void PARTIALDECOMP::buildDecChainString(
   char* buffer
   )
{
   /* set detector chain info string */
   SCIPsnprintf( buffer, SCIP_MAXSTRLEN, "" );
   if( this->usergiven == USERGIVEN::PARTIAL || this->usergiven == USERGIVEN::COMPLETE
      || this->usergiven == USERGIVEN::COMPLETED_CONSTOMASTER || this->getDetectorchain().empty() )
   {
      char str1[2] = "\0"; /* gives {\0, \0} */
      str1[0] = 'U';
      (void) strncat( buffer, str1, 1 );
   }

   for( int d = 0; d < this->getNDetectors(); ++ d )
   {
      if( d == 0 && this->getDetectorchain()[d] == NULL )
         continue;
      char str[2] = "\0"; /* gives {\0, \0} */
      str[0] = DECdetectorGetChar( this->getDetectorchain()[d] );
      (void) strncat( buffer, str, 1 );
   }
}


SCIP_Real PARTIALDECOMP::getClassicScore()
{
   return classicscore;
}

void PARTIALDECOMP::setClassicScore(
   SCIP_Real score
   )
{
   classicscore = score;
}


SCIP_Real PARTIALDECOMP::getBorderAreaScore()
{
   return borderareascore;
}

void PARTIALDECOMP::setBorderAreaScore(
   SCIP_Real score
   )
{
   borderareascore = score;
}


SCIP_Real PARTIALDECOMP::getMaxWhiteScore()
{
   return getScore(SCORETYPE::MAX_WHITE);
}

void PARTIALDECOMP::setMaxWhiteScore(
   SCIP_Real score
   )
{
   maxwhitescore = score;
}


SCIP_Real PARTIALDECOMP::getMaxForWhiteScore()
{
   return getScore(scoretype::MAX_FORESSEEING_WHITE);
}

void PARTIALDECOMP::setMaxForWhiteScore(
   SCIP_Real score
   )
{
   maxforeseeingwhitescore = score;
}


SCIP_Real PARTIALDECOMP::getSetPartForWhiteScore()
{
   return setpartfwhitescore;
}

void PARTIALDECOMP::setSetPartForWhiteScore(
   SCIP_Real score
   )
{
   setpartfwhitescore = score;
}


SCIP_Real PARTIALDECOMP::getMaxForWhiteAggScore()
{
   return maxforeseeingwhitescoreagg;
}

void PARTIALDECOMP::setMaxForWhiteAggScore(
   SCIP_Real score
   )
{
   maxforeseeingwhitescoreagg = score;
}


SCIP_Real PARTIALDECOMP::getSetPartForWhiteAggScore()
{
   return setpartfwhitescoreagg;
}

void PARTIALDECOMP::setSetPartForWhiteAggScore(
   SCIP_Real score
   )
{
   setpartfwhitescoreagg = score;
}

SCIP_Real PARTIALDECOMP::getBendersScore()
{
   return bendersscore;
}

void PARTIALDECOMP::setBendersScore(
   SCIP_Real score
   )
{
   bendersscore = score;
}


SCIP_Real PARTIALDECOMP::getStrongDecompScore()
{
   return strongdecompositionscore;
}

void PARTIALDECOMP::setStrongDecompScore(
   SCIP_Real score
   )
{
   strongdecompositionscore = score;
}


void PARTIALDECOMP::prepare()
{
   considerImplicits();
   deleteEmptyBlocks(true);
   calcHashvalue();
}


std::vector<std::vector<int>>& PARTIALDECOMP::getConssForBlocks()
{
   return conssforblocks;
}

bool PARTIALDECOMP::aggInfoCalculated()
{
   return getNBlocks() == 0 || getNReps() > 0;
}

int PARTIALDECOMP::getTranslatedpartialdecid() const
{
   return translatedpartialdecid;
}

void PARTIALDECOMP::setTranslatedpartialdecid(
   int decid
   )
{
   PARTIALDECOMP::translatedpartialdecid = decid;
}

} /* namespace gcg */
