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

/**@file   class_partialdecomp.h
 * @brief  class storing (potentially incomplete) decompositions
 * @note   formerly called "Seeed"
 * @author Michael Bastubbe
 * @author Hannah Hechenrieder
 * @author Hanna Franzen
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_CLASS_PARTIALDECOMP_H__
#define GCG_CLASS_PARTIALDECOMP_H__

#include "objscip/objscip.h"
#include "struct_detector.h"

#include <vector>
#include <string>

#include "class_conspartition.h"
#include "class_varpartition.h"
#include "graph/graph_gcg.h"
#include "graph/graph.h"
#include "type_scoretype.h"

#include "reader_gp.h"

namespace gcg {


/**
 * @brief enumeration to display if a decomposition was given by the user and if so, how it was processed after adding
 */
enum USERGIVEN
{
   NOT = 0,                            /**< this partialdec was not given by the user */
   PARTIAL = - 1,                      /**< this partial partialdec was given by the user as it is*/
   COMPLETE = - 2,                     /**< this complete partialdec was given by the user as it is*/
   COMPLETED_CONSTOMASTER = - 3        /**< this partialdec was partially given by the user and then completed by setting all missing constraints to the master*/
};

class DETPROBDATA;


/*!
 * @brief class to manage partial decompositions
 *
 * each partialdec corresponds to one detprobdata which contains the problem information,
 * there is one detprobdata for the original and the transformed problem.
 */
class PARTIALDECOMP
{
private:
   SCIP* scip;                                                 /**< SCIP data structure */
   int id;                                                     /**< unique id of the partialdec, unique */
   int nblocks;                                                /**< number of blocks the partial decomposition currently has */
   int nvars;                                                  /**< number of variables */
   int nconss;                                                 /**< number of constraints */
   std::vector<int> masterconss;                               /**< vector containing indices of master constraints */
   std::vector<int> mastervars;                                /**< vector containing indices of master variables (these variables are supposed to have all nonzero entries in master constraints) */
   std::vector<std::vector<int>> conssforblocks;               /**< conssforblocks[k] contains a vector of indices of all
                                                                 *< constraints assigned to block k */
   std::vector<std::vector<int>> varsforblocks;                /**< varsforblocks[k] contains a vector of indices of all
                                                                 *< variables assigned to block k */
   std::vector<int> linkingvars;                               /**< vector containing indices of linking variables */
   std::vector<std::vector<int>> stairlinkingvars;             /**< vector containing indices of staircase linking variables
                                                                 *< of the blocks (stair-linking variables are registered only
                                                                 *< in their first block) */
   std::vector<int> openvars;                                  /**< vector containing indices of variables that are not
                                                                 *< assigned yet*/
   std::vector<int> openconss;                                 /**< vector containing indices of constraints that are not
                                                                 *< assigned yet*/
   std::vector<bool> isvaropen;                                /**< whether ith variable is still open */
   std::vector<bool> isconsopen;                               /**< whether ith constraint is still open */
   std::vector<bool> isvarmaster;                              /**< whether ith variable is assigned to be a only-master variable */
   std::vector<bool> isconsmaster;                             /**< whether ith constraint is assigned to be a master constraint */

   std::vector<int>  ncoeffsforblock;                          /**< number of coeffs per block */

   SCIP_Bool         calculatedncoeffsforblock;                /**< is the number of coeff per block already calculated*/
   int               ncoeffsformaster;                         /**< number of master coefficients */
   std::vector<std::vector<int>> ncoeffsforblockformastercons; /**< number of coeffs a block has in a certain master constraint */

   bool varsforblocksorted;                                    /**< bool to store if the varsforblocks datastructure is sorted atm */
   bool stairlinkingvarsforblocksorted;                        /**< bool to store if the stairlinkingvarsforblock datastructure is sorted atm */
   bool conssforblocksorted;                                   /**< bool to store if the conssforblock datastructure is sorted atm */
   bool linkingvarssorted;                                     /**< bool to store if the linkingvars datastructure is sorted atm */
   bool mastervarssorted;                                      /**< bool to store if the mastervars datastructure is sorted atm */
   bool masterconsssorted;                                     /**< bool to store if the masterconsssorted datastructure is sorted atm */

   unsigned long hashvalue;                                    /**< hash value of this partial decomposition, decompositions with same has value are considered to be identical */
   bool hvoutdated;                                            /**< true if the internal structure changed such that the hash value needs to be recalculated */

   bool isselected;                                            /**< is this partialdec selected */

   bool isagginfoalreadytoexpensive;                            /**< is agginfo already known to be to expensive to calculate*/

   const static int primes[];                                   /**< an array of prime numbers to calculate the hash value */
   const static int nprimes;                                    /**< size of the array of prime numbers */

   bool isfinishedbyfinisher;                                   /**< was this partialdec finished by the finishpartialdec() method of a detector */

   /* aggregation information */
   int                  nrepblocks;                                    /**< number of block representatives */
   std::vector<std::vector<int>> reptoblocks;                          /**< translation of the block representatives to (old) blocks */
   std::vector<int>     blockstorep;                                   /**< translation of the (old) blocks to the block representatives */
   std::vector<std::vector<std::vector<int> > > pidtopidvarmaptofirst; /**< [nrepblocks][blockstorep[k].size()][nvarsforprob] collection of varmaps of probindices from k-th subproblem to the zeroth block that is represented */

   /* statistic information */
   std::vector<DEC_DETECTOR*> detectorchain;          /**< vector containing detectors that worked on that partialdec */
   std::vector<std::string> detectorchaininfo;        /**< vector containing information about the detector call */
   std::vector<SCIP_Real> detectorclocktimes;         /**< vector containing detector times in seconds  */
   std::vector<SCIP_Real> pctvarstoborder;            /**< vector containing the fraction of variables assigned to the
                                                        *< border for each detector working on that partialdec*/
   std::vector<SCIP_Real> pctvarstoblock;             /**< vector containing the fraction of variables assigned to a block
                                                        *< for each detector working on that partialdec*/
   std::vector<SCIP_Real> pctvarsfromfree;            /**< vector containing the fraction of variables that are not longer
                                                        *< open for each detector working on that partialdec*/
   std::vector<SCIP_Real> pctconsstoborder;           /**< vector containing the fraction of constraints assigned to the
                                                        *< border for each detector working on that partialdec*/
   std::vector<SCIP_Real> pctconsstoblock;            /**< vector containing the fraction of constraints assigned to a block
                                                        *< for each detector working on that partialdec*/
   std::vector<SCIP_Real> pctconssfromfree;           /**< vector containing the fraction of constraints that are not longer
                                                         *< open for each detector working on that partialdec*/
   std::vector<int> nnewblocks;                       /**< vector containing information how many new blocks a detector has assigned */

   std::vector<IndexPartition*> usedpartition;      /**< vector containing pointer to the cons- or varpartitions
                                                         *< a detector made use of for each detector working on that partialdec
                                                         *< (NULL if no partition was used) */
   std::vector<std::vector<int>> classestomaster;     /**< vector containing the vector of classindices that were assigned
                                                         *< to master by the partition used by a detector
                                                         *< (empty vector if no partition was used) */
   std::vector<std::vector<int>> classestolinking;    /**< vector containing the vector of classindices that were assigned
                                                         *< to linking by the partition used by a detector
                                                         *< (empty vector if no partition was used) */

   std::vector<int> listofancestorids;                /**< vector containing decomposition indices that are ancestors of this partialdec */

   USERGIVEN usergiven;                               /**< is this partialdec partially or completely given by user */

   /* score values (or -1 iff nott computed yet) */

   SCIP_Real maxwhitescore;                           /**< score corresponding to the max white measure */
   SCIP_Real borderareascore;                         /**< 1 - fraction of border area to complete area */
   SCIP_Real classicscore;                            /**< classic score to evaluate the partial */
   SCIP_Real maxforeseeingwhitescore;                 /**< maximum foreseeing white area score (i.e. maximize fraction of white area score considering problem with copied linking variables and corresponding master constraints; white area is nonblock and nonborder area, stairlinking variables count as linking) */
   SCIP_Real setpartfwhitescore;                      /**< setpartitioning maximum foreseeing white area score (i.e. convex combination of maximum foreseeing white area score and a boolean score rewarding a master containing only setppc and cardinality constraints )*/
   SCIP_Real maxforeseeingwhitescoreagg;              /**< maximum foreseeing white area score with respect to aggregatable blocks (i.e. maximize fraction of white area score considering problem with copied linking variables and corresponding master constraints; white area is nonblock and nonborder area, stairlinking variables count as linking) */
   SCIP_Real setpartfwhitescoreagg;                   /**< setpartitioning maximum foreseeing white area score with respect to aggregateable (i.e. convex combination of maximum foreseeing white area score and a boolean score rewarding a master containing only setppc and cardinality constraints )*/
   SCIP_Real bendersscore;                            /**< score to evaluate the partialdecs */
   SCIP_Real strongdecompositionscore;                /**< strong decomposition score  */

   /* datastructure to store information if this partialdec stems from a partialdec concerning the orig problem */
   bool stemsfromorig;                    /**< partialdec has at least one ancestor that is a partialdec from orig problem */
   bool original;                        /**< indicates whether partialdec is from original problem */
   bool isfinishedbyfinisherorig;         /**< was the ancestor partialdec for the unpresolved problem finished by the
                                            *< finishpartialdec() method of a detector */
   DEC_DETECTOR* finishedorigby;          /**< index of finishing detector of orig ancestor partialdec */

   int translatedpartialdecid;

private:
   /**< id of the translated partialdec */

   /**
    *
    * @brief simple bliss automorphism check for blocks
    *
    * Checks blocks for equality by graph automorphism check, done by bliss
    * @note equality is only found if variables are in correct order
    */
   void checkIdenticalBlocksBliss(
      int                  b1,         /**< block id of first block */
      int                  b2,         /**< block id of second block */
      std::vector<int>&    varmap,     /**< maps variable indices (corresponding to  detprobdata indices) of block 2 to block 1 */
      SCIP_HASHMAP*        varmap2,    /**< maps variable pointers of block 2 to those of block 1 if both blocks (problems) are identical*/
      SCIP_Bool*           identical,  /**< pointer to store if the subproblems are identical  */
      unsigned int         searchnodelimit,    /**< bliss search node limit (requires patched bliss version) */
      unsigned int         generatorlimit      /**< bliss generator limit (requires patched bliss version) */

   );


   /**
    * @brief brute force equality check for blocks
    *
    * Checks blocks for equality by brute force
    * @note equality is only found if variables are in correct order
    */
   void checkIdenticalBlocksBrute(
      int                  b1,         /**< block id of first block */
      int                  b2,         /**< block id of second block */
      std::vector<int>&    varmap,     /**< maps variable indices (corresponding to  detprobdata indices) of prob2 to prob1 */
      SCIP_HASHMAP*        varmap2,    /**< maps variable pointers of block 2 to those of block 1 if both blocks (problems) are identical*/
      SCIP_Bool*           identical   /**< pointer to store if the subproblems are identical  */
      );


   /**
    * @brief plausibility check whether two blocks could be identical
    *
    * Check some necessary conditions for two blocks to be identical
    */
   void checkIdenticalBlocksTrivial(
      int                  b1,                     /**< block id of first block */
      int                  b2,                     /**< block id of second block */
      SCIP_Bool*           notidentical            /**< pointer to store whether or not the non-equality is proven */
      );

public:

   /**
    * @brief Standard constructor, creates empty partialdec with unique id
    * @note initially, all conss and vars are open
    */
   PARTIALDECOMP(
      SCIP* scip,                                  /**< scip data structure */
      bool originalProblem                             /**< true iff partialdec is for presolved problem (else for original problem) */
      );

   /**
    * @brief copy constructor
    */
   PARTIALDECOMP(
      const PARTIALDECOMP *partialdecToCopy /**< partialdec to be copied */
      );

   /**
    * Standard destructor
    */
   ~PARTIALDECOMP();

    /**
     * @brief adds a block
     * @returns the number (id) of the new block
     * */
   int addBlock();

   /**
    * @brief adds detection time of one detector
    *
    * incorporates the needed time of some detector in the detector chain
    */
   void addClockTime(
      SCIP_Real clocktime /**< time to be added */
      );

   /**
    * @brief adds the statistical differences to an ancestor
    *
    * incorporates the changes from ancestor partialdec into the statistical data structures
    */
   void addDecChangesFromAncestor(
      PARTIALDECOMP* ancestor    /**< partialdec whose propagation yielded to the current partialdec */
      );

   /**
    * @brief add information about the detector chain
    *
    * adds a detectorchain information string to the corresponding vector
    * (that carries information for each detector call)
    * */
   void addDetectorChainInfo(
      const char* decinfo              /**< information string (about the detector call) to add  */
      );

   /**
    * @brief adds how many new blocks were introduced
    *
    * bookkeeping information: adds number of new blocks created by a detector added to detector chain
    */
   void addNNewBlocks(
      int nnewblocks                   /**< number of new added blocks by latest detector call */
      );

   /**
    * @brief adds percentage of closed constraints
    *
    * bookkeeping information: fraction of constraints that are not longer open for a detector added to detector chain
    */
   void addPctConssFromFree(
      SCIP_Real pct                    /**< fraction of constraints that are not longer open */
      );

   /**
    * @brief adds percentage of constraints assigned to blocks
    *
    * bookkeeping information: adds fraction of constraints assigned to a block for a detector added to detector chain
    *  */
   void addPctConssToBlock(
      SCIP_Real pct                    /**< fraction of constraints assigned to a block */
      );

   /**
    * @brief adds percentage of constraints assigned to border
    *
    * bookkeeping information: adds fraction of constraints assigned to the border for a detector added to detector chain
    */
   void addPctConssToBorder(
      SCIP_Real pct                    /**< fraction constraints assigned to the border */
      );

   /**
    *  @brief adds percentage of closed variables
    *
    *  bookkeeping information: adds fraction of variables that are not longer open for a detector added to detector chain
    */
   void addPctVarsFromFree(
      SCIP_Real pct                    /**< fraction of variables that are not longer open */
      );


   /**
    *  @brief adds percentage of variables assigned to blocks
    *
    *  bookkeeping information: adds fraction of variables assigned to a block for a detector added to detector chain
    *  */
   void addPctVarsToBlock(
      SCIP_Real pct                     /**< fraction of variables assigned to a block */
      );

   /**
    * @brief adds percentage of variables assigned to border
    *
    * bookkeeping information: adds fraction of variables assigned to the border for a detector added to detector chain
    */
   void addPctVarsToBorder(
      SCIP_Real pct                    /**< fraction of variables assigned to the border */
      );

   /**
    * @brief method to check if at least one constraint is assigned to some block
    * @returns true iff at least one constraint is assigned to a block
    *  */
   bool alreadyAssignedConssToBlocks();

   /**
    * @brief assigns open conss to master
    *
    * assigns open constraints to master according to the cons assignment information given in constoblock hashmap
    * @returns scip return code
    * @note for conss assigned to blocks according to constoblock there is no assignment \see assignPartialdecFromConstoblock
    * @note master assignment is indicated by assigning cons to index additionalNBlocks
    * */
   SCIP_RETCODE assignBorderFromConstoblock(
      SCIP_HASHMAP* constoblock, /**< hashmap assigning cons indices (not SCIP_Cons*) to block indices */
      int givenNBlocks           /**< number of blocks the hashmap contains */
       );

   /**
    * @brief assigns open vars to stairlinking if appropriate
    *
    * assigns open vars to stairlinking if they can be found in exactly two consecutive blocks
    * @returns true iff at least one stairlinkingvar was assigned
    */
   bool assignCurrentStairlinking(
      );

   /**
    * @brief assigns open conss to master
    */
   void assignOpenConssToMaster(
      );

   /**
    * @brief assigns conss structure according to given hashmap
    *
    *  adds blocks and assigns open conss to a new block or to master
    *  according to the cons assignment information given in constoblock hashmap
    *  @returns scip return code
    *  \see assignPartialdecFromConstoblockVector()
    *  @note master assignment is indicated by assigning cons to index additionalNBlocks
    *  */
   SCIP_RETCODE assignPartialdecFromConstoblock(
      SCIP_HASHMAP* constoblock, /**< hashmap assigning cons indices (not SCIP_Cons*) to block indices */
      int additionalNBlocks      /**< number of (additional) blocks the hashmap contains */
       );

   /*!
    * @brief assigns conss structure according to given vector
    *
    *  adds blocks and assigns open conss to a new block or to master
    *  according to the cons assignment information given in constoblock vector
    *  @returns scip return code
    *  \see  assignPartialdecFromConstoblock()
    *  @note master is indicated by assigning cons to index additionalNBlocks */
   SCIP_RETCODE assignPartialdecFromConstoblockVector(
      std::vector<int> constoblock, /**< vector containing an assignment of conss to a block or to master */
      int additionalNBlocks         /**< number of (additional) blocks the vector contains */
      );

   /**
    * @brief computes components by connectedness of conss and vars
    *
    * computes components corresponding to connectedness of conss and vars
    * and assigns them accordingly (all but one of largest components)
    *
    * strategy: assigns all conss same block if they are connected
    * two constraints are adjacent if there is a common variable
    *
    * @note this relies on the consadjacency structure of the detprobdata
    *  hence it cannot be applied in presence of linking variables
    */
   void assignSmallestComponentsButOneConssAdjacency(
      );

   /**
    * @brief reassigns linking vars to stairlinkingvars if possible
    *
    *  potentially reorders blocks for making a maximum number of linking vars stairlinking
    *  if all vars that connect exactly two blocks have a staircase structure, all of them become stairlinkingvars
    *  otherwise, the stairlinking assignment is done greedily
    *  @note precondition: partialdec does not have any stairlinking vars
    */
   void calcStairlinkingVars(
        );

   /**
    * @brief checks if all conss are assigned
    *
    * returns true iff all constraints are assigned and deletes the vector open conss if so
    * @return true iff all constraints are assigned
    * */
   bool checkAllConssAssigned();

   /**
    * @brief Checks whether the assignments in the partialdec are consistent
    *
    * The following checks are performed:
    * - check if nblocks is set appropriately
    * - check for empty (row- and col-wise) blocks
    * - every variable is assigned at most once
    * - check if all not assigned variables are open vars
    * - check if all open vars are not assigned
    * - every constraint is assigned at most once
    * - check if all not assigned constraints are open cons
    * - check if all open conss are not assigned
    * - check if the data structures are sorted
    * - check if variables hitting a cons are either in the cons's block or border or still open
    * @returns true iff the partialdec seems to be consistent
    * */
   bool checkConsistency(
      );

   /**
    * @brief assigns all open constraints and open variables trivially
    *
    *  strategy: assigns all open conss and vars to blocks if they can be refined there, otherwise to the master
    *
    *  @note partialdecomps should usually be completed by a detector, only use this function if you know what you are doing.
    */
   void complete(
      );

   /**
    * @brief assigns all open constraints and open variables
    *
    *  strategy: assigns all conss and vars to the same block if they are connected,
    *  a cons and a var are adjacent if the var appears in the cons
    */
   void completeByConnected(
      );

   /**
    * @brief assigns all open constraints and open variables
    *
    *  strategy: assigns all conss and vars to the same block if they are connected
    *  a cons and a var are adjacent if the var appears in the cons
    *  \note this relies on the consadjacency structure of the detprobdata
    *  hence it cannot be applied in presence of linking variables
    */
   void completeByConnectedConssAdjacency(
      );

   /**
    * @brief assigns all open constraints and open variables
    *
    *  strategy: assigns a cons (and related vars) to a new block if possible,
    *  if not to an existing block if possible (by means of prior var assignments)
    *  and finally to master, if there does not exist such a block
    */
   void completeGreedily(
      );

   /** @brief removes the given cons from master
    */
   void removeMastercons(
      int consid      /**< id of cons */
      );

   /**
    * @brief: assigns every open cons/var
    *
    * Assignments happen as follows:
    *  - to the respective block if it hits exactly one blockvar/blockcons and no open vars/conss
    *  - to master/linking if it hits blockvars/blockconss assigned to different blocks
    *  - and every cons to master that hits a master var
    *  - and every var to master if it does not hit any blockcons and has no open cons
    *  - leave the cons/variableopen if nothing from the above holds
    *  */
   void considerImplicits(
      );

   /**
    * @brief copies the given partialdec's partition statistics
    *
    * @param otherpartialdec partialdec whose partition statistics are to be copied
    */
   void copyPartitionStatistics(
      const PARTIALDECOMP* otherpartialdec
      );

   /**
    * @brief deletes empty blocks and sets nblocks accordingly
    *
    *  A block is considered to be empty if no constraint is assigned to it,
    *  variables in blocks with no constraints become open
    *
    * @param variables if true, then blocks with no constraints but at least one variable are considered to be nonempty
    */
   void deleteEmptyBlocks(
      bool variables
   );

   /**
    * @brief deletes a cons from list of open conss
    *
    * @param opencons id of the cons that is not considered open anymore
    */
   void deleteOpencons(
      int opencons
      );

   /**
    * @brief deletes a cons from list of open conss
    *
    * @param itr valid iterator pointing to elements of PARTIALDECOMP::openconss
    */
   std::vector<int>::const_iterator deleteOpencons(
      std::vector<int>::const_iterator itr
   );

   /**
    * @brief deletes a var from the list of open vars
    *
    * @param openvar id of the var that is not considered open anymore
    */
   void deleteOpenvar(
      int openvar
      );

   /**
    * @brief deletes a var from the list of open vars
    *
    * @param itr valid iterator pointing to elements of PARTIALDECOMP::openvars
    */
   std::vector<int>::const_iterator deleteOpenvar(
      std::vector<int>::const_iterator itr
   );

   /**
    * @brief displays the relevant information of the partialdec
    *
    * @param detailLevel pass a value that indicates how detailed the output should be:
    *                         0: brief overview
    *                         1: block and detector info
    *                         2: cons and var assignments
    */
   void displayInfo(
      int detailLevel
      );

   /**
    * @brief every constraint is either assigned to master or open
    *
    *  Assignment happens according to the cons assignment information given in constoblock hashmap,
    *  variables are set accordingly
    * @note precondition: no constraint or variable is already assigned to a block
    * @return scip return code
    */
   SCIP_RETCODE filloutBorderFromConstoblock(
      SCIP_HASHMAP* constoblock, /**< hashmap assigning cons indices (not SCIP_Cons*) to block indices
                                   *< (master assignment is indicated by assigning cons to index additionalNBlocks) */
      int givenNBlocks          /**< number of blocks the hashmap contains */
      );

   /**
    * @brief assigns all conss to master or a block
    *
    *  Assignment happens according to the cons assignment information given in constoblock hashmap
    *
    * @return scip return code
    *  calculates implicit variable assignment through cons assignment
    * @note precondition: no cons or var is already assigned to a block and constoblock contains information for every cons */

   SCIP_RETCODE filloutPartialdecFromConstoblock(
      SCIP_HASHMAP* constoblock, /**< hashmap assigning cons indices (not SCIP_Cons*) to block indices
                                   *< (master assignment is indicated by assigning cons to index additionalNBlocks) */
      int givenNBlocks          /**< number of blocks the hashmap contains */
      );

   /**
    * @brief reassigns linking variables to master if appropriate
    *
    * Variables are reassigned as master if the variable only hits master conss
    */
   void findVarsLinkingToMaster(
      );

   /**
    * @brief reassigns variables classified as linking to stairlinking if appropriate
    *
    * Variables are reassigned as master if the variable hits conss in exactly two consecutive
    * blocks
    */
   void findVarsLinkingToStairlinking(
      );

   /**
    * @brief gets partialdec id of given ancestor id
    * @return partialdec id of given ancestor id
    */
   int getAncestorID(
      int ancestorindex /**< index of ancestor in list of ancestor ids data structure */
      );

   /**
    * @brief get ancestor ids as vector
    * @return vector of ids of all ancestors id
    */
   std::vector<int>& getAncestorList(
      );

   /**
    * set ancestor list directly
    * @param newlist new list of ancestor ids
    */
   void setAncestorList(
      std::vector<int>& newlist
      );

   /** removes ancestor id from list */
   void removeAncestorID(
      int ancestorid    /**< id to remove */
   );

   /**
    * adds ancestor id to back of list
    * @param ancestor id of ancestor that is to be added
    */
   void addAncestorID(
      int ancestor
      );

   /**
    * @brief get a vector of block ids that are identical to block with id repid
    * @param repid id of the representative block
    * @return vector of block ids that are identical to block with id repid
    */
   const std::vector<int> & getBlocksForRep(
      int  repid
      );

   /**
    * @brief returns the time that the detector related to the given detectorchainindex needed for detecting
    * @return the clock time for the corresponding detector in the chain
    */
   SCIP_Real getDetectorClockTime(
      int detectorchainindex /**< index of the detector in the detectorchain */
      );

   /**
    * @brief returns a vector of the clock times that each detector needed that was involved in this partialdec
    * @return vector of the clock times
    */
   std::vector<SCIP_Real>& getDetectorClockTimes();

   /**
    * @brief returns array containing constraints assigned to a block
    * @param block id of the block the constraint indices are returned
    * @return array containing constraints assigned to a block
    */
   std::vector<int>& getConssForBlock(
      int block
      );

   /**
    * @brief returns detector chain as vector of detector pointers
    * @return detector chain as array of detector pointers
    */
   std::vector<DEC_DETECTOR*>& getDetectorchain();

   /**
    * @brief returns true iff this partialdec was finished by finishPartialdec() method of a detector
    * @return true iff this partialdec was finished by finishPartialdec() method of a detector
    */
   bool getFinishedByFinisher();

   /**
    * @brief returns the calculated hash value of this partialdec
    * @return the calculated hash value of this partialdec
    */
   unsigned long getHashValue();

   /**
    * @brief returns the unique id of the partialdec
    * @return the unique id of the partialdec
    */
   int getID();

   /**
    * @brief returns array containing all linking vars indices
    * @return vector containing all linking vars indices
    * @note when accessed it is supposed to be sorted
    */
   std::vector<int>& getLinkingvars();

   /**
    * @brief Gets array containing all master conss indices
    * @return array containing all master conss indices
    * @note when accessed it is supposed to be sorted
    */
   std::vector<int>& getMasterconss();

   /**
    * @brief Gets array containing all master vars indices
    *
    * master vars hit only constraints in the master, aka static variables
    * @return array containing all master vars indices
    */
   std::vector<int>& getMastervars();

   /**
    * @brief Gets the number of nonzero coeffs in a certain block
    * @param blockid of the block the number of nozerors are requested for
    * @return number of nonzero coeffs in a certain block
    */
   int getNCoeffsForBlock(
      int blockid
      );

   /**
    * Gets the number of nonzero coeffs in master
    * @return the number of nonzero coeffs in master
    */
   int getNCoeffsForMaster(
      );

   /**
    * @brief returns the score of the partialdec (depending on used scoretype)
    * @param type the scoretype
    * @return the score
    * @see enum scoretype in cons_decomp.h
    */
   SCIP_Real getScore(
      SCORETYPE type
      );

   /**
    * @brief checks if all master constraints set partitioning, set packing, set cover, or cardinality constraints
    * @return TRUE iff all master constraints set partitioning, set packing, set cover, or cardinality constraints
    */
   SCIP_Bool hasSetppccardMaster(
   );

   /**
    * @brief checks iff all master constraints set partitioning, set packing, or set cover constraints
    * @return TRUE iff all master constraints set partitioning, set packing, or set cover
    */
   SCIP_Bool hasSetppcMaster(
   );

   /**
    * @brief checks iff all master constraints set partitioning, or set packing constraints
    * @return TRUE iff all master constraints set partitioning, or set packing constraints
    */
   SCIP_Bool hasSetppMaster(
   );

   /**
    * @brief Gets the USERGIVEN status of this partialdecs
    * @return the USERGIVEN status of this partialdecs
    * @see enum USERGIVEN
    */
   USERGIVEN getUsergiven();

   /**
    * @brief Gets number of ancestor partialdecs
    * @return number of ancestor partialdecs
    */
   int getNAncestors();

   /**
    * @brief Gets the number of blocks
    * @return number of blocks
    */
   int getNBlocks();

   /**
    * @brief Gets the number of constraints
    * @return number of constraints
    */
   int getNConss();

   /**
    * @brief Gets size of the vector containing conss assigned to a block
    * @param block id of the block the number of constraints is asked for
    * @return size of the vector containing conss assigned to a block
    */
   int getNConssForBlock(
      int block
      );

   /**
    * @brief Gets the detectorchain info vector
    * @return detectorchain info vector
    */
   std::vector<std::string>& getDetectorchainInfo();

   /**
    * @brief Gets the number of detectors the partialdec is propagated by
    * @return number of detectors the partialdec is propagated by
    */
   int getNDetectors();

   /**
    * @brief Gets size of the vector containing linking vars
    * @return size of the vector containing linking vars
    */
   int getNLinkingvars();

   /**
    * @brief Gets size of the vector containing master conss
    * @returns size of the vector containing master conss
    */
   int getNMasterconss();

   /**
    * @brief Gets size of the vector containing master vars
    *
    * master vars hit only constraints in the master
    * @return size of the vector containing master vars
    */
   int getNMastervars();

   /**
    * @brief Gets number of blocks a detector added
    *
    * @return number of blocks a detector added
    */
   int getNNewBlocks(
      int detectorchainindex /**< index of the detector in the detectorchain */
      );

   /**
    * @brief gets number of blocks the detectors in the detectorchain added
    * @return number of blocks the detectors in the detectorchain added
    */
   std::vector<int> getNNewBlocksVector();

   /**
    * @brief Gets total number of stairlinking vars
    * @return total number of stairlinking vars
    */
   int getNTotalStairlinkingvars();

   /**
    * @brief Gets size of vector containing constraints not assigned yet
    * @return returns size of vector containing constraints not assigned yet
    */
   int getNOpenconss();

   /**
    * @brief Gets size of vector containing variables not assigned yet
    * @return size of vector containing variables not assigned yet
    */
   int getNOpenvars();

   /**
    * @brief Gets the number of blockrepresentatives
    * @return the number of blockrepresentatives
    */
   int getNReps();

   /**
    * @brief Gets size of the vector containing stairlinking vars
    * @param block id of the block the size of the stairlinking vector is asked for
    * @return size of the vector containing stairlinking vars
    */
   int getNStairlinkingvars(
      int block
      );

   /**
    * @brief Gets number of vars
    * @return number of vars
    */
   int getNVars();

   /**
    * @brief Gets size of the vector containing vars assigned to a block
    * @param block id of the block the number of variables is asked for
    * @return size of the vector containing vars assigned to a block
    */
   int getNVarsForBlock(
      int block
      );

   /**
    * @brief Gets overall number of vars assigned to a block
    * @return number of vars that are assigned to any block
    */
   int getNVarsForBlocks(
      );

   /**
    * @brief Gets array containing constraints not assigned yet
    * @return array containing constraints not assigned yet
    */
   const int* getOpenconss();

   /**
    * @brief Gets a vector containing constraint ids not assigned yet as vector
    * @return returns a vector containing constraint ids not assigned yet as vector
    */
   std::vector<int>& getOpenconssVec();

   /**
    * @brief Gets array containing variables not assigned yet
    * @return returns array containing variables not assigned yet
    */
   const int* getOpenvars();

   /**
    * Gets array containing variables not assigned yet as vector
    * @return array containing variables not assigned yet as vector
    */
   std::vector<int>& getOpenvarsVec();

   /**
    * @brief Gets fraction of variables assigned to the border for a detector
    *
    * @return fraction of variables assigned to the border for a detector
    */
   SCIP_Real getPctVarsToBorder(
      int detectorchainindex /**< index of the detector in the detectorchain */
      );

   /**
    * @brief Gets fraction of variables assigned to the border for detectors in detectorchain
    * @return vector of fractions of variables assigned to the border for detectors in detectorchain
    */
   std::vector<SCIP_Real>& getPctVarsToBorderVector();

   /**
    * @brief Gets fraction of variables assigned to a block for a detector
    *
    * @return fraction of variables assigned to a block for a detector
    */
   SCIP_Real getPctVarsToBlock(
      int detectorchainindex /**< index of the detector in the detectorchain */
      );

   /**
    * @brief returns fraction of variables assigned to a block for detectors in detectorchain
    * @return vector of fractions of variables assigned to a block for detectors in detectorchain
    */
   std::vector<SCIP_Real>& getPctVarsToBlockVector();

   /**
    * @brief Gets fraction of variables that are not longer open for a detector
    *
    * @return index of the detector in the detectorchain
    */
   SCIP_Real getPctVarsFromFree(
      int detectorchainindex /**< index of the detector in the detectorchain */
      );

   /**
    * @brief Gets fraction of variables that are not longer open for detectors in detectorchain
    * @return vector or fractions of variables that are not longer open for detectors in detectorchain
    */
   std::vector<SCIP_Real>& getPctVarsFromFreeVector();

   /**
    * @brief Gets fraction of constraints assigned to the border for a detector
    * @return returns fraction of constraints assigned to the border for a detector
    */
   /**  */
   SCIP_Real getPctConssToBorder(
      int detectorchainindex /**< index of the detector in the detectorchain */
      );

   /**
    * @brief Gets fraction of constraints assigned to the border for detectors in detectorchain
    * @return vector of fractions of constraints assigned to the border for detectors in detectorchain
    */
   std::vector<SCIP_Real>& getPctConssToBorderVector();

   /**
    * @brief Gets fraction of constraints assigned to a block for a detector
    * @return fraction of constraints assigned to a block for a detector
    */
   SCIP_Real getPctConssToBlock(
      int detectorchainindex /**< index of the detector in the detectorchain */
      );

   /**
    * @brief Gets fraction of constraints assigned to a block for detectors in detectorchain
    * @return vector of fractions of constraints assigned to a block for detectors in detectorchain
    */
   std::vector<SCIP_Real>& getPctConssToBlockVector();

   /**
    * @brief Gets fraction of constraints that are not longer open for a detector
    * @return fraction of constraints that are not longer open for a detector
    */
   SCIP_Real getPctConssFromFree(
      int detectorchainindex /**< index of the detector in the detectorchain */
      );

   /**
    * @brief Gets fraction of constraints that are not longer open for detectors in detectorchain
    * @return vector of fractions of constraints that are not longer open for detectors in detectorchain
    */
   std::vector<SCIP_Real>& getPctConssFromFreeVector();

   /**
    * @brief Gets index of the representative block for a block, this might be blockid itself
    * @param blockid id of the block the representative is asked for
    * @return index of the representative block for a block, this might be blockid itself
    */
   int getRepForBlock(
      int blockid
      );

   /**
    * @brief Gets the represenation varmap
    *
    * Var map is vector for represenative repid and the blockrepid-th block that is represented by repid
    * @param repid id of representative
    * @param blockrepid id of block
    * @return the represenation varmap as vector for represenative repid and the blockrepid-th block that is represented by repid
    */

   std::vector<int> & getRepVarmap(
      int repid,
      int blockrepid
      );

   /**
    * @brief Gets the corresponding detprobdata
    * @return corresponding detprobdata
    */
   DETPROBDATA* getDetprobdata();

   /**
    * @brief Gets array containing stairlinking vars,
    * @note if a stairlinking variable links block i and i+1 it is only stored in vector of block i
    * @param block id of the block the stairlinking variable varctor is asked for
    * @return array containing stairlinking vars,
    */
   const int* getStairlinkingvars(
      int block
      );

   /**
    * @brief Gets array containing vars of a block
    * @param block id of the block the vars are requested for
    * @return returns array containing vars of a block
    */
   std::vector<int>& getVarsForBlock(
      int block
      );

   /**
    * @brief  Gets index in variables array of a block for a variable
    * @param varid the id of the variable the index
    * @param block the corresponding block id
    * @return  returns index in variables array of a block for a variable
    */
   int getVarProbindexForBlock(
      int varid,
      int block
   );

   /**
    * @brief Gets whether this partialdec is complete,
    *  i.e. it has no more open constraints and variables
    * @return TRUE iff this partialdec is complete
    */
   bool isComplete();

   /**
    * @brief Gets whether the cons is a master cons
    * @param cons id of ccons to check if it is master constraint
    * @return true iff the cons is a master cons
    */
   bool isConsMastercons(
      int cons
      );

   /**
    * @brief Gets whether the cons is an open cons
    * @param cons id of cons to check
    * @return true iff the cons is an open cons
    */
   bool isConsOpencons(
      int cons
      );

   /**
    * @brief Gets whether the partialdec is from the presolved problem
    * @return true iff the partialdec is from the presolved problem
    */
   bool isAssignedToOrigProb();

   /**
    * Gets whether the partialdec is currently selected in explore menue
    * @return true iff the partialdec is currently selected in explore menue
    */
   bool isSelected();

   /**
    * @brief method to check whether this partialdec is equal to a given other partialdec ( \see  isEqual(PARTIALDECOMP*))

    * @return scip return code
    */
   SCIP_RETCODE isEqual(
      PARTIALDECOMP* otherpartialdec,   /**< other partialdec */
      SCIP_Bool* isequal,  /**< pointer to store whether partialdecs are identical */
      bool sortpartialdecs      /**< should conss and vars be sorted before comparing the partialdecs? */
      );

   /**
    * @brief method to check whether this partialdec is equal to a given other partialdec
    *
    * @return true iff partialdecs are equal
    */
   bool isEqual(
      PARTIALDECOMP* other /**< other partialdec to check equality with */
      );

   /**
    * @brief Gets whether this partialdec was propagated by specified detector
    * @param detector pointer to detector to check for
    * @return true iff this partialdec was propagated by detectorID
    */
   bool isPropagatedBy(
      DEC_DETECTOR* detector
      );

   /**
    * @brief Gets whether this partialdec is considered to be trivial
    *
    *  PARTIALDECOMP is considered trivial if all conss are in one block, all conss are in border,
    *  all variables linking or mastervars, or all constraints and variables are open
    * @return true iff this partialdec is considered to be trivial
    */
   bool isTrivial();

   /**
    * @brief Checks whether the var is assigned to the block
    * @param var id of var to check
    * @param block id of block to check
    * @return true iff the var is assigned to the block
    */
   bool isVarBlockvarOfBlock(
      int var,
      int block
      );

   /**
    * @brief Checks whether the var is a linking var
    * @param var id of var to check
    * @return true iff the var is a linking var
    */
   bool isVarLinkingvar(
      int var
      );

   /**
    * @brief Checks whether the var is a master var
    * @param var id of var to check
    * @return true iff the var is a master var
    */
   bool isVarMastervar(
      int var
      );

   /**
    * @brief Checks whether the var is an open var
    * @param var id of var to check
    * @return true iff the var is an open var
    */
   /**  */
   bool isVarOpenvar(
      int var
      );

   /**
    * @brief Checks whether the var is a stairlinking var
    * @param var id of var to check
    * @return true iff the var is a stairlinking var
    */
   bool isVarStairlinkingvar(
      int var
      );

   /**
    * @brief Checks whether the var is a stairlinkingvar of a specified block
    * @param var id of var to check if it is a stairlinking variable hitting specified block
    * @param block id of block to check
    * @return true iff the var is a stairlinkingvar of a specified block
    */
   bool isVarStairlinkingvarOfBlock(
      int var,
      int block
      );

   /**
    * @brief prints partition information as described in \see cls reader
    * @param givenscip scip data structure
    * @param file output file
    */
   void printPartitionInformation(
      SCIP*                givenscip,
      FILE*                file
      );

   /**
    * @brief refine partialdec with focus on blocks
    *
    * strategy: assigns open conss and vars if they can be found in blocks
    * (without respect to open vars and conss  @see assignHittingOpenconss(), @see assignHittingOpenvars())
    * @note partialdec might be not complete
    */
   void refineToBlocks(
      );

   /**
    * @brief refine partialdec with focus on master
    *
    * strategy: do obvious ( @see considerImplicits()) assignments and
    *  assign other conss and vars to master if possible (@see assignOpenPartialHittingToMaster())
    */
   void refineToMaster(
      );

   /**
    * @brief registers statistics for a used conspartition
    */
   void setConsPartitionStatistics(
      int detectorchainindex, /**< index of the detector in the detectorchain */
      ConsPartition* partition, /**< the used conspartition */
      std::vector<int>& consclassesmaster /**< vector of classindices that were assigned to master */
      );

   /**
    * @brief adds a constraint to a block, does not delete this cons from list of open conss
    * @param consToBlock id of cons to add
    * @param block id of block to add
    */
   void setConsToBlock(
      int consToBlock,
      int block
      );

   /**
    * @brief adds a constraint to a block
    * @param cons id of cons to add
    * @param block id of block to add
    */
   void fixConsToBlock(
      int cons,
      int block
      );

   /**
    * @brief adds a constraint to a block
    * @param cons pointer of cons to add
    * @param block id of block to add
    * @returns true iff successful
    */
   bool fixConsToBlock(
      SCIP_CONS* cons,
      int block
      );

   /**
    * @brief adds a constraint to the master constraints, does not delete this cons from list of open conss
    * @param consToMaster id of cons to add
    */
   void setConsToMaster(
      int consToMaster
      );

   /**
    * @brief fixes a constraint to the master constraints
    * @param itr valid iterator pointing to elements of PARTIALDECOMP::openconss
    * @return iterator that points to the next element of PARTIALDECOMP::openconss
    * @warning This method modifies the vector PARTIALDECOMP::openconss! Hence, any kind of iterator might be invalid afterwards!
    */
   std::vector<int>::const_iterator fixConsToMaster(
         std::vector<int>::const_iterator itr
   );

   /**
    * @brief fixes a constraint to the master constraints
    * @param cons id of cons to add
    * @warning This method modifies the vector PARTIALDECOMP::openconss! Hence, any kind of iterator might be invalid afterwards!
    */
   void fixConsToMaster(
      int cons
      );

   /**
    * @brief fixes a constraint to the master constraints
    * @param cons pointer of cons to add
    * @warning This method modifies the vector PARTIALDECOMP::openconss! Hence, any kind of iterator might be invalid afterwards!
    * @returns true iff successful
    */
   bool fixConsToMaster(
      SCIP_CONS* cons
      );

   /**
    * @brief sets the detectorchain with the given vector of detector pointers
    * @param givenDetectorChain vector of detector pointers
    */
   void setDetectorchain(
      std::vector<DEC_DETECTOR*>& givenDetectorChain
      );

   /**
    * @brief sets partialdec to be propagated by a detector
    * @param detector pointer to detector that is registered for this partialdec
    */
   void setDetectorPropagated(
      DEC_DETECTOR* detector
      );

   /**
    * @brief sets detector that finished the partialdec
    * @param detector pointer to detector that has finished this partialdecs
    */
   void setDetectorFinished(
      DEC_DETECTOR* detector
      );

   /**
    * @brief sets detector that finished the partialdec in the original problem
    * @param detectorID pointer to detector that has finished this partialdecs
    * @note does not add the detector to the detectorchain and does not modify partition statistics
    */
   void setDetectorFinishedOrig(
      DEC_DETECTOR* detectorID
      );

   /**
    * @brief sets whether this partialdec was finished by a finishing detector
    * @param finished is this partialdecs finished by a finishing detector
    */
   void setFinishedByFinisher(
      bool finished
      );

   /**
    * @brief sets whether this partialdec was finished by a finishing detector in the original problem
    *
    * (in case this partialdec was translated)
    * @param finished was this partialdecs finished by a finishing detector in orig
    */
   void setFinishedByFinisherOrig(
      bool finished
      );

   /**
    * @brief sets number of blocks, only increasing number allowed
    * @param nblocks new number of blocks
    */
   void setNBlocks(
      int nblocks
      );

   /**
    * @brief set the selection status of this partialdecs
    * @param selected whether the partialdec is selected
    */
   void setSelected(
      bool selected
      );

   /**
    * @brief sets whether this partialdec stems from an orig problem partialdec
    * @param fromorig has this partialdec ancestors from the orig problem
    */
   void setStemsFromOrig(
      bool fromorig
      );

   /**
    * @brief sets whether this partialdec is user given
    * @param usergiven is this partialdec user given
    */
   void setUsergiven(
      USERGIVEN usergiven
      );

   /**
    * @brief registers statistics for a used varpartition
    */
   void setVarPartitionStatistics(
      int detectorchainindex, /**< index of the detector in the detectorchain */
      VarPartition* partition, /**< the used varpartition */
      std::vector<int>& varclasseslinking, /**< vector of classindices that were assigned to linking */
      std::vector<int>& varclassesmaster /**< vector of classindices that were assigned to master */
      );

   /**
    * @brief adds a variable to the linking variables, does not delete this var from list of open vars
    * @param varToBlock id of var to be added
    * @param block id of block to be added
    */
   void setVarToBlock(
      int varToBlock,
      int block
      );

   /**
    * @brief adds a variable to the linking variables
    * @param var id of var to be added
    * @param block id of block to be added
    */
   void fixVarToBlock(
      int var,
      int block
      );

   /**
    * @brief adds a variable to the linking variables
    * @param itr valid iterator pointing to elements of PARTIALDECOMP::openvars
    * @param block id of block to be added
    * @return iterator that points to the next element of PARTIALDECOMP::openvars
    * @warning This method modifies the vector PARTIALDECOMP::openvars! Hence, any kind of iterator might be invalid afterwards!
    */
   std::vector<int>::const_iterator fixVarToBlock(
      std::vector<int>::const_iterator itr,
      int block
   );

   /**
    * @brief adds a variable to the linking variables, does not delete this var from list of open vars
    * @param varToLinking var to be set to linking
    */
   void setVarToLinking(
      int varToLinking
      );

   /**
    * @brief adds a variable to the linking variables
    * @param var var to be set to linking
    */
   void fixVarToLinking(
      int var
      );

   /**
    * @brief adds a variable to the linking variables
    * @param itr valid iterator pointing to elements of PARTIALDECOMP::openvars
    * @return iterator that points to the next element of PARTIALDECOMP::openvars
    * @warning This method modifies the vector PARTIALDECOMP::openvars! Hence, any kind of iterator might be invalid afterwards!
    */
   std::vector<int>::const_iterator fixVarToLinking(
      std::vector<int>::const_iterator itr
   );

   /** @brief adds a variable to the master variables, does not delete this var from list of open vars
    *
    *  master variables hit only constraints in the master
    */
   void setVarToMaster(
      int varToMaster   /**< var to be set to master */
      );

   /** @brief adds a variable to the master variables
    *
    *  master variables hit only constraints in the master
    */
   void fixVarToMaster(
      int var     /**< var to be set to master */
      );

   /** @brief adds a variable to the master variables
    *
    * @param itr valid iterator pointing to elements of PARTIALDECOMP::openvars
    * @return iterator that points to the next element of PARTIALDECOMP::openvars
    * @warning This method modifies the vector PARTIALDECOMP::openvars! Hence, any kind of iterator might be invalid afterwards!
    */
   std::vector<int>::const_iterator fixVarToMaster(
      std::vector<int>::const_iterator itr     /**< var to be set to master */
   );

   /**
    * @brief adds a variable to the stairlinking variables, does not delete this var from list of open vars
    * @param varToStairLinking id of variable to be added
    * @param block1 id of block one
    * @param block2 id of block two
    * @note stairlinking variables are only registered in block with smaller index
    */
   void setVarToStairlinking(
      int varToStairLinking,
      int block1,
      int block2
      );

   /**
    * @brief adds a variable to the stairlinking variables
    * @param var id of variable to be added
    * @param firstblock stairlinking variables hit exactly two consecutive blocks, this is the index of the first of these blocks
    * @note stairlinking variables are only registered in block with smaller index
    */
   void fixVarToStairlinking(
      int var,
      int firstblock
      );

   /**
    * @brief adds a variable to the stairlinking variables
    * @param itr valid iterator pointing to elements of PARTIALDECOMP::openvars
    * @param firstblock stairlinking variables hit exactly two consecutive blocks, this is the index of the first of these blocks
    * @return iterator that points to the next element of PARTIALDECOMP::openvars
    * @warning This method modifies the vector PARTIALDECOMP::openvars! Hence, any kind of iterator might be invalid afterwards!
    * @note stairlinking variables are only registered in block with smaller index
    */
   std::vector<int>::const_iterator fixVarToStairlinking(
      std::vector<int>::const_iterator itr,
      int firstblock
   );

   /**
    * @brief assigns a constraint by name to a block
    * @see fixConsToBlock
    * @returns true iff successful
    */
   bool fixConsToBlockByName(
      const char*           consname,            /**< name of the constraint */
      int                   blockid              /**< block index (counting from 0) */
      );

   /**
    * @brief assigns a variable by name to a block
    * @see fixVarToBlock
    * @returns true iff successful
    */
   bool fixVarToBlockByName(
      const char*           varname,             /**< name of the variable */
      int                   blockid              /**< block index (counting from 0) */
      );

   /**
    * @brief assgins a constraint by name as master
    * @see fixConsToMaster
    * @returns true iff successful
    */
   bool fixConsToMasterByName(
      const char*           consname   /**< name of cons to fix as master cons */
      );

   /**
    * @brief assigns a variable with given name as master
    * @see fixVarToMaster
    * @returns true iff successful
    */
   bool fixVarToMasterByName(
      const char*           varname              /**< name of the variable */
      );

   /**
    * @brief assigns a variable by name to the linking variables
    * @see fixVarToLinking
    * @returns true iff successful
    */
   bool fixVarToLinkingByName(
      const char*           varname              /**< name of the variable */
      );

   /**
    * @brief generates and opens a gp visualization of the partialdec
    * @see visual/pdfreader and
    * @note linux only
    */
   void showVisualization();

   /**
    * @brief generates a visualization of the partialdec using gnuplot
    * @param filename Path where to store the gp file
    * @param outname Path at which gnuplot will output its result
    * @param outputformat The format of the gnuplot output. Should match the file extension of outname
    * @note linux only, requires gnuplot
    */
   void generateVisualization(
      char* filename,
      char* outname,
      GP_OUTPUT_FORMAT outputformat = GP_OUTPUT_FORMAT_PDF
      );

   /**
    * @brief writes a gp visualization of the partialdec to a file
    * @param filename Path where to store the gp file
    * @param outname Path at which gnuplot will output its result
    * @param outputformat The format of the gnuplot output. Should match the file extension of outname
    */
   void writeVisualizationFile(
      char* filename,
      char* outname,
      GP_OUTPUT_FORMAT outputformat = GP_OUTPUT_FORMAT_PDF
      );
   
   /**
    * @brief generates a gp visualization of the partialdec without compilation or opening
    */
   void exportVisualization();

   /**
    * @brief Checks whether this partialdec is a userpartialdec that should be completed
    *
    * the completion should be done by setting unspecified constraints to master
    * @return TRUE iff this partialdec is a userpartialdec that should be completed
    */
   SCIP_Bool shouldCompletedByConsToMaster();

   /**
    * @brief sorts the vars and conss data structures by their indices
    * @returns true if the internal order of variables or constraints changed
    */
   bool sort();

   /**
    * @brief set statistical vector of fractions of constraints set to blocks per involved detector
    * @param newvector vector of fractions of constraints set to blocks per involved detector
    */
   void setPctConssToBlockVector(
      std::vector<SCIP_Real>& newvector
      );

   /**
    * @brief set statistical vector of fractions of constraints that are not longer open per involved detector
    * @param newvector vector of fractions of constraints that are not longer open per involved detector
    */
   void setPctConssFromFreeVector(
      std::vector<SCIP_Real>& newvector
   );

   /**
    * @brief set statistical vector of fractions of constraints assigned to the border per involved detector
    * @param newvector vector of fractions of constraints assigned to the border per involved detector
    */
   void setPctConssToBorderVector(
      std::vector<SCIP_Real>& newvector
      );

   /**
    * @brief set statistical vector of fraction of variables assigned to the border per involved detector
    * @param newvector vector of fractions of variables assigned to the border per involved detector
    */
   void setPctVarsToBorderVector(
      std::vector<SCIP_Real>& newvector
      );

   /**
    * @brief set statistical vector of fractions of variables assigned to a block per involved detector
    * @param newvector vector of fractions of variables assigned to a block per involved detector
    */
   void setPctVarsToBlockVector(
      std::vector<SCIP_Real>& newvector
   );

   /**
    * @brief set statistical vector of variables that are not longer open per involved detector
    * @param newvector vector of fractions of variables that are not longer open per involved detector
    */
   void setPctVarsFromFreeVector(
      std::vector<SCIP_Real>& newvector
      );

   /**
    * @brief set statistical vector of the times that the detectors needed for detecting per involved detector
    * @param newvector vector of the times that the detectors needed for detecting per involved detector
    */
   void setDetectorClockTimes(
      std::vector<SCIP_Real>& newvector
      );

   /** @brief gets the classic score
    *
    * @note -1 iff not calculated yet, \see GCGconshdlrDecompCalcClassicScore
    * @returns border area score
    */
   SCIP_Real getClassicScore();

   /** @brief set the classic score
    */
   void setClassicScore(
      SCIP_Real score   /**< new score value in [0,1] */
   );

   /** @brief gets the border area score
    *
    * @note -1 iff not calculated yet, \see GCGconshdlrDecompCalcBorderAreaScore
    * @returns border area score
    */
   SCIP_Real getBorderAreaScore();

   /** @brief set the border area score
    */
   void setBorderAreaScore(
      SCIP_Real score   /**< new score value in [0,1] */
   );

    /** @brief gets the maximum white area score
    *
    * "maximum white score" is fraction of the area of the decomposed matrix that is neither block or border
    * @note -1 iff not calculated yet, \see GCGconshdlrDecompCalcMaxWhiteScore
    * @returns maximum  white area score
    * */
   SCIP_Real getMaxWhiteScore();

   /** @brief set the maximum white area score
    */
   void setMaxWhiteScore(
      SCIP_Real score   /**< new score value in [0,1] */
   );

   /** @brief gets the maximum foreseeing white area score
    *
    * @note -1 iff not calculated yet, \see GCGconshdlrDecompCalcMaxForseeingWhiteScore
    * @returns maximum foreseeing white area score
    * */
   SCIP_Real getMaxForWhiteScore();

   /** @brief set the maximum foreseeing white area score
    */
   void setMaxForWhiteScore(
      SCIP_Real score   /**< new score value in [0,1] */
   );

   /** @brief gets the setpartitioning maximum foreseeing white area score
    *
    * @note -1 iff not calculated yet, \see GGCGconshdlrDecompCalcSetPartForseeingWhiteScore
    * @returns setpartitioning maximum foreseeing white area score
    * */
   SCIP_Real getSetPartForWhiteScore();

   /** @brief set the setpartitioning maximum foreseeing white area score
    */
   void setSetPartForWhiteScore(
      SCIP_Real score   /**< new score value in [0,1] */
   );

   /** @brief gets the maximum foreseeing white area score with respect to aggregatable blocks
    *
    * @note -1 iff not calculated yet, \see GCGconshdlrDecompCalcMaxForeseeingWhiteAggScore
    * @returns maximum foreseeing white area score with respect to aggregatable blocks
    * */
   SCIP_Real getMaxForWhiteAggScore();

   /** @brief set the maximum foreseeing white area score with respect to aggregatable blocks
    */
   void setMaxForWhiteAggScore(
      SCIP_Real score   /**< new score value in [0,1] */
   );

   /** @brief gets the setpartitioning maximum foreseeing white area score with respect to aggregateable
    *
    * @note -1 iff not calculated yet, \see GCGconshdlrDecompCalcSetPartForWhiteAggScore
    * @returns setpartitioning maximum foreseeing white area score with respect to aggregateable
    */
   SCIP_Real getSetPartForWhiteAggScore();

   /** @brief set the setpartitioning maximum foreseeing white area score with respect to aggregateable
    */
   void setSetPartForWhiteAggScore(
      SCIP_Real score   /**< new score value in [0,1] */
   );

   /** @brief gets the benders score
    *
    * @note -1 iff not calculated yet, \see GCGconshdlrDecompCalcBendersScore
    * @returns benders score
    */
   SCIP_Real getBendersScore();

   /** @brief set the benders score
    */
   void setBendersScore(
      SCIP_Real score   /**< new score value in [0,1] */
   );

   /** @brief gets the strong decomposition score
    *
    * @note -1 iff not calculated yet, \see GCGconshdlrDecompCalcStrongDecompositionScore
    * @returns strong decomposition score
    */
   SCIP_Real getStrongDecompScore();

   /** @brief set the strong decomposition score
    */
   void setStrongDecompScore(
      SCIP_Real score   /**< new score value in [0,1] */
   );

   /** sorts the partialdec and calculates a its implicit assignments, hashvalue and evaluation
    *
    * @returns SCIP_OKAY if the result is consistent, SCIP_ERROR if there was an inconsistency
   */
   void prepare();

   /**
    * @brief Checks if the aggregation information was already calculated
    * @return true iff the aggregation information was already calculated
    */
   bool aggInfoCalculated();

   /**
    * @brief computes if aggregation of sub problems is possible
    *
    * checks if aggregation of sub problems is possible and stores the corresponding aggregation information
    *
    * @param ignoreDetectionLimits Set to true if computation should ignore detection limits. This parameter is ignored if the patched bliss version is not present.
   */
   void calcAggregationInformation(
      bool ignoreDetectionLimits
      );

   /**< @brief gets vector of indices of all constraints assigned to blocks
    *
    * @note conssforblocks[k] contains a vector of indices of all constraints assigned to block k
    * @returns vector of a vector of indices for each block */
   std::vector<std::vector<int>>& getConssForBlocks(
   );

   int getTranslatedpartialdecid() const;

   void setTranslatedpartialdecid(
      int decid
      );

   /**
   * @brief creates a detector chain short string for this partialdec, is built from detector chain
   */
   void buildDecChainString(
      char* buffer /**< will contain string of detector chars in chronological order afterwards*/
      );

private:

   /**
    * @brief adds empty entries for all partition statistics for a detector added to the detector chain
    */
   void addEmptyPartitionStatistics();

   /** @brief assigns open cons
    *
    * Assignments as follows:
    *  - to master if it hits blockvars of different blocks
    *  - to the respective block if it hits a blockvar of exactly one block and no stairlinking var
    *  - to master if it hits a stairlinking var but there is no block the cons may be assigned to
    *  - to the block with the lowest number of conss if it hits a stairlinking var and there are blocks the cons may be
    *    assigned to
    *  - leave it open if it hits no blocks yet
    *  @return true iff some assignment was made by the method
    */
   bool assignHittingOpenconss(
      );

   /** @brief assigns every open var
    *
    * Assignments as follows:
    *  - to the respective block if it hits blockconss of exactly one block
    *  - to linking if it hits blockconss of more than one different blocks
    *  - leave the var open otherwise
    *  @return true iff there is a var that has been assigned in this call*/
   bool assignHittingOpenvars(
      );

   /**
    * @brief assigns every open cons to master that hits
    *
    * Assignments as follows:
    *  - exactly one block var and at least one open var or
    *  - a master var
    *  - or leave it open elsewise
    */
   void assignOpenPartialHittingConsToMaster(
      );

   /**
    * @brief assigns open conss/vars that hit exactly one block and at least one open var/cons to border
    */
   void assignOpenPartialHittingToMaster(
      );

   /**
    * @brief assigns every open var to linking that hits
    *
    * Assignments as follows:
    *  - exactly one block cons and at least one open cons
    *  - leave it open otherwise
    */
   void assignOpenPartialHittingVarsToMaster(
      );

   /**
    * @brief calculates the number of nonzero coefficients for the blocks
    */
   void calcNCoeffsForBlocks(
   );

   /**
    * @brief calculates the hash value of the partialdec for comparing
    */
   void calcHashvalue();

   /**
    * @brief blockwise calculation of how many master conss contain the block vars
    *
    * counts for each pair of block and master constraint, how many nonzero entries the variables of the blocks
    * have in the master constraint
    */
   void calcNCoeffsForBlockForMastercons(
        );

   /**
    *  @brief optimizes block order to max stairlinking vars
    *
    *  changes the block order in a way such that all linking vars that are potentially stairlinking
    *  may be reassigned to stairlinking
    * @note precondition: all potentially stairlinking vars have a staircase structure */
   void changeBlockOrderStaircase(
        GraphGCG* g /**< graph with blocks as nodes and weighted edges for the number of
                         potentially stairlinkingvars connecting two blocks */
        );

   /**
    * @brief changes the order of the blocks according to the given mapping
    *
    * \note precondition: given mapping needs to be an adequately sized permutation */
   void changeBlockOrder(
        std::vector<int> oldToNewBlockIndex /**< the mapping from old to new block indices */
        );

   /**
    * @brief returns true if the given detector used a conspartition
    * @return true iff the given detector used a conspartition
    */
   bool consPartitionUsed(
      int detectorchainindex /**< index of the detector in the detectorchain */
      );

   /**
    * @brief prints out the current aggregation information
    *
    *  aggregation information: if there has been identified identical blocks
    */
   void displayAggregationInformation();

   /**
    * @brief calculates potential stairlinking variables with their blocks
    *
    * @return a vector of pairs of var index and vector of (two) block indices
    *  the related linking variable hits exactly these two blocks given in the related vector
    */
   std::vector< std::pair< int, std::vector< int > > > findLinkingVarsPotentiallyStairlinking(
      );

   /**
    * @brief returns the data of the conspartition that the given detector made use of
    */
   void getConsPartitionData(
      int detectorchainindex, /**< index of the detector in the detectorchain */
      ConsPartition** partition, /**< a pointer to the used conspartition (set by method)*/
      std::vector<int>& consclassesmaster /**< a vector containing all indices of the consclasses assigned to master
                                            *  (set by method) */
      );

   /**
    * @brief returns a string displaying all detector-related clock times and assignment data
    *
    * @return string displaying all detector-related clock times and assignment data
    */
   std::string getDetectorStatistics(
      int detectorchainindex /**< index of the detector in the detectorchain */
      );

   /**
    * @brief returns a string displaying partition information if a partition was used
    *
    * @return string displaying partition information if a partition was used
    */
   std::string getDetectorPartitionInfo(
      int detectorchainindex, /**< index of the detector in the detectorchain */
      bool displayConssVars /**< pass true if constraints and variables of the respective classes should be displayed */
      );

   /**
    * @brief Gets the number used partitions
    * @return number used partitions
    */
   int getNUsedPartitions();

   /**
    * @brief Gets the data of the varpartition that the given detector made use of
    * @return data of the varpartition that the given detector made use of
    */
   void getVarPartitionData(
      int detectorchainindex, /**< index of the detector in the detectorchain */
      VarPartition** partition, /**< a pointer to the used varpartition (set by method) */
      std::vector<int>& varclasseslinking, /**< a vector containing all indices of the varclasses assigned to linking (set by method) */
      std::vector<int>& varclassesmaster /**< a vector containing all indices of the varclasses assigned to master (set by method)*/
      );

   /**
    * @brief checks if calculation of aggregation information is considered to be too expensive
    * @return TRUE iff calculation of aggregation information is considered to be too expensive
    */
   SCIP_Bool isAgginfoTooExpensive();

   /**
    * @brief Gets whether the cons is a cons of the block
    * @param cons id of constraint to check
    * @param block id of the blocks
    * @return true iff the cons is a cons of the block
    */
   /**  */
   bool isConsBlockconsOfBlock(
      int cons,
      int block
      );

   /**
    * @brief returns true if the given detector used a varpartition
    *
    * @return true if the given detector used a varpartition
    */
   bool varPartitionUsed(
      int detectorchainindex /**< index of the detector in the detectorchain */
      );

};


} /* namespace gcg */
#endif /* GCG_CLASS_PARTIALDECOMP_H__ */
