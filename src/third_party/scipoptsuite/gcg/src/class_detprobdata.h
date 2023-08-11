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

/**@file   class_detprobdata.h
 * @ingroup DECOMP
 * @brief  class storing partialdecs and the problem matrix
 * @note   formerly called "Seeedpool"
 * @author Michael Bastubbe
 * @author Julius Hense
 * @author Hanna Franzen
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_CLASS_DETPROBDATA_H__
#define GCG_CLASS_DETPROBDATA_H__

#include <vector>

#if __cplusplus >= 201103L
#include <unordered_map>
using std::unordered_map;
#else
#include <tr1/unordered_map>
using std::tr1::unordered_map;
#endif

#include <functional>
#include <string>
#include <utility>
#include "gcg.h"

#include "class_partialdecomp.h"
#include "class_conspartition.h"
#include "class_varpartition.h"

/** constraint type */
enum SCIP_Constype_orig
{
   SCIP_CONSTYPE_EMPTY         =  0,         /**<  */
   SCIP_CONSTYPE_FREE          =  1,         /**<  */
   SCIP_CONSTYPE_SINGLETON     =  2,         /**<  */
   SCIP_CONSTYPE_AGGREGATION   =  3,         /**<  */
   SCIP_CONSTYPE_VARBOUND      =  4,         /**<  */
   SCIP_CONSTYPE_SETPARTITION  =  5,         /**<  */
   SCIP_CONSTYPE_SETPACKING    =  6,         /**<  */
   SCIP_CONSTYPE_SETCOVERING   =  7,         /**<  */
   SCIP_CONSTYPE_CARDINALITY   =  8,         /**<  */
   SCIP_CONSTYPE_INVKNAPSACK   =  9,         /**<  */
   SCIP_CONSTYPE_EQKNAPSACK    = 10,         /**<  */
   SCIP_CONSTYPE_BINPACKING    = 11,         /**<  */
   SCIP_CONSTYPE_KNAPSACK      = 12,         /**<  */
   SCIP_CONSTYPE_INTKNAPSACK   = 13,         /**<  */
   SCIP_CONSTYPE_MIXEDBINARY   = 14,         /**<  */
   SCIP_CONSTYPE_GENERAL       = 15          /**<  */
};
typedef enum SCIP_Constype_orig SCIP_CONSTYPE_ORIG;


namespace gcg{

/**
 *  @brief combine two hash function of objects of a pair to get a vaulue for the pair
 */
struct pair_hash
{
   template<class T1, class T2>
   std::size_t operator()(
      const std::pair<T1, T2> &p) const
   {
      auto h1 = std::hash<T1>{}( p.first );
      auto h2 = std::hash<T2>{}( p.second );

      /* overly simple hash combination */
      return h1 ^ h2;
   }
};

/**
 * class to manage the detection process and data for one coefficient matrix of a MIP, usually there is one detprobdata for the original and one detprobdata for the presolved problem
 */
class DETPROBDATA
{ /*lint -esym(1712,DETPROBDATA)*/

private:
   SCIP* scip;                                           /**< SCIP data structure */
   std::vector<PARTIALDECOMP*> openpartialdecs;          /**< vector of open partialdecs */
   std::vector<PARTIALDECOMP*> finishedpartialdecs;      /**< vector of finished partialdecs */
   std::vector<PARTIALDECOMP*> ancestorpartialdecs;      /**< old partialdecs that were previously used */

   std::vector<SCIP_CONS*> relevantconss;                /**< stores all constraints that are not marked as deleted or obsolete */
   std::vector<SCIP_VAR*> relevantvars;                  /**< stores all prob variables that are not fixed to 0 */
   std::vector<std::vector<int>> varsforconss;           /**< stores for every constraint the indices of variables
                                                           *< that are contained in the constraint */
   std::vector<std::vector<double>> valsforconss;        /**< stores for every constraint the coefficients of
                                                           *< variables that are contained in the constraint (i.e.
                                                           *< have a nonzero coefficient) */
   std::vector<std::vector<int>> conssforvars;           /**< stores for every variable the indices of constraints
                                                           *< containing this variable */
   
   std::vector<std::vector<int>> conssadjacencies;
   unordered_map<SCIP_CONS*, int> constoindex;            /**< maps SCIP_CONS* to the corresponding internal index */
   unordered_map<SCIP_VAR*, int> vartoindex;              /**< maps SCIP_VAR* to the corresponding internal index */

   unordered_map<std::pair<int, int>, SCIP_Real, pair_hash> valsMap;   /**< maps an entry of the matrix to its
                                                            *< value, zeros are omitted */

   std::vector<SCIP_VAR*> origfixedtozerovars;           /**< collection of SCIP_VAR* that are fixed to zero in transformed problem*/

   int nvars;                                            /**< number of variables */
   int nconss;                                           /**< number of constraints */
   int nnonzeros;                                        /**< number of nonzero entries in the coefficient matrix */

   SCIP_Bool original;                                  /**< corresponds the matrix data structure to the original
                                                           *< problem */

public:
   std::vector<std::pair<int, int>> candidatesNBlocks;   /**< candidate for the number of blocks, second int indicates how often a candidate was added */

   std::vector<ConsPartition*> conspartitioncollection;   /**< collection of different constraint class distributions  */
   std::vector<VarPartition*> varpartitioncollection;     /**< collection of different variable class distributions   */

   SCIP_Real classificationtime;                         /**< time that was consumed by the classification of the constraint and variables classifiers */
   SCIP_Real nblockscandidatescalctime;                  /**< time that was used to calulate the candidates of te block number */
   SCIP_Real postprocessingtime;                         /**< time that was spent in postproceesing decomposigtions */
   SCIP_Real translatingtime;                            /**< time that was spent by transforming partialdecs between presolved and orig problem */

private:

   /**
    * @brief calculates necessary data for translating partialdecs and partitions
    */
   void calcTranslationMapping(
      DETPROBDATA* origdata,              /** original detprobdata */
      std::vector<int>& rowothertothis,   /** constraint index mapping from old to new detprobdata */
      std::vector<int>& rowthistoother,   /** constraint index mapping new to old detprobdata */
      std::vector<int>& colothertothis,   /** variable index mapping from old to new detprobdata */
      std::vector<int>& colthistoother,   /** variable index mapping from new to old detprobdata */
      std::vector<int>& missingrowinthis  /** missing constraint indices in new detprobdata */
      );

   /**
    * @brief returns translated partialdecs derived from given mapping data

    * @return vector of translated partialdec pointers
    */
    void getTranslatedPartialdecs(
       std::vector<PARTIALDECOMP*>& otherpartialdecs,   /**< partialdecs to be translated */
       std::vector<int>& rowothertothis,   /** constraint index mapping from old to new detprobdata */
       std::vector<int>& rowthistoother,   /** constraint index mapping new to old detprobdata */
       std::vector<int>& colothertothis,   /** variable index mapping from old to new detprobdata */
       std::vector<int>& colthistoother,   /** variable index mapping from new to old detprobdata */
       std::vector<PARTIALDECOMP*>& translatedpartialdecs   /**< will contain translated partialdecs */
       );

public:

   /**
    * @brief constructor
    * @param scip SCIP data structure
    * @param _originalProblem true iff the detprobdata is created for the presolved problem
    */
   DETPROBDATA(
      SCIP* scip,
      SCIP_Bool _originalProblem
      );

   /**
    * destructor
    */
   ~DETPROBDATA();

   /**
    * @brief adds a constraint partition if it is no duplicate of an existing constraint partition
    */
   void addConsPartition(
      ConsPartition* partition /**< conspartition to be added*/
      );

   /**
    * @brief adds a candidate for block number and counts how often a candidate is added
    */
   void addCandidatesNBlocksNVotes(
      int candidate, /**< candidate for block size */
      int nvotes     /**< number of votes this candidates will get */
   );

   /**
    * @brief adds a partialdec to ancestor partialdecs
    * @param partialdec partialdec that is added to the ancestor partialdecs
    */
   void addPartialdecToAncestor(
      PARTIALDECOMP* partialdec
      );

   /**
    * @brief adds a partialdec to current partialdecs (data structure for partialdecs that are goin to processed in the propagation rounds)
    * @param partialdec pointer of partialdec to be added
    * @returns TRUE if the partialdecs was successfully added (i.e. it is no duplicate of a known partialdec)
    */
   bool addPartialdecToOpen(
      PARTIALDECOMP* partialdec
      );

   /**
    * @brief adds a partialdec to finished partialdecs
    * @param partialdec pointer of partialdec that is going to be added to the finished partialdecs (data structure to carry finished decompositions)
    * @returns TRUE if the partialdecs was successfully added (i.e. it is no duplicate of a known partialdec)
    * @see addPartialdecToFinishedUnchecked()
    */
   bool addPartialdecToFinished(
      PARTIALDECOMP* partialdec
      );

   /**
    * @brief adds a partialdec to finished partialdecs without checking for duplicates, dev has to check this on his own
    * @param partialdec pointer of partialdec that is going to be added unchecked to the finished partialdecs (data structure to carry finished decompositions)
    * @see addPartialdecToFinished()
    */
   void addPartialdecToFinishedUnchecked(
      PARTIALDECOMP* partialdec
      );

   /**
    * @brief  adds a variable partition if it is no duplicate of an existing variable partition
    * @param partition varpartition to be added
    */
   void addVarPartition(
      VarPartition* partition
      );

   /**
    * @brief clears ancestor partialdec data structure,
    * @note does not free the partialdecs themselves
    */
   void clearAncestorPartialdecs();

   /**
    * @brief clears current partialdec data structure
    * @note does not free the partialdecs themselves
    */
   void clearCurrentPartialdecs();

   /**
    * @brief clears finished partialdec data structure
    * 
    * @note does not free the partialdecs themselves
    */
   void clearFinishedPartialdecs();

   /**
    * @brief create the constraint adjacency datastructure that is used (if created) for some methods to faster access the constarints that have variables in common
    */
   void createConssAdjacency();

   /**
    * @brief frees temporary data that is only needed during the detection process
    */
   void freeTemporaryData();

   /**
    * @brief returns a partialdec from ancestor partialdec data structure with given index
    * 
    * @returns partialdec from ancestor partialdec data structure
    */
   PARTIALDECOMP* getAncestorPartialdec(
      int partialdecindex /**< index of partialdec in ancestor partialdec data structure */
      );

   /**
    * @brief returns pointer to a constraint partition
    * @return pointer to a cosntraint partition with the given index
    */
   ConsPartition* getConsPartition(
      int partitionIndex /**< index of constraint partition */
      );

   /**
    * @brief returns the SCIP constraint related to a constraint index
    * @return the SCIP constraint related to a constraint index
    */
   SCIP_CONS* getCons(
      int consIndex /**< index of the constraint to be considered */
      );

   /**
    * @brief return array of constraint indices that have a common variable with the given constraint
    * @return return vector of constraint indices that have a common variable with the given constraint
    * @note constraint adjacency data structure has to initilized
    */
   std::vector<int>& getConssForCons(
      int consIndex /**< index of the constraint to be considered */
      );

   /**
    * \brief returns the constraint indices of the coefficient matrix for a variable
    * @return vector of constraint indices that have a nonzero entry with this variable
    */
   std::vector<int>& getConssForVar(
      int varIndex /**< index of the variable to be considered */
      );

   /**
    * @brief determines all partialdecs from current (open) partialdec data structure
    * @returns  all partialdecs in current (open) partialdec data structure
    */
   /**  */
   std::vector<PARTIALDECOMP*>& getOpenPartialdecs();

   /**
    * @brief returns a partialdec from finished partialdec data structure
    * @return  partialdec from finished partialdec data structure
    */
   PARTIALDECOMP* getFinishedPartialdec(
      int partialdecindex /**< index of partialdec in finished partialdec data structure */
      );

   /**
    * @brief gets all finished partialdecs
    * @returns all finished partialdecs
    */
   std::vector<PARTIALDECOMP*>& getFinishedPartialdecs();

   /**
    * @brief returns the constraint index related to a SCIP constraint
    * @param cons the SCIP constraint pointer the index is asked for
    * @return the constraint index related to a SCIP constraint
    */
   int getIndexForCons(
      SCIP_CONS* cons
      );

   /**
    * @brief returns the constraint index related to a SCIP constraint
    * @param consname name of the constraint the index is asked for
    * @return the constraint index related to a SCIP constraint
    */
   int getIndexForCons(
      const char* consname
   );

   /**
    * @brief returns the variable index related to a SCIP variable
    * @param var variable pointer the index is asked for
    * @return the variable index related to a SCIP variable
    */
   int getIndexForVar(
      SCIP_VAR* var
      );

   /**
    * @brief returns the variable index related to a SCIP variable
    * @param varname name of the variable the index is asked for
    * @return the variable index related to a SCIP variable
    */
   int getIndexForVar(
      const char* varname
   );

   /**
    * @brief returns size of ancestor partialdec data structure
    * @return size of ancestor partialdec data structure
    */
   int getNAncestorPartialdecs();

   /**
    * @brief returns number of different constraint partitions
    * @return number of different constraint partitions
    */
   int getNConsPartitions();

   /**
    * @brief returns the number of variables considered in the detprobdata
    * @return number of variables considered in the detprobdata
    */
   int getNConss();

   /**
    * @brief returns the number of constraints for a given constraint
    * @return the number of constraints for a given constraint
    */
   int getNConssForCons(
      int consIndex /**< index of the constraint to be considered */
      );

   /**
    * @brief returns the number of constraints for a given variable where the var has a nonzero entry in
    * @return the number of constraints for a given variable
    */
   int getNConssForVar(
      int varIndex /**< index of the variable to be considered */
      );

   /**
    * @brief returns size of current (open) partialdec data structure
    * @return size of current (open) partialdec data structure
    */
   int getNOpenPartialdecs();

   /**
    * returns size of finished partialdec data structure
    * @return  size of finished partialdec data structure
    */
   int getNFinishedPartialdecs();

   /**
    * returns the number of stored partialdecs
    * @return  number of stored partialdecs
    */
   int getNPartialdecs();

   /**
    * @brief returns the number of nonzero entries in the coefficient matrix
    * @return the number of nonzero entries in the coefficient matrix
    */
   int getNNonzeros();

   /**
    * @brief returns number of different variable partitions
    * @return  number of different variable partitions
    */
   int getNVarPartitions();

   /**
    * @brief return the number of variables considered in the detprobdata
    * @return the number of variables considered in the detprobdata
    */
   int getNVars();

   /**
    * @brief returns the number of variables for a given constraint
    * @return the number of variables for a given constraint
    */
   int getNVarsForCons(
      int consIndex /**< index of the constraint to be considered */
      );

   /**
    * @brief returns pointers to all orig vars that are fixed to zero
    * @returns vector of vars
    */
   std::vector<SCIP_VAR*> getOrigVarsFixedZero();

   /**
    * @brief returns pointers to all constraints that are not marked as deleted or obsolete
    * @returns vector of conss
    */
   std::vector<SCIP_CONS*> getRelevantConss();

   /**
    * @brief returns pointers to all problem vars that are not fixed to 0
    * @returns vector of vars
    */
   std::vector<SCIP_VAR*> getRelevantVars();

   /**
    * @brief returns the corresponding scip data structure
    * @return the corresponding scip data structure
    */
   SCIP* getScip();

   /**
    * @brief gets the candidates for number of blocks added by the user followed by the found ones sorted in descending order by how often a candidate was proposed
    * @param candidates will contain the candidates for number of blocks sorted in descending order by how often a candidate was added
    */
   void getSortedCandidatesNBlocks(
      std::vector<int>& candidates
      );

   /**
    * @brief returns a coefficient from the coefficient matrix
    * @return a coefficient from the coefficient matrix
    */
   SCIP_Real getVal(
      int row, /**< index of the constraint to be considered */
      int col  /**< index of the variable to be considered */
      );

   /**
    * @brief returns the nonzero coefficients of the coefficient matrix for a constraint
    * @return vector of coefficients of in matrix for constraints
    * @note same order as in @see getVarsForCons()
    */
   std::vector<SCIP_Real>& getValsForCons(
      int consIndex /**< index of the constraint to be considered */
      );

   /**
    * @brief returns pointer to a variable partition with given index
    * @return pointer to a variable partition with given index
    */
   VarPartition* getVarPartition(
      int partitionIndex /**< index of variable partition */
      );

   /**
    * @brief returns vector to stored variable partitions
    * @return returns vector to stored variable partitions
    */
   std::vector<VarPartition*> getVarPartitions();

   /**
    * @brief returns SCIP variable related to a variable index
    * @return SCIP variable pointer related to a variable index
    */
   SCIP_VAR* getVar(
      int varIndex /**< index of the variable to be considered */
      );

   /**
    * @brief returns the variable indices of the coefficient matrix for a constraint
    * @return the variable indices of the coefficient matrix for a constraint
    */
   std::vector<int>& getVarsForCons(
      int consIndex /**< index of the constraint to be considered */
      );

   /**
    * @brief returns whether a constraint is a cardinality constraint, i.e. of the \f$\sum_{i} x_i = b\f$
    * @param consindexd index of constraint that is be checked
    * @return returns whether a constraint is a cardinality constraint
    */
   bool isConsCardinalityCons(
         int  consindexd
         );

   /**
    * @brief determines whether or not the constraint-constraint adjacency data structure is initilized
    * 
    * @returns true iff the constraint-constraint adjacency data structure is initilized
    */
   SCIP_Bool isConssAdjInitialized();

   /**
    * @brief is cons with specified indec partitioning, or packing covering constraint?
    * @param consindexd index of the given cons
    * @return is cons with specified indec partitioning, or packing covering constraint
    */
   bool isConsSetpp(
      int  consindexd
      );

   /**
    * @brief is cons with specified index partitioning packing, or covering constraint?
    * @param consindexd index of cons to be checked
    * @return whether a constraint is partitioning packing, or covering constraint?
    */
   bool isConsSetppc(
      int  consindexd
      );

   /** @brief is constraint ranged row, i.e., -inf < lhs < rhs < inf?
    * 
    * @returns whether val is ranged row
    */
   SCIP_Bool isFiniteNonnegativeIntegral(
      SCIP*                 scip,               /**< SCIP data structure */
      SCIP_Real             x                   /**< value */
   );

   /**
    * @brief check if partialdec is a duplicate of an existing finished partialdec
    * @param partialdec partialdec to be checked
    * @returns TRUE iff partialdec is a duplicate of an existing finished partialdec
    */
   SCIP_Bool isPartialdecDuplicateofFinished(
      PARTIALDECOMP* partialdec
      );

   /**
     * @brief returns true if the matrix structure corresponds to the presolved problem
     * @return TRUE if the matrix structure corresponds to the presolved problem
     */
   SCIP_Bool isAssignedToOrigProb();

   /** @brief is constraint ranged row, i.e., -inf < lhs < rhs < inf? 
    * @returns whether constraint is ranged row
   */
   SCIP_Bool isRangedRow(
      SCIP*                 scip,   /**< SCIP data structure */
      SCIP_Real             lhs,    /**< left hand side */
      SCIP_Real             rhs     /**< right hand side */
   );

   /**
    * @brief check if partialdec is a duplicate of any given partialdecs
    * @param comppartialdec partialdec to be checked
    * @param partialdecs partialdecs to compare with
    * @param sort sort the vars and conss data structures in the partialdecs by their indices
    * @return TRUE iff partialdec is no duplicate of any given partialdecs
    */
   SCIP_Bool partialdecIsNoDuplicateOfPartialdecs(
      PARTIALDECOMP* comppartialdec,
      std::vector<PARTIALDECOMP*> const & partialdecs,
      bool sort
   );

   /**
    * @brief output method for json file writer to write block candidate information
    * @param scip SCIP data structure
    * @param file  output file or NULL for standard output
    */
   void printBlockcandidateInformation(
    SCIP*                 scip,               /**< SCIP data structure */
    FILE*                 file                /**< output file or NULL for standard output */
   );

   /**
    * @brief output method for json file writer to write partition candidate information
    * @param file output file or NULL for standard output
    */
   void printPartitionInformation(
    FILE*                 file                /**< output file or NULL for standard output */
   );

   /**
    * @brief sorts partialdecs in finished partialdecs data structure according to the current scoretype
    */
   void sortFinishedForScore();

   /**
    * @brief translates partialdecs if the index structure of the problem has changed, e.g. due to presolving
    * @return translated partialdecs
    */
   std::vector<PARTIALDECOMP*> translatePartialdecs(
      DETPROBDATA* otherdata,                       /**< old detprobdata */
      std::vector<PARTIALDECOMP*> otherpartialdecs  /**< partialdecs to be translated */
      );

   /**
    * @brief translates partialdecs if the index structure of the problem has changed, e.g. due to presolving
    * @return translated partialdecs
    */
   std::vector<PARTIALDECOMP*> translatePartialdecs(
      DETPROBDATA* otherdata                        /**< old detprobdata */
   );


};
/* class DETPROBDATA */


} /* namespace gcg */

/**
 * \brief interface data structure for the detector calling methods
 */
struct Partialdec_Detection_Data
{
   gcg::DETPROBDATA* detprobdata;         /**< current detprobdata to consider */
   gcg::PARTIALDECOMP* workonpartialdec;  /**< partialdec (aka partial decomposition) to be propagted in next detector call */
   gcg::PARTIALDECOMP** newpartialdecs;   /**< array of new partialdecs to filled by the detetcor methods */
   int nnewpartialdecs;                   /**< number of new neeed, set by the detector methods */
   double detectiontime;                  /**< time spent on detection */
};

#endif /* GCG_CLASS_DETPROBDATA_H__ */
