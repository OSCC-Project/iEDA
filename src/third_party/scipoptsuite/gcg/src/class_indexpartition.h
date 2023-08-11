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

/**@file   class_indexpartition.h
 * @brief  generalization of ConsPartition and VarPartition
 * @author Julius Hense
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_CLASS_INDEXPARTITION_H__
#define GCG_CLASS_INDEXPARTITION_H__

#include "objscip/objscip.h"
#include <vector>
#include <string>

namespace gcg
{

class IndexPartition
{

protected:
   SCIP*                      scip;                   /**< scip data structure */

private:
   std::string                name;                   /**< name of the partition */
   int                        nClasses;               /**< number of classes the partition provides */
   int                        nIndices;               /**< number of indices */
   std::vector<int>           indicesToClasses;       /**< index i is assigned to class indicesToClasses[i] (-1 if not assigned)*/
   std::vector<std::string>   classNames;             /**< name of class k is classNames[k] */
   std::vector<std::string>   classDescriptions;      /**< information text describing class k is classDescriptions[k] */
   std::vector<int>           classDecompInfo;        /**< decomposition information of class k is classDecompInfo[k] */


protected:

   /** constructor */
   IndexPartition(
      SCIP*                scip,                /**< scip data structure */
      const char*          name,                /**< name of partition (will be copied) */
      int                  nClasses,            /**< initial number of classes */
      int                  nIndices             /**< number of indices to be classified */
   );

   /** copy constructor */
   IndexPartition(
      const IndexPartition* toCopy              /**< IndexPartition to be copied */
   );


   /** destructor */
   virtual ~IndexPartition();

   /** creates a new class, returns index of the class */
   int addClass(
      const char* name,                /**< name of the class (will be copied) */
      const char* desc                 /**< description of the class (will be copied) */
   );

   /** assigns an index to a class */
   void assignIndexToClass(
      int index,                       /**< index to be assigned */
      int classindex                   /**< index of the class */
   );

   /** returns a vector containing all possible subsets of the given classindices */
   std::vector<std::vector<int>> getAllSubsets(
      std::vector<int>& classindices   /**< classindices to be considered */
   );

   /** returns the decomposition info of the a class */
   int getClassDecompInfo(
      int classindex                   /**< index of the class */
   );

   /** returns the name of the class an index is assigned to */
   const char* getClassNameOfIndex(
      int index
   );

   /** returns the index of the class an index is assigned to */
   int getClassOfIndex(
      int index
   );

   /** returns vector containing the assigned class of each index */
   std::vector<int>& getIndicesToClasses(
   );

   /** returns the number of indices */
   int getNIndices(
   );

   /** returns a vector with the numbers of indices that are assigned to the classes */
   std::vector<int> getNIndicesOfClasses(
   );

   /** returns whether an index is already assigned to a class */
   bool isIndexClassified(
      int index
   );

   /** sets the decomposition info of the a class */
   void setClassDecompInfo(
      int classindex,                  /**< index of the class */
      int decompInfo                   /**< decomposition info */
   );


public:

   /** returns true if the other partition has an equivalent index structure,
    *  meaning that the partition of the set of constraints is the same ignoring the concrete classindices, classnames, etc. */
   bool isDuplicateOf(
      IndexPartition* otherPartition    /**< other partition to be checked */
   );

   /** returns the information text of a class */
   const char* getClassDescription(
      int classindex                   /**< index of class */
   );

   /** returns the name of a class */
   const char* getClassName(
      int classindex                   /**< index of class */
   );

   /** returns the name of the partition */
   const char* getName(
   );


   /** returns the number of classes the partition provides */
   int getNClasses(
   );

   /** returns a class index mapping for creating a new partition
     * the enlarged class is always the class with index 0
     * returns empty vector if the current number of classes is lower than an upper bound
     * or greater than 2*(upper bound) */
   std::vector<int> reduceClasses(
      int maxNumberOfClasses           /**< upper bound */
   );

   /** removes all classes which do not have any assigned index (classindices may change)
    *  returns number of removed classes */
   int removeEmptyClasses(
   );

   /** sets the information text of a class */
   void setClassDescription(
      int classindex,                  /**< index of class */
      const char* desc                 /**< description of class (will be copied) */
   );

   /** sets the name of a class */
   void setClassName(
      int classindex,                  /**< index of class */
      const char* name                 /**< name of class (will be copied) */
   );

};


} /* namespace gcg */
#endif /* SRC_CLASS_INDEXPARTITION_H_ */
