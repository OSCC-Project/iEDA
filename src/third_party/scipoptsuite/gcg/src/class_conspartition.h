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

/**@file   class_conspartition.h
 * @brief  class representing a partition of a set of constraints
 * @author Julius Hense
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_CLASS_CONSPARTITION_H__
#define GCG_CLASS_CONSPARTITION_H__

#include "class_indexpartition.h"

namespace gcg
{

enum ConsClassDecompInfo
{
   BOTH = 0,                     /**< assign class to master or pricing problem */
   ONLY_MASTER = 1,              /**< assign class only to master problem */
   ONLY_PRICING = 2              /**< assign class only to pricing problem */
};
typedef enum ConsClassDecompInfo CONS_DECOMPINFO;


class ConsPartition : public IndexPartition
{

public:

   /** constructor */
   ConsPartition(
      SCIP*                scip,                /**< scip data structure */
      const char*          name,                /**< name of partition (will be copied) */
      int                  nClasses,            /**< initial number of classes */
      int                  nConss               /**< number of constraints to be classified */
   );

   /** copy constructor */
   ConsPartition(
      const ConsPartition* toCopy              /**< ConsPartition to be copied */
   );


   /** destructor */
   ~ConsPartition();


   /** creates a new class, returns index of the class */
   int addClass(
      const char* name,                /**< name of the class (will be copied) */
      const char* desc,                /**< description of the class (will be copied) */
      CONS_DECOMPINFO decompInfo            /**< decomposition code of the class */
   );

   /** assigns a constraint to a class */
   void assignConsToClass(
      int consindex,                   /**< index of the constraint */
      int classindex                   /**< index of the class */
   );

   /** returns a vector containing all possible subsets of the chosen classindices */
   std::vector<std::vector<int>> getAllSubsets(
      bool both,                       /**< true, if BOTH classes should be considered */
      bool only_master,                /**< true, if ONLY_MASTER classes should be considered */
      bool only_pricing                /**< true, if ONLY_PRICING classes should be considered */
   );

   /** returns the decomposition info of a class */
   CONS_DECOMPINFO getClassDecompInfo(
      int classindex                   /**< index of class */
   );

   /** returns the name of the class a constraint is assigned to */
   const char* getClassNameOfCons(
      int consindex                    /**< index of constraint */
   );


   /** returns the index of the class a constraint is assigned to */
   int getClassOfCons(
      int consindex                    /**< index of constraint */
   );

   /** returns vector containing the assigned class of each constraint */
   const int* getConssToClasses(
   );

   /** returns the number of constraints */
   int getNConss(
   );

   /** returns a vector with the numbers of constraints that are assigned to the classes */
   std::vector<int> getNConssOfClasses(
   );

   /** returns whether a constraint is already assigned to a class */
   bool isConsClassified(
      int consindex                    /**< index of constraint */
   );


   /** returns partition with reduced number of classes
    *  if the current number of classes is greater than an upper bound
    *  and lower than 2*(upper bound) (returns NULL otherwise) */
   ConsPartition* reduceClasses(
      int maxNumberOfClasses           /**< upper bound */
   );

   /** sets the decomposition code of a class */
   void setClassDecompInfo(
      int classindex,                  /**< index of class */
      CONS_DECOMPINFO decompInfo            /**< decomposition code of class */
   );

};


} /* namespace gcg */
#endif /* GCG_CLASS_CONSPARTITION_H__ */
