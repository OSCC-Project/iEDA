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

/**@file   class_varpartition.h
 * @brief  class representing a partition of a set of variables
 * @author Julius Hense
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_CLASS_VARPARTITION_H__
#define GCG_CLASS_VARPARTITION_H__

#include "class_indexpartition.h"

namespace gcg
{

enum VarClassDecompInfo
{
   ALL = 0,
   LINKING = 1,
   MASTER = 2,
   BLOCK = 3
};
typedef enum VarClassDecompInfo VAR_DECOMPINFO;


class VarPartition : public IndexPartition
{

public:

   /** constructor */
   VarPartition(
      SCIP*                scip,                /**< scip data structure */
      const char*          name,                /**< name of partition (will be copied) */
      int                  nClasses,            /**< initial number of classes */
      int                  nVars                /**< number of variables to be classified */
   );

   /** copy constructor */
   VarPartition(
      const VarPartition* toCopy              /**< VarPartition to be copied */
   );


   /** destructor */
   ~VarPartition();


   /** creates a new class, returns index of the class */
   int addClass(
      const char* name,                /**< name of the class (will be copied) */
      const char* desc,                /**< description of the class (will be copied) */
      VAR_DECOMPINFO decompInfo        /**< decomposition code of the class */
   );

   /** assigns a variable to a class */
   void assignVarToClass(
      int varindex,                    /**< index of the variable */
      int classindex                   /**< index of the class */
   );

   /** returns a vector containing all possible subsets of the chosen classindices */
   std::vector<std::vector<int>> getAllSubsets(
      bool all,                        /**< true, if ALL classes should be considered */
      bool linking,                    /**< true, if LINKING classes should be considered */
      bool master,                     /**< true, if MASTER classes should be considered */
      bool block                       /**< true, if BLOCK classes should be considered */
   );

   /** returns the decomposition info of a class */
   VAR_DECOMPINFO getClassDecompInfo(
      int classindex                   /**< index of class */
   );

   /** returns the name of the class a variable is assigned to */
   const char* getClassNameOfVar(
      int varindex                    /**< index of variable */
   );


   /** returns the index of the class a variable is assigned to */
   int getClassOfVar(
      int varindex                    /**< index of variable */
   );

   /** returns vector containing the assigned class of each variable */
   const int* getVarsToClasses(
   );

   /** returns the number of variables */
   int getNVars(
   );

   /** returns a vector with the numbers of variables that are assigned to the classes */
   std::vector<int> getNVarsOfClasses(
   );

   /** returns whether a variable is already assigned to a class */
   bool isVarClassified(
      int varindex                    /**< index of variable */
   );


   /** returns partition with reduced number of classes
    *  if the current number of classes is greater than an upper bound
    *  and lower than 2*(upper bound) (returns NULL otherwise) */
   VarPartition* reduceClasses(
      int maxNumberOfClasses           /**< upper bound */
   );

   /** sets the decomposition code of a class */
   void setClassDecompInfo(
      int classindex,                  /**< index of class */
      VAR_DECOMPINFO decompInfo        /**< decomposition code of class */
   );

};


} /* namespace gcg */
#endif /* GCG_CLASS_VARPARTITION_H__ */
