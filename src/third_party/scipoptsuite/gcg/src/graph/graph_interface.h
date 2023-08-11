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

/**@file   graph_interface.h
 * @brief  miscellaneous graph interface methods
 * @author Martin Bergner
 * @author Annika Thome
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/



#ifndef GCG_GRAPHINTERFACE_H_
#define GCG_GRAPHINTERFACE_H_

#include "objscip/objscip.h"
#include "weights.h"

#include <vector>
#include <fstream>

using std::ifstream;
namespace gcg {

class GraphInterface {
protected:
   std::vector<int> partition;
public:

   GraphInterface() {}

   virtual ~GraphInterface() {}

   /** return a partition of the nodes */
   virtual std::vector<int> getPartition() const { return partition; }

   /** assigns partition to a given node*/
   virtual void setPartition(int i, int nodeid) = 0;

   /** writes the graph to the given file.
    *  The format is graph dependent
    */
   virtual SCIP_RETCODE writeToFile(
      int                fd,                  /**< filename where the graph should be written to */
      SCIP_Bool          writeweights        /**< whether to write weights */
    ) = 0;


   /**
    * reads the partition from the given file.
    * The format is graph dependent. The default is a file with one line for each node a
    */
   virtual SCIP_RETCODE readPartition(
      const char*        filename            /**< filename where the partition is stored */
   ) = 0;


   /** create decomposition based on the read in partition */
   virtual SCIP_RETCODE createDecompFromPartition(
      DEC_DECOMP**       decomp              /**< decomposition structure to generate */
   )
   { /*lint -e715*/
      return SCIP_ERROR;
   }

   virtual SCIP_RETCODE flush() = 0;

};

}

#endif
