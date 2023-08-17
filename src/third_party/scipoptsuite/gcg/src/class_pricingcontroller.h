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

/**@file   class_pricingcontroller.h
 * @brief  pricing controller managing the pricing strategy
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef CLASS_PRICINGCONTROLLER_H_
#define CLASS_PRICINGCONTROLLER_H_

#include "colpool.h"
#include "pricestore_gcg.h"
#include "class_pricingtype.h"
#include "type_gcgpqueue.h"
#include "type_pricingjob.h"
#include "type_pricingprob.h"
#include "objscip/objscip.h"

namespace gcg {

class Pricingcontroller
{

private:
   SCIP*                 scip_;              /**< SCIP instance (master problem) */
   GCG_PRICINGPROB**     pricingprobs;       /**< pricing problems */
   int                   npricingprobs;      /**< number of pricing problems */
   int                   maxpricingprobs;    /**< capacity of pricingprobs */
   GCG_PRICINGJOB**      pricingjobs;        /**< pricing jobs */
   int                   npricingjobs;       /**< number of pricing jobs */
   int                   maxpricingjobs;     /**< capacity of pricingjobs */

   /* parameters */
   int                   heurpricingiters;   /**< maximum number of heuristic pricing iterations per pricing call and problem */
   int                   maxheurdepth;       /**< maximum depth at which heuristic pricing should be performed (-1 for infinity) */
   char                  sorting;            /**< order by which the pricing problems should be sorted */
   int                   nroundscol;         /**< number of previous pricing rounds for which the number of improving columns should be counted */
   int                   chunksize;          /**< maximal number of pricing problems to be solved during one pricing loop */
   int                   eagerfreq;          /**< frequency at which all pricing problems should be solved */

   /* strategy */
   GCG_PQUEUE*           pqueue;             /**< priority queue containing the pricing jobs */
   SCIP_Real*            score;              /**< scores of the pricing problems */
   int                   maxniters;          /**< maximal possible number of pricing iterations */
   int                   nchunks;            /**< number of pricing problem 'chunks' */
   int                   curchunk;           /**< index of current chunk of pricing problems */
   int                   startchunk;         /**< first chunk considered in a pricing call */
   PricingType*          pricingtype_;       /**< current pricing type */

   /* statistics */
   int                   eagerage;           /**< iterations since last eager iteration */
   int                   nsolvedprobs;       /**< number of completely solved pricing problems during the current pricing loop */


public:
   /** default constructor */
   Pricingcontroller();

   /** constructor */
   Pricingcontroller(SCIP* scip);

   /** destructor */
   virtual ~Pricingcontroller();

   SCIP_RETCODE addParameters();

   SCIP_RETCODE initSol();

   SCIP_RETCODE exitSol();

   /** pricing initialization, called right at the beginning of pricing */
   SCIP_RETCODE initPricing(
      PricingType*          pricingtype         /**< type of pricing */
   );

   /** pricing deinitialization, called when pricing is finished */
   void exitPricing();

   /** setup the priority queue (done once per stabilization round): add all pricing jobs to be performed */
   SCIP_RETCODE setupPriorityQueue(
      SCIP_Real*            dualsolconv         /**< dual solution values / Farkas coefficients of convexity constraints */
   );

   /** get the next pricing job to be performed */
   GCG_PRICINGJOB* getNextPricingjob();

   /** add the information that the next branching constraint must be added,
    * and for the pricing job, reset heuristic pricing counter and flag
    */
   SCIP_RETCODE pricingprobNextBranchcons(
      GCG_PRICINGPROB*      pricingprob         /**< pricing problem structure */
      );

   /** set an individual time limit for a pricing job */
   SCIP_RETCODE setPricingjobTimelimit(
      GCG_PRICINGJOB*       pricingjob          /**< pricing job */
      );

   /** update solution information of a pricing problem */
   void updatePricingprob(
      GCG_PRICINGPROB*      pricingprob,        /**< pricing problem structure */
      GCG_PRICINGSTATUS     status,             /**< new pricing status */
      SCIP_Real             lowerbound,         /**< new lower bound */
      int                   nimpcols            /**< number of new improving columns */
      );

   /** update solution statistics of a pricing job */
   void updatePricingjobSolvingStats(
      GCG_PRICINGJOB*       pricingjob          /**< pricing job */
   );

   /** decide whether a pricing job must be treated again */
   void evaluatePricingjob(
      GCG_PRICINGJOB*       pricingjob,        /**< pricing job */
      GCG_PRICINGSTATUS     status             /**< status of pricing job */
      );

   /** collect solution results from all pricing problems */
   void collectResults(
      GCG_COL**             bestcols,           /**< best found columns per pricing problem */
      SCIP_Bool*            infeasible,         /**< pointer to store whether pricing is infeasible */
      SCIP_Bool*            optimal,            /**< pointer to store whether all pricing problems were solved to optimality */
      SCIP_Real*            bestobjvals,        /**< array to store best lower bounds */
      SCIP_Real*            beststabobj,        /**< pointer to store total lower bound */
      SCIP_Real*            bestredcost,        /**< pointer to store best total reduced cost */
      SCIP_Bool*            bestredcostvalid    /**< pointer to store whether best reduced cost is valid */
      );

   /** check if the next chunk of pricing problems is to be used */
   SCIP_Bool checkNextChunk();

   /** decide whether the pricing loop can be aborted */
   SCIP_Bool canPricingloopBeAborted(
      PricingType*          pricingtype,        /**< type of pricing (reduced cost or Farkas) */
      int                   nfoundcols,         /**< number of negative reduced cost columns found so far */
      int                   nsuccessfulprobs    /**< number of pricing problems solved successfully so far */
      ) const;

   void resetEagerage();

   void increaseEagerage();

   /** for a given problem index, get the corresponding pricing problem (or NULL, if it does not exist) */
   GCG_PRICINGPROB* getPricingprob(
      int                   probnr              /**< index of the pricing problem */
      );

   /** get maximal possible number of pricing iterations */
   int getMaxNIters() const;


private:
   /** comparison operator for pricing jobs w.r.t. their solution priority */
   static
   SCIP_DECL_SORTPTRCOMP(comparePricingjobs);

   /** for each pricing problem, get its corresponding generic branching constraints */
   SCIP_RETCODE getGenericBranchconss();

   /** check if a pricing job is done */
   SCIP_Bool pricingprobIsDone(
      GCG_PRICINGPROB*      pricingprob        /**< pricing problem structure */
      ) const;

   /** check whether the next generic branching constraint of a pricing problem must be considered */
   SCIP_Bool pricingprobNeedsNextBranchingcons(
      GCG_PRICINGPROB*      pricingprob        /**< pricing problem structure */
      ) const;
};

} /* namespace gcg */
#endif /* CLASS_PRICINGCONTROLLER_H_ */
