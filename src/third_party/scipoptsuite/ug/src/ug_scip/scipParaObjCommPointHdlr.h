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

/**@file    scipParaObjCommPointHdlr.h
 * @brief   Event handlr for communication point.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_PARA_COMM_POINT_HDLR_H__
#define __SCIP_PARA_COMM_POINT_HDLR_H__

#include <cstring>
#include "scipParaComm.h"
#include "scipParaInstance.h"
#include "scipParaSolver.h"
#include "scipParaObjLimitUpdator.h"
#include "scipParaParamSet.h"
#include "objscip/objeventhdlr.h"
#include "scip/scipdefplugins.h"

namespace ParaSCIP
{

class ScipParaSolver;

/** C++ wrapper object for event handlers */
class ScipParaObjCommPointHdlr : public scip::ObjEventhdlr
{
   UG::ParaComm   *paraComm;
   ScipParaSolver *scipParaSolver;
   ScipParaObjLimitUpdator *scipParaObjLimitUpdator;
   SCIP           *scipToCheckRootSolvability;
   SCIP           *originalScip;
   bool           needToSendNode;
   bool           originalSelectionStrategy;
   // int            originalPriority;
   SCIP_Longint   previousNNodesSolved;
   SCIP_Longint   previousLpIter;
   const char     *changeNodeSelName;
   // double         previousCommTime;
   bool           cloned;                  // indicate that this hander is cloned or not
   bool           interrupting;            // indicate that this handler called interrupt or not
   bool           startedCollectingNodesForInitialRampUp;  // initial ramp-up collecting has been started
   // int            nThghtendLbs;
   // int            nTightendUbs;
   void           processNewSolution(SCIP *scip, SCIP_EVENT* event);
   // void           checkRootNodeSolvabilityAndSendParaNode(SCIP *scip);
   bool           checkRootNodeSolvabilityAndSendParaNode(SCIP *scip);
   void           sendNode( SCIP *scip, SCIP_NODE* node, int depth, int nBranchVars, SCIP_VAR **branchVars, SCIP_Real *branchBounds, SCIP_BOUNDTYPE *boundTypes );
   void           changeSearchStrategy(SCIP *scip);
   bool           ifFeasibleInOriginalProblem(SCIP *scip, int nNewBranchVars, SCIP_VAR **newBranchVars, SCIP_Real *newBranchBounds);
public:
   ScipParaObjCommPointHdlr(
            UG::ParaComm   *comm,
            ScipParaSolver *solver,
	    ScipParaObjLimitUpdator *updator
         )
         : scip::ObjEventhdlr::ObjEventhdlr(solver->getScip(), "ScipParaObjCommPointHdlr", "Event handler to communicate with LC"),
           paraComm(comm), scipParaSolver(solver), scipParaObjLimitUpdator(updator), scipToCheckRootSolvability(0), originalScip(0), needToSendNode(false),
           originalSelectionStrategy(true), previousNNodesSolved(0), previousLpIter(0),
           cloned(false), interrupting(false), startedCollectingNodesForInitialRampUp(false) // ,
           // nThghtendLbs(0), nTightendUbs(0)
   {
      changeNodeSelName = scipParaSolver->getChangeNodeSelName();
      if( !cloned && scipParaSolver->getParaParamSet()->getBoolParamValue(RootNodeSolvabilityCheck) )
      {
         /* initialize SCIP to check root solvability */
         SCIP_CALL_ABORT( SCIPcreate(&scipToCheckRootSolvability) );
         /* include default SCIP plugins */
         SCIP_CALL_ABORT( SCIPincludeDefaultPlugins(scipToCheckRootSolvability) );
         ScipParaInstance* scipParaInstance = dynamic_cast< ScipParaInstance* >(scipParaSolver->getParaInstance());
         scipParaInstance->createProblem(scipToCheckRootSolvability,
               solver->getParaParamSet()->getIntParamValue(UG::InstanceTransferMethod),
               solver->getParaParamSet()->getBoolParamValue(UG::NoPreprocessingInLC),
               solver->getParaParamSet()->getBoolParamValue(UG::UseRootNodeCuts),
               NULL,
               NULL,
               NULL,
               NULL
         );   // LC presolving setting file should not be set, when it does this check!
      }
      if( scipParaSolver->getParaParamSet()->getIntParamValue(UG::RampUpPhaseProcess) != 0 ||
            !scipParaSolver->getParaParamSet()->getBoolParamValue(UG::CollectOnce) )
      {
         startedCollectingNodesForInitialRampUp = true;
      }
   }

   ScipParaObjCommPointHdlr(
		    UG::ParaComm   *comm,
            ScipParaSolver *solver,
            SCIP           *subScip,
            SCIP           *inOriginalScip,
            bool inCloned
         )          : scip::ObjEventhdlr::ObjEventhdlr(subScip, "ScipParaObjCommPointHdlr", "Event handler to communicate with LC"),
         paraComm(comm), scipParaSolver(solver), scipParaObjLimitUpdator(0), scipToCheckRootSolvability(0), originalScip(inOriginalScip), needToSendNode(false),
         originalSelectionStrategy(true), previousNNodesSolved(0), previousLpIter(0), changeNodeSelName(0),
         cloned(inCloned), interrupting(false), startedCollectingNodesForInitialRampUp(false) //,
         // nThghtendLbs(0), nTightendUbs(0)
   {
      if( scipParaSolver->getParaParamSet()->getIntParamValue(UG::RampUpPhaseProcess) != 0 ||
            !scipParaSolver->getParaParamSet()->getBoolParamValue(UG::CollectOnce) )
      {
         startedCollectingNodesForInitialRampUp = true;
      }
   }

   /** destructor */
   ~ScipParaObjCommPointHdlr(
         )
   {
      if( scipToCheckRootSolvability )
      {
         SCIP_CALL_ABORT( SCIPfree(&scipToCheckRootSolvability) );
      }
   }

   void resetCommPointHdlr()
   {
      scipParaSolver->setOriginalPriority();
      originalSelectionStrategy = true;
      needToSendNode = false;
      interrupting = false;
   }

   /** clone method, used to copy plugins which are not constraint handlers or variable pricer plugins */
   ObjCloneable* clone(
      SCIP*           scip                /**< SCIP data structure */
      ) const
   {
      return new ScipParaObjCommPointHdlr(paraComm, scipParaSolver, scip, scipParaSolver->getScip(), true);
   }
   /** returns whether the objective plugin is copyable */
   SCIP_Bool iscloneable(
      void
      ) const
   {
      return true;
   }

   // SCIP *getParentScip(){ return parentScip; }
   bool isColne(){return cloned;}

   void setOriginalNodeSelectionStrategy()
   {
      scipParaSolver->setOriginalPriority();
      originalSelectionStrategy = true;
   }

   /** destructor of event handler to free user data (called when SCIP is exiting) */
   virtual SCIP_RETCODE scip_free(
      SCIP*              scip,               /**< SCIP data structure */
      SCIP_EVENTHDLR*    eventhdlr           /**< the event handler itself */
      )
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** initialization method of event handler (called after problem was transformed) */
   virtual SCIP_RETCODE scip_init(
      SCIP*              scip,               /**< SCIP data structure */
      SCIP_EVENTHDLR*    eventhdlr           /**< the event handler itself */
      )
   {  /*lint --e{715}*/
      SCIP_CALL( SCIPcatchEvent( scip,
            ( SCIP_EVENTTYPE_GBDCHANGED |
                  SCIP_EVENTTYPE_BOUNDTIGHTENED |
                  SCIP_EVENTTYPE_LPEVENT |
                  SCIP_EVENTTYPE_ROWEVENT |
                  // SCIP_EVENTTYPE_NODEFOCUSED |
                  SCIP_EVENTTYPE_NODEEVENT |
                  SCIP_EVENTTYPE_BESTSOLFOUND // |
		            // SCIP_EVENTTYPE_COMM
                  )
            , eventhdlr, NULL, NULL) );

      if( !cloned )
      {
         int        i;
         int        nvars;
         SCIP_VAR** vars;
         // std::cout << "catching events in commpoint eventhdlr" << std::endl;
         nvars = SCIPgetNVars(scip);
         vars = SCIPgetVars(scip);
         // SCIP_CALL( SCIPcatchEvent(scip, SCIP_EVENTTYPE_VARADDED, eventhdlr, NULL, NULL) );
         for( i = 0; i < nvars ; ++i )
         {
            SCIP_CALL( SCIPcatchVarEvent(scip, vars[i], SCIP_EVENTTYPE_GBDCHANGED, eventhdlr, NULL, NULL) );
         }
      }

      interrupting = false;
      return SCIP_OKAY;
   }

   void issueInterrupt()
   {
      interrupting = true;
   }

   bool isInterrupting()
   {
      return interrupting;
   }

   /** deinitialization method of event handler (called before transformed problem is freed) */
   virtual SCIP_RETCODE scip_exit(
      SCIP*              scip,               /**< SCIP data structure */
      SCIP_EVENTHDLR*    eventhdlr           /**< the event handler itself */
      )
   {  /*lint --e{715}*/
      /* notify SCIP that your event handler wants to drop the event type best solution found */
      return SCIP_OKAY;
   }

   /** solving process initialization method of event handler (called when branch and bound process is about to begin)
    *
    *  This method is called when the presolving was finished and the branch and bound process is about to begin.
    *  The event handler may use this call to initialize its branch and bound specific data.
    *
    */
   virtual SCIP_RETCODE scip_initsol(
      SCIP*              scip,               /**< SCIP data structure */
      SCIP_EVENTHDLR*    eventhdlr           /**< the event handler itself */
      )
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** solving process deinitialization method of event handler (called before branch and bound process data is freed)
    *
    *  This method is called before the branch and bound process is freed.
    *  The event handler should use this call to clean up its branch and bound data.
    */
   virtual SCIP_RETCODE scip_exitsol(
      SCIP*              scip,               /**< SCIP data structure */
      SCIP_EVENTHDLR*    eventhdlr           /**< the event handler itself */
      )
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** frees specific constraint data */
   virtual SCIP_RETCODE scip_delete(
      SCIP*              scip,               /**< SCIP data structure */
      SCIP_EVENTHDLR*    eventhdlr,          /**< the event handler itself */
      SCIP_EVENTDATA**   eventdata           /**< pointer to the event data to free */
      )
   {  /*lint --e{715}*/
      return SCIP_OKAY;
   }

   /** execution method of event handler
    *
    *  Processes the event. The method is called every time an event occurs, for which the event handler
    *  is responsible. Event handlers may declare themselves resposible for events by calling the
    *  corresponding SCIPcatch...() method. This method creates an event filter object to point to the
    *  given event handler and event data.
    */
   virtual SCIP_RETCODE scip_exec(
      SCIP*              scip,               /**< SCIP data structure */
      SCIP_EVENTHDLR*    eventhdlr,          /**< the event handler itself */
      SCIP_EVENT*        event,              /**< event to process */
      SCIP_EVENTDATA*    eventdata           /**< user data for the event */
      );
};

}  /* namespace ParaSCIP */

#endif // __SCIP_PARA_COMM_POINT_HDLR_H__
