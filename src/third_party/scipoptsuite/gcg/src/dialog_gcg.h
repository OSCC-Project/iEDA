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

/**@file   dialog_gcg.h
 * @ingroup DIALOGS
 * @brief  GCG user interface dialog
 * @author Tobias Achterberg
 * @author Timo Berthold
 * @author Gerald Gamrath
 * @author Martin Bergner
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_DIALOG_GCG_H__
#define GCG_DIALOG_GCG_H__


#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif


/** dialog execution method for the display additionalstatistics command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecDisplayAdditionalStatistics);

/** dialog execution method for the display statistics command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecDisplayStatistics);

/** dialog execution method print complete detection information */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecPrintDetectionInformation);

/** dialog execution method for adding block number candidate */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecChangeAddBlocknr);

/** dialog execution method for the display detectors command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecDisplayDetectors);

/** dialog execution method for the display solvers command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecDisplaySolvers);

/** dialog execution method for the display decomposition command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecDisplayDecomposition);

/** dialog execution method for the display nblockscandidates command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecDisplayNBlockcandidates);

/** dialog execution method for the presolve command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecPresolve);

/** dialog execution method for the master command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecSetMaster);

/** dialog execution method for the set loadmaster command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecSetLoadmaster);

/** dialog execution method for the detect command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecDetect);

/** dialog execution method for the select command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecSelect);

/** dialog execution method for the transform command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecTransform);

/** dialog execution method for the optimize command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecOptimize);

/** dialog execution method for the set detectors fast command */
extern
SCIP_DECL_DIALOGEXEC(SCIPdialogExecSetDetectorsFast);

/** dialog execution method for the set detectors off command */
extern
SCIP_DECL_DIALOGEXEC(SCIPdialogExecSetDetectorsOff);

/** dialog execution method for the set detectors default command */
extern
SCIP_DECL_DIALOGEXEC(SCIPdialogExecSetDetectorsDefault);

/** dialog execution method for the set detectors aggressive command */
extern
SCIP_DECL_DIALOGEXEC(SCIPdialogExecSetDetectorsAggressive);

/** dialog execution method for the set heuristics fast command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecSetHeuristicsFast);

/** dialog execution method for the set heuristics off command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecSetHeuristicsOff);

/** dialog execution method for the set heuristics aggressive command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecSetHeuristicsAggressive);

/** dialog execution method for the set separators default command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecSetSeparatorsDefault);

/** dialog execution method for the set separators fast command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecSetSeparatorsFast);

/** dialog execution method for the set separators off command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecSetSeparatorsOff);

/** dialog execution method for the set separators aggressive command */
extern
SCIP_DECL_DIALOGEXEC(GCGdialogExecSetSeparatorsAggressive);

/** creates a root dialog
 * @returns SCIP return code */
extern
SCIP_RETCODE GCGcreateRootDialog(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIALOG**         root                /**< pointer to store the root dialog */
   );

/** includes or updates the GCG dialog menus in SCIP */
extern
SCIP_RETCODE SCIPincludeDialogGcg(
   SCIP*                 scip                /**< SCIP data structure */
   );

#ifdef __cplusplus
}
#endif

#endif
