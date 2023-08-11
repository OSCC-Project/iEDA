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

/**@file   disp_master.c
 * @ingroup DISPLAYS
 * @brief  master display columns
 * @author Gerald Gamrath
 * @author Christian Puchert
 * @author Martin Bergner
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "disp_master.h"
#include "scip/disp_default.h"
#include "gcg.h"

#define DISP_NAME_ORIGINAL         "original"
#define DISP_DESC_ORIGINAL         "display column printing a display line of the original SCIP instance"
#define DISP_HEAD_ORIGINAL         ""
#define DISP_WIDT_ORIGINAL         5
#define DISP_PRIO_ORIGINAL         80000
#define DISP_POSI_ORIGINAL         3550
#define DISP_STRI_ORIGINAL         TRUE

/*
 * Callback methods
 */

/** copy method for display plugins (called when SCIP copies plugins) */
static
SCIP_DECL_DISPCOPY(dispCopyMaster)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(disp != NULL);

   /* call inclusion method of default SCIP display plugin */
   SCIP_CALL( SCIPincludeDispDefault(scip) );

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' printing a display column of the original SCIP instance */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputOriginal)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_ORIGINAL) == 0);
   assert(scip != NULL);

   SCIP_CALL( SCIPprintDisplayLine(GCGgetOriginalprob(scip), file, SCIP_VERBLEVEL_HIGH, FALSE) );

   return SCIP_OKAY;
}


/*
 * default display columns specific interface methods
 */

/** includes the default display columns in SCIP */
SCIP_RETCODE SCIPincludeDispMaster(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_ORIGINAL, DISP_DESC_ORIGINAL, DISP_HEAD_ORIGINAL,
         SCIP_DISPSTATUS_AUTO, dispCopyMaster, NULL, NULL, NULL, NULL, NULL, SCIPdispOutputOriginal, NULL,
         DISP_WIDT_ORIGINAL, DISP_PRIO_ORIGINAL, DISP_POSI_ORIGINAL, DISP_STRI_ORIGINAL) );

   return SCIP_OKAY;
}
