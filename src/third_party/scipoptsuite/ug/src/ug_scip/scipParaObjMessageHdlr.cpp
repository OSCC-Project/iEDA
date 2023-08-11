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

/**@file    scipParaObjMessageHdlr.cpp
 * @brief   SCIP message handler for ParaSCIP and FiberSCIP.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <cassert>
#include <iostream>

#include "scipParaObjMessageHdlr.h"

using namespace ParaSCIP;


/** constructor */
ScipParaObjMessageHdlr::ScipParaObjMessageHdlr(
   UG::ParaComm           *inComm,
   FILE*              inFile,
   SCIP_Bool          inQuiet,
   SCIP_Bool          inBufferedoutput      /**< should the output be buffered up to the next newline? */
   )  : ObjMessagehdlr(inBufferedoutput)
{
   comm = inComm;     // for debugging
   logfile = inFile;
   quiet = inQuiet;
}

ScipParaObjMessageHdlr::~ScipParaObjMessageHdlr(
      )
{
}


void ScipParaObjMessageHdlr::logMessage(
   FILE*                 file,               /**< file stream to print message into */
   const char*           msg                 /**< message to print */
   )
{
   if( file == NULL )
      file = stdout;
   if( !quiet || (file != stdout && file != stderr) )
   {
      fputs(msg, file);
      fflush(file);
   }
   if( logfile != NULL && (file == stdout || file == stderr) )
   {
      fputs(msg, logfile);
      fflush(logfile);
   }
}

/** error message print method of message handler
*
*  This method is invoked, if SCIP wants to display an error message to the screen or a file
*/
void ScipParaObjMessageHdlr::scip_error(
   SCIP_MESSAGEHDLR*  messagehdlr,        /**< the message handler itself */
   FILE*              file,               /**< file stream to print into */
   const char*        msg                 /**< string to output into the file */
   )
{
   logMessage(file, msg);
}

/** warning message print method of message handler
 *
 *  This method is invoked, if SCIP wants to display a warning message to the screen or a file
 */
void ScipParaObjMessageHdlr::scip_warning(
   SCIP_MESSAGEHDLR*  messagehdlr,        /**< the message handler itself */
   FILE*              file,               /**< file stream to print into */
   const char*        msg                 /**< string to output into the file */
   )
{
   // logMessage(mymessagehdlrdata, file, msg);
   logMessage(file, msg);
}

/** dialog message print method of message handler
 *
 *  This method is invoked, if SCIP wants to display a dialog message to the screen or a file
 */
void ScipParaObjMessageHdlr::scip_dialog(
   SCIP_MESSAGEHDLR*  messagehdlr,        /**< the message handler itself */
   FILE*              file,               /**< file stream to print into */
   const char*        msg                 /**< string to output into the file */
   )
{
   logMessage(file, msg);
}

/** info message print method of message handler
 *
 *  This method is invoked, if SCIP wants to display an information message to the screen or a file
 */
void ScipParaObjMessageHdlr::scip_info(
   SCIP_MESSAGEHDLR*  messagehdlr,        /**< the message handler itself */
   FILE*              file,               /**< file stream to print into */
   const char*        msg                 /**< string to output into the file */
   )
{
   logMessage(file, msg);
}

extern "C" {
/** error message function as used by SCIP */
SCIP_DECL_ERRORPRINTING(scip_errorfunction)
{
   assert( data != 0 );
   ScipParaObjMessageHdlr* objmessagehdlr = (ScipParaObjMessageHdlr*) data;

   objmessagehdlr->scip_error(0, objmessagehdlr->getlogfile(), msg);
}
}
