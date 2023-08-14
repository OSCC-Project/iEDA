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

/**@file   reader_tex.h
 * @brief  tex file reader for writing decomposition details to LaTeX files
 * @author Hanna Franzen
 * @ingroup FILEREADERS

 * This reader can write visualizations, family trees and reports of partialdecs to a .tex LaTeX file.
 * The gp reader might be required for visualizations.

 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_READER_TEX_H__
#define GCG_READER_TEX_H__

#include "scip/scip.h"
#include "type_decomp.h"
#include "cons_decomp.h"

#ifdef __cplusplus
extern "C" {
#endif


/** Includes the tex file reader into SCIP
 *
 * @returns SCIP status */
extern SCIP_RETCODE SCIPincludeReaderTex(
   SCIP* scip     /**< SCIP data structure */
   );

/** Writes visualization LaTeX code for the given partialdec
 *
 * @returns SCIP status */
extern SCIP_RETCODE GCGwriteTexVisualization(
   SCIP* scip,             /**< SCIP data structure */
   FILE* file,             /**< file in which to write the LaTeX code */
   int partialdecid,            /**< id of partialdec to visualize */
   SCIP_Bool statistics,   /**< additionally to picture show statistics */
   SCIP_Bool usegp         /**< true if the gp reader should be used for the image generation (instead of tikz) */
   );

/** Writes a report for the given partialdecs
 *
 * @note  *npartialdecs will be set to the number of actually written decompositions.
 * @returns SCIP status */
extern SCIP_RETCODE GCGwriteTexReport(
   SCIP* scip,             /**< SCIP data structure */
   FILE* file,             /**< file in which to put the LaTeX code */
   int* partialdecids,     /**< ids of partialdecs to visualize */
   int* npartialdecs,      /**< number of partialdecs to visualize */
   SCIP_Bool titlepage,    /**< true if a title page should be included in the document */
   SCIP_Bool toc,          /**< true if an interactive table of contents should be included */
   SCIP_Bool statistics,   /**< true if statistics for each partialdec should be included */
   SCIP_Bool usegp         /**< true if the gp reader should be used for the image generation */
   );

/** Makes a new makefile and readme for the given .tex file
 *
 * @returns SCIP status */
extern SCIP_RETCODE GCGtexWriteMakefileAndReadme(
   SCIP* scip,          /**< SCIP data structure */
   FILE* file,          /**< tex file for which the makefile & readme are generated */
   SCIP_Bool usegp,     /**< true if the gp reader was used for creation of images */
   SCIP_Bool compiletex /**< true if there are tex files to be compiled before main document */
   );

#ifdef __cplusplus
}
#endif


#endif /* GCG_READER_TEX_H__ */

