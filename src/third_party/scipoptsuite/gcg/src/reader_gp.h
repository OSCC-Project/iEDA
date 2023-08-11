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

/**@file   reader_gp.h
 * @brief  GP file reader writing decompositions to gnuplot files
 * @author Martin Bergner
 * @author Hanna Franzen
 * @ingroup FILEREADERS
 *
 * This reader can write visualizations of partialdecs to a .gp file.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_READER_GP_H__
#define GCG_READER_GP_H__

#include "scip/scip.h"
#include "type_decomp.h"

#ifdef __cplusplus
extern "C" {
#endif


/** Output format of gnuplot. Specifies the output format that gnuplot will produce. */
enum GPOutputFormat
{
   GP_OUTPUT_FORMAT_PDF,
   GP_OUTPUT_FORMAT_PNG,
   GP_OUTPUT_FORMAT_SVG
};
typedef enum GPOutputFormat GP_OUTPUT_FORMAT;

/** Includes the gp file reader into SCIP
 * @returns SCIP status */
extern
SCIP_RETCODE SCIPincludeReaderGp(
   SCIP*                 scip                /**< SCIP data structure */
   );

/* Writes a visualization for the given partialdec */
extern SCIP_RETCODE GCGwriteGpVisualizationFormat(
   SCIP* scip,             /**< SCIP data structure */
   char* filename,         /**< filename (including path) to write to */
   char* outputname,       /**< filename for compiled output file */
   int partialdecid,       /**< id of partialdec to visualize */
   GP_OUTPUT_FORMAT outputformat /**< the output format which gnuplot should emit */
   );

/** Writes a visualization as .pdf file for the given partialdec
 * @returns SCIP status */
extern
SCIP_RETCODE GCGwriteGpVisualization(
   SCIP* scip,             /**< SCIP data structure */
   char* filename,         /**< filename (including path), location of the output*/
   char* outputname,       /**< outputname is the name of the file for the compiled gnuplot output file */
   int partialdecid             /**< id of partialdec to visualize */
   );

/** Creates a block matrix and outputs its visualization as .pdf file
 * @returns SCIP return code
 * */
extern
SCIP_RETCODE GCGWriteGpDecompMatrix(
   SCIP*                 scip,               /**< scip data structure */
   const char*           filename,           /**< filename the output should be written to (including directory) */
   const char*           workfolder,         /**< directory in which should be worked */
   SCIP_Bool             originalmatrix      /**< should the original (or transformed) matrix be written */
);

#ifdef __cplusplus
}
#endif

#endif
