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

/**@file   reader_gp.cpp
 * @brief  GP file reader writing decompositions to gnuplot files
 * @author Martin Bergner
 * @author Hanna Franzen
 * @author Michael Bastubbe
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include <cstring>
#include <fstream>

#include "scip/scip.h"

#include "reader_gp.h"
#include "scip_misc.h"
#include "struct_decomp.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include "pub_decomp.h"
#include "params_visu.h"

#include "class_partialdecomp.h"
#include "class_detprobdata.h"
#include "miscvisualization.h"

#define READER_NAME             "gpreader"
#define READER_DESC             "gnuplot file writer for partialdec visualization"
#define READER_EXTENSION        "gp"

#define SCALING_FACTOR_NONZEROS 0.6

using namespace gcg;

/*
 * Callback methods of reader
 */


/** Destructor of reader to free user data (called when SCIP is exiting) */
static
SCIP_DECL_READERFREE(readerFreeGp)
{
   assert(strcmp(SCIPreaderGetName(reader), READER_NAME) == 0);
   return SCIP_OKAY;
}


/** Problem writing method of reader */
static
SCIP_DECL_READERWRITE(readerWriteGp)
{
   PARTIALDECOMP* partialdec;
   char filename[PATH_MAX];
   char outputname[SCIP_MAXSTRLEN];

   assert(scip != NULL);
   assert(file != NULL);

   /* get partialdec to write */
   partialdec = DECgetPartialdecToWrite(scip, transformed);

   if(partialdec == NULL)
   {
      SCIPerrorMessage("Could not find Partialdecomp to write!\n");
      *result = SCIP_DIDNOTRUN;
   }
   else
   {
      /* reader internally works with the filename instead of the C FILE type */
      GCGgetFilePath(file, filename);

      /* get filename for compiled file */
      GCGgetVisualizationFilename(scip, partialdec, "pdf", outputname);
      strcat(outputname, ".pdf");

      GCGwriteGpVisualization(scip, filename, outputname, partialdec->getID() );

      *result = SCIP_SUCCESS;
   }

   return SCIP_OKAY;
}


/** Write gnuplot file header with terminal etc.
 * @returns SCIP status */
static
SCIP_RETCODE writeGpHeader(
   SCIP*                 scip,               /**< SCIP data structure */
   char*                 filename,           /**< filename (including path) to write to */
   const char*           outputname,         /**< the filename to which gnuplot should compile the visualization */
   GP_OUTPUT_FORMAT      outputformat        /**< the output format which gnuplot should emit */
   )
{
   std::ofstream ofs;

   ofs.open( filename, std::ofstream::out );

   /* set output format and file */
   ofs << "set encoding utf8" << std::endl;

   ofs << "set terminal ";
   switch (outputformat)
   {
   case GP_OUTPUT_FORMAT_PDF:
      ofs << "pdf";
      break;
   case GP_OUTPUT_FORMAT_PNG:
      ofs << "pngcairo";
      break;
   case GP_OUTPUT_FORMAT_SVG:
      ofs << "svg";
      break;
   }
   ofs << std::endl;

   ofs << "set output \"" << outputname << "\"" << std::endl;

   ofs.close();

   return SCIP_OKAY;
}


/** Adds gnuplot code to given file that contains a box with given coordinates and color
 * @returns SCIP status */
static
SCIP_RETCODE drawGpBox(
   SCIP* scip,       /**< SCIP data structure */
   char* filename,   /**< filename (including path) to write to */
   int objectid,     /**< id number of box (>0), must be unique */
   int x1,           /**< x value of lower left vertex coordinate */
   int y1,           /**< y value of lower left vertex coordinate */
   int x2,           /**< x value of upper right vertex coordinate */
   int y2,           /**< y value of upper right vertex coordinate */
   const char* color /**< color hex code (e.g. #000000) for box filling */
   )
{
   std::ofstream ofs;
   ofs.open( filename, std::ofstream::out | std::ofstream::app );

   ofs << "set object " << objectid << " rect from " << x1 << "," << y1 << " to " << x2 << "," << y2
      << " fc rgb \"" << color << "\"" << " lc rgb \"" << SCIPvisuGetColorLine(scip) << "\"" << std::endl;

   ofs.close();
   return SCIP_OKAY;
}


/** Writes gnuplot code to given file that contains all nonzero points
 * @returns SCIP status */
static
SCIP_RETCODE writeGpNonzeros(
   SCIP* scip,             /**< SCIP data structure */
   const char* filename,   /**< filename to write to (including path & extension) */
   PARTIALDECOMP* partialdec,           /**< PARTIALDECOMP for which the nonzeros should be visualized */
   float radius            /**< radius of the dots (scaled concerning matrix dimensions)*/
   )
{
   int radiusscale;
   std::vector<int> orderToRows(partialdec->getNConss(), -1);
   std::vector<int> rowToOrder(partialdec->getNConss(), -1);
   std::vector<int> orderToCols(partialdec->getNVars(), -1);
   std::vector<int> colsToOrder(partialdec->getNVars(), -1);
   int counterrows = 0;
   int countercols = 0;
   std::ofstream ofs;
   DETPROBDATA* detprobdata;

   detprobdata = partialdec->getDetprobdata();

   /* order of constraints */
   /* master constraints */
   for( int i = 0; i < partialdec->getNMasterconss() ; ++i )
   {
      int rowidx = partialdec->getMasterconss()[i];
      orderToRows[counterrows] = rowidx;
      rowToOrder[rowidx] = counterrows;
      ++counterrows;
   }

   /* block constraints */
   for( int b = 0; b < partialdec->getNBlocks(); ++b )
   {
      for(int i = 0; i < partialdec->getNConssForBlock(b); ++i )
      {
         int rowidx = partialdec->getConssForBlock(b)[i];
         orderToRows[counterrows] = rowidx;
         rowToOrder[rowidx] = counterrows;
         ++counterrows;
      }
   }

   /* open constraints */
   for( int i = 0; i < partialdec->getNOpenconss(); ++i )
   {
      int rowidx = partialdec->getOpenconss()[i];
      orderToRows[counterrows] = rowidx;
      rowToOrder[rowidx] = counterrows;
      ++counterrows;
   }

   /* order of variables */

   /* linking variables */
   for( int i = 0; i < partialdec->getNLinkingvars() ; ++i )
   {
      int colidx = partialdec->getLinkingvars()[i];
      orderToCols[countercols] = colidx;
      colsToOrder[colidx] = countercols;
      ++countercols;
   }

   /* master variables */
   for( int i = 0; i < partialdec->getNMastervars() ; ++i )
   {
      int colidx = partialdec->getMastervars()[i];
      orderToCols[countercols] = colidx;
      colsToOrder[colidx] = countercols;
      ++countercols;
   }

   /* block variables */
   for( int b = 0; b < partialdec->getNBlocks(); ++b )
   {
      for(int i = 0; i < partialdec->getNVarsForBlock(b); ++i )
      {
         int colidx = partialdec->getVarsForBlock(b)[i];
         orderToCols[countercols] = colidx;
         colsToOrder[colidx] = countercols;
         ++countercols;
      }
      for(int i = 0; i < partialdec->getNStairlinkingvars(b); ++i )
      {
         int colidx = partialdec->getStairlinkingvars(b)[i];
         orderToCols[countercols] = colidx;
         colsToOrder[colidx] = countercols;
         ++countercols;
      }
   }

   /* open vars */
   for( int i = 0; i < partialdec->getNOpenvars() ; ++i )
   {
      int colidx = partialdec->getOpenvars()[i];
      orderToCols[countercols] = colidx;
      colsToOrder[colidx] = countercols;
      ++countercols;
   }

   ofs.open (filename, std::ofstream::out | std::ofstream::app );

   /* scaling factor concerning user wishes */
   SCIPgetIntParam(scip, "visual/nonzeroradius", &radiusscale);
   radius *= radiusscale;


  /* dot should be visible, so enforce minimum radius of 0.01 */
   if( radius < 0.01 )
      radius = 0.01;

   /* start writing dots */
   ofs << "set style line 99 lc rgb \"" << SCIPvisuGetColorNonzero(scip) << "\"  " << std::endl;
   ofs << "plot \"-\" using 1:2:(" << radius << ") with dots ls 99 notitle " << std::endl;
   /* write scatter plot */
   for( int row = 0; row < partialdec->getNConss(); ++row )
   {
      int cons;
      cons = orderToRows[row];
      for( int v = 0; v < detprobdata->getNVarsForCons(cons); ++v )
      {
         int col;
         int var;
         var = detprobdata->getVarsForCons(cons)[v];
         col = colsToOrder[var];
         ofs << col + 0.5 << " " << row + 0.5 << std::endl;
      }

   }

   /* end writing dots */
   ofs << "e" << std::endl;

   ofs.close();

   return SCIP_OKAY;
}

/** \brief Adds the gnuplot body of the partialdec visualization to the given file
 *
 * Adds the gnuplot body of the partialdec visualization to the given file.
 * This includes axes, blocks and nonzeros. */
static
SCIP_RETCODE writeGpPartialdec(
   SCIP* scip,             /**< SCIP data structure */
   char* filename,         /**< filename (including path) to write to */
   PARTIALDECOMP* partialdec            /**< PARTIALDECOMP for which the nonzeros should be visualized */
   )
{
   int rowboxcounter = 0;
   int colboxcounter = 0;
   int objcounter = 0;
   int nvars;
   int nconss;
   SCIP_Bool writematrix;

   nvars = partialdec->getNVars();
   nconss = partialdec->getNConss();

   std::ofstream ofs;
   ofs.open( filename, std::ofstream::out | std::ofstream::app );

   writematrix = FALSE;

   if ( partialdec->getNBlocks() == 1 && partialdec->isComplete() && partialdec->getNMasterconss() == 0
      && partialdec->getNLinkingvars() == 0  && partialdec->getNMastervars() == 0 )
      writematrix = TRUE;

   /* set coordinate range */
   if( !writematrix )
   {
      ofs << "set xrange [-1:" << nvars << "]" << std::endl;
      ofs << "set yrange[" << nconss << ":-1]" << std::endl;
   }
   else
   {
      ofs << "set xrange [0:" << nvars << "]" << std::endl;
      ofs << "set yrange[" << nconss << ":0]" << std::endl;

      ofs << " set xtics nomirror " << std::endl;
      ofs << " set ytics nomirror" << std::endl;
      ofs << " set xtics out " << std::endl;
      ofs << " set ytics out" << std::endl;
   }

   /* --- draw boxes ---*/

   if( !writematrix )
   {
      /* linking vars */
      if(partialdec->getNLinkingvars() != 0)
      {
         ++objcounter; /* has to start at 1 for gnuplot */
         drawGpBox( scip, filename, objcounter, 0, 0, partialdec->getNLinkingvars(), partialdec->getNConss(),
            SCIPvisuGetColorLinking(scip) );
         colboxcounter += partialdec->getNLinkingvars();
      }

      /* masterconss */
      if(partialdec->getNMasterconss() != 0)
      {
         ++objcounter;
         drawGpBox( scip, filename, objcounter, 0, 0, partialdec->getNVars(), partialdec->getNMasterconss(),
            SCIPvisuGetColorMasterconss(scip) );
         rowboxcounter += partialdec->getNMasterconss();
      }

      /* mastervars */
      if(partialdec->getNMastervars() != 0)
      {
         ++objcounter;
         //      drawGpBox( scip, filename, objcounter, colboxcounter, 0, partialdec->getNMastervars()+colboxcounter,
         //         partialdec->getNMasterconss(), SCIPvisuGetColorMastervars() );
         colboxcounter += partialdec->getNMastervars();
      }

      /* blocks (blocks are not empty) */
      for( int b = 0; b < partialdec->getNBlocks() ; ++b )
      {
         ++objcounter;
         drawGpBox(scip, filename, objcounter, colboxcounter, rowboxcounter,
            colboxcounter + partialdec->getNVarsForBlock(b), rowboxcounter + partialdec->getNConssForBlock(b),
            SCIPvisuGetColorBlock(scip));
         colboxcounter += partialdec->getNVarsForBlock(b);

         if( partialdec->getNStairlinkingvars(b) != 0 )
         {
            ++objcounter;
            drawGpBox( scip, filename, objcounter, colboxcounter, rowboxcounter,
               colboxcounter + partialdec->getNStairlinkingvars(b),
               rowboxcounter + partialdec->getNConssForBlock(b) + partialdec->getNConssForBlock(b+1),
               SCIPvisuGetColorStairlinking(scip) );
         }
         colboxcounter += partialdec->getNStairlinkingvars(b);
         rowboxcounter += partialdec->getNConssForBlock(b);
      }

      /* open */
      if(partialdec->getNOpenvars() != 0)
      {
         ++objcounter;
         drawGpBox( scip, filename, objcounter, colboxcounter, rowboxcounter, colboxcounter + partialdec->getNOpenvars(),
            rowboxcounter+partialdec->getNOpenconss(), SCIPvisuGetColorOpen(scip) );
         colboxcounter += partialdec->getNOpenvars();
         rowboxcounter += partialdec->getNOpenconss();
      }
   }
   /* --- draw nonzeros --- */
   if( SCIPvisuGetDraftmode(scip) == FALSE )
   {
      /* scale the dots according to matrix dimensions here */
      writeGpNonzeros(scip, filename, partialdec, SCIPvisuGetNonzeroRadius(scip, partialdec->getNVars(), partialdec->getNConss(),
         SCALING_FACTOR_NONZEROS) );
   }
   else
   {
      ofs << "plot \"-\" using 1:2:(0) notitle with circles fill solid lw 2 fc rgb \"black\" "
         << std::endl << "0 0" << std::endl << "e" << std::endl;
   }

   ofs.close();

   return SCIP_OKAY;
}


/* Writes a visualization for the given partialdec */
SCIP_RETCODE GCGwriteGpVisualizationFormat(
   SCIP* scip,             /**< SCIP data structure */
   char* filename,         /**< filename (including path) to write to */
   char* outputname,       /**< filename for compiled output file */
   int partialdecid,       /**< id of partialdec to visualize */
   GP_OUTPUT_FORMAT outputformat /**< the output format which gnuplot should emit */
   )
{
   DETPROBDATA* detprobdata;
   PARTIALDECOMP* partialdec;

   /* get partialdec and detprobdata */
   partialdec = GCGconshdlrDecompGetPartialdecFromID(scip, partialdecid);
   if( partialdec == NULL )
   {
      SCIPerrorMessage("Could not find PARTIALDECOMP!\n");
      return SCIP_ERROR;
   }

   detprobdata = partialdec->getDetprobdata();
   if( detprobdata == NULL )
   {
      SCIPerrorMessage("Could not find DETPROBDATA!\n");
      return SCIP_ERROR;
   }

   /* write file */
   writeGpHeader(scip, filename, outputname, outputformat );
   writeGpPartialdec(scip, filename, partialdec );

   return SCIP_OKAY;
}

/* Writes a visualization as .pdf file for the given partialdec */
SCIP_RETCODE GCGwriteGpVisualization(
   SCIP* scip,             /**< SCIP data structure */
   char* filename,         /**< filename (including path) to write to */
   char* outputname,       /**< filename for compiled output file */
   int partialdecid             /**< id of partialdec to visualize */
   )
{
   return GCGwriteGpVisualizationFormat(scip, filename, outputname, partialdecid, GP_OUTPUT_FORMAT_PDF);
}


/* Creates a block matrix and outputs its visualization as .pdf file
 * @returns SCIP return code*/
SCIP_RETCODE GCGWriteGpDecompMatrix(
   SCIP*                 scip,               /* scip data structure */
   const char*           filename,           /* filename the output should be written to (including directory) */
   const char*           workfolder,         /* directory in which should be worked */
   SCIP_Bool             originalmatrix      /* should the original (or transformed) matrix be written */
   )
{
   char outputname[SCIP_MAXSTRLEN];
   char filename2[SCIP_MAXSTRLEN];

   int id = GCGconshdlrDecompAddMatrixPartialdec(scip, !originalmatrix);

   GCGgetVisualizationFilename(scip, GCGconshdlrDecompGetPartialdecFromID(scip, id), "pdf", outputname);

   strcat(outputname, ".pdf");
   strcpy(filename2, filename);
   SCIPinfoMessage(scip, NULL, "filename for matrix plot is %s \n", filename );
   SCIPinfoMessage(scip, NULL, "foldername for matrix plot is %s \n", workfolder );

   /* actual writing */
   GCGwriteGpVisualization(scip, filename2, outputname, id );

   return SCIP_OKAY;
}


/*
 * reader include
 */

/* includes the gp file reader into SCIP */
SCIP_RETCODE SCIPincludeReaderGp(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( SCIPincludeReader(scip, READER_NAME, READER_DESC, READER_EXTENSION,
      NULL, readerFreeGp, NULL, readerWriteGp, NULL) );

   return SCIP_OKAY;
}

