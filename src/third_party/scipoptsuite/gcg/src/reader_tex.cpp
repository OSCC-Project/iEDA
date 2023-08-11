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

/**@file   reader_tex.cpp
 * @brief  tex file reader for writing partialdecs to LaTeX files
 * @author Hanna Franzen
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <cassert>
#include <cstring>

#if defined(_WIN32) || defined(_WIN64)
#else
#include <strings.h> /*lint --e{766}*/ /* needed for strcasecmp() */
#endif
#include <cstdio>
#include <vector>
#include <sstream>
#include <algorithm>    // std::sort

#include "reader_tex.h"
#include "scip_misc.h"
#include "reader_gp.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include "pub_decomp.h"
#include "miscvisualization.h"
#include "class_partialdecomp.h"
#include "class_detprobdata.h"
#include "params_visu.h"
#include "scoretype.h"

#define READER_NAME             "texreader"
#define READER_DESC             "LaTeX file writer for partialdec visualization"
#define READER_EXTENSION        "tex"

using namespace gcg;

/** Destructor of reader to free user data (called when SCIP is exiting) */
SCIP_DECL_READERFREE(readerFreeTex)
{
   return SCIP_OKAY;
}

/** Problem reading method of reader.
 *  Since the reader is not supposed to read files this returns a reading error. */
SCIP_DECL_READERREAD(readerReadTex)
{  /*lint --e{715}*/
   return SCIP_READERROR;
}

/** Problem writing method of reader */
SCIP_DECL_READERWRITE(readerWriteTex)
{
   PARTIALDECOMP* pd;
   assert(scip != NULL);
   assert(reader != NULL);


   /* get partialdec to write */
   pd = DECgetPartialdecToWrite(scip, transformed);

   if( pd == NULL )
   {
      SCIPerrorMessage("Could not find Partialdecomp to write!\n");
      *result = SCIP_DIDNOTRUN;
   }
   else
   {
      GCGwriteTexVisualization(scip, file, pd->getID(), TRUE, FALSE);
      *result = SCIP_SUCCESS;
   }

   return SCIP_OKAY;
}


/** Outputs the r, g, b decimal values for the rgb hex input
 *
 * @returns SCIP status */
static
SCIP_RETCODE getRgbFromHex(
   const char* hex,  /**< input hex rgb code of form "#000000" */
   int*        red,  /**< output decimal r */
   int*        green,/**< output decimal g */
   int*        blue  /**< output decimal b */
   )
{
   char temp[SCIP_MAXSTRLEN];
   unsigned int r = 0;
   unsigned int g = 0;
   unsigned int b = 0;

   assert( hex[0] == '#' );

   /* remove the '#' at the beginning */
   strcpy( temp, hex );
   memmove( temp, temp+1, strlen( temp ) );

   /* extract int values from the rest */
   sscanf( temp, "%02x%02x%02x", &r, &g, &b );

   *red = (int) r;
   *green = (int) g;
   *blue = (int) b;

   return SCIP_OKAY;
}


/** Converts a hex color code into a tex-conform line of code that defines the color as "colorname"
 *
 * @returns SCIP status */
static
SCIP_RETCODE getTexColorFromHex(
   const char* hex,        /**< hex code for color */
   const char* colorname,  /**< name of color */
   char* code              /**< output resulting code line */
   )
{
   char texcode[SCIP_MAXSTRLEN];
   char colorcode[SCIP_MAXSTRLEN];
   int r;
   int g;
   int b;

   /* convert hex color code to rgb color */
   getRgbFromHex( hex, &r, &g, &b );

   /* make tex code line that defines a rgb color with the computed values */
   strcpy( texcode, "\\definecolor{" );
   strcat( texcode, colorname );
   strcat( texcode, "}{RGB}{" );
   snprintf(colorcode, SCIP_MAXSTRLEN, "%d", r);
   strcat( texcode, colorcode );
   strcat( texcode, "," );
   snprintf(colorcode, SCIP_MAXSTRLEN, "%d", g);
   strcat( texcode, colorcode );
   strcat( texcode, "," );
   snprintf(colorcode, SCIP_MAXSTRLEN, "%d", b);
   strcat( texcode, colorcode );
   strcat( texcode, "}" );

   /* copy the code line into the output variable */
   strcpy( code, texcode );

   return SCIP_OKAY;
}


/** Write LaTeX code header & begin of document to given file
 *
 *  @returns SCIP status */
static
SCIP_RETCODE writeTexHeader(
   SCIP*                scip,               /**< SCIP data structure */
   FILE*                file,               /**< File pointer to write to */
   SCIP_Bool            externalizepics     /**< whether to use the tikz externalize package */
   )
{
   char temp[SCIP_MAXSTRLEN];

   /* write header */
   SCIPinfoMessage(scip, file, "%% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n");
   SCIPinfoMessage(scip, file, "%% *                                                                           * \n");
   SCIPinfoMessage(scip, file, "%% *                  This file is part of the program                         * \n");
   SCIPinfoMessage(scip, file, "%% *          GCG --- Generic Column Generation                                * \n");
   SCIPinfoMessage(scip, file, "%% *                  a Dantzig-Wolfe decomposition based extension            * \n");
   SCIPinfoMessage(scip, file, "%% *                  of the branch-cut-and-price framework                    * \n");
   SCIPinfoMessage(scip, file, "%% *         SCIP --- Solving Constraint Integer Programs                      * \n");
   SCIPinfoMessage(scip, file, "%% *                                                                           * \n");
   SCIPinfoMessage(scip, file, "%% * Copyright (C) 2010-2022 Operations Research, RWTH Aachen University       * \n");
   SCIPinfoMessage(scip, file, "%% *                         Zuse Institute Berlin (ZIB)                       * \n");
   SCIPinfoMessage(scip, file, "%% *                                                                           * \n");
   SCIPinfoMessage(scip, file, "%% * This program is free software; you can redistribute it and/or             * \n");
   SCIPinfoMessage(scip, file, "%% * modify it under the terms of the GNU Lesser General Public License        * \n");
   SCIPinfoMessage(scip, file, "%% * as published by the Free Software Foundation; either version 3            * \n");
   SCIPinfoMessage(scip, file, "%% * of the License, or (at your option) any later version.                    * \n");
   SCIPinfoMessage(scip, file, "%% *                                                                           * \n");
   SCIPinfoMessage(scip, file, "%% * This program is distributed in the hope that it will be useful,           * \n");
   SCIPinfoMessage(scip, file, "%% * but WITHOUT ANY WARRANTY; without even the implied warranty of            * \n");
   SCIPinfoMessage(scip, file, "%% * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             * \n");
   SCIPinfoMessage(scip, file, "%% * GNU Lesser General Public License for more details.                       * \n");
   SCIPinfoMessage(scip, file, "%% *                                                                           * \n");
   SCIPinfoMessage(scip, file, "%% * You should have received a copy of the GNU Lesser General Public License  * \n");
   SCIPinfoMessage(scip, file, "%% * along with this program; if not, write to the Free Software               * \n");
   SCIPinfoMessage(scip, file, "%% * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.* \n");
   SCIPinfoMessage(scip, file, "%% *                                                                           * \n");
   SCIPinfoMessage(scip, file, "%% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n");
   SCIPinfoMessage(scip, file, "%%\n");
   SCIPinfoMessage(scip, file, "%% @author Hanna Franzen\n");
   SCIPinfoMessage(scip, file, "\n");
   SCIPinfoMessage(scip, file, "\n");
   SCIPinfoMessage(scip, file, "\\documentclass[a4paper,10pt]{article}\n");
   SCIPinfoMessage(scip, file, "\n");
   SCIPinfoMessage(scip, file, "%% packages\n");
   SCIPinfoMessage(scip, file, "\\usepackage[utf8]{inputenc}\n");
   SCIPinfoMessage(scip, file, "\\usepackage[hidelinks]{hyperref}\n");
   SCIPinfoMessage(scip, file, "\\usepackage{pdfpages}\n");
   SCIPinfoMessage(scip, file, "\\usepackage{fancybox}\n");
   if(!GCGgetUseGp(scip))
   {
      SCIPinfoMessage(scip, file, "\\usepackage{pgfplots}\n");
      SCIPinfoMessage(scip, file, "\\pgfplotsset{compat=1.12}\n");
//      SCIPinfoMessage(scip, file, "\\pgfplotsset{compat=newest}                                                    \n");
//      SCIPinfoMessage(scip, file, "\\pgfplotsset{legend image with text/.style={\nlegend image code/.code={%       \n");
//      SCIPinfoMessage(scip, file, "\\node[anchor=center] at (0.3cm,0cm) {#1};\n}},}\n"                                );
//      SCIPinfoMessage(scip, file, "\\usepackage{tikz}                                                               \n");
      SCIPinfoMessage(scip, file, "\\usetikzlibrary{positioning}\n");
      if(externalizepics)
      {
         SCIPinfoMessage(scip, file, " \\usetikzlibrary{external}\n");
         SCIPinfoMessage(scip, file, " \\tikzexternalize\n");
      }
   }
   SCIPinfoMessage(scip, file, "\n");

  /* introduce colors of current color scheme */
   getTexColorFromHex(SCIPvisuGetColorMasterconss(scip), "colormasterconss", temp);
   SCIPinfoMessage(scip, file, "%s\n", temp);

   getTexColorFromHex(SCIPvisuGetColorMastervars(scip), "colormastervars", temp);
   SCIPinfoMessage(scip, file, "%s\n", temp);

   getTexColorFromHex(SCIPvisuGetColorLinking(scip), "colorlinking", temp);
   SCIPinfoMessage(scip, file, "%s\n", temp);

   getTexColorFromHex(SCIPvisuGetColorStairlinking(scip), "colorstairlinking", temp);
   SCIPinfoMessage(scip, file, "%s\n", temp);

   getTexColorFromHex(SCIPvisuGetColorBlock(scip), "colorblock", temp);
   SCIPinfoMessage(scip, file, "%s\n", temp);

   getTexColorFromHex(SCIPvisuGetColorOpen(scip), "coloropen", temp);
   SCIPinfoMessage(scip, file, "%s\n", temp);

   getTexColorFromHex(SCIPvisuGetColorNonzero(scip), "colornonzero", temp);
   SCIPinfoMessage(scip, file, "%s\n", temp);

   getTexColorFromHex(SCIPvisuGetColorLine(scip), "colorline", temp);
   SCIPinfoMessage(scip, file, "%s\n", temp);

   /* start writing the document */
   SCIPinfoMessage(scip, file, "\n");
   SCIPinfoMessage(scip, file, "\\begin{document}\n");
   SCIPinfoMessage(scip, file, "\n");

   return SCIP_OKAY;
}


/** Write LaTeX code title page that includes general statistics about the problem to given file
 *
 * @returns SCIP status */
static
SCIP_RETCODE writeTexTitlepage(
   SCIP*                scip,               /**< SCIP data structure */
   FILE*                file,               /**< File pointer to write to */
   int*                 npresentedpartialdecs    /**< Number of decompositions to be shown in the file or NULL if unknown */
   )
{
   char* pname;
   char ppath[SCIP_MAXSTRLEN];
   int ndecomps;

   ndecomps = GCGconshdlrDecompGetNDecomps(scip);
   strcpy(ppath, SCIPgetProbName(scip));
   SCIPsplitFilename(ppath, NULL, &pname, NULL, NULL);

   SCIPinfoMessage(scip, file, "\\begin{titlepage}\n");
   SCIPinfoMessage(scip, file, "  \\centering\n");
   SCIPinfoMessage(scip, file, "  \\thispagestyle{empty}\n");
   SCIPinfoMessage(scip, file, "  {\\Huge Report: %s} \\\\ \\today \n",
      pname);
   SCIPinfoMessage(scip, file, "\n");
   SCIPinfoMessage(scip, file, "\\vspace{2cm}\n");
   SCIPinfoMessage(scip, file, "\\begin{tabular}{{lp{10cm}}}\n");
   SCIPinfoMessage(scip, file, "  \\textbf{Problem}: & \\begin{minipage}{10cm}\n");
   SCIPinfoMessage(scip, file, "                         \\begin{verbatim}%s\\end{verbatim}\n",
      pname);
   SCIPinfoMessage(scip, file, "                       \\end{minipage} \\\\ \n");
   SCIPinfoMessage(scip, file, "  Number of variables in original problem: & %i  \\\\ \n",
      SCIPgetNOrigVars(scip));
   SCIPinfoMessage(scip, file, "  \\vspace{0.5cm}\n");
   SCIPinfoMessage(scip, file, "  Number of constraints in original problem: & %i  \\\\ \n",
      SCIPgetNOrigConss(scip));
   SCIPinfoMessage(scip, file, "  Number of found finished decompositions: & %i  \\\\ \n",
      GCGconshdlrDecompGetNDecomps(scip));
   SCIPinfoMessage(scip, file, "  Number of found incomplete decompositions: & %i  \\\\ \n",
      GCGconshdlrDecompGetNPartialdecs(scip) - GCGconshdlrDecompGetNDecomps(scip));
   if(npresentedpartialdecs != NULL){
      if( ndecomps > *npresentedpartialdecs )
      {
         SCIPinfoMessage(scip, file, "  Number of decompositions presented in this document: & %i \\\\ \n",
            *npresentedpartialdecs);
      }
      else
      {
         SCIPinfoMessage(scip, file, "  Number of decompositions presented in this document: & %i \\\\ \n", ndecomps);
      }
   }
   
   SCIPinfoMessage(scip, file, "Score info: & \\begin{minipage}{5cm}\n");
   SCIPinfoMessage(scip, file, "                  %s\n",
                   GCGscoretypeGetDescription(GCGconshdlrDecompGetScoretype(scip)));
   SCIPinfoMessage(scip, file, "              \\end{minipage} \\\\ \n");
   
   SCIPinfoMessage(scip, file, "\\end{tabular}\n");
   SCIPinfoMessage(scip, file, "\n");
   SCIPinfoMessage(scip, file, "\\end{titlepage}\n");
   SCIPinfoMessage(scip, file, "\\newpage\n");

   return SCIP_OKAY;
}


/** Write LaTeX code for table of contents to given file
 *
 *  @returns SCIP status */
static
SCIP_RETCODE writeTexTableOfContents(
   SCIP*                scip,               /**< SCIP data structure */
   FILE*                file                /**< File pointer to write to */
   )
{
   SCIPinfoMessage(scip, file, "\\thispagestyle{empty}\n");
   SCIPinfoMessage(scip, file, "\\tableofcontents\n");
   SCIPinfoMessage(scip, file, "\\newpage\n");
   SCIPinfoMessage(scip, file, "\n");

   return SCIP_OKAY;
}


/** Writes tikz code for a box
 *
 *  @returns SCIP status */
static
SCIP_RETCODE writeTikzBox(
   SCIP* scip,       /**< SCIP data structure */
   FILE* file,       /**< File pointer to write to */
   int xmax,         /**< maximum x axis value */
   int ymax,         /**< maximum y axis value */
   int x1,           /**< x value of lower left vertex coordinate */
   int y1,           /**< y value of lower left vertex coordinate */
   int x2,           /**< x value of upper right vertex coordinate */
   int y2,           /**< y value of upper right vertex coordinate */
   const char* color /**< color name of box color */
   )
{
   SCIPinfoMessage(scip, file,
      "    \\filldraw [fill=%s, draw=colorline] (%f*\\textwidth*0.75,%f*\\textwidth*0.75) rectangle (%f*\\textwidth*0.75,%f*\\textwidth*0.75);\n",
      color, ( (float) x1 / (float) xmax ), ( (float) y1 / (float) ymax ), ( (float) x2 / (float) xmax ),
      ( (float) y2 / (float) ymax ));
   return SCIP_OKAY;
}


/** Writes tikz code for dots representing nonzeros
 *
 *  @returns SCIP status */
static
SCIP_RETCODE writeTikzNonzeros(
   SCIP* scip,             /**< SCIP data structure */
   FILE* file,             /**< file to write to  */
   PARTIALDECOMP* partialdec,           /**< PARTIALDECOMP for which the nonzeros should be visualized */
   float radius,           /**< radius of the dots */
   int xmax,               /**< maximum x axis value */
   int ymax                /**< maximum y axis value */
   )
{
   std::vector<int> orderToRows(partialdec->getNConss(), -1);
   std::vector<int> rowToOrder(partialdec->getNConss(), -1);
   std::vector<int> orderToCols(partialdec->getNVars(), -1);
   std::vector<int> colsToOrder(partialdec->getNVars(), -1);
   int counterrows = 0;
   int countercols = 0;
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

   /* write scatter plot */
   for( int row = 0; row < partialdec->getNConss(); ++row )
   {
      for ( int col = 0; col < partialdec->getNVars(); ++col )
      {
         assert( orderToRows[row] != -1 );
         assert( orderToCols[col] != -1 );
         if( detprobdata->getVal( orderToRows[row], orderToCols[col] ) != 0 )
         {
            SCIPinfoMessage(scip, file,
               "    \\draw [fill] (%f*\\textwidth*0.75,%f*\\textwidth*0.75) circle [radius=%f*0.75];\n",
               ( (float) col + 0.5 ) / (float) xmax, ( (float) row + 0.5 ) / (float) ymax, radius);
         }
      }
   }

   return SCIP_OKAY;
}


/** Writes LaTeX code that contains a figure with a tikz picture of the given partialdec
 *
 * @returns SCIP status */
static
SCIP_RETCODE writeTexPartialdec(
   SCIP* scip,                /**< SCIP data structure */
   FILE* file,                /**< file to write to */
   PARTIALDECOMP* partialdec, /**< PARTIALDECOMP to be visualized */
   SCIP_Bool nofigure         /**< if true there will be no figure environment around tikz code*/
   )
{
   int rowboxcounter = 0;
   int colboxcounter = 0;
   int nvars;
   int nconss;

   nvars = partialdec->getNVars();
   nconss = partialdec->getNConss();

   if(!nofigure)
   {
      SCIPinfoMessage(scip, file, "\\begin{figure}[!htb]\n");
      SCIPinfoMessage(scip, file, "  \\begin{center}\n");
   }
   SCIPinfoMessage(scip, file, "  \\begin{tikzpicture}[yscale=-1]\n");
   /* --- draw boxes ---*/

   /* linking vars */
   if(partialdec->getNLinkingvars() != 0)
   {
      writeTikzBox(scip, file, nvars, nconss, 0, 0, partialdec->getNLinkingvars(), partialdec->getNConss(),
         (const char*) "colorlinking");
      colboxcounter += partialdec->getNLinkingvars();
   }

   /* masterconss */
   if(partialdec->getNMasterconss() != 0)
   {
      writeTikzBox(scip, file, nvars, nconss, 0, 0, partialdec->getNVars(), partialdec->getNMasterconss(),
         (const char*) "colormasterconss");
      rowboxcounter += partialdec->getNMasterconss();
   }

   /* mastervars */
   if(partialdec->getNMastervars() != 0)
   {
      writeTikzBox(scip, file, nvars, nconss, colboxcounter, 0, partialdec->getNMastervars()+colboxcounter,
         partialdec->getNMasterconss(), (const char*) "colormastervars");
      colboxcounter += partialdec->getNMastervars();
   }

   /* blocks (blocks are not empty) */
   for( int b = 0; b < partialdec->getNBlocks() ; ++b )
   {
      writeTikzBox(scip, file, nvars, nconss, colboxcounter, rowboxcounter,
         colboxcounter + partialdec->getNVarsForBlock(b), rowboxcounter + partialdec->getNConssForBlock(b),
         (const char*) "colorblock");
      colboxcounter += partialdec->getNVarsForBlock(b);

      if( partialdec->getNStairlinkingvars(b) != 0 )
      {
         writeTikzBox(scip, file, nvars, nconss, colboxcounter, rowboxcounter,
            colboxcounter + partialdec->getNStairlinkingvars(b),
            rowboxcounter + partialdec->getNConssForBlock(b) + partialdec->getNConssForBlock(b+1),
            (const char*) "colorstairlinking");
      }
      colboxcounter += partialdec->getNStairlinkingvars(b);
      rowboxcounter += partialdec->getNConssForBlock(b);
   }

   /* open */
   if(partialdec->getNOpenvars() != 0)
   {
      writeTikzBox(scip, file, nvars, nconss, colboxcounter, rowboxcounter, colboxcounter + partialdec->getNOpenvars(),
         rowboxcounter+partialdec->getNOpenconss(), (const char*) "coloropen" );
      colboxcounter += partialdec->getNOpenvars();
      rowboxcounter += partialdec->getNOpenconss();
   }

   /* --- draw nonzeros --- */
   if(SCIPvisuGetDraftmode(scip) == FALSE)
   {
      writeTikzNonzeros(scip, file, partialdec, SCIPvisuGetNonzeroRadius(scip, partialdec->getNVars(), partialdec->getNConss(), 1),
         nvars, nconss);
   }

   SCIPinfoMessage(scip, file, "  \\end{tikzpicture}\n");
   if(!nofigure)
   {
      SCIPinfoMessage(scip, file, "  \\end{center}\n");
      SCIPinfoMessage(scip, file, "\\end {figure}\n");
   }

   return SCIP_OKAY;
}


/** Writes LaTeX code for some statistics about the partialdec:
 * - amount of blocks
 * - amount of master, linking, stairlinking variables
 * - amount of master constraints
 * - score
 *
 * @returns SCIP status */
static
SCIP_RETCODE writeTexPartialdecStatistics(
   SCIP* scip,             /**< SCIP data structure */
   FILE* file,             /**< file to write to */
   PARTIALDECOMP* partialdec            /**< statistics are about this partialdec */
   )
{
   std::ostringstream fulldetectorstring;

   /* get detector chain full-text string*/
   if( partialdec->getUsergiven() != gcg::USERGIVEN::NOT )
   {
      fulldetectorstring << "user";
   }
   for( auto detector : partialdec->getDetectorchain() )
   {
      if( fulldetectorstring.tellp() > 0 )
         fulldetectorstring << ", ";
      fulldetectorstring << DECdetectorGetName(detector);
   }

   SCIPinfoMessage(scip, file, "\n");
   SCIPinfoMessage(scip, file, "\\vspace{0.3cm}\n");
   SCIPinfoMessage(scip, file, "\\begin{tabular}{lp{10cm}}\n");
   SCIPinfoMessage(scip, file,
      "  Found by detector(s): & \\begin{minipage}{10cm}\\begin{verbatim}%s\\end{verbatim}\\end{minipage} \\\\ \n",
      fulldetectorstring.str().c_str());
   SCIPinfoMessage(scip, file, "  Number of blocks: & %i \\\\ \n",
      partialdec->getNBlocks());
   SCIPinfoMessage(scip, file, "  Number of master variables: & %i \\\\ \n",
      partialdec->getNMastervars());
   SCIPinfoMessage(scip, file, "  Number of master constraints: & %i \\\\ \n",
      partialdec->getNMasterconss());
   SCIPinfoMessage(scip, file, "  Number of linking variables: & %i \\\\ \n",
      partialdec->getNLinkingvars());
   SCIPinfoMessage(scip, file, "  Number of stairlinking variables: & %i \\\\ \n",
      partialdec->getNTotalStairlinkingvars());
   SCIPinfoMessage(scip, file, "  Score: & %f \\\\ \n",
      partialdec->getScore(GCGconshdlrDecompGetScoretype(scip)));
   SCIPinfoMessage(scip, file, "\\end{tabular}\n");

   SCIPinfoMessage(scip, file, "\\clearpage\n");
   SCIPinfoMessage(scip, file, "\n");

   return SCIP_OKAY;
}


/** Write LaTeX code for end of document to given file
 *
 * @returns SCIP status */
static
SCIP_RETCODE writeTexEnding(
   SCIP* scip, /**< SCIP data structure */
   FILE* file  /**< File pointer to write to */
   )
{
   SCIPinfoMessage(scip, file, "\\end{document}                                                                  \n");

   return SCIP_OKAY;
}


/* Writes a report for the given partialdecs */
SCIP_RETCODE GCGwriteTexReport(
   SCIP* scip,             /* SCIP data structure */
   FILE* file,             /* file to write to */
   int* partialdecids,     /* ids of partialdecs to visualize */
   int* npartialdecs,      /* number of partialdecs to visualize */
   SCIP_Bool titlepage,    /* true if a title page should be included in the document */
   SCIP_Bool toc,          /* true if an interactive table of contents should be included */
   SCIP_Bool statistics,   /* true if statistics for each partialdec should be included */
   SCIP_Bool usegp         /* true if the gp reader should be used to visualize the individual partialdecs */
   )
{
   PARTIALDECOMP* partialdec;
   char gppath[PATH_MAX];
   char filepath[PATH_MAX];
   char* path;
   char gpname[SCIP_MAXSTRLEN];
   char pdfname[SCIP_MAXSTRLEN];

   /* write tex code into file */
   writeTexHeader(scip, file, TRUE);
   if(titlepage)
      writeTexTitlepage(scip, file, npartialdecs);
   if(toc)
      writeTexTableOfContents(scip, file);

   /* get partialdec pointers and sort them according to current score */
   std::vector<PARTIALDECOMP*> partialdecs;
   partialdecs.reserve(*npartialdecs);
   for(int i = 0; i < *npartialdecs; i++)
   {
      partialdecs.push_back(GCGconshdlrDecompGetPartialdecFromID(scip, partialdecids[i]));
   }
   SCORETYPE sctype = GCGconshdlrDecompGetScoretype(scip);
   std::sort(partialdecs.begin(), partialdecs.end(), [&](PARTIALDECOMP* a, PARTIALDECOMP* b) {return (a->getScore(sctype) > b->getScore(sctype)); });

   /* if there are more decomps than the maximum, reset npartialdecs */
   if(*npartialdecs > GCGreportGetMaxNDecomps(scip))
      *npartialdecs = GCGreportGetMaxNDecomps(scip);

   for(int i = 0; i < *npartialdecs; i++)
   {
      /* write each partialdec */
      partialdec = partialdecs.at(i);

      if(toc)
      {
         char decompname[SCIP_MAXSTRLEN];
         char buffer[SCIP_MAXSTRLEN];
         partialdec->buildDecChainString(buffer);
         SCIPsnprintf( decompname, SCIP_MAXSTRLEN, "%s-%d-%d", buffer, partialdec->getID(), partialdec->getNBlocks() );

         SCIPinfoMessage(scip, file, "\\section*{Decomposition: %s}\n", decompname);
         SCIPinfoMessage(scip, file, "\\addcontentsline{toc}{section}{Decomposition: %s}\n", decompname);
         SCIPinfoMessage(scip, file, "\n");
      }

      if(!usegp)
      {
         writeTexPartialdec(scip, file, partialdec, FALSE);
      }
      else
      {
         /* in case a gp file should be generated include it in the tex code */
         GCGgetVisualizationFilename(scip, partialdec, "gp", gpname);
         GCGgetVisualizationFilename(scip, partialdec, "pdf", pdfname);
         strcat(pdfname, ".pdf");

         GCGgetFilePath(file, filepath);
         SCIPsplitFilename(filepath, &path, NULL, NULL, NULL);

         SCIPsnprintf(gppath, PATH_MAX, "%s/%s.gp", path, gpname);

         GCGwriteGpVisualization(scip, gppath, pdfname, partialdecids[i]);

         SCIPinfoMessage(scip, file, "\\begin{figure}[!htb]\n");
         SCIPinfoMessage(scip, file, "  \\begin{center}\n");
         SCIPinfoMessage(scip, file, "    \\includegraphics{%s}\n", pdfname);
         SCIPinfoMessage(scip, file, "  \\end{center}\n");
         SCIPinfoMessage(scip, file, "\\end {figure}\n");
      }
      if(statistics)
         writeTexPartialdecStatistics(scip, file, partialdec);
   }
   writeTexEnding(scip, file);

   GCGtexWriteMakefileAndReadme(scip, file, usegp, FALSE);

   return SCIP_OKAY;
}


/* Writes a visualization for the given partialdec */
SCIP_RETCODE GCGwriteTexVisualization(
   SCIP* scip,             /* SCIP data structure */
   FILE* file,             /* file to write to */
   int partialdecid,            /* id of partialdec to visualize */
   SCIP_Bool statistics,   /* additionally to picture show statistics */
   SCIP_Bool usegp         /* true if the gp reader should be used to visualize the individual partialdecs */
   )
{
   PARTIALDECOMP* partialdec;
   char gpname[SCIP_MAXSTRLEN];
   char pdfname[SCIP_MAXSTRLEN];

   /* get partialdec */
   partialdec = GCGconshdlrDecompGetPartialdecFromID(scip, partialdecid);

   /* write tex code into file */
   writeTexHeader(scip, file, FALSE);

   if(!usegp)
   {
      writeTexPartialdec(scip, file, partialdec, FALSE);
   }
   else
   {
      /* in case a gp file should be generated include it */
       GCGgetVisualizationFilename(scip, partialdec, "gp", gpname);
       GCGgetVisualizationFilename(scip, partialdec, "pdf", pdfname);

      GCGwriteGpVisualization(scip, gpname, pdfname, partialdecid);

      SCIPinfoMessage(scip, file, "\\begin{figure}[!htb]\n");
      SCIPinfoMessage(scip, file, "  \\begin{center}\n");
      SCIPinfoMessage(scip, file, "    \\input{%s}\n", pdfname);
      SCIPinfoMessage(scip, file, "  \\end{center}\n");
      SCIPinfoMessage(scip, file, "\\end {figure}\n");
   }
   if(statistics)
      writeTexPartialdecStatistics(scip, file, partialdec);

   writeTexEnding(scip, file);

   return SCIP_OKAY;
}


/* Makes a new makefile and readme for the given .tex file */
SCIP_RETCODE GCGtexWriteMakefileAndReadme(
   SCIP*                scip,               /* SCIP data structure */
   FILE*                file,               /* File for which the makefile & readme are generated */
   SCIP_Bool            usegp,              /* true if there are gp files to be included in the makefile */
   SCIP_Bool            compiletex          /* true if there are tex files to be compiled before main document */

   )
{
   FILE* makefile;
   FILE* readme;
   char* filepath;
   char* filename;
   char pfile[PATH_MAX];
   char makefilename[PATH_MAX];
   char readmename[PATH_MAX];
   char name[PATH_MAX];
   const char makename[SCIP_MAXSTRLEN] = "makepdf";

   /* --- create a Makefile --- */

   /* get path to write to and put it into makefilename */
   GCGgetFilePath(file, pfile);
   SCIPsplitFilename(pfile, &filepath, &filename, NULL, NULL);
   SCIPsnprintf(name, PATH_MAX, "%s_%s.make", makename, filename);
   SCIPsnprintf(makefilename, PATH_MAX, "%s/%s", filepath, name);

   /* open and write makefile */
   makefile = fopen(makefilename, "w");
   if( makefile == NULL )
   {
      return SCIP_FILECREATEERROR;
   }

   if( usegp )
   {
      SCIPinfoMessage(scip, makefile, "GPFILES := $(wildcard *.gp)\n");
   }
   if( compiletex )
   {
      /* will only be applied if the filename ends with "-tex.tex" due to the standard naming scheme */
      SCIPinfoMessage(scip, makefile, "TEXFILES := $(wildcard *-pdf.tex)\n");
   }
   SCIPinfoMessage(scip, makefile, "\n");
   SCIPinfoMessage(scip, makefile, "# latexmk automatically manages the .tex files\n");
   SCIPinfoMessage(scip, makefile, "%s.pdf: %s.tex\n",
      filename, filename);
   if( usegp )
   {
      SCIPinfoMessage(scip, makefile, "\t@echo ------------\n");
      SCIPinfoMessage(scip, makefile, "\t@echo \n");
      SCIPinfoMessage(scip, makefile, "\t@echo Compiling gp files to tex\n");
      SCIPinfoMessage(scip, makefile, "\t@echo \n");
      SCIPinfoMessage(scip, makefile, "\t@echo ------------\n");
      SCIPinfoMessage(scip, makefile, "\t$(SHELL) -ec  'for i in $(GPFILES); \\\n");
      SCIPinfoMessage(scip, makefile, "\t\tdo \\\n");
      SCIPinfoMessage(scip, makefile, "\t\tgnuplot $$i; \\\n");
      SCIPinfoMessage(scip, makefile, "\t\tdone'\n");
   }
   SCIPinfoMessage(scip, makefile, "\t@echo ------------\n");
   SCIPinfoMessage(scip, makefile, "\t@echo \n");
   SCIPinfoMessage(scip, makefile, "\t@echo Compiling tex code. This may take a while.\n");
   SCIPinfoMessage(scip, makefile, "\t@echo \n");
   SCIPinfoMessage(scip, makefile, "\t@echo ------------\n");
   if( compiletex )
   {
      SCIPinfoMessage(scip, makefile, "\t$(SHELL) -ec  'for j in $(TEXFILES); \\\n");
      SCIPinfoMessage(scip, makefile, "\t\tdo \\\n");
      SCIPinfoMessage(scip, makefile, "\t\tpdflatex $$j; \\\n");
      SCIPinfoMessage(scip, makefile, "\t\tdone'\n");
   }
   SCIPinfoMessage(scip, makefile,
      "\t@latexmk -pdf -pdflatex=\"pdflatex -interaction=batchmode -shell-escape\" -use-make %s.tex \n", filename);
   SCIPinfoMessage(scip, makefile, "\t@make -f %s clean\n", name);
   SCIPinfoMessage(scip, makefile, "\n");
   SCIPinfoMessage(scip, makefile, "clean:\n");
   SCIPinfoMessage(scip, makefile, "\t@latexmk -c\n");
   SCIPinfoMessage(scip, makefile, "\t@rm -f report_*figure*.*\n");
   SCIPinfoMessage(scip, makefile, "\t@rm -f *.auxlock\n");
   SCIPinfoMessage(scip, makefile, "\t@rm -f *figure*.md5\n");
   SCIPinfoMessage(scip, makefile, "\t@rm -f *figure*.log\n");
   SCIPinfoMessage(scip, makefile, "\t@rm -f *figure*.dpth\n");
   if( usegp )
   {
      SCIPinfoMessage(scip, makefile, "\t@rm -f *.gp\n");
   }
   SCIPinfoMessage(scip, makefile, "\n");
   SCIPinfoMessage(scip, makefile, "cleanall:\n");
   SCIPinfoMessage(scip, makefile, "\t@latexmk -C\n");
   SCIPinfoMessage(scip, makefile, "\t@make -f %s clean\n", name);

   /* close makefile */
   fclose(makefile);

   /* --- create a readme file --- */

   /* use same file path as the makefile */
   SCIPsnprintf(readmename, PATH_MAX, "%s/README_%s", filepath, makename);

   /* open and write readme */
   readme = fopen(readmename, "w");
   if( readme == NULL )
   {
      return SCIP_FILECREATEERROR;
   }

   SCIPinfoMessage(scip, readme, "README: How to create a PDF file from the .tex file(s) using the %s file.\n", name);
   SCIPinfoMessage(scip, readme, "Note: The package pdflatex is required.\n");
   SCIPinfoMessage(scip, readme, "\n");
   SCIPinfoMessage(scip, readme, "Use the command\n\t'make -f %s'\nto compile.\n", name);
   SCIPinfoMessage(scip, readme, "Depending on the size of your problem that may take some time.\n");
   SCIPinfoMessage(scip, readme,
      "Please do not delete any new files that might be generated during the compile process.\n");
   SCIPinfoMessage(scip, readme, "All access files will be deleted automatically once the compilation is complete.\n");
   SCIPinfoMessage(scip, readme, "\n");
   SCIPinfoMessage(scip, readme, "Clean options:\n");
   SCIPinfoMessage(scip, readme, "\t'make -f %s clean' clears all present intermediate files (if any exist)\n", name);
   SCIPinfoMessage(scip, readme, "\t'make -f %s cleanall' clears all generated files INCLUDING .pdf\n", name);

   /* close readme file */
   fclose(readme);

   return SCIP_OKAY;
}


/* Includes the tex file reader in SCIP */
SCIP_RETCODE SCIPincludeReaderTex(
   SCIP*                 scip                /*< SCIP data structure */
   )
{
   SCIP_CALL(SCIPincludeReader(scip, READER_NAME, READER_DESC, READER_EXTENSION, NULL,
           readerFreeTex, readerReadTex, readerWriteTex, NULL));

   return SCIP_OKAY;
}

