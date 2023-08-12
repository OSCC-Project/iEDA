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

/**@file    params_visu.c
 * @brief   parameter settings for visualization readers
 * @author  Hanna Franzen
 * @author  Michael Bastubbe
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "params_visu.h"
#include "type_decomp.h"
#include "cons_decomp.h"

#include "relax_gcg.h"

#include "scip/scip.h"

#include <limits.h>
#include <string.h>

#define PARAM_NAME          "paramsvisu"
#define PARAM_DESC          "parameters for visualization"

/* color defaults to build default color layout with */
#define COLOR_WHITE     "#FFFFFF"   /**< standard white */
#define COLOR_BLUE1     "#ACBCE9"   /**< very light blue */
#define COLOR_BLUE2     "#718CDB"   /**< light blue */
#define COLOR_BLUE3     "#3C64DD"   /**< middle blue */
#define COLOR_BLUE4     "#1340C7"   /**< dark blue */
#define COLOR_BLUE5     "#1F377D"   /**< very dark blue */
#define COLOR_ORANGE1   "#FFD88F"   /**< very light orange */
#define COLOR_ORANGE2   "#FFCB69"   /**< light orange */
#define COLOR_ORANGE3   "#FFB72D"   /**< orange */
#define COLOR_BROWN1    "#B38208"   /**< light brown */
#define COLOR_BROWN2    "#886100"   /**< brown */
#define COLOR_BROWN3    "#443000"   /**< dark brown */
#define COLOR_BLACK     "#000000"   /**< standard black */

/* default colors (use defines above for changes) */
#define DEFAULT_COLOR_MASTERVARS   COLOR_BLUE4     /**< for mastervars (in block area) */
#define DEFAULT_COLOR_MASTERCONSS  COLOR_BLUE4     /**< for masterconss */
#define DEFAULT_COLOR_LINKING      COLOR_ORANGE3   /**< for linking areas */
#define DEFAULT_COLOR_STAIRLINKING COLOR_BROWN2    /**< for stairlinking areas */
#define DEFAULT_COLOR_BLOCK        COLOR_BLUE2     /**< for finished blocks */
#define DEFAULT_COLOR_OPEN         COLOR_ORANGE1   /**< for open (not assigned) elements */
#define DEFAULT_COLOR_NONZERO      COLOR_BLACK     /**< for nonzero dots */
#define DEFAULT_COLOR_LINE         COLOR_BLACK     /**< for outlines of blocks */

/* 8 shades of grey */
#define GREY_COLOR_MASTERVARS   "#323232"    /**< for mastervars (in block area) */
#define GREY_COLOR_MASTERCONS   "#666666"    /**< for masterconss */
#define GREY_COLOR_LINKING      "#4C4C4C"    /**< for linking areas */
#define GREY_COLOR_STAIRLINKING "#191919"    /**< for stairlinking areas */
#define GREY_COLOR_BLOCK        "#d3d3d3"    /**< for finished blocks */
#define GREY_COLOR_OPEN         "#7F7F7F"    /**< for open (not assigned) elements */
#define GREY_COLOR_NONZERO      COLOR_BLACK  /**< for nonzero dots */
#define GREY_COLOR_LINE         COLOR_BLACK  /**< for outlines of blocks */

/* visualization imaging defaults */
#define DEFAULT_VISU_DRAFTMODE   FALSE                /**< if true no nonzeros are shown in visualizations */
#define DEFAULT_VISU_COLORSCHEME COLORSCHEME_DEFAULT  /**< is of type VISU_COLORSCHEME */
#define DEFAULT_VISU_RADIUS      2                    /**< possible scale: 1-10 */
#define DEFAULT_VISU_USEGP       FALSE                /**< if true gnuplot is used for visualizations,
                                                       * otherwise LaTeX/Tikz */

/* pdf reader default */
/**< name of pdf reader, must be callable by system */
#if defined(__linux__)
#define DEFAULT_PDFREADER        "xdg-open"
#elif defined(__APPLE__)
#define DEFAULT_PDFREADER        "open"
#elif defined(_WIN32)
#define DEFAULT_PDFREADER        "start"
#else
#define DEFAULT_PDFREADER        "evince"
#endif

/* report parameter defaults */
#define DEFAULT_REPORT_MAXNDECOMPS     20       /**< maximum number of decomps to be shown in report */
#define DEFAULT_REPORT_SHOWTYPE        0        /**< what type of decomps to show
                                                 * (DEC_DECTYPE, but 0 corresponds to 'show all') */
#define DEFAULT_REPORT_SHOWTITLEPAGE   TRUE     /**< if true a titlepage is included */
#define DEFAULT_REPORT_SHOWTOC         TRUE     /**< if true a table of contents is included */
#define DEFAULT_REPORT_SHOWSTATISTICS  TRUE     /**< if true statistics are included for each decomp */

/** data structure for visualization parameters */
struct GCG_ParamData
{
   SCIP_Bool         visudraftmode;    /**< true if no nonzeros should be shown */
   VISU_COLORSCHEME  visucolorscheme;  /**< stores the current color scheme */
   int               visuradius;       /**< radius for nonzeros */
   SCIP_Bool         visuusegp;        /**< if true gnuplot is used for visualizations, otherwise LaTeX/Tikz */

   char* mancolormastervars;           /**< manual color for master variables */
   char* mancolormasterconss;          /**< manual color for master constraints */
   char* mancolorlinking;              /**< manual color for linking */
   char* mancolorstairlinking;         /**< manual color for stairlinking */
   char* mancolorblock;                /**< manual color for blocks */
   char* mancoloropen;                 /**< manual color for nonassigned areas */
   char* mancolornonzero;              /**< manual color for nonzeros */
   char* mancolorline;                 /**< manual color for lines */

   char* greycolormastervars;          /**< black and white color for master variables */
   char* greycolormasterconss;         /**< black and white color for master constraints */
   char* greycolorlinking;             /**< black and white color for linking */
   char* greycolorstairlinking;        /**< black and white color for stairlinking */
   char* greycolorblock;               /**< black and white color for blocks */
   char* greycoloropen;                /**< black and white color for nonassigned areas */
   char* greycolornonzero;             /**< black and white color for nonzeros */
   char* greycolorline;                /**< black and white color for lines */

   char* pdfreader;                    /**< name of pdfreader to open files with */

   int         rep_maxndecomps;        /**< maximum number of decomps to be shown in report */
   SCIP_Bool   rep_showtitle;          /**< if true a titlepage is included */
   SCIP_Bool   rep_showtoc;            /**< if true a table of contents is included */
   SCIP_Bool   rep_statistics;         /**< if true statistics are included for each decomp */

   int         nmaxdecompstowrite;     /**< maximum number of decompositions to write */
};


/* getter & setter */


/* gets if draftmode is on
 * draftmode lets visualizations omit nonzeros
 * @returns true if draftmode is on  */
SCIP_Bool SCIPvisuGetDraftmode(
   SCIP* scip       /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   SCIP_Bool draftmode;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   draftmode = paramdata->visudraftmode;
   return draftmode;
}


/* sets draftmode
 * draftmode lets visualizations omit nonzeros */
void SCIPvisuSetDraftmode(
   SCIP* scip,       /**< SCIP data structure */
   SCIP_Bool setmode /**< true iff draftmode should be on */
   )
{
   GCG_PARAMDATA* paramdata;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   paramdata->visudraftmode = setmode;
}

/** gets the colorscheme for visualizations
 *  @returns current colorscheme */
VISU_COLORSCHEME SCIPvisuGetColorscheme(
   SCIP* scip  /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   VISU_COLORSCHEME curscheme;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   curscheme = paramdata->visucolorscheme;
   return curscheme;
}


/* sets colorscheme for visualizations */
void SCIPvisuSetColorscheme(
   SCIP* scip,                /**< SCIP data structure */
   VISU_COLORSCHEME newscheme /**< new colorscheme */
   )
{
   GCG_PARAMDATA* paramdata;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   paramdata->visucolorscheme = newscheme;
}


/* gets color for mastercon block in current color scheme
 * @returns mastercons color */
const char* SCIPvisuGetColorMasterconss(
   SCIP* scip       /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   const char* color;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   switch(SCIPvisuGetColorscheme(scip))
   {
   case COLORSCHEME_GREY:
      color = paramdata->greycolormasterconss;
      break;
   case COLORSCHEME_MANUAL:
      color = paramdata->mancolormasterconss;
      break;
   default:
      color = DEFAULT_COLOR_MASTERCONSS;
   }
   return color;
}


/* gets color for mastervar block in current color scheme
 * @returns mastervars color */
const char* SCIPvisuGetColorMastervars(
   SCIP* scip       /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   const char* color;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   switch(SCIPvisuGetColorscheme(scip))
   {
   case COLORSCHEME_GREY:
      color =  paramdata->greycolormastervars;
      break;
   case COLORSCHEME_MANUAL:
      color =  paramdata->mancolormastervars;
      break;
   default:
      color =  DEFAULT_COLOR_MASTERVARS;
   }
   return color;
}


/* gets color for linking blocks in current color scheme
 * @returns linking color */
const char* SCIPvisuGetColorLinking(
   SCIP* scip       /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   const char* color;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   switch(SCIPvisuGetColorscheme(scip))
   {
   case COLORSCHEME_GREY:
      color = paramdata->greycolorlinking;
      break;
   case COLORSCHEME_MANUAL:
      color = paramdata->mancolorlinking;
      break;
   default:
      color = DEFAULT_COLOR_LINKING;
   }
   return color;
}


/* gets color for stairlinking blocks in current color scheme
 * @returns stairlinking color */
const char* SCIPvisuGetColorStairlinking(
   SCIP* scip       /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   const char* color;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   switch(SCIPvisuGetColorscheme(scip))
   {
   case COLORSCHEME_GREY:
      color = paramdata->greycolorstairlinking;
      break;
   case COLORSCHEME_MANUAL:
      color = paramdata->mancolorstairlinking;
      break;
   default:
      color = DEFAULT_COLOR_STAIRLINKING;
   }
   return color;
}


/* gets color for normal decomp blocks in current color scheme
 * @returns block color */
const char* SCIPvisuGetColorBlock(
   SCIP* scip       /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   const char* color;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   switch(SCIPvisuGetColorscheme(scip))
   {
   case COLORSCHEME_GREY:
      color = paramdata->greycolorblock;
      break;
   case COLORSCHEME_MANUAL:
      color = paramdata->mancolorblock;
      break;
   default:
      color = DEFAULT_COLOR_BLOCK;
   }
   return color;
}


/* gets color for open blocks in current color scheme
 * @returns open color */
const char* SCIPvisuGetColorOpen(
   SCIP* scip       /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   const char* color;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   switch(SCIPvisuGetColorscheme(scip))
   {
   case COLORSCHEME_GREY:
      color = paramdata->greycoloropen;
      break;
   case COLORSCHEME_MANUAL:
      color = paramdata->mancoloropen;
      break;
   default:
      color = DEFAULT_COLOR_OPEN;
   }
   return color;
}


/* gets color for non-zero points in current color scheme
 * @returns non-zero color */
const char* SCIPvisuGetColorNonzero(
   SCIP* scip       /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   const char* color;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   switch(SCIPvisuGetColorscheme(scip))
   {
   case COLORSCHEME_GREY:
      color = paramdata->greycolornonzero;
      break;
   case COLORSCHEME_MANUAL:
      color = paramdata->mancolornonzero;
      break;
   default:
      color = DEFAULT_COLOR_NONZERO;
   }
   return color;
}


/* gets color for lines in current color scheme
 * @returns line color */
const char* SCIPvisuGetColorLine(
   SCIP* scip       /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   const char* color;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   switch(SCIPvisuGetColorscheme(scip))
   {
   case COLORSCHEME_GREY:
      color = paramdata->greycolorline;
      break;
   case COLORSCHEME_MANUAL:
      color = paramdata->mancolorline;
      break;
   default:
      color = DEFAULT_COLOR_LINE;
   }
   return color;
}


/* sets color for mastercon block in current color scheme */
void SCIPvisuSetColorManMasterconss(
   SCIP* scip,          /* SCIP data structure */
   const char* newcolor /* new color */
   )
{
   GCG_PARAMDATA* paramdata;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   SCIPsnprintf(paramdata->mancolormasterconss, SCIP_MAXSTRLEN, "%s", newcolor);
}


/* sets manual color for mastervar block in current color scheme */
void SCIPvisuSetColorManMastervars(
   SCIP* scip,          /* SCIP data structure */
   const char* newcolor /* new color */
   )
{
   GCG_PARAMDATA* paramdata;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   SCIPsnprintf(paramdata->mancolormastervars, SCIP_MAXSTRLEN, "%s", newcolor);
}


/* sets manual color for linking blocks in current color scheme */
void SCIPvisuSetColorManLinking(
   SCIP* scip,          /* SCIP data structure d refere*/
   const char* newcolor /* new color */
   )
{
   GCG_PARAMDATA* paramdata;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   SCIPsnprintf(paramdata->mancolorlinking, SCIP_MAXSTRLEN, "%s", newcolor);
}


/* sets manual color for stairlinking blocks in current color scheme */
void SCIPvisuSetColorManStairlinking(
   SCIP* scip,          /* SCIP data structure */
   const char* newcolor /* new color */
   )
{
   GCG_PARAMDATA* paramdata;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   SCIPsnprintf(paramdata->mancolorstairlinking, SCIP_MAXSTRLEN, "%s", newcolor);
}


/* sets manual color for normal decomp blocks in current color scheme */
void SCIPvisuSetColorManBlock(
   SCIP* scip,          /* SCIP data structure */
   const char* newcolor /* new color */
   )
{
   GCG_PARAMDATA* paramdata;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   SCIPsnprintf(paramdata->mancolorblock, SCIP_MAXSTRLEN, "%s", newcolor);
}


/* sets manual color for open blocks in current color scheme */
void SCIPvisuSetColorManOpen(
   SCIP* scip,          /* SCIP data structure */
   const char* newcolor /* new color */
   )
{
   GCG_PARAMDATA* paramdata;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   SCIPsnprintf(paramdata->mancoloropen, SCIP_MAXSTRLEN, "%s", newcolor);
}

/* sets manual color for non-zero points in current color scheme */
void SCIPvisuSetColorManNonzero(
   SCIP* scip,          /* SCIP data structure */
   const char* newcolor /* new color */
   )
{
   GCG_PARAMDATA* paramdata;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   SCIPsnprintf(paramdata->mancolornonzero, SCIP_MAXSTRLEN, "%s", newcolor);
}


/* sets manual color for lines in current color scheme */
void SCIPvisuSetColorManLine(
   SCIP* scip,          /* SCIP data structure */
   const char* newcolor /* new color */
   )
{
   GCG_PARAMDATA* paramdata;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);
   SCIPsnprintf(paramdata->mancolorline, SCIP_MAXSTRLEN, "%s", newcolor);
}


/* gets appropriate radius for nonzeros
 * needs highest indices of both axes
 * @returns radius */
float SCIPvisuGetNonzeroRadius(
   SCIP* scip,          /* SCIP data structure */
   int maxindx,         /* highest index x-axis */
   int maxindy,         /* highest index y-axis */
   float scalingfactor  /* percentage to scale radius, 1 if no scaling */
   )
{
   int maxind;
   float radius;
   GCG_PARAMDATA* paramdata;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   maxind = 0;

   /* the max indices must be at least one to be compatible with division */
   if(maxindx <= 0)
      maxindx = 1;

   if(maxindy <= 0)
      maxindy = 1;

   /* determine the highest index */
   if(maxindx > maxindy)
      maxind = maxindx;
   else
      maxind = maxindy;

   /* scale by coordinate system size and given factor */
   radius = ( (float) paramdata->visuradius / (float) maxind) * scalingfactor;

   return radius;
}


/* if true gp reader should be used for sub-visualizations, otherwise tex reader
 * @returns true if gp reader should be used, false if tex reader should be used */
SCIP_Bool GCGgetUseGp(
   SCIP* scip          /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   SCIP_Bool usegp;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   usegp = paramdata->visuusegp;
   return usegp;
}


/* gets the name of the pdf reader that should be used
 * @returns name of pdf reader */
const char* GCGVisuGetPdfReader(
   SCIP* scip          /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   char* pdfreader;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   pdfreader = paramdata->pdfreader;
   return pdfreader;
}


/* gets the max number of decomps to be included in reports
 * @returns max number of decomps */
int GCGreportGetMaxNDecomps(
   SCIP* scip          /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   int max;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   max = paramdata->rep_maxndecomps;
   return max;
}


/* gets whether a titlepage should be included in reports
 * @returns true iff title page should be generated */
SCIP_Bool GCGreportGetShowTitlepage(
   SCIP* scip          /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   SCIP_Bool showtitle;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   showtitle = paramdata->rep_showtitle;
   return showtitle;
}


/* gets whether a table of contents should be included in reports
 * @returns true iff table of contents should be generated */
SCIP_Bool GCGreportGetShowToc(
   SCIP* scip          /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   SCIP_Bool showtoc;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   showtoc = paramdata->rep_showtoc;
   return showtoc;
}


/* gets whether statistics should be included for each decomp in reports
 * @returns true iff statistics for each decomp should be generated */
SCIP_Bool GCGreportGetShowStatistics(
   SCIP* scip          /**< SCIP data structure */
   )
{
   GCG_PARAMDATA* paramdata;
   SCIP_Bool showstats;

   paramdata = GCGgetParamsVisu(scip);
   assert(paramdata != NULL);

   showstats = paramdata->rep_statistics;
   return showstats;
}


#define paramInitVisu NULL

/* frees all visualization parameters */
extern void GCGVisuFreeParams(
   SCIP* scip,                /* SCIP data structure */
   GCG_PARAMDATA* paramdata   /* input empty paramdata, oputput new set of param data */
   )
{
   assert(scip != NULL);
   assert(paramdata != NULL);

   SCIPfreeMemory(scip, &(paramdata->greycolormastervars));
   SCIPfreeMemory(scip, &(paramdata->greycolormasterconss));
   SCIPfreeMemory(scip, &(paramdata->greycolorlinking));
   SCIPfreeMemory(scip, &(paramdata->greycolorstairlinking));
   SCIPfreeMemory(scip, &(paramdata->greycolorblock));
   SCIPfreeMemory(scip, &(paramdata->greycoloropen));
   SCIPfreeMemory(scip, &(paramdata->greycolornonzero));
   SCIPfreeMemory(scip, &(paramdata->greycolorline));

   SCIPfreeMemory(scip, &paramdata);
}

/** includes the visualization parameters into GCG & initializes them */
SCIP_RETCODE SCIPcreateParamsVisu(
   SCIP* scip,                /**< SCIP data structure */
   GCG_PARAMDATA** paramdata  /**< input empty paramdata, output new set of param data */
   )
{
   assert(*paramdata == NULL);

   SCIP_CALL( SCIPallocMemory(scip, &(*paramdata)) );

   /* init string params with NULL pointer */
   (*paramdata)->pdfreader = NULL;
   (*paramdata)->mancolormastervars = NULL;
   (*paramdata)->mancolormasterconss = NULL;
   (*paramdata)->mancolorlinking = NULL;
   (*paramdata)->mancolorstairlinking = NULL;
   (*paramdata)->mancolorblock = NULL;
   (*paramdata)->mancoloropen = NULL;
   (*paramdata)->mancolornonzero = NULL;
   (*paramdata)->mancolorline = NULL;

   /* initialize black and white color scheme */
   SCIP_CALL( SCIPallocMemoryArray(scip, &((*paramdata)->greycolormastervars), SCIP_MAXSTRLEN) );
   SCIPsnprintf((*paramdata)->greycolormastervars, SCIP_MAXSTRLEN, "%s", GREY_COLOR_MASTERVARS);
   SCIP_CALL( SCIPallocMemoryArray(scip, &((*paramdata)->greycolormasterconss), SCIP_MAXSTRLEN) );
   SCIPsnprintf((*paramdata)->greycolormasterconss, SCIP_MAXSTRLEN, "%s", GREY_COLOR_MASTERCONS);
   SCIP_CALL( SCIPallocMemoryArray(scip, &((*paramdata)->greycolorlinking), SCIP_MAXSTRLEN) );
   SCIPsnprintf((*paramdata)->greycolorlinking, SCIP_MAXSTRLEN, "%s", GREY_COLOR_LINKING);
   SCIP_CALL( SCIPallocMemoryArray(scip, &((*paramdata)->greycolorstairlinking), SCIP_MAXSTRLEN) );
   SCIPsnprintf((*paramdata)->greycolorstairlinking, SCIP_MAXSTRLEN, "%s", GREY_COLOR_STAIRLINKING);
   SCIP_CALL( SCIPallocMemoryArray(scip, &((*paramdata)->greycolorblock), SCIP_MAXSTRLEN) );
   SCIPsnprintf((*paramdata)->greycolorblock, SCIP_MAXSTRLEN, "%s", GREY_COLOR_BLOCK);
   SCIP_CALL( SCIPallocMemoryArray(scip, &((*paramdata)->greycoloropen), SCIP_MAXSTRLEN) );
   SCIPsnprintf((*paramdata)->greycoloropen, SCIP_MAXSTRLEN, "%s", GREY_COLOR_OPEN);
   SCIP_CALL( SCIPallocMemoryArray(scip, &((*paramdata)->greycolornonzero), SCIP_MAXSTRLEN) );
   SCIPsnprintf((*paramdata)->greycolornonzero, SCIP_MAXSTRLEN, "%s", GREY_COLOR_NONZERO);
   SCIP_CALL( SCIPallocMemoryArray(scip, &((*paramdata)->greycolorline), SCIP_MAXSTRLEN) );
   SCIPsnprintf((*paramdata)->greycolorline, SCIP_MAXSTRLEN, "%s", GREY_COLOR_LINE);

   /* add general parameters */
   SCIP_CALL( SCIPaddBoolParam(scip,
      "visual/draftmode", "if true no nonzeros are shown (may improve performance)",
      &(*paramdata)->visudraftmode, FALSE, DEFAULT_VISU_DRAFTMODE, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip,
      "visual/colorscheme", "type number: 0=default, 1=black and white, 2=manual",
      (int*) &(*paramdata)->visucolorscheme, FALSE, DEFAULT_VISU_COLORSCHEME, 0, 2, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip,
      "visual/nonzeroradius", "integer value to scale points on range 1-10",
      &(*paramdata)->visuradius, FALSE, DEFAULT_VISU_RADIUS, 1, 10, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip,
      "visual/nmaxdecompstowrite", "maximum number of decompositions to write (-1: no limit)",
      &(*paramdata)->nmaxdecompstowrite, FALSE, -1, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddStringParam(scip,
      "visual/pdfreader", "pdf reader that opens visualizations in decomposition explorer",
      &(*paramdata)->pdfreader, FALSE,
      DEFAULT_PDFREADER,
      NULL, NULL) );

   /* add parameters for manual colors */
   SCIP_CALL( SCIPaddStringParam(scip,
      "visual/colors/colormastervars", "color for master variables in hex code",
      &(*paramdata)->mancolormastervars, FALSE, DEFAULT_COLOR_MASTERVARS, NULL, NULL) );

   SCIP_CALL( SCIPaddStringParam(scip,
      "visual/colors/colormasterconss", "color for master constraints in hex code",
      &(*paramdata)->mancolormasterconss, FALSE, DEFAULT_COLOR_MASTERCONSS, NULL, NULL) );

   SCIP_CALL( SCIPaddStringParam(scip,
      "visual/colors/colorlinking", "color for linking variables in hex code",
      &(*paramdata)->mancolorlinking, FALSE, DEFAULT_COLOR_LINKING, NULL, NULL) );

   SCIP_CALL( SCIPaddStringParam(scip,
      "visual/colors/colorstairlinking", "color for stairlinking variables in hex code",
      &(*paramdata)->mancolorstairlinking, FALSE, DEFAULT_COLOR_STAIRLINKING, NULL, NULL) );

   SCIP_CALL( SCIPaddStringParam(scip,
      "visual/colors/colorblock", "color for found blocks in hex code",
      &(*paramdata)->mancolorblock, FALSE, DEFAULT_COLOR_BLOCK, NULL, NULL) );

   SCIP_CALL( SCIPaddStringParam(scip,
      "visual/colors/coloropen", "color for open areas in hex code",
      &(*paramdata)->mancoloropen, FALSE, DEFAULT_COLOR_OPEN, NULL, NULL) );

   SCIP_CALL( SCIPaddStringParam(scip,
      "visual/colors/colornonzeros", "color for nonzeros in hex code",
      &(*paramdata)->mancolornonzero, FALSE, DEFAULT_COLOR_NONZERO, NULL, NULL) );

   SCIP_CALL( SCIPaddStringParam(scip,
      "visual/colors/colorlines", "color for lines in hex code",
      &(*paramdata)->mancolorline, FALSE, DEFAULT_COLOR_LINE, NULL, NULL) );

   /* add parameters for report */
   SCIP_CALL( SCIPaddIntParam(scip,
      "visual/report/maxndecomps", "maximum number of decompositions shown in report (best scores first)",
      &(*paramdata)->rep_maxndecomps, FALSE, DEFAULT_REPORT_MAXNDECOMPS, 1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip,
      "visual/report/showtitle", "if true a title page is included",
      &(*paramdata)->rep_showtitle, FALSE, DEFAULT_REPORT_SHOWTITLEPAGE, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip,
      "visual/report/showtoc", "if true a table of contents is included",
      &(*paramdata)->rep_showtoc, FALSE, DEFAULT_REPORT_SHOWTOC, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip,
      "visual/report/showstatistics", "if true statistics are included for each decomp",
      &(*paramdata)->rep_statistics, FALSE, DEFAULT_REPORT_SHOWSTATISTICS, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip,
      "visual/report/usegp", "if true gnuplot is used for sub-visualizations in report, otherwise LaTeX/Tikz",
      &(*paramdata)->visuusegp, FALSE, DEFAULT_VISU_USEGP, NULL, NULL) );

   return SCIP_OKAY;
}
