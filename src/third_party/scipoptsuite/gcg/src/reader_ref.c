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

/**@file   reader_ref.c
 * @brief  REF file reader for structure information
 * @author Gerald Gamrath
 * @author Christian Puchert
 * @author Martin Bergner
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#if defined(_WIN32) || defined(_WIN64)
#else
#include <strings.h> /*lint --e{766}*/ /* needed for strcasecmp() */
#endif
#include <ctype.h>

#include "reader_ref.h"
#include "gcg.h"
#include "scip/cons_linear.h"
#include "cons_decomp.h"
#include "relax_gcg.h"

#define READER_NAME             "refreader"
#define READER_DESC             "file reader for blocks corresponding to a mip in lpb format"
#define READER_EXTENSION        "ref"

/*
 * Data structures
 */
#define REF_MAX_LINELEN       65536
#define REF_MAX_PUSHEDTOKENS  2

/** section in REF File */
enum RefSection
{
   REF_START, REF_NBLOCKS, REF_BLOCKSIZES, REF_BLOCKS, REF_END
};
typedef enum RefSection REFSECTION;

enum RefExpType
{
   REF_EXP_NONE, REF_EXP_UNSIGNED, REF_EXP_SIGNED
};
typedef enum RefExpType REFEXPTYPE;


/** REF reading data */
struct RefInput
{
   SCIP_FILE*            file;               /**< file to read */
   char                  linebuf[REF_MAX_LINELEN]; /**< line buffer */
   char*                 token;              /**< current token */
   char*                 tokenbuf;           /**< token buffer */
   char*                 pushedtokens[REF_MAX_PUSHEDTOKENS]; /**< token stack */
   int                   npushedtokens;      /**< size of token stack */
   int                   linenumber;         /**< current line number */
   int                   linepos;            /**< current line position (column) */
   int                   nblocks;            /**< number of blocks */
   int                   blocknr;            /**< current block number */
   int                   nassignedvars;      /**< number of assigned variables */
   int*                  blocksizes;         /**< array of block sizes */
   int                   totalconss;         /**< total number of constraints */
   int                   totalreadconss;     /**< total number of read constraints */
   SCIP_CONS**           masterconss;        /**< array of constraints to be in the master */
   int                   nmasterconss;       /**< number of constraints to be in the master */
   REFSECTION            section;            /**< current section */
   SCIP_Bool             haserror;           /**< flag to indicate an error occurence */
   SCIP_HASHMAP*         vartoblock;         /**< hashmap mapping variables to blocks (1..nblocks) */
   SCIP_HASHMAP*         constoblock;        /**< hashmap mapping constraints to blocks (1..nblocks) */
};
typedef struct RefInput REFINPUT;

static const char delimchars[] = " \f\n\r\t\v";
static const char tokenchars[] = "-+:<>=";
static const char commentchars[] = "\\";


/*
 * Local methods (for reading)
 */

/** issues an error message and marks the REF data to have errors */
static
void syntaxError(
   SCIP*                 scip,               /**< SCIP data structure */
   REFINPUT*             refinput,           /**< REF reading data */
   const char*           msg                 /**< error message */
   )
{
   char formatstr[256];

   assert(refinput != NULL);

   SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "Syntax error in line %d: %s ('%s')\n",
      refinput->linenumber, msg, refinput->token);
   if( refinput->linebuf[strlen(refinput->linebuf)-1] == '\n' )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "  input: %s", refinput->linebuf);
   }
   else
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "  input: %s\n", refinput->linebuf);
   }
   (void) SCIPsnprintf(formatstr, 256, "         %%%ds\n", refinput->linepos);
   SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, formatstr, "^");
   refinput->section  = REF_END;
   refinput->haserror = TRUE;
}

/** returns whether a syntax error was detected */
static
SCIP_Bool hasError(
   REFINPUT*             refinput            /**< REF reading data */
   )
{
   assert(refinput != NULL);

   return refinput->haserror;
}

/** returns whether the given character is a token delimiter */
static
SCIP_Bool isDelimChar(
   char                  c                   /**< input character */
   )
{
   return (c == '\0') || (strchr(delimchars, c) != NULL);
}

/** returns whether the given character is a single token */
static
SCIP_Bool isTokenChar(
   char                  c                   /**< input character */
   )
{
   return (strchr(tokenchars, c) != NULL);
}

/** returns whether the current character is member of a value string */
static
SCIP_Bool isValueChar(
   char                  c,                  /**< input character */
   char                  nextc,              /**< next input character */
   SCIP_Bool             firstchar,          /**< is the given character the first char of the token? */
   SCIP_Bool*            hasdot,             /**< pointer to update the dot flag */
   REFEXPTYPE*           exptype             /**< pointer to update the exponent type */
   )
{
   assert(hasdot != NULL);
   assert(exptype != NULL);

   if( isdigit(c) )
      return TRUE;
   else if( (*exptype == REF_EXP_NONE) && !(*hasdot) && (c == '.') )
   {
      *hasdot = TRUE;
      return TRUE;
   }
   else if( !firstchar && (*exptype == REF_EXP_NONE) && (c == 'e' || c == 'E') )
   {
      if( nextc == '+' || nextc == '-' )
      {
         *exptype = REF_EXP_SIGNED;
         return TRUE;
      }
      else if( isdigit(nextc) )
      {
         *exptype = REF_EXP_UNSIGNED;
         return TRUE;
      }
   }
   else if( (*exptype == REF_EXP_SIGNED) && (c == '+' || c == '-') )
   {
      *exptype = REF_EXP_UNSIGNED;
      return TRUE;
   }

   return FALSE;
}

/** reads the next line from the input file into the line buffer; skips comments;
 *  returns whether a line could be read
 */
static
SCIP_Bool getNextLine(
   REFINPUT*             refinput            /**< REF reading data */
   )
{
   int i;

   assert(refinput != NULL);

   /* clear the line */
   BMSclearMemoryArray(refinput->linebuf, REF_MAX_LINELEN);

   /* read next line */
   refinput->linepos = 0;
   refinput->linebuf[REF_MAX_LINELEN-2] = '\0';
   if( SCIPfgets(refinput->linebuf, REF_MAX_LINELEN, refinput->file) == NULL )
      return FALSE;
   refinput->linenumber++;
   if( refinput->linebuf[REF_MAX_LINELEN-2] != '\0' )
   {
      SCIPerrorMessage("Error: line %d exceeds %d characters\n", refinput->linenumber, REF_MAX_LINELEN-2);
      refinput->haserror = TRUE;
      return FALSE;
   }
   refinput->linebuf[REF_MAX_LINELEN-1] = '\0';
   refinput->linebuf[REF_MAX_LINELEN-2] = '\0'; /* we want to use lookahead of one char -> we need two \0 at the end */

   /* skip characters after comment symbol */
   for( i = 0; commentchars[i] != '\0'; ++i )
   {
      char* commentstart;

      commentstart = strchr(refinput->linebuf, commentchars[i]);
      if( commentstart != NULL )
      {
         *commentstart = '\0';
         *(commentstart+1) = '\0'; /* we want to use lookahead of one char -> we need two \0 at the end */
      }
   }

   return TRUE;
}

/** reads the next token from the input file into the token buffer; returns whether a token was read */
static
SCIP_Bool getNextToken(
   REFINPUT*             refinput            /**< REF reading data */
   )
{
   SCIP_Bool hasdot;
   REFEXPTYPE exptype;
   char* buf;
   int tokenlen;

   assert(refinput != NULL);
   assert(refinput->linepos < REF_MAX_LINELEN);

   /* check the token stack */
   if( refinput->npushedtokens > 0 )
   {
      SCIPswapPointers((void**)&refinput->token, (void**)&refinput->pushedtokens[refinput->npushedtokens-1]);
      refinput->npushedtokens--;
      SCIPdebugMessage("(line %d) read token again: '%s'\n", refinput->linenumber, refinput->token);
      return TRUE;
   }

   /* skip delimiters */
   buf = refinput->linebuf;
   while( isDelimChar(buf[refinput->linepos]) )
   {
      if( buf[refinput->linepos] == '\0' )
      {
         if( !getNextLine(refinput) )
         {
            refinput->section = REF_END;
            refinput->blocknr++;
            SCIPdebugMessage("(line %d) end of file\n", refinput->linenumber);
            return FALSE;
         }
         else
         {
            if( refinput->section == REF_START )
               refinput->section = REF_NBLOCKS;
            else if( refinput->section == REF_BLOCKSIZES )
            {
               refinput->section = REF_BLOCKS;
               refinput->blocknr = 0;
            }
            return FALSE;
         }
      }
      else
         refinput->linepos++;
   }
   assert(refinput->linepos < REF_MAX_LINELEN);
   assert(!isDelimChar(buf[refinput->linepos]));

   /* check if the token is a value */
   hasdot = FALSE;
   exptype = REF_EXP_NONE;
   if( isValueChar(buf[refinput->linepos], buf[refinput->linepos+1], TRUE, &hasdot, &exptype) ) /*lint !e679*/
   {
      /* read value token */
      tokenlen = 0;
      do
      {
         assert(tokenlen < REF_MAX_LINELEN);
         assert(!isDelimChar(buf[refinput->linepos]));
         refinput->token[tokenlen] = buf[refinput->linepos];
         ++tokenlen;
         ++(refinput->linepos);
         assert(refinput->linepos < REF_MAX_LINELEN);
      }
      while( isValueChar(buf[refinput->linepos], buf[refinput->linepos+1], FALSE, &hasdot, &exptype) ); /*lint !e679*/
   }
   else
   {
      /* read non-value token */
      tokenlen = 0;
      do
      {
         assert(tokenlen < REF_MAX_LINELEN);
         refinput->token[tokenlen] = buf[refinput->linepos];
         tokenlen++;
         refinput->linepos++;
         if( tokenlen == 1 && isTokenChar(refinput->token[0]) )
            break;
      }
      while( !isDelimChar(buf[refinput->linepos]) && !isTokenChar(buf[refinput->linepos]) );

      /* if the token is an equation sense '<', '>', or '=', skip a following '='
       * if the token is an equality token '=' and the next character is a '<' or '>', replace the token by the inequality sense
       */
      if( tokenlen >= 1
         && (refinput->token[tokenlen-1] == '<' || refinput->token[tokenlen-1] == '>' || refinput->token[tokenlen-1] == '=')
         && buf[refinput->linepos] == '=' )
      {
         refinput->linepos++;
      }
      else if( refinput->token[tokenlen-1] == '=' && (buf[refinput->linepos] == '<' || buf[refinput->linepos] == '>') )
      {
         refinput->token[tokenlen-1] = buf[refinput->linepos];
         refinput->linepos++;
      }
   }
   assert(tokenlen < REF_MAX_LINELEN);
   refinput->token[tokenlen] = '\0';

   return TRUE;
}

/** returns whether the current token is a value */
static
SCIP_Bool isInt(
   SCIP*                 scip,               /**< SCIP data structure */
   REFINPUT*             refinput,           /**< REF reading data */
   int*                  value               /**< pointer to store the value (unchanged, if token is no value) */
   )
{
   long val;
   char* endptr;

   assert(refinput != NULL);
   assert(value != NULL);
   assert(!(strcasecmp(refinput->token, "INFINITY") == 0) && !(strcasecmp(refinput->token, "INF") == 0));

   val = strtol(refinput->token, &endptr, 0);
   if( endptr != refinput->token && *endptr == '\0' )
   {
      if( val < INT_MIN || val > INT_MAX ) /*lint !e685*/
         return FALSE;

      *value = (int) val;
      return TRUE;
   }


   return FALSE;
}

/** reads the header of the file */
static
SCIP_RETCODE readStart(
   SCIP*                 scip,               /**< SCIP data structure */
   REFINPUT*             refinput            /**< REF reading data */
   )
{
   assert(refinput != NULL);

   (void) getNextToken(refinput);

   return SCIP_OKAY;
}

/** reads the nblocks section */
static
SCIP_RETCODE readNBlocks(
   SCIP*                 scip,               /**< SCIP data structure */
   REFINPUT*             refinput            /**< REF reading data */
   )
{
   int nblocks;

   assert(refinput != NULL);

   if( getNextToken(refinput) )
   {
      /* read number of blocks */
      if( isInt(scip, refinput, &nblocks) )
      {
         if( refinput->nblocks == -1 )
         {
            refinput->nblocks = nblocks;
            SCIP_CALL( SCIPallocBufferArray(scip, &refinput->blocksizes, nblocks) );
         }
         SCIPdebugMessage("Number of blocks = %d\n", refinput->nblocks);
      }
      else
         syntaxError(scip, refinput, "NBlocks: Value not an integer.\n");
   }
   else
      syntaxError(scip, refinput, "Could not read number of blocks.\n");

   refinput->section = REF_BLOCKSIZES;

   return SCIP_OKAY;
}

/** reads the blocksizes section */
static
SCIP_RETCODE readBlockSizes(
   SCIP*                 scip,               /**< SCIP data structure */
   REFINPUT*             refinput            /**< REF reading data */
   )
{
   int blocknr;
   int blocksize;

   assert(refinput != NULL);

   for( blocknr = 0; getNextToken(refinput) && blocknr < refinput->nblocks; blocknr++ )
   {
      if( isInt(scip, refinput, &blocksize) )
      {
         refinput->blocksizes[blocknr] = blocksize;
         refinput->totalconss += blocksize;
      }
      else
         syntaxError(scip, refinput, "Blocksize: Value not integer.\n");
   }
   if( blocknr != refinput->nblocks )
      syntaxError(scip, refinput, "Could not get sizes for all blocks.\n");

   return SCIP_OKAY;
}

/** reads the blocks section */
static
SCIP_RETCODE readBlocks(
   SCIP*                 scip,               /**< SCIP data structure */
   REFINPUT*             refinput            /**< REF reading data */
   )
{
   int consctr;

   assert(refinput != NULL);

   consctr = 0;

   while( refinput->blocknr < refinput->nblocks )
   {
      SCIPdebugMessage("Reading constraints of block %d/%d\n", refinput->blocknr + 1, refinput->nblocks);
      while( getNextToken(refinput) )
      {
         SCIP_VAR** vars;
         int consnr;
         SCIP_CONS** conss = SCIPgetConss(scip);

         if( isInt(scip, refinput, &consnr) )
         {
            SCIP_CONSHDLR* conshdlr;
            int nvars;
            int v;
            SCIP_CONS* cons;

            SCIPdebugMessage("  -> constraint %d\n", consnr);

            cons = conss[consnr];
            conshdlr = SCIPconsGetHdlr(cons);

            if( strcmp(SCIPconshdlrGetName(conshdlr), "linear") == 0 )
            {
               vars = SCIPgetVarsLinear(scip, cons);
               nvars = SCIPgetNVarsLinear(scip, cons);
            }
            else
            {
               SCIPdebugMessage("    constraint of unknown type.\n");
               continue;
            }

            SCIP_CALL( SCIPhashmapSetImage(refinput->constoblock, cons, (void*) (size_t) (refinput->blocknr+1) ) );

            for( v = 0; v < nvars; v++ )
            {
               SCIP_VAR* var = vars[v];

               SCIPdebugMessage("    -> variable %s\n", SCIPvarGetName(var));

               /* set the block number of the variable to the number of the current block */
               if( SCIPhashmapExists(refinput->vartoblock, var) )
               {
                  int block;
                  block = (int)(size_t) SCIPhashmapGetImage(refinput->vartoblock, var); /*lint !e507*/
                  if( block != refinput->blocknr+1 && block != refinput->nblocks+1 )
                  {
                     SCIP_CALL( SCIPhashmapRemove(refinput->vartoblock, var) );
                     SCIP_CALL( SCIPhashmapSetImage(refinput->vartoblock, var, (void*) (size_t) (refinput->nblocks+1)) );
                  }
               }
               else
               {
                  SCIP_CALL( SCIPhashmapSetImage(refinput->vartoblock, var, (void*) (size_t) (refinput->blocknr+1)) );
               }
               refinput->nassignedvars++;
            }
            consctr++;
            refinput->totalreadconss++;
         }
         else
            syntaxError(scip, refinput, "ConsNr: Value not an integer.\n");
      }

      if( consctr == refinput->blocksizes[refinput->blocknr] )
      {
         refinput->blocknr++;
         consctr = 0;
      }
   }

   return SCIP_OKAY;
}


/** reads an REF file */
static
SCIP_RETCODE readREFFile(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_READER*          reader,             /**< reader data structure */
   REFINPUT*             refinput,           /**< REF reading data */
   DEC_DECOMP*           decomp,             /**< decomposition structure */
   const char*           filename            /**< name of the input file */
   )
{

   assert(scip != NULL);
   assert(reader != NULL);
   assert(refinput != NULL);
   assert(filename != NULL);

   if( SCIPgetStage(scip) < SCIP_STAGE_TRANSFORMED )
      SCIP_CALL( SCIPtransformProb(scip) );

   /* open file */
   refinput->file = SCIPfopen(filename, "r");
   if( refinput->file == NULL )
   {
      SCIPerrorMessage("cannot open file <%s> for reading\n", filename);
      SCIPprintSysError(filename);
      return SCIP_NOFILE;
   }

   if( SCIPgetStage(scip) >= SCIP_STAGE_PRESOLVED )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "Reader 'ref' can only read in unpresolved structures.\n Please re-read the problem and read the decomposition again.\n");
      return SCIP_OKAY;
   }

   /* parse the file */
   refinput->section = REF_START;
   while( refinput->section != REF_END && !hasError(refinput) )
   {
      switch( refinput->section )
      {
      case REF_START:
         SCIP_CALL( readStart(scip, refinput) );
         break;

      case REF_NBLOCKS:
         SCIP_CALL( readNBlocks(scip, refinput) );
         DECdecompSetNBlocks(decomp, refinput->nblocks);
         break;

      case REF_BLOCKSIZES:
         SCIP_CALL( readBlockSizes(scip, refinput) );
         break;

      case REF_BLOCKS:
         SCIP_CALL( readBlocks(scip, refinput) );
         break;

      case REF_END: /* this is already handled in the while() loop */
      default:
         SCIPerrorMessage("invalid REF file section <%d>\n", refinput->section);
         return SCIP_INVALIDDATA;
      }
   }

   /* close file */
   SCIPfclose(refinput->file);

   /* copy information to decomp */
   SCIP_CALL_QUIET( DECfilloutDecompFromHashmaps(scip, decomp, refinput->vartoblock, refinput->constoblock, refinput->nblocks, FALSE) );

   DECdecompSetPresolved(decomp, FALSE);
   DECdecompSetDetector(decomp, NULL);

   return SCIP_OKAY;
}

/** writes a Ref file */
static
SCIP_RETCODE writeREFFile(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_READER*          reader,             /**< ref reader */
   FILE*                 file                /**< target file */

   )
{
   SCIP_HASHMAP *cons2origindex;
   DEC_DECOMP* decomp;

   SCIP_CONS** conss;
   int nconss;
   SCIP_CONS*** subscipconss;
   int* nsubscipconss;
   int i;
   int j;
   int nblocks;
   assert(scip != NULL);
   assert(reader != NULL);
   assert(file != NULL);

   decomp = DECgetBestDecomp(scip, TRUE);

   if( decomp == NULL )
   {
      decomp = GCGgetStructDecomp(scip);
   }

   if( decomp == NULL )
   {
      SCIPwarningMessage(scip, "No reformulation exists, cannot write reformulation file!\n");
      return SCIP_OKAY;
   }
   nblocks = DECdecompGetNBlocks(decomp);
   conss = SCIPgetOrigConss(scip);
   nconss = SCIPgetNOrigConss(scip);

   SCIP_CALL( SCIPhashmapCreate(&cons2origindex, SCIPblkmem(scip), 2*nconss) );
   for( i = 0; i < nconss; ++i )
   {
      int ind;
      SCIP_CONS* cons;

      ind = i+1;

      assert(ind > 0);
      assert(ind <= nconss);
      cons = SCIPfindCons(scip, SCIPconsGetName(conss[i]));

      SCIPdebugMessage("cons added: %d\t%p\t%s\n", ind, (void*)cons, SCIPconsGetName(cons));
      SCIP_CALL( SCIPhashmapInsert(cons2origindex, cons, (void*)(size_t)(ind)) ); /* shift by 1 to enable error checking */
   }

   subscipconss = DECdecompGetSubscipconss(decomp);
   nsubscipconss = DECdecompGetNSubscipconss(decomp);
   SCIPinfoMessage(scip, file, "%d ", nblocks);

   assert(nsubscipconss != NULL);
   assert(subscipconss != NULL);

   for( i = 0; i < nblocks; ++i )
   {
      SCIPinfoMessage(scip, file, "%d ", nsubscipconss[i]);
   }
   SCIPinfoMessage(scip, file, "\n");

   for( i = 0; i < nblocks; ++i )
   {
      for( j = 0; j < nsubscipconss[i]; ++j )
      {
         int ind;
         SCIP_CONS* cons;

         cons = SCIPfindCons(scip, SCIPconsGetName(subscipconss[i][j]));
         ind = (int)(size_t) SCIPhashmapGetImage(cons2origindex, cons); /*lint !e507*/
         SCIPdebugMessage("cons retrieve (o): %d\t%p\t%s\n", ind, (void*)cons, SCIPconsGetName(cons));

         assert(ind > 0); /* shift by 1 */
         assert(ind <= nconss); /* shift by 1 */
         SCIPinfoMessage(scip, file, "%d ", ind-1);
      }
      SCIPinfoMessage(scip, file, "\n");
   }
   SCIPhashmapFree(&cons2origindex);

   DECdecompFree(scip, &decomp);

   return SCIP_OKAY;
}


/*
 * Callback methods of reader
 */

/** destructor of reader to free user data (called when SCIP is exiting) */
#define readerFreeRef NULL

/** problem reading method of reader */
static
SCIP_DECL_READERREAD(readerReadRef)
{
   if( SCIPgetStage(scip) == SCIP_STAGE_INIT || SCIPgetNVars(scip) == 0 || SCIPgetNConss(scip) == 0 )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_DIALOG, NULL, "No problem exists, will not detect structure!\n");
      return SCIP_OKAY;
   }

   SCIP_CALL( SCIPreadRef(scip, reader, filename, result) );

   return SCIP_OKAY;
}


/** problem writing method of reader */
static
SCIP_DECL_READERWRITE(readerWriteRef)
{
   /*lint --e{715}*/
   SCIP_CALL( writeREFFile(scip, reader, file) );
   *result = SCIP_SUCCESS;
   return SCIP_OKAY;
}

/*
 * reader specific interface methods
 */

/** includes the ref file reader in SCIP */
SCIP_RETCODE SCIPincludeReaderRef(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);
   /* include lp reader */
   SCIP_CALL( SCIPincludeReader(scip, READER_NAME, READER_DESC, READER_EXTENSION,
         NULL, readerFreeRef, readerReadRef, readerWriteRef, NULL) );

   return SCIP_OKAY;
}


/** reads problem from file */
SCIP_RETCODE SCIPreadRef(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_READER*          reader,             /**< the file reader itself */
   const char*           filename,           /**< full path and name of file to read, or NULL if stdin should be used */
   SCIP_RESULT*          result              /**< pointer to store the result of the file reading call */
   )
{
   SCIP_RETCODE retcode;
   REFINPUT refinput;
   DEC_DECOMP* decomp;
   int i;
#ifdef SCIP_DEBUG
   SCIP_VAR** vars;
   int nvars;
#endif

   /* initialize REF input data */
   refinput.file = NULL;
   refinput.linebuf[0] = '\0';
   SCIP_CALL( SCIPallocMemoryArray(scip, &refinput.token, REF_MAX_LINELEN) ); /*lint !e506*/
   refinput.token[0] = '\0';
   SCIP_CALL( SCIPallocMemoryArray(scip, &refinput.tokenbuf, REF_MAX_LINELEN) ); /*lint !e506*/
   refinput.tokenbuf[0] = '\0';
   for( i = 0; i < REF_MAX_PUSHEDTOKENS; ++i )
   {
      SCIP_CALL( SCIPallocMemoryArray(scip, &refinput.pushedtokens[i], REF_MAX_LINELEN) ); /*lint !e506 !e866*/
   }
   SCIP_CALL( SCIPallocBufferArray(scip, &refinput.masterconss, 1) );

   refinput.npushedtokens = 0;
   refinput.linenumber = 0;
   refinput.linepos = 0;
   refinput.nblocks = -1;
   refinput.blocknr = -2;
   refinput.totalconss = 0;
   refinput.totalreadconss = 0;
   refinput.nassignedvars = 0;
   refinput.nmasterconss = 0;
   refinput.haserror = FALSE;

   SCIP_CALL( SCIPhashmapCreate(&refinput.vartoblock, SCIPblkmem(scip), SCIPgetNVars(scip)) );
   SCIP_CALL( SCIPhashmapCreate(&refinput.constoblock, SCIPblkmem(scip), SCIPgetNConss(scip)) );

   /* read the file */
   SCIP_CALL( DECdecompCreate(scip, &decomp) );

   retcode = readREFFile(scip, reader, &refinput, decomp, filename);

   if( retcode == SCIP_OKAY )
   {
      SCIP_CALL( GCGconshdlrDecompAddPreexistingDecomp(scip, decomp) );
      SCIPdebugMessage("Read %d/%d conss in ref-file\n", refinput.totalreadconss, refinput.totalconss);
      SCIPdebugMessage("Assigned %d variables to %d blocks.\n", refinput.nassignedvars, refinput.nblocks);
#ifdef SCIP_DEBUG
      SCIP_CALL( SCIPgetVarsData(scip, &vars, &nvars, NULL, NULL, NULL, NULL) );

      for( i = 0; i < nvars; i++ )
      {
         if( GCGvarGetBlock(vars[i]) == -1 )
         {
            SCIPdebugMessage("  -> not assigned: variable %s\n", SCIPvarGetName(vars[i]));
         }
      }
#endif
   }

   SCIP_CALL( DECdecompFree(scip, &decomp) );

   /* free dynamically allocated memory */
   SCIPfreeMemoryArray(scip, &refinput.token);
   SCIPfreeMemoryArray(scip, &refinput.tokenbuf);
   for( i = 0; i < REF_MAX_PUSHEDTOKENS; ++i )
   {
      SCIPfreeMemoryArray(scip, &refinput.pushedtokens[i]);
   }
   SCIPfreeBufferArray(scip, &refinput.masterconss);
   SCIPfreeBufferArray(scip, &refinput.blocksizes);

   /* evaluate the result */
   if( refinput.haserror )
      return SCIP_READERROR;
   else if( retcode == SCIP_OKAY )
   {
      *result = SCIP_SUCCESS;
   }

   return retcode;
}
