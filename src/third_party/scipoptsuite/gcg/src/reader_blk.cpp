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

/**@file   reader_blk.cpp
 * @brief  BLK file reader for structure information
 * @author Gerald Gamrath
 * @author Martin Bergner
 * @author Christian Puchert
 *
 * This reader reads in a blk-file that defines the structur to be used for the decomposition.
 * The structure is defined variable-wise, i.e., the number of blocks and the variables belonging to each block are
 * defined. Afterwards, each constraint that has only variables of one block is added to that block,
 * constraints having variables of more than one block go into the master. If needed, constraints can also be
 * forced into the master, even if they could be transferred to one block.
 *
 * The keywords are:
 * - Presolved: to be followed by either 0 or 1 indicating that the decomposition is for the presolved or unpresolved problem
 * - NBlocks: to be followed by a line giving the number of blocks
 * - Block i with 1 <= i <= nblocks: to be followed by the names of the variables belonging to block i, one per line.
 * - Masterconss: to be followed by names of constraints, one per line, that should go into the master,
 *                even if they only contain variables of one block and could thus be added to this block.
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

#include "reader_blk.h"
#include "relax_gcg.h"
#include "pub_gcgvar.h"
#include "pub_decomp.h"
#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include "scip_misc.h"
#include "class_partialdecomp.h"

#define READER_NAME             "blkreader"
#define READER_DESC             "file reader for structures in blk format"
#define READER_EXTENSION        "blk"

/*
 * Data structures
 */
#define BLK_MAX_LINELEN       65536
#define BLK_MAX_PUSHEDTOKENS  2

/** section in BLK File */
enum BlkSection
{
   BLK_START, BLK_PRESOLVED, BLK_NBLOCKS, BLK_BLOCK, BLK_MASTERCONSS, BLK_END
};
typedef enum BlkSection BLKSECTION;

/** exponent indicator of the a value */
enum BlkExpType
{
   BLK_EXP_NONE, BLK_EXP_UNSIGNED, BLK_EXP_SIGNED
};
typedef enum BlkExpType BLKEXPTYPE;


/** BLK reading data */
struct BlkInput
{
   SCIP_FILE* file;                          /**< file to read */
   char linebuf[BLK_MAX_LINELEN];            /**< line buffer */
   char* token;                              /**< current token */
   char* tokenbuf;                           /**< token buffer */
   char* pushedtokens[BLK_MAX_PUSHEDTOKENS]; /**< token stack */
   int npushedtokens;                        /**< size of token buffer */
   int linenumber;                           /**< current line number */
   int linepos;                              /**< current line position (column) */
   SCIP_Bool presolved;                      /**< does the decomposition refer to the presolved problem? */
   SCIP_Bool haspresolvesection;             /**< does the decomposition have a presolved section  */
   int nblocks;                              /**< number of blocks */
   int blocknr;                              /**< number of the currentblock between 0 and Nblocks-1*/
   BLKSECTION section;                       /**< current section */
   SCIP_Bool haserror;                       /**< flag to indicate an error occurence */
};
typedef struct BlkInput BLKINPUT;

/** data for blk reader */
struct SCIP_ReaderData
{
   int*                  varstoblock;        /**< index=varid; value= -1 or blockID or -2 for multipleblocks */
   int*                  nblockvars;         /**< number of variable per block that are not linkingvars */
   int**                 linkingvarsblocks;  /**< array with blocks assigned to one linking var */
   int*                  nlinkingvarsblocks; /**< array with number of blocks assigned to each linking var */
   SCIP_HASHMAP*         constoblock;        /**< hashmap key=constaint value=block*/
   SCIP_CONS***          blockcons;          /**< array of assignments from constraints to their blocks [blocknr][consid]  */
   int*                  nblockcons;         /**< number of block-constraints for blockID*/
   int                   nlinkingcons;       /**< number of linking constraints*/
   int                   nlinkingvars;       /**< number of linking vars*/
};

static const int NOVALUE = -1;
static const int LINKINGVALUE = -2;
static const char delimchars[] = " \f\n\r\t\v";
static const char tokenchars[] = "-+:<>=";
static const char commentchars[] = "\\";




/*
 * Local methods (for reading)
 */

/** issues an error message and marks the BLK data to have errors */
static
void syntaxError(
   SCIP*                 scip,               /**< SCIP data structure */
   BLKINPUT*             blkinput,           /**< BLK reading data */
   const char*           msg                 /**< error message */
   )
{
   char formatstr[256];

   assert(blkinput != NULL);

   SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "Syntax error in line %d: %s ('%s')\n",
      blkinput->linenumber, msg, blkinput->token);
   if( blkinput->linebuf[strlen(blkinput->linebuf)-1] == '\n' )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "  input: %s", blkinput->linebuf);
   }
   else
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "  input: %s\n", blkinput->linebuf);
   }
   (void) SCIPsnprintf(formatstr, 256, "         %%%ds\n", blkinput->linepos);
   SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, formatstr, "^");
   blkinput->section  = BLK_END;
   blkinput->haserror = TRUE;
}

/** returns whether a syntax error was detected */
static
SCIP_Bool hasError(
   BLKINPUT*             blkinput            /**< BLK reading data */
   )
{
   assert(blkinput != NULL);

   return blkinput->haserror;
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
   BLKEXPTYPE*           exptype             /**< pointer to update the exponent type */
   )
{
   assert(hasdot != NULL);
   assert(exptype != NULL);

   if( isdigit(c) )
      return TRUE;
   else if( (*exptype == BLK_EXP_NONE) && !(*hasdot) && (c == '.') )
   {
      *hasdot = TRUE;
      return TRUE;
   }
   else if( !firstchar && (*exptype == BLK_EXP_NONE) && (c == 'e' || c == 'E') )
   {
      if( nextc == '+' || nextc == '-' )
      {
         *exptype = BLK_EXP_SIGNED;
         return TRUE;
      }
      else if( isdigit(nextc) )
      {
         *exptype = BLK_EXP_UNSIGNED;
         return TRUE;
      }
   }
   else if( (*exptype == BLK_EXP_SIGNED) && (c == '+' || c == '-') )
   {
      *exptype = BLK_EXP_UNSIGNED;
      return TRUE;
   }

   return FALSE;
}

/** reads the next line from the input file into the line buffer; skips comments;
 *  returns whether a line could be read
 */
static
SCIP_Bool getNextLine(
   BLKINPUT*             blkinput            /**< BLK reading data */
   )
{
   int i;

   assert(blkinput != NULL);

   /* clear the line */
   BMSclearMemoryArray(blkinput->linebuf, BLK_MAX_LINELEN);

   /* read next line */
   blkinput->linepos = 0;
   blkinput->linebuf[BLK_MAX_LINELEN-2] = '\0';
   if( SCIPfgets(blkinput->linebuf, BLK_MAX_LINELEN, blkinput->file) == NULL )
      return FALSE;
   blkinput->linenumber++;
   if( blkinput->linebuf[BLK_MAX_LINELEN-2] != '\0' )
   {
      SCIPerrorMessage("Error: line %d exceeds %d characters\n", blkinput->linenumber, BLK_MAX_LINELEN-2);
      blkinput->haserror = TRUE;
      return FALSE;
   }
   blkinput->linebuf[BLK_MAX_LINELEN-1] = '\0';
   blkinput->linebuf[BLK_MAX_LINELEN-2] = '\0'; /* we want to use lookahead of one char -> we need two \0 at the end */

   /* skip characters after comment symbol */
   for( i = 0; commentchars[i] != '\0'; ++i )
   {
      char* commentstart;

      commentstart = strchr(blkinput->linebuf, commentchars[i]);
      if( commentstart != NULL )
      {
         *commentstart = '\0';
         *(commentstart+1) = '\0'; /* we want to use lookahead of one char -> we need two \0 at the end */
      }
   }

   return TRUE;
}

/** swaps the addresses of two pointers */
static
void swapPointers(
   char**                pointer1,           /**< first pointer */
   char**                pointer2            /**< second pointer */
   )
{
   char* tmp;

   tmp = *pointer1;
   *pointer1 = *pointer2;
   *pointer2 = tmp;
}

/** reads the next token from the input file into the token buffer; returns whether a token was read */
static
SCIP_Bool getNextToken(
   BLKINPUT*             blkinput            /**< BLK reading data */
   )
{
   SCIP_Bool hasdot;
   BLKEXPTYPE exptype;
   char* buf;
   int tokenlen;

   assert(blkinput != NULL);
   assert(blkinput->linepos < BLK_MAX_LINELEN);

   /* check the token stack */
   if( blkinput->npushedtokens > 0 )
   {
      swapPointers(&blkinput->token, &blkinput->pushedtokens[blkinput->npushedtokens-1]);
      blkinput->npushedtokens--;
      SCIPdebugMessage("(line %d) read token again: '%s'\n", blkinput->linenumber, blkinput->token);
      return TRUE;
   }

   /* skip delimiters */
   buf = blkinput->linebuf;
   while( isDelimChar(buf[blkinput->linepos]) )
   {
      if( buf[blkinput->linepos] == '\0' )
      {
         if( !getNextLine(blkinput) )
         {
            blkinput->section = BLK_END;
            SCIPdebugMessage("(line %d) end of file\n", blkinput->linenumber);
            return FALSE;
         }
         assert(blkinput->linepos == 0);
      }
      else
         blkinput->linepos++;
   }
   assert(blkinput->linepos < BLK_MAX_LINELEN);
   assert(!isDelimChar(buf[blkinput->linepos]));

   /* check if the token is a value */
   hasdot = FALSE;
   exptype = BLK_EXP_NONE;
   if( isValueChar(buf[blkinput->linepos], buf[blkinput->linepos+1], TRUE, &hasdot, &exptype) ) /*lint !e679*/
   {
      /* read value token */
      tokenlen = 0;
      do
      {
         assert(tokenlen < BLK_MAX_LINELEN);
         assert(!isDelimChar(buf[blkinput->linepos]));
         blkinput->token[tokenlen] = buf[blkinput->linepos];
         ++tokenlen;
         ++(blkinput->linepos);
         assert(blkinput->linepos < BLK_MAX_LINELEN);
      }
      while( isValueChar(buf[blkinput->linepos], buf[blkinput->linepos+1], FALSE, &hasdot, &exptype) ); /*lint !e679*/
   }
   else
   {
      /* read non-value token */
      tokenlen = 0;
      do
      {
         assert(tokenlen < BLK_MAX_LINELEN);
         blkinput->token[tokenlen] = buf[blkinput->linepos];
         tokenlen++;
         blkinput->linepos++;
         if( tokenlen == 1 && isTokenChar(blkinput->token[0]) )
            break;
      }
      while( !isDelimChar(buf[blkinput->linepos]) && !isTokenChar(buf[blkinput->linepos]) );

      /* if the token is an equation sense '<', '>', or '=', skip a following '='
       * if the token is an equality token '=' and the next character is a '<' or '>', replace the token by the inequality sense
       */
      if( tokenlen >= 1
         && (blkinput->token[tokenlen-1] == '<' || blkinput->token[tokenlen-1] == '>' || blkinput->token[tokenlen-1] == '=')
         && buf[blkinput->linepos] == '=' )
      {
         blkinput->linepos++;
      }
      else if( blkinput->token[tokenlen-1] == '=' && (buf[blkinput->linepos] == '<' || buf[blkinput->linepos] == '>') )
      {
         blkinput->token[tokenlen-1] = buf[blkinput->linepos];
         blkinput->linepos++;
      }
   }
   assert(tokenlen < BLK_MAX_LINELEN);
   blkinput->token[tokenlen] = '\0';

   SCIPdebugMessage("(line %d) read token: '%s'\n", blkinput->linenumber, blkinput->token);

   return TRUE;
}

/** puts the current token on the token stack, such that it is read at the next call to getNextToken() */
static
void pushToken(
   BLKINPUT*             blkinput            /**< BLK reading data */
   )
{
   assert(blkinput != NULL);
   assert(blkinput->npushedtokens < BLK_MAX_PUSHEDTOKENS);

   swapPointers(&blkinput->pushedtokens[blkinput->npushedtokens], &blkinput->token);
   blkinput->npushedtokens++;
}

/** swaps the current token with the token buffer */
static
void swapTokenBuffer(
   BLKINPUT*             blkinput            /**< BLK reading data */
   )
{
   assert(blkinput != NULL);

   swapPointers(&blkinput->token, &blkinput->tokenbuf);
}

/** returns whether the current token is a value */
static
SCIP_Bool isInt(
   SCIP*                 scip,               /**< SCIP data structure */
   BLKINPUT*             blkinput,           /**< BLK reading data */
   int*                  value               /**< pointer to store the value (unchanged, if token is no value) */
   )
{
   long val;
   char* endptr;

   assert(blkinput != NULL);
   assert(value != NULL);
   assert(!(strcasecmp(blkinput->token, "INFINITY") == 0) && !(strcasecmp(blkinput->token, "INF") == 0));

   val = strtol(blkinput->token, &endptr, 0);
   if( endptr != blkinput->token && *endptr == '\0' )
   {
      if(val < INT_MIN || val > INT_MAX ) /*lint !e685*/
         return FALSE;

      *value = (int) val;
      return TRUE;
   }

   return FALSE;
}

/** checks whether the current token is a section identifier, and if yes, switches to the corresponding section */
static
SCIP_Bool isNewSection(
   SCIP*                 scip,               /**< SCIP data structure */
   BLKINPUT*             blkinput            /**< BLK reading data */
   )
{
   SCIP_Bool iscolon;

   assert(blkinput != NULL);

   /* remember first token by swapping the token buffer */
   swapTokenBuffer(blkinput);

   /* look at next token: if this is a ':', the first token is a name and no section keyword */
   iscolon = FALSE;
   if( getNextToken(blkinput) )
   {
      iscolon = (strcmp(blkinput->token, ":") == 0);
      pushToken(blkinput);
   }

   /* reinstall the previous token by swapping back the token buffer */
   swapTokenBuffer(blkinput);

   /* check for ':' */
   if( iscolon )
      return FALSE;

   if( strcasecmp(blkinput->token, "PRESOLVED") == 0 )
   {
      SCIPdebugMessage("(line %d) new section: PRESOLVED\n", blkinput->linenumber);
      blkinput->section = BLK_PRESOLVED;
      return TRUE;
   }

   if( strcasecmp(blkinput->token, "NBLOCKS") == 0 )
   {
      SCIPdebugMessage("(line %d) new section: NBLOCKS\n", blkinput->linenumber);
      blkinput->section = BLK_NBLOCKS;
      return TRUE;
   }

   if( strcasecmp(blkinput->token, "BLOCK") == 0 )
   {
      int blocknr;

      blkinput->section = BLK_BLOCK;

      if( getNextToken(blkinput) )
      {
         /* read block number */
         if( isInt(scip, blkinput, &blocknr) )
         {
            assert(blocknr >= 0);
            assert(blocknr <= blkinput->nblocks);

            blkinput->blocknr = blocknr-1;
         }
         else
            syntaxError(scip, blkinput, "no block number after block keyword!\n");
      }
      else
         syntaxError(scip, blkinput, "no block number after block keyword!\n");

      SCIPdebugMessage("new section: BLOCK %d\n", blkinput->blocknr);

      return TRUE;

   }

   if( strcasecmp(blkinput->token, "MASTERCONSS") == 0 )
   {
      blkinput->section = BLK_MASTERCONSS;

      SCIPdebugMessage("new section: MASTERCONSS\n");

      return TRUE;
   }

   if( strcasecmp(blkinput->token, "END") == 0 )
   {
      SCIPdebugMessage("(line %d) new section: END\n", blkinput->linenumber);
      blkinput->section = BLK_END;
      return TRUE;
   }

   return FALSE;
}

/** reads the header of the file */
static
SCIP_RETCODE readStart(
   SCIP*                 scip,               /**< SCIP data structure */
   BLKINPUT*             blkinput            /**< BLK reading data */
   )
{
   assert(blkinput != NULL);

   /* everything before first section is treated as comment */
   do
   {
      /* get token */
      if( !getNextToken(blkinput) )
         return SCIP_OKAY;
   }
   while( !isNewSection(scip, blkinput) );

   return SCIP_OKAY;
}

/** reads the presolved section */
static
SCIP_RETCODE readPresolved(
   SCIP*                 scip,               /**< SCIP data structure */
   BLKINPUT*             blkinput            /**< DEC reading data */
   )
{
   int presolved;

   assert(scip != NULL);
   assert(blkinput != NULL);

   while( getNextToken(blkinput) )
   {
      /* check if we reached a new section */
      if( isNewSection(scip, blkinput) )
         return SCIP_OKAY;

      /* read number of blocks */
      if( isInt(scip, blkinput, &presolved) )
      {
         blkinput->haspresolvesection = TRUE;
         if( presolved == 1 )
            blkinput->presolved = TRUE;
         else if ( presolved == 0 )
            blkinput->presolved = FALSE;
         else
            syntaxError(scip, blkinput, "presolved parameter must be 0 or 1");
         SCIPdebugMessage("Decomposition is%s from presolved problem\n",
            blkinput->presolved ? "" : " not");
      }
   }

   return SCIP_OKAY;
}

/** reads the nblocks section */
static
SCIP_RETCODE readNBlocks(
   SCIP*                 scip,               /**< SCIP data structure */
   gcg::PARTIALDECOMP*           partialdec,              /**< partialdec to edit */
   BLKINPUT*             blkinput            /**< BLK reading data */
   )
{
   int nblocks;

   assert(scip != NULL);
   assert(blkinput != NULL);
   assert(partialdec != NULL);

   while( getNextToken(blkinput) )
   {
      /* check if we reached a new section */
      if( isNewSection(scip, blkinput) )
      {
         if( blkinput->nblocks == NOVALUE )
            syntaxError(scip, blkinput, "no integer value in nblocks section");
         else
            return SCIP_OKAY;
      }

      /* read number of blocks */
      if( isInt(scip, blkinput, &nblocks) )
      {
         if( blkinput->nblocks == NOVALUE )
         {
            blkinput->nblocks = nblocks;
            partialdec->setNBlocks(nblocks);
         }
         else
            syntaxError(scip, blkinput, "2 integer values in nblocks section");
         SCIPdebugMessage("Number of blocks = %d\n", blkinput->nblocks);
      }
   }

   return SCIP_OKAY;
}

/** reads a block section */
static
SCIP_RETCODE readBlock(
   SCIP*                 scip,               /**< SCIP data structure */
   BLKINPUT*             blkinput,           /**< BLK reading data */
   gcg::PARTIALDECOMP*           partialdec,              /**< partialdec to edit */
   SCIP_READERDATA*      readerdata          /**< reader data */
   )
{
   int blockid;

   assert(blkinput != NULL);
   assert(partialdec != NULL);

   blockid = blkinput->blocknr;

   while( getNextToken(blkinput) )
   {
      SCIP_VAR* var;
      int varidx;
      int oldblock;

      /* check if we reached a new section */
      if( isNewSection(scip, blkinput) )
         return SCIP_OKAY;

      /* the token must be the name of an existing variable */
      var = SCIPfindVar(scip, blkinput->token);
      if( var == NULL )
      {
         syntaxError(scip, blkinput, "unknown variable in block section");
         return SCIP_OKAY;
      }

      varidx = SCIPvarGetProbindex(var);
      oldblock = readerdata->varstoblock[varidx];

      /* set the block number of the variable to the number of the current block */
      if( oldblock == NOVALUE )
      {
         SCIPdebugMessage("\tVar %s temporary in block %d.\n", SCIPvarGetName(var), blockid);
         readerdata->varstoblock[varidx] = blockid;
         ++(readerdata->nblockvars[blockid]);
         partialdec->fixVarToBlockByName(blkinput->token, blockid);
      }
      /* variable was assigned to another (non-linking) block before, so it becomes a linking variable, now */
      else if( (oldblock != LINKINGVALUE) )
      {
         assert(oldblock != blockid);
         SCIPdebugMessage("\tVar %s is linking (old %d != %d new).\n", SCIPvarGetName(var), oldblock, blockid);

         readerdata->varstoblock[varidx] = LINKINGVALUE;

         /* decrease the number of variables in the old block and increase the number of linking variables */
         --(readerdata->nblockvars[oldblock]);
         ++(readerdata->nlinkingvars);

         assert(readerdata->nlinkingvarsblocks[varidx] == 0);
         assert(readerdata->linkingvarsblocks[varidx] == NULL);
         SCIP_CALL( SCIPallocMemoryArray(scip, &readerdata->linkingvarsblocks[varidx], 2) ); /*lint !e506 !e866*/
         readerdata->linkingvarsblocks[varidx][0] = oldblock;
         readerdata->linkingvarsblocks[varidx][1] = blockid;
         readerdata->nlinkingvarsblocks[varidx] = 2;

         partialdec->fixVarToLinkingByName(blkinput->token);
      }
      /* variable is a linking variable already, store the new block to which it belongs */
      else
      {
         assert(oldblock == LINKINGVALUE);
         assert(readerdata->nlinkingvarsblocks[varidx] >= 2);
         assert(readerdata->linkingvarsblocks[varidx] != NULL);
         SCIP_CALL( SCIPreallocMemoryArray(scip, &readerdata->linkingvarsblocks[varidx], (size_t) readerdata->nlinkingvarsblocks[varidx] + 1) ); /*lint !e866*/
         readerdata->linkingvarsblocks[varidx][readerdata->nlinkingvarsblocks[varidx]] = blockid;
         ++(readerdata->nlinkingvarsblocks[varidx]);
      }
   }

   return SCIP_OKAY;
}

/** reads the masterconss section */
static
SCIP_RETCODE readMasterconss(
   SCIP*                 scip,               /**< SCIP data structure */
   BLKINPUT*             blkinput,           /**< BLK reading data */
   gcg::PARTIALDECOMP*           partialdec,              /**< PARTIALDECOMP to edit */
   SCIP_READERDATA*      readerdata          /**< reader data */
   )
{
   assert(blkinput != NULL);
   assert(partialdec != NULL);

   while( getNextToken(blkinput) )
   {
      SCIP_CONS* cons;

      /* check if we reached a new section */
      if( isNewSection(scip, blkinput) )
         return SCIP_OKAY;

      /* the token must be the name of an existing constraint */
      cons = SCIPfindCons(scip, blkinput->token);
      if( cons == NULL )
      {
         syntaxError(scip, blkinput, "unknown constraint in masterconss section");
         return SCIP_OKAY;
      }
      else
      {
         assert(SCIPhashmapGetImage(readerdata->constoblock, cons) == (void*) (size_t) NOVALUE);
         SCIP_CALL( SCIPhashmapSetImage(readerdata->constoblock, cons, (void*) (size_t) (blkinput->nblocks +1)) );
         partialdec->fixConsToMasterByName(blkinput->token);
      }
   }

   return SCIP_OKAY;
}

/** fills the whole Decomp struct after the blk file has been read */
static
SCIP_RETCODE fillDecompStruct(
   SCIP*                 scip,               /**< SCIP data structure */
   BLKINPUT*             blkinput,           /**< blk reading data */
   DEC_DECOMP*           decomp,             /**< DEC_DECOMP structure to fill */
   gcg::PARTIALDECOMP*           partialdec,              /**< partialdec to fill for internal handling */
   SCIP_READERDATA*      readerdata          /**< reader data*/
   )
{
   SCIP_HASHMAP* constoblock;
   SCIP_CONS** allcons;
   SCIP_VAR** allvars;

   SCIP_VAR** consvars;
   SCIP_RETCODE retcode;
   int i;
   int j;
   int nvars;
   int blocknr;
   int nconss;
   int nblocks;

   assert(scip != NULL);
   assert(blkinput != NULL);
   assert(readerdata != NULL);
   assert(partialdec != NULL);

   allcons = SCIPgetConss(scip);
   allvars = SCIPgetVars(scip);
   nvars = SCIPgetNVars(scip);
   nconss = SCIPgetNConss(scip);
   nblocks = blkinput->nblocks;

   DECdecompSetPresolved(decomp, blkinput->presolved);
   DECdecompSetNBlocks(decomp, nblocks);
   DECdecompSetDetector(decomp, NULL);

   SCIP_CALL( DECdecompSetType(decomp, DEC_DECTYPE_ARROWHEAD) );

   /* hashmaps */
   SCIP_CALL( SCIPhashmapCreate(&constoblock, SCIPblkmem(scip), nconss) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &consvars, nvars) );

   /* assign unassigned variables as master variables */
   for( i = 0; i < nvars; ++i)
   {
      SCIP_VAR* var;
      var = allvars[i];
      if( readerdata->varstoblock[i] == NOVALUE )
      {
         partialdec->fixVarToMasterByName(SCIPvarGetName(var));
      }
   }

   /* assign constraints to blocks or declare them linking */
   for( i = 0; i < nconss;  ++i )
   {
      SCIP_CONS* cons;

      cons = allcons[i];

      if( SCIPhashmapGetImage(readerdata->constoblock, cons) == (void*) (size_t) LINKINGVALUE )
      {
         SCIP_CALL( SCIPhashmapInsert(constoblock, cons, (void*) (size_t) (nblocks+1)) );

         SCIPdebugMessage("cons %s is linking\n", SCIPconsGetName(cons));
      }
      /* check whether all variables in the constraint belong to one block */
      else
      {
         int nconsvars;

         nconsvars = GCGconsGetNVars(scip, cons);
         assert(nconsvars < nvars);

         SCIP_CALL( GCGconsGetVars(scip, cons, consvars, nvars) );

         blocknr = -1;

         /* find the first unique assignment of a contained variable to a block */
         for( j = 0; j < nconsvars; ++j )
         {
            /* if a contained variable is directly transferred to the master, the constraint is a linking constraint */
            if( readerdata->varstoblock[SCIPvarGetProbindex(consvars[j])] == NOVALUE )
            {
               blocknr = -1;
               break;
            }
            /* assign the constraint temporarily to the block of the variable, if it is unique */
            if( blocknr == -1 && readerdata->varstoblock[SCIPvarGetProbindex(consvars[j])] != LINKINGVALUE )
            {
               blocknr = readerdata->varstoblock[SCIPvarGetProbindex(consvars[j])];
            }
         }
         if( blocknr != -1 )
         {
            /* check whether all contained variables are copied into the assigned block;
             * if not, the constraint is treated as a linking constraint
             */
            for( j = 0; j < nconsvars; ++j )
            {
               int varidx = SCIPvarGetProbindex(consvars[j]);
               int varblock = readerdata->varstoblock[varidx];
               assert(varblock != NOVALUE);

               if( varblock != LINKINGVALUE && varblock != blocknr )
               {
                  blocknr = -1;
                  break;
               }
               else if( varblock == LINKINGVALUE )
               {
                  int k;

                  for( k = 0; k < readerdata->nlinkingvarsblocks[varidx]; ++k )
                  {
                     if( readerdata->linkingvarsblocks[varidx][k] == blocknr )
                        break;
                  }
                  /* we did not break, so the variable is not assigned to the block */
                  if( k == readerdata->nlinkingvarsblocks[varidx] )
                  {
                     blocknr = -1;
                     break;
                  }
               }
            }
         }

         if( blocknr == -1 )
         {
            SCIP_CALL( SCIPhashmapInsert(constoblock, cons, (void*) (size_t) (nblocks+1)) );
            partialdec->fixConsToMasterByName(SCIPconsGetName(cons));

            SCIPdebugMessage("constraint <%s> is a linking constraint\n",
               SCIPconsGetName(cons));
         }
         else
         {
            SCIP_CALL( SCIPhashmapInsert(constoblock, cons, (void*) (size_t) (blocknr+1)) );
            partialdec->fixConsToBlockByName(SCIPconsGetName(cons), blocknr);
            SCIPdebugMessage("constraint <%s> is assigned to block %d\n", SCIPconsGetName(cons), blocknr);
         }
      }
   }

   SCIPinfoMessage(scip, NULL, "just read blk file:\n");

   retcode = DECfilloutDecompFromConstoblock(scip, decomp, constoblock, nblocks, FALSE);
   SCIPfreeMemoryArray(scip, &consvars);

   return retcode;
}

/** reads an BLK file */
static
SCIP_RETCODE readBLKFile(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_READER*          reader,             /**< reader data structure */
   BLKINPUT*             blkinput,           /**< BLK reading data */
   const char*           filename            /**< name of the input file */
   )
{
   SCIP_RETCODE retcode = SCIP_ERROR;
   DEC_DECOMP *decdecomp;
   int i;
   int nconss;
   int nblocksread;
   int nvars;
   SCIP_READERDATA* readerdata;
   SCIP_CONS** conss;
   nblocksread = FALSE;

   assert(scip != NULL);
   assert(reader != NULL);
   assert(blkinput != NULL);

   if( SCIPgetStage(scip) < SCIP_STAGE_TRANSFORMED )
      SCIP_CALL( SCIPtransformProb(scip) );

   readerdata = SCIPreaderGetData(reader);
   assert(readerdata != NULL);

   readerdata->nlinkingcons = SCIPgetNConss(scip);
   readerdata->nlinkingvars = 0;
   nvars = SCIPgetNVars(scip);
   conss = SCIPgetConss(scip);
   nconss = SCIPgetNConss(scip);

   /* alloc: var -> block mapping */
   SCIP_CALL( SCIPallocMemoryArray(scip, &readerdata->varstoblock, nvars) );
   for( i = 0; i < nvars; i ++ )
   {
      readerdata->varstoblock[i] = NOVALUE;
   }

   /* alloc: linkingvar -> blocks mapping */
   SCIP_CALL( SCIPallocMemoryArray(scip, &readerdata->linkingvarsblocks, nvars) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &readerdata->nlinkingvarsblocks, nvars) );
   BMSclearMemoryArray(readerdata->linkingvarsblocks, nvars);
   BMSclearMemoryArray(readerdata->nlinkingvarsblocks, nvars);

   /* cons -> block mapping */
   SCIP_CALL( SCIPhashmapCreate(&readerdata->constoblock, SCIPblkmem(scip), nconss) );
   for( i = 0; i < SCIPgetNConss(scip); i ++ )
   {
      SCIP_CALL( SCIPhashmapInsert(readerdata->constoblock, conss[i], (void*)(size_t) NOVALUE) );
   }

   /* open file */
   blkinput->file = SCIPfopen(filename, "r");
   if( blkinput->file == NULL )
   {
      SCIPerrorMessage("cannot open file <%s> for reading\n", filename);
      SCIPprintSysError(filename);
      return SCIP_NOFILE;
   }

   /* parse the file */
   blkinput->section = BLK_START;
   gcg::PARTIALDECOMP* newpartialdec = NULL;
   while( blkinput->section != BLK_END && !hasError(blkinput) )
   {
      switch( blkinput->section )
      {
      case BLK_START:
         SCIP_CALL( readStart(scip, blkinput) );
         break;

      case BLK_PRESOLVED:
         SCIP_CALL( readPresolved(scip, blkinput) );
         if( blkinput->presolved && SCIPgetStage(scip) < SCIP_STAGE_PRESOLVED )
         {
            SCIPpresolve(scip);
            assert(blkinput->haspresolvesection);

            /** @bug GCG should be able to presolve the problem first */

            //            SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "decomposition belongs to the presolved problem, please presolve the problem first.\n");
         }
         break;

      case BLK_NBLOCKS:
         if( blkinput->haspresolvesection )
         {
            newpartialdec = new gcg::PARTIALDECOMP(scip, !blkinput->presolved);
            newpartialdec->setUsergiven(gcg::USERGIVEN::COMPLETED_CONSTOMASTER);
         }
         SCIP_CALL( readNBlocks(scip, newpartialdec, blkinput) );
         if( blkinput->haspresolvesection && !blkinput->presolved && SCIPgetStage(scip) >= SCIP_STAGE_PRESOLVED )
         {
            SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "decomposition belongs to the original problem, please re-read the problem and read the decomposition without presolving.\n");
            goto TERMINATE;
            }
         if( !blkinput->haspresolvesection )
         {
            SCIPwarningMessage(scip, "decomposition has no presolve section at beginning. It is assumed to belong to the unpresolved problem but the behaviour is undefined. See the FAQ for further information.\n");
            blkinput->presolved = FALSE;
            newpartialdec = new gcg::PARTIALDECOMP(scip, !blkinput->presolved);
            newpartialdec->setUsergiven(gcg::USERGIVEN::COMPLETED_CONSTOMASTER);
         }

         break;

      case BLK_BLOCK:
         if( nblocksread == FALSE )
         {
            /* alloc n vars per block */
            SCIP_CALL( SCIPallocMemoryArray(scip, &readerdata->nblockvars, blkinput->nblocks) );
            SCIP_CALL( SCIPallocMemoryArray(scip, &readerdata->nblockcons, blkinput->nblocks) );
            SCIP_CALL( SCIPallocMemoryArray(scip, &readerdata->blockcons, blkinput->nblocks) );
            for( i = 0; i < blkinput->nblocks; ++i )
            {
               readerdata->nblockvars[i] = 0;
               readerdata->nblockcons[i] = 0;
               SCIP_CALL( SCIPallocMemoryArray(scip, &(readerdata->blockcons[i]), nconss) ); /*lint !e866*/
            }
            nblocksread = TRUE;
         }
         SCIP_CALL( readBlock(scip, blkinput, newpartialdec, readerdata) );
         break;

      case BLK_MASTERCONSS:
         SCIP_CALL( readMasterconss(scip, blkinput, newpartialdec, readerdata) );
         break;

      case BLK_END: /* this is already handled in the while() loop */
      default:
         SCIPerrorMessage("invalid BLK file section <%d>\n", blkinput->section);
         return SCIP_INVALIDDATA;
      }
   }

   SCIP_CALL( DECdecompCreate(scip, &decdecomp) );

   /* fill decomp */
   retcode = fillDecompStruct(scip, blkinput, decdecomp, newpartialdec, readerdata);

   GCGconshdlrDecompAddPreexisitingPartialDec(scip, newpartialdec);

   SCIP_CALL( DECdecompFree(scip, &decdecomp) );

   for( i = 0; i < nvars; ++i )
   {
      assert(readerdata->linkingvarsblocks[i] != NULL || readerdata->nlinkingvarsblocks[i] == 0);
      if( readerdata->nlinkingvarsblocks[i] > 0 )
      {
         SCIPfreeMemoryArray(scip, &readerdata->linkingvarsblocks[i]);
      }
   }

 TERMINATE:
   if( nblocksread )
   {
      for( i = blkinput->nblocks - 1; i >= 0; --i )
      {
         SCIPfreeMemoryArray(scip, &(readerdata->blockcons[i]));
      }
      SCIPfreeMemoryArray(scip, &readerdata->blockcons);
      SCIPfreeMemoryArray(scip, &readerdata->nblockcons);
      SCIPfreeMemoryArray(scip, &readerdata->nblockvars);
   }

   SCIPhashmapFree(&readerdata->constoblock);

   SCIPfreeMemoryArray(scip, &readerdata->nlinkingvarsblocks);
   SCIPfreeMemoryArray(scip, &readerdata->linkingvarsblocks);
   SCIPfreeMemoryArray(scip, &readerdata->varstoblock);

   /* close file */
   SCIPfclose(blkinput->file);

   return retcode;
}


/*
 * Callback methods of reader
 */

/** destructor of reader to free user data (called when SCIP is exiting) */
static
SCIP_DECL_READERFREE(readerFreeBlk)
{
   SCIP_READERDATA* readerdata;

   readerdata = SCIPreaderGetData(reader);
   assert(readerdata != NULL);

   SCIPfreeMemory(scip, &readerdata);

   return SCIP_OKAY;
}


/** problem reading method of reader */
static
SCIP_DECL_READERREAD(readerReadBlk)
{  /*lint --e{715} */

   if( SCIPgetStage(scip) == SCIP_STAGE_INIT || SCIPgetNVars(scip) == 0 || SCIPgetNConss(scip) == 0 )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_DIALOG, NULL, "Please read in a problem before reading in the corresponding structure file!\n");
      return SCIP_OKAY;
   }
   SCIP_CALL( SCIPreadBlk(scip, filename, result) );

   return SCIP_OKAY;
}


/** problem writing method of reader */
static
SCIP_DECL_READERWRITE(readerWriteBlk)
{ /*lint --e{715}*/
   return SCIP_OKAY;
}

/*
 * reader specific interface methods
 */

/** includes the blk file reader in SCIP */
SCIP_RETCODE SCIPincludeReaderBlk(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_READERDATA* readerdata;

   /* create blk reader data */
   SCIP_CALL( SCIPallocMemory(scip, &readerdata) );

   /* include blk reader */
   SCIP_CALL( SCIPincludeReader(scip, READER_NAME, READER_DESC, READER_EXTENSION, NULL,
         readerFreeBlk, readerReadBlk, readerWriteBlk, readerdata) );

   return SCIP_OKAY;
}


/* reads problem from file */
SCIP_RETCODE SCIPreadBlk(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename,           /**< full path and name of file to read, or NULL if stdin should be used */
   SCIP_RESULT*          result              /**< pointer to store the result of the file reading call */
   )
{
   SCIP_RETCODE retcode;
   SCIP_READER* reader;
   BLKINPUT blkinput;
   int i;
   char* ext;
   char  copyfilename[SCIP_MAXSTRLEN];

   reader = SCIPfindReader(scip, READER_NAME);
   assert(reader != NULL);

   (void) SCIPsnprintf(copyfilename, SCIP_MAXSTRLEN, "%s", filename);
   SCIPsplitFilename(copyfilename, NULL, NULL, &ext, NULL);

   if ( strcmp(ext, "blk") != 0 )
   {
      return SCIP_READERROR;
   }

   /* initialize BLK input data */
   blkinput.file = NULL;
   blkinput.linebuf[0] = '\0';
   SCIP_CALL( SCIPallocMemoryArray(scip, &blkinput.token, BLK_MAX_LINELEN) ); /*lint !e506*/
   blkinput.token[0] = '\0';
   SCIP_CALL( SCIPallocMemoryArray(scip, &blkinput.tokenbuf, BLK_MAX_LINELEN) ); /*lint !e506*/
   blkinput.tokenbuf[0] = '\0';
   for( i = 0; i < BLK_MAX_PUSHEDTOKENS; ++i )
   {
      SCIP_CALL( SCIPallocMemoryArray(scip, &blkinput.pushedtokens[i], BLK_MAX_LINELEN) ); /*lint !e506 !e866*/
   }

   blkinput.npushedtokens = 0;
   blkinput.linenumber = 0;
   blkinput.linepos = 0;
   blkinput.section = BLK_START;
   blkinput.presolved = FALSE;
   blkinput.haspresolvesection = FALSE;
   blkinput.nblocks = -1;
   blkinput.blocknr = -2;
   blkinput.haserror = FALSE;

   /* read the file */
   retcode = readBLKFile(scip, reader, &blkinput, filename);

   /* free dynamically allocated memory */
   SCIPfreeMemoryArray(scip, &blkinput.token);
   SCIPfreeMemoryArray(scip, &blkinput.tokenbuf);
   for( i = 0; i < BLK_MAX_PUSHEDTOKENS; ++i )
   {
      SCIPfreeMemoryArray(scip, &blkinput.pushedtokens[i]);
   }

   /* evaluate the result */
   if( blkinput.haserror )
      return SCIP_READERROR;
   else if( retcode == SCIP_OKAY )
   {
      *result = SCIP_SUCCESS;
   }

   return SCIP_OKAY;
}
