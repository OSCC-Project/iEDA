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

/**@file   reader_dec.cpp
 * @brief  DEC file reader for structure information
 * @author Lukas Kirchhart
 * @author Martin Bergner
 * @author Gerald Gamrath
 * @author Christian Puchert
 * @author Michael Bastubbe
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

/* #define SCIP_DEBUG */

#include <assert.h>
#include <string.h>
#if defined(_WIN32) || defined(_WIN64)
#else
#include <strings.h> /*lint --e{766}*/ /* needed for strcasecmp() */
#endif
#include <ctype.h>

#include "reader_dec.h"
#include "scip_misc.h"
#include "pub_gcgvar.h"

#include "cons_decomp.h"
#include "cons_decomp.hpp"
#include "pub_decomp.h"

#include "class_partialdecomp.h"
#include "class_detprobdata.h"

#define READER_NAME             "decreader"
#define READER_DESC             "file reader for blocks in dec format"
#define READER_EXTENSION        "dec"


/*
 * Data structures
 */
#define DEC_MAX_LINELEN       65536
#define DEC_MAX_PUSHEDTOKENS  2

/** section in DEC File */
enum DecSection
{
   DEC_START, DEC_INCOMPLETE, DEC_PRESOLVED, DEC_NBLOCKS, DEC_BLOCKCONSS, DEC_MASTERCONSS, DEC_BLOCKVARS, DEC_MASTERVARS, DEC_LINKINGVARS, DEC_END
};
typedef enum DecSection DECSECTION;

/** exponent indicator of the a value */
enum DecExpType
{
   DEC_EXP_NONE
};
typedef enum DecExpType DECEXPTYPE;

/** DEC reading data */
struct DecInput
{
   SCIP_FILE* file;                          /**< file to read */
   char linebuf[DEC_MAX_LINELEN];            /**< line buffer */
   char* token;                              /**< current token */
   char* tokenbuf;                           /**< token buffer */
   char* pushedtokens[DEC_MAX_PUSHEDTOKENS]; /**< token stack */
   int npushedtokens;                        /**< size of token buffer */
   int linenumber;                           /**< current line number */
   int linepos;                              /**< current line position (column) */
   SCIP_Bool presolved;                      /**< does the decomposition refer to the presolved problem? */
   SCIP_Bool haspresolvesection;             /**< does the decomposition have a presolved section  */
   SCIP_Bool incomplete;                     /**< if false the unspecified constraints should be forced to the master (for downward compatibility)  */
   int nblocks;                              /**< number of blocks */
   int blocknr;                              /**< number of the currentblock between 0 and Nblocks-1*/
   DECSECTION section;                       /**< current section */
   SCIP_Bool haserror;                       /**< flag to indicate an error occurence */
   gcg::PARTIALDECOMP* partialdec;           /**< incomplete decomposition */
};
typedef struct DecInput DECINPUT;

/** data for dec reader */
struct SCIP_ReaderData
{

};
static const int NOVALUE = -1;
static const int LINKINGVALUE = -2;
static const char delimchars[] = " \f\n\r\t\v";
static const char tokenchars[] = "-+:<>=";
static const char commentchars[] = "\\";

/*
 * Local methods (for reading)
 */

/** issues an error message and marks the DEC data to have errors */
static
void syntaxError(
   SCIP*                 scip,               /**< SCIP data structure */
   DECINPUT*             decinput,           /**< DEC reading data */
   const char*           msg                 /**< error message */
   )
{
   char formatstr[256];

   assert(decinput != NULL);

   SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "Syntax error in line %d: %s ('%s')\n",
           decinput->linenumber, msg, decinput->token);
   if( decinput->linebuf[strlen(decinput->linebuf) - 1] == '\n' )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "  input: %s", decinput->linebuf);
   }
   else
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, "  input: %s\n", decinput->linebuf);
   }
   (void) SCIPsnprintf(formatstr, 256, "         %%%ds\n", decinput->linepos);
   SCIPverbMessage(scip, SCIP_VERBLEVEL_MINIMAL, NULL, formatstr, "^");
   decinput->section = DEC_END;
   decinput->haserror = TRUE;
}

/** returns whether a syntax error was detected */
static
SCIP_Bool hasError(
   DECINPUT*             decinput            /**< DEC reading data */
   )
{
   assert(decinput != NULL);
   return decinput->haserror;
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
   DECEXPTYPE*           exptype             /**< pointer to update the exponent type */
   )
{  /*lint --e{715}*/
   assert(hasdot != NULL);
   assert(exptype != NULL);

   if( isdigit(c) )
      return TRUE;

   return FALSE;
}

/** reads the next line from the input file into the line buffer; skips comments;
 *  returns whether a line could be read
 */
static
SCIP_Bool getNextLine(
   DECINPUT*             decinput            /**< DEC reading data */
   )
{
   int i;

   assert(decinput != NULL);

   /* clear the line */
   BMSclearMemoryArray(decinput->linebuf, DEC_MAX_LINELEN);

   /* read next line */
   decinput->linepos = 0;
   decinput->linebuf[DEC_MAX_LINELEN - 2] = '\0';
   if( SCIPfgets(decinput->linebuf, DEC_MAX_LINELEN, decinput->file) == NULL )
      return FALSE;
   decinput->linenumber ++;
   if( decinput->linebuf[DEC_MAX_LINELEN - 2] != '\0' )
   {
      SCIPerrorMessage("Error: line %d exceeds %d characters\n", decinput->linenumber, DEC_MAX_LINELEN - 2);
      decinput->haserror = TRUE;
      return FALSE;
   }
   decinput->linebuf[DEC_MAX_LINELEN - 1] = '\0';
   decinput->linebuf[DEC_MAX_LINELEN - 2] = '\0'; /* we want to use lookahead of one char -> we need two \0 at the end */

   /* skip characters after comment symbol */
   for( i = 0; commentchars[i] != '\0'; ++ i )
   {
      char* commentstart;

      commentstart = strchr(decinput->linebuf, commentchars[i]);
      if( commentstart != NULL )
      {
         *commentstart = '\0';
         *(commentstart + 1) = '\0'; /* we want to use lookahead of one char -> we need two \0 at the end */
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

   tmp = * pointer1;
   *pointer1 = * pointer2;
   *pointer2 = tmp;
}

/** reads the next token from the input file into the token buffer; returns whether a token was read */
static
SCIP_Bool getNextToken(
   DECINPUT*             decinput            /**< DEC reading data */
   )
{
   SCIP_Bool hasdot;
   DECEXPTYPE exptype;
   char* buf;
   int tokenlen;

   assert(decinput != NULL);
   assert(decinput->linepos < DEC_MAX_LINELEN);

   /* check the token stack */
   if( decinput->npushedtokens > 0 )
   {
      swapPointers(&decinput->token, &decinput->pushedtokens[decinput->npushedtokens - 1]);
      decinput->npushedtokens --;
      SCIPdebugMessage("(line %d) read token again: '%s'\n", decinput->linenumber, decinput->token);
      return TRUE;
   }

   /* skip delimiters */
   buf = decinput->linebuf;
   while( isDelimChar(buf[decinput->linepos]) )
   {
      if( buf[decinput->linepos] == '\0' )
      {
         if( !getNextLine(decinput) )
         {
            decinput->section = DEC_END;
            SCIPdebugMessage("(line %d) end of file\n", decinput->linenumber);
            return FALSE;
         }
         assert(decinput->linepos == 0);
      }
      else
         decinput->linepos ++;
   }
   assert(decinput->linepos < DEC_MAX_LINELEN);
   assert(! isDelimChar(buf[decinput->linepos]));

   /* check if the token is a value */
   hasdot = FALSE;
   exptype = DEC_EXP_NONE;
   if( isValueChar(buf[decinput->linepos], buf[decinput->linepos + 1], TRUE, &hasdot, &exptype) ) /*lint !e679*/
   {
      /* read value token */
      tokenlen = 0;
      do
      {
         assert(tokenlen < DEC_MAX_LINELEN);
         assert(! isDelimChar(buf[decinput->linepos]));
         decinput->token[tokenlen] = buf[decinput->linepos];
         ++tokenlen;
         ++(decinput->linepos);
         assert(decinput->linepos < DEC_MAX_LINELEN-1);
      }
      while( isValueChar(buf[decinput->linepos], buf[decinput->linepos + 1], FALSE, &hasdot, &exptype) ); /*lint !e679*/
   }
   else
   {
      /* read non-value token */
      tokenlen = 0;
      do
      {
         assert(tokenlen < DEC_MAX_LINELEN);
         decinput->token[tokenlen] = buf[decinput->linepos];
         tokenlen ++;
         decinput->linepos ++;
         if( tokenlen == 1 && isTokenChar(decinput->token[0]) )
            break;
      }
      while( !isDelimChar(buf[decinput->linepos]) && ! isTokenChar(buf[decinput->linepos]) );

      /* if the token is an equation sense '<', '>', or '=', skip a following '='
       * if the token is an equality token '=' and the next character is a '<' or '>', replace the token by the inequality sense
       */
      if( tokenlen >= 1
              && (decinput->token[tokenlen - 1] == '<' || decinput->token[tokenlen - 1] == '>' || decinput->token[tokenlen - 1] == '=')
              && buf[decinput->linepos] == '=' )
      {
         decinput->linepos ++;
      }
      else if( decinput->token[tokenlen - 1] == '=' && (buf[decinput->linepos] == '<' || buf[decinput->linepos] == '>') )
      {
         decinput->token[tokenlen - 1] = buf[decinput->linepos];
         decinput->linepos ++;
      }
   }
   assert(tokenlen < DEC_MAX_LINELEN);
   decinput->token[tokenlen] = '\0';

   SCIPdebugMessage("(line %d) read token: '%s'\n", decinput->linenumber, decinput->token);

   return TRUE;
}

/** puts the current token on the token stack, such that it is read at the next call to getNextToken() */
static
void pushToken(
   DECINPUT*             decinput            /**< DEC reading data */
   )
{
   assert(decinput != NULL);
   assert(decinput->npushedtokens < DEC_MAX_PUSHEDTOKENS);

   swapPointers(&decinput->pushedtokens[decinput->npushedtokens], &decinput->token);
   decinput->npushedtokens ++;
}

/** swaps the current token with the token buffer */
static
void swapTokenBuffer(
   DECINPUT*             decinput            /**< DEC reading data */
   )
{
   assert(decinput != NULL);

   swapPointers(&decinput->token, &decinput->tokenbuf);
}

/** returns whether the current token is a value */
static
SCIP_Bool isInt(
   SCIP*                 scip,               /**< SCIP data structure */
   DECINPUT*             decinput,           /**< DEC reading data */
   int*                  value               /**< pointer to store the value (unchanged, if token is no value) */
   )
{
   long val;
   char* endptr;

   assert(decinput != NULL);
   assert(value != NULL);
   assert(!(strcasecmp(decinput->token, "INFINITY") == 0) && !(strcasecmp(decinput->token, "INF") == 0));

   val = strtol(decinput->token, &endptr, 0);
   if( endptr != decinput->token && * endptr == '\0' )
   {
      if( val < INT_MIN || val > INT_MAX ) /*lint !e685*/
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
   DECINPUT*             decinput            /**< DEC reading data */
   )
{

   assert(decinput != NULL);

   /* remember first token by swapping the token buffer */
   swapTokenBuffer(decinput);

   /* look at next token: if this is a ':', the first token is a name and no section keyword */
   if( getNextToken(decinput) )
   {
      pushToken(decinput);
   }

   /* reinstall the previous token by swapping back the token buffer */
   swapTokenBuffer(decinput);

   if( strcasecmp(decinput->token, "INCOMPLETE") == 0 )
   {
      SCIPdebugMessage("(line %d) new section: INCOMPLETE\n", decinput->linenumber);
      decinput->section = DEC_INCOMPLETE;
      return TRUE;
   }

   if( strcasecmp(decinput->token, "PRESOLVED") == 0 )
   {
      SCIPdebugMessage("(line %d) new section: PRESOLVED\n", decinput->linenumber);
      decinput->section = DEC_PRESOLVED;
      return TRUE;
   }

   if( strcasecmp(decinput->token, "NBLOCKS") == 0 )
   {
      SCIPdebugMessage("(line %d) new section: NBLOCKS\n", decinput->linenumber);
      decinput->section = DEC_NBLOCKS;
      return TRUE;
   }

   if( strcasecmp(decinput->token, "BLOCK") == 0 || strcasecmp(decinput->token, "BLOCKCONSS") == 0 || strcasecmp(decinput->token, "BLOCKCONS") == 0)
   {
      int blocknr;

      decinput->section = DEC_BLOCKCONSS;

      if( getNextToken(decinput) )
      {
         /* read block number */
         if( isInt(scip, decinput, &blocknr) )
         {
            assert(blocknr >= 0);
            assert(blocknr <= decinput->nblocks);

            decinput->blocknr = blocknr - 1;
         }
         else
            syntaxError(scip, decinput, "no block number after block keyword!\n");
      }
      else
         syntaxError(scip, decinput, "no block number after block keyword!\n");

      SCIPdebugMessage("new section: BLOCKCONSS %d\n", decinput->blocknr);

      return TRUE;

   }

   if( strcasecmp(decinput->token, "MASTERCONSS") == 0 || strcasecmp(decinput->token, "MASTERCONS") == 0 )
   {
      decinput->section = DEC_MASTERCONSS;

      SCIPdebugMessage("new section: MASTERCONSS\n");

      return TRUE;
   }

   if( strcasecmp(decinput->token, "BLOCKVARS") == 0 || strcasecmp(decinput->token, "BLOCKVAR") == 0 )
   {
      int blocknr;

      decinput->section = DEC_BLOCKVARS;

      if( getNextToken(decinput) )
      {
         /* read block number */
         if( isInt(scip, decinput, &blocknr) )
         {
            assert(blocknr >= 0);
            assert(blocknr <= decinput->nblocks);

            decinput->blocknr = blocknr - 1;
         }
         else
            syntaxError(scip, decinput, "no block number after block keyword!\n");
      }
      else
         syntaxError(scip, decinput, "no block number after block keyword!\n");

      SCIPdebugMessage("new section: BLOCKVARS %d\n", decinput->blocknr);

      return TRUE;

   }

   if( strcasecmp(decinput->token, "MASTERVARS") == 0 || strcasecmp(decinput->token, "MASTERVAR") == 0
      || strcasecmp(decinput->token, "STATICVAR") == 0 || strcasecmp(decinput->token, "STATICVARS") == 0 )
   {
      decinput->section = DEC_MASTERVARS;

      SCIPdebugMessage("new section: MASTERVARS\n");

      return TRUE;
   }

   if( strcasecmp(decinput->token, "LINKINGVARS") == 0 || strcasecmp(decinput->token, "LINKINGVAR") == 0 )
   {
      decinput->section = DEC_LINKINGVARS;

      SCIPdebugMessage("new section: LINKINGVARS\n");

      return TRUE;
   }

   return FALSE;
}

/** reads the header of the file */
static
SCIP_RETCODE readStart(
   SCIP*                 scip,               /**< SCIP data structure */
   DECINPUT*             decinput            /**< DEC reading data */
   )
{
   assert(decinput != NULL);

   /* everything before first section is treated as comment */
   do
   {
      /* get token */
      if( !getNextToken(decinput) )
         return SCIP_OKAY;
   }
   while( !isNewSection(scip, decinput) );

   return SCIP_OKAY;
}

/** reads the incomplete section */
static
SCIP_RETCODE readIncomplete(
   SCIP*                 scip,               /**< SCIP data structure */
   DECINPUT*             decinput            /**< DEC reading data */
   )
{
   int incomplete;

   assert(scip != NULL);
   assert(decinput != NULL);

   while( getNextToken(decinput) )
   {
      /* check if we reached a new section */
      if( isNewSection(scip, decinput) )
         return SCIP_OKAY;

      /* read if the consdefaultmaster */
      if( isInt(scip, decinput, &incomplete) )
      {
         if( incomplete == 1 )
            decinput->incomplete = TRUE;
         else if ( incomplete == 0 )
            decinput->incomplete = FALSE;
         else
            syntaxError(scip, decinput, "incomplete parameter must be 0 or 1");

         SCIPdebugMessage("The constraints that are not specified in this decomposition are %s forced to the master\n",
            decinput->incomplete ? "" : " not");
      }
   }

   return SCIP_OKAY;
}


/** reads the presolved section */
static
SCIP_RETCODE readPresolved(
   SCIP*                 scip,               /**< SCIP data structure */
   DECINPUT*             decinput            /**< DEC reading data */
   )
{
   int presolved;

   assert(scip != NULL);
   assert(decinput != NULL);

   while( getNextToken(decinput) )
   {
      /* check if we reached a new section */
      if( isNewSection(scip, decinput) )
         return SCIP_OKAY;

      /* read number of blocks */
      if( isInt(scip, decinput, &presolved) )
      {
         decinput->haspresolvesection = TRUE;
         if( presolved == 1 )
         {
            decinput->presolved = TRUE;
         }
         else if ( presolved == 0 )
         {
            decinput->presolved = FALSE;
         }
         else
            syntaxError(scip, decinput, "presolved parameter must be 0 or 1");
         SCIPdebugMessage("Decomposition is%s from presolved problem\n",
            decinput->presolved ? "" : " not");
      }
   }

   return SCIP_OKAY;
}

/** reads the nblocks section */
static
SCIP_RETCODE readNBlocks(
   SCIP*                 scip,               /**< SCIP data structure */
   DECINPUT*             decinput            /**< DEC reading data */
   )
{
   int nblocks;

   assert(scip != NULL);
   assert(decinput != NULL);
   assert(decinput->partialdec != NULL);

   while( getNextToken(decinput) )
   {
      /* check if we reached a new section */
      if( isNewSection(scip, decinput) )
      {
         if( decinput->nblocks == NOVALUE )
            syntaxError(scip, decinput, "no integer value in nblocks section");
         else
            return SCIP_OKAY;
      }

      /* read number of blocks */
      if( isInt(scip, decinput, &nblocks) )
      {
         if( decinput->nblocks == NOVALUE )
         {
            decinput->nblocks = nblocks;
            decinput->partialdec->setNBlocks(nblocks);
         }
         else
            syntaxError(scip, decinput, "2 integer values in nblocks section");
         SCIPdebugMessage("Number of blocks = %d\n", decinput->nblocks);
      }
   }

   return SCIP_OKAY;
}

/** reads the blocks section */
static
SCIP_RETCODE readBlockconss(
   SCIP*                 scip,               /**< SCIP data structure */
   DECINPUT*             decinput,           /**< DEC reading data */
   SCIP_READERDATA*      readerdata          /**< reader data */
   )
{
   int blockid;
   int currblock;

   SCIP_Bool success;
   assert(decinput != NULL);
   assert(readerdata != NULL);
   assert(decinput->partialdec != NULL);

   currblock = 0;

   while( getNextToken(decinput) )
   {
      int i;
      SCIP_CONS* cons;
      SCIP_VAR** curvars = NULL;
      int ncurvars;

      SCIP_Bool conshasvar = FALSE;
      /* check if we reached a new section */
      if( isNewSection(scip, decinput) )
         break;

      /* the token must be the name of an existing cons */
      if( decinput->presolved )
         cons = SCIPfindCons(scip, decinput->token);
      else
         cons = SCIPfindOrigCons(scip, decinput->token);
      if( cons == NULL )
      {
         syntaxError(scip, decinput, "unknown constraint in block section");
         decinput->haserror = TRUE;
         break;
      }

      if( !SCIPconsIsActive(cons) && decinput->presolved )
      {
         SCIPdebugMessage("cons is not active, skip it \n");
         continue;
      }

      /* get all curvars for the specific constraint */
      SCIP_CALL( SCIPgetConsNVars(scip, cons, &ncurvars, &success) );
      assert(success);
      if( ncurvars > 0 )
      {
         SCIP_CALL( SCIPallocBufferArray(scip, &curvars, ncurvars) );
         SCIP_CALL( SCIPgetConsVars(scip, cons, curvars, ncurvars, &success) );
         assert(success);
      }

      blockid = decinput->blocknr;

      for( i = 0; i < ncurvars; i ++ )
      {
         assert(curvars != NULL); /* for flexelint */
         if( decinput->presolved )
         {
            SCIP_VAR* var = SCIPvarGetProbvar(curvars[i]);
            if( !GCGisVarRelevant(var) )
               continue;
         }

         conshasvar = TRUE;
         break; /* found var */
      }

      SCIPfreeBufferArrayNull(scip, &curvars);

      if( !conshasvar )
      {
         SCIPdebugMessage("Cons <%s> has been deleted by presolving or has no variable at all.\n",  SCIPconsGetName(cons) );
         decinput->partialdec->fixConsToBlockByName(decinput->token, currblock);
         ++currblock;
         currblock = currblock % decinput->nblocks;
         continue;
      }
      /*
       * saving block <-> constraint
       */

      if( !decinput->partialdec->isConsOpencons(decinput->partialdec->getDetprobdata()->getIndexForCons(cons))  )
      {
         decinput->haserror = TRUE;
         SCIPwarningMessage(scip, "cons %s is already assigned but is supposed to assigned to %d\n", SCIPconsGetName(cons), (blockid+1));
         return SCIP_OKAY;
      }

      SCIPdebugMessage("cons %s is in block %d\n", SCIPconsGetName(cons), blockid);
      decinput->partialdec->fixConsToBlockByName(decinput->token, blockid);
   }

   return SCIP_OKAY;
}

/** reads the block vars section */
static
SCIP_RETCODE readBlockvars(
   SCIP*                 scip,               /**< SCIP data structure */
   DECINPUT*             decinput,           /**< DEC reading data */
   SCIP_READERDATA*      readerdata          /**< reader data */
   )
{
   int blockid;

   assert(decinput != NULL);
   assert(readerdata != NULL);
   assert(decinput->partialdec != NULL);

   while( getNextToken(decinput) )
   {
      SCIP_Var* var;

      /* check if we reached a new section */
      if( isNewSection(scip, decinput) )
         break;

      /* the token must be the name of an existing cons */
      var = SCIPfindVar(scip, decinput->token);
      if( var == NULL )
      {
         syntaxError(scip, decinput, "unknown variable in block section");
         break;
      }

      if( !SCIPvarIsActive(var) )
      {
         SCIPwarningMessage(scip, "Var <%s> has been fixed or aggregated by presolving, skipping.\n",  SCIPvarGetName(var));
         continue;
      }

      blockid = decinput->blocknr;
      decinput->partialdec->fixVarToBlockByName(decinput->token, blockid);
   }

   return SCIP_OKAY;
}

/** reads the masterconss section */
static
SCIP_RETCODE readMasterconss(
   SCIP*                 scip,               /**< SCIP data structure */
   DECINPUT*             decinput,           /**< DEC reading data */
   SCIP_READERDATA*      readerdata          /**< reader data */
   )
{
   assert(scip != NULL);
   assert(decinput != NULL);
   assert(readerdata != NULL);
   assert(decinput->partialdec != NULL);

   while( getNextToken(decinput) )
   {
      int cons;

      /* check if we reached a new section */
      if( isNewSection(scip, decinput) )
         break;

      /* the token must be the name of an existing constraint */
      cons = decinput->partialdec->getDetprobdata()->getIndexForCons(decinput->token);

      if( cons < 0 )
      {
         syntaxError(scip, decinput, "unknown or deleted constraint in masterconss section");
         break;
      }
      else
      {
         decinput->partialdec->fixConsToMaster(cons);
         SCIPdebugMessage("cons %s is linking constraint\n", decinput->token);
      }
   }

   return SCIP_OKAY;
}

/** reads the mastervars section */
static
SCIP_RETCODE readMastervars(
   SCIP*                 scip,               /**< SCIP data structure */
   DECINPUT*             decinput,           /**< DEC reading data */
   SCIP_READERDATA*      readerdata          /**< reader data */
   )
{
   assert(scip != NULL);
   assert(decinput != NULL);
   assert(readerdata != NULL);
   assert(decinput->partialdec != NULL);

   while( getNextToken(decinput) )
   {
      SCIP_VAR* var;

      /* check if we reached a new section */
      if( isNewSection(scip, decinput) )
         break;

      /* the token must be the name of an existing constraint */
      var = SCIPfindVar(scip, decinput->token);
      if( var == NULL )
      {
         syntaxError(scip, decinput, "unknown constraint in mastervars section");
         break;
      }
      else
      {
         if( !SCIPvarIsActive(var) )
         {
            SCIPdebugMessage("Var <%s> has been fixed or aggregated by presolving, skipping.\n", SCIPvarGetName(var));
            continue;
         }

         decinput->partialdec->fixVarToMasterByName(decinput->token);

         SCIPdebugMessage("var %s is master constraint\n", decinput->token);
      }
   }

   return SCIP_OKAY;
}

/** reads the linkingvars section */
static
SCIP_RETCODE readLinkingvars(
   SCIP*                 scip,               /**< SCIP data structure */
   DECINPUT*             decinput,           /**< DEC reading data */
   SCIP_READERDATA*      readerdata          /**< reader data */
   )
{
   assert(scip != NULL);
   assert(decinput != NULL);
   assert(readerdata != NULL);
   assert(decinput->partialdec != NULL);

   while( getNextToken(decinput) )
   {
      SCIP_Var* var;

      /* check if we reached a new section */
      if( isNewSection(scip, decinput) )
         break;

      /* the token must be the name of an existing constraint */
      var = SCIPfindVar(scip, decinput->token);
      if( var == NULL )
      {
         syntaxError(scip, decinput, "unknown constraint in masterconss section");
         break;
      }
      else
      {
         if( !SCIPvarIsActive(var) )
         {
            SCIPwarningMessage(scip, "Var <%s> has been fixed or aggregated by presolving, skipping.\n", SCIPvarGetName(var));
            continue;
         }

         decinput->partialdec->fixVarToLinkingByName(decinput->token);

         SCIPdebugMessage("cons %s is linking constraint\n", decinput->token);
      }
   }

   return SCIP_OKAY;
}

/** Reads the file and sets the decinput->presolved flag. Resets the file stream afterward.*/
static
SCIP_RETCODE setPresolved(
   SCIP*                 scip,               /**< SCIP data structure */
   DECINPUT*             decinput            /**< DEC reading data */
   )
{
   while( getNextToken(decinput) )
   {
      if( isNewSection(scip, decinput) && decinput->section == DEC_PRESOLVED )
      {
         SCIP_CALL( readPresolved(scip, decinput) );
         break;
      }
   }
   SCIPrewind(decinput->file);
   decinput->section = DEC_START;
   return SCIP_OKAY;
}

/** reads a DEC file */
static
SCIP_RETCODE readDECFile(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_READER*          reader,             /**< Reader data structure */
   DECINPUT*             decinput,           /**< DEC reading data */
   const char*           filename            /**< name of the input file */
   )
{
   SCIP_READERDATA* readerdata;

   assert(decinput != NULL);
   assert(scip != NULL);
   assert(reader != NULL);

   if( SCIPgetStage(scip) == SCIP_STAGE_INIT || SCIPgetNVars(scip) == 0 || SCIPgetNConss(scip) == 0 )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_DIALOG, NULL, "No problem exists, will not read structure!\n");
      return SCIP_OKAY;
   }

   /* open file */
   decinput->file = SCIPfopen(filename, "r");
   if( decinput->file == NULL )
   {
      SCIPerrorMessage("cannot open file <%s> for reading\n", filename);
      SCIPprintSysError(filename);
      return SCIP_NOFILE;
   }

   readerdata = SCIPreaderGetData(reader);
   assert(readerdata != NULL);

   /* parse the file */
   decinput->section = DEC_START;

   setPresolved(scip, decinput);
   if( decinput->presolved && SCIPgetStage(scip) < SCIP_STAGE_PRESOLVED )
   {
      SCIPinfoMessage(scip, NULL, "read presolved decomposition but problem is not presolved yet -> presolve()\n");
      SCIPpresolve(scip);
      assert(decinput->haspresolvesection);
   }

   decinput->partialdec = new gcg::PARTIALDECOMP(scip, !decinput->presolved);

   while( decinput->section != DEC_END && !hasError(decinput) )
   {
      switch( decinput->section )
      {
         case DEC_START:
            SCIP_CALL( readStart(scip, decinput) );
            break;
         case DEC_INCOMPLETE:
            SCIP_CALL( readIncomplete(scip, decinput) );
            break;
         case DEC_PRESOLVED:
            while( getNextToken(decinput) )
            {
               if( isNewSection(scip, decinput) )
               {
                  break;
               }
            }
            break;
         case DEC_NBLOCKS:
            SCIP_CALL( readNBlocks(scip, decinput) );
            if( decinput->haspresolvesection && !decinput->presolved && SCIPgetStage(scip) >= SCIP_STAGE_PRESOLVED )
            {
               SCIPwarningMessage(scip, "decomposition belongs to the unpresolved problem, but the problem is already presolved, please consider to re-read the problem and read the decomposition without presolving when transforming do not succeed.\n");
               break;

            }
            if( !decinput->haspresolvesection )
            {
               SCIPwarningMessage(scip, "decomposition has no presolve section at beginning. The behaviour is undefined. Please add a presolve section. File reading is aborted. \n");
            }
            break;

         case DEC_BLOCKCONSS:
            SCIP_CALL( readBlockconss(scip, decinput, readerdata) );
            break;

         case DEC_MASTERCONSS:
            SCIP_CALL( readMasterconss(scip, decinput, readerdata) );
            break;

         case DEC_BLOCKVARS:
            SCIP_CALL( readBlockvars(scip, decinput, readerdata) );
            break;

         case DEC_MASTERVARS:
            SCIP_CALL( readMastervars(scip, decinput, readerdata) );
            break;

         case DEC_LINKINGVARS:
            SCIP_CALL( readLinkingvars(scip, decinput, readerdata) );
            break;

         case DEC_END: /* this is already handled in the while() loop */
         default:
            SCIPerrorMessage("invalid DEC file section <%d>\n", decinput->section);
            return SCIP_INVALIDDATA;
      }
   }

   decinput->partialdec->prepare();

   if( !decinput->partialdec->isComplete() && !decinput->incomplete )
      decinput->partialdec->setUsergiven(gcg::USERGIVEN::COMPLETED_CONSTOMASTER);

   if( decinput->haserror)
   {
      SCIPinfoMessage(scip, NULL, "error occurred while reading dec file");
      delete decinput->partialdec;
   }
   else
   {
      SCIPinfoMessage(scip, NULL, "just read dec file:\n");
      decinput->partialdec->sort();
      /* if the partialdec was to be completed, add a "vanilla" version as well */
      if( decinput->partialdec->shouldCompletedByConsToMaster() )
      {
         gcg::PARTIALDECOMP* partial = new gcg::PARTIALDECOMP(decinput->partialdec);
         partial->setUsergiven(gcg::USERGIVEN::PARTIAL);
         GCGconshdlrDecompAddPreexisitingPartialDec(scip, partial);
      }
      GCGconshdlrDecompAddPreexisitingPartialDec(scip, decinput->partialdec);
   }

   /* close file */
   SCIPfclose(decinput->file);

   return SCIP_OKAY;
}


/**
 * @brief write partialdec to file in dec format

 * @return scip return code
 */
static
SCIP_RETCODE writePartialdec(
   SCIP* scip,             /**< SCIP data structure */
   FILE* file,             /**< pointer to file to write to */
   gcg::PARTIALDECOMP* partialdec,      /**< partialdec to write */
   SCIP_RESULT* result     /**< will be set to SCIP_SUCCESS if writing was successful */
   )
{
   int nconss;
   int nvars;
   std::vector<int> consindex;
   std::vector<int> varindex;

   assert(partialdec != NULL);

   gcg::DETPROBDATA* detprobdata = partialdec->getDetprobdata();

   nconss = detprobdata->getNConss();
   nvars = detprobdata->getNVars();

   consindex = std::vector<int>(nconss);
   varindex = std::vector<int>(nvars);

   for( int i = 0; i < nconss; ++i )
      consindex[i] = i;
   for( int i = 0; i < nvars; ++i )
      varindex[i] = i;

   /* write meta data of decomposition as comment */
   if( partialdec->getUsergiven() == gcg::USERGIVEN::PARTIAL )
      SCIPinfoMessage(scip, file, "%s%s stems from a partial decomposition provided by the user\n", commentchars, commentchars);
   else if( partialdec->getUsergiven() != gcg::USERGIVEN::NOT )
      SCIPinfoMessage(scip, file, "%s%s provided by the user\n", commentchars, commentchars);
   auto& detectorchain = partialdec->getDetectorchain();
   auto& detectorchaininfo = partialdec->getDetectorchainInfo();
   SCIPinfoMessage(scip, file, "%s%s ndetectors \n", commentchars, commentchars);
   SCIPinfoMessage(scip, file, "%s%s %ld \n", commentchars, commentchars, detectorchain.size());

   SCIPinfoMessage(scip, file, 
      "%s%s name info time nnewblocks %%ofnewborderconss %%ofnewblockconss %%ofnewlinkingvars %%ofnewblockvars\n",
      commentchars, commentchars);

   for( unsigned int i = 0; i < detectorchain.size(); ++i )
   {
      SCIPinfoMessage(scip, file, "%s%s %s %s %f %d %f %f %f %f \n", commentchars, commentchars,
      DECdetectorGetName(detectorchain[i]), detectorchaininfo[i].c_str(), partialdec->getDetectorClockTimes().at(i),
         partialdec->getNNewBlocks(i), partialdec->getPctConssToBorder(i), partialdec->getPctConssToBlock(i),
         partialdec->getPctVarsToBorder(i), partialdec->getPctVarsToBlock(i)) ;
   }

   if( !partialdec->isComplete() )
      SCIPinfoMessage(scip, file, "INCOMPLETE\n1\n");

   if( partialdec->isAssignedToOrigProb() )
      SCIPinfoMessage(scip, file, "PRESOLVED\n0\n");
   else
      SCIPinfoMessage(scip, file, "PRESOLVED\n1\n");

   SCIPinfoMessage(scip, file, "NBLOCKS\n%d\n", partialdec->getNBlocks());

   for( int b = 0; b < partialdec->getNBlocks(); ++b )
   {
      SCIPinfoMessage(scip, file, "BLOCK %d\n", b+1 );
      for( int c = 0; c < partialdec->getNConssForBlock(b); ++c )
      {
         SCIPinfoMessage(scip, file, "%s\n", SCIPconsGetName(detprobdata->getCons(partialdec->getConssForBlock(b)[c])));
      }
   }

   SCIPinfoMessage(scip, file, "MASTERCONSS\n" );
   for( int mc = 0; mc < partialdec->getNMasterconss(); ++mc )
   {
      SCIPinfoMessage(scip, file, "%s\n", SCIPconsGetName(detprobdata->getCons(partialdec->getMasterconss()[mc])));
   }

   if( partialdec->isComplete() )
   {
      *result = SCIP_SUCCESS;
      return SCIP_OKAY;
   }

   SCIPinfoMessage(scip, file, "LINKINGVARS\n" );
   for( int lv = 0; lv < partialdec->getNLinkingvars(); ++lv )
   {
      SCIPinfoMessage(scip, file, "%s\n", SCIPvarGetName(detprobdata->getVar(partialdec->getLinkingvars()[lv])));
   }

   SCIPinfoMessage(scip, file, "MASTERVARS\n%s%s aka STATICVARS\n", commentchars, commentchars );
   for( int mv = 0; mv < partialdec->getNMastervars(); ++mv )
   {
      SCIPinfoMessage(scip, file, "%s\n", SCIPvarGetName(detprobdata->getVar(partialdec->getMastervars()[mv])));
   }

   for( int b = 0; b < partialdec->getNBlocks(); ++b )
   {
      SCIPinfoMessage(scip, file, "BLOCKVARS %d\n", b+1 );
      for( int v = 0; v < partialdec->getNVarsForBlock(b); ++v )
      {
         SCIPinfoMessage(scip, file, "%s\n", SCIPvarGetName(detprobdata->getVar(partialdec->getVarsForBlock(b)[v])));
      }
   }

   *result = SCIP_SUCCESS;

   return SCIP_OKAY;
}


/* reads problem from file */
SCIP_RETCODE readDec(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename,           /**< full path and name of file to read, or NULL if stdin should be used */
   SCIP_RESULT*          result              /**< pointer to store the result of the file reading call */
)
{
   SCIP_RETCODE retcode;
   SCIP_READER* reader;
   DECINPUT decinput;
   int i;

   reader = SCIPfindReader(scip, READER_NAME);
   assert(reader != NULL);

   /* initialize DEC input data */
   decinput.file = NULL;
   decinput.linebuf[0] = '\0';
   SCIP_CALL( SCIPallocMemoryArray(scip, &decinput.token, DEC_MAX_LINELEN) ); /*lint !e506*/
   decinput.token[0] = '\0';
   SCIP_CALL( SCIPallocMemoryArray(scip, &decinput.tokenbuf, DEC_MAX_LINELEN) ); /*lint !e506*/
   decinput.tokenbuf[0] = '\0';
   for( i = 0; i < DEC_MAX_PUSHEDTOKENS; ++ i )
   {
      SCIP_CALL( SCIPallocMemoryArray(scip, &decinput.pushedtokens[i], DEC_MAX_LINELEN) ); /*lint !e506 !e866*/
   }

   decinput.npushedtokens = 0;
   decinput.linenumber = 0;
   decinput.linepos = 0;
   decinput.section = DEC_START;
   decinput.presolved = FALSE;
   decinput.haspresolvesection = FALSE;
   decinput.nblocks = NOVALUE;
   decinput.blocknr = - 2;
   decinput.haserror = FALSE;
   decinput.incomplete = FALSE;

   /* read the file */
   retcode = readDECFile(scip, reader, &decinput, filename);

   /* free dynamically allocated memory */
   SCIPfreeMemoryArray(scip, &decinput.token);
   SCIPfreeMemoryArray(scip, &decinput.tokenbuf);
   for( i = 0; i < DEC_MAX_PUSHEDTOKENS; ++ i )
   {
      SCIPfreeMemoryArray(scip, &decinput.pushedtokens[i]);
   }

   /* evaluate the result */
   if( decinput.haserror )
      return SCIP_READERROR;
   else if( retcode == SCIP_OKAY )
   {
      *result = SCIP_SUCCESS;
   }

   return retcode;
}


/*
 * Callback methods of reader
 */

/** destructor of reader to free user data (called when SCIP is exiting) */
static
SCIP_DECL_READERFREE(readerFreeDec)
{
   SCIP_READERDATA* readerdata;

   readerdata = SCIPreaderGetData(reader);
   assert(readerdata != NULL);

   SCIPfreeMemory(scip, &readerdata);

   return SCIP_OKAY;
}

/** problem reading method of reader */
static
SCIP_DECL_READERREAD(readerReadDec)
{  /*lint --e{715}*/

   if( SCIPgetStage(scip) == SCIP_STAGE_INIT || SCIPgetNVars(scip) == 0 || SCIPgetNConss(scip) == 0 )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_DIALOG, NULL, "Please read in a problem before reading in the corresponding structure file!\n");
      return SCIP_OKAY;
   }

   SCIP_CALL( readDec(scip, filename, result) );

   return SCIP_OKAY;
}

/** problem writing method of reader */
static
SCIP_DECL_READERWRITE(readerWriteDec)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(reader != NULL);

   gcg::PARTIALDECOMP* partialdec = DECgetPartialdecToWrite(scip, transformed);

   if(partialdec == NULL) {
      SCIPwarningMessage(scip, "There is no writable partialdec!\n");
      return SCIP_OKAY;
   }

   writePartialdec(scip, file, partialdec, result);

   return SCIP_OKAY;
}

/*
 * reader specific interface methods
 */

/** includes the dec file reader in SCIP */
SCIP_RETCODE GCGincludeReaderDec(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_READERDATA* readerdata;

   /* create dec reader data */
   SCIP_CALL( SCIPallocMemory(scip, &readerdata) );

   /* include dec reader */
   SCIP_CALL(SCIPincludeReader(scip, READER_NAME, READER_DESC, READER_EXTENSION, NULL,
           readerFreeDec, readerReadDec, readerWriteDec, readerdata));

   return SCIP_OKAY;
}
