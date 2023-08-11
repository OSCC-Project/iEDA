/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: metaio.c                                                      */
/*   Name....: Meta Input/Output                                             */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2006-2022 by Thorsten Koch <koch@zib.de>
 * 
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
 */
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <fcntl.h>
#include <assert.h>

#ifndef WITHOUT_ZLIB
#include <zlib.h>
#endif

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"

#include "zimpl/mme.h"
#include "zimpl/metaio.h"

enum file_type { 
  MFP_ERR, 
  MFP_STRG, 
  MFP_FILE,
  MFP_PIPE
#ifndef WITHOUT_ZLIB
  , MFP_ZLIB
#endif /* ! WITHOUT_ZLIB */
 };

typedef struct strg_file     StrgFile;
typedef enum file_type       FileType;

#define STRGFILE_SID  0x53544649
#define MFP_SID       0x4d46505f
#define BUF_SIZE_INC  4096

struct strg_file
{
   SID
   char const* name;
   char const* content;
   int         length;
   int         offset;
   bool        use_copy;
   StrgFile*   next;
};

struct meta_file_ptr
{
   SID
   FileType     type;
   union 
   {
      StrgFile* strg;
      FILE*     file;
#ifndef WITHOUT_ZLIB
      gzFile    zlib;
#endif /* ! WITHOUT_ZLIB */
   } fp;
};

static StrgFile* strg_file_root = NULL;

static bool strgfile_is_valid(StrgFile const* sf)
{
   return (sf != NULL)
      && SID_ok(sf, STRGFILE_SID)
      && (sf->name != NULL)
      && (sf->content != NULL)
      && (sf->length >= 0)
      && (sf->offset >= 0)
      && (sf->offset <= sf->length);
}

void mio_add_strg_file(char const* name, char const* content, bool use_copy)
{
   StrgFile* sf = calloc(1, sizeof(*sf));
   
   assert(name    != NULL);
   assert(content != NULL);
   assert(sf      != NULL);

   sf->name     = strdup(name);
   sf->content  = use_copy ? strdup(content) : content;
   sf->length   = (int)strlen(content); /* the final '\0' is not part of the file */
   sf->offset   = 0;
   sf->use_copy = use_copy;
   sf->next     = strg_file_root;

   SID_set(sf, STRGFILE_SID);
   assert(strgfile_is_valid(sf));
   
   strg_file_root = sf;
}

static int strgfile_getc(StrgFile* sf)
{
   int c;
   
   assert(strgfile_is_valid(sf));
   
   if (sf->offset == sf->length)
      c = EOF;
   else
   {
      assert(sf->offset < sf->length);
         
      c = sf->content[sf->offset];

      sf->offset++;
   }
   return c;
}

static char* strgfile_gets(StrgFile* sf, char* buf, int size)
{
   char* s = NULL;
   
   assert(strgfile_is_valid(sf));
   assert(buf  != NULL);
   assert(size > 0);
   
   if (sf->offset < sf->length)
   {
      int i = 0;
      
      while(sf->offset < sf->length && i < size - 1)
      {
         assert(sf->content[sf->offset] != '\0');
            
         buf[i] = sf->content[sf->offset];
         sf->offset++;
         i++;

         if (buf[i - 1] == '\n')
            break;
      }
      buf[i] = '\0';
      s      = buf;
   }
   return s;
   
}

static bool mfp_is_valid(MFP const* mfp)
{
   return mfp != NULL && SID_ok(mfp, MFP_SID) && mfp->type != MFP_ERR;
}

MFP* mio_open(char const* name, char const* ext)
{
   MFP*      mfp = calloc(1, sizeof(*mfp));
   char*     filename;
   StrgFile* sf;
   
   assert(name != NULL);
   assert(ext  != NULL);
   assert(mfp  != NULL);
   
   filename = malloc(strlen(name) + strlen(ext) + 1);
   assert(filename != NULL);
   strcpy(filename, name);
   
   /* Check if a String File is openend
    */
   for(sf = strg_file_root; sf != NULL; sf = sf->next)
      if (!strcmp(name, sf->name))
         break;

   /* Did we find something ?
    */
   if (sf != NULL)
   {
      mfp->type    = MFP_STRG;
      mfp->fp.strg = sf;
      mfp->fp.strg->offset = 0;

      SID_set(mfp, MFP_SID);
      assert(mfp_is_valid(mfp));
   }
   else
   {
      /* Do we have a pipe?
       */
      if (*filename == '#')
      {
         mfp->type = MFP_PIPE;

         if (NULL == (mfp->fp.file = popen(&filename[1], "r")))
         {
            perror(filename);
            free(mfp);
            mfp = NULL;
         }
         else
         {
            SID_set(mfp, MFP_SID);
            assert(mfp_is_valid(mfp));
         }
      }
      else
      {
         int len;
      
         /* This seems to be a real file, is it available ?
          */
         if (access(filename, R_OK) != 0)
         {
            strcat(filename, ext);

            /* If .gz/.zpl also does not work, revert to the old name
             * to get a better error message.
             */
            if (access(filename, R_OK) != 0)
               strcpy(filename, name);
         }
#ifndef WITHOUT_ZLIB
         len = (int)strlen(filename);
      
         if (len > 3 && !strcmp(&filename[len - 3], ".gz"))
         {
            mfp->type = MFP_ZLIB;
         
            if (NULL == (mfp->fp.zlib = gzopen(filename, "r")))
            {
               perror(filename);
               free(mfp);
               mfp = NULL;
            }
            else
            {
               SID_set(mfp, MFP_SID);
               assert(mfp_is_valid(mfp));
            }
         }
         else
#endif /* !WITHOUT_ZLIB */
         {
            mfp->type = MFP_FILE;

            if (NULL == (mfp->fp.file = fopen(filename, "r")))
            {
               perror(filename);
               free(mfp);
               mfp = NULL;
            }
            else
            {
               SID_set(mfp, MFP_SID);
               assert(mfp_is_valid(mfp));
            }
         }
      }
   }
   free(filename);
   
   return mfp;
}

void mio_close(MFP* mfp)
{
   assert(mfp_is_valid(mfp));

   switch(mfp->type)
   {
   case MFP_STRG :
      break;
   case MFP_FILE :
      fclose(mfp->fp.file);
      break;
   case MFP_PIPE :
      pclose(mfp->fp.file);
      break;
#ifndef WITHOUT_ZLIB
   case MFP_ZLIB :
      gzclose(mfp->fp.zlib);
      break;
#endif /* ! WITHOUT_ZLIB */
   default :
      abort();
   }
   SID_del(mfp);
      
   free(mfp);
}

int mio_getc(MFP const* mfp)
{
   int c;
   
   assert(mfp_is_valid(mfp));

   switch(mfp->type)
   {
   case MFP_STRG :
      c = strgfile_getc(mfp->fp.strg);
      break;
   case MFP_FILE :
   case MFP_PIPE :
      c = fgetc(mfp->fp.file);
      break;
#ifndef WITHOUT_ZLIB
   case MFP_ZLIB :
      c = gzgetc(mfp->fp.zlib);
      break;
#endif /* ! WITHOUT_ZLIB */
   default :
      abort();
   }
   return c;
}

char* mio_gets(MFP const* mfp, char* buf, int len)
{
   char* s = NULL;
   
   assert(mfp_is_valid(mfp));

   switch(mfp->type)
   {
   case MFP_STRG :
      s = strgfile_gets(mfp->fp.strg, buf, len);
      break;
   case MFP_FILE :
   case MFP_PIPE :
      s = fgets(buf, len, mfp->fp.file);
      break;
#ifndef WITHOUT_ZLIB
   case MFP_ZLIB :
      s = gzgets(mfp->fp.zlib, buf, len);
      break;
#endif /* ! WITHOUT_ZLIB */
   default :
      abort();
   }
   return s;
}

char* mio_get_line(MFP const* mfp)
{
   int    size = 1;
   char*  buf = NULL;
   char*  s;
   
   assert(mfp_is_valid(mfp));

   do
   {
      char* t;
      int   pos     = size - 1;
      
      size         += BUF_SIZE_INC - 1;
      buf           = (buf == NULL) ? malloc((size_t)size) : realloc(buf, (size_t)size);
      t             = &buf[pos];
      buf[size - 1] = '@';
      s             = mio_gets(mfp, t, BUF_SIZE_INC);
   }
   while(s != NULL && buf[size - 1] == '\0' && buf[size - 2] != '\n');

   /* nothing read at all ?
    */
   if (s == NULL && size == BUF_SIZE_INC)
   {
      free(buf);
      buf = NULL;
   }
   return buf;
}

void mio_init()
{
   /* Setup for internal test
    */
   static char const* const progstrg = 
      "# $Id: metaio.c,v 1.21 2014/03/03 16:44:14 bzfkocht Exp $\n"
      "#\n"
      "# Generic formulation of the Travelling Salesmen Problem\n"
      "#\n"
      "set V   := { read \"@selftest_cities.dat\" as \"<1s>\" comment \"#\" use cities };\n"
      "set E   := { <i,j> in V * V with i < j };\n"
      "set P[] := powerset(V);\n"
      "set K   := indexset(P) \\ { 0 };\n"
      "\n"
      "do print cities;\n"
      "do print V;\n"
      "do print E;\n"
      "param px[V] := read \"@selftest_cities.dat\" as \"<1s> 2n\" comment \"#\" use cities;\n"
      "param py[V] := read \"@selftest_cities.dat\" as \"<1s> 3n\" comment \"#\" use cities;\n"
      "\n"
      "defnumb dist(a,b) := sqrt((px[a] - px[b])^2 + (py[a] - py[b])^2);\n"
      "\n"
      "var x[E] binary;\n"
      "\n"
      "minimize cost: sum <i,j> in E : dist(i,j) * x[i, j];\n"
      "\n"
      "subto two_connected:\n"
      "   forall <v> in V do\n"
      "      (sum <v,j> in E : x[v,j]) + (sum <i,v> in E : x[i,v]) == 2;\n"
      "\n"
      "subto no_subtour:\n"
      "   forall <k> in K with card(P[k]) > 2 and card(P[k]) < card(V) - 2 do\n"
      "      sum <i,j> in E with <i> in P[k] and <j> in P[k] : x[i,j] \n"
      "      <= card(P[k]) - 1;\n";

   static char const* const datastrg =
      "Berlin     5251 1340\n"
      "Frankfurt  5011  864\n"
      "Leipzig    5133 1237\n"
      "Heidelberg 4941  867\n"
      "Karlsruhe  4901  840\n"
      "Hamburg    5356  998\n"
      "Bayreuth   4993 1159\n"
      "Trier      4974  668\n"
      "Hannover   5237  972\n"
      "Stuttgart  4874  909\n"
      "Passau     4856 1344\n"
      "Augsburg   4833 1089\n"
      "Koblenz    5033  759\n"
      "Dortmund   5148  741\n";

   mio_add_strg_file("@selftest_tspste.zpl", progstrg, false);
   mio_add_strg_file("@selftest_cities.dat", datastrg, false);
}

void mio_exit()
{
   StrgFile* p;
   StrgFile* q;
   
   for(p = strg_file_root; p != NULL; p = q)
   {
      q = p->next;

      assert(strgfile_is_valid(p));

      CLANG_WARN_OFF(-Wcast-qual)

         free((void*)p->name);
      
      if (p->use_copy)
         free((void*)p->content);

      CLANG_WARN_ON
         
      SID_del(p);
      
      free(p);
   }
   strg_file_root = NULL;
}

   
