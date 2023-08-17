/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: iread.c                                                       */
/*   Name....: Read Instruction                                              */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2001-2022 by Thorsten Koch <koch@zib.de>
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <sys/types.h>
#include <assert.h>

#ifdef WITH_PCRE
#include <pcre2posix.h>
#else
#include <regex.h>
#endif

#ifndef WITHOUT_ZLIB
#include <zlib.h>
#endif

#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"

#include "zimpl/ratlptypes.h"
#include "zimpl/numb.h"
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/set.h"
#include "zimpl/symbol.h"
#include "zimpl/define.h"
#include "zimpl/bound.h"
#include "zimpl/idxset.h"
#include "zimpl/mono.h"
#include "zimpl/term.h"
#include "zimpl/rdefpar.h"
#include "zimpl/conname.h"
#include "zimpl/stmt.h"
#include "zimpl/local.h"
#include "zimpl/list.h"
#include "zimpl/entry.h"
#include "zimpl/heap.h"
#include "zimpl/code.h"
#include "zimpl/inst.h"
#include "zimpl/xlpglue.h"
#include "zimpl/metaio.h"
#include "zimpl/strstore.h"

/* read "datei" as "<1n,3s,12s,4n> %6s" skip 17 use 2000 fs " ;"
 */
#define MAX_FIELDS   65536

CodeNode* i_read_new(CodeNode* self)
{
   char const* filename;
   char const* pattern;
   
   Trace("i_read_new");
   
   assert(code_is_valid(self));

   filename = code_eval_child_strg(self, 0);
   pattern  = code_eval_child_strg(self, 1);   

   code_value_rdef(self, rdef_new(filename, pattern));

   return self;
}

CodeNode* i_read_param(CodeNode* self)
{
   RDef*       rdef;
   RPar const* rpar;
   
   Trace("i_read_param");
   
   assert(code_is_valid(self));

   rdef = rdef_copy(code_eval_child_rdef(self, 0));
   rpar = code_eval_child_rpar(self, 1);   

   rdef_set_param(rdef, rpar);
   
   code_value_rdef(self, rdef);

   return self;
}

CodeNode* i_read_comment(CodeNode* self)
{
   char const* comment;
   
   Trace("i_read_comment");
   
   assert(code_is_valid(self));

   comment = code_eval_child_strg(self, 0);

   code_value_rpar(self, rpar_new_comment(comment));

   return self;
}

CodeNode* i_read_match(CodeNode* self)
{
   char const* match;
   
   Trace("i_read_match");
   
   assert(code_is_valid(self));

   match = code_eval_child_strg(self, 0);

   code_value_rpar(self, rpar_new_match(match));

   return self;
}

CodeNode* i_read_use(CodeNode* self)
{
   Numb const* use;
   int         int_use;
   
   Trace("i_read_use");
   
   assert(code_is_valid(self));

   use = code_eval_child_numb(self, 0);

   if (!numb_is_int(use))
   {
      fprintf(stderr, "*** Error 147: use value ");
      numb_print(stderr, use);
      fprintf(stderr, " is too big or not an integer\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   int_use = numb_toint(use);

   if (int_use <= 0)
   {
      fprintf(stderr, "*** Error 148: use value %d is not positive\n", int_use);
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   code_value_rpar(self, rpar_new_use(int_use));

   return self;
}

CodeNode* i_read_skip(CodeNode* self)
{
   Numb const* skip;
   int         int_skip;
   
   Trace("i_read_skip");
   
   assert(code_is_valid(self));

   skip = code_eval_child_numb(self, 0);

   if (!numb_is_int(skip))
   {
      fprintf(stderr, "*** Error 149: skip value ");
      numb_print(stderr, skip);
      fprintf(stderr, " is too big or not an integer\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   int_skip = numb_toint(skip);
   
   if (int_skip < 0)
   {
      fprintf(stderr, "*** Error 150: skip value %d is negative\n", int_skip);
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   code_value_rpar(self, rpar_new_skip(int_skip));

   return self;
}

static int parse_pattern(
   CodeNode const* self,
   char const*     pattern,
   int*            param_field,
   int*            param_type,
   bool*           is_tuple_list,
   bool*           is_streaming,
   int*            hi_field_no)
{
   char const* sep = " ,<>";
   
   char* temp = strdup(pattern);
   char* s;
   char* t;
   int   field;
   char  type;
   int   params = 0;
   bool  is_single_value = false;
   
   assert(self          != NULL);
   assert(pattern       != NULL);
   assert(param_field   != NULL);
   assert(param_type    != NULL);
   assert(is_tuple_list != NULL);
   assert(hi_field_no   != NULL);
   
   *is_streaming  = false;
   *is_tuple_list = false;
   *hi_field_no   = 0;
   
   /* Is this a tuple_list "<1n,2s>" or
    * an entry_list "<1n,2n> 3s" pattern
    * or a single value "2n"
    * or a stream "s+" or "n+"
    * or a stream tuple list "<n+>" or "<s+>"
    */
   /*lint -e{731} supress "Boolean argument to equal/not equal"
    */
   if (( (NULL == strchr(temp, '>')) != (NULL == strchr(temp, '<')))
      || (strrchr(temp, '>') != strchr(temp, '>'))
      || (strrchr(temp, '<') != strchr(temp, '<')))
   {
      fprintf(stderr, "*** Error 151: Not a valid read template\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }

   /* Is there an + involved ? Than it is a stream.
    */
   if (NULL != strchr(temp, '+'))
   {
      t = temp;
      
      /* We have a stream template
       */
      if (t[0] == '<')
      {

         if (strlen(t) != 4 || (t[1] != 'n' && t[1] != 's') || t[2] != '+' || t[3] != '>')
         {
            fprintf(stderr, "*** Error 151: Not a valid read template\n");
            code_errmsg(self);
            zpl_exit(EXIT_FAILURE);
         }
         *is_tuple_list = true;
         
         t++;
         t[2] = '\0';
      }
      if (strlen(t) != 2 || (t[0] != 'n' && t[0] != 's') || t[1] != '+')
      {
         fprintf(stderr, "*** Error 151: Not a valid read template\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      *is_streaming = true;
      
      param_field[0] = 0;
      param_type [0] = t[0];
      *hi_field_no   = MAX_FIELDS - 1;
      
      free(temp);
      
      return 1;
   }

   /* Single value or tuple list or entry list ?
    */
   s = strchr(temp, '>');
      
   if (NULL == s)
   {
      /* Single value */
      is_single_value = true;
   }
   else
   {
      for(++s; isspace(*s); s++)
         ;

      *is_tuple_list = (*s == '\0');
   }
   /* Here we start to rip the template apart
    */
   for(s = strtok(temp, sep);
       (s != NULL) && (params < MAX_FIELDS);
       s = strtok(NULL, sep))
   {
      if (2 != sscanf(s, "%d%c", &field, &type))
      {
         fprintf(stderr, "*** Error 152: Invalid read template syntax\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      field--;
      
      if ((field < 0) || (field >= MAX_FIELDS))
      {
         fprintf(stderr, "*** Error 153: Invalid field number [%d]\n", field + 1);
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      if ((type != 'n') && (type != 's'))
      {
         fprintf(stderr, "*** Error 154: Invalid field type [%c]\n", type);
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      if (field > *hi_field_no)
         *hi_field_no = field;
      
      param_field[params] = field;
      param_type [params] = type;
      params++;
   }
   free(temp);

   if (is_single_value)
   {
      if (params != 1)
      {
         fprintf(stderr, "*** Error 201: Invalid read template, only one field allowed\n");
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
   }
   else if (params - (*is_tuple_list ? 0 : 1) < 1)
   {
      fprintf(stderr, "*** Error 155: Invalid read template, not enough fields\n");
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   return params;
}

//lint -sem(split_fields, 1p, chneg(2), 2n < MAX_FIELDS, 3p, @n >= 0)
static int split_fields(char* s, int hi_field_no, char* field[])
{
   char* t = s;
   int   fields = 0;

   assert(s           != NULL);
   assert(hi_field_no >= 0);
   assert(hi_field_no <  MAX_FIELDS);
   assert(field       != NULL);
   
   for(;;)
   {
      bool new_field = false;
      
      switch(*s)
      {
      case '\"' :
         s++;
         t = s;
         while((*s != '\0') && (*s != '\"'))
            s++;
         new_field = true;
         break;
      case '\0' :
      case '\t' :
      case ' ' :
      case ',' :
      case ';' :
      case ':' :
         new_field = true;
         break;
      default :
         s++;
         break;
      }
      if (new_field)
      {
         assert(fields <= hi_field_no);
         assert(fields <  MAX_FIELDS);
         
         /*xint --e{661}  Possible access of out-of-bounds pointer */
         char* u = s;
         field[fields] = t;
         fields++;

         if (*s == '\"')
            s++;
         while(isspace(*s))
            s++;
         if (*s == ',' || *s == ';' || *s == ':')
            s++;
         while(isspace(*s))
            s++;
         *u = '\0';

         if (*s == '\0')
            break;

         /* Have we collected all fields we are interested in?
          */
         if (fields - 1 == hi_field_no)
            break;
         
         t = s;
      }
   }
   return fields;
}


static List* process_entry_stream(
   CodeNode const* self,
   List*          list,
   RDef const*    rdef,
   int            line,
   int            fields,
   char**         field,
   int            param_type) /* param_type[0] */
{
   Tuple*      tuple;
   Entry*      entry;
   Numb*       numb;
   int         i;

   assert(rdef   != NULL);
   assert(line   >= 1);
   assert(fields >= 1);
   assert(field  != NULL);
   assert(param_type == 'n' || param_type == 's');

   tuple = tuple_new(0);
   
   for(i = 0; i < fields; i++)
   {
      if (param_type == 'n')
      {
         if (!numb_is_number(field[i]))
         {
            fprintf(stderr, "*** Error 174: Numeric field");
            fprintf(stderr, " read as \"%s\". This is not a number.\n", field[i]);
            fprintf(stderr, "***            File: %s line %d\n",
               rdef_get_filename(rdef), line);
            code_errmsg(self);
            zpl_exit(EXIT_FAILURE);
         }
         numb = numb_new_ascii(field[i]);
         entry = entry_new_numb(tuple, numb);
         numb_free(numb);
      }
      else
      {
         entry = entry_new_strg(tuple, str_new(field[i]));
      }
      if (list == NULL)
         list = list_new_entry(entry);
      else
         list_add_entry(list, entry);
                  
      entry_free(entry);
   }
   tuple_free(tuple);

   return list;
}

static List* process_tuple_stream(
   CodeNode const*   self,
   List*             list,
   RDef const*       rdef,
   int               line,
   int               fields,
   char**            field,
   int               param_type) /* param_type[0] */
{
   Elem*       elem;
   Numb*       numb;
   int         i;

   assert(rdef   != NULL);
   assert(line   >= 1);
   assert(fields >= 1);
   assert(field  != NULL);
   assert(param_type == 'n' || param_type == 's');

   
   for(i = 0; i < fields; i++)
   {
      Tuple* tuple = tuple_new(1);
      
      if (param_type == 'n')
      {
         if (!numb_is_number(field[i]))
         {
            fprintf(stderr, "*** Error 174: Numeric field");
            fprintf(stderr, " read as \"%s\". This is not a number.\n", field[i]);
            fprintf(stderr, "***            File: %s line %d\n",
               rdef_get_filename(rdef), line);
            code_errmsg(self);
            zpl_exit(EXIT_FAILURE);
         }
         numb = numb_new_ascii(field[i]);
         elem = elem_new_numb(numb);
         numb_free(numb);
      }
      else
      {
         elem = elem_new_strg(str_new(field[i]));
      }
      tuple_set_elem(tuple, 0, elem);
      
      if (list == NULL)
         list = list_new_tuple(tuple);
      else
         list_add_tuple(list, tuple);
                  
      tuple_free(tuple);
   }
   return list;
}

static List* process_line(
   CodeNode const* self,
   List*           list,
   RDef const*     rdef,
   int             line,
   bool            is_tuple_list,
   int             dim,
   int             fields,
   char**          field,
   int const*      param_field,
   int const*      param_type) 
{
   Tuple*      tuple;
   Elem*       elem;
   Numb*       numb;
   char*       t;
   int         i;

   assert(rdef        != NULL);
   assert(line        >= 1);
   assert(dim         >= 0);
   assert(fields      >= 1);
   assert(field       != NULL);
   assert(param_type  != NULL);
   assert(param_field != NULL);

   tuple = tuple_new(dim);
         
   for(i = 0; i < dim; i++)
   {
      if (param_field[i] >= fields)
      {
         fprintf(stderr, "*** Error 156: Not enough fields in data\n");
         fprintf(stderr, "***            File: %s line %d\n",
            rdef_get_filename(rdef), line);
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      t = field[param_field[i]];
            
      if (param_type[i] == 'n')
      {
         if (!numb_is_number(t))
         {
            fprintf(stderr, "*** Error 174: Numeric field %d", param_field[i] + 1);
            fprintf(stderr, " read as \"%s\". This is not a number.\n", t);
            fprintf(stderr, "***            File: %s line %d\n",
               rdef_get_filename(rdef), line);
            code_errmsg(self);
            zpl_exit(EXIT_FAILURE);
         }
         numb = numb_new_ascii(t);
         elem = elem_new_numb(numb);
         numb_free(numb);
      }
      else
      {
         elem = elem_new_strg(str_new(t));
      }
      tuple_set_elem(tuple, i, elem);
   }
         
   if (is_tuple_list)
   {
      if (list == NULL)
         list = list_new_tuple(tuple);
      else
         list_add_tuple(list, tuple);
   }
   else
   {
      Entry* entry;
      
      if (param_field[i] >= fields)
      {
         fprintf(stderr, "*** Error 157: Not enough fields in data (value)\n");
         fprintf(stderr, "***            File: %s line %d\n",
            rdef_get_filename(rdef), line);
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
      t = field[param_field[i]];

      if (param_type[i] == 'n')
      {
         if (!numb_is_number(t))
         {
            fprintf(stderr, "*** Error 174: Numeric field %d", param_field[i] + 1);
            fprintf(stderr, " read as \"%s\". This is not a number.\n", t);
            fprintf(stderr, "***            File: %s line %d\n",
               rdef_get_filename(rdef), line);
            code_errmsg(self);
            zpl_exit(EXIT_FAILURE);
         }
         numb = numb_new_ascii(t);
         entry = entry_new_numb(tuple, numb);
         numb_free(numb);
      }
      else
      {
         entry = entry_new_strg(tuple, str_new(t));
      }
      if (list == NULL)
         list = list_new_entry(entry);
      else
         list_add_entry(list, entry);
               
      entry_free(entry);
   }
   tuple_free(tuple);

   return list;
}

/* The result of this function is either a tuple_list "<1n,2s>" or
 * an entry_list "<2n,3s> 1s" or an elem_list "n+".
 * The single value "1s" generates an entry_list. 
 */
CodeNode* i_read(CodeNode* self)
{
   MFP*        fp;
   char*       buf;
   char**      field;
   int*        param_field;  /* template parameter field number */
   int*        param_type;  /* template parameter field type */
   bool        is_tuple_list;
   bool        is_streaming;
   int         hi_field_no;
   int         dim;
   RDef const* rdef;
   List*       list = NULL;
   char*       filename;
   char*       comment;
   char const* match;
   int         skip;
   int         use;
   regex_t     regex;
   
   Trace("i_read");
   
   assert(code_is_valid(self));

   field       = malloc(MAX_FIELDS * sizeof(*field));
   param_field = malloc(MAX_FIELDS * sizeof(*param_field));
   param_type  = malloc(MAX_FIELDS * sizeof(*param_type));

   assert(field       != NULL);
   assert(param_field != NULL);
   assert(param_type  != NULL);
   
   rdef     = code_eval_child_rdef(self, 0);
   use      = rdef_get_use(rdef);
   skip     = rdef_get_skip(rdef);
   dim      = parse_pattern(self,
      rdef_get_pattern(rdef), param_field, param_type,
      &is_tuple_list, &is_streaming, &hi_field_no);

   filename = malloc(strlen(rdef_get_filename(rdef)) + 4);
   assert(filename != NULL);
   strcpy(filename, rdef_get_filename(rdef));

   comment  = malloc(strlen(rdef_get_comment(rdef)) + 3);
   assert(comment  != NULL);
   comment[0] = '\n';
   comment[1] = '\r';
   strcpy(&comment[2], rdef_get_comment(rdef));

   if (NULL != (match = rdef_get_match(rdef)))
   {
      int err = regcomp(&regex, match, REG_EXTENDED | REG_NOSUB);
   
      if (err != 0)
      {
         char errmsg[1024];
      
         regerror(err, &regex, errmsg, sizeof(errmsg));
         fprintf(stderr, "Error 802: %s\n", errmsg);
         code_errmsg(self);
         zpl_exit(EXIT_FAILURE);
      }
   }
   
   /* The last template parameter is the value for the entry_list.
    */
   if (!is_tuple_list)
      dim--;

   if (NULL == (fp = mio_open(filename, ".gz")))
   {
      code_errmsg(self);
      zpl_exit(EXIT_FAILURE);
   }
   else
   {
      int line = 0;
      
      if (verbose >= VERB_NORMAL)
         printf("Reading %s\n", filename);

      while(NULL != (buf = mio_get_line(fp)))
      {
         char* s;
         
         /* Count the line
          */
         line++;

         /* Is this a comment line or is there a comment on the line,
          * then remove it.
          */
         if (NULL != (s = strpbrk(buf, comment)))
            *s = '\0';

         /* If this is an empty line...
          */
         for(s = buf; (*s != '\0') && isspace(*s); s++)
            ;

         /* ... we skip it.
          */
         if (*s == '\0')
         {
            free(buf);
            continue;
         }
         /* do we have regex and if yes it is matched ?
          * if not, skip the line.
          */
         if (match != NULL && regexec(&regex, s, 0, NULL, 0))
         {
            free(buf);
            continue;
         }
         /* Should we skip this line ?
          */
         if (skip-- > 0)
         {
            free(buf);
            continue;
         }

         /* Now we break the line in fields.
          */
         int fields = split_fields(s, hi_field_no, field);
#ifdef DEBUGGING
         {
            fprintf(stdout, "Fields=%d\n", fields);
            for(int i = 0; i < fields; i++)
               fprintf(stdout, "Field[%d]=[%s]\n", i, field[i]);
         }
#endif

         if (is_streaming)
         {
            if (fields == MAX_FIELDS)
            {
               if (stmt_trigger_warning(213))
               {
                  fprintf(stderr, "--- Warning 213: More than %d input fields in line %d of %s\n",
                     fields, line, filename);
                  code_errmsg(self);
               }
            }
            if (is_tuple_list)
               list = process_tuple_stream(self, list, rdef, line,
                  fields, field, param_type[0]);
            else 
               list = process_entry_stream(self, list, rdef, line,
                  fields, field, param_type[0]);
         }
         else
         {
            list = process_line(self, list, rdef, line, is_tuple_list, dim,
               fields, field, param_field, param_type);
         }
         free(buf);
         
         if (--use == 0)
            break;
      }
      mio_close(fp);
   }

   /* If we found nothing to put into the list, we add a dummy entry
    */
   if (list == NULL)
   {
      Tuple* tuple = tuple_new(0);

      if (is_tuple_list)
         list = list_new_tuple(tuple);           
      else
      {
         Set*   set   = set_pseudo_new();
         Entry* entry = entry_new_set(tuple, set);
               
         list = list_new_entry(entry);
         
         entry_free(entry);
         set_free(set);
      }
      tuple_free(tuple);

      /*
        fprintf(stderr, "*** Error 158: Read from file found no data\n");
        code_errmsg(self);
        zpl_exit(EXIT_FAILURE);
      */
   }
   code_value_list(self, list);

   if (match != NULL)
      regfree(&regex);
   
   free(comment);
   free(filename);
   free(param_type);
   free(param_field);
   free(field);
   
   return self;
}



