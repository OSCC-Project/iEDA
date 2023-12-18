%code requires {
// OpenSTA, Static Timing Analyzer
// Copyright (c) 2021, Parallax Software, Inc.
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.


#include "Liberty.hh"
#include "Vector.hh"
#include "log/Log.hh"
#include "string/Str.hh"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

using namespace ista;

#define YYDEBUG 1

typedef void* yyscan_t;

}

%code provides {
#undef  YY_DECL
#define YY_DECL int lib_lex(LIB_STYPE *yylval_param, yyscan_t yyscanner, ista::LibertyReader *lib_reader)
YY_DECL;

void yyerror(yyscan_t scanner, ista::LibertyReader *lib_reader, const char *str);
}

%union {
  char *string;
  float number;
  int line;
  void* obj;
}

%define api.pure full
%define api.prefix {lib_}
%parse-param {yyscan_t yyscanner}
%parse-param { ista::LibertyReader *lib_reader }
%lex-param   {yyscan_t yyscanner }
%lex-param   { ista::LibertyReader *lib_reader }

%token <number> FLOAT
%token <string> STRING KEYWORD

%type <obj> statement statements complex_attr simple_attr variable group file
%type <obj> attr_values
%type <obj> attr_value simple_attr_value
%type <string> string
%type <line> line
%type <number> volt_expr volt_token

%start file

%{
%}

%%

file:
    group
    {
        lib_reader->set_library_group((LibertyGroupStmt*)$1);
        $$ = $1;
        LOG_INFO << "load liberty file " << lib_reader->get_file_name() << " success.";
    }
    ;

group:
    KEYWORD '(' ')' line '{'
    {
        lib_reader->stringDelete((char*)$1);
        //empty lib TODO
    }
    '}' semi_opt
    {  }
|   KEYWORD '(' ')' line '{'
    {  
        const char* key_word = (const char*)$1;
        LibertyGroupStmt* group = new LibertyGroupStmt(key_word, lib_reader->get_file_name(), lib_reader->get_line_no());
        lib_reader->stringDelete(key_word);
        $<obj>$ = group;
    }
    statements '}' semi_opt
    {  
        LibertyGroupStmt* group = (LibertyGroupStmt*)$<obj>6;
        std::vector<std::unique_ptr<LibertyStmt>>* stmts = (std::vector<std::unique_ptr<LibertyStmt>>*)$7;
        group->set_stmts(std::move(*stmts));
        delete stmts;
        $$ = group;
    }
|   KEYWORD '(' attr_values ')' line '{'
    {
        lib_reader->stringDelete((char*)$1);

        auto attr_values = (std::vector<std::unique_ptr<LibertyAttrValue>>*)$3;
        delete attr_values;

        //empty lib TODO
    }
    '}' semi_opt
    {  }
|   KEYWORD '(' attr_values ')' line '{'
    {  
        const char* key_word = (const char*)$1;
        LibertyGroupStmt* group = new LibertyGroupStmt(key_word, lib_reader->get_file_name(), lib_reader->get_line_no());
        auto attr_values = (std::vector<std::unique_ptr<LibertyAttrValue>>*)$3;
        group->set_attri_values(std::move(*attr_values));
        delete attr_values;
        lib_reader->stringDelete(key_word);
        $<obj>$ = group;
    }
    statements '}' semi_opt
    {  
        LibertyGroupStmt* group = (LibertyGroupStmt*)$<obj>7;
        std::vector<std::unique_ptr<LibertyStmt>>* stmts = (std::vector<std::unique_ptr<LibertyStmt>>*)$8;
        group->set_stmts(std::move(*stmts));
        delete stmts;
        $$ = group;
    }
    ;

line: /* empty */
    { $$ = 1; }
    ;

statements:
    statement
    { 
        std::vector<std::unique_ptr<LibertyStmt>>* stmts = new std::vector<std::unique_ptr<LibertyStmt>>;
        auto stmt = std::unique_ptr<LibertyStmt>((LibertyStmt*)$1); 
        stmts->emplace_back(std::move(stmt));

        $$ = stmts;
    }
   | statements statement
    {
        std::vector<std::unique_ptr<LibertyStmt>>* stmts = (std::vector<std::unique_ptr<LibertyStmt>>*)($1);
        auto stmt = std::unique_ptr<LibertyStmt>((LibertyStmt*)$2); 
        stmts->emplace_back(std::move(stmt));

        $$ = stmts;
    }
    ;

statement:
    simple_attr
    { $$ = $1; }
|   complex_attr
    { $$ = $1; }
|   group
    { $$ = $1; }
|   variable
    ;

simple_attr:
    KEYWORD ':' simple_attr_value line semi_opt
    {  
        const char* key_word = (const char*)$1;
        LibertySimpleAttrStmt* simple_attr_stmt = new LibertySimpleAttrStmt(key_word, lib_reader->get_file_name(), lib_reader->get_line_no());        
        auto attribute_value = std::unique_ptr<LibertyAttrValue>((LibertyAttrValue*)$3);
        simple_attr_stmt->set_attribute_value(std::move(attribute_value));
        lib_reader->stringDelete(key_word);

        $$ = simple_attr_stmt;
    }
    ;

simple_attr_value:
    attr_value
    { $$ = $1; }
|   volt_expr
    { $$ = nullptr; }
/* Unquoted NOT function. */
/* clocked_on : !CP; */
|   '!' string
    { 
        lib_reader->stringDelete((char*)$2);
        $$ = nullptr; 
    }
    ;

complex_attr:
    KEYWORD '(' ')' line semi_opt
    {  
        const char* key_word = (const char*)$1;
        LibertyComplexAttrStmt* complex_attr_stmt = new LibertyComplexAttrStmt(key_word, lib_reader->get_file_name(), lib_reader->get_line_no());
        lib_reader->stringDelete(key_word);
        $$ = complex_attr_stmt;
    }
|   KEYWORD '(' attr_values ')' line semi_opt
    {
        const char* key_word = (const char*)$1;
        std::vector<std::unique_ptr<LibertyAttrValue>>* attri_vals = (std::vector<std::unique_ptr<LibertyAttrValue>>*)$3;
        LibertyComplexAttrStmt* complex_attr_stmt = new LibertyComplexAttrStmt(key_word, lib_reader->get_file_name(), lib_reader->get_line_no());
        complex_attr_stmt->set_attribute_values(std::move(*attri_vals));
        delete attri_vals;
        lib_reader->stringDelete(key_word);
        $$ = complex_attr_stmt;
    }
    ;

attr_values:
    attr_value
    { 
        auto attri_val = std::unique_ptr<LibertyAttrValue>((LibertyAttrValue*)$1);
        std::vector<std::unique_ptr<LibertyAttrValue>>* attri_vals = new std::vector<std::unique_ptr<LibertyAttrValue>>;
        attri_vals->emplace_back(std::move(attri_val));
        $$ = attri_vals;
    }
|   attr_values ',' attr_value
    { 
        auto attri_val = std::unique_ptr<LibertyAttrValue>((LibertyAttrValue*)$3);
        std::vector<std::unique_ptr<LibertyAttrValue>>* attri_vals = (std::vector<std::unique_ptr<LibertyAttrValue>>*)$1;
        attri_vals->emplace_back(std::move(attri_val));

        $$ = $1;
    }
|   attr_values attr_value
    {
        auto attri_val = std::unique_ptr<LibertyAttrValue>((LibertyAttrValue*)$2);
        std::vector<std::unique_ptr<LibertyAttrValue>>* attri_vals = (std::vector<std::unique_ptr<LibertyAttrValue>>*)$1;
        attri_vals->emplace_back(std::move(attri_val));

        $$ = $1;
    }
    ;

variable:
    string '=' FLOAT line semi_opt
    { lib_reader->stringDelete((char*)$1); }
    ;

string:
    STRING
    { $$ = $1; }
|   KEYWORD
    { $$ = $1; }
    ;

attr_value:
    FLOAT
    { 
        LibertyFloatValue* float_value = new LibertyFloatValue((double)$1);
        $$ = float_value; 
    }
    | string
    { 
        LibertyStringValue* string_value = new LibertyStringValue((const char*)$1);
        lib_reader->stringDelete((char*)$1);
        $$ = string_value; 
    }
    ;

/* Voltage expressions are ignored. */
volt_expr:
    volt_token expr_op volt_token
|   volt_expr expr_op volt_token
    ;

volt_token:
    FLOAT
|   KEYWORD
    { 
     lib_reader->stringDelete((char*)$1);
      $$ = 0.0;
    }
    ;

expr_op:
    '+'
|   '-'
|   '*'
|   '/'
    ;

semi_opt:
    /* empty */
|   semi_opt ';'
    ;

%%

void lib_error(yyscan_t scanner, ista::LibertyReader *lib_reader, const char *str)
{
   char* error_msg = Str::printf("Error found in line %lu in verilog file %s\n", 
   lib_reader->get_line_no(), lib_reader->get_file_name()); 
   LOG_ERROR << error_msg << ":" << str ;
}
