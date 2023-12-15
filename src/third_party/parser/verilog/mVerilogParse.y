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

#include <vector>

#include "log/Log.hh"
#include "string/Str.hh"
#include "VerilogReader.hh"

using namespace ista;

#define YYDEBUG 1

typedef void* yyscan_t;

}

%code provides {
#undef  YY_DECL
#define YY_DECL int verilog_lex(VERILOG_STYPE *yylval_param, yyscan_t yyscanner, ista::VerilogReader *verilog_reader)
YY_DECL;

void yyerror(yyscan_t scanner,ista::VerilogReader *verilog_reader, const char *str);
}

%union {
  int integer;
  char* string;
  const char* constant;
  void*  obj;
}

%define api.pure full
%define api.prefix {verilog_}
%parse-param {yyscan_t yyscanner}
%parse-param { ista::VerilogReader* verilog_reader }
%lex-param   {yyscan_t yyscanner }
%lex-param   { ista::VerilogReader* verilog_reader }

%token INT CONSTANT ID STRING MODULE ENDMODULE ASSIGN PARAMETER DEFPARAM
%token WIRE WAND WOR TRI INPUT OUTPUT INOUT SUPPLY1 SUPPLY0 REG

%left '-' '+'
%left '*' '/'
%left NEG     /* negation--unary minus */

%type <string> ID STRING
%type <integer> WIRE WAND WOR TRI INPUT OUTPUT INOUT SUPPLY1 SUPPLY0
%type <integer> INT parameter_exprs parameter_expr module_begin
%type <constant> CONSTANT
%type <integer> dcl_type port_dcl_type
%type <obj> stmt declaration instance parameter parameter_dcls parameter_dcl
%type <obj> defparam param_values param_value port_dcl
%type <obj> stmts stmt_seq net_assignments continuous_assign port_dcls
%type <obj> net_assignment
%type <obj> dcl_arg
%type <obj> dcl_args
%type <obj> port net_scalar net_bit_select net_part_select net_assign_lhs
%type <obj> net_constant net_expr port_ref port_expr named_pin_net_expr
%type <obj> inst_named_pin net_named net_expr_concat
%type <obj> port_list port_refs inst_ordered_pins
%type <obj> inst_named_pins net_exprs inst_pins
%type <obj> module_parameter module_parameters

%start file

%{
%}

%%

file:
    modules
    ;

modules:
    // empty
|   modules module
    ;

module_begin:
    MODULE { $<integer>$ = verilog_reader->get_line_no(); }
    { $$ = $<integer>2; }
    ;

module:
    module_begin ID ';' stmts ENDMODULE
    { LOG_FATAL << "module 1"; }
|   module_begin ID '(' ')' ';' stmts ENDMODULE
    { 
        auto* module_stmts = static_cast<std::vector<std::unique_ptr<VerilogStmt>>*>($6); 
        verilog_reader->makeModule(static_cast<const char*>($2), std::move(*module_stmts), static_cast<int>($1));
        delete module_stmts;
    }
|   module_begin ID '(' port_list ')' ';' stmts ENDMODULE
    {         
        auto* port_list = static_cast<std::vector<std::unique_ptr<VerilogID>>*>($4);
        auto* module_stmts = static_cast<std::vector<std::unique_ptr<VerilogStmt>>*>($7); 
        verilog_reader->makeModule(static_cast<const char*>($2), std::move(*port_list), std::move(*module_stmts), static_cast<int>($1));
        
        delete port_list;
        delete module_stmts; 
    }
|   module_begin ID '(' port_dcls ')' ';' stmts ENDMODULE
    { LOG_FATAL << "module 3"; }

|   module_begin ID '#' '(' module_parameters ')' '(' port_dcls ')' ';' stmts ENDMODULE
    {   
     }
    ;

port_list:
    port
    { 
        auto* port_list = new std::vector<std::unique_ptr<VerilogID>>;
        port_list->emplace_back(static_cast<VerilogID*>($1));
        $$ = port_list;
    }
|   port_list ',' port
    {         
        auto* port_list = static_cast<std::vector<std::unique_ptr<VerilogID>>*>($1);
        port_list->emplace_back(static_cast<VerilogID*>($3));
        $$ = port_list; 
    }
    ;

port:
    port_expr
    { $$ = $1; }
|   '.' ID '(' ')'
    { $$=nullptr;}
|   '.' ID '(' port_expr ')'
    { $$=nullptr;}
    ;

port_expr:
    port_ref
    { $$ = $1; }
|   '{' port_refs '}'
    { $$ = nullptr; }   ;

port_refs:
    port_ref
    { $$ = nullptr;
    }
|   port_refs ',' port_ref
    { $$ = $1; }
    ;

port_ref:
    net_scalar
    { $$ = $1; }
|   net_bit_select
|   net_part_select
    ;

port_dcls:
    port_dcl
    { $$ = nullptr;;
    }
|   port_dcls ',' port_dcl
    { $$ = $1;
    }
|   port_dcls ',' dcl_arg
    {
      $$ = $1;
    }
    ;

port_dcl:
    port_dcl_type { $<integer>$ = verilog_reader->get_line_no(); } dcl_arg
    { $$ = nullptr; }
|   port_dcl_type { $<integer>$ = verilog_reader->get_line_no(); }
    '[' parameter_expr ':' parameter_expr ']' dcl_arg
    { $$ = nullptr; }


    ;

port_dcl_type:
    INPUT { $$ = static_cast<int>(VerilogModule::PortDclType::kInput); }
|   INPUT WIRE { $$ = static_cast<int>(VerilogModule::PortDclType::kInputWire); }
|   INOUT { $$ = static_cast<int>(VerilogModule::PortDclType::kInout); }
|   INOUT REG { $$ = static_cast<int>(VerilogModule::PortDclType::kInoutReg); }
|   INOUT WIRE { $$ = static_cast<int>(VerilogModule::PortDclType::kInoutWire); }
|   OUTPUT { $$ = static_cast<int>(VerilogModule::PortDclType::kOutput); }
|   OUTPUT WIRE { $$ = static_cast<int>(VerilogModule::PortDclType::kOputputWire); }
|   OUTPUT REG { $$ = static_cast<int>(VerilogModule::PortDclType::kOutputReg); }
    ;

stmts:
    // empty
    { 
        auto* verilog_stmts = new std::vector<std::unique_ptr<VerilogStmt>>;
        $$ = verilog_stmts;
    }
|   stmts stmt
    { 
        auto* verilog_stmts = static_cast<std::vector<std::unique_ptr<VerilogStmt>>*>($1); 
        if ($2) {
            auto* verilog_stmt = static_cast<VerilogStmt*>($2);
            verilog_stmts->emplace_back(verilog_stmt);
        } 
        $$ = $1; 
    }
|   stmts stmt_seq
    // Append stmt_seq to stmts.
    { 
        auto* verilog_stmts = static_cast<std::vector<std::unique_ptr<VerilogStmt>>*>($1); 
        if ($2) {
            auto* verilog_stmt = static_cast<VerilogStmt*>($2);
            verilog_stmts->emplace_back(verilog_stmt);
        } 
        $$ = $1; 
    }
    ;

stmt:
    parameter
|   defparam
|   declaration
    { $$ = $1; }
|   instance
    { $$ = $1; }
|   error ';'
    { yyerrok; $$ = nullptr; }
    ;

stmt_seq:
    continuous_assign
    { $$ = $1; }
    ;

/* Parameters are parsed and ignored. */
parameter:
    PARAMETER parameter_dcls ';'
    { $$ = nullptr; }
|   PARAMETER '[' INT ':' INT ']' parameter_dcls ';'
    { $$ = nullptr; }
    ;

module_parameters:
    module_parameter
|   module_parameters ',' module_parameter
;

module_parameter:
    PARAMETER parameter_dcl 
    { $$ = nullptr; }
|   PARAMETER '[' INT ':' INT ']' parameter_dcl 
    { $$ = nullptr; }
    ;

parameter_dcls:
    parameter_dcl
    { $$ = nullptr; }
|   parameter_dcls ',' parameter_dcl
    { $$ = nullptr; }
    ;

parameter_dcl:
    ID '=' parameter_expr
    { 
      $$ = nullptr;
    }
|   ID '=' STRING
    { 
      $$ = nullptr;
    }
;

parameter_expr:
    ID
    { 
      $$ = 0;
    }
|   '`' ID
    { 
      $$ = 0;
    }
|   CONSTANT
    { 
      $$ = 0;
    }
|   INT
    {   }
|   '-' parameter_expr %prec NEG
    { $$ = - $2; }
|   parameter_expr '+' parameter_expr
    {  }
|   parameter_expr '-' parameter_expr
    {  }
|   parameter_expr '*' parameter_expr
    {  }
|   parameter_expr '/' parameter_expr
    {  }
|   '(' parameter_expr ')'
    { $$ = $2; }
|  '.' parameter_expr '('parameter_expr ')'
    {}
    ;

defparam:
    DEFPARAM param_values ';'
    { $$ = nullptr; }
    ;

param_values:
    param_value
    { $$ = nullptr; }
|   param_values ',' param_value
    { $$ = nullptr; }
    ;

param_value:
    ID '=' parameter_expr
    { 
      $$ = nullptr;
    }
|   ID '=' STRING
    {
      $$ = nullptr;
    }
    ;

declaration:
    dcl_type { $<integer>$ = verilog_reader->get_line_no(); } dcl_args ';'
    { 
        auto* dcl_args = static_cast<std::vector<const char*>*>($3);
        auto* declaration = verilog_reader->makeDcl(static_cast<VerilogDcl::DclType>($1), std::move(*dcl_args), $<integer>2);
        delete dcl_args;
        $$ = declaration;
    }
|   dcl_type { $<integer>$ = verilog_reader->get_line_no(); }
    '[' INT ':' INT ']' dcl_args ';'
    { 
        auto* dcl_args = static_cast<std::vector<const char*>*>($8);
        std::pair<int, int> range = std::make_pair($4, $6);
        auto* declaration = verilog_reader->makeDcl(static_cast<VerilogDcl::DclType>($1), std::move(*dcl_args), $<integer>2, range);
        delete dcl_args;
        $$ = declaration; 
    }
    ;

dcl_type:
    INPUT { $$ = static_cast<int>(VerilogDcl::DclType::kInput); }
|   INOUT { $$ = static_cast<int>(VerilogDcl::DclType::kInout); }
|   OUTPUT { $$ = static_cast<int>(VerilogDcl::DclType::kOutput); }
|   SUPPLY0 { $$ = static_cast<int>(VerilogDcl::DclType::kSupply0); }
|   SUPPLY1 { $$ = static_cast<int>(VerilogDcl::DclType::kSupply1); }
|   TRI { $$ = static_cast<int>(VerilogDcl::DclType::kTri); }
|   WAND { $$ = static_cast<int>(VerilogDcl::DclType::kWand); }
|   WIRE { $$ = static_cast<int>(VerilogDcl::DclType::kWire); }
|   WOR { $$ = static_cast<int>(VerilogDcl::DclType::kWor); }
    ;

dcl_args:
    dcl_arg
    { 
        auto* dcl_args = new std::vector<const char*>;
        dcl_args->push_back(static_cast<const char*>($1));
        $$ = dcl_args;
    }
|   dcl_args ',' dcl_arg
    { 
        auto* dcl_args = static_cast<std::vector<const char*>*>($1);
        dcl_args->push_back(static_cast<const char*>($3));
        $$ = dcl_args;
    }
    ;

dcl_arg:
    ID
    { $$ = $1; }
|   net_assignment
    { $$ = nullptr; }
    ;

continuous_assign:
    ASSIGN net_assignments ';'
    { $$ = $2; }
    ;

net_assignments:
    net_assignment
    { $$ = $1;
    }
|   net_assignments ',' net_assignment
    { $$ = $1; }
    ;

net_assignment:
    net_assign_lhs { $<integer>$ = verilog_reader->get_line_no(); } '=' net_expr
    {  
        auto* lhs = static_cast<VerilogNetExpr*>($1);
        auto* rhs = static_cast<VerilogNetExpr*>($4);

        auto* module_assign = verilog_reader->makeModuleAssign(lhs, rhs,  $<integer>2); 
        $$ = module_assign; 
        }
    ;

net_assign_lhs:
        net_named
        { $$ = $1; }
        | net_expr_concat
        { $$ = $1; }
        ;

instance:
    ID { $<integer>$ = verilog_reader->get_line_no(); } ID '(' inst_pins ')' ';'
    { 
        std::vector<std::unique_ptr<VerilogPortRefPortConnect>> inst_port_connection;
        if(auto* port_connection = static_cast<std::vector<std::unique_ptr<VerilogPortRefPortConnect>>*>($5);port_connection) {
            inst_port_connection = std::move(*port_connection);
            delete port_connection;        
        }
        
        auto* module_inst = verilog_reader->makeModuleInst(static_cast<const char*>($1), static_cast<const char*>($3), std::move(inst_port_connection), $<integer>2); 
        $$ = module_inst;
    }
    |  ID { $<integer>$ = verilog_reader->get_line_no(); } parameter_values
       ID '(' inst_pins ')' ';'
    { $$ = nullptr; }
    ;


parameter_values:
    '#' '(' parameter_exprs ')'
    ;

parameter_exprs:
    parameter_expr
    { }
|   '{' parameter_exprs '}'
    { $$ = $2; }
|   parameter_exprs ',' parameter_expr
    {}
    ;

inst_pins:
    // empty
    { $$ = nullptr; }
|   inst_ordered_pins
|   inst_named_pins
    { $$ = $1;}
    ;

// Positional pin connections.
inst_ordered_pins:
    net_expr
    { $$ = $1;
    }
|   inst_ordered_pins ',' net_expr
    { $$ = $1; }
    ;

// Named pin connections.
inst_named_pins:
    inst_named_pin
    { 
        auto* port_connect_vec = new std::vector<std::unique_ptr<VerilogPortConnect>>;
        port_connect_vec->emplace_back(static_cast<VerilogPortConnect*>($1));
        $$ = port_connect_vec;
    }
|   inst_named_pins ',' inst_named_pin
    { 
        auto* port_connect_vec = static_cast<std::vector<std::unique_ptr<VerilogPortConnect>>*>($1);
        port_connect_vec->emplace_back(static_cast<VerilogPortConnect*>($3));
        $$ = port_connect_vec;
    }
    ;

// The port reference is split out into cases to special case
// the most frequent case of .port_scalar(net_scalar).
inst_named_pin:
//      Scalar port.
    '.' ID '(' ')'
    { 
        VerilogID* inst_port_id = verilog_reader->makeVerilogID(static_cast<char*>($2));
        $$ = verilog_reader->makePortConnect(inst_port_id, nullptr); 
    }
|   '.' ID '(' ID ')'
    { 
        VerilogID* inst_port_id = verilog_reader->makeVerilogID(static_cast<char*>($2));
        VerilogID* net_id = verilog_reader->makeVerilogID(static_cast<char*>($4));
        auto* net_expr = verilog_reader->makeVerilogNetExpr(net_id, verilog_reader->get_line_no());
        $$ = verilog_reader->makePortConnect(inst_port_id, net_expr); 
    }
|   '.' ID '(' ID '[' INT ']' ')'
    { 
        VerilogID* inst_port_id = new VerilogID(static_cast<char*>($2));
        VerilogID* net_id = verilog_reader->makeVerilogID(static_cast<char*>($4), static_cast<int>($6));
        auto* net_expr = verilog_reader->makeVerilogNetExpr(net_id, verilog_reader->get_line_no());
        $$ = verilog_reader->makePortConnect(inst_port_id, net_expr); 
    }
|   '.' ID '(' named_pin_net_expr ')'
    {   VerilogID* inst_port_id = verilog_reader->makeVerilogID(static_cast<char*>($2));
        $$ = verilog_reader->makePortConnect(inst_port_id, static_cast<VerilogNetExpr*>($4));  
    }    
//      Bus port bit select.
|   '.' ID '[' INT ']' '(' ')'
    { $$ = nullptr; }
|   '.' ID '[' INT ']' '(' net_expr ')'
    { $$ = nullptr; }
//      Bus port part select.
|   '.'  ID '[' INT ':' INT ']' '(' ')'
    { $$ = nullptr; }
|   '.'  ID '[' INT ':' INT ']' '(' net_expr ')'
    { $$ = nullptr; }
    ;

named_pin_net_expr:
    net_part_select
    {
        VerilogID* net_id = static_cast<VerilogID*>($1);
        auto* net_expr = verilog_reader->makeVerilogNetExpr(net_id, verilog_reader->get_line_no());
        $$ = net_expr;
    }
|   net_constant
    { $$ = $1; }
|   net_expr_concat
    { $$ = $1; }
    ;

net_named:
    net_scalar
    { $$ = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>($1), verilog_reader->get_line_no());  }
|   net_bit_select
    { $$ = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>($1), verilog_reader->get_line_no()); }
|   net_part_select
    { $$ = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>($1), verilog_reader->get_line_no()); }
    ;

net_scalar:
    ID
    {  $$ = verilog_reader->makeVerilogID(static_cast<char*>($1));  }
    ;

net_bit_select:
    ID '[' INT ']'
    { $$ = verilog_reader->makeVerilogID(static_cast<char*>($1), static_cast<int>($3)); }
    ;

net_part_select:
    ID '[' INT ':' INT ']'
    { $$ = verilog_reader->makeVerilogID(static_cast<char*>($1), static_cast<int>($3), static_cast<int>($5)); }
    ;

net_constant:
    CONSTANT
    { $$ = verilog_reader->makeVerilogNetExpr(static_cast<const char*>($1), verilog_reader->get_line_no()); }
    ;

net_expr_concat:
    '{' net_exprs '}'
    { 
        auto* verilog_id_concat = static_cast<Vector<std::unique_ptr<VerilogNetExpr>>*>($2);
        auto* verilog_net_expr_concat = verilog_reader->makeVerilogNetExpr(std::move(*verilog_id_concat), verilog_reader->get_line_no()); 
        delete verilog_id_concat;

        $$ = verilog_net_expr_concat;
    }
    ;

net_exprs:
    net_expr
    { 
        auto* verilog_id_concat = new Vector<std::unique_ptr<VerilogNetExpr>>();
        verilog_id_concat->emplace_back(static_cast<VerilogNetExpr*>($1));

        $$ = verilog_id_concat;
    }
|   net_exprs ',' net_expr
    { 
        auto* verilog_id_concat = static_cast<Vector<std::unique_ptr<VerilogNetExpr>>*>($1);
        verilog_id_concat->emplace_back(static_cast<VerilogNetExpr*>($3));

        $$ = verilog_id_concat;
    }
    ;

net_expr:
    net_scalar
    { $$ = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>($1), verilog_reader->get_line_no()); }
|   net_bit_select
    { $$ = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>($1), verilog_reader->get_line_no()); }
|   net_part_select
    { $$ = verilog_reader->makeVerilogNetExpr(static_cast<VerilogID*>($1), verilog_reader->get_line_no()); }
|   net_constant
    { $$ = $1; }
|   net_expr_concat
    { $$ = $1; }
    ;

%%


void verilog_error(yyscan_t scanner,ista::VerilogReader *verilog_reader, const char *str)
{
   char* error_msg = Str::printf("Error found in line %lu in verilog file %s\n", 
   verilog_reader->get_line_no(), verilog_reader->get_file_name().c_str()); 
   LOG_ERROR << error_msg;
}