%{

#include <ctype.h>

#include "SdfReader.hh"
#include "string/Str.hh"
#include "log/Log.hh"

using namespace ista;
using namespace ieda;

int Sdf_lex();

extern int g_sdf_line;
SdfReader* g_sdf_reader = nullptr;

// use yacc generated parser errors
#define YYERROR_VERBOSE

void yyerror(const char* s);

%}

// expected shift/reduce conflicts
%expect 4

%union {
  char character;
  char *string;
  int integer;
  float number;
  void *obj;
}

%token DELAYFILE SDFVERSION DESIGN DATE VENDOR PROGRAM PVERSION
%token DIVIDER VOLTAGE PROCESS TEMPERATURE TIMESCALE
%token CELL CELLTYPE INSTANCE DELAY ABSOLUTE INCREMENTAL
%token INTERCONNECT PORT DEVICE RETAIN
%token IOPATH TIMINGCHECK
%token SETUP HOLD SETUPHOLD RECOVERY REMOVAL RECREM WIDTH PERIOD SKEW NOCHANGE
%token POSEDGE NEGEDGE COND CONDELSE
%token QSTRING ID PATH NUMBER EXPR_OPEN_IOPATH EXPR_OPEN EXPR_ID_CLOSE

%type <number> NUMBER
%type <obj> number_opt
%type <obj> value triple
%type <obj> delval_list
%type <obj> tchk_def tchk_defs timing_spec 
%type <string> QSTRING ID PATH path 
%type <string> EXPR_OPEN_IOPATH EXPR_OPEN EXPR_ID_CLOSE
%type <obj> port_spec port_tchk port_instance
%type <obj> retains retain
%type <integer> port_transition
%type <character> hierarchical_char

%start file

%{
%}

%%

file:
    '(' DELAYFILE  header cells ')' {}
;

header:
    header_stmt
|   header header_stmt
;

// technically the ordering of these statements is fixed by the spec
header_stmt:
    '(' SDFVERSION QSTRING ')' { Str::free((char*)$3); }
|   '(' DESIGN QSTRING ')' { Str::free((char*)$3); }
|   '(' DATE QSTRING ')' { Str::free((char*)$3); }
|   '(' VENDOR QSTRING ')' { Str::free((char*)$3); }
|   '(' PROGRAM QSTRING ')' { Str::free((char*)$3); }
|   '(' PVERSION QSTRING ')' { Str::free((char*)$3); }
|   '(' DIVIDER hierarchical_char ')' {  }
|   '(' VOLTAGE triple ')' {  }
|   '(' VOLTAGE NUMBER ')'
|   '(' VOLTAGE ')'  // Illegal SDF (from OC).
|   '(' PROCESS QSTRING ')' { Str::free($3); }
|   '(' PROCESS ')'  // Illegal SDF (from OC).
|   '(' TEMPERATURE NUMBER ')'
|   '(' TEMPERATURE triple ')' {  }
|   '(' TEMPERATURE ')' // Illegal SDF (from OC).
|   '(' TIMESCALE NUMBER ID ')'
    {  }
;

hierarchical_char:
    '/'
    { $$ = '/'; }
|   '.'
    { $$ = '.'; }
;

number_opt: { $$ = nullptr; }
|   NUMBER   { $$ = new float($1); }
;

cells:
    cell
|   cells cell
;

cell:
    '(' CELL celltype cell_instance timing_specs ')'
    {  }
;

celltype:
    '(' CELLTYPE QSTRING ')'
    {  }
;

cell_instance:
    '(' INSTANCE ')'   {  }
|   '(' INSTANCE '*' ')'   {  }
|   '(' INSTANCE path ')'
    {  }
;

timing_specs:
    /* empty */
|   timing_specs timing_spec
;

timing_spec:
    '(' DELAY deltypes ')'
     { $$ = nullptr; }
|   '(' TIMINGCHECK tchk_defs ')'
    {
        auto* tchk_defs = static_cast<std::vector<std::unique_ptr<SdfTimingCheckDef>>*>($3);
        auto* timing_spec = g_sdf_reader->makeTimingSpec(std::move(*tchk_defs));

        delete tchk_defs;

        $$ = timing_spec;
    }
;

deltypes:
|       deltypes deltype
;

deltype:
    '(' ABSOLUTE
    {  }
    del_defs ')'
|   '(' INCREMENTAL
    {  }
    del_defs ')'
;

del_defs:
|   del_defs del_def
;

path:
    ID
|   PATH
;

del_def:
    '(' IOPATH port_spec port_instance retains delval_list ')'
    {  }
|   '(' CONDELSE '(' IOPATH port_spec port_instance
            retains delval_list ')' ')'
    {  }
|   '(' COND EXPR_OPEN_IOPATH port_spec port_instance
            retains delval_list ')' ')'
    {  }
|   '(' INTERCONNECT port_instance port_instance delval_list ')'
    {  }
|   '(' PORT port_instance delval_list ')'
    {  }
|   '(' DEVICE delval_list ')'
    {  }
|   '(' DEVICE port_instance delval_list ')'
    {  }
;

retains:
    /* empty */
    { $$ = nullptr; }
|   retains retain
;

retain:
    '(' RETAIN delval_list ')'
    {  }
;

delval_list:
    value
    { 
        auto* delay_values = new std::vector<std::unique_ptr<SdfTripleValue>>();
        auto* delay_value = static_cast<SdfTripleValue*>($1);
        delay_values->emplace_back(delay_value);
        $$ = delay_values; 
    }
|   delval_list value
    {   
        auto* delay_values = static_cast<std::vector<std::unique_ptr<SdfTripleValue>>*>($1);
        auto* delay_value = static_cast<SdfTripleValue*>($2);
        delay_values->emplace_back(delay_value);
        $$ = delay_values; 
    }
;

/*timing check defines*/
tchk_defs:
    { $$ = nullptr; }
|   tchk_defs tchk_def
    { 
        auto* timing_check_defs = static_cast<std::vector<std::unique_ptr<SdfTimingCheckDef>>*>($1);
        if (!timing_check_defs) {
            timing_check_defs = new std::vector<std::unique_ptr<SdfTimingCheckDef>>();
        }

        auto* timing_check_def = static_cast<SdfTimingCheckDef*>($2);
        timing_check_defs->emplace_back(timing_check_def);

        $$ = timing_check_defs;
    }
;

tchk_def:
    '(' SETUP { g_sdf_reader->set_parse_timing_check(true); }
    port_tchk port_tchk value ')'
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>($4);
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>($5);
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>($6);
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kSetup, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      $$ = timing_check_def;
      
    }
|   '(' HOLD { g_sdf_reader->set_parse_timing_check(true); }
    port_tchk port_tchk value ')'
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>($4);
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>($5);
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>($6);
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kHold, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      $$ = timing_check_def;
      
    }
|   '(' SETUPHOLD { g_sdf_reader->set_parse_timing_check(true); }
    port_tchk port_tchk value value ')'
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>($4);
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>($5);
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>($6);
      SdfTripleValue* value1 = static_cast<SdfTripleValue*>($7);
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kSetupHold, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::move(*value1));
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      delete value1;
      $$ = timing_check_def;
      
    }
|   '(' RECOVERY { g_sdf_reader->set_parse_timing_check(true); }
    port_tchk port_tchk value ')'
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>($4);
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>($5);
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>($6);
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kRecovery, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      $$ = timing_check_def;
      
    }
|   '(' REMOVAL { g_sdf_reader->set_parse_timing_check(true); }
    port_tchk port_tchk value ')'
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>($4);
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>($5);
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>($6);
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kRemoval, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      $$ = timing_check_def;
      
    }
|   '(' RECREM { g_sdf_reader->set_parse_timing_check(true); }
    port_tchk port_tchk value value ')'
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>($4);
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>($5);
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>($6);
      SdfTripleValue* value1 = static_cast<SdfTripleValue*>($7);
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kRecRem, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::move(*value1));
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      delete value1;
      $$ = timing_check_def;
      
    }
|   '(' SKEW { g_sdf_reader->set_parse_timing_check(true); }
    port_tchk port_tchk value ')'
    // Sdf skew clk/ref are reversed from liberty.
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>($4);
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>($5);
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>($6);
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kSkew, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      $$ = timing_check_def;
      
    }
|   '(' WIDTH { g_sdf_reader->set_parse_timing_check(true); }
    port_tchk value ')'
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>($4);
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>($4);
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>($5);
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kWidth, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      $$ = timing_check_def;
      
    }
|   '(' PERIOD { g_sdf_reader->set_parse_timing_check(true); }
    port_tchk value ')'   
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>($4);
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>($4);
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>($5);
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kPeriod, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      $$ = timing_check_def;
      
    }
|   '(' NOCHANGE { g_sdf_reader->set_parse_timing_check(true); }
    port_tchk port_tchk value value ')'
    { 
      SdfPortSpec* snk_port_tchk = static_cast<SdfPortSpec*>($4);
      SdfPortSpec* src_port_tchk = static_cast<SdfPortSpec*>($5);
      SdfTripleValue* value0 = static_cast<SdfTripleValue*>($6);
      auto* timing_check_def = g_sdf_reader->makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType::kNoChange, std::move(*snk_port_tchk), std::move(*src_port_tchk), 
      std::move(*value0), std::nullopt);
      delete snk_port_tchk;
      delete src_port_tchk;
      delete value0;
      $$ = timing_check_def;
    }
;

port_instance:
    ID
    { 
        $$ = g_sdf_reader->makePortInstance(static_cast<const char*>($1));
    }
|   PATH
    { $$ = g_sdf_reader->makePortInstance(static_cast<const char*>($1));}
;

port_spec:
    ID
    { $$ = g_sdf_reader->makePortSpec(static_cast<const char*>($1)); }
|   '(' port_transition ID ')'
    { $$ = g_sdf_reader->makePortSpec(static_cast<SdfPortSpec::TransitionType>($2), static_cast<const char*>($3)); }
;

port_transition:
    POSEDGE   { $$ = static_cast<int>(SdfPortSpec::TransitionType::kPOSEDGE); }
|   NEGEDGE   { $$ = static_cast<int>(SdfPortSpec::TransitionType::kNEGEDGE); }
;

/*port timing check*/
port_tchk:
    port_spec
    { $$ = $1; }    
|   '(' COND EXPR_ID_CLOSE 
    { 
        /* (COND expr port) */
        $$ = nullptr; 
    }
|   '(' COND EXPR_OPEN port_transition ID ')' ')' 
    { 
        /* (COND expr (posedge port)) */
        $$ = nullptr; 
    }
;

value:
    '(' ')'
    {
      std::array<std::optional<float>, 3> triple_value;
      $$ = g_sdf_reader->makeTriple(triple_value);
    }
|   '(' NUMBER ')'
    {
      float n = (float)$2;
      std::array<std::optional<float>, 3> triple_value = {n, n, n};
      $$ = g_sdf_reader->makeTriple(triple_value);
    }
|   '(' triple ')' { $$ = $2; }
;

triple:
    NUMBER ':' number_opt ':' number_opt
    {      
      float n1 = (float)$1;
      std::optional<float> n2 = $3 ? std::optional<float>(*((float*)$3)) : std::nullopt; 
      std::optional<float> n3 = $5 ? std::optional<float>(*((float*)$5)) : std::nullopt; 

      delete (float*)$3;
      delete (float*)$5;

      std::array<std::optional<float>, 3> triple_value = {n1, n2, n3};
      $$ = g_sdf_reader->makeTriple(triple_value);
    }
|  number_opt ':' NUMBER ':' number_opt
    {
      std::optional<float> n1 = $1 ? std::optional<float>(*((float*)$1)) : std::nullopt; 
      float n2 = (float)$3;      
      std::optional<float> n3 = $5 ? std::optional<float>(*((float*)$5)) : std::nullopt; 

      delete (float*)$1;
      delete (float*)$5;

      std::array<std::optional<float>, 3> triple_value = {n1, n2, n3};
      $$ = g_sdf_reader->makeTriple(triple_value);
    }
|  number_opt ':' number_opt ':' NUMBER
    {      
      std::optional<float> n1 = $1 ? std::optional<float>(*((float*)$1)) : std::nullopt; 
      std::optional<float> n2 = $3 ? std::optional<float>(*((float*)$3)) : std::nullopt; 
      float n3 = (float)$5;

      delete (float*)$1;
      delete (float*)$3;

      std::array<std::optional<float>, 3> triple_value = {n1, n2, n3};
      $$ = g_sdf_reader->makeTriple(triple_value);
    }
;

%%

// Global namespace

void sdfFlushBuffer();

void yyerror(const char *msg) {

  sdfFlushBuffer();

  LOG_ERROR << "error line " << g_sdf_line << " " << msg;

}