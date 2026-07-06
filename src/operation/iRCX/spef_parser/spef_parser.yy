%code requires {
#include <cstdlib>

#include "SpefParser.hh"
}

%code provides {
int spef_lex(void);
void spef_error(spef::ParserContext* context, const char* message);
}

%union {
  char* str;
  int ival;
}

%define api.prefix {spef_}
%parse-param { spef::ParserContext* context }

%token EOL
%token K_NAME_MAP K_PORTS K_CONN K_CAP K_RES K_END K_D_NET
%token K_COORD K_LOAD K_DRIVE K_LL K_UR K_LAYER K_IGNORE_ATTR
%token <str> HEADER_KEY CONN_TYPE DIRECTION NAME_REF SPEF_NAME QUOTED_STRING NUMBER

%type <str> name_token name_value_token
%type <ival> direction_opt direction conn_type

%destructor { std::free($$); } <str>

%start spef_file

%%

spef_file:
    lines
  ;

lines:
    /* empty */
  | lines line
  ;

line:
    EOL
  | section_line
  | header_line
  | name_map_entry
  | port_entry
  | dnet_entry
  | conn_entry
  | cap_res_entry
  ;

section_line:
    K_NAME_MAP { context->setSection(spef::SectionType::kNameMap); }
  | K_PORTS { context->setSection(spef::SectionType::kPorts); }
  | K_CONN { context->setSection(spef::SectionType::kConn); }
  | K_CAP { context->setSection(spef::SectionType::kCap); }
  | K_RES { context->setSection(spef::SectionType::kRes); }
  | K_END { context->setSection(spef::SectionType::kEnd); }
  ;

header_line:
    HEADER_KEY
    {
      context->startHeader(spef::tokenToString($1));
      std::free($1);
    }
    header_values
    EOL
    {
      context->finishHeader();
    }
  ;

header_values:
    header_value
  | header_values header_value
  ;

header_value:
    name_value_token
    {
      context->addHeaderValue(spef::stripQuotes(spef::tokenToString($1)));
      std::free($1);
    }
  ;

name_map_entry:
    NAME_REF name_value_token EOL
    {
      if (context->section() == spef::SectionType::kNameMap) {
        context->addNameMap(spef::tokenToString($1), spef::stripQuotes(spef::tokenToString($2)));
      }
      std::free($1);
      std::free($2);
    }
  ;

port_entry:
    name_token direction EOL
    {
      if (context->section() == spef::SectionType::kPorts) {
        context->addPort(spef::tokenToString($1), static_cast<spef::ConnectionDirection>($2),
                         spef::Coord{});
      }
      std::free($1);
    }
  | name_token direction K_COORD NUMBER NUMBER EOL
    {
      if (context->section() == spef::SectionType::kPorts) {
        context->addPort(spef::tokenToString($1), static_cast<spef::ConnectionDirection>($2),
                         spef::Coord{spef::toDouble($4), spef::toDouble($5)});
      }
      std::free($1);
      std::free($4);
      std::free($5);
    }
  ;

dnet_entry:
    K_D_NET name_token NUMBER EOL
    {
      context->startNet(spef::tokenToString($2), spef::toDouble($3), 0);
      std::free($2);
      std::free($3);
    }
  ;

conn_entry:
    conn_start conn_attrs EOL
    {
      context->finishConn();
    }
  ;

conn_start:
    conn_type name_token direction_opt
    {
      context->startConn(static_cast<spef::ConnectionType>($1), spef::tokenToString($2),
                         static_cast<spef::ConnectionDirection>($3));
      std::free($2);
    }
  ;

conn_attrs:
    /* empty */
  | conn_attrs conn_attr
  ;

conn_attr:
    K_COORD NUMBER NUMBER
    {
      context->setConnCoordinate(spef::Coord{spef::toDouble($2), spef::toDouble($3)});
      std::free($2);
      std::free($3);
    }
  | K_LOAD NUMBER
    {
      context->setConnLoad(spef::toDouble($2));
      std::free($2);
    }
  | K_DRIVE name_token
    {
      context->setConnDrivingCell(spef::stripQuotes(spef::tokenToString($2)));
      std::free($2);
    }
  | K_LL NUMBER NUMBER
    {
      context->setConnLowerLeft(spef::Coord{spef::toDouble($2), spef::toDouble($3)});
      std::free($2);
      std::free($3);
    }
  | K_UR NUMBER NUMBER
    {
      context->setConnUpperRight(spef::Coord{spef::toDouble($2), spef::toDouble($3)});
      std::free($2);
      std::free($3);
    }
  | K_LAYER NUMBER
    {
      context->setConnLayer(spef::toInt($2));
      std::free($2);
    }
  | K_IGNORE_ATTR
  ;

cap_res_entry:
    NUMBER name_token NUMBER EOL
    {
      if (context->section() == spef::SectionType::kCap) {
        context->addCap(spef::tokenToString($2), "", spef::toDouble($3));
      }
      std::free($1);
      std::free($2);
      std::free($3);
    }
  | NUMBER name_token name_token NUMBER EOL
    {
      context->addCapOrRes(spef::tokenToString($2), spef::tokenToString($3), spef::toDouble($4));
      std::free($1);
      std::free($2);
      std::free($3);
      std::free($4);
    }
  ;

direction_opt:
    /* empty */ { $$ = static_cast<int>(spef::ConnectionDirection::kUninitialized); }
  | direction { $$ = $1; }
  ;

direction:
    DIRECTION
    {
      $$ = static_cast<int>(spef::parseDirection($1));
      std::free($1);
    }
  ;

conn_type:
    CONN_TYPE
    {
      $$ = static_cast<int>(spef::parseConnectionType($1));
      std::free($1);
    }
  ;

name_token:
    NAME_REF { $$ = $1; }
  | SPEF_NAME { $$ = $1; }
  | QUOTED_STRING { $$ = $1; }
  | DIRECTION { $$ = $1; }
  ;

name_value_token:
    NAME_REF { $$ = $1; }
  | SPEF_NAME { $$ = $1; }
  | QUOTED_STRING { $$ = $1; }
  | NUMBER { $$ = $1; }
  | DIRECTION { $$ = $1; }
  ;

%%

void spef_error(spef::ParserContext* context, const char* message)
{
  if (context != nullptr) {
    context->setError(message == nullptr ? "parse error" : message);
  }
}
