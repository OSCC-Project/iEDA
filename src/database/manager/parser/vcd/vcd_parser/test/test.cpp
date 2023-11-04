#include <cstring>
#include <iostream>

#include "VcdParserRustC.hh"

int main()
{
  ipower::RustVcdReader vcd_reader;

  vcd_reader.readVcdFile("/home/shaozheqing/iEDA/src/database/manager/parser/vcd/vcd_parser/benchmark/test1.vcd");

  vcd_reader.buildAnnotateDB("top_i");
  vcd_reader.calcScopeToggleAndSp("top_i");
  vcd_reader.printAnnotateDB(std::cout);
  return 1;
}