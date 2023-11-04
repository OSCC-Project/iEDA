#include <cstring>
#include <iostream>

#include "VcdParserRustC.hh"

int main()
{
  ipower::RustVcdReader vcd_reader;

  vcd_reader.readVcdFile("/home/shaozheqing/iEDA/src/database/manager/parser/vcd/vcd_parser/benchmark/test1.vcd");

  return 1;
}