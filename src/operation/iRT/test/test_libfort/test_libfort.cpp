#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "fort.hpp"

int main()
{
  fort::char_table table;
  table << fort::header << "Access Type"
        << "Pin Number" << fort::endr;
  table << "layer 1"
        << "1" << fort::endr;
  table << ""
        << "2" << fort::endr;
  table[3][0].set_cell_text_align(fort::text_align::center);
  table[3][0].set_cell_span(2);
  table << fort::header << "Total"
        << "3" << fort::endr;

  std::cout << table.to_string() << std::endl;
  return 0;
}