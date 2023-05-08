//
// Created by xiebiwei on 2019/12/18.
//

#include <iostream>

#include "NetlistReader.h"
#include "module_test.h"

using namespace std;

int main(int argc, char** argv) {
  if (argc != 1) {
    cout << argv[0] << " " << argv[1] << endl;
  }
  //    Utility::testGrammar();

  //test case
  string filename_str = "/home/huangzengrong/Project/data/asic_top.v";
  //    string filename_str = "../test/beihai/asic_top.syn.0822_final.vg";
  //    string filename_str = "../test/NOOPSoC.rtlnopwr.v";
  //    string filename_str = "../test/map9v3.rtlnopwr.v";
//   string topModuleName = "soc_asic_top";
  //    string topModuleName = "IFU_0";
  //    string topModuleName = "asic_top";
  //    string topModuleName = "soc_top_0";
  //    string filename_str = "../test/map9v3.rtlnopwr.v";

  string topModuleName = "\\$paramod$7b55cc7d09a4092e982886ac2a7c332114632b02\\spi_flash";
  if (!ebase::gNetlistReader)
    ebase::gNetlistReader = new ebase::NetlistReader(filename_str);
  ebase::gNetlistReader->read();

  testMakeSingleModule(ebase::gNetlistReader, topModuleName);

  ebase::gNetlistReader->printNetlist();
  return 0;
}
