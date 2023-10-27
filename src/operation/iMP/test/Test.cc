#include <vector>

#include "MP.hh"
// #include "PyPlot.hh"
int main(int argc, char* argv[])
{
  std::string idb_json = argv[1];
  imp::MacroPlacer mp;
  mp.setDataManager(idb_json);
  mp.runMP();
  return 0;
}