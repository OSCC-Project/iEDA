#include "MacroPlacer2.hh"
int main(int argc, char* argv[])
{
  std::string idb_json = argv[1];
  imp::MacroPlacer mp;
  mp.setDataManager(idb_json);
  // iPLAPIInst.initAPI(ipl_json, idb_builder);
  // iPLAPIInst.runMP();
  return 0;
}