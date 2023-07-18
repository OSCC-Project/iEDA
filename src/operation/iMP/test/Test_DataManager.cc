#include <vector>

#include "MP.hh"
int main(int argc, char* argv[])
{
  std::string idb_json = argv[1];
  imp::MacroPlacer mp;
  mp.setDataManager(idb_json);
  return 0;
}