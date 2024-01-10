#include <string>
#include <vector>

#include "MP.hh"
#include "idm.h"
#include "IDBParserEngine.hh"

int main(int argc, char* argv[])
{
  std::string idb_json = argv[1];
  dmInst->init(idb_json);
  imp::MP mp(new imp::IDBParser(dmInst->get_idb_builder()));
  mp.runMP();
  return 0;
}