
#include <string>
#include "PlacerDB.hh"
#include "iPL_API.hh"
#include "idm.h"

using namespace ipl;
using namespace imp;

int main(int argc, char *argv[]) {
    std::string idb_json = argv[1];
    std::string ipl_json = argv[2];
    dmInst->init(idb_json);

    auto* idb_builder = dmInst->get_idb_builder();

    iPLAPIInst.initAPI(ipl_json, idb_builder);
    iPLAPIInst.runMP();
    return 1;
}
