// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************

#include <string>
#include "PlacerDB.hh"
#include "PLAPI.hh"
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
