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
/**
 * @file iPNPApi.hh
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#include "iPNPApi.hh"

#include "iPNP.hh"
#include "log/Log.hh"

namespace ipnp {

iPNP* iPNPApi::_ipnp_instance = nullptr;

void iPNPApi::setInstance(iPNP* ipnp) {
    _ipnp_instance = ipnp;
}

iPNP* iPNPApi::getInstance() {
    return _ipnp_instance;
}

void iPNPApi::run_pnp(idb::IdbBuilder* idb_builder) {
    if (!_ipnp_instance) {
        LOG_ERROR << "iPNP instance is not set. Please call setInstance() first." << std::endl;
        return;
    }
    
    if (!idb_builder) {
        LOG_ERROR << "Input idb_builder is null." << std::endl;
        return;
    }
    
    auto* idb_design = idb_builder->get_def_service()->get_design();
    if (!idb_design) {
        LOG_ERROR << "Failed to get IDB design from builder." << std::endl;
        return;
    }
    
    _ipnp_instance->setIdb(idb_design);
    _ipnp_instance->setIdbBuilder(idb_builder);

    // run
    _ipnp_instance->init();
    _ipnp_instance->runSynthesis();
    _ipnp_instance->saveToIdb();

}

} // namespace ipnp 