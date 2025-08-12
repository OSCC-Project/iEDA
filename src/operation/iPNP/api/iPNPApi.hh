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

#ifndef IPNP_API_HH
#define IPNP_API_HH

#pragma once
#include <iostream>
#include <string>
#include <filesystem>

#include "log/Log.hh"

namespace ipnp {

#define iPNPApiInst ipnp::iPNPApi::getInstance()

class iPNP;

class iPNPApi
{
public:
    static void setInstance(iPNP* ipnp);
    static iPNP* getInstance();

private:
    static iPNP* _ipnp_instance;
};

}  // namespace ipnp

#endif // IPNP_API_HH