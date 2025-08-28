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
#pragma once
#include <string>
namespace python_interface {

bool placerAutoRun(const std::string& config);
bool placerRunFiller(const std::string& config);
bool placerIncrementalFlow(const std::string& config);
bool placerIncrementalLG();
bool placerCheckLegality();
bool placerReport();
bool placerAiRun(const std::string& config, const std::string& onnx_path, const std::string& normalization_path);

bool placerInit(const std::string& config);
bool placerDestroy();
bool placerRunMP();
bool placerRunGP();
bool placerRunLG();
bool placerRunDP();

}  // namespace python_interface