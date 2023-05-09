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
 * @file sdcCommand.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The file is the implemention of
 * @version 0.1
 * @date 2020-11-22
 */

#include "SdcCommand.hh"
#include "tcl/ScriptEngine.hh"

namespace ista {

using ieda::ScriptEngine;

SdcCommandObj::SdcCommandObj() {
  auto* script_engine = ScriptEngine::getOrCreateInstance();
  _file_name = Str::copy(script_engine->getTclFileName());
  _line_no = script_engine->getTclLineNo();
}

SdcCommandObj::~SdcCommandObj() { Str::free(_file_name); }

SdcIOConstrain::SdcIOConstrain(const char* constrain_name)
    : _constrain_name(Str::copy(constrain_name)) {}

SdcIOConstrain::~SdcIOConstrain() { Str::free(_constrain_name); }

}  // namespace ista
