/**
 * @file sdcCommand.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The file is the implemention of
 * @version 0.1
 * @date 2020-11-22
 *
 * @copyright Copyright (c) 2020
 *
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
