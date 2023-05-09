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
 * @file SdfReader.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-07-28
 */

#include "SdfReader.hh"

#include "string/Str.hh"

extern int SdfParse_parse();

namespace ista {

SdfTripleValue::SdfTripleValue(std::array<std::optional<float>, 3> triple_value) : _triple_value(triple_value)
{
}

SdfPortSpec::SdfPortSpec(const char* port_name) : _port_name(port_name)
{
}

SdfPortSpec::~SdfPortSpec()
{
  Str::free(_port_name);
  Str::free(_sdf_condition);
}

SdfPortInstance::SdfPortInstance(const char* port_instance_name) : _port_instance_name(port_instance_name)
{
}

SdfPortInstance::~SdfPortInstance()
{
  Str::free(_port_instance_name);
  _port_instance_name = nullptr;
}

SdfTimingCheckDef::SdfTimingCheckDef(SdfCheckType sdf_check_type, SdfPortSpec&& snk_port_tchk, SdfPortSpec&& src_port_tchk,
                                     SdfTripleValue value0, std::optional<SdfTripleValue> value1)
    : _sdf_check_type(sdf_check_type),
      _snk_port_tchk(snk_port_tchk),
      _src_port_tchk(src_port_tchk),
      _value0(std::move(value0)),
      _value1(std::move(value1))
{
}

SdfIOPathDelayTimingSpec::SdfIOPathDelayTimingSpec(SdfPortSpec* src_port_spec, SdfPortInstance* snk_port_instance)
    : _src_port_spec(src_port_spec), _snk_port_instance(snk_port_instance)
{
}

SdfIOPathDelayTimingSpec::~SdfIOPathDelayTimingSpec()
{
  Str::free(_cond);
  _cond = nullptr;
}

SdfCheckTimingSpec::SdfCheckTimingSpec(std::vector<std::unique_ptr<SdfTimingCheckDef>>&& tchk_defs) : _tchk_defs(std::move(tchk_defs))
{
}

SdfReader::SdfReader(const char* file_name) : _file_name(file_name), _parse_timing_check(0)
{
}

/**
 * @brief Make the triple value.
 *
 * @param triple_value
 * @return SdfTripleValue*
 */
SdfTripleValue* SdfReader::makeTriple(std::array<std::optional<float>, 3> triple_value)
{
  auto* sdf_triple_value = new SdfTripleValue(triple_value);
  return sdf_triple_value;
}

/**
 * @brief Make the port spec
 *
 * @param port_name
 * @return SdfPortSpec*
 */
SdfPortSpec* SdfReader::makePortSpec(const char* port_name)
{
  auto* sdf_port_spec = new SdfPortSpec(port_name);
  return sdf_port_spec;
}

/**
 * @brief Make the port spec include posedge or negedge.
 *
 */
SdfPortSpec* SdfReader::makePortSpec(SdfPortSpec::TransitionType transition_type, const char* port_name)
{
  auto* sdf_port_spec = new SdfPortSpec(port_name);
  sdf_port_spec->set_transition_type(transition_type);
  return sdf_port_spec;
}

SdfPortInstance* SdfReader::makePortInstance(const char* port_instance_name)
{
  auto* port_instance = new SdfPortInstance(port_instance_name);
  return port_instance;
}

/**
 * @brief Make timing check define, timing check is setup/hold, recovery/removal
 * check.
 *
 * @param sdf_check_type
 * @param snk_port_tchk
 * @param src_port_tchk
 * @param value0
 * @param value1 setup/hold include value1.
 * @return SdfTimingCheckDef*
 */
SdfTimingCheckDef* SdfReader::makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType sdf_check_type, SdfPortSpec snk_port_tchk,
                                                 SdfPortSpec src_port_tchk, SdfTripleValue value0, std::optional<SdfTripleValue> value1)
{
  auto* sdf_timing_check_def
      = new SdfTimingCheckDef(sdf_check_type, std::move(snk_port_tchk), std::move(src_port_tchk), std::move(value0), std::move(value1));
  return sdf_timing_check_def;
}

/**
 * @brief Make timing spec, timing spec include delay spec and check spec.
 *
 * @param tchk_defs
 * @return SdfTimingSpec*
 */
SdfTimingSpec* SdfReader::makeTimingSpec(std::vector<std::unique_ptr<SdfTimingCheckDef>>&& tchk_defs)
{
  auto* timing_spec = new SdfCheckTimingSpec(std::move(tchk_defs));
  return timing_spec;
}

}  // namespace ista
