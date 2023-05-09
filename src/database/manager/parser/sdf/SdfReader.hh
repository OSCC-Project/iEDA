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
 * @file SdfReader.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-07-28
 */

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "DisallowCopyAssign.hh"
#include "string/Str.hh"

using ieda::Str;

namespace ista {

/**
 * @brief The triple value in sdf file such as : (.5:.6:.7), which means
 * fast:typical:slow.
 *
 */
class SdfTripleValue
{
 public:
  explicit SdfTripleValue(std::array<std::optional<float>, 3> triple_value);
  ~SdfTripleValue() = default;

  SdfTripleValue(SdfTripleValue&& other) = default;
  SdfTripleValue& operator=(SdfTripleValue&& other) = default;

 private:
  std::array<std::optional<float>, 3> _triple_value;  //!< The triple value fast:typical:slow.

  DISALLOW_COPY_AND_ASSIGN(SdfTripleValue);
};

/**
 * @brief Specify the port name, rise/fall, condition.
 *
 */
class SdfPortSpec
{
 public:
  enum class TransitionType : int
  {
    kPOSEDGE = 0,
    kNEGEDGE = 1,
    kBoth = 2
  };
  explicit SdfPortSpec(const char* port_name);
  ~SdfPortSpec();

  void set_transition_type(TransitionType transition_type) { _transition_type = transition_type; }

  TransitionType get_transition_type() { return _transition_type; }

 private:
  const char* _port_name;
  TransitionType _transition_type = TransitionType::kBoth;
  const char* _sdf_condition = nullptr;
};

/**
 * @brief Specify the port instance name for interconnect net.
 *
 */
class SdfPortInstance
{
 public:
  explicit SdfPortInstance(const char* port_instance_name);
  ~SdfPortInstance();

 private:
  const char* _port_instance_name;
};

/**
 * @brief The sdf timing check include setup/hold,recovery/removal,
 * max skew,  min width, min period, no change.
 *
 */
class SdfTimingCheckDef
{
 public:
  enum class SdfCheckType
  {
    kSetup = 0,
    kHold = 1,
    kSetupHold = 2,
    kRecovery = 3,
    kRemoval = 4,
    kRecRem = 5,
    kSkew = 6,
    kWidth = 7,
    kPeriod = 8,
    kNoChange = 9
  };

  SdfTimingCheckDef(SdfCheckType sdf_check_type, SdfPortSpec&& snk_port_tchk, SdfPortSpec&& src_port_tchk, SdfTripleValue value0,
                    std::optional<SdfTripleValue> value1);
  ~SdfTimingCheckDef() = default;

 private:
  SdfCheckType _sdf_check_type;
  SdfPortSpec _snk_port_tchk;
  SdfPortSpec _src_port_tchk;

  SdfTripleValue _value0;
  std::optional<SdfTripleValue> _value1;  //!< if SetupHold, RecoveryRemoval, then should two value.
};

/**
 * @brief The sdf timing spec may be delay timing spec, or check timing spec.
 *
 */
class SdfTimingSpec
{
 public:
  SdfTimingSpec() = default;
  virtual ~SdfTimingSpec() = default;
  virtual unsigned isDelayTimingSpec() { return 0; }
  virtual unsigned isCheckTimingSpec() { return 0; }
};

/**
 * @brief The sdf delay timing spec, which include tco etc.
 *
 */
class SdfDelayTimingSpec : public SdfTimingSpec
{
 public:
  unsigned isDelayTimingSpec() override { return 1; }
};

/**
 * @brief The io path delay timing spec, which means tco timing arc.
 *
 */
class SdfIOPathDelayTimingSpec : SdfDelayTimingSpec
{
 public:
  SdfIOPathDelayTimingSpec(SdfPortSpec* src_port_spec, SdfPortInstance* snk_port_instance);
  ~SdfIOPathDelayTimingSpec() override;

 private:
  std::unique_ptr<SdfPortSpec> _src_port_spec;
  std::unique_ptr<SdfPortInstance> _snk_port_instance;
  const char* _cond = nullptr;
};

/**
 * @brief The sdf check timing spec, which include setup/hold, recovery/removal,
 * min width, min period etc.
 *
 */
class SdfCheckTimingSpec : public SdfTimingSpec
{
 public:
  explicit SdfCheckTimingSpec(std::vector<std::unique_ptr<SdfTimingCheckDef>>&& tchk_defs);
  ~SdfCheckTimingSpec() override = default;
  unsigned isCheckTimingSpec() override { return 1; }

 private:
  std::vector<std::unique_ptr<SdfTimingCheckDef>> _tchk_defs;
};

/**
 * @brief The sdf reader for read the sdf file.
 *
 */
class SdfReader
{
 public:
  explicit SdfReader(const char* file_name);
  ~SdfReader() = default;

  [[nodiscard]] unsigned isParseTimingCheck() const { return _parse_timing_check; }
  void set_parse_timing_check(bool is_parse_timing_check) { _parse_timing_check = is_parse_timing_check ? 1 : 0; }

  bool read();

  SdfTripleValue* makeTriple(std::array<std::optional<float>, 3> triple_value);
  SdfPortSpec* makePortSpec(const char* port_name);
  SdfPortSpec* makePortSpec(SdfPortSpec::TransitionType transition_type, const char* port_name);
  SdfPortInstance* makePortInstance(const char* port_instance_name);
  SdfTimingCheckDef* makeTimingCheckDef(SdfTimingCheckDef::SdfCheckType sdf_check_type, SdfPortSpec snk_port_tchk,
                                        SdfPortSpec src_port_tchk, SdfTripleValue value0, std::optional<SdfTripleValue> value1);
  SdfTimingSpec* makeTimingSpec(SdfPortSpec* src_port_spec, SdfPortInstance* snk_port_instance);
  SdfTimingSpec* makeTimingSpec(std::vector<std::unique_ptr<SdfTimingCheckDef>>&& tchk_defs);

 private:
  const char* _file_name;
  double _timescale = 1.0;

  unsigned _parse_timing_check : 1;
  unsigned _reserverd : 31;

  DISALLOW_COPY_AND_ASSIGN(SdfReader);
};

}  // namespace ista
