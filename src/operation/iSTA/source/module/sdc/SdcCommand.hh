// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file sdcCommand.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The file is the base command obj of sdc.
 * @version 0.1
 * @date 2020-11-22
 *
 */

#pragma once

#include "Config.hh"
#include "Type.hh"
#include "string/Str.hh"

namespace ista {
/**
 * @brief The sdc command base class.
 *
 */

class SdcCommandObj {
 public:
  SdcCommandObj();
  virtual ~SdcCommandObj();
  const char* get_file_name() { return _file_name; }
  [[nodiscard]] unsigned get_line_no() const { return _line_no; }
  void set_file_name(const char* file_name) { _file_name = file_name; }
  void set_line_no(unsigned line_no) { _line_no = line_no; }

  virtual unsigned isIOConstrain() { return 0; }
  virtual unsigned isTimingDRC() { return 0; }
  virtual unsigned isSdcCollection() { return 0; }
  virtual unsigned isAllClock() { return 0; }
  virtual unsigned isAllInputPorts() { return 0; }
  virtual unsigned isAllOutputPorts() { return 0; }
  virtual unsigned isClock() { return 0; }

 private:
  const char* _file_name = nullptr;  //!< the sdc file name.
  unsigned _line_no = 0;             //!< the sdc file line no.

  FORBIDDEN_COPY(SdcCommandObj);
};

/**
 * @brief The IO constrain, include set_input_delay,
 * set_output_delay,set_input_transition.
 *
 */
class SdcIOConstrain : public SdcCommandObj {
 public:
  explicit SdcIOConstrain(const char* constrain_name);
  ~SdcIOConstrain() override;

  const char* get_constrain_name() { return _constrain_name; }
  unsigned isIOConstrain() override { return 1; }
  unsigned isTimingDRC() override { return 0; }
  virtual unsigned isSetInputTransition() { return 0; }
  virtual unsigned isSetLoad() { return 0; }
  virtual unsigned isSetInputDelay() { return 0; }
  virtual unsigned isSetOutputDelay() { return 0; }

 private:
  const char* _constrain_name;
};

}  // namespace ista
