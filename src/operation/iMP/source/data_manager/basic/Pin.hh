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
/*
 * @Author: S.J Chen
 * @Date: 2022-01-20 21:52:59
 * @LastEditTime: 2022-12-08 22:45:56
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/imp/src/database/Pin.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IMP_PIN_H
#define IMP_PIN_H

#include <string>
#include <vector>

#include "Geometry.hh"

namespace imp {

enum class PIN_TYPE
{
  kNone,
  kInstancePort,
  kIOPort,
  kFakePin
};

enum class PIN_IO_TYPE
{
  kNone,
  kInput,
  kOutput,
  kInputOutput
};

class Pin
{
 public:
  Pin() = delete;
  explicit Pin(std::string name);
  Pin(const Pin&) = delete;
  Pin(Pin&&) = delete;
  ~Pin() = default;

  Pin& operator=(const Pin&) = delete;
  Pin& operator=(Pin&&) = delete;

  // getter.
  std::string get_name() const { return _name; }

  bool isIOPort() const { return _pin_type == PIN_TYPE::kIOPort; }
  bool isIOInput() const { return (_pin_type == PIN_TYPE::kIOPort) && (_pin_io_type == PIN_IO_TYPE::kInput); }
  bool isIOOutput() const { return (_pin_type == PIN_TYPE::kIOPort) && (_pin_io_type == PIN_IO_TYPE::kOutput); }
  bool isIOInputOutput() const { return (_pin_type == PIN_TYPE::kIOPort) && (_pin_io_type == PIN_IO_TYPE::kInputOutput); }

  bool isInstancePort() const { return _pin_type == PIN_TYPE::kInstancePort; }
  bool isInstanceInput() const { return (_pin_type == PIN_TYPE::kInstancePort) && (_pin_io_type == PIN_IO_TYPE::kInput); }
  bool isInstanceOutput() const { return (_pin_type == PIN_TYPE::kInstancePort) && (_pin_io_type == PIN_IO_TYPE::kOutput); }
  bool isInstanceInputOutput() const { return (_pin_type == PIN_TYPE::kInstancePort) && (_pin_io_type == PIN_IO_TYPE::kInputOutput); }

  bool isFakePin() const { return _pin_type == PIN_TYPE::kFakePin; }

  // setter.

  void set_coordi(int32_t x, int32_t y) { _coordi.set<0>(x); _coordi.set<1>(y);}

  void set_offset(int32_t x, int32_t y) { _offset.set<0>(x); _offset.set<1>(y);}

  void set_pin_type(PIN_TYPE type) { _pin_type = type; }
  void set_pin_io_type(PIN_IO_TYPE type) { _pin_io_type = type; }

 private:
  std::string _name;

  /* (0,0) for io port ; offset is relative to instance min corner */
  geo::point<int32_t> _offset;
  geo::point<int32_t> _coordi;

  PIN_TYPE _pin_type;
  PIN_IO_TYPE _pin_io_type;
};

inline Pin::Pin(std::string name) : _name(std::move(name)), _pin_type(PIN_TYPE::kNone), _pin_io_type(PIN_IO_TYPE::kNone)
{
}

}  // namespace imp

#endif