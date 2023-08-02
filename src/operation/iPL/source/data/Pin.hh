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
 * @FilePath: /iEDA/src/iPL/src/database/Pin.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_PIN_H
#define IPL_PIN_H

#include <string>
#include <vector>

#include "Point.hh"

namespace ipl {

class Instance;
class Net;

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
  int32_t get_pin_id() const { return _pin_id; }
  std::string get_name() const { return _name; }
  Net* get_net() const { return _net; }
  Instance* get_instance() const { return _instance; }

  Point<int32_t> get_offset_coordi() const { return _offset_coordi; }
  Point<int32_t> get_center_coordi() const { return _center_coordi; }

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
  void set_pin_id(int32_t id) { _pin_id = id; }
  void set_net(Net* net) { _net = net; }
  void set_instance(Instance* inst) { _instance = inst; }

  void set_offset_coordi(int32_t x, int32_t y)
  {
    _offset_coordi.set_x(x);
    _offset_coordi.set_y(y);
  }
  void set_offset_coordi(Point<int32_t> offset) { _offset_coordi = std::move(offset); }
  void set_center_coordi(int32_t x, int32_t y)
  {
    _center_coordi.set_x(x);
    _center_coordi.set_y(y);
  }
  void set_center_coordi(Point<int32_t> center) { _center_coordi = std::move(center); }

  void set_pin_type(PIN_TYPE type) { _pin_type = type; }
  void set_pin_io_type(PIN_IO_TYPE type) { _pin_io_type = type; }

 private:
  int32_t _pin_id;
  /* _name format as "instace name" + separator(e.g. ":","/") + "port name"  */
  std::string _name;
  Net* _net;
  Instance* _instance;

  /* (0,0) for io port ; offset is relative to instance center */
  Point<int32_t> _offset_coordi;
  Point<int32_t> _center_coordi;

  PIN_TYPE _pin_type;
  PIN_IO_TYPE _pin_io_type;
};

inline Pin::Pin(std::string name)
    : _name(std::move(name)), _net(nullptr), _instance(nullptr), _pin_type(PIN_TYPE::kNone), _pin_io_type(PIN_IO_TYPE::kNone)
{
}

}  // namespace ipl

#endif