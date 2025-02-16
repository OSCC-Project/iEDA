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
 * @file IdbBus.h
 * @author pengming
 * @brief
 * @version 0.1
 * @date 2022-09-15
 */

#pragma once
#include <functional>
#include <map>
#include <optional>
#include <vector>

#include "IdbBusBitChars.h"
#include "IdbObject.h"
#include "IdbPins.h"
namespace idb {
class IdbBus
{
 public:
  enum kBusType
  {
    kBusNet,
    kBusInstancePin,
    kBusIo
  };
  IdbBus() = default;
  IdbBus(std::string name, unsigned left, unsigned right) : _bus_name(std::move(name)), _left(left), _right(right) {};
  ~IdbBus() = default;
  IdbBus(IdbBus&& other) = default;
  IdbBus& operator=(IdbBus&& other) = default;

  [[nodiscard]] const std::string& get_name() const { return _bus_name; };
  // get upper-bound of bus
  [[nodiscard]] unsigned get_left() const { return _left; }
  // get lower-bound of bus
  [[nodiscard]] unsigned get_right() const { return _right; };

  void updateRange(unsigned index);
  void updateRange(const IdbBus& bus_object);
  void set_name(const std::string& bus_name) { _bus_name = bus_name; }
  static std::optional<IdbBus> parseBusObj(const std::string& name_str, const IdbBusBitChars* bus_bit_chars);

  /**
   * @brief
   * Parse a bus object from name string.
   * If the given name does not belong to a bus, return nullopt.
   * For example (default busbitchars "[]"") :
   *  buffer[0] is a bus named "buffer" and index 0;
   *  buffer\[0\] is not a bus.
   * @param name_str
   * @param bus_bit_chars
   * @return std::optional<std::pair<std::string, unsigned>>
   */
  static std::optional<std::pair<std::string, unsigned>> parseBusName(std::string name_str, const IdbBusBitChars& bus_bit_chars);
  void set_type(kBusType type) { _bus_type = type; }
  [[nodiscard]] kBusType get_type() const { return _bus_type; }

  void addPin(IdbPin* pin);
  void addPin(IdbPin* pin, unsigned index);
  void addNet(IdbNet* net);
  void addNet(IdbNet* net, unsigned index);

  [[nodiscard]] const std::vector<IdbPin*>& getPins() const { return _pins; }
  IdbPin* getPin(unsigned index)
  {
    if (index >= _pins.size()) {
      return nullptr;
    }
    return _pins[index];
  }

  IdbNet* getNet(unsigned index);
  [[nodiscard]] const std::vector<IdbNet*>& getNets() const { return _nets; }

 private:
  std::string _bus_name;
  // upper bound
  unsigned _left{0};
  // lower bound
  unsigned _right{0};

  kBusType _bus_type{kBusNet};
  std::vector<IdbPin*> _pins;
  std::vector<IdbNet*> _nets;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief IdbBusList, used for bus query and storage
 *
 */
class IdbBusList
{
 public:
  /**
   * @brief find a bus object by name. If it doesn't exist, return nullopt.
   *
   * @param bus_name
   * @return std::optional<IdbBus>
   */
  std::optional<std::reference_wrapper<IdbBus>> findBus(const std::string& full_name);
  std::optional<std::reference_wrapper<IdbBus>> findBus(const std::string& instance_name, const std::string& pin_name);

  void addBusObject(IdbBus&& idb_bus);
  void addOrUpdate(const std::pair<std::string, unsigned>& info, const std::function<void(IdbBus&)>& setter);

  /**
   * @brief given a busname with an index, add it to buslist or update the
   * current bus with that busname in the buslist.
   * @param name  Use IdbBus::parseBusName to get busname from raw name.
   * @param index
   * @param setter :
   * A callable object that do some property settings after bus has been added
   * or updated. For example:  [](IdbBus& bus){ bus.set_type(kBusNet);}
   */
  void addOrUpdate(const std::string& name, unsigned index, const std::function<void(IdbBus&)>& setter);

  // get all buses as a const vector of IdbBus
  const std::vector<IdbBus>& get_bus_list() { return _bus_arr; }

 private:
  std::map<std::string, size_t> _bus_map;
  std::vector<IdbBus> _bus_arr;
};
}  // namespace idb