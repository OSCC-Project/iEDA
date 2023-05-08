/**
 * @file Port.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class for port in the netlist.
 * @version 0.1
 * @date 2021-02-04
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include <map>
#include <string>
#include <utility>

#include "Array.hh"
#include "DesignObject.hh"
#include "DisallowCopyAssign.hh"
#include "Type.hh"

namespace ista {

class Net;
class PortBus;

enum class PortDir { kIn, kOut, kInOut, kOther };
/**
 * @brief The class for port of design.
 *
 */
class Port : public DesignObject {
 public:
  explicit Port(const char* name, PortDir port_dir);
  Port(Port&& other) noexcept;
  Port& operator=(Port&& rhs) noexcept;
  ~Port() override = default;

  unsigned isPort() override { return 1; }
  unsigned isPin() override { return 0; }
  unsigned isInput() override {
    return (_port_dir == PortDir::kIn || _port_dir == PortDir::kInOut);
  }
  unsigned isOutput() override {
    return (_port_dir == PortDir::kOut || _port_dir == PortDir::kInOut);
  }
  unsigned isInout() override { return _port_dir == PortDir::kInOut; }
  double cap() override;
  double cap(AnalysisMode mode, TransType trans_type) override;

  void set_cap(double cap);
  void set_cap(AnalysisMode mode, TransType trans_type, double cap);

  void set_net(Net* net) override { _net = net; }
  Net* get_net() override { return _net; }

  std::string getFullName() override { return get_name(); }
  PortDir get_port_dir() { return _port_dir; }

  void set_port_bus(PortBus* port_bus) { _port_bus = port_bus; }
  auto* get_port_bus() { return _port_bus; }

 private:
  std::map<ModeTransPair, double> _caps;
  PortDir _port_dir;  //!< The port direction.
  Net* _net;          //!< The port connected net.

  PortBus* _port_bus = nullptr;  //!< The port owned by the port bus.

  DISALLOW_COPY_AND_ASSIGN(Port);
};

/**
 * @brief The class for port bus of design.
 *
 */
class PortBus : public DesignObject {
 public:
  PortBus(const char* name, unsigned left, unsigned right, unsigned size,
          PortDir port_dir);
  PortBus(PortBus&& other) noexcept = default;
  PortBus& operator=(PortBus&& rhs) noexcept = default;
  ~PortBus() override = default;

  unsigned isPortBus() override { return 1; }

  [[nodiscard]] unsigned get_left() const { return _left; }
  [[nodiscard]] unsigned get_right() const { return _right; }
  auto get_port_dir() { return _port_dir; }

  void addPort(unsigned index, Port* port) {
    LOG_FATAL_IF(index >= _size) << "beyond bus size";
    _ports[index] = port;
    port->set_port_bus(this);
  }

  Port* getPort(unsigned index) { return _ports[index]; }
  auto& getPorts() { return _ports; }

  std::string getFullName() override { return get_name(); }

 private:
  unsigned _left;     //!< The left range.
  unsigned _right;    //!< The right range.
  PortDir _port_dir;  //!< The port direction.

  std::unique_ptr<Port*[]> _ports;  //!< The ports.
  unsigned _size;                   //!< The port bus size.
  DISALLOW_COPY_AND_ASSIGN(PortBus);
};

}  // namespace ista