#pragma once

#include "Logger.hpp"
#include "PinNode.hpp"
#include "ViaNode.hpp"
#include "WireNode.hpp"

namespace irt {

class PHYNode
{
 public:
  PHYNode() = default;
  PHYNode(const PHYNode& other) { copy(other); }
  PHYNode(PHYNode&& other) { move(std::forward<PHYNode>(other)); }
  ~PHYNode() { free(); }
  PHYNode& operator=(const PHYNode& other)
  {
    copy(other);
    return (*this);
  }
  PHYNode& operator=(PHYNode&& other)
  {
    move(std::forward<PHYNode>(other));
    return (*this);
  }
  // function
  template <typename T>
  T& getNode()
  {
    if (std::holds_alternative<std::monostate>(_data_node)) {
      _data_node = new T();
    } else if (!std::holds_alternative<T*>(_data_node)) {
      LOG_INST.error(Loc::current(), "The phy data_node is not the expected type!");
      exit(1);
    }
    return *std::get<T*>(_data_node);
  }

  bool isEmpty() { return std::holds_alternative<std::monostate>(_data_node); }

  template <typename T>
  bool isType()
  {
    return std::holds_alternative<T*>(_data_node);
  }

 private:
  std::variant<std::monostate, PinNode*, WireNode*, ViaNode*> _data_node;

  // function
  void copy(const PHYNode& other)
  {
    freeDataNode();
    copyDataNode(other);
  }

  void copyDataNode(const PHYNode& other)
  {
    std::visit(Overload{[&](std::monostate other_data_node) { _data_node = other_data_node; },
                        [&](PinNode* other_data_node) { _data_node = new PinNode(*other_data_node); },
                        [&](WireNode* other_data_node) { _data_node = new WireNode(*other_data_node); },
                        [&](ViaNode* other_data_node) { _data_node = new ViaNode(*other_data_node); }},
               other._data_node);
  }

  void move(PHYNode&& other)
  {
    _data_node = other._data_node;
    other._data_node = std::monostate();
  }

  void free() { freeDataNode(); }

  void freeDataNode()
  {
    std::visit(Overload{
                   [](std::monostate data_node) { return; },
                   [](PinNode* data_node) { delete data_node; },
                   [](WireNode* data_node) { delete data_node; },
                   [](ViaNode* data_node) { delete data_node; },
               },
               _data_node);
    _data_node = std::monostate();
  }
};

}  // namespace irt
