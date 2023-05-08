#pragma once

namespace irt {

template <typename T>
class TNode
{
 public:
  TNode() = default;
  explicit TNode(const T& v) { _v = v; }
  ~TNode() = default;
  // getter
  T& value() { return _v; }
  std::vector<TNode<T>*>& get_child_list() { return _child_list; }
  // setter
  void set_value(const T& v) { _v = v; }
  // function
  irt_int getChildrenNum() { return static_cast<irt_int>(_child_list.size()); }
  bool isLeafNode() { return getChildrenNum() == 0; }

  void addChild(TNode<T>* child) { _child_list.push_back(child); }
  void addChildren(const std::vector<TNode<T>*>& child_list)
  {
    for (size_t i = 0; i < child_list.size(); i++) {
      addChild(child_list[i]);
    }
  }
  void delChild(TNode<T>* child) { _child_list.erase(find(_child_list.begin(), _child_list.end(), child)); }
  void delChildren(const std::vector<TNode<T>*>& child_list)
  {
    for (size_t i = 0; i < child_list.size(); i++) {
      delChild(child_list[i]);
    }
  }
  void clearChildren() { _child_list.clear(); }

 private:
  T _v;
  std::vector<TNode<T>*> _child_list;
};

}  // namespace irt
