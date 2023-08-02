#ifndef IMP_CLUSTER_H
#define IMP_CLUSTER_H
#include <cassert>
#include <memory>
#include <vector>

namespace imp {

template <typename T = void>
class TreeNode : public std::enable_shared_from_this<TreeNode<T>>
{
 public:
  std::shared_ptr<TreeNode<T>> makeTreeNode() { return std::make_shared<TreeNode<T>>(); }
  T* get() { return _Tp.get(); }
  void set(T* Tp) { _Tp.reset(Tp); }
  void addChild(std::shared_ptr<TreeNode<T>> cluster)
  {
    cluster->_parent = shared_from_this();
    _Children.push_back(cluster);
  }
  bool mergeChildren(std::vector<int> parts, std::vector<T*> parts_Tp, T* new_Tp);
  using std::enable_shared_from_this<TreeNode<T>>::shared_from_this;

 private:
  TreeNode() = default;
  TreeNode(T* Tp) : _Tp(Tp) {}
  ~TreeNode() = default;
  std::weak_ptr<TreeNode<T>> _parent;
  std::shared_ptr<T> _Tp;
  // std::vector<int> _idmap;
  std::vector<std::shared_ptr<TreeNode<T>>> _Children;
};

template <typename T>
bool TreeNode<T>::mergeChildren(std::vector<int> parts, std::vector<T*> parts_Tp, T* new_Tp)
{
  if (parts.size() != _Children.size()) {
    return false;
  }
  int num_parts = parts_Tp.size();
  assert(num_parts == *std::max_element(parts.begin(), parts.end()));
  std::vector<std::shared_ptr<TreeNode<T>>> new_sub_clusters(num_parts, std::make_shared<TreeNode<T>>());
  for (size_t i = 0; i < num_parts; i++) {
    new_sub_clusters[i]->set(parts_Tp[i]);
  }
  for (size_t i = 0; i < parts.size(); i++) {
    new_sub_clusters[parts[i]]->addChild(_Children[i]);
  }

  _Tp.reset(new_Tp);
  _Children = std::move(new_sub_clusters);

  return true;
}

}  // namespace imp

#endif