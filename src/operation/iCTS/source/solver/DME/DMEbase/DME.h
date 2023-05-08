#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "BST.h"
#include "DmeNode.h"
#include "Params.h"
#include "UST.h"
#include "ZST.h"

namespace icts {
using std::pair;
using std::vector;

template <typename T, typename P>
void dme(Topology<T> &topo, const P &params) {
  typedef typename P::DmeTag DmeTag;
  dme(topo, params, DmeTag());
}

template <typename T, typename P>
void dme(Topology<T> &topo, const P &params, ZstTag tag) {
  Topology<ZstNode<T>> zst_topo;
  zst_topo.copy(topo, [](ZstNode<T> &zst_node, const T &t) {
    DmeNode<T> dme_node(t);
    zst_node = ZstNode<T>(dme_node);
  });

  ZeroSkewTree zst(params);
  zst.build(zst_topo);

  topo.clear();
  topo.copy(zst_topo,
            [](T &t, ZstNode<T> &zst_node) { t = zst_node.get_data(); });
}

template <typename T, typename P>
void dme(Topology<T> &topo, const P &params, BstTag tag) {
  Topology<BstNode<T>> bst_topo;
  bst_topo.copy(topo, [](BstNode<T> &bst_node, const T &t) {
    DmeNode<T> dme_node(t);
    bst_node = BstNode<T>(dme_node);
  });

  BoundedSkewTree bst(params);
  bst.build(bst_topo);

  topo.clear();
  topo.copy(bst_topo,
            [](T &t, BstNode<T> &bst_node) { t = bst_node.get_data(); });
}

template <typename T, typename P>
void dme(Topology<T> &topo, const P &params, SkewScheduler *skew_scheduler) {
  Topology<UstNode<T>> ust_topo;
  ust_topo.copy(topo, [](UstNode<T> &ust_node, const T &t) {
    DmeNode<T> dme_node(t);
    ust_node = UstNode<T>(dme_node);
  });
  UsefulSkewTree ust(params, skew_scheduler);
  ust.build(ust_topo);

  topo.clear();
  topo.copy(ust_topo,
            [](T &t, UstNode<T> &ust_node) { t = ust_node.get_data(); });
}
}  // namespace icts
