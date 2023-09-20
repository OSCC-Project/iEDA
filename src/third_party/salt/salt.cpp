#include "salt.h"

#include "base/flute.h"
#include "base/rsa.h"
#include "refine/refine.h"

namespace salt {

void SaltInterface::Init(Tree& min_tree, shared_ptr<Pin> src_pin)
{
  min_tree.UpdateId();
  auto mt_nodes = min_tree.ObtainNodes();
  sl_nodes.resize(mt_nodes.size());
  shortest_dists.resize(mt_nodes.size());
  cur_dists.resize(mt_nodes.size());
  for (auto mt_node : mt_nodes) {
    sl_nodes[mt_node->id] = make_shared<TreeNode>(mt_node->loc, mt_node->pin, mt_node->id);
    shortest_dists[mt_node->id] = Dist(mt_node->loc, src_pin->loc);
    cur_dists[mt_node->id] = numeric_limits<DTYPE>::max();
  }
  cur_dists[src_pin->id] = 0;
  sl_src = sl_nodes[src_pin->id];
}

void SaltInterface::Finalize(const Net& net, Tree& tree)
{
  for (auto n : sl_nodes)
    if (n->parent)
      sl_nodes[n->parent->id]->children.push_back(n);
  tree.source = sl_src;
  tree.net = &net;
}

void SaltBuilder::Run(const Net& net, Tree& tree, double eps, int refineLevel)
{
  // SMT
  Tree smt;
  FluteBuilder flute_builder;
  flute_builder.Run(net, smt);

  // Refine SMT
  if (refineLevel >= 1) {
    Refine::flip(smt);
    Refine::uShift(smt);
  }

  // Init
  Init(smt, net.source());

  // DFS
  DFS(smt.source, sl_src, eps);
  Finalize(net, tree);
  tree.RemoveTopoRedundantSteiner();

  // Connect breakpoints to source by RSA
  salt::RsaBuilder rsa_builder;
  rsa_builder.ReplaceRootChildren(tree);

  // Refine SALT
  if (refineLevel >= 1) {
    Refine::cancelIntersect(tree);
    Refine::flip(tree);
    Refine::uShift(tree);
    if (refineLevel >= 2) {
      Refine::substitute(tree, eps, refineLevel == 3);
    }
  }
}

bool SaltBuilder::Relax(const shared_ptr<TreeNode>& u, const shared_ptr<TreeNode>& v)
{
  DTYPE new_dist = cur_dists[u->id] + Dist(u->loc, v->loc);
  if (cur_dists[v->id] > new_dist) {
    cur_dists[v->id] = new_dist;
    v->parent = u;
    return true;
  } else if (cur_dists[v->id] == new_dist && Dist(u->loc, v->loc) < v->WireToParentChecked()) {
    v->parent = u;
    return true;
  } else
    return false;
}

void SaltBuilder::DFS(const shared_ptr<TreeNode>& mst_node, const shared_ptr<TreeNode>& sl_node, double eps)
{
  if (mst_node->pin && cur_dists[sl_node->id] > (1 + eps) * shortest_dists[sl_node->id]) {
    sl_node->parent = sl_src;
    cur_dists[sl_node->id] = shortest_dists[sl_node->id];
  }
  for (auto c : mst_node->children) {
    Relax(sl_node, sl_nodes[c->id]);
    DFS(c, sl_nodes[c->id], eps);
    Relax(sl_nodes[c->id], sl_node);
  }
}

}  // namespace salt
