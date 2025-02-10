#include "flute.h"

#include <boost/functional/hash.hpp>
#include <unordered_map>

#include "flute3/flute.h"  // should be included after boost/functional/hash.hpp
#define MAXD 300000        // max. degree that can be handled
void salt::FluteBuilder::Run(const salt::Net& net, salt::Tree& saltTree)
{
  // load LUT
  static bool once = false;
  if (!once) {
    Flute::readLUT();
    once = true;
  }

  // Obtain flute tree
  Flute::Tree flute_tree;
  flute_tree.branch = nullptr;
  int d = net.pins.size();
  assert(d <= MAXD);
  int x[MAXD], y[MAXD];
  for (size_t i = 0; i < d; ++i) {
    x[i] = net.pins[i]->loc.x;
    y[i] = net.pins[i]->loc.y;
  }
  if (flute_tree.branch)
    free(flute_tree.branch);  // is it complete for mem leak?
  flute_tree = Flute::flute(d, x, y, FLUTE_ACCURACY);

  // Build adjacency list
  unordered_map<pair<DTYPE, DTYPE>, shared_ptr<salt::TreeNode>, boost::hash<pair<DTYPE, DTYPE>>> key2node;
  for (auto p : net.pins) {
    key2node[{p->loc.x, p->loc.y}] = make_shared<salt::TreeNode>(p);
  }
  auto& t = flute_tree;

  auto find_or_create = [&](DTYPE x, DTYPE y) {
    auto it = key2node.find({x, y});
    if (it == key2node.end()) {
      shared_ptr<salt::TreeNode> node = make_shared<salt::TreeNode>(x, y);
      key2node[{x, y}] = node;
      return node;
    } else
      return it->second;
  };

  for (int i = 0; i < 2 * t.deg - 2; i++) {
    int j = t.branch[i].n;
    if (t.branch[i].x == t.branch[j].x && t.branch[i].y == t.branch[j].y)
      continue;
    // any more duplicate?
    shared_ptr<salt::TreeNode> n1 = find_or_create(t.branch[i].x, t.branch[i].y);
    shared_ptr<salt::TreeNode> n2 = find_or_create(t.branch[j].x, t.branch[j].y);
    // printlog(LOG_INFO, "%d - %d\n", n1->pin?n1->pin->id:-1, n2->pin?n2->pin->id:-1);
    n1->children.push_back(n2);
    n2->children.push_back(n1);
  }

  // Reverse parent-child orders
  saltTree.source = key2node[{net.source()->loc.x, net.source()->loc.y}];
  saltTree.SetParentFromUndirectedAdjList();
  saltTree.net = &net;

  free(flute_tree.branch);
}