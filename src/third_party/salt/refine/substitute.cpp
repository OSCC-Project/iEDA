#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/segment.hpp>
#include <boost/geometry/index/rtree.hpp>

#include "refine.h"
#include "salt/base/eval.h"
#include "salt/base/mst.h"

namespace salt {

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using BPoint = bg::model::point<DTYPE, 2, bg::cs::cartesian>;
using BSegment = bg::model::segment<BPoint>;
using BBox = bg::model::box<BPoint>;
using BPolygon = bg::model::polygon<BPoint>;
using RNode = pair<BBox, shared_ptr<TreeNode>>;  // R-Tree node
struct RNodeComp
{
  bool operator()(const RNode& l, const RNode& r) const { return bg::equals(l.first, r.first) && l.second == r.second; }
};

void Refine::substitute(Tree& tree, double eps, bool useRTree)
{
  bgi::rtree<RNode, bgi::rstar<8>, bgi::indexable<RNode>, RNodeComp> rtree;
  if (useRTree) {
    tree.postOrder([&](const shared_ptr<TreeNode>& n) {
      if (n->parent) {
        BBox s;
        bg::envelope(BSegment(BPoint(n->loc.x, n->loc.y), BPoint(n->parent->loc.x, n->parent->loc.y)), s);
        rtree.insert({s, n});
      }
    });
  }
  auto disconnect = [&](const shared_ptr<TreeNode>& n) {
    if (useRTree) {
      BBox s;
      bg::envelope(BSegment(BPoint(n->loc.x, n->loc.y), BPoint(n->parent->loc.x, n->parent->loc.y)), s);
      rtree.remove({s, n});
    }
    TreeNode::resetParent(n);
  };
  auto connect = [&](const shared_ptr<TreeNode>& n, const shared_ptr<TreeNode>& parent) {
    TreeNode::setParent(n, parent);
    if (useRTree) {
      BBox s;
      bg::envelope(BSegment(BPoint(n->loc.x, n->loc.y), BPoint(parent->loc.x, parent->loc.y)), s);
      rtree.insert({s, n});
    }
  };
  while (true) {
    // Get nearest neighbors
    tree.UpdateId();
    vector<shared_ptr<TreeNode>> nodes = tree.ObtainNodes(),
                                 ordered_nodes(nodes.size());  // note: all pins should be covered
    vector<Point> points(nodes.size());
    for (int i = 0; i < nodes.size(); ++i) {
      ordered_nodes[nodes[i]->id] = nodes[i];
      points[nodes[i]->id] = nodes[i]->loc;  // TODO: move within bracket
    }
    nodes = ordered_nodes;
    vector<vector<int>> nearest_neighbors;
    if (!useRTree) {
      MstBuilder mst_builder;
      mst_builder.GetAllNearestNeighbors(points, nearest_neighbors);
    } else {
      nearest_neighbors.resize(nodes.size());
      for (auto n : nodes) {
        if (n->parent) {
          Point c = n->loc;  // center
          DTYPE radius = n->WireToParent();
          // diamond is too slow...
          // BPolygon diamond;
          // diamond.outer().emplace_back(c.x - radius, c.y);
          // diamond.outer().emplace_back(c.x, c.y + radius);
          // diamond.outer().emplace_back(c.x + radius, c.y);
          // diamond.outer().emplace_back(c.x, c.y - radius);
          // diamond.outer().emplace_back(c.x - radius, c.y);
          BBox query_box{{c.x - radius, c.y - radius}, {c.x + radius, c.y + radius}};
          vector<RNode> cands;
          rtree.query(bgi::intersects(query_box), back_inserter(cands));  // TODO: change back_inserter
          for (const auto& cand : cands) {
            nearest_neighbors[n->id].push_back(cand.second->id);
          }
        }
      }
    }

    // Prune descendants in nearest neighbors
    vector<int> pre_order_idxes(nodes.size(), -1);
    int global_pre_order_idx = 0;
    function<void(const shared_ptr<TreeNode>&)> remove_descendants = [&](const shared_ptr<TreeNode>& node) {
      pre_order_idxes[node->id] = global_pre_order_idx++;
      for (auto child : node->children) {
        remove_descendants(child);
      }
      for (auto& neigh_idx : nearest_neighbors[node->id]) {
        int neigh_pre_order_idx = pre_order_idxes[neigh_idx];
        if (neigh_pre_order_idx != -1 && neigh_pre_order_idx >= pre_order_idxes[node->id]) {
          neigh_idx = -1;  // -1 stands for "descendant"
        }
      }
    };
    remove_descendants(tree.source);

    // Init path lengths and subtree slacks
    vector<DTYPE> path_lengths(nodes.size());
    vector<DTYPE> slacks(nodes.size());
    auto update_path_lengths = [&](const shared_ptr<TreeNode>& node) {
      if (node->parent) {
        path_lengths[node->id] = path_lengths[node->parent->id] + node->WireToParent();
      } else {
        path_lengths[node->id] = 0;
      }
    };
    auto update_slacks = [&](const shared_ptr<TreeNode>& node) {
      if (node->children.empty()) {
        slacks[node->id] = Dist(node->loc, tree.source->loc) * (1 + eps) - path_lengths[node->id];  // floor here...
      } else {
        DTYPE min_slack = Dist(node->loc, tree.source->loc) * (1 + eps) - path_lengths[node->id];
        for (auto child : node->children) {
          min_slack = min(min_slack, slacks[child->id]);
        }
        slacks[node->id] = min_slack;
      }
    };
    tree.preOrder(update_path_lengths);
    tree.postOrder(update_slacks);

    // Find legal candidate moves
    using MoveT = tuple<DTYPE, shared_ptr<TreeNode>, shared_ptr<TreeNode>>;
    vector<MoveT> candidate_moves;  // <wire_length_delta, node, newParent>
    auto get_nearest_point = [](const shared_ptr<TreeNode>& target, const shared_ptr<TreeNode>& neigh) {
      Box box(neigh->loc, neigh->parent->loc);
      box.Legalize();
      return box.GetNearestPointTo(target->loc);
    };
    for (auto node : nodes) {
      if (!(node->parent)) {
        continue;
      }
      DTYPE best_wire_length_delta = 0;  // the negative, the better
      shared_ptr<TreeNode> best_new_parent;
      for (int neigh_idx : nearest_neighbors[node->id]) {
        if (neigh_idx == -1 || !nodes[neigh_idx]->parent)
          continue;
        auto neigh = nodes[neigh_idx];
        auto neigh_parent = neigh->parent;
        auto steiner_pt = get_nearest_point(node, neigh);
        DTYPE wire_length_delta = Dist(node->loc, steiner_pt) - node->WireToParent();
        if (wire_length_delta < best_wire_length_delta) {  // has wire length improvement
          DTYPE path_length_delta = path_lengths[neigh_parent->id] + Dist(node->loc, neigh_parent->loc) - path_lengths[node->id];
          if (path_length_delta <= slacks[node->id]) {  // make path length under control
            best_wire_length_delta = wire_length_delta;
            best_new_parent = neigh;
          }
        }
      }
      if (best_new_parent) {
        candidate_moves.emplace_back(best_wire_length_delta, node, best_new_parent);
      }
    }
    if (candidate_moves.empty()) {
      break;
    }

    // Try candidate moves in the order of descending wire length savings
    // Note that earlier moves may influence the legality of later one
    sort(candidate_moves.begin(), candidate_moves.end(), [](const MoveT& lhs, const MoveT& rhs) { return get<0>(lhs) < get<0>(rhs); });
    for (const auto& move : candidate_moves) {
      auto node = get<1>(move), neigh = get<2>(move);
      auto neigh_parent = neigh->parent;
      // check due to earlier moves
      if (TreeNode::isAncestor(node, neigh_parent))
        continue;
      DTYPE path_length_delta = path_lengths[neigh_parent->id] + Dist(node->loc, neigh_parent->loc) - path_lengths[node->id];
      if (path_length_delta > slacks[node->id])
        continue;
      auto steiner_pt = get_nearest_point(node, neigh);
      DTYPE wire_length_delta = Dist(node->loc, steiner_pt) - node->WireToParent();
      if (wire_length_delta >= 0)
        continue;
      // break
      disconnect(node);
      // reroot
      if (steiner_pt == neigh->loc) {
        connect(node, neigh);
      } else if (steiner_pt == neigh_parent->loc) {
        connect(node, neigh_parent);
      } else {
        auto steiner_node = make_shared<TreeNode>(steiner_pt);
        connect(steiner_node, neigh_parent);
        disconnect(neigh);
        connect(neigh, steiner_node);
        connect(node, steiner_node);
        // for later moves
        steiner_node->id = nodes.size();
        nodes.push_back(steiner_node);
        path_lengths.push_back(path_lengths[neigh_parent->id] + steiner_node->WireToParent());
        slacks.push_back(Dist(steiner_node->loc, tree.source->loc) * (1 + eps) - path_lengths.back());
      }
      // update slack for later moves: first subtree, then path to source
      TreeNode::preOrder(neigh_parent, update_path_lengths);
      TreeNode::postOrder(neigh_parent, update_slacks);
      auto tmp = neigh_parent;
      while (tmp->parent) {
        slacks[tmp->parent->id] = min(slacks[tmp->parent->id], slacks[tmp->id]);
        tmp = tmp->parent;
      }
    }

    // Finalize
    // tree.RemoveTopoRedundantSteiner();
    tree.postOrderCopy([&](const shared_ptr<TreeNode>& node) {
      // degree may change after post-order traversal of its children
      if (node->pin)
        return;
      if (node->children.empty()) {
        disconnect(node);
      } else if (node->children.size() == 1) {
        auto old_parent = node->parent, old_child = node->children[0];
        disconnect(node);
        disconnect(old_child);
        connect(old_child, old_parent);
      }
    });
  }
}

}  // namespace salt