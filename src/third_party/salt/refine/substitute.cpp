#include "refine.h"

#include "salt/base/eval.h"
#include "salt/base/mst.h"

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/segment.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace salt {

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using BPoint = bg::model::point<DTYPE, 2, bg::cs::cartesian>;
using BSegment = bg::model::segment<BPoint>;
using BBox = bg::model::box<BPoint>;
using BPolygon = bg::model::polygon<BPoint>;
using RNode = pair<BBox, shared_ptr<TreeNode>>;  // R-Tree node
struct RNodeComp {
    bool operator()(const RNode& l, const RNode& r) const {
        return bg::equals(l.first, r.first) && l.second == r.second;
    }
};

void Refine::Substitute(Tree& tree, double eps, bool useRTree) {
    bgi::rtree<RNode, bgi::rstar<8>, bgi::indexable<RNode>, RNodeComp> rtree;
    if (useRTree) {
        tree.PostOrder([&](const shared_ptr<TreeNode>& n) {
            if (n->parent) {
                BBox s;
                bg::envelope(BSegment(BPoint(n->loc.x, n->loc.y), BPoint(n->parent->loc.x, n->parent->loc.y)), s);
                rtree.insert({s, n});
            }
        });
    }
    auto Disconnect = [&](const shared_ptr<TreeNode>& n) {
        if (useRTree) {
            BBox s;
            bg::envelope(BSegment(BPoint(n->loc.x, n->loc.y), BPoint(n->parent->loc.x, n->parent->loc.y)), s);
            rtree.remove({s, n});
        }
        TreeNode::ResetParent(n);
    };
    auto Connect = [&](const shared_ptr<TreeNode>& n, const shared_ptr<TreeNode>& parent) {
        TreeNode::SetParent(n, parent);
        if (useRTree) {
            BBox s;
            bg::envelope(BSegment(BPoint(n->loc.x, n->loc.y), BPoint(parent->loc.x, parent->loc.y)), s);
            rtree.insert({s, n});
        }
    };
    while (true) {
        // Get nearest neighbors
        int num = tree.UpdateId();
        vector<shared_ptr<TreeNode>> nodes = tree.ObtainNodes(),
                                     orderedNodes(nodes.size());  // note: all pins should be covered
        vector<Point> points(nodes.size());
        for (int i = 0; i < nodes.size(); ++i) {
            orderedNodes[nodes[i]->id] = nodes[i];
            points[nodes[i]->id] = nodes[i]->loc;  // TODO: move within bracket
        }
        nodes = orderedNodes;
        vector<vector<int>> nearestNeighbors;
        if (!useRTree) {
            MstBuilder mstB;
            mstB.GetAllNearestNeighbors(points, nearestNeighbors);
        } else {
            nearestNeighbors.resize(nodes.size());
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
                    BBox queryBox{{c.x - radius, c.y - radius}, {c.x + radius, c.y + radius}};
                    vector<RNode> cands;
                    rtree.query(bgi::intersects(queryBox), back_inserter(cands));  // TODO: change back_inserter
                    for (const auto& cand : cands) {
                        nearestNeighbors[n->id].push_back(cand.second->id);
                    }
                }
            }
        }

        // Prune descendants in nearest neighbors
        vector<int> preOrderIdxes(nodes.size(), -1);
        int globalPreOrderIdx = 0;
        function<void(const shared_ptr<TreeNode>&)> removeDescendants = [&](const shared_ptr<TreeNode>& node) {
            preOrderIdxes[node->id] = globalPreOrderIdx++;
            for (auto child : node->children) {
                removeDescendants(child);
            }
            for (auto& neighIdx : nearestNeighbors[node->id]) {
                int neighPreOrderIdx = preOrderIdxes[neighIdx];
                if (neighPreOrderIdx != -1 && neighPreOrderIdx >= preOrderIdxes[node->id]) {
                    neighIdx = -1;  // -1 stands for "descendant"
                }
            }
        };
        removeDescendants(tree.source);

        // Init path lengths and subtree slacks
        vector<DTYPE> pathLengths(nodes.size());
        vector<DTYPE> slacks(nodes.size());
        auto UpdatePathLengths = [&](const shared_ptr<TreeNode>& node) {
            if (node->parent) {
                pathLengths[node->id] = pathLengths[node->parent->id] + node->WireToParent();
            } else {
                pathLengths[node->id] = 0;
            }
        };
        auto UpdateSlacks = [&](const shared_ptr<TreeNode>& node) {
            if (node->children.empty()) {
                slacks[node->id] =
                    Dist(node->loc, tree.source->loc) * (1 + eps) - pathLengths[node->id];  // floor here...
            } else {
                DTYPE minSlack = Dist(node->loc, tree.source->loc) * (1 + eps) - pathLengths[node->id];
                for (auto child : node->children) {
                    minSlack = min(minSlack, slacks[child->id]);
                }
                slacks[node->id] = minSlack;
            }
        };
        tree.PreOrder(UpdatePathLengths);
        tree.PostOrder(UpdateSlacks);

        // Find legal candidate moves
        using MoveT = tuple<DTYPE, shared_ptr<TreeNode>, shared_ptr<TreeNode>>;
        vector<MoveT> candidateMoves;  // <wireLengthDelta, node, newParent>
        auto GetNearestPoint = [](const shared_ptr<TreeNode>& target, const shared_ptr<TreeNode>& neigh) {
            Box box(neigh->loc, neigh->parent->loc);
            box.Legalize();
            return box.GetNearestPointTo(target->loc);
        };
        for (auto node : nodes) {
            if (!(node->parent)) {
                continue;
            }
            DTYPE bestWireLengthDelta = 0;  // the negative, the better
            shared_ptr<TreeNode> bestNewParent;
            for (int neighIdx : nearestNeighbors[node->id]) {
                if (neighIdx == -1 || !nodes[neighIdx]->parent) continue;
                auto neigh = nodes[neighIdx];
                auto neighParent = neigh->parent;
                auto steinerPt = GetNearestPoint(node, neigh);
                DTYPE wireLengthDelta = Dist(node->loc, steinerPt) - node->WireToParent();
                if (wireLengthDelta < bestWireLengthDelta) {  // has wire length improvement
                    DTYPE pathLengthDelta =
                        pathLengths[neighParent->id] + Dist(node->loc, neighParent->loc) - pathLengths[node->id];
                    if (pathLengthDelta <= slacks[node->id]) {  // make path length under control
                        bestWireLengthDelta = wireLengthDelta;
                        bestNewParent = neigh;
                    }
                }
            }
            if (bestNewParent) {
                candidateMoves.emplace_back(bestWireLengthDelta, node, bestNewParent);
            }
        }
        if (candidateMoves.empty()) {
            break;
        }

        // Try candidate moves in the order of descending wire length savings
        // Note that earlier moves may influence the legality of later one
        sort(candidateMoves.begin(), candidateMoves.end(), [](const MoveT& lhs, const MoveT& rhs){
            return get<0>(lhs) < get<0>(rhs);
        });
        for (const auto& move : candidateMoves) {
            auto node = get<1>(move), neigh = get<2>(move);
            auto neighParent = neigh->parent;
            // check due to earlier moves
            if (TreeNode::IsAncestor(node, neighParent)) continue;
            DTYPE pathLengthDelta =
                pathLengths[neighParent->id] + Dist(node->loc, neighParent->loc) - pathLengths[node->id];
            if (pathLengthDelta > slacks[node->id]) continue;
            auto steinerPt = GetNearestPoint(node, neigh);
            DTYPE wireLengthDelta = Dist(node->loc, steinerPt) - node->WireToParent();
            if (wireLengthDelta >= 0) continue;
            // break
            Disconnect(node);
            // reroot
            if (steinerPt == neigh->loc) {
                Connect(node, neigh);
            } else if (steinerPt == neighParent->loc) {
                Connect(node, neighParent);
            } else {
                auto steinerNode = make_shared<TreeNode>(steinerPt);
                Connect(steinerNode, neighParent);
                Disconnect(neigh);
                Connect(neigh, steinerNode);
                Connect(node, steinerNode);
                // for later moves
                steinerNode->id = nodes.size();
                nodes.push_back(steinerNode);
                pathLengths.push_back(pathLengths[neighParent->id] + steinerNode->WireToParent());
                slacks.push_back(Dist(steinerNode->loc, tree.source->loc) * (1 + eps) - pathLengths.back());
            }
            // update slack for later moves: first subtree, then path to source
            TreeNode::PreOrder(neighParent, UpdatePathLengths);
            TreeNode::PostOrder(neighParent, UpdateSlacks);
            auto tmp = neighParent;
            while (tmp->parent) {
                slacks[tmp->parent->id] = min(slacks[tmp->parent->id], slacks[tmp->id]);
                tmp = tmp->parent;
            }
        }

        // Finalize
        // tree.RemoveTopoRedundantSteiner();
        tree.PostOrderCopy([&](const shared_ptr<TreeNode>& node) {
            // degree may change after post-order traversal of its children
            if (node->pin) return;
            if (node->children.empty()) {
                Disconnect(node);
            } else if (node->children.size() == 1) {
                auto oldParent = node->parent, oldChild = node->children[0];
                Disconnect(node);
                Disconnect(oldChild);
                Connect(oldChild, oldParent);
            }
        });
    }
}

}  // namespace salt