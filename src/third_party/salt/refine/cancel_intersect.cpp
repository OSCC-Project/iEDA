#include "refine.h"

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
using RNode = pair<BBox, shared_ptr<TreeNode>>;  // R-Tree node
struct RNodeComp {
    bool operator()(const RNode& l, const RNode& r) const {
        return bg::equals(l.first, r.first) && l.second == r.second;
    }
};

// Remove cases where the intersecion point is two end points
void Purify(const shared_ptr<TreeNode> n, vector<RNode>& cands) {
    vector<RNode> res;
    for (auto c : cands) {
        auto s0 = n->loc;
        auto s1 = n->parent->loc;
        auto c0 = c.second->loc;
        auto c1 = c.second->parent->loc;
        // if (!bg::equals(s0, c0) && !bg::equals(s0, c1) && !bg::equals(s1, c0) && !bg::equals(s1, c1))
        if (s0 != c0 && s0 != c1 && s1 != c0 && s1 != c1) res.push_back(c);
    }
    cands = move(res);
}

// Process the identified intersected edges
void Cancel(array<shared_ptr<TreeNode>, 2> oric) {
    int d;  // direction id: 0-x 1-y

    // make the primary direction for (oric[0] - oric[0]->parent) x
    vector<DTYPE> xys[2];
    for (int d = 0; d < 2; ++d) {
        for (auto sn : {oric[0], oric[1], oric[0]->parent, oric[1]->parent}) xys[d].push_back(sn->loc[d]);
        sort(xys[d].begin(), xys[d].end());
    }
    auto minx = xys[0][0], maxx = xys[0][3], miny = xys[1][0], maxy = xys[1][3];
    if ((minx == oric[1]->loc.x || minx == oric[1]->parent->loc.x) &&
        (maxx == oric[1]->loc.x || maxx == oric[1]->parent->loc.x))  // for corner cases where three is the same
        // || ((miny == oric[0]->loc.y || miny == oric[0]->parent->loc.y) &&
        // (maxy == oric[0]->loc.y || maxy == oric[0]->parent->loc.y)))
        swap(oric[0], oric[1]);

    // make sure oric[d] is in the left side (xys[d][0]) of xys[d]
    for (d = 0; d < 2; ++d) {
        auto v = oric[d]->loc[d];
        if (v != xys[d][0] && v != xys[d][1]) {
            reverse(xys[d].begin(), xys[d].end());
            assert(v == xys[d][0] || v == xys[d][1]);
        }
    }

    // get closest corners in the overlapped box
    Point cclosest[2];
    for (d = 0; d < 2; ++d) {
        // coordinate of secondary direction is the same
        cclosest[d][1 - d] = oric[d]->loc[1 - d];
        // but cannot be extreme
        if (cclosest[d][1 - d] == xys[1 - d][0])
            cclosest[d][1 - d] = xys[1 - d][1];
        else if (cclosest[d][1 - d] == xys[1 - d][3])
            cclosest[d][1 - d] = xys[1 - d][2];
        // coordinate of primary direction is in xys
        cclosest[d][d] = xys[d][1];
        // log() << oric[d]->loc << ": " << cclosest[d] << endl;
    }

    // select oric[minD] (i.e., oric[1-minD]->parent)
    // two choices: 1. select shorter path; 2. select shorter edge
    function<DTYPE(const shared_ptr<TreeNode>&)> DistToSrc = [&](const shared_ptr<TreeNode>& node) -> DTYPE {
        if (!node->parent) return 0;
        return DistToSrc(node->parent) + node->WireToParent();
    };
    DTYPE totDist[2], edgeL[2];
    for (d = 0; d < 2; ++d) {
        edgeL[d] = Dist(oric[d]->parent->loc, cclosest[d]);
        totDist[d] = DistToSrc(oric[d]->parent) + edgeL[d];
    }
    int minD = ((totDist[0] < totDist[1]) || (totDist[0] == totDist[1] && edgeL[0] < edgeL[1])) ? 1 : 0;
    // special case
    auto trace = oric[1 - minD];
    while (trace->parent && trace != oric[minD]) trace = trace->parent;
    if (trace->parent) minD = 1 - minD;

    // clean, create & connect
    auto rootN = oric[1 - minD]->parent;
    shared_ptr<TreeNode> mergeN = nullptr;
    for (int d = 0; d < 2; ++d) {
        TreeNode::ResetParent(oric[d]);
        if (oric[d]->loc == cclosest[d]) mergeN = oric[d];
    }
    if (!mergeN) mergeN = make_shared<TreeNode>(cclosest[minD]);
    TreeNode::SetParent(mergeN, rootN);
    for (int d = 0; d < 2; ++d)
        if (oric[d] != mergeN) TreeNode::SetParent(oric[d], mergeN);
}

void Refine::CancelIntersect(Tree& tree) {
    bgi::rtree<RNode, bgi::linear<16>, bgi::indexable<RNode>, RNodeComp> rtree;
    auto nodes = tree.ObtainNodes();
    for (int i = 0; i < nodes.size(); ++i) {
        auto n = nodes[i];
        if (n->parent) {
            // n->parent & nodes may change
            BBox s;
            while (true) {
                bg::envelope(BSegment(BPoint(n->loc.x, n->loc.y), BPoint(n->parent->loc.x, n->parent->loc.y)), s);
                vector<RNode> cands;
                rtree.query(bgi::intersects(s), back_inserter(cands));
                Purify(n, cands);
                if (cands.empty()) break;
                auto c = cands[0];
                // log() << n->id << "," << n->parent->id << " " << c.second->id << "," << c.second->parent->id << endl;
                Cancel({n, c.second});
                assert(n->parent == c.second->parent || n->parent == c.second || n == c.second->parent);
                rtree.remove(c);
                if (n->parent == c.second->parent) nodes.push_back(n->parent);
                nodes.push_back(c.second);
            }
            rtree.insert({s, n});
        }
    }
    tree.RemoveTopoRedundantSteiner();
}

}  // namespace salt