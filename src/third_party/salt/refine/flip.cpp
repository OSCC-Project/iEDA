#include "refine.h"

#include "salt/base/eval.h"

#include <algorithm>

namespace salt {

void GetDirId(const Point& a, const Point& b, bool low, int& dir, DTYPE& dist) {  // b is the center/base
    if (a.y < b.y && (low || a.x == b.x)) {                                       // down
        dir = 0;
        dist = b.y - a.y;
    } else if (a.y > b.y && (!low || a.x == b.x)) {  // up
        dir = 1;
        dist = a.y - b.y;
    } else if (a.x < b.x) {  // left
        dir = 2;
        dist = b.x - a.x;
    } else {  // right (or the same point)
        dir = 3;
        dist = a.x - b.x;
    }
}

void UpdateNode(const shared_ptr<TreeNode> node,       // central node
                const vector<bool>& align,             // a child is aligned or not?
                const vector<array<DTYPE, 2>>& gains,  // the gain of each child
                int i,                                 // current child
                array<vector<DTYPE>, 4>& segs,         // segments in each direction
                DTYPE accGain,                         // gain accumulated by now
                DTYPE& bestGain,                       // best gain
                vector<bool>& lowOrUp,                 // each child edge is low or up
                vector<bool>& bestLowOrUp              // each child edge is low or up for best gain
) {
    if (i == node->children.size()) {  // get to the end of enumeration
        for (int j = 0; j < 4; ++j) {
            if (segs[j].size() <= 1) continue;  // many stops here
            DTYPE max = 0;
            for (auto len : segs[j]) {
                if (len > max) {
                    accGain += max;
                    max = len;
                } else
                    accGain += len;
            }
        }
        if (accGain > bestGain) {
            bestGain = accGain;
            for (auto c : node->children) bestLowOrUp[c->id] = lowOrUp[c->id];
        }
    } else if (align[i]) {  // already consider, pass this child
        UpdateNode(node, align, gains, i + 1, segs, accGain, bestGain, lowOrUp, bestLowOrUp);
    } else {  // make a branch because this node-child edge can be L/U
        int dir;
        DTYPE dist;
        bool low = true;  // L/U
        for (int j = 0; j < 2; ++j) {
            GetDirId(node->children[i]->loc, node->loc, low, dir, dist);
            accGain += gains[i][j];
            segs[dir].push_back(dist);
            lowOrUp[node->children[i]->id] = low;
            UpdateNode(node, align, gains, i + 1, segs, accGain, bestGain, lowOrUp, bestLowOrUp);
            accGain -= gains[i][j];
            segs[dir].pop_back();
            low = false;  // L/U
        }
    }
}

void UpdateSubTree(const shared_ptr<TreeNode> node, array<DTYPE, 2>& res, array<vector<bool>, 2>& bestLowOrUp) {
    res = {0, 0};  // low, up
    if (node->children.empty()) return;

    // Calc gains of children
    unsigned nc = node->children.size();
    vector<array<DTYPE, 2>> gains(nc);
    for (unsigned i = 0; i < nc; ++i) UpdateSubTree(node->children[i], gains[i], bestLowOrUp);

    // Process aligned children
    vector<bool> align(nc, false);
    array<vector<DTYPE>, 4> segs;
    DTYPE accGain = 0;
    vector<bool> lowOrUp(bestLowOrUp[0].size());
    for (unsigned i = 0; i < nc; ++i) {
        Point &a = node->children[i]->loc, b = node->loc;
        if (a.x == b.x || a.y == b.y) {
            int dir;
            DTYPE dist;
            GetDirId(a, node->loc, true, dir, dist);
            segs[dir].push_back(dist);
            align[i] = true;  // no need for further processing
            accGain += gains[i][0];
        }
    }

    // Enumerate L flipping around the node
    if (node->parent) {
        bool low = true;  // L/U
        int dir;
        DTYPE dist;
        for (int j = 0; j < 2; ++j) {
            GetDirId(node->parent->loc, node->loc, low, dir, dist);
            segs[dir].push_back(dist);
            UpdateNode(node, align, gains, 0, segs, accGain, res[j], lowOrUp, bestLowOrUp[j]);
            segs[dir].pop_back();
            low = false;  // L/U
        }
    } else  // for source node, update res[0] only
        UpdateNode(node, align, gains, 0, segs, accGain, res[0], lowOrUp, bestLowOrUp[0]);

    // cout<<*node->pin<<", low="<<res[0]<<", up="<<res[1]<<endl;
}

void AddNode(shared_ptr<TreeNode> c1, shared_ptr<TreeNode> c2, shared_ptr<TreeNode> p, const shared_ptr<TreeNode>& steiner) {
    // Make sure p is the parent (p>c1, p>c2)
    if (p->parent == c1) swap(p, c1);
    if (p->parent == c2) swap(p, c2);
    if (p->parent == c1) swap(p, c1);

    // Update
    // case 1: p > c1 = c2
    // case 2. p > c1 > c2
    if (!(c1->parent == p && c2->parent == p)) {
        if (c1->parent == c2) swap(c1, c2);  // make sure (c1>=c2)
        assert(c2->parent == c1 && c1->parent == p);
    }
    TreeNode::ResetParent(c1);
    TreeNode::SetParent(c1, steiner);
    TreeNode::ResetParent(c2);
    TreeNode::SetParent(c2, steiner);
    TreeNode::SetParent(steiner, p);
}

int dir2xy[4] = {0, 0, 1, 1};  // down, up, left, right

void TraverseAndAddSteiner(const shared_ptr<TreeNode>& node, bool low, array<vector<bool>, 2>& bestLowOrUp) {
    // cout<<node->id<<(low?" low":"  up")<<endl;
    // node->Print(0,true);
    if (node->children.empty()) return;

    // go down first
    int luid = low ? 0 : 1;
    auto children_cpy = node->children;  // it may be chagned later
    for (auto c : children_cpy) TraverseAndAddSteiner(c, bestLowOrUp[luid][c->id], bestLowOrUp);

    // get overlap info
    array<vector<pair<int, shared_ptr<TreeNode>>>, 4> dist2node;  // for four directions
    int dir;
    DTYPE dist;
    if (node->parent) {
        GetDirId(node->parent->loc, node->loc, low, dir, dist);
        dist2node[dir].emplace_back(dist, node->parent);
    }
    for (auto c : node->children) {
        if (c->id == -1) continue;
        GetDirId(c->loc, node->loc, bestLowOrUp[luid][c->id], dir, dist);
        dist2node[dir].emplace_back(dist, c);
    }

    // add steiner point
    auto oriParent = node->parent;
    for (dir = 0; dir < 4; ++dir) {
        auto& cands = dist2node[dir];
        if (cands.size() <= 1) continue;

        sort(cands.begin(),
             cands.end(),
             [](const pair<int, shared_ptr<TreeNode>>& a, const pair<int, shared_ptr<TreeNode>>& b) {
                 return a.first > b.first;
             });

        auto out = cands[0].second;
        for (int cid = 1; cid < cands.size(); ++cid) {
            int xy = dir2xy[dir];
            Point p;
            p[xy] = node->loc[xy];
            p[1 - xy] = cands[cid].second->loc[1 - xy];
            shared_ptr<TreeNode> steiner = make_shared<TreeNode>(p, nullptr, -1);
            AddNode(out, cands[cid].second, node, steiner);
            out = steiner;
        }
    }
}

void Refine::Flip(Tree& tree) {
    WireLengthEvalBase cur(tree), pre;

    do {
        pre = cur;
        array<DTYPE, 2> g;  // low, up
        int numNode = tree.UpdateId();
        array<vector<bool>, 2> bestLowOrUp = {vector<bool>(numNode), vector<bool>(numNode)};
        UpdateSubTree(tree.source, g, bestLowOrUp);
        TraverseAndAddSteiner(tree.source, true, bestLowOrUp);
        tree.RemovePhyRedundantSteiner();
        cur.Update(tree);
    } while (cur.wireLength < pre.wireLength);
}

}  // namespace salt