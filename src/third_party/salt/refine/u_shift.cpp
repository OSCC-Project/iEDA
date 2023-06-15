#include "refine.h"

namespace salt {

// cur - p
//  |    |
//  c    pp -> root
// four Nodes:      c       - cur   - p     - pp    -> root
// four points:     out[0]  - in[0] - in[1] - out[1]-> root
void Refine::UShift(Tree& tree) {
    auto nodes = tree.ObtainNodes();  // fix the nodes considered (no deletion will happen)
    for (auto cur : nodes) {
        Point in[2], out[2];
        auto p = cur->parent;
        if (p == nullptr) continue;
        auto pp = p->parent;
        if (pp == nullptr) continue;
        in[0] = cur->loc;
        in[1] = p->loc;
        out[1] = pp->loc;
        int d;  // primary direction (0-x: in[0]-in[1] is vertical; 1-y)
        for (d = 0; d < 2; ++d) {
            if (in[0][d] == in[1][d]) break;
        }
        if (d == 2) continue;  // not aligned
        bool larger;           // out[i] is in the larger side of in[i] (in secondary direction)
        if (out[1][d] == in[1][d]) continue;
        larger = out[1][d] > in[1][d];
        for (auto c : cur->children) {
            out[0] = c->loc;
            if (out[0][d] == in[0][d]) continue;
            if ((out[0][d] > in[0][d]) != larger) continue;  // should be at the same side
            Point newP[2];
            DTYPE closer = larger ? min(out[0][d], out[1][d]) : max(out[0][d], out[1][d]);
            for (int i = 0; i < 2; ++i) newP[i][d] = closer;            // set pri dir
            for (int i = 0; i < 2; ++i) newP[i][1 - d] = in[i][1 - d];  // set sec dir

            TreeNode::ResetParent(c);
            TreeNode::ResetParent(cur);
            TreeNode::ResetParent(p);
            //           inS[0]    inS[1]
            //              |         |
            // outS[0] - newS[0] - newS[1] - outS[1] -> root
            shared_ptr<TreeNode> inS[2] = {cur, p};
            shared_ptr<TreeNode> outS[2] = {c, pp};
            shared_ptr<TreeNode> newS[2];
            for (int i = 0; i < 2; ++i) {
                if (newP[i] == out[i])
                    newS[i] = outS[i];
                else {
                    newS[i] = make_shared<TreeNode>(newP[i]);
                    if (i == 0)
                        TreeNode::SetParent(outS[i], newS[i]);
                    else
                        TreeNode::SetParent(newS[i], outS[i]);
                }
                TreeNode::SetParent(inS[i], newS[i]);
            }
            TreeNode::SetParent(newS[0], newS[1]);
        }
    }

    tree.RemoveTopoRedundantSteiner();
}

}  // namespace salt