#pragma once

#include "base/tree.h"

namespace salt {

class SaltBase {
protected:
    vector<shared_ptr<TreeNode>> slNodes; // nodes of the shallow-light tree
    vector<DTYPE> shortestDists;
    vector<DTYPE> curDists;
    shared_ptr<TreeNode> slSrc;  // source node of the shallow-light tree

    void Init(Tree& minTree, shared_ptr<Pin> srcP);   // tree of minimum weight
    void Finalize(const Net& net, Tree& tree);
    virtual bool Relax(const shared_ptr<TreeNode>& u, const shared_ptr<TreeNode>& v) = 0;  // from u to v
    virtual void DFS(const shared_ptr<TreeNode>& mstNode, const shared_ptr<TreeNode>& slNode, double eps) = 0;
};

class SaltBuilder : public SaltBase {
public:
    void Run(const Net& net, Tree& tree, double eps, int refineLevel = 3);

protected:
    bool Relax(const shared_ptr<TreeNode>& u, const shared_ptr<TreeNode>& v);  // from u to v
    void DFS(const shared_ptr<TreeNode>& mstNode, const shared_ptr<TreeNode>& slNode, double eps);
};

}  // namespace salt