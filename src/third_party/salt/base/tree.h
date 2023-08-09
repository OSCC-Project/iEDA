#pragma once

#include "net.h"

#include <functional>

namespace salt {

class TreeNode {
public:
    int id;
    shared_ptr<Pin> pin;  // nullptr if it is not a pin
    Point loc;
    vector<shared_ptr<TreeNode>> children;  // empty for leaf
    shared_ptr<TreeNode> parent;  // nullptr for source

    // Constructors
    TreeNode(const Point& l = Point(), shared_ptr<Pin> p = nullptr, int i = -1) : loc(l), pin(p), id(i) {}
    TreeNode(DTYPE x, DTYPE y, shared_ptr<Pin> p = nullptr, int i = -1) : loc(x, y), pin(p), id(i) {}
    TreeNode(shared_ptr<Pin> p) : loc(p->loc), pin(p), id(p->id) {}

    // Accessors
    DTYPE WireToParent() const { return Dist(loc, parent->loc); }
    DTYPE WireToParentChecked() const { return parent ? WireToParent() : 0.0; }

    // Hunman-readable print
    void PrintSingle(ostream& os = cout) const; 
    void PrintRecursiveHelp(ostream& os, vector<bool>& prefix) const; // hunman-readable print
    void PrintRecursive(ostream& os = cout) const; 
    friend ostream& operator<<(ostream& os, const TreeNode& node) { node.PrintRecursive(os); return os; }

    // Set/reset/check parent/ancestor
    static void SetParent(const shared_ptr<TreeNode>& childNode, const shared_ptr<TreeNode>& parentNode);
    static void ResetParent(const shared_ptr<TreeNode>& node);
    static void Reroot(const shared_ptr<TreeNode>& node);
    static bool IsAncestor(const shared_ptr<TreeNode>& ancestor, const shared_ptr<TreeNode>& descendant);

    // Traverse
    static void PreOrder(const shared_ptr<TreeNode>& node, const function<void(const shared_ptr<TreeNode>&)>& visit);
    static void PostOrder(const shared_ptr<TreeNode>& node, const function<void(const shared_ptr<TreeNode>&)>& visit);
    static void PostOrderCopy(const shared_ptr<TreeNode>& node, const function<void(const shared_ptr<TreeNode>&)>& visit);
};

class Tree {
public:
    shared_ptr<TreeNode> source;
    const Net* net;

    // Note: take care when using copy assign operator and copy constructor. use swap().
    Tree(const shared_ptr<TreeNode>& sourceNode = nullptr, const Net* associatedNet = nullptr) : source(sourceNode), net(associatedNet) {}
    void Reset(bool freeTreeNodes = true);
    ~Tree() { Reset(); }

    // File read/write
    // ------
    // Format:
    // Tree <net_id> <net_name> <pin_num> [-cap]
    // 0 x0 y0 -1 [cap0]
    // 1 x1 y1 parent_idx1 [cap1]
    // 2 x2 y2 parent_idx2 [cap2]
    // ...
    // k xk yk parent_idxk
    // ...
    // ------
    // Notes:
    // 1. Nodes with indexes [0, pin_num) are pins, others are Steiner
    // 2. Steiner nodes have no cap
    void Read(istream& is);
    void Read(const string& fileName);
    void Write(ostream& os);  // TODO: const? but UpdateId
    void Write(const string& prefix, bool withNetInfo = true);

    // Hunman-readable print
    void Print(ostream& os = cout) const;
    friend ostream& operator<<(ostream& os, const Tree& tree) { tree.Print(os); return os; }

    // Traverse
    void PreOrder(const function<void(const shared_ptr<TreeNode>&)>& visit) { if (source) TreeNode::PreOrder(source, visit); }
    void PostOrder(const function<void(const shared_ptr<TreeNode>&)>& visit) { if (source) TreeNode::PostOrder(source, visit); }
    void PostOrderCopy(const function<void(const shared_ptr<TreeNode>&)>& visit) { if (source) TreeNode::PostOrderCopy(source, visit); }
    void PreOrder(const function<void(const shared_ptr<TreeNode>&)>& visit) const { if (source) TreeNode::PreOrder(source, visit); }
    void PostOrder(const function<void(const shared_ptr<TreeNode>&)>& visit) const { if (source) TreeNode::PostOrder(source, visit); }

    // Flatten
    int UpdateId();  // update node ids to [0, nodeNum), return nodeNum
    vector<shared_ptr<TreeNode>> ObtainNodes() const;

    // Legalize parent-child relationship
    void SetParentFromChildren();
    void SetParentFromUndirectedAdjList();
    void Reroot();
    void QuickCheck();  // check parent-child and pin coverage

    // Remove redundant Steiner nodes
    void RemovePhyRedundantSteiner();   // remove physically redudant ones (i.e., with the same locatoin of a pin)
    void RemoveTopoRedundantSteiner();  // remove topologically redudant ones (i.e., with 0/1 children)

    // Remove empty children
    void RemoveEmptyChildren();
};

}  // namespace salt