#include "tree.h"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace salt {

void TreeNode::PrintSingle(ostream& os) const {
    os << "Node " << id << ": " << loc << (pin ? ", pin" : "") << ", " << children.size() << " children";
}

void TreeNode::PrintRecursiveHelp(ostream& os, vector<bool>& prefix) const {
    for (auto pre : prefix) os << (pre ? "  |" : "   ");
    if (!prefix.empty()) os << "-> ";
    PrintSingle(os);
    os << endl;
    if (children.size() > 0) {
        prefix.push_back(true);
        for (size_t i = 0; i < children.size() - 1; ++i) {
            if (children[i])
                children[i]->PrintRecursiveHelp(os, prefix);
            else
                os << "<null>" << endl;
        }
        prefix.back() = false;
        children.back()->PrintRecursiveHelp(os, prefix);
        prefix.pop_back();
    }
}

void TreeNode::PrintRecursive(ostream& os) const {
    vector<bool> prefix;  // prefix indicates whether an ancestor is a last child or not
    PrintRecursiveHelp(os, prefix);
}

void TreeNode::SetParent(const shared_ptr<TreeNode>& childNode, const shared_ptr<TreeNode>& parentNode) {
    childNode->parent = parentNode;
    parentNode->children.push_back(childNode);
}

void TreeNode::ResetParent(const shared_ptr<TreeNode>& node) {
    assert(node->parent);

    auto& n = node->parent->children;
    auto it = find(n.begin(), n.end(), node);
    assert(it != n.end());
    *it = n.back();
    n.pop_back();

    node->parent.reset();
}

void TreeNode::Reroot(const shared_ptr<TreeNode>& node) {
    if (node->parent) {
        Reroot(node->parent);
        auto oldParent = node->parent;
        TreeNode::ResetParent(node);
        TreeNode::SetParent(oldParent, node);
    }
}

bool TreeNode::IsAncestor(const shared_ptr<TreeNode>& ancestor, const shared_ptr<TreeNode>& descendant) {
    auto tmp = descendant;
    do {
        if (tmp == ancestor) {
            return true;
        }
        tmp = tmp->parent;
    } while (tmp);
    return false;
}

void TreeNode::PreOrder(const shared_ptr<TreeNode>& node, const function<void(const shared_ptr<TreeNode>&)>& visit) {
    visit(node);
    for (auto c : node->children) PreOrder(c, visit);
}

void TreeNode::PostOrder(const shared_ptr<TreeNode>& node, const function<void(const shared_ptr<TreeNode>&)>& visit) {
    for (auto c : node->children) PostOrder(c, visit);
    visit(node);
}

void TreeNode::PostOrderCopy(const shared_ptr<TreeNode>& node, const function<void(const shared_ptr<TreeNode>&)>& visit) {
    auto tmp = node->children;
    for (auto c : tmp) PostOrderCopy(c, visit);
    visit(node);
}

void Tree::Reset(bool freeTreeNodes) {
    if (freeTreeNodes) {
        PostOrder([](const shared_ptr<TreeNode>& node) { node->children.clear(); });
    }
    source.reset();
    net = nullptr;
}

void Tree::Read(istream& is) {
    Net* pNet = new Net;

    // header
    string buf, option;
    int numPin = 0;
    while (is >> buf && buf != "Tree")
        ;
    assert(buf == "Tree");
    getline(is, buf);
    istringstream iss(buf);
    iss >> pNet->id >> pNet->name >> numPin >> option;
    assert(numPin > 0);
    pNet->withCap = (option == "-cap");

    // pins
    int i, parentIdx;
    DTYPE x, y;
    double c = 0.0;
    pNet->pins.resize(numPin);
    vector<int> parentIdxs;
    vector<shared_ptr<TreeNode>> treeNodes;
    for (auto& pin : pNet->pins) {
        is >> i >> x >> y >> parentIdx;
        assert(i == treeNodes.size());
        if (pNet->withCap) is >> c;
        pin = make_shared<Pin>(x, y, i, c);
        treeNodes.push_back(make_shared<TreeNode>(x, y, pin, i));
        parentIdxs.push_back(parentIdx);
    }
    assert(treeNodes.size() == numPin);

    // non-pin nodes
    getline(is, buf);  // consume eol
    streampos pos;
    while (true) {
        pos = is.tellg();
        getline(is, buf);
        istringstream iss2(buf);
        iss2 >> i >> x >> y >> parentIdx;
        if (iss2.fail()) break;
        assert(i == treeNodes.size());
        treeNodes.push_back(make_shared<TreeNode>(x, y, nullptr, i));
        parentIdxs.push_back(parentIdx);
    }
    is.seekg(pos);  // go back

    // parents
    for (unsigned i = 0; i < treeNodes.size(); ++i) {
        parentIdx = parentIdxs[i];
        if (parentIdx >= 0) {
            assert(parentIdx < treeNodes.size());
            salt::TreeNode::SetParent(treeNodes[i], treeNodes[parentIdx]);
        } else {
            assert(parentIdx == -1);
            source = treeNodes[i];
        }
    }

    net = pNet;
    // TODO: check dangling nodes
}

void Tree::Read(const string& fileName) {
    ifstream is(fileName);
    if (is.fail()) {
        cout << "ERROR: Cannot open file " << fileName << endl;
        exit(1);
    }
    Read(is);
}

void Tree::Write(ostream& os) {
    // header
    os << "Tree " << net->GetHeader() << endl;

    // nodes
    // Note: source pin may not be covered in some intermediate state
    int numNodes = UpdateId();
    auto nodes = ObtainNodes();
    vector<shared_ptr<TreeNode>> sortedNodes(numNodes, nullptr);  // bucket sort
    for (auto node : nodes) {
        sortedNodes[node->id] = node;
    }
    for (auto node : sortedNodes) {
        if (!node) continue;
        int parentId = node->parent ? node->parent->id : -1;
        os << node->id << " " << node->loc.x << " " << node->loc.y << " " << parentId;
        if (net->withCap && node->pin) os << " " << node->pin->cap;
        os << endl;
    }
}

void Tree::Write(const string& prefix, bool withNetInfo) {
    ofstream ofs(prefix + (withNetInfo ? ("_" + net->name) : "") + ".tree");
    Write(ofs);
    ofs.close();
}

void Tree::Print(ostream& os) const {
    os << "Tree ";
    if (net)
        os << net->id << ": #pins=" << net->pins.size() << endl;
    else
        os << "<no_net_associated>" << endl;
    if (source)
        source->PrintRecursive(os);
    else
        os << "<null>" << endl;
}

int Tree::UpdateId() {
    int numNode = net->pins.size();
    PreOrder([&](const shared_ptr<TreeNode>& node) {
        if (node->pin) {
            assert(node->pin->id < net->pins.size());
            node->id = node->pin->id;
        } else
            node->id = numNode++;
    });
    return numNode;
}

vector<shared_ptr<TreeNode>> Tree::ObtainNodes() const {
    vector<shared_ptr<TreeNode>> nodes;
    PreOrder([&](const shared_ptr<TreeNode>& node) { nodes.push_back(node); });
    return nodes;
}

void Tree::SetParentFromChildren() {
    PreOrder([](const shared_ptr<TreeNode>& node) {
        for (auto& c : node->children) {
            c->parent = node;
        }
    });
}

void Tree::SetParentFromUndirectedAdjList() {
    PreOrder([](const shared_ptr<TreeNode>& node) {
        for (auto& c : node->children) {
            auto& n = c->children;
            auto it = find(n.begin(), n.end(), node);
            assert(it != n.end());
            *it = n.back();
            n.pop_back();
            c->parent = node;
        }
    });
}

void Tree::Reroot() {
    TreeNode::Reroot(source);
}

void Tree::QuickCheck() {
    int numPin = net->pins.size(), numChecked = 0;
    vector<bool> pinExist(numPin, false);
    PreOrder([&](const shared_ptr<TreeNode>& node) {
        if (!node) {
            cerr << "Error: empty node" << endl;
        }
        if (node->pin) {
            auto id = node->pin->id;
            if (!(id >= 0 && id < numPin && pinExist[id] == false)) {
                cerr << "Error: Steiner node with incorrect id" << endl;
            }
            pinExist[id] = true;
            ++numChecked;
        }
        for (auto& c : node->children) {
            if (!c->parent || c->parent != node) {
                cerr << "Error: inconsistent parent-child relationship" << endl;
            }
        }
    });
    if (numChecked != numPin) {
        cerr << "Error: pin not covered" << endl;
    }
}

void Tree::RemovePhyRedundantSteiner() {
    PostOrderCopy([](const shared_ptr<TreeNode>& node) {
        if (!node->parent || node->loc != node->parent->loc) return;
        if (node->pin) {
            if (node->parent->pin && node->parent->pin != node->pin) return;
            node->parent->pin = node->pin;
        }
        for (auto c : node->children) TreeNode::SetParent(c, node->parent);
        TreeNode::ResetParent(node);
    });
}

void Tree::RemoveTopoRedundantSteiner() {
    PostOrderCopy([](const shared_ptr<TreeNode>& node) {
        // degree may change after post-order traversal of its children
        if (node->pin) return;
        if (node->children.empty()) {
            TreeNode::ResetParent(node);
        } else if (node->children.size() == 1) {
            auto oldParent = node->parent, oldChild = node->children[0];
            TreeNode::ResetParent(node);
            TreeNode::ResetParent(oldChild);
            TreeNode::SetParent(oldChild, oldParent);
        }
    });
}

void Tree::RemoveEmptyChildren() {
    PreOrder([](const shared_ptr<TreeNode>& node) {
        int size = 0;
        for (int i = 0; i < node->children.size(); ++i) {
            if (node->children[i]) node->children[size++] = node->children[i];
        }
        node->children.resize(size);
    });
}

}  // namespace salt