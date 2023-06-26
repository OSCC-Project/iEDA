#include "rsa.h"

#include <unordered_map>
#include <unordered_set>

namespace salt {

constexpr double PI_VALUE = 3.14159265358979323846; /* pi */

void RsaBase::ReplaceRootChildren(Tree& tree) {
    const Net* oldNet = tree.net;

    // create tmpNet and fakePins
    Net tmpNet = *oldNet;
    tmpNet.pins.clear();
    unordered_map<shared_ptr<Pin>, shared_ptr<TreeNode>> pinToOldNode;
    tmpNet.pins.push_back(tree.source->pin);
    unordered_set<shared_ptr<Pin>> fakePins;
    pinToOldNode[tree.source->pin] = tree.source;
    for (auto c : tree.source->children) {  // only contains the direct children of tree.source
        shared_ptr<Pin> pin = c->pin;
        if (!pin) {
            pin = make_shared<Pin>(c->loc);
            fakePins.insert(pin);
        }
        tmpNet.pins.push_back(pin);
        pinToOldNode[pin] = c;
    }
    tree.source->children.clear();  // free them...

    // get rsa and graft the old subtrees to it
    Run(tmpNet, tree);
    tree.PostOrder([&](const shared_ptr<TreeNode>& node) {
        if (node->pin) {
            auto oldNode = pinToOldNode[node->pin];
            if (fakePins.find(node->pin) != fakePins.end()) node->pin = nullptr;
            if (node->parent)
                for (auto c : oldNode->children) TreeNode::SetParent(c, node);
        }
    });
    tree.net = oldNet;
    tree.RemoveTopoRedundantSteiner();
}

DTYPE RsaBase::MaxOvlp(DTYPE z1, DTYPE z2) {
    if (z1 >= 0 && z2 >= 0)
        return min(z1, z2);
    else if (z1 <= 0 && z2 <= 0)
        return max(z1, z2);
    else
        return 0;
}

void RsaBuilder::Run(const Net& net, Tree& tree) {
    // Shift all pins to make source (0,0)
    auto oriSrcLoc = net.source()->loc;
    for (auto& p : net.pins) p->loc -= oriSrcLoc;

    // Init innerNodes with all sinks
    for (auto p : net.pins)
        if (p->IsSink()) innerNodes.insert(new InnerNode(make_shared<TreeNode>(p)));

    // Process a inner node in each iteration
    while (!innerNodes.empty()) {
        if ((*innerNodes.begin())->dist == 0) break;  // TODO: clear
        shared_ptr<TreeNode> node = (*innerNodes.begin())->tn;
        auto forDelete = *innerNodes.begin();
        innerNodes.erase(innerNodes.begin());
        if (!node->pin) {  // steiner node
            assert(node->children.size() == 2);
            if (node->children[0]) RemoveAnOuterNode(node->children[0], true, false);
            if (node->children[1]) RemoveAnOuterNode(node->children[1], false, true);
            node->children[0]->parent = node;
            node->children[1]->parent = node;
        } else {  // pin node
            TryDominating(node);
        }
        delete forDelete;
        AddAnOuterNode(node);
    }

    // connet the remaining outerNodes to the source
    tree.source = make_shared<TreeNode>(net.source());
    for (const auto& on : outerNodes) TreeNode::SetParent(on.second.cur, tree.source);
    tree.net = &net;

    // shift all pins back
    for (auto& p : net.pins) p->loc += oriSrcLoc;
    tree.PreOrder([&](const shared_ptr<TreeNode>& node) { node->loc += oriSrcLoc; });

    // clear
    for (auto in : innerNodes) delete in;
    innerNodes.clear();
    outerNodes.clear();
}

map<double, OuterNode>::iterator RsaBuilder::NextOuterNode(const map<double, OuterNode>::iterator& it) {
    auto res = next(it);
    if (res != outerNodes.end())
        return res;
    else
        return outerNodes.begin();
}

map<double, OuterNode>::iterator RsaBuilder::PrevOuterNode(const map<double, OuterNode>::iterator& it) {
    if (it != outerNodes.begin())
        return prev(it);
    else
        return prev(outerNodes.end());
}

bool RsaBuilder::TryMaxOvlpSteinerNode(OuterNode& left, OuterNode& right) {
    double rlAng = atan2(right.cur->loc.y, right.cur->loc.x) - atan2(left.cur->loc.y, left.cur->loc.x);
    // if (rlAng>-PI_VALUE && rlAng<0) return false; // there is smaller arc
    DTYPE newX, newY;
    if ((rlAng > -PI_VALUE && rlAng < 0) || rlAng > PI_VALUE) {  // large arc
        newX = 0;
        newY = 0;
    } else {
        newX = MaxOvlp(left.cur->loc.x, right.cur->loc.x);
        newY = MaxOvlp(left.cur->loc.y, right.cur->loc.y);
    }
    // if (newX==0 && newY==0) return false; // non-neighboring quadrant
    auto tn = make_shared<TreeNode>(newX, newY);
    tn->children = {left.cur, right.cur};
    auto in = new InnerNode(tn);
    left.rightP = in;
    right.leftP = in;
    innerNodes.insert(in);
    // cout << "add a tmp steiner point" << endl;
    // tn->Print(0,true);
    return true;
}

void RsaBuilder::RemoveAnOuterNode(const shared_ptr<TreeNode>& node, bool delL, bool delR) {
    auto outerCur = outerNodes.find(OuterNodeKey(node));
    assert(outerCur != outerNodes.end());
    InnerNode *innerL = outerCur->second.leftP, *innerR = outerCur->second.rightP;
    auto outerL = outerNodes.end(), outerR = outerNodes.end();
    if (innerL != nullptr) {
        outerL = PrevOuterNode(outerCur);
        assert(outerL->second.cur == innerL->tn->children[0]);
        assert(outerCur->second.cur == innerL->tn->children[1]);
        innerNodes.erase(innerL);
        if (delL) delete innerL;  // inner parent become invalid now
        outerL->second.rightP = nullptr;
    }
    if (innerR != nullptr) {
        outerR = NextOuterNode(outerCur);
        assert(outerCur->second.cur == innerR->tn->children[0]);
        assert(outerR->second.cur == innerR->tn->children[1]);
        innerNodes.erase(innerR);
        if (delR) delete innerR;  // inner parent become invalid now
        outerR->second.leftP = nullptr;
    }
    // delete outerCur->second.first; //  outer child should be kept
    outerNodes.erase(outerCur);
    if (delL && delR && outerR != outerNodes.end() && outerL != outerNodes.end() && outerL != outerR) {
        TryMaxOvlpSteinerNode(outerL->second, outerR->second);
    }
}

// p dominates c
inline bool Dominate(const Point& p, const Point& c) {
    return ((p.x >= 0 && c.x >= 0 && p.x <= c.x) || (p.x <= 0 && c.x <= 0 && p.x >= c.x))      // x
           && ((p.y >= 0 && c.y >= 0 && p.y <= c.y) || (p.y <= 0 && c.y <= 0 && p.y >= c.y));  // y
}

bool RsaBuilder::TryDominatingOneSide(OuterNode& p, OuterNode& c) {
    if (!Dominate(p.cur->loc, c.cur->loc)) return false;
    TreeNode::SetParent(c.cur, p.cur);
    RemoveAnOuterNode(c.cur);
    return true;
}

void RsaBuilder::TryDominating(const shared_ptr<TreeNode>& node) {
    OuterNode outerCur(node);
    if (outerNodes.empty())
        return;
    else if (outerNodes.size() == 1) {
        TryDominatingOneSide(outerCur, outerNodes.begin()->second);
        return;
    }
    // get outerR & outerL
    auto outerR = outerNodes.upper_bound(OuterNodeKey(node));
    if (outerR == outerNodes.end()) outerR = outerNodes.begin();
    auto outerL = PrevOuterNode(outerR);
    assert(outerL != outerNodes.end() && outerR != outerNodes.end());
    assert(outerL->second.rightP == outerR->second.leftP);
    // try dominating twice
    TryDominatingOneSide(outerCur, outerL->second);
    TryDominatingOneSide(outerCur, outerR->second);
}

// suppose no case of [node = min(node, an outer node)]
void RsaBuilder::AddAnOuterNode(const shared_ptr<TreeNode>& node) {
    OuterNode outerCur(node);
    if (!outerNodes.empty()) {
        // get outerR & outerL
        auto outerR = outerNodes.upper_bound(OuterNodeKey(node));
        if (outerR == outerNodes.end()) outerR = outerNodes.begin();
        auto outerL = PrevOuterNode(outerR);
        assert(outerL != outerNodes.end() && outerR != outerNodes.end());
        assert(outerL->second.rightP == outerR->second.leftP);
        // delete parent(outerR, outerL)
        if (outerL->second.rightP) {
            innerNodes.erase(outerL->second.rightP);
            delete outerL->second.rightP;  // inner parent become invalid now
        }
        // add two parents
        TryMaxOvlpSteinerNode(outerL->second, outerCur);
        TryMaxOvlpSteinerNode(outerCur, outerR->second);
    }
    outerNodes.insert({OuterNodeKey(node), outerCur});
}

void RsaBuilder::PrintInnerNodes() {
    cout << "Inner nodes (# = " << innerNodes.size() << ")" << endl;
    for (auto in : innerNodes) {
        cout << in->tn;
    }
}

void RsaBuilder::PrintOuterNodes() {
    cout << "Outer nodes (# = " << outerNodes.size() << ")" << endl;
    for (auto on : outerNodes) {
        cout << on.second.cur;
    }
}

}  // namespace salt