#include "eval.h"
#include "flute.h"

#include <algorithm>

namespace salt {

void WireLengthEvalBase::Update(const Tree& tree) {
    wireLength = 0;
    tree.PostOrder([&](const shared_ptr<TreeNode>& node) {
        if (node->parent) {
            wireLength += node->WireToParent();
        }
    });
}

void WireLengthEval::Update(const Tree& tree) {
    // wirelength
    WireLengthEvalBase::Update(tree);
    // path
    vector<DTYPE> pathLength(tree.net->pins.size());
    function<void(const shared_ptr<TreeNode>&, DTYPE)> traverse = [&](const shared_ptr<TreeNode>& node, DTYPE curDist) {
        if (node->pin) pathLength[node->pin->id] = curDist;
        for (auto c : node->children) traverse(c, curDist + c->WireToParent());
    };
    traverse(tree.source, 0);
    maxPathLength = 0;
    double totalPathLength = 0;
    double totalShortestPathLength = 0;
    maxStretch = 0;
    avgStretch = 0;
    for (auto p : tree.net->pins) {
        if (p->IsSink()) {
            DTYPE pl = pathLength[p->id], sp = Dist(tree.source->loc, p->loc);
            double stretch = double(pl) / sp;
            // cout << p->id << " " << stretch << endl;
            if (pl > maxPathLength) maxPathLength = pl;
            totalPathLength += pl;
            totalShortestPathLength += sp;
            if (stretch > maxStretch) maxStretch = stretch;
            avgStretch += stretch;
        }
    }
    auto numSink = tree.net->pins.size() - 1;
    avgPathLength = totalPathLength / numSink;
    norPathLength = totalPathLength / totalShortestPathLength;
    avgStretch /= numSink;
}

//********************************************************************************

double ElmoreDelayEval::unitRes = -1;
double ElmoreDelayEval::unitCap = -1;

void ElmoreDelayEval::Update(double rd, Tree& tree, bool normalize) {
    assert(rd > 0);
    assert(unitRes > 0 && unitCap > 0);
    int numPins = tree.net->pins.size();
    int numNodes = tree.UpdateId();
    maxDelay = avgDelay = maxNorDelay = avgNorDelay = 0;

    auto delay = GetDelay(rd, tree, numNodes);  // delay for all tree nodes
    tree.PreOrder([&](const shared_ptr<TreeNode>& node) {
        if (!node->pin || node == tree.source) return;
        maxDelay = max(maxDelay, delay[node->id]);
        avgDelay += delay[node->id];
    });
    avgDelay /= (numPins - 1);

    if (!normalize) return;

    auto lb = GetDelayLB(rd, tree);  // delay lb for all pins, 0 is source
    // tree.PreOrder([&](const shared_ptr<TreeNode>& node){
    // 	if(!node->pin || node == tree.source) return;
    // 	double norDelay = delay[node->id] / lb[node->id];
    // 	maxNorDelay = max(maxNorDelay, norDelay);
    // 	avgNorDelay += norDelay;
    // });
    // avgNorDelay /= (numPins-1);
    auto maxLb = *max_element(lb.begin(), lb.end());
    maxNorDelay = maxDelay / maxLb;
    avgNorDelay = avgDelay / maxLb;
}

vector<double> ElmoreDelayEval::GetDelay(double rd, const Tree& tree, int numNode) {
    // get node cap by post-order traversal
    vector<double> cap(numNode, 0);
    tree.PostOrder([&](const shared_ptr<TreeNode>& node) {
        if (node->pin && node != tree.source) cap[node->id] = node->pin->cap;
        for (auto c : node->children) {
            cap[node->id] += cap[c->id];
            cap[node->id] += c->WireToParent() * unitCap;
        }
    });

    // get delay by post-order traversal
    vector<double> delay(numNode, 0);
    tree.PreOrder([&](const shared_ptr<TreeNode>& node) {
        if (node == tree.source)
            delay[node->id] = rd * cap[node->id];
        else {
            double dist = node->WireToParent();
            delay[node->id] = dist * unitRes * (0.5 * dist * unitCap + cap[node->id]) + delay[node->parent->id];
        }
    });
    return delay;
}

vector<double> ElmoreDelayEval::GetDelayLB(double rd, const Tree& tree) {
    vector<double> lb(tree.net->pins.size(), 0);

    // call flute and get smt
    Tree flute;
    FluteBuilder fluteB;
    fluteB.Run(*tree.net, flute);
    WireLengthEvalBase wl(flute);
    fluteWL = wl.wireLength;

    double totalcap = 0;
    for (auto p : tree.net->pins) totalcap += p->cap;

    double lb_sd = rd * (fluteWL * unitCap + totalcap);
    for (auto pin : tree.net->pins) {
        if (pin->IsSource()) continue;
        double dist = Dist(tree.source->loc, pin->loc);
        lb[pin->id] = dist * unitRes * (0.5 * dist * unitCap + pin->cap) + lb_sd;
    }

    return lb;
}

}  // namespace salt