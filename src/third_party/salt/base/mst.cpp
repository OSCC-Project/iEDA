#include "mst.h"

#include <queue>

namespace salt {

void MstBuilder::Run(const Net& net, Tree& tree) {
    vector<vector<int>> adjLists;
    GetSpanGraph(net, adjLists);
    RunPrimAlg(net, adjLists, tree);
}

void MstBuilder::GetAllNearestNeighbors(const vector<Point>& points, vector<vector<int>>& adjLists) {
    vector<pair<Point, int>> pointIdxes(points.size());
    for (int i = 0; i < points.size(); ++i) {
        pointIdxes[i] = make_pair(points[i], i);
    }
    adjLists.assign(points.size(), {});
    MstBuilder::GetNearestNeighbors(8, pointIdxes, [&](int i, int j) {
        adjLists[i].push_back(j);
    });
}

void MstBuilder::GetSpanGraph(const Net& net, vector<vector<int>>& adjLists) {
    vector<pair<Point, int>> points(net.pins.size());
    for (int i = 0; i < net.pins.size(); ++i) {
        points[i] = make_pair(net.pins[i]->loc, i);
    }
    adjLists.assign(points.size(), {});
    MstBuilder::GetNearestNeighbors(4, points, [&](int i, int j) {
        adjLists[i].push_back(j);
        adjLists[j].push_back(i);
    });
}

void MstBuilder::RunPrimAlg(const Net& net, const vector<vector<int>>& adjLists, Tree& tree) {
    // init visited flags, costs, prefixes, and heap
    int numPins = net.pins.size();
    vector<char> visited(numPins, false);
    vector<DTYPE> costs(numPins, numeric_limits<DTYPE>::max());
    vector<int> prefixes(numPins, -1);
    auto heapCmp = [](const pair<DTYPE, int>& lhs, const pair<DTYPE, int>& rhs) { return lhs.first > rhs.first; };
    priority_queue<pair<DTYPE, int>, vector<pair<DTYPE, int>>, decltype(heapCmp)> minHeap(heapCmp);

    // loop start from the source (0)
    minHeap.emplace(0, 0);
    costs[0] = 0;
    while (!minHeap.empty()) {
        int cur = minHeap.top().second;
        minHeap.pop();
        if (!visited[cur]) {  // lazy deletion
            for (int adj : adjLists[cur]) {
                DTYPE cost = Dist(net.pins[cur]->loc, net.pins[adj]->loc);
                if (!visited[adj] && cost < costs[adj]) {
                    costs[adj] = cost;
                    prefixes[adj] = cur;
                    minHeap.emplace(cost, adj);
                }
            }
            visited[cur] = true;
        }
    }

    // convert to salt::Tree
    vector<shared_ptr<TreeNode>> nodes(numPins);
    for (int i = 0; i < numPins; ++i) {
        nodes[i] = make_shared<TreeNode>(net.pins[i]);
    }
    for (int i = 0; i < numPins; ++i) {
        if (prefixes[i] >= 0) {
            TreeNode::SetParent(nodes[i], nodes[prefixes[i]]);
        }
    }
    tree.source = nodes[0];
    tree.net = &net;
}

void MstBuilder::GetNearestNeighbors(int maxOctant, vector<pair<Point, int>>& points, function<void(int, int)> handle) {
    for (int i = 0; i < maxOctant; ++i) {
        if (i % 2 == 0) {  // octant 1, 3, 5, ...
            GetFirstOctantNearestNeighbors(points, handle);
        } else {  // octant 2, 4, 6, ...
            // mirror wrt y = x
            for (auto& point : points) {
                point.first = {point.first.y, point.first.x};
            }
            GetFirstOctantNearestNeighbors(points, handle);
            // mirror wrt y = x
            // rotate counter-clockwise by 90 degree
            for (auto& point : points) {
                point.first = {point.first.y, point.first.x};
                point.first = {-point.first.y, point.first.x};
            }
        }
    }
}

void MstBuilder::GetFirstOctantNearestNeighbors(vector<pair<Point, int>>& points, function<void(int, int)> handle) {
    sort(points.begin(), points.end(), [](const pair<Point, int>& lhs, const pair<Point, int>& rhs) {
        return (lhs.first.x + lhs.first.y) < (rhs.first.x + rhs.first.y);
    });
    auto activeSetCmp = [](const pair<Point, int>& lhs, const pair<Point, int>& rhs) {
        return lhs.first.x < rhs.first.x;
    };
    set<pair<Point, int>, decltype(activeSetCmp)> activeSet(activeSetCmp);
    for (const auto& point : points) {
        auto endIter = activeSet.upper_bound(point);
        auto beginIter = endIter;
        if (beginIter != activeSet.begin()) --beginIter;
        while (beginIter != activeSet.begin() &&
               ((*beginIter).first.x - (*beginIter).first.y) >= (point.first.x - point.first.y)) {
            --beginIter;
        }
        if (beginIter != activeSet.end() &&
            ((*beginIter).first.x - (*beginIter).first.y) < (point.first.x - point.first.y)) {
            ++beginIter;
        }
        for (auto it = beginIter; it != endIter; ++it) {
            handle(it->second, point.second);
        }
        activeSet.erase(beginIter, endIter);
        activeSet.insert(point);
    }
}

}  // namespace salt