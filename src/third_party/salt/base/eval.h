#pragma once

#include "tree.h"

namespace salt {

class WireLengthEvalBase {
public:
    DTYPE wireLength;

    WireLengthEvalBase() = default;
    void Update(const Tree& tree);
    WireLengthEvalBase(const Tree& tree) { Update(tree); }
};

class WireLengthEval : public WireLengthEvalBase {
public:
    DTYPE maxPathLength;
    double avgPathLength;
    double norPathLength;  // avgPathLength / avgShortestPathLength
    double maxStretch;     // max{pathLength / shortestPathLength}
    double avgStretch;     // avg{pathLength / shortestPathLength}

    WireLengthEval() = default;
    void Update(const Tree& tree);
    WireLengthEval(const Tree& tree) { Update(tree); }
};

inline ostream& operator<<(ostream& os, const WireLengthEval& eval) {
    os << " wl=" << eval.wireLength << " mp=" << eval.maxPathLength << " ap=" << eval.avgPathLength
       << " ms=" << eval.maxStretch << " as=" << eval.avgStretch;
    return os;
}

//********************************************************************************

class ElmoreDelayEval {
public:
    static double unitRes;
    static double unitCap;

    DTYPE fluteWL = -1;
    double maxDelay;
    double avgDelay;
    double maxNorDelay;
    double avgNorDelay;
    ElmoreDelayEval() {}
    void Update(double rd, Tree& tree, bool normalize = true);  // tree node id will be updated
    ElmoreDelayEval(double rd, Tree& tree, bool normalize = true) { Update(rd, tree, normalize); }

private:
    vector<double> GetDelay(double rd, const Tree& tree, int numNode);
    vector<double> GetDelayLB(double rd, const Tree& tree);
};

inline ostream& operator<<(ostream& os, const ElmoreDelayEval& eval) {
    os << " md=" << eval.maxDelay << " ad=" << eval.avgDelay << " mnd=" << eval.maxNorDelay
       << " and=" << eval.avgNorDelay;
    return os;
}

//********************************************************************************

class CompleteEval : public WireLengthEval, public ElmoreDelayEval {
public:
    double norWL;

    CompleteEval() = default;
    void Update(double rd, Tree& tree) {
        WireLengthEval::Update(tree);
        ElmoreDelayEval::Update(rd, tree);
        norWL = double(wireLength) / fluteWL;
    }
    CompleteEval(double rd, Tree& tree) { Update(rd, tree); }
};

class CompleteStat {
public:
    double norWL = 0, maxStretch = 0, avgStretch = 0, norPathLength = 0, maxNorDelay = 0, avgNorDelay = 0;
    int cnt = 0;
    double eps;
    double time = 0;
    void Inc(const CompleteEval& eval, double runtime = 0.0) {
        ++cnt;
        norWL += eval.norWL;
        maxStretch += eval.maxStretch;
        avgStretch += eval.avgStretch;
        norPathLength += eval.norPathLength;
        maxNorDelay += eval.maxNorDelay;
        avgNorDelay += eval.avgNorDelay;
        time += runtime;
    }
    void Avg() {
        norWL /= cnt;
        maxStretch /= cnt;
        avgStretch /= cnt;
        norPathLength /= cnt;
        maxNorDelay /= cnt;
        avgNorDelay /= cnt;
        time /= cnt;
    }
};

}  // namespace salt