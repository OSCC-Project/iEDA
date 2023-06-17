#include "net.h"

#include <boost/functional/hash.hpp>
#include <fstream>
#include <unordered_set>

namespace salt {

void Net::RanInit(int i, int numPin, DTYPE width, DTYPE height) {
    pins.clear();
    id = i;
    name = "random_pin" + to_string(numPin) + "_id" + to_string(i);
    int k = 0;
    unordered_set<pair<DTYPE, DTYPE>, boost::hash<pair<DTYPE, DTYPE>>> exist;
    int trial = 0;
    while (exist.size() < numPin) {
        DTYPE x = rand() % (int)width, y = rand() % (int)height;  // TODO: better random generator
        if (exist.find({x, y}) == exist.end()) {
            pins.push_back(make_shared<Pin>(x, y, k++));
            exist.insert({x, y});
        }
        if (trial++ > numPin * 2) {
            break;
            log() << "randomly try 2 * #pins times. give up." << endl;
        }
    }
    // log() << "specify " << numPin << ", get " << k << endl;
}

// TODO: use catch to handle istream error
bool Net::Read(istream& is) {
    // header
    string buf, option;
    int numPin = 0;
    while (is >> buf && buf != "Net")
        ;
    if (buf != "Net") return false;
    getline(is, buf);
    istringstream iss(buf);
    iss >> id >> name >> numPin >> option;
    assert(numPin > 0);
    withCap = (option == "-cap");

    // pins
    int i;
    DTYPE x, y;
    double c = 0.0;
    pins.resize(numPin);
    for (auto& pin : pins) {
        is >> i >> x >> y;
        if (withCap) is >> c;
        pin = make_shared<Pin>(x, y, i, c);
    }

    return true;
}

void Net::Read(const string& fileName) {
    ifstream is(fileName);
    if (is.fail()) {
        cout << "ERROR: Cannot open file " << fileName << endl;
        exit(1);
    }
    Read(is);
}

string Net::GetHeader() const {
    string header = to_string(id) + " " + name + " " + to_string(pins.size());
    // string header = name + " " + to_string(pins.size());
    if (withCap) header += " -cap";
    return header;
}

void Net::Write(ostream& os) const {
    // header
    os << "Net " << GetHeader() << endl;

    // pins
    for (const auto& pin : pins) {
        os << pin->id << " " << pin->loc.x << " " << pin->loc.y;
        if (withCap) os << " " << pin->cap;
        os << endl;
    }
}

void Net::Write(const string& prefix, bool withNetInfo) const {
    ofstream os(prefix + (withNetInfo ? ("_" + name) : "") + ".net");
    Write(os);
}

}  // namespace salt