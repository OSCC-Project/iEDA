#pragma once

#include <string>
#include <vector>

namespace ista {
class TimingEngine;
class TimingIDBAdapter;
class TimingDBAdapter;
class DesignObject;
class RctNode;
class Net;
class Pin;
class Instance;
class StaVertex;
class LibertyCell;
class LibertyPort;
class LibertyArc;
class StaSeqPathData;
enum class AnalysisMode;
enum class TransType;
} // namespace ista

namespace idb {
class IdbPin;
class IdbPins;
class IdbInstance;
class IdbInstanceList;
class IdbNet;
class IdbBuilder;
class IdbDesign;
class IdbCellMaster;
class IdbBlockage;
class IdbRow;
class IdbLayout;
class IdbCellMasterList;
} // namespace idb

namespace ino {
class NoConfig;
class iNO;
}