#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace spef {

enum class SectionType
{
  kHeader,
  kNameMap,
  kPorts,
  kConn,
  kCap,
  kRes,
  kEnd
};

enum class ConnectionDirection
{
  kInput,
  kOutput,
  kInout,
  kInternal,
  kUninitialized
};

enum class ConnectionType
{
  kInternal,
  kExternal,
  kUninitialized
};

struct Coord
{
  double x = -1.0;
  double y = -1.0;
};

struct HeaderEntry
{
  std::string key;
  std::string value;
};

struct PortEntry
{
  std::string name;
  ConnectionDirection direction = ConnectionDirection::kUninitialized;
  Coord coordinate;
};

struct ConnEntry
{
  ConnectionType conn_type = ConnectionType::kUninitialized;
  ConnectionDirection conn_direction = ConnectionDirection::kUninitialized;
  std::string pin_port_name;
  std::string driving_cell;
  double load = 0.0;
  int layer = 0;
  Coord coordinate;
  Coord ll_coordinate;
  Coord ur_coordinate;
};

struct ResCap
{
  std::string node1;
  std::string node2;
  double res_or_cap = 0.0;
};

struct Net
{
  std::string name;
  std::size_t line_no = 0;
  double lcap = 0.0;
  std::vector<ConnEntry> conns;
  std::vector<ResCap> caps;
  std::vector<ResCap> ress;
};

struct Exchange
{
  explicit Exchange(std::string file_name) : file_name(std::move(file_name)) {}

  std::string file_name;
  std::vector<HeaderEntry> header;
  std::unordered_map<std::size_t, std::string> index_to_name_map;
  std::unordered_map<std::string, std::size_t> name_to_index_map;
  std::vector<PortEntry> ports;
  std::vector<Net> nets;
};

#define FOREACH_SPEF_NET(spef_file, spef_net) \
  for (const auto& spef_net : (spef_file)->nets)

#define FOREACH_SPEF_CONN(spef_net, conn) \
  for (const auto& conn : (spef_net).conns)

#define FOREACH_SPEF_CAP(spef_net, cap) \
  for (const auto& cap : (spef_net).caps)

#define FOREACH_SPEF_RES(spef_net, res) \
  for (const auto& res : (spef_net).ress)

class ParserContext
{
 public:
  explicit ParserContext(std::string file_name);

  Exchange& exchange() { return exchange_; }
  const Exchange& exchange() const { return exchange_; }

  void setSection(SectionType section);
  SectionType section() const { return current_section_; }

  void startHeader(std::string key);
  void addHeaderValue(std::string value);
  void finishHeader();

  void addNameMap(std::string index_name, std::string mapped_name);
  void addPort(std::string name, ConnectionDirection direction, Coord coordinate);

  void startNet(std::string name, double lcap, std::size_t line_no);
  void finishNet();

  void startConn(ConnectionType type, std::string name, ConnectionDirection direction);
  void setConnCoordinate(Coord coordinate);
  void setConnLoad(double load);
  void setConnDrivingCell(std::string driving_cell);
  void setConnLowerLeft(Coord coordinate);
  void setConnUpperRight(Coord coordinate);
  void setConnLayer(int layer);
  void finishConn();

  void addCap(std::string node1, std::string node2, double cap);
  void addRes(std::string node1, std::string node2, double res);
  void addCapOrRes(std::string node1, std::string node2, double value);

  void setError(std::string message);
  bool ok() const { return error_message_.empty(); }
  const std::string& errorMessage() const { return error_message_; }

 private:
  static std::size_t parseNameIndex(const std::string& index_name);

  Exchange exchange_;
  SectionType current_section_ = SectionType::kHeader;
  Net current_net_;
  bool has_current_net_ = false;
  ConnEntry current_conn_;
  bool has_current_conn_ = false;
  std::string pending_header_key_;
  std::vector<std::string> pending_header_values_;
  std::string error_message_;
};

double toDouble(const char* text);
int toInt(const char* text);
std::string tokenToString(const char* text);
std::string stripQuotes(std::string text);
ConnectionDirection parseDirection(const char* text);
ConnectionType parseConnectionType(const char* text);
void expandAllNames(Exchange& exchange);
std::string expandName(const Exchange& exchange, const std::string& name);
std::string removeEscapes(const std::string& name);

Exchange* parseSpefFile(const char* spef_path);
std::string getSpefCapUnit(const Exchange& exchange);
std::string getSpefResUnit(const Exchange& exchange);

class SpefReader
{
 public:
  SpefReader() = default;
  ~SpefReader() = default;

  bool read(const std::string& file_path);

  Exchange* getSpefFile() { return spef_file_.get(); }
  const Exchange* getSpefFile() const { return spef_file_.get(); }

  void expandName();
  std::string getSpefCapUnit() const;
  std::string getSpefResUnit() const;

 private:
  std::unique_ptr<Exchange> spef_file_;
};

}  // namespace spef
