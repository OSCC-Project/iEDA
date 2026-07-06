#include "SpefParser.hh"

#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "log/Log.hh"

int spef_parse(spef::ParserContext* context);
void spef_restart(FILE* input_file);
extern FILE* spef_in;

namespace spef {
namespace {

std::string joinHeaderValues(const std::vector<std::string>& values)
{
  std::string result;
  for (const auto& value : values) {
    if (!result.empty()) {
      result += ' ';
    }
    result += value;
  }
  return result;
}

bool startsWithNameIndex(const std::string& name)
{
  return name.size() >= 2 && name.front() == '*' && std::isdigit(static_cast<unsigned char>(name[1]));
}

}  // namespace

ParserContext::ParserContext(std::string file_name) : exchange_(std::move(file_name)) {}

void ParserContext::setSection(SectionType section)
{
  current_section_ = section;
  if (section == SectionType::kEnd) {
    finishNet();
  }
}

void ParserContext::startHeader(std::string key)
{
  pending_header_key_ = std::move(key);
  pending_header_values_.clear();
}

void ParserContext::addHeaderValue(std::string value)
{
  pending_header_values_.push_back(std::move(value));
}

void ParserContext::finishHeader()
{
  if (pending_header_key_.empty()) {
    return;
  }
  exchange_.header.push_back(HeaderEntry{pending_header_key_, joinHeaderValues(pending_header_values_)});
  pending_header_key_.clear();
  pending_header_values_.clear();
}

void ParserContext::addNameMap(std::string index_name, std::string mapped_name)
{
  const std::size_t index = parseNameIndex(index_name);
  exchange_.index_to_name_map[index] = mapped_name;
  exchange_.name_to_index_map[std::move(mapped_name)] = index;
}

void ParserContext::addPort(std::string name, ConnectionDirection direction, Coord coordinate)
{
  if (startsWithNameIndex(name)) {
    name.erase(0, 1);
  }
  exchange_.ports.push_back(PortEntry{std::move(name), direction, coordinate});
}

void ParserContext::startNet(std::string name, double lcap, std::size_t line_no)
{
  if (has_current_net_) {
    finishNet();
  }
  current_net_ = Net{};
  current_net_.name = std::move(name);
  current_net_.lcap = lcap;
  current_net_.line_no = line_no;
  has_current_net_ = true;
}

void ParserContext::finishNet()
{
  if (!has_current_net_) {
    return;
  }
  exchange_.nets.push_back(std::move(current_net_));
  current_net_ = Net{};
  has_current_net_ = false;
}

void ParserContext::startConn(ConnectionType type, std::string name, ConnectionDirection direction)
{
  current_conn_ = ConnEntry{};
  current_conn_.conn_type = type;
  current_conn_.pin_port_name = std::move(name);
  current_conn_.conn_direction = direction;
  if (type == ConnectionType::kInternal && direction == ConnectionDirection::kUninitialized) {
    current_conn_.conn_direction = ConnectionDirection::kInternal;
  }
  has_current_conn_ = true;
}

void ParserContext::setConnCoordinate(Coord coordinate)
{
  if (has_current_conn_) {
    current_conn_.coordinate = coordinate;
  }
}

void ParserContext::setConnLoad(double load)
{
  if (has_current_conn_) {
    current_conn_.load = load;
  }
}

void ParserContext::setConnDrivingCell(std::string driving_cell)
{
  if (has_current_conn_) {
    current_conn_.driving_cell = std::move(driving_cell);
  }
}

void ParserContext::setConnLowerLeft(Coord coordinate)
{
  if (has_current_conn_) {
    current_conn_.ll_coordinate = coordinate;
  }
}

void ParserContext::setConnUpperRight(Coord coordinate)
{
  if (has_current_conn_) {
    current_conn_.ur_coordinate = coordinate;
  }
}

void ParserContext::setConnLayer(int layer)
{
  if (has_current_conn_) {
    current_conn_.layer = layer;
  }
}

void ParserContext::finishConn()
{
  if (!has_current_net_ || !has_current_conn_) {
    return;
  }
  current_net_.conns.push_back(std::move(current_conn_));
  current_conn_ = ConnEntry{};
  has_current_conn_ = false;
}

void ParserContext::addCap(std::string node1, std::string node2, double cap)
{
  if (has_current_net_) {
    current_net_.caps.push_back(ResCap{std::move(node1), std::move(node2), cap});
  }
}

void ParserContext::addRes(std::string node1, std::string node2, double res)
{
  if (has_current_net_) {
    current_net_.ress.push_back(ResCap{std::move(node1), std::move(node2), res});
  }
}

void ParserContext::addCapOrRes(std::string node1, std::string node2, double value)
{
  if (current_section_ == SectionType::kCap) {
    addCap(std::move(node1), std::move(node2), value);
  } else if (current_section_ == SectionType::kRes) {
    addRes(std::move(node1), std::move(node2), value);
  }
}

void ParserContext::setError(std::string message)
{
  if (error_message_.empty()) {
    error_message_ = std::move(message);
  }
}

std::size_t ParserContext::parseNameIndex(const std::string& index_name)
{
  const std::size_t begin = index_name.front() == '*' ? 1 : 0;
  const std::size_t end = index_name.find(':', begin);
  return static_cast<std::size_t>(std::strtoull(index_name.substr(begin, end - begin).c_str(), nullptr, 10));
}

double toDouble(const char* text)
{
  if (text == nullptr) {
    return 0.0;
  }
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(text, &end);
  if (errno != 0 || end == text) {
    return 0.0;
  }
  return value;
}

int toInt(const char* text)
{
  if (text == nullptr) {
    return 0;
  }
  return static_cast<int>(std::strtol(text, nullptr, 10));
}

std::string tokenToString(const char* text)
{
  return text == nullptr ? std::string{} : std::string{text};
}

std::string stripQuotes(std::string text)
{
  if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
    return text.substr(1, text.size() - 2);
  }
  return text;
}

ConnectionDirection parseDirection(const char* text)
{
  if (text == nullptr) {
    return ConnectionDirection::kUninitialized;
  }
  if (std::strcmp(text, "I") == 0) {
    return ConnectionDirection::kInput;
  }
  if (std::strcmp(text, "O") == 0) {
    return ConnectionDirection::kOutput;
  }
  if (std::strcmp(text, "B") == 0) {
    return ConnectionDirection::kInout;
  }
  return ConnectionDirection::kUninitialized;
}

ConnectionType parseConnectionType(const char* text)
{
  if (text == nullptr) {
    return ConnectionType::kUninitialized;
  }
  if (std::strcmp(text, "*P") == 0) {
    return ConnectionType::kExternal;
  }
  if (std::strcmp(text, "*I") == 0 || std::strcmp(text, "*N") == 0) {
    return ConnectionType::kInternal;
  }
  return ConnectionType::kInternal;
}

std::string removeEscapes(const std::string& name)
{
  std::string result;
  result.reserve(name.size());
  for (char ch : name) {
    if (ch != '\\') {
      result.push_back(ch);
    }
  }
  return result;
}

std::string expandName(const Exchange& exchange, const std::string& name)
{
  if (!startsWithNameIndex(name)) {
    return name;
  }

  const std::size_t begin = 1;
  const std::size_t colon = name.find(':', begin);
  const std::size_t index = static_cast<std::size_t>(std::strtoull(name.substr(begin, colon - begin).c_str(), nullptr, 10));
  const auto map_it = exchange.index_to_name_map.find(index);
  if (map_it == exchange.index_to_name_map.end()) {
    return name;
  }

  std::string expanded = removeEscapes(map_it->second);
  if (colon != std::string::npos) {
    expanded += name.substr(colon);
  }
  return expanded;
}

void expandAllNames(Exchange& exchange)
{
  if (exchange.index_to_name_map.empty()) {
    return;
  }

  for (auto& net : exchange.nets) {
    const std::string current_net_name = net.name;
    net.name = expandName(exchange, net.name);
    for (auto& conn : net.conns) {
      conn.pin_port_name = expandName(exchange, conn.pin_port_name);
    }
    for (auto& cap : net.caps) {
      cap.node1 = expandName(exchange, cap.node1);
      if (!cap.node2.empty()) {
        cap.node2 = expandName(exchange, cap.node2);
      } else if (cap.node1 == current_net_name) {
        cap.node1 = net.name;
      }
    }
    for (auto& res : net.ress) {
      res.node1 = expandName(exchange, res.node1);
      if (!res.node2.empty()) {
        res.node2 = expandName(exchange, res.node2);
      }
    }
  }
}

Exchange* parseSpefFile(const char* spef_path)
{
  if (spef_path == nullptr) {
    return nullptr;
  }

  FILE* file = std::fopen(spef_path, "r");
  if (file == nullptr) {
    LOG_ERROR << "open spef file failed: " << spef_path;
    return nullptr;
  }

  ParserContext context(spef_path);
  spef_in = file;
  spef_restart(file);
  const int parse_status = spef_parse(&context);
  std::fclose(file);
  spef_in = nullptr;

  if (parse_status != 0 || !context.ok()) {
    LOG_ERROR << "parse spef file failed: " << spef_path << " " << context.errorMessage();
    return nullptr;
  }

  context.finishNet();
  return new Exchange(std::move(context.exchange()));
}

std::string getSpefCapUnit(const Exchange& exchange)
{
  for (const auto& header : exchange.header) {
    if (header.key == "*C_UNIT") {
      return header.value;
    }
  }
  return "";
}

std::string getSpefResUnit(const Exchange& exchange)
{
  for (const auto& header : exchange.header) {
    if (header.key == "*R_UNIT") {
      return header.value;
    }
  }
  return "";
}

bool SpefReader::read(const std::string& file_path)
{
  spef_file_.reset(parseSpefFile(file_path.c_str()));
  return spef_file_ != nullptr;
}

void SpefReader::expandName()
{
  if (spef_file_ != nullptr) {
    expandAllNames(*spef_file_);
  }
}

std::string SpefReader::getSpefCapUnit() const
{
  return spef_file_ == nullptr ? std::string{} : spef::getSpefCapUnit(*spef_file_);
}

std::string SpefReader::getSpefResUnit() const
{
  return spef_file_ == nullptr ? std::string{} : spef::getSpefResUnit(*spef_file_);
}

}  // namespace spef
