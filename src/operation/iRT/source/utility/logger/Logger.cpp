#include "Logger.hpp"

namespace irt {

// public

void Logger::initInst()
{
  if (_log_instance == nullptr) {
    _log_instance = new Logger();
  }
}

Logger& Logger::getInst()
{
  if (_log_instance == nullptr) {
    initInst();
  }
  return *_log_instance;
}

void Logger::destroyInst()
{
  if (_log_instance != nullptr) {
    delete _log_instance;
    _log_instance = nullptr;
  }
}

void Logger::openLogFileStream(const std::string& log_file_path)
{
  _log_file_path = log_file_path;
  _log_file = new std::ofstream(_log_file_path);
}

void Logger::closeLogFileStream()
{
  if (_log_file != nullptr) {
    _log_file->close();
    delete _log_file;
  }
}

void Logger::printLogFilePath()
{
  if (!_log_file_path.empty()) {
    info(Loc::current(), "The log file path is '", _log_file_path, "'!");
  }
}

// private

Logger* Logger::_log_instance = nullptr;

}  // namespace irt
