#include "Reporter.h"

namespace ino {

/**
 * @brief report start or end time
 *
 * @param begin
 * true: start time.   false: end time.
 */
void Reporter::reportTime(bool begin) {
  _outfile.open(_output_path, std::ios::app);
  time_t timep;
  time(&timep);
  char tmp[256];
  strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&timep));

  if (begin) {
    _outfile << "\n\n======================= Program start time " << tmp
             << "=======================" << std::endl;
  } else {
    _outfile << "======================= Program end time " << tmp
             << "=======================" << std::endl;
  }
  _outfile.close();
}

void Reporter::report(string info) {
  _outfile.open(_output_path, std::ios::app);
  _outfile << info << std::endl;
  _outfile.close();
}
} // namespace ino
