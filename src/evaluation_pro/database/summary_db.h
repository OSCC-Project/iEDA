/*
 * @FilePath: summary_db.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-30 12:30:58
 * @Description:
 */

#include "congestion_db.h"
#include "density_db.h"
#include "timing_db.hh"
#include "wirelength_db.h"

namespace ieda_eval {

class SummaryDB
{
 public:
  SummaryDB();
  ~SummaryDB();
};

}  // namespace ieda_eval