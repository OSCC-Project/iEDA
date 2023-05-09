// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#ifndef GUILOADING_H
#define GUILOADING_H

#include <QThread>
#include <QtWidgets>
class GuiLoading : public QDialog {
  Q_OBJECT
 public:
  explicit GuiLoading(QMovie *gif, QWidget *parent = nullptr);

 signals:
};
class LoadingThread : public QThread {
  Q_OBJECT

 public:
  explicit LoadingThread(QMovie *gif, QWidget *parent = nullptr);
  //    ~LoadingThread();

 protected:
  void run();

 signals:

 public slots:
  void isDone();  //处理完成信号

 private:
  GuiLoading *load;
};

#endif  // GUILOADING_H
