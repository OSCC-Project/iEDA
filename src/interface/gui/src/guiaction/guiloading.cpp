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
#include "guiloading.h"

GuiLoading::GuiLoading(QMovie *gif, QWidget *parent) : QDialog(parent) {
  // setAttribute(Qt::WA_DeleteOnClose);
  setFixedSize(600, 500);
  setWindowFlags(Qt::FramelessWindowHint);
  setWindowModality(Qt::ApplicationModal);
  QLabel *lab_gif = new QLabel(this);
  lab_gif->setMovie(gif);
  lab_gif->setScaledContents(true);
  lab_gif->resize(this->size());
  lab_gif->setFrameStyle(QFrame::Panel | QFrame::Raised);

  gif->start();
}
LoadingThread::LoadingThread(QMovie *gif, QWidget *parent) {
  load = new GuiLoading(gif, parent);
}
void LoadingThread::run() {
  load->show();
  qDebug() << "isshow";
}
void LoadingThread::isDone() {
  load->close();
  delete load;
  this->quit();
}
