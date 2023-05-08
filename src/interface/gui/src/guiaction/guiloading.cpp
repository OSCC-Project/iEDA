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
