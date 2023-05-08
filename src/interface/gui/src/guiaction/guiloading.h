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
