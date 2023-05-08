#ifndef GUISPLASH_H
#define GUISPLASH_H

#include <QLabel>
#include <QSplashScreen>

class GuiSplash : public QSplashScreen {
  Q_OBJECT
 public:
  explicit GuiSplash(QWidget *parent = nullptr,
                     const QPixmap &pixmap = QPixmap(),
                     Qt::WindowFlags f = Qt::WindowFlags());
  void loadingSlot(const QString &text);

 signals:
  void loading(const QString &text);

 private:
  QLabel *load;
  void setlabel(const QString &text);
};

#endif  // GUISPLASH_H
