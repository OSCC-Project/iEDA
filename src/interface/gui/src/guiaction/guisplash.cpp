#include "guisplash.h"

GuiSplash::GuiSplash(QWidget *parent, const QPixmap &pixmap, Qt::WindowFlags f)
    : QSplashScreen(parent, pixmap, f) {
  load = new QLabel(tr("Initialize...."), this);
  load->move(0, height() - load->height());
  connect(this, &GuiSplash::loading, this, &GuiSplash::loadingSlot);
}
void GuiSplash::loadingSlot(const QString &text) { setlabel(text); }
void GuiSplash::setlabel(const QString &text) {
  load->setText(QString(tr("Initialize....         %1")).arg(text));
  load->adjustSize();
}
