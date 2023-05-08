#ifndef COMPLETELINEEDIT_H
#define COMPLETELINEEDIT_H
#include <QDebug>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QListView>
#include <QStringList>
#include <QStringListModel>

class GuiSearchEdit : public QLineEdit {
  Q_OBJECT
 public:
  GuiSearchEdit(QWidget *parent = 0);
  ~GuiSearchEdit();
  QStringList get_names() { return _names; }
  void set_name_list(QStringList list) { _names = list; }

 public slots:
  void setCompleter(const QString &text);
  void completeText(const QModelIndex &index);

 signals:
  void search();

 protected:
  virtual void keyPressEvent(QKeyEvent *e);
  virtual void focusOutEvent(QFocusEvent *e);

 private slots:
  void replyMoveSignal();

 private:
  QListView *_list_view;
  QStringList _names;
  QStringListModel *_model;
};
#endif  // COMPLETELINEEDIT_H
