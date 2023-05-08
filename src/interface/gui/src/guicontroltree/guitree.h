
#ifndef GUITREE_H
#define GUITREE_H

#include <QMouseEvent>
#include <QTreeWidget>

#include "../guiDB/dbsetup.h"
#include "guiConfig.h"
#include "guiConfigTree.h"
#include "guiattribute.h"

class GuiTree : public QTreeWidget {
  Q_OBJECT
 public:
  GuiTree(QWidget* parent = nullptr);
  virtual ~GuiTree() = default;

  void updateLayer();
  void setDbSetup(DbSetup* db_setup) { _db_setup = db_setup; }

 public slots:
  void onItemClicked(QTreeWidgetItem* item, int column);

 protected:
  //   void mousePressEvent(QMouseEvent* event) override;

 private:
  DbSetup* _db_setup = nullptr;

  void initHeader();

  void initByTreeNode(igui::GuiTreeNode& tree_node);

  void updateTreeNode(igui::GuiTreeNode& tree_node);
};

#endif  // GUITREE_H
