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
#include "fileimport.h"

#include <QDebug>
#include <iostream>

FileImport::FileImport(QWidget* parent) : QDialog(parent) { init(); }
void FileImport::init() {
  setWindowIcon(QIcon(":/icon/FileImport.png"));
  setWindowTitle(tr("FileImport File"));
  setWindowFlags(Qt::Window);
  setWindowModality(Qt::ApplicationModal);
  resize(900, 800);

  initDirLayout();
  initTechLayout();
  initListLayout();
  initListItem();
  initControl();

  _topLayout = new QVBoxLayout;
  _topLayout->addLayout(_layout_directory);
  _topLayout->addLayout(_layout_tech);
  _topLayout->addLayout(_layout_list);
  _topLayout->addLayout(_control);

  setLayout(_topLayout);
}

FileImport::~FileImport() {
  if (_edit_directory != nullptr) {
    delete _edit_directory;
    _edit_directory = nullptr;
  };

  if (_tech_directory != nullptr) {
    delete _tech_directory;
    _tech_directory = nullptr;
  };

  if (_layout_tech != nullptr) {
    delete _layout_tech;
    _layout_tech = nullptr;
  };

  if (_btn_tech != nullptr) {
    delete _btn_tech;
    _btn_tech = nullptr;
  };

  if (_select_layout != nullptr) {
    delete _select_layout;
    _select_layout = nullptr;
  };

  if (_btn_select_all != nullptr) {
    delete _btn_select_all;
    _btn_select_all = nullptr;
  };

  if (_btn_open != nullptr) {
    delete _btn_open;
    _btn_open = nullptr;
  };

  if (_control != nullptr) {
    delete _control;
    _control = nullptr;
  };

  if (_list_def != nullptr) {
    delete _list_def;
    _list_def = nullptr;
  };

  if (_list_lef != nullptr) {
    delete _list_lef;
    _list_lef = nullptr;
  };

  if (_file_list != nullptr) {
    delete _file_list;
    _file_list = nullptr;
  };

  if (_layout_list != nullptr) {
    delete _layout_list;
    _layout_list = nullptr;
  };

  if (_topLayout != nullptr) {
    delete _topLayout;
    _topLayout = nullptr;
  };
}

void FileImport::initDirLayout() {
  _edit_directory = new QLineEdit;
  _btn_open       = new QPushButton(tr("···"));
  _btn_open->setFixedSize(QSize(80, 20));
  connect(_btn_open, &QPushButton::clicked, this, &FileImport::openFile);
  _layout_directory = new QHBoxLayout;
  _layout_directory->addWidget(_edit_directory);
  _layout_directory->addWidget(_btn_open);
}

void FileImport::initTechLayout() {
  _tech_directory = new QLineEdit;
  _btn_tech       = new QPushButton(tr("Tech File"));
  _btn_tech->setFixedSize(QSize(80, 20));
  connect(_btn_tech, &QPushButton::clicked, this, &FileImport::openTechFile);
  _layout_tech = new QHBoxLayout;
  _layout_tech->addWidget(_tech_directory);
  _layout_tech->addWidget(_btn_tech);
}

void FileImport::initListLayout() {
  _select_layout  = new QHBoxLayout;
  _btn_select_all = new QPushButton(tr("All Select"));
  connect(_btn_select_all, &QPushButton::clicked, this, &FileImport::allSelect);
  _select_layout->addStretch();
  _select_layout->addWidget(_btn_select_all);

  _file_list = new QListWidget;
  // _file_list -> setSelectionMode(QAbstractItemView::MultiSelection);
  _file_list->setStyleSheet("QListWidget::Item{padding-top:-7px; padding-bottom:-7px; }");
  // connect(_file_list,&QListWidget::itemEntered,this,&FileImport::setItemChecked);
  connect(_file_list, &QListWidget::itemClicked, this, &FileImport::setItemChecked);

  _layout_list = new QVBoxLayout;
  _layout_list->addLayout(_select_layout);
  _layout_list->addWidget(_file_list);
}
void FileImport::initListItem() {
  _list_def = new QListWidgetItem(tr("DEF File"), _file_list);
  _list_def->setSizeHint(QSize(32, 32));
  _list_def->setFont(QFont("Arial", 10, 75, false));
  _list_def->setIcon(QIcon(":/icon/folder.png"));
  _list_def->setBackground(QColor(Qt::lightGray));
  _def_index = 1;

  _list_lef = new QListWidgetItem(tr("LEF File"), _file_list);
  _list_lef->setSizeHint(QSize(32, 32));
  _list_lef->setFont(QFont("Arial", 10, 75, false));
  _list_lef->setIcon(QIcon(":/icon/folder.png"));
  _list_lef->setBackground(QColor(Qt::lightGray));
  _lef_index = 2;
}

void FileImport::clearListItem() { _file_list->clear(); }

void FileImport::initControl() {
  _ok      = new QPushButton(tr("OK"));
  _cancel  = new QPushButton(tr("Cancel"));
  _control = new QHBoxLayout;
  _control->addStretch();
  _control->addWidget(_ok);
  _control->addStretch();
  _control->addWidget(_cancel);
  _control->addStretch();
  connect(_ok, &QPushButton::clicked, this, &FileImport::okAct);
  connect(_cancel, &QPushButton::clicked, this, &FileImport::cancelAct);
}
void FileImport::openFile() {
  QString directory = QFileDialog::getExistingDirectory(this, tr("FileImport"), "../");
  if (directory.isEmpty())
    return;

  _directory = directory;
  _edit_directory->setText(_directory);
  QDir dir(_directory);
  QFileInfoList fileList = dir.entryInfoList();

  clearListItem();

  initListItem();

  foreach (QFileInfo info, fileList) {
    QListWidgetItem* item = new QListWidgetItem;
    item->setSizeHint(QSize(35, 35));
    if (info.suffix() == QString(tr("def"))) {
      _file_list->insertItem(_def_index++, item);
      _file_list->setItemWidget(item, new ItemWidget(info, _file_list));
      _lef_index++;
    } else if (info.suffix() == QString(tr("lef"))) {
      if (_tech_file.compare(info.filePath().toStdString()) == 0) {
        continue;
      }
      _file_list->insertItem(_lef_index++, item);
      _file_list->setItemWidget(item, new ItemWidget(info, _file_list));
    }
  }
}

void FileImport::openTechFile() {
  QString file_name = QFileDialog::getOpenFileName(this, "Tech File", _directory, "*.lef");
  if (file_name.isEmpty())
    return;

  _tech_directory->setText(file_name);
  _tech_file = file_name.toStdString();

  /// if tech file in lef list item, remove it
  for (int i = 0; i < _file_list->count(); ++i) {
    ItemWidget* widget = qobject_cast<ItemWidget*>(_file_list->itemWidget(_file_list->item(i)));
    if (widget != nullptr) {
      QFileInfo fileInfo = widget->getFileInfo();
      if (fileInfo.filePath().toStdString().compare(_tech_file) == 0) {
        // _file_list->removeItemWidget(_file_list->item(i));
        _file_list->takeItem(i);
      }
    }
  }
  /// remove tech file from file list
  std::list<std::string>& lef_file_list = _fileMap["lef"];
  for (auto it = lef_file_list.begin(); it != lef_file_list.end(); ++it) {
    if (_tech_file.compare(*it) == 0) {
      _fileMap["lef"].erase(it);
    }
  }
}

void FileImport::allSelect() {
  bool all_checked = true;
  for (int i = 0; i < _file_list->count(); ++i) {
    if (ItemWidget* widget = qobject_cast<ItemWidget*>(_file_list->itemWidget(_file_list->item(i)))) {
      if (!widget->isChecked()) {
        widget->check();
        all_checked = false;
      }
    }
  }

  if (all_checked) {
    for (int i = 0; i < _file_list->count(); ++i) {
      if (ItemWidget* widget = qobject_cast<ItemWidget*>(_file_list->itemWidget(_file_list->item(i)))) {
        widget->check();
      }
    }
  }
}
void FileImport::okAct() {
  if (_tech_file.empty()) {
    std::cout << "_tech_file empty" << std::endl;
    return;
  }

  if (!_fileMap.empty()) {
    _fileMap.clear();
  }

  /// add tech file
  _fileMap["lef"].push_back(_tech_file);

  for (int i = 0; i < _file_list->count(); ++i) {
    ItemWidget* widget;
    if ((widget = qobject_cast<ItemWidget*>(_file_list->itemWidget(_file_list->item(i)))) && widget->isChecked()) {
      QFileInfo fileInfo = widget->getFileInfo();

      if (fileInfo.filePath().toStdString().compare(_tech_file) == 0) {
        continue;
      }

      if (fileInfo.suffix() == QString(tr("def")))
        _fileMap["def"].push_back(fileInfo.filePath().toStdString());
      else if (fileInfo.suffix() == QString(tr("lef"))) {
        std::string str_lef = fileInfo.filePath().toStdString();
        if (str_lef.compare(_tech_file) == 0) {
          continue;
        }
        _fileMap["lef"].push_back(str_lef);
      }
    }
  }
  close();
  this->setResult(QDialog::Accepted);
}
void FileImport::cancelAct() {
  if (!_fileMap.empty())
    _fileMap.clear();
  close();
  this->setResult(QDialog::Rejected);
}
void FileImport::setItemChecked(QListWidgetItem* item) {
  const QListWidget* listWidget = qobject_cast<const QListWidget*>(sender());
  if (ItemWidget* widget = qobject_cast<ItemWidget*>(listWidget->itemWidget(item)))
    widget->check();
}

ItemWidget::ItemWidget(const QFileInfo& info, QWidget* parent) : QWidget(parent) {
  fileInfo = info;

  QLabel* lable = new QLabel;
  lable->setScaledContents(true);
  lable->setPixmap(QPixmap(":/icon/file.png"));

  checkBox = new QCheckBox;

  QHBoxLayout* layout = new QHBoxLayout;
  layout->addWidget(lable);
  layout->addWidget(new QLabel(info.baseName()));
  layout->addStretch();
  layout->addWidget(checkBox);

  setLayout(layout);
}
bool ItemWidget::isChecked() { return checkBox->isChecked(); }
QFileInfo ItemWidget::getFileInfo() { return fileInfo; }
void ItemWidget::check() { checkBox->isChecked() ? checkBox->setChecked(false) : checkBox->setChecked(true); }
