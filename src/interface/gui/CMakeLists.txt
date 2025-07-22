# cmake_minimum_required(VERSION 3.5)

# project(iGUI LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations ")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${HOME_BUILD}/lib/iGUI)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${HOME_BUILD}/lib/iGUI)
if(NOT DEFINED CMD_BUILD)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${HOME_BUILD}/lib/iGUI)
endif()


set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fexec-charset=GBK")
# set(CMAKE_PREFIX_PATH "/opt/Qt/5.15.2/gcc_64") 
# set(Qt5_DIR ${CMAKE_PREFIX_PATH}/lib/cmake/Qt5)
# set(Qt5Widgets_DIR ${CMAKE_PREFIX_PATH}/lib/cmake/Qt5Widgets)
find_package(Qt5 COMPONENTS Core Xml Widgets REQUIRED)
include_directories(${QT_INCLUDE})
include_directories(${Qt5Widgets_INCLUDE_DIRS})

set(sources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow_menu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow_toolbar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow_statusbar.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow_file.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow_search.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow_fp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow_pdn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow_placement.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow_cts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow_routing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mainwindow_drc.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/guigraphicsview.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guigraphicsscene.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB/dbsetup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB/idbsetup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB/idbfastsetup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB/idbfastsetupupdate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB/idbfastsetupsearch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB/idbfastsetupdrc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB/idbfastsetupclocktree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB/idbfastsetupInstancedemo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB/idbfastsetupCellMaster.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB/idbfastsetupgraph.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/graphicsitem/guiitem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/graphicsitem/guirect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/graphicsitem/guiline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/graphicsitem/guipolygon.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/graphicsitem/transform.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiparser/guixmlparser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiparser/guijsonparser.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/guicontroltree/guitree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guicontroltree/guicontroltreeitem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guicontroltree/guitreeitemcolumn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guicontroltree/guitreeitemhandler.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiaction/fileimport.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiaction/guiloading.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiaction/guisplash.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiaction/guisearchedit.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guidie.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guicore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guirow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guistandardcell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guipad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guiblock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guivia.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guipower.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guipin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guiinstance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guiflightline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guiwire.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guigr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/arrowitem.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupdesign.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupitem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupwire.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupvia.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupinstance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupgrid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupdrc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupclocktree.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupitemsearch.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/utility/guiattribute.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/utilityitem/shape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utilityitem/line.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utilityitem/ruler.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/interface/gui_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/idb_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/ifp_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/ipl_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/irt_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/icts_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/idrc_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/web_io.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/config/guiConfig.cpp

    res/source.qrc
)

set(headers
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mainwindow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/guigraphicsview.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/guigraphicsscene.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB/guidatabase.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB/dbsetup.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB/idbfastsetup.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB/idbsetup.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/graphicsitem/guiitem.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/graphicsitem/guirect.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/graphicsitem/guiline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/graphicsitem/guipolygon.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/graphicsitem/transform.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiparser/guixmlparser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiparser/guijsonparser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guicontroltree/guitree.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guicontroltree/guicontroltreeitem.h

    ${CMAKE_CURRENT_SOURCE_DIR}/src/guicontroltree/guitreeitemcolumn.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guicontroltree/guitreeitemhandler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiaction/fileimport.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiaction/guiloading.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiaction/guisplash.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guiaction/guisearchedit.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guidie.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guicore.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guirow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guistandardcell.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guipad.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guiblock.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guivia.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guipower.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guipin.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guiinstance.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guiflightline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guiwire.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/guigr.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem/arrowdrc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupdesign.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupitem.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupwire.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupvia.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupinstance.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupgrid.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupdrc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupitemsearch.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems/guispeedupclocktree.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utility/guiattribute.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utility/guistring.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utilityitem/shape.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utilityitem/line.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utilityitem/ruler.h
    
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/gui_io.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config/guiConfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config/guiConfigTree.h
)

include(${HOME_CMAKE}/operation/idb.cmake)

add_library(iGUI
      ${headers}
      ${sources}
)
target_link_libraries(iGUI Qt5${QT_VERSION_MAJOR}::Widgets Qt5${QT_VERSION_MAJOR}::Xml Qt5${QT_VERSION_MAJOR}::Core)
# target_link_libraries(iGUI idrc_api file_manager_cts file_manager_drc file_manager_placement tool_api_ipl OpenMP::OpenMP_CXX)
target_link_libraries(iGUI 
    file_manager_cts 
    # file_manager_drc 
    file_manager_placement
    tool_api_ipl
    ieda_feature
    ilm_layout_db
)
if(OpenMP_CXX_FOUND)
target_link_libraries(iGUI 
    OpenMP::OpenMP_CXX
)  
endif()

target_include_directories(iGUI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(iGUI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/graphicsitem)
target_include_directories(iGUI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/guiaction)
target_include_directories(iGUI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/guicontroltree)
target_include_directories(iGUI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/guiDB)
target_include_directories(iGUI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/guiparser)
target_include_directories(iGUI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/designitem)
target_include_directories(iGUI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/utilityitem)
target_include_directories(iGUI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/utility)
target_include_directories(iGUI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/guispeedupitems)
target_include_directories(iGUI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/interface)
target_include_directories(iGUI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/config)
target_include_directories(iGUI 
    PUBLIC 
    ${HOME_DATABASE}/interaction/RT_DRC
)
