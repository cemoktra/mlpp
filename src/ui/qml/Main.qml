import QtQuick 2.12
import QtQuick.Window 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12
import QtQuick.Layouts 1.13
import QtQuick.Dialogs 1.0

ApplicationWindow {
    id: window
    width: 1024
    height: 800
    visible: true
    title: "mlpp-ui"
    flags: Qt.FramelessWindowHint

    Material.theme: Material.Dark
    Material.accent: Material.DeepOrange
    Material.primary: Material.Blue

    header: 
        ToolBar {
            RowLayout {
                anchors.fill: parent
                ToolButton {
                    text: "\u205D"
                    onClicked: mainMenu.open()
                }
                Label {
                    text: "MachineLearning UI"
                    elide: Label.ElideRight
                    horizontalAlignment: Qt.AlignHCenter
                    verticalAlignment: Qt.AlignVCenter
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    MouseArea {
                        anchors.fill: parent
                        property variant previousPosition  
                        onPressed: {
                            previousPosition = Qt.point(mouseX, mouseY)
                        }
                        onPositionChanged: {
                            if (pressedButtons == Qt.LeftButton) {
                                var dx = mouseX - previousPosition.x
                                var dy = mouseY - previousPosition.y
                                window.x = window.x + dx
                                window.y = window.y + dy
                            }
                        }
                    }
                }
                ToolButton {
                    text: "\u2573"
                    onClicked: window.close()
                }
            }
        }

    Menu {
        id: mainMenu
        MenuItem { 
            text: "&Load CSV..." 
            onTriggered: fileDialog.open()
        }
        MenuSeparator { }
        MenuItem { 
            text: "&Exit" 
            onTriggered: window.close()
        }
    }

    FileDialog {
        id: fileDialog
        nameFilters: ["CSV files (*.csv)"]
        // folder: StandardPaths.writableLocation(StandardPaths.DocumentsLocation)
        onAccepted: {
        }
    }
}