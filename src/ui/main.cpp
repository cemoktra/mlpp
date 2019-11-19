#include <QApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickView>
#include <QQuickStyle>

int main(int argc, char *argv[])
{
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QQuickWindow::setSceneGraphBackend(QSGRendererInterface::OpenGL);
    
    QApplication app(argc, argv);
    app.setOrganizationName("mlpp");
    app.setOrganizationDomain("mlpp");

    QQmlApplicationEngine engine;
    QQuickStyle::setStyle("Material");
    const QUrl url(QStringLiteral("qrc:/qml/Main.qml"));
    engine.load(url);
    return app.exec();
}
