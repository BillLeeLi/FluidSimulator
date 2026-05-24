#include "Simulation/App.h"
#include "Assets/bundled.h"

namespace VCX::MainScene {

    App::App():
        _ui(Common::UIOptions {}),
        _mainScene({ Assets::ExampleScene::Fluid }) {
    }
    void App::OnFrame() {
        _ui.Setup(_cases, _caseId);
    }
}