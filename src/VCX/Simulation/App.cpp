#include "Simulation/App.h"
#include "Assets/bundled.h"

namespace VCX::Fluid {

    App::App():
        _ui(Common::UIOptions {}),
        _casefluid({ Assets::ExampleScene::Fluid }) {
    }
    void App::OnFrame() {
        _ui.Setup(_cases, _caseId);
    }
}