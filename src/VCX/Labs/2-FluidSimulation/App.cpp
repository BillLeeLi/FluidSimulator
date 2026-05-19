#include "Labs/2-FluidSimulation/App.h"

namespace VCX::Labs::Fluid {

    App::App() {
        _ui.Setup(_cases, _caseId);
    }

    void App::OnFrame() {
        _ui.Setup(_cases, _caseId);
    }

} // namespace VCX::Labs::Fluid
