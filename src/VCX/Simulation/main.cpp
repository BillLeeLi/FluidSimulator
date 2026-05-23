#include "Assets/bundled.h"
#include "Simulation/App.h"

int main() {
    using namespace VCX;
    return Engine::RunApp<Fluid::App>(Engine::AppContextOptions {
        .Title         = "VCX-sim 2: Fluid Simulation",
        .WindowSize    = { 1024, 768 },
        .FontSize      = 16,
        .IconFileNames = Assets::DefaultIcons,
        .FontFileNames = Assets::DefaultFonts,
    });
}
