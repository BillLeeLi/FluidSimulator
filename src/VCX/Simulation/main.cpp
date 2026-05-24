#include "Assets/bundled.h"
#include "Simulation/App.h"

int main() {
    using namespace VCX;
    return Engine::RunApp<MainScene::App>(Engine::AppContextOptions {
        .Title         = "Simulation",
        .WindowSize    = { 1024, 768 },
        .FontSize      = 16,
        .IconFileNames = Assets::DefaultIcons,
        .FontFileNames = Assets::DefaultFonts,
    });
}
