#include "Simulation/SimulationWorld.h"

#include <algorithm>
#include <chrono>

namespace VCX::MainScene {

    void SimulationWorld::Setup(int res) {
        _resolution = std::max(1, res);
        _fluid.setupScene(_resolution);
        _rigidBodies.Clear();
        _timeAccumulator = 0.0f;
        _lastSimTimeMs   = 0.0f;
    }

    void SimulationWorld::Reset() {
        Setup(_resolution);
    }

    void SimulationWorld::Reset(int res) {
        Setup(res);
    }

    void SimulationWorld::ApplyExternalForceToBody(int bodyIndex, glm::vec3 const & force) {
        // 这里对接刚体系统中将外力应用到指定刚体上
        // 例如: _rigidBodies.ApplyForce(bodyIndex, force);
    }

    void SimulationWorld::Step(float dt) {
        _coupler.ResetDebug();

        int const   numSubSteps = std::max(1, _numSubSteps);
        float const sdt         = dt / float(numSubSteps);
        float const flipRatio   = _fluid.m_fRatio;

        glm::vec3 const obstaclePos(0.0f);
        glm::vec3 const obstacleVel(0.0f);

        for (int step = 0; step < numSubSteps; ++step) {
            _fluid.integrateParticles(sdt);
            _fluid.handleParticleCollisions(obstaclePos, 0.0f, obstacleVel);
            if (_separateParticles)
                _fluid.pushParticlesApart(_numParticleIters);
            _fluid.handleParticleCollisions(obstaclePos, 0.0f, obstacleVel);
            _fluid.transferVelocities(true, flipRatio);
            _fluid.updateParticleDensity();
            _fluid.solveIncompressibility(_numPressureIters, sdt, _overRelaxation, _compensateDrift);
            _fluid.transferVelocities(false, flipRatio);
        }

        _fluid.updateParticleColors();
    }

    void SimulationWorld::StepFrame(float frameDt) {
        frameDt = std::min(frameDt, 0.1f);
        _timeAccumulator += frameDt;

        float const dt = 1.0f / float(_invDeltaTime);
        int         stepCount = 0;

        while (_timeAccumulator >= dt && stepCount < 2) {
            auto const startSim = std::chrono::high_resolution_clock::now();
            Step(dt);
            auto const endSim = std::chrono::high_resolution_clock::now();

            _lastSimTimeMs = std::chrono::duration<float, std::milli>(endSim - startSim).count();
            _timeAccumulator -= dt;
            ++stepCount;
        }

        if (_timeAccumulator > dt) {
            _timeAccumulator = 0.0f;
        }
    }

    void SimulationWorld::SetInvDeltaTime(int invDeltaTime) {
        _invDeltaTime = std::max(1, invDeltaTime);
    }

    void SimulationWorld::SetFlipRatio(float flipRatio) {
        _fluid.m_fRatio = std::clamp(flipRatio, 0.0f, 1.0f);
    }

} // namespace VCX::MainScene
