#include "Simulation/SimulationWorld.h"

#include <algorithm>
#include <chrono>

namespace VCX::MainScene {

    std::vector<SimulationWorld::RigidBodyResetFlag> SimulationWorld::CaptureRigidBodyResetFlags() const {
        std::vector<RigidBodyResetFlag> flags;
        flags.reserve(_rigidBodies.Bodies.size());

        for (int i = 0; i < static_cast<int>(_rigidBodies.Bodies.size()); ++i) {
            if (_rigidBodies.IsInternalTankBoundary(i)) continue;
            auto const & body = _rigidBodies.Bodies[i];
            flags.push_back(RigidBodyResetFlag { body.name, body.isStatic, body.useGravity });
        }
        return flags;
    }

    void SimulationWorld::RestoreRigidBodyResetFlags(std::vector<RigidBodyResetFlag> const & flags) {
        for (auto const & saved : flags) {
            for (int i = 0; i < static_cast<int>(_rigidBodies.Bodies.size()); ++i) {
                if (_rigidBodies.IsInternalTankBoundary(i)) continue;
                if (_rigidBodies.Bodies[i].name != saved.name) continue;

                _rigidBodies.SetBodyStatic(i, saved.isStatic);
                _rigidBodies.SetBodyGravity(i, saved.useGravity);
                break;
            }
        }
    }

    void SimulationWorld::Setup(int res) {
        auto const oldFlags = CaptureRigidBodyResetFlags();
        bool const oldFluidGravity = _fluid.enableGravity;
        bool const oldSurfaceModeling = _fluid.enableSurfaceModeling;

        _resolution = std::max(1, res);
        _fluid.setupScene(_resolution);
        _fluid.enableGravity = oldFluidGravity;
        _fluid.SetSurfaceModelingEnabled(oldSurfaceModeling);

        _rigidBodies.SetupDefaultScene(_rigidBodyPreset);
        RestoreRigidBodyResetFlags(oldFlags);

        _timeAccumulator = 0.0f;
        _lastSimTimeMs   = 0.0f;
        _externalForce.setZero();
        _externalTorque.setZero();
        _externalImpulse.setZero();
        _externalBody = -1;
        SetSelectedRigidBody(_selectedRigidBody);
    }

    void SimulationWorld::Reset() {
        Setup(_resolution);
    }

    void SimulationWorld::Reset(int res) {
        Setup(res);
    }

    void SimulationWorld::ApplyExternalForceToBody(int bodyIndex, glm::vec3 const & force) {
        ApplyExternalForceToBody(bodyIndex, ToEigen(force));
    }

    void SimulationWorld::ApplyExternalForceToBody(int bodyIndex, Eigen::Vector3f const & force) {
        if (! _rigidBodies.IsValidBody(bodyIndex)) return;
        _externalBody = bodyIndex;
        _externalForce += force;
    }

    void SimulationWorld::ApplyExternalTorqueToBody(int bodyIndex, Eigen::Vector3f const & torque) {
        if (! _rigidBodies.IsValidBody(bodyIndex)) return;
        _externalBody = bodyIndex;
        _externalTorque += torque;
    }

    void SimulationWorld::ApplyImpulseToBody(int bodyIndex, Eigen::Vector3f const & impulse) {
        if (! _rigidBodies.IsValidBody(bodyIndex)) return;
        _externalBody = bodyIndex;
        _externalImpulse += impulse;
    }

    void SimulationWorld::Step(float dt) {
        _coupler.ResetDebug();

        int const   numSubSteps = std::max(1, _numSubSteps);
        float const sdt         = dt / float(numSubSteps);
        float const flipRatio   = _fluid.m_fRatio;

        glm::vec3 const obstaclePos(0.0f);
        glm::vec3 const obstacleVel(0.0f);

        for (int step = 0; step < numSubSteps; ++step) {
            // 1. Rigid body part.  This preserves the Lab1 style: explicit Euler
            // integration + FCL narrow-phase + sequential impulse contact solve.
            if (_rigidBodies.IsValidBody(_externalBody)) {
                if (_externalForce.squaredNorm() > 0.0f) {
                    _rigidBodies.ApplyForceToCenter(_externalBody, _externalForce);
                }
                if (_externalTorque.squaredNorm() > 0.0f) {
                    _rigidBodies.ApplyTorque(_externalBody, _externalTorque);
                }
                if (_externalImpulse.squaredNorm() > 0.0f) {
                    _rigidBodies.ApplyImpulseToCenter(_externalBody, _externalImpulse);
                    _externalImpulse.setZero();
                }
            }
            _rigidBodies.Step(sdt);
            _rigidBodies.ResolveTankBounds(-0.5f + _fluid.m_h, 0.5f - _fluid.m_h);

            // 2. Fluid part.  This is intentionally kept in the original order so
            // the old FLIP fluid behavior is not disturbed before coupling is added.
            _fluid.integrateParticles(sdt);
            _fluid.handleParticleCollisions(obstaclePos, 0.0f, obstacleVel);
            if (_separateParticles)
                _fluid.pushParticlesApart(_numParticleIters);
            _fluid.handleParticleCollisions(obstaclePos, 0.0f, obstacleVel);
            if (_fluid.enableSurfaceModeling) {
                _fluid.EnsureSurfaceFields();
                _fluid.updateSurfaceField();
                _fluid.computeSurfaceGeometry();
                _fluid.applySurfaceTension(sdt);
            }
            _fluid.transferVelocities(true, flipRatio);
            _fluid.updateParticleDensity();
            _fluid.solveIncompressibility(_numPressureIters, sdt, _overRelaxation, _compensateDrift);
            _fluid.transferVelocities(false, flipRatio);
        }

        _externalForce.setZero();
        _externalTorque.setZero();
        _externalBody = -1;
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

    void SimulationWorld::SetSurfaceModelingEnabled(bool enabled) {
        _fluid.SetSurfaceModelingEnabled(enabled);
    }

    void SimulationWorld::SetGravityEnabled(bool enabled) {
        _fluid.enableGravity = enabled;
    }

    void SimulationWorld::SetSelectedRigidBody(int id) {
        if (_rigidBodies.Bodies.empty()) {
            _selectedRigidBody = -1;
            return;
        }

        int const n = static_cast<int>(_rigidBodies.Bodies.size());
        int const start = std::clamp(id, 0, n - 1);

        for (int i = start; i < n; ++i) {
            if (! _rigidBodies.IsInternalTankBoundary(i)) {
                _selectedRigidBody = i;
                return;
            }
        }
        for (int i = start - 1; i >= 0; --i) {
            if (! _rigidBodies.IsInternalTankBoundary(i)) {
                _selectedRigidBody = i;
                return;
            }
        }

        _selectedRigidBody = -1;
    }

    void SimulationWorld::SetRigidKeyboardForce(float force) {
        _rigidKeyboardForce = std::max(0.0f, force);
    }

    void SimulationWorld::SetRigidPreset(RigidBodyPreset preset) {
        if (_rigidBodyPreset == preset) return;
        _rigidBodyPreset = preset;
        Reset();
    }

} // namespace VCX::MainScene
