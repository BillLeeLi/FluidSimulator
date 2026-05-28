#pragma once

#include "Simulation/FluidSimulator.h"
#include "Simulation/FluidSolidCoupler.h"
#include "Simulation/RigidBodySystem.h"

namespace VCX::MainScene {

    class SimulationWorld {
    public:
        void Setup(int res);
        void Reset();
        void Reset(int res);

        void Step(float dt);
        void StepFrame(float frameDt);

        FluidSimulator const & GetFluid() const { return _fluid; }
        FluidSimulator &       GetFluid() { return _fluid; }

        FluidSolidCoupler const & GetCoupler() const { return _coupler; }
        FluidSolidCoupler &       GetCoupler() { return _coupler; }

        RigidBodySystem const & GetRigidBodies() const { return _rigidBodies; }
        RigidBodySystem &       GetRigidBodies() { return _rigidBodies; }

        void ApplyExternalForceToBody(int bodyIndex, glm::vec3 const & force);
        void ApplyExternalForceToBody(int bodyIndex, Eigen::Vector3f const & force);
        void ApplyExternalTorqueToBody(int bodyIndex, Eigen::Vector3f const & torque);
        void ApplyImpulseToBody(int bodyIndex, Eigen::Vector3f const & impulse);

        int   Resolution() const { return _resolution; }
        int   InvDeltaTime() const { return _invDeltaTime; }
        float FlipRatio() const { return _fluid.m_fRatio; }
        float LastSimTimeMs() const { return _lastSimTimeMs; }

        int   SelectedRigidBody() const { return _selectedRigidBody; }
        float RigidKeyboardForce() const { return _rigidKeyboardForce; }
        RigidBodyPreset RigidPreset() const { return _rigidBodyPreset; }

        void SetInvDeltaTime(int invDeltaTime);
        void SetFlipRatio(float flipRatio);
        void SetSelectedRigidBody(int id);
        void SetRigidKeyboardForce(float force);
        void SetRigidPreset(RigidBodyPreset preset);

    private:
        FluidSimulator    _fluid;
        RigidBodySystem   _rigidBodies;
        FluidSolidCoupler _coupler;

        // External actions accumulated from UI during the current render frame.
        Eigen::Vector3f _externalForce  { 0.0f, 0.0f, 0.0f };
        Eigen::Vector3f _externalTorque { 0.0f, 0.0f, 0.0f };
        Eigen::Vector3f _externalImpulse { 0.0f, 0.0f, 0.0f };
        int             _externalBody { -1 };

        int _numSubSteps { 1 };
        int _numParticleIters { 5 };
        int _numPressureIters { 30 };
        bool _separateParticles { true };
        float _overRelaxation { 1.9f };
        bool _compensateDrift { false };

        int _resolution { 16 };
        int _invDeltaTime { 60 };
        float _timeAccumulator { 0.0f };
        float _lastSimTimeMs { 0.0f };

        int _selectedRigidBody { 0 };
        float _rigidKeyboardForce { 8.0f };
        RigidBodyPreset _rigidBodyPreset { RigidBodyPreset::FluidCouplingMixed };
    };

} // namespace VCX::MainScene
