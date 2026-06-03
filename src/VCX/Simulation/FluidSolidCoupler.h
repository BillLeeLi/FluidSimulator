#pragma once

namespace VCX::MainScene {

    struct FluidSimulator;
    class RigidBodySystem;

    class FluidSolidCoupler {
    public:
        void ResetDebug();
        // 将所有穿透刚体的粒子投影到最近的表面点，并调整速度以消除穿透分量
        int  ProjectParticlesOutOfRigidBodies(FluidSimulator & fluid, RigidBodySystem const & rigid);

        int ProjectedParticleCount() const { return _projectedParticleCount; }

    private:
        int _projectedParticleCount { 0 };
    };

} // namespace VCX::MainScene
