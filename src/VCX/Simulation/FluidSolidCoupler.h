#pragma once

#include <vector>

#include <Eigen/Core>

namespace VCX::MainScene {

    struct FluidSimulator;
    class RigidBodySystem;

    class FluidSolidCoupler {
    public:
        bool  enableRigidSolidMask { true };
        bool  enablePressureForce  { true };
        bool  enableMovingSolidVelocity { true };
        bool  enableParticleCollisionImpulse { true };
        float pressureForceScale   { 2.0f };
        float maxPressureForForce  { 80.0f };
        float particleImpulseScale { 0.05f };

        void ResetDebug();
        // 将所有穿透刚体的粒子投影到最近的表面点，并调整速度以消除穿透分量
        int  ProjectParticlesOutOfRigidBodies(FluidSimulator & fluid, RigidBodySystem & rigid);
        // 将刚体光栅化到流体网格
        int  RasterizeRigidBodiesToFluid(FluidSimulator & fluid, RigidBodySystem const & rigid);
        // 将刚体接触面的速度写入流体MAC边界速度
        int  ApplyRigidBoundaryVelocitiesToFluid(FluidSimulator & fluid, RigidBodySystem const & rigid);
        // 根据流体压力场计算作用在刚体上的压力力，并施加到刚体系统中
        int  ApplyPressureForcesFromFluid(FluidSimulator const & fluid, RigidBodySystem & rigid);

        // === 统计信息接口 ===
        int ProjectedParticleCount() const { return _projectedParticleCount; }
        int RigidSolidCellCount() const { return _rigidSolidCellCount; }
        int PressureContactFaceCount() const { return _pressureContactFaceCount; }
        int MovingBoundaryFaceCount() const { return _movingBoundaryFaceCount; }
        Eigen::Vector3f PressureForceOnBody(int bodyId) const;
        Eigen::Vector3f ParticleImpulseOnBody(int bodyId) const;

    private:
        int _projectedParticleCount    { 0 };
        int _rigidSolidCellCount       { 0 };
        int _pressureContactFaceCount  { 0 };
        int _movingBoundaryFaceCount   { 0 };
        std::vector<Eigen::Vector3f> _pressureForcesByBody;
        std::vector<Eigen::Vector3f> _particleImpulsesByBody;
    };

} // namespace VCX::MainScene
