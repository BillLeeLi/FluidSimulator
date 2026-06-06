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
        bool  enableBoatBuoyancy { false };
        float pressureForceScale   { 200.0f };
        float maxPressureForForce  { 80.0f };
        float particleImpulseScale { 0.5f };
        float boatWaterLevel       { 0.0f }; // 这里作为根据真实粒子表面估计水位后的微调量
        float boatBuoyancyScale    { 1.08f };
        float boatWaterDrag        { 3.0f };

        void ResetDebug();
        // 将所有穿透刚体的粒子投影到最近的表面点，并调整速度以消除穿透分量
        int  ProjectParticlesOutOfRigidBodies(FluidSimulator & fluid, RigidBodySystem & rigid);
        // 将刚体光栅化到流体网格
        int  RasterizeRigidBodiesToFluid(FluidSimulator & fluid, RigidBodySystem const & rigid);
        // 将刚体接触面的速度写入流体MAC边界速度
        int  ApplyRigidBoundaryVelocitiesToFluid(FluidSimulator & fluid, RigidBodySystem const & rigid);
        // 根据流体压力场计算作用在刚体上的压力力，并施加到刚体系统中
        int  ApplyPressureForcesFromFluid(FluidSimulator const & fluid, RigidBodySystem & rigid);
        // 可选的备用船体浮力。压力场 demo 默认关闭。
        int  ApplyBoatBuoyancyForces(FluidSimulator const & fluid, RigidBodySystem & rigid);

        // === 统计信息接口 ===
        int ProjectedParticleCount() const { return _projectedParticleCount; }
        int RigidSolidCellCount() const { return _rigidSolidCellCount; }
        int PressureContactFaceCount() const { return _pressureContactFaceCount; }
        int MovingBoundaryFaceCount() const { return _movingBoundaryFaceCount; }
        Eigen::Vector3f PressureForceOnBody(int bodyId) const;
        Eigen::Vector3f ParticleImpulseOnBody(int bodyId) const;
        Eigen::Vector3f BoatBuoyancyForceOnBody(int bodyId) const;

    private:
        int _projectedParticleCount    { 0 };
        int _rigidSolidCellCount       { 0 };
        int _pressureContactFaceCount  { 0 };
        int _movingBoundaryFaceCount   { 0 };
        std::vector<Eigen::Vector3f> _pressureForcesByBody;
        std::vector<Eigen::Vector3f> _particleImpulsesByBody;
        std::vector<Eigen::Vector3f> _boatBuoyancyForcesByBody;
    };

} // namespace VCX::MainScene
