#pragma once

#include "Simulation/FluidSimulator.h"
#include "Simulation/FluidSolidCoupler.h"
#include "Simulation/RigidBodySystem.h"

namespace VCX::MainScene {

    class SimulationWorld {
    public:
        // 设置场景
        void Setup(int res);
        void Reset();
        void Reset(int res);
        // 主要函数：执行一个模拟步骤，dt为时间步长（秒）
        void Step(float dt);      
        // 执行一个渲染帧的模拟更新，frameDt为帧时间（秒），内部会累积时间并调用Step进行固定步长更新  
        void StepFrame(float frameDt);
        // 获取当前模拟状态（不可修改）
        FluidSimulator const & GetFluid() const { return _fluid; }
        FluidSimulator &       GetFluid() { return _fluid; }

        FluidSolidCoupler const & GetCoupler() const { return _coupler; }
        FluidSolidCoupler &       GetCoupler() { return _coupler; }

        RigidBodySystem const & GetRigidBodies() const { return _rigidBodies; }
        RigidBodySystem &       GetRigidBodies() { return _rigidBodies; }

        void ApplyExternalForceToBody(int bodyIndex, glm::vec3 const & force); // 这里与刚体系统交互，施加外力
        // 获取当前模拟参数（不可修改）
        int   Resolution() const { return _resolution; }
        int   InvDeltaTime() const { return _invDeltaTime; }
        float FlipRatio() const { return _fluid.m_fRatio; }
        float LastSimTimeMs() const { return _lastSimTimeMs; }
        // ===设置模拟参数===
        void SetInvDeltaTime(int invDeltaTime); // 设置步长
        void SetFlipRatio(float flipRatio);     // 设置FLIP/PIC混合比例 (0=纯PIC, 1=纯FLIP, 默认0.95）

    private:
        FluidSimulator    _fluid;
        RigidBodySystem   _rigidBodies;
        FluidSolidCoupler _coupler;
        glm::vec3         _externalforce { 0.0f, 0.0f, 0.0f };

        // 流体模拟参数
        int _numSubSteps { 1 };                 // 每个固定物理步内部再细分的次数
        int _numParticleIters { 5 };            // 粒子分离计算迭代次数
        int _numPressureIters { 30 };           // 压力投影迭代次数
        bool _separateParticles { true };       // 是否执行粒子分离
        float _overRelaxation { 1.9f };         // 压力求解的超松弛系数
        bool _compensateDrift { false };        // 是否在压力求解中补偿密度漂移

        // 全场景模拟参数
        int _resolution { 16 };                 // 网格分辨率
        int _invDeltaTime { 60 };               // 固定物理步长的倒数
        float _timeAccumulator { 0.0f };        // 渲染帧时间累积器
        float _lastSimTimeMs { 0.0f };          // 最近一次物理 step 的耗时
    };

} // namespace VCX::MainScene
