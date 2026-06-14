#pragma once

#include "Simulation/SDFField.h"

#include <glm/glm.hpp>
#include <cstdint>
#include <utility>
#include <vector>

namespace VCX::MainScene {
    class RigidBodySystem;

    // 单元格类型标记
    const int EMPTY_CELL = 0;
    const int FLUID_CELL = 1;
    const int SOLID_CELL = 2;

    struct FluidSurfaceMesh {
        std::vector<glm::vec3>      positions;
        std::vector<glm::vec3>      normals;
        std::vector<std::uint32_t>  indices;
    };

    // 纯流体模式的可配置时间步参数
    struct FluidStepConfig {
        int   numSubSteps       = 1;     // 每帧的子步数
        int   numParticleIters  = 5;     // 粒子分离迭代次数 (O(n) 借助哈希表)
        int   numPressureIters  = 30;    // 压力求解迭代次数
        bool  separateParticles = true;  // 是否启用粒子分离
        float overRelaxation    = 1.9f;  // 超松弛因子
        bool  compensateDrift   = false; // 是否补偿漂移
    };

    // 两张映射表：
    // pressureDof: 从网格 cellId 找到它在 Ax=b 里的未知量编号；不是流体则为 -1。
    // dofCell:     从未知量编号反查对应的三维 cell 坐标，解完以后把压力写回 m_p。
    struct FluidPressureDofs {
        std::vector<int>        pressureDof; // cellId -> pressure dof, -1表示不是压力未知量
        std::vector<glm::ivec3> dofCell;     // pressure dof -> cell index
    };

    struct FluidSimulator {
        // ==================== 粒子数据 ====================
        std::vector<glm::vec3> m_particlePos;   // 粒子位置 (世界坐标, 范围 [-0.5, 0.5])
        std::vector<glm::vec3> m_particleVel;   // 粒子速度
        std::vector<glm::vec3> m_particleColor; // 粒子颜色 (用于渲染)

        // ==================== 模拟参数 ====================
        float m_fRatio = 0.95f; // FLIP ratio: 0=PIC, 1=FLIP, 0.95=FLIP95

        // ==================== 网格几何参数 ====================
        int   m_iCellX; // 网格点数 (res+1)
        int   m_iCellY;
        int   m_iCellZ;
        float m_h;           // 网格间距 = 1.0 / res
        float m_fInvSpacing; // 网格间距的倒数 = res
        int   m_iNumCells;   // 总网格点数 = m_iCellX * m_iCellY * m_iCellZ

        int   m_iNumSpheres;    // 粒子总数
        float m_particleRadius; // 粒子半径 (约为 0.3*h)

        // ==================== 交错网格速度 (MAC Grid) ====================
        // 网格点 (i,j,k) 存储三个速度面分量:
        //   m_vel[idx].x = u(i, j+1/2, k+1/2) — x方向速度面
        //   m_vel[idx].y = v(i+1/2, j, k+1/2) — y方向速度面
        //   m_vel[idx].z = w(i+1/2, j+1/2, k) — z方向速度面
        std::vector<glm::vec3> m_vel;         // 当前网格速度
        std::vector<glm::vec3> m_pre_vel;     // 压力求解前的网格速度 (用于FLIP增量计算)
        std::vector<float>     m_near_num[3]; // 每个速度面的权重累积和 (用于P2G归一化)
        std::vector<glm::vec3> m_solidVel;    // 运动固体边界速度: x/y/z分别对应三个MAC face方向
        std::vector<glm::ivec3> m_solidVelMask; // 运动固体边界速度是否有效: 0/1

        // ==================== 空间哈希表 ====================
        std::vector<int>        m_hashtable;      // 按格点排序的粒子索引列表
        std::vector<int>        m_hashtableindex; // 每个格点的前缀和, 大小 m_iNumCells+1
        std::vector<glm::ivec3> m_particleSlot;   // 每个粒子所在的格点索引

        // ==================== 压力与单元格状态 ====================
        std::vector<float> m_p;                          // 压力 (用于可视化)
        std::vector<float> m_s;                          // 固体分数: 0.0=固体, 1.0=非固体
        std::vector<int>   m_type;                       // 单元格类型 (EMPTY/FLUID/SOLID)
        std::vector<float> m_particleDensity;            // 每个格点的核密度估计
        float              m_particleRestDensity = 1.0f; // 静止密度 (初始化时估算)

        // ==================== 仿真 SDF 几何场 ====================
        // SDF 本身只描述边界位置；下面的 fraction 才是投影矩阵最终要消费的几何权重。
        SDFField m_phiFluidSim; // 粒子构建的仿真用液体 SDF；不做渲染平滑
        SDFField m_phiSolid;    // 刚体/水槽边界构建的仿真用固体 SDF
        float    m_simSdfParticleRadius = 0.0f; // 0 => 默认使用 m_particleRadius
        float    m_simSdfNarrowBand     = 0.0f; // 0 => 默认窄带宽度
        float    m_solidSdfMaxDistance  = 0.0f; // 0 => 默认固体 SDF 更新范围
        // cell fraction: 1 表示该压力 cell 完全被对应几何占据，0 表示完全没有占据。
        std::vector<float> m_liquidCellFraction;
        std::vector<float> m_solidCellFraction;
        // face open fraction: x/y/z 分量分别对应同一 grid index 上的 u/v/w MAC face。
        // 1 表示 face 完全开放，0 表示完全被固体遮挡。
        std::vector<glm::vec3> m_faceOpenFraction;

        // ==================== 流体表面建模 ====================
        std::vector<float>     m_surfacePhi;       // 近似 signed distance（不保证梯度大小为1），只在表面窄带内可靠
        std::vector<float>     m_surfaceColor;     // 标量场c(x)，表示x附近的粒子加权数量（越小越接近流体表面）
        std::vector<glm::vec3> m_surfaceNormal;    // 表面法线
        std::vector<float>     m_surfaceCurvature; // 表面曲率

        bool  enableSurfaceModeling = false; // 是否启用表面建模
        float m_surfaceTension      = 0.02f; // 表面张力系数
        float m_surfaceIsoValue     = 0.5f;  // 自由表面等值面阈值，意思是m_surfaceColor=m_surfaceIsoValue的地方被认为是表面
        float m_surfaceKernelRadius = 0.0f;  // 表面建模核半径.默认值2*m_h
        float m_surfaceRestField    = 0.0f;  // m_surfaceColor的参考值，用于归一化(在第一次被使用时取为全部液体cell的color均值)
        float m_surfaceBandWidth    = 0.0f;  // 表面窄带宽度，表面张力只施加在|phi|<bandWidth的范围内.默认值2*m_h
        float m_surfaceCurvatureMax = 0.0f;  // 最大曲率 (用于限制过大曲率引起的数值不稳定).默认值1/m_h

        float m_renderSurfaceResolutionScale = 1.5f;
        int   m_renderSurfaceCellX           = 0;
        int   m_renderSurfaceCellY           = 0;
        int   m_renderSurfaceCellZ           = 0;
        float m_renderSurfaceH               = 0.0f;  // 等于 m_h/m_renderSurfaceResolutionScale
        float m_renderSurfaceInvH            = 0.0f;
        float m_renderSurfaceIsoValue        = 0.45f;
        float m_renderSurfaceKernelRadius    = 0.0f;
        int   m_renderSurfaceBlurIters       = 1;
        int   m_renderSurfaceUpdateInterval  = 2;
        int   m_renderSurfaceFrameCounter    = 0;

        // Offline render surface reconstruction mode.
        // density mode: smooth color-field / metaball-like surface.
        // sdf mode: union-of-particles signed distance, closer to FluidRigidCoupling3D.
        bool  m_renderSurfaceUseParticleSdf  = false;
        float m_renderSurfaceSdfParticleRadius = 0.0f; // 0 => default multiple of m_particleRadius
        int   m_renderSurfaceSdfSmoothIters   = 1;    // render-only smoothing for particle SDF; not used by the solver

        std::vector<float> m_renderSurfaceColor;
        std::vector<float> m_renderSurfacePhi;
        std::vector<glm::vec3> m_renderSurfaceNormal;
        FluidSurfaceMesh   m_renderSurfaceMesh;

        glm::vec3 gravity { 0, -9.81f, 0 };
        bool      enableGravity = true;

        // ==================== 核心模拟函数 ====================

        // Step 1: 粒子平流 — 施加重力并更新位置
        void integrateParticles(float timeStep);

        // Step 2: 边界碰撞处理 — "无粘"边界条件
        void handleParticleCollisions();

        // Step 3: 粒子分离 — 防止粒子过度重叠 (O(n) 借助哈希表)
        void pushParticlesApart(int numIters);

        // Step 4 & 8: 速度传递 (粒子↔网格)
        // toGrid=true  → P2G: 将粒子速度加权累积到交错网格面上
        // toGrid=false → G2P: 从网格插值回粒子, 混合PIC与FLIP
        void transferVelocities(bool toGrid, float flipRatio);

        // Step 5: 更新粒子密度 — 对每个格点计算其周围粒子的核密度
        void updateParticleDensity();

        // Step 6: 求解不可压性 — 使用Gauss-Seidel迭代驱散速度散度
        void solveIncompressibility(int numIters, float dt, float overRelaxation, bool compensateDrift);

        void updateSurfaceField();
        void computeSurfaceGeometry();
        void applySurfaceTension(float dt);

        // 刷新仿真 SDF/fraction。调用点放在 P2G/密度更新之后、压力投影之前。
        void EnsureSimulationSDFFields();
        void ClearSimulationSDFFields();
        void BuildFluidSimulationSDF();
        void BuildSolidSimulationSDF(RigidBodySystem const & rigid);
        void BuildSimulationSDFFields(RigidBodySystem const & rigid);
        void EnsureSimulationFractionFields();
        void ClearSimulationFractionFields();
        void ComputeLiquidCellFractions();
        void ComputeSolidCellFractions();
        void ComputeFaceOpenFractions();
        void ComputeSimulationFractions();
        void PromoteBoundaryLiquidCellsToFluid();
        float FaceOpenFraction(glm::ivec3 face, int dir) const;

        void updateRenderableSurface();
        FluidSurfaceMesh const & GetRenderableSurface() const { return m_renderSurfaceMesh; }
        void SetSurfaceModelingEnabled(bool enabled);
        void EnsureSurfaceFields();
        void ClearSurfaceFields();
        void EnsureRenderableSurfaceFields();

        // Step 7: 更新粒子颜色 — 根据压力大小着色 (蓝→青→红)
        void updateParticleColors();

        // 场景初始化
        void setupScene(int res);

        // 在已分配好的网格上，按 HCP 布局重新生成指定形状的初始粒子。
        void ResetParticlesToBoxRegion(glm::vec3 minCorner, glm::vec3 maxCorner, glm::vec3 initialVelocity = glm::vec3(0.0f));
        void ResetParticlesToSphereRegion(glm::vec3 center, float radius, glm::vec3 initialVelocity = glm::vec3(0.0f));
        void FinalizeManualParticleSetup(std::vector<glm::vec3> positions, glm::vec3 initialVelocity = glm::vec3(0.0f));

        // 将3D网格索引->线性数组偏移
        int GridIndex(glm::ivec3 idx) const;

        // 判断网格索引是否越界
        bool IsInsideGrid(glm::ivec3 idx) const;

        // 世界坐标 → 网格点索引
        glm::ivec3 worldToCell(glm::vec3 const & p) const;

        // 网格索引 -> 世界坐标 (格点中心)
        glm::vec3 CellCenter(glm::ivec3 idx) const;

        // 重置`m_s`，只保留水槽六面体为固体
        void ResetSolidMaskToTank();

        // 设置固体Cell
        void SetCellSolid(glm::ivec3 idx, bool solid);

        // 重置/设置运动固体边界速度
        void ResetSolidBoundaryVelocity();
        void SetSolidBoundaryVelocity(glm::ivec3 idx, int dir, float velocity, bool updatePreVel = true);
        bool HasSolidBoundaryVelocity(int i, int j, int k, int dir) const;
        float SolidBoundaryVelocity(int i, int j, int k, int dir) const;

        // 判断某个 MAC 速度面是否是固体/玻璃边界面，并把所有这类面的速度强制设为边界速度。
        // 这一步很重要：否则 P2G 会把粒子速度写到墙面 face 上，压力投影虽然在散度里把墙当成 0 通量，
        // 但 G2P 仍可能把未投影的墙面速度采回粒子，导致玻璃边缘附近持续抖动/自发流动。
        bool IsVelocityFaceSolidBoundary(glm::ivec3 face, int dir) const;
        void EnforceSolidBoundaryVelocities();

        // 变分投影/矩阵压力求解会用到的 MAC face 和 pressure dof 辅助接口。
        // 这些接口把网格内部实现暴露成更清晰的“离散算子”：face 两侧 cell、face 中心、face 法向等。
        bool IsVelocityFaceInRange(glm::ivec3 face, int dir) const;
        // 返回某个 MAC 速度面两侧的 cell：lower 是负方向侧，upper 是正方向侧。
        std::pair<glm::ivec3, glm::ivec3> FaceNeighborCells(glm::ivec3 face, int dir) const;
        // 返回 MAC face 的世界坐标中心，用于计算刚体接触点速度 v + w x r。
        glm::vec3 FaceCenter(glm::ivec3 face, int dir) const;
        // 返回 face 法向轴：dir=0/1/2 分别对应 x/y/z 方向。
        glm::vec3 FaceAxis(int dir) const;
        // 从给定速度场读取某个 face 的法向速度分量，常用于读取投影前速度 u*。
        float FaceVelocity(std::vector<glm::vec3> const & velocities, glm::ivec3 face, int dir) const;
        // 为所有流体 cell 分配压力未知量编号。
        FluidPressureDofs BuildPressureDofs() const;
        // 查询 cell 对应的压力未知量编号；非流体/越界返回 -1。
        int PressureDofForCell(FluidPressureDofs const & dofs, glm::ivec3 cell) const;

        // 查看Cell类型，用于调试和边界判断
        // 网格外视为固体
        bool IsCellSolid(glm::ivec3 idx) const;
        bool IsNearSolidBoundary(glm::ivec3 idx) const;
        bool IsBoundaryLiquidCell(glm::ivec3 idx) const;
        bool IsFaceCutBySolidNormal(glm::ivec3 face, int dir) const;

        // 从压力网格`m_p`三线性插值采样流体速度
        float SamplePressure(glm::vec3 const & p) const;

        // 从MAC网格`m_vel`插值采样流体速度
        glm::vec3 SampleVelocityPIC(glm::vec3 const & p) const;

        // 纯流体模式的可配置时间步
        void SimulateTimestep(float dt, FluidStepConfig const & cfg);

    private:
        // ==================== 辅助函数 ====================

        // 将3D网格索引转换为线性数组偏移
        int index2GridOffset(glm::ivec3 idx) const;

        // 将值钳制到 [0, hi]
        static int clampSlot(int v, int hi);

        // 将值钳制到 [lo, hi]
        static int clampCoord(int v, int lo, int hi);

        // 线性B样条核函数: 三个轴的 (1-|d|) 乘积, |d|>=1 时返回0
        // 速度快, 无需sqrt, 支持范围为1个网格间距
        float weight(glm::vec3 gridPos, glm::vec3 partPos) const;

        // 构建空间哈希表 (计数排序, O(n))
        void buildHash();

        // 检查速度面 u/v/w 在 (i,j,k,dir) 处是否有效 (未越界且不在固体中)
        bool isVelocityFaceInRange(int i, int j, int k, int dir) const;
        bool isValidVelocity(int i, int j, int k, int dir) const;
    };
} // namespace VCX::MainScene
