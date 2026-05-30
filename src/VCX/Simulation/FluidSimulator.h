#pragma once

#include <glm/glm.hpp>
#include <vector>

namespace VCX::MainScene {
    // 单元格类型标记
    const int EMPTY_CELL = 0;
    const int FLUID_CELL = 1;
    const int SOLID_CELL = 2;

    // 纯流体模式的可配置时间步参数
    struct FluidStepConfig {
        int   numSubSteps       = 1;     // 每帧的子步数
        int   numParticleIters  = 5;     // 粒子分离迭代次数 (O(n) 借助哈希表)
        int   numPressureIters  = 30;    // 压力求解迭代次数
        bool  separateParticles = true;  // 是否启用粒子分离
        float overRelaxation    = 1.9f;  // 超松弛因子
        bool  compensateDrift   = false; // 是否补偿漂移
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

        // ==================== 流体表面建模 ====================
        std::vector<float>     m_surfacePhi;       // 近似 signed distance（不保证梯度大小为1），只在表面窄带内可靠
        std::vector<float>     m_surfaceColor;     // 标量场c(x)，表示x附近的粒子加权数量（越小越接近流体表面）
        std::vector<glm::vec3> m_surfaceNormal;    // 表面法线
        std::vector<float>     m_surfaceCurvature; // 表面曲率

        bool  enableSurfaceModeling = true;  // 是否启用表面建模
        float m_surfaceTension      = 0.02f; // 表面张力系数
        float m_surfaceIsoValue     = 0.5f;  // 自由表面等值面阈值，意思是m_surfaceColor=m_surfaceIsoValue的地方被认为是表面
        float m_surfaceKernelRadius = 0.0f;  // 表面建模核半径.默认值2*m_h
        float m_surfaceRestField    = 0.0f;  // m_surfaceColor的参考值，用于归一化(在第一次被使用时取为全部液体cell的color均值)
        float m_surfaceBandWidth    = 0.0f;  // 表面窄带宽度，表面张力只施加在|phi|<bandWidth的范围内.默认值2*m_h
        float m_surfaceCurvatureMax = 0.0f;  // 最大曲率 (用于限制过大曲率引起的数值不稳定).默认值1/m_h

        glm::vec3 gravity { 0, -9.81f, 0 };

        // ==================== 核心模拟函数 ====================

        // Step 1: 粒子平流 — 施加重力并更新位置
        void integrateParticles(float timeStep);

        // Step 2: 边界碰撞处理 — "无粘"边界条件
        void handleParticleCollisions(glm::vec3 obstaclePos, float obstacleRadius, glm::vec3 obstacleVel);

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

        // Step 7: 更新粒子颜色 — 根据压力大小着色 (蓝→青→红)
        void updateParticleColors();

        // 场景初始化
        void setupScene(int res);

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

        // 查看Cell类型，用于调试和边界判断
        // 网格外视为固体
        bool IsCellSolid(glm::ivec3 idx) const;

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
        bool isValidVelocity(int i, int j, int k, int dir) const;
    };
} // namespace VCX::MainScene
