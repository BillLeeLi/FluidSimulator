#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glm/glm.hpp>

namespace fcl {
    template <typename S> class CollisionGeometry;
}

namespace VCX::MainScene {

    enum class RigidBodyShape {
        Box,
        Sphere,
        BoatHull,
    };

    enum class RigidBodyPreset {
        FluidCouplingMixed = 0,
        BoxCollision       = 1,
        MixedStack         = 2,
        BoatInWater        = 3,
        BoatDropIntoPool   = 4,
        SurfaceTensionBlob = 5,
    };

    struct RigidBuoyancySample {
        Eigen::Vector3f localPosition = Eigen::Vector3f::Zero();
        float           volumeWeight  = 1.0f;
        float           radius        = 0.04f;
    };

    struct RigidContact {
        int             idA         = -1;
        int             idB         = -1;
        Eigen::Vector3f position    = Eigen::Vector3f::Zero(); // 世界坐标下的接触点
        Eigen::Vector3f normal      = Eigen::Vector3f::UnitY(); // 从 A 指向 B 的接触法线
        float           penetration = 0.0f;

        // 同一个接触点迭代时用的累计冲量，主要是为了堆叠时不乱跳。
        float           accumulatedNormalImpulse  = 0.0f;
        float           accumulatedTangentImpulse = 0.0f;
        Eigen::Vector3f tangent                   = Eigen::Vector3f::Zero();
        bool            tangentInitialized        = false;
    };

    // 给后面的流固耦合留的表面采样点。刚体这边只负责给点、法线和面积。
    struct RigidSurfaceSample {
        int             bodyId   = -1;
        Eigen::Vector3f position = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal   = Eigen::Vector3f::UnitY();
        float           area     = 0.0f;
    };

    // 复杂刚体的局部 SDF 缓存。用于把船体 mesh 查询变成 O(1) 网格插值。
    struct RigidSdfGrid {
        bool            valid      = false;
        Eigen::Vector3f localMin   = Eigen::Vector3f::Zero();
        Eigen::Vector3f localMax   = Eigen::Vector3f::Zero();
        Eigen::Vector3i resolution = Eigen::Vector3i::Zero();
        Eigen::Vector3f spacing    = Eigen::Vector3f::Ones();
        std::vector<float> phi; // 局部空间 signed distance，<=0 表示有效固体区域

        // 三维 SDF 网格坐标转一维数组下标。
        int Index(int i, int j, int k) const {
            return (i * resolution.y() + j) * resolution.z() + k;
        }
    };

    // 刚体耦合求解使用 6 维广义速度 V = [线速度v, 角速度w]。
    // 压力冲量会同时改变平动和转动，因此 coupler 不再只读写 body.v。
    using RigidGeneralizedVelocity = Eigen::Matrix<float, 6, 1>;
    // 广义逆质量矩阵 M_s^-1 = diag(1/m I, I_world^-1)，静态刚体返回零矩阵。
    using RigidInverseMassBlock    = Eigen::Matrix<float, 6, 6>;

    struct RigidBody {
        RigidBodyShape     shape = RigidBodyShape::Box;
        Eigen::Vector3f    dim   = Eigen::Vector3f::Ones(); // 盒子是完整长宽高，球用直径
        Eigen::Vector3f    x     = Eigen::Vector3f::Zero(); // 质心位置
        Eigen::Quaternionf q     = Eigen::Quaternionf::Identity(); // 姿态

        Eigen::Vector3f v = Eigen::Vector3f::Zero(); // 线速度
        Eigen::Vector3f w = Eigen::Vector3f::Zero(); // 角速度

        // 每个时间步里暂存外力，Step 结束会清掉。
        Eigen::Vector3f force  = Eigen::Vector3f::Zero();
        Eigen::Vector3f torque = Eigen::Vector3f::Zero();

        Eigen::Vector3f color = Eigen::Vector3f(0.75f, 0.75f, 0.8f);

        // BoatHull 使用同一套 Kenney 小船三角网格参与渲染、FCL 碰撞、流体投影和压力采样。
        std::vector<Eigen::Vector3f>     meshVertices;
        std::vector<std::uint32_t>       meshTriIndices;
        // 浮力采样点固定在船体局部坐标里，Step 时按当前姿态转到世界坐标施力。
        std::vector<RigidBuoyancySample> buoyancySamples;
        // 薄壳模型给流体网格用的有效厚度。FCL 碰撞仍然用原始三角面。
        float solidShellThickness = 0.0f;
        // 船体/复杂刚体给流固耦合用的局部 SDF。球和盒子仍使用解析 SDF。
        RigidSdfGrid localSdf;

        float       mass        = 1.0f;
        float       invMass     = 1.0f;
        float       restitution = 0.15f;
        float       friction    = 0.45f;
        bool        isStatic    = false;
        bool        useGravity  = true;
        std::string name;

        // 刚体局部坐标系下的转动惯量。求解时再转到世界坐标。
        Eigen::Matrix3f inertiaBody    = Eigen::Matrix3f::Identity();
        Eigen::Matrix3f inertiaBodyInv = Eigen::Matrix3f::Identity();

        // 返回一个保守包围球半径，主要用于粗碰撞/粗范围筛选。
        float GetBoundingSphereRadius() const;
        // 根据 shape/dim/mass/isStatic 重新计算 invMass 和转动惯量；改质量或尺寸后要调用。
        void UpdateMassProperties();
        // 当前姿态四元数 q 对应的世界旋转矩阵。
        Eigen::Matrix3f GetRotationMatrix() const;
        // 世界坐标下的逆转动惯量，用于冲量/流体压力改变角速度。
        Eigen::Matrix3f GetWorldInertiaInv() const;
        // box 的 8 个世界角点；主要用于 AABB 和调试绘制，BoatHull 不走这个。
        std::array<Eigen::Vector3f, 8> GetWorldCorners() const;
        // 返回世界空间 AABB；coupler 用它缩小刚体和流体网格的搜索范围。
        std::pair<Eigen::Vector3f, Eigen::Vector3f> GetWorldAABB() const;

        // 世界点转刚体局部坐标。做几何查询时通常先调用它。
        Eigen::Vector3f WorldToLocal(Eigen::Vector3f const & p) const;
        // 局部点转世界坐标。表面点、采样点写回场景时用。
        Eigen::Vector3f LocalToWorld(Eigen::Vector3f const & p) const;
        // 为 BoatHull 预计算局部空间 SDF；box/sphere 不需要，直接用解析公式。
        // maxResolution 控制最长轴采样数；padding < 0 时自动按薄壳厚度扩边。
        void            BuildLocalSdfGrid(int maxResolution = 48, float padding = -1.0f);
        // 查询局部坐标点到刚体表面的 signed distance，<=0 表示在有效固体区域内。
        float           LocalSignedDistance(Eigen::Vector3f const & localPoint) const;
        // 查询世界坐标点的 signed distance；内部会先转到刚体局部坐标。
        float           SignedDistance(Eigen::Vector3f const & worldPoint) const;
        // 只判断世界坐标点是否在刚体内；比 SignedDistance 更适合高频布尔查询。
        bool            ContainsPoint(Eigen::Vector3f const & worldPoint) const;
        // 返回世界坐标点在刚体表面的近似最近点，用于粒子推出/拾取等。
        Eigen::Vector3f ClosestSurfacePoint(Eigen::Vector3f const & worldPoint) const;
        // 返回表面法线；调用点通常应位于表面附近。
        Eigen::Vector3f SurfaceNormalAt(Eigen::Vector3f const & worldPoint) const;
    };

    class RigidBodySystem {
    public:
        std::vector<RigidBody>   Bodies;
        std::vector<RigidContact> Contacts;

        Eigen::Vector3f Gravity                      = Eigen::Vector3f(0.0f, -9.8f, 0.0f);
        int             ImpulseIterations            = 10; // 接触冲量迭代次数
        int             Substeps                     = 6;  // 刚体内部子步
        float           LinearDamping                = 0.01f;
        float           AngularDamping               = 0.01f;
        float           RestitutionVelocityThreshold = 0.25f;
        float           PositionCorrectionPercent    = 0.70f;
        float           PositionCorrectionSlop       = 0.0025f;

        bool  EnableFriction              = true;
        bool  SortContactsForStability    = false; // 堆叠时先处理靠下的接触
        bool  EnableRestingStabilization  = false;
        bool  AlternateIterationSweep     = false; // 迭代时正反扫交替，少一点方向偏差
        bool  UseSequentialImpulseCaching = true;
        float RestingLinearThreshold      = 0.08f;
        float RestingAngularThreshold     = 0.12f;

        // 清空刚体和接触数据，通常用于彻底重建场景。
        void Clear();
        // 重置为默认场景配置。
        void Reset();
        // 添加一个刚体并返回它在 Bodies 里的 id；会同步更新质量/碰撞缓存。
        int AddBody(RigidBody body);
        // 按预设创建测试场景，包括水箱边界、box/sphere 或船体。
        void SetupDefaultScene(RigidBodyPreset preset = RigidBodyPreset::FluidCouplingMixed);

        // 刚体主更新：外力积分、FCL 碰撞检测、接触求解和位置修正。
        void Step(float dt, int substeps = -1);
        // 清掉所有 body.force/body.torque；每步结束或重置力时调用。
        void ClearForces();
        // 只做速度/位置积分，不做碰撞求解；Step 内部会调用。
        void Integrate(float dt);
        // 使用 FCL 找刚体之间的接触点，结果写入 Contacts。
        void DetectCollisionsFCL();
        // 对 Contacts 做冲量迭代，修正 v/w。
        void SolveContacts();
        // 对穿透做位置级别修正，减少堆叠下沉。
        void PositionalCorrection();
        // 对近似静止接触做额外稳定化，减少小抖动。
        void StabilizeRestingContacts();
        // 把刚体限制在水箱范围内，并生成/处理墙面接触。
        void ResolveTankBounds(float minBound, float maxBound);

        // 注意这里传的是作用点，不是默认打在质心上；鼠标拖拽和水压力都要用这个。
        void ApplyForce(int id, Eigen::Vector3f const & forceWorld, Eigen::Vector3f const & worldPoint);
        void ApplyForceToCenter(int id, Eigen::Vector3f const & forceWorld);
        void ApplyTorque(int id, Eigen::Vector3f const & torqueWorld);
        void ApplyImpulse(int id, Eigen::Vector3f const & impulseWorld, Eigen::Vector3f const & worldPoint);
        void ApplyImpulseToCenter(int id, Eigen::Vector3f const & impulseWorld);

        // 返回第一个非静态刚体 id，找不到则返回 -1。
        int GetFirstDynamicBody() const;
        // 查询刚体上某个世界点的速度 v + w x r，流固边界速度会用它。
        Eigen::Vector3f VelocityAtPoint(int id, Eigen::Vector3f const & worldPoint) const;
        // 读取/写回 [v, w]，供流固耦合投影统一修正刚体速度。
        RigidGeneralizedVelocity GetGeneralizedVelocity(int id) const;
        void SetGeneralizedVelocity(int id, RigidGeneralizedVelocity const & velocity);
        // 返回 6x6 的 M_s^-1，用于组装刚体项 J M_s^-1 J^T。
        RigidInverseMassBlock GetInverseMassBlock(int id) const;
        // 采样刚体表面点；之后如果做压力积分/浮力细化，可以直接遍历这些点。
        void            CollectSurfaceSamples(int bodyId, int samplesPerAxis, std::vector<RigidSurfaceSample> & samples) const;
        // id 是否落在 Bodies 有效范围内。
        bool            IsValidBody(int id) const;
        // 区分水箱墙体和普通动态刚体，coupler 不应把 tank 当成可动固体求解。
        bool            IsInternalTankBoundary(int id) const;

        // 修改质量并刷新 invMass/惯量；静态刚体会保持零 invMass。
        void SetBodyMass(int id, float mass);
        // 修改尺寸并刷新质量属性和碰撞几何缓存。
        void SetBodyDim(int id, Eigen::Vector3f const & dim);
        // 设置是否静态；静态刚体不响应力/冲量。
        void SetBodyStatic(int id, bool isStatic);
        // 设置是否受重力；水箱墙体等边界通常关闭。
        void SetBodyGravity(int id, bool useGravity);
        // 修改几何类型；切走 BoatHull 时会清掉局部 SDF。
        void SetBodyShape(int id, RigidBodyShape shape);

    private:
        using CollisionGeometryPtr = std::shared_ptr<fcl::CollisionGeometry<float>>;

        void collisionDetectPairFCL(int idA, int idB);
        void sortContactsForStability();
        void resolveVelocityContact(RigidContact & contact);

        void InvalidateCollisionGeometryCache();
        void EnsureCollisionGeometryCache();
        CollisionGeometryPtr const & CollisionGeometryAt(int id);

        // 只缓存几何，不缓存位姿。刚体每帧的 x/q 仍然在 CollisionObject 里更新。
        std::vector<CollisionGeometryPtr> _collisionGeometryCache;
        bool                              _collisionGeometryCacheDirty = true;
    };

    glm::vec3        ToGlm(Eigen::Vector3f const & v);
    Eigen::Vector3f ToEigen(glm::vec3 const & v);

} // namespace VCX::MainScene
