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
        BoatInWater       = 3,
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

        float                          GetBoundingSphereRadius() const;
        void                           UpdateMassProperties();
        Eigen::Matrix3f                GetRotationMatrix() const;
        Eigen::Matrix3f                GetWorldInertiaInv() const;
        std::array<Eigen::Vector3f, 8> GetWorldCorners() const;
        std::pair<Eigen::Vector3f, Eigen::Vector3f> GetWorldAABB() const;

        Eigen::Vector3f WorldToLocal(Eigen::Vector3f const & p) const;
        Eigen::Vector3f LocalToWorld(Eigen::Vector3f const & p) const;
        bool            ContainsPoint(Eigen::Vector3f const & worldPoint) const;
        Eigen::Vector3f ClosestSurfacePoint(Eigen::Vector3f const & worldPoint) const;
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

        void Clear();
        void Reset();
        int  AddBody(RigidBody body);
        void SetupDefaultScene(RigidBodyPreset preset = RigidBodyPreset::FluidCouplingMixed);

        void Step(float dt, int substeps = -1);
        void ClearForces();
        void Integrate(float dt);
        void DetectCollisionsFCL();
        void SolveContacts();
        void PositionalCorrection();
        void StabilizeRestingContacts();
        void ResolveTankBounds(float minBound, float maxBound);

        // 注意这里传的是作用点，不是默认打在质心上；鼠标拖拽和水压力都要用这个。
        void ApplyForce(int id, Eigen::Vector3f const & forceWorld, Eigen::Vector3f const & worldPoint);
        void ApplyForceToCenter(int id, Eigen::Vector3f const & forceWorld);
        void ApplyTorque(int id, Eigen::Vector3f const & torqueWorld);
        void ApplyImpulse(int id, Eigen::Vector3f const & impulseWorld, Eigen::Vector3f const & worldPoint);
        void ApplyImpulseToCenter(int id, Eigen::Vector3f const & impulseWorld);

        int             GetFirstDynamicBody() const;
        Eigen::Vector3f VelocityAtPoint(int id, Eigen::Vector3f const & worldPoint) const;
        // 后面算流体压力时可以直接遍历这些采样点。
        void            CollectSurfaceSamples(int bodyId, int samplesPerAxis, std::vector<RigidSurfaceSample> & samples) const;
        bool            IsValidBody(int id) const;
        bool            IsInternalTankBoundary(int id) const;

        void SetBodyMass(int id, float mass);
        void SetBodyDim(int id, Eigen::Vector3f const & dim);
        void SetBodyStatic(int id, bool isStatic);
        void SetBodyGravity(int id, bool useGravity);
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
