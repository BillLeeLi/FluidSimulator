#pragma once

#include <array>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glm/glm.hpp>

namespace VCX::MainScene {

    enum class RigidBodyShape {
        Box,
        Sphere,
    };

    enum class RigidBodyPreset {
        FluidCouplingMixed = 0,
        BoxCollision       = 1,
        MixedStack         = 2,
    };

    struct RigidContact {
        int             idA         = -1;
        int             idB         = -1;
        Eigen::Vector3f position    = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal      = Eigen::Vector3f::UnitY();
        float           penetration = 0.0f;

        // Sequential impulse caches used within one velocity solve.
        float           accumulatedNormalImpulse  = 0.0f;
        float           accumulatedTangentImpulse = 0.0f;
        Eigen::Vector3f tangent                   = Eigen::Vector3f::Zero();
        bool            tangentInitialized        = false;
    };

    struct RigidBody {
        RigidBodyShape     shape = RigidBodyShape::Box;
        Eigen::Vector3f    dim   = Eigen::Vector3f::Ones(); // box: full size; sphere: diameter in x/y/z
        Eigen::Vector3f    x     = Eigen::Vector3f::Zero();
        Eigen::Quaternionf q     = Eigen::Quaternionf::Identity();

        Eigen::Vector3f v = Eigen::Vector3f::Zero();
        Eigen::Vector3f w = Eigen::Vector3f::Zero();

        Eigen::Vector3f force  = Eigen::Vector3f::Zero();
        Eigen::Vector3f torque = Eigen::Vector3f::Zero();

        Eigen::Vector3f color = Eigen::Vector3f(0.75f, 0.75f, 0.8f);

        float       mass        = 1.0f;
        float       invMass     = 1.0f;
        float       restitution = 0.15f;
        float       friction    = 0.45f;
        bool        isStatic    = false;
        bool        useGravity  = true;
        std::string name;

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
        int             ImpulseIterations            = 10;
        int             Substeps                     = 6;
        float           LinearDamping                = 0.01f;
        float           AngularDamping               = 0.01f;
        float           RestitutionVelocityThreshold = 0.25f;
        float           PositionCorrectionPercent    = 0.70f;
        float           PositionCorrectionSlop       = 0.0025f;

        bool  EnableFriction              = true;
        bool  SortContactsForStability    = false;
        bool  EnableRestingStabilization  = false;
        bool  AlternateIterationSweep     = false;
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

        void ApplyForce(int id, Eigen::Vector3f const & forceWorld, Eigen::Vector3f const & worldPoint);
        void ApplyForceToCenter(int id, Eigen::Vector3f const & forceWorld);
        void ApplyTorque(int id, Eigen::Vector3f const & torqueWorld);
        void ApplyImpulse(int id, Eigen::Vector3f const & impulseWorld, Eigen::Vector3f const & worldPoint);
        void ApplyImpulseToCenter(int id, Eigen::Vector3f const & impulseWorld);

        int             GetFirstDynamicBody() const;
        Eigen::Vector3f VelocityAtPoint(int id, Eigen::Vector3f const & worldPoint) const;
        bool            IsValidBody(int id) const;

        void SetBodyMass(int id, float mass);
        void SetBodyDim(int id, Eigen::Vector3f const & dim);
        void SetBodyStatic(int id, bool isStatic);
        void SetBodyGravity(int id, bool useGravity);
        void SetBodyShape(int id, RigidBodyShape shape);

    private:
        void collisionDetectPairFCL(int idA, int idB);
        void sortContactsForStability();
        void resolveVelocityContact(RigidContact & contact);
    };

    glm::vec3        ToGlm(Eigen::Vector3f const & v);
    Eigen::Vector3f ToEigen(glm::vec3 const & v);

} // namespace VCX::MainScene
