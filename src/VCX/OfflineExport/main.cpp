#include "Simulation/SimulationWorld.h"

#include <Eigen/Core>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
    using VCX::MainScene::FluidSurfaceMesh;
    using VCX::MainScene::RigidBody;
    using VCX::MainScene::RigidBodyPreset;
    using VCX::MainScene::RigidBodyShape;
    using VCX::MainScene::SimulationWorld;

    struct Options {
        int   frames              = 181;
        int   fps                 = 30;
        int   res                 = 20;
        int   preset              = 3;
        int   stepSubsteps        = 2;
        bool  exportParticles     = true;
        bool  enableBoatBuoyancy  = true;
        bool  enableSurfaceModel  = false;
        bool  enableVariational   = false;
        std::string surfaceMethod = "sdf"; // sdf is the new offline default; density keeps the old look.
        float surfaceScale        = 2.8f;
        float surfaceIso          = 0.40f;
        float surfaceBlur         = 3.0f;
        float surfaceKernelScale  = 2.45f;
        float surfaceSdfRadiusScale = 1.85f;
        int   surfaceSdfSmooth      = 1;
        float dt                  = 0.0f; // 0 means 1/fps.
        fs::path out              = "exports/blender_offline";
    };

    std::string ToLower(std::string s) {
        for (char & c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }

    std::string Sanitize(std::string s) {
        if (s.empty()) return "body";
        for (char & c : s) {
            bool const ok = std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.';
            if (! ok) c = '_';
        }
        return s;
    }

    std::string FrameName(int frame) {
        std::ostringstream oss;
        oss << std::setw(6) << std::setfill('0') << frame;
        return oss.str();
    }

    glm::vec3 ToGlm(Eigen::Vector3f const & v) {
        return glm::vec3(v.x(), v.y(), v.z());
    }

    Eigen::Vector3f ToEigen(glm::vec3 const & v) {
        return Eigen::Vector3f(v.x, v.y, v.z);
    }

    // Coordinate convention fix for Blender offline rendering.
    //
    // The simulator is Y-up: gravity is (0, -9.81, 0), water height is the
    // Y coordinate, and the tank bounds are [-0.5, 0.5]^3 in simulation space.
    // Blender is normally Z-up.  The previous exporter wrote simulation
    // coordinates directly and the Blender script then treated Y as vertical;
    // this made the liquid read like a vertical curtain in the final image.
    //
    // We now export all geometry in Blender's standard Z-up coordinates:
    //     sim (x, y_up, z_depth) -> blender (x, -z_depth, y_up)
    // The minus sign keeps the transform right-handed, so face winding and
    // normals remain consistent.
    glm::vec3 SimToBlender(glm::vec3 const & p) {
        return glm::vec3(p.x, -p.z, p.y);
    }

    std::vector<glm::vec3> SimToBlender(std::vector<glm::vec3> const & points) {
        std::vector<glm::vec3> out;
        out.reserve(points.size());
        for (glm::vec3 const & p : points) out.push_back(SimToBlender(p));
        return out;
    }

    struct Bounds {
        glm::vec3 min { std::numeric_limits<float>::max() };
        glm::vec3 max { std::numeric_limits<float>::lowest() };
        bool valid = false;

        void Add(glm::vec3 const & p) {
            min = valid ? glm::min(min, p) : p;
            max = valid ? glm::max(max, p) : p;
            valid = true;
        }
    };

    void WritePlyMesh(
        fs::path const &                  path,
        std::vector<glm::vec3> const &    positions,
        std::vector<glm::vec3> const &    normals,
        std::vector<std::uint32_t> const &indices) {
        std::ofstream out(path, std::ios::binary);
        if (! out) throw std::runtime_error("Cannot write " + path.string());

        bool const hasNormals = normals.size() == positions.size();
        out << "ply\n";
        out << "format ascii 1.0\n";
        out << "comment exported by FluidSimulator offline-export\n";
        out << "element vertex " << positions.size() << "\n";
        out << "property float x\nproperty float y\nproperty float z\n";
        if (hasNormals) out << "property float nx\nproperty float ny\nproperty float nz\n";
        out << "element face " << indices.size() / 3 << "\n";
        out << "property list uchar uint vertex_indices\n";
        out << "end_header\n";
        out << std::fixed << std::setprecision(7);
        for (std::size_t i = 0; i < positions.size(); ++i) {
            glm::vec3 const p = SimToBlender(positions[i]);
            out << p.x << ' ' << p.y << ' ' << p.z;
            if (hasNormals) {
                glm::vec3 const n = SimToBlender(normals[i]);
                out << ' ' << n.x << ' ' << n.y << ' ' << n.z;
            }
            out << '\n';
        }
        for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
            out << "3 " << indices[i] << ' ' << indices[i + 1] << ' ' << indices[i + 2] << '\n';
        }
    }

    void WritePlyPoints(fs::path const & path, std::vector<glm::vec3> const & positions) {
        std::ofstream out(path, std::ios::binary);
        if (! out) throw std::runtime_error("Cannot write " + path.string());
        out << "ply\n";
        out << "format ascii 1.0\n";
        out << "comment exported by FluidSimulator offline-export\n";
        out << "element vertex " << positions.size() << "\n";
        out << "property float x\nproperty float y\nproperty float z\n";
        out << "end_header\n";
        out << std::fixed << std::setprecision(7);
        for (glm::vec3 const & p0 : positions) {
            glm::vec3 const p = SimToBlender(p0);
            out << p.x << ' ' << p.y << ' ' << p.z << '\n';
        }
    }

    void AddTri(std::vector<std::uint32_t> & indices, std::uint32_t a, std::uint32_t b, std::uint32_t c) {
        indices.push_back(a);
        indices.push_back(b);
        indices.push_back(c);
    }

    void BuildBoxMesh(RigidBody const & body, std::vector<glm::vec3> & pos, std::vector<std::uint32_t> & idx) {
        Eigen::Vector3f const h = 0.5f * body.dim;
        std::array<Eigen::Vector3f, 8> const local {
            Eigen::Vector3f(-h.x(), -h.y(), -h.z()), Eigen::Vector3f( h.x(), -h.y(), -h.z()),
            Eigen::Vector3f( h.x(),  h.y(), -h.z()), Eigen::Vector3f(-h.x(),  h.y(), -h.z()),
            Eigen::Vector3f(-h.x(), -h.y(),  h.z()), Eigen::Vector3f( h.x(), -h.y(),  h.z()),
            Eigen::Vector3f( h.x(),  h.y(),  h.z()), Eigen::Vector3f(-h.x(),  h.y(),  h.z())
        };
        pos.reserve(8);
        for (auto const & p : local) pos.push_back(ToGlm(body.LocalToWorld(p)));
        AddTri(idx, 0, 2, 1); AddTri(idx, 0, 3, 2);
        AddTri(idx, 4, 5, 6); AddTri(idx, 4, 6, 7);
        AddTri(idx, 0, 1, 5); AddTri(idx, 0, 5, 4);
        AddTri(idx, 3, 6, 2); AddTri(idx, 3, 7, 6);
        AddTri(idx, 1, 2, 6); AddTri(idx, 1, 6, 5);
        AddTri(idx, 0, 4, 7); AddTri(idx, 0, 7, 3);
    }

    void BuildSphereMesh(RigidBody const & body, std::vector<glm::vec3> & pos, std::vector<std::uint32_t> & idx) {
        int const rings = 16;
        int const segs  = 32;
        float const r   = 0.5f * body.dim.x();
        pos.reserve(2 + (rings - 1) * segs);

        pos.push_back(ToGlm(body.LocalToWorld(Eigen::Vector3f(0.0f, r, 0.0f))));
        for (int y = 1; y < rings; ++y) {
            float const v     = float(y) / float(rings);
            float const theta = 3.14159265358979323846f * v;
            for (int x = 0; x < segs; ++x) {
                float const u   = float(x) / float(segs);
                float const phi = 2.0f * 3.14159265358979323846f * u;
                Eigen::Vector3f local(
                    r * std::sin(theta) * std::cos(phi),
                    r * std::cos(theta),
                    r * std::sin(theta) * std::sin(phi));
                pos.push_back(ToGlm(body.LocalToWorld(local)));
            }
        }
        std::uint32_t const southPole = static_cast<std::uint32_t>(pos.size());
        pos.push_back(ToGlm(body.LocalToWorld(Eigen::Vector3f(0.0f, -r, 0.0f))));

        auto ringVertex = [&](int ring, int segment) {
            return std::uint32_t(1 + (ring - 1) * segs + (segment % segs));
        };

        for (int x = 0; x < segs; ++x) {
            int const next = (x + 1) % segs;
            AddTri(idx, 0, ringVertex(1, next), ringVertex(1, x));
        }

        for (int y = 1; y < rings - 1; ++y) {
            for (int x = 0; x < segs; ++x) {
                int const next = (x + 1) % segs;
                AddTri(idx, ringVertex(y, x), ringVertex(y + 1, next), ringVertex(y + 1, x));
                AddTri(idx, ringVertex(y, x), ringVertex(y, next), ringVertex(y + 1, next));
            }
        }

        for (int x = 0; x < segs; ++x) {
            int const next = (x + 1) % segs;
            AddTri(idx, southPole, ringVertex(rings - 1, x), ringVertex(rings - 1, next));
        }
    }

    void BuildRigidBodyMesh(RigidBody const & body, std::vector<glm::vec3> & pos, std::vector<std::uint32_t> & idx) {
        pos.clear();
        idx.clear();
        if (body.shape == RigidBodyShape::BoatHull && ! body.meshVertices.empty() && body.meshTriIndices.size() >= 3) {
            pos.reserve(body.meshVertices.size());
            for (Eigen::Vector3f const & p : body.meshVertices)
                pos.push_back(ToGlm(body.LocalToWorld(p)));
            idx = body.meshTriIndices;
            return;
        }
        if (body.shape == RigidBodyShape::Sphere) {
            BuildSphereMesh(body, pos, idx);
            return;
        }
        BuildBoxMesh(body, pos, idx);
    }

    std::vector<glm::vec3> ComputeVertexNormals(std::vector<glm::vec3> const & pos, std::vector<std::uint32_t> const & idx) {
        std::vector<glm::vec3> n(pos.size(), glm::vec3(0.0f));
        for (std::size_t i = 0; i + 2 < idx.size(); i += 3) {
            std::uint32_t const ia = idx[i + 0], ib = idx[i + 1], ic = idx[i + 2];
            if (ia >= pos.size() || ib >= pos.size() || ic >= pos.size()) continue;
            glm::vec3 const e1 = pos[ib] - pos[ia];
            glm::vec3 const e2 = pos[ic] - pos[ia];
            glm::vec3 const nn = glm::cross(e1, e2);
            n[ia] += nn; n[ib] += nn; n[ic] += nn;
        }
        for (glm::vec3 & v : n) {
            float const len = glm::length(v);
            v = (len > 1e-7f) ? v / len : glm::vec3(0.0f, 1.0f, 0.0f);
        }
        return n;
    }

    RigidBodyPreset ParsePreset(std::string s) {
        s = ToLower(s);
        if (s == "0" || s == "mixed" || s == "fluid") return RigidBodyPreset::FluidCouplingMixed;
        if (s == "1" || s == "box") return RigidBodyPreset::BoxCollision;
        if (s == "2" || s == "stack") return RigidBodyPreset::MixedStack;
        if (s == "3" || s == "boat" || s == "boatinwater") return RigidBodyPreset::BoatInWater;
        if (s == "4" || s == "boatdrop" || s == "boatdroppool" || s == "boatfall") return RigidBodyPreset::BoatDropIntoPool;
        if (s == "5" || s == "blob" || s == "surfaceblob" || s == "tension" || s == "surfacetensionblob") return RigidBodyPreset::SurfaceTensionBlob;
        return RigidBodyPreset::BoatInWater;
    }

    void PrintHelp(char const * exe) {
        std::cout << "Usage: " << exe << " [options]\n"
                  << "  --out PATH                 Output folder, default exports/blender_offline\n"
                  << "  --frames N                 Number of frames, default 181\n"
                  << "  --fps N                    Playback FPS, default 30\n"
                  << "  --res N                    Fluid resolution, default 20\n"
                  << "  --preset boat|mixed|box|stack|boatdrop|blob|0..5, default boat\n"
                  << "  --step-substeps N          Simulation steps per exported frame, default 2\n"
                  << "  --dt SECONDS               Physical time per exported frame. Default 1/fps\n"
                  << "  --surface-method sdf|density  Offline surface reconstruction, default sdf\n"
                  << "  --surface-scale F          Render surface grid scale, default 2.8\n"
                  << "  --surface-iso F            Density-mode iso value, default 0.40\n"
                  << "  --surface-blur N           Density-mode blur iterations, default 3\n"
                  << "  --surface-kernel-scale F   Density-mode kernel radius in multiples of grid h, default 2.45\n"
                  << "  --surface-sdf-radius-scale F SDF particle radius in multiples of simulation particle radius, default 1.85\n"
                  << "  --surface-sdf-smooth N      SDF-mode phi smoothing iterations, default 1\n"
                  << "  --no-particles             Do not export particle fallback PLYs\n"
                  << "  --no-boat-buoyancy         Disable boat buoyancy helper\n";
    }

    Options ParseArgs(int argc, char ** argv) {
        Options opt;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto needValue = [&](std::string const & name) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("Missing value after " + name);
                return argv[++i];
            };
            if (a == "--help" || a == "-h") { PrintHelp(argv[0]); std::exit(0); }
            else if (a == "--out") opt.out = needValue(a);
            else if (a == "--frames") opt.frames = std::max(1, std::stoi(needValue(a)));
            else if (a == "--fps") opt.fps = std::max(1, std::stoi(needValue(a)));
            else if (a == "--res") opt.res = std::max(4, std::stoi(needValue(a)));
            else if (a == "--preset") opt.preset = static_cast<int>(ParsePreset(needValue(a)));
            else if (a == "--step-substeps") opt.stepSubsteps = std::max(1, std::stoi(needValue(a)));
            else if (a == "--dt") opt.dt = std::max(0.0f, std::stof(needValue(a)));
            else if (a == "--surface-method") opt.surfaceMethod = ToLower(needValue(a));
            else if (a == "--surface-scale") opt.surfaceScale = std::max(1.0f, std::stof(needValue(a)));
            else if (a == "--surface-iso") opt.surfaceIso = std::stof(needValue(a));
            else if (a == "--surface-blur") opt.surfaceBlur = std::max(0.0f, std::stof(needValue(a)));
            else if (a == "--surface-kernel-scale") opt.surfaceKernelScale = std::max(0.5f, std::stof(needValue(a)));
            else if (a == "--surface-sdf-radius-scale") opt.surfaceSdfRadiusScale = std::max(0.5f, std::stof(needValue(a)));
            else if (a == "--surface-sdf-smooth") opt.surfaceSdfSmooth = std::max(0, std::stoi(needValue(a)));
            else if (a == "--no-particles") opt.exportParticles = false;
            else if (a == "--no-boat-buoyancy") opt.enableBoatBuoyancy = false;
            else if (a == "--enable-surface-tension") opt.enableSurfaceModel = true;
            else if (a == "--variational") opt.enableVariational = true;
            else std::cerr << "[OfflineExport] ignoring unknown option: " << a << "\n";
        }
        return opt;
    }

    void WriteMetadata(fs::path const & root, Options const & opt) {
        std::ofstream out(root / "metadata.json");
        out << std::fixed << std::setprecision(6);
        out << "{\n";
        out << "  \"format\": \"FluidSimulatorOfflineExport-v1\",\n";
        out << "  \"frames\": " << opt.frames << ",\n";
        out << "  \"fps\": " << opt.fps << ",\n";
        out << "  \"res\": " << opt.res << ",\n";
        out << "  \"preset\": " << opt.preset << ",\n";
        out << "  \"coordinate_system\": \"blender_z_up\",\n";
        out << "  \"axis_mapping\": \"sim(x,y,z)->blender(x,-z,y)\",\n";
        out << "  \"surface_method\": \"" << opt.surfaceMethod << "\",\n";
        out << "  \"surface_sdf_radius_scale\": " << opt.surfaceSdfRadiusScale << ",\n";
        out << "  \"surface_sdf_smooth\": " << opt.surfaceSdfSmooth << ",\n";
        out << "  \"bounds\": { \"min\": [-0.5, -0.5, -0.5], \"max\": [0.5, 0.5, 0.5] },\n";
        out << "  \"fluid_pattern\": \"frames/fluid_######.ply\",\n";
        out << "  \"particle_pattern\": \"frames/particles_######.ply\",\n";
        out << "  \"rigid_pattern\": \"frames/rigid_######_*.ply\"\n";
        out << "}\n";
    }

    void ExportFrame(SimulationWorld & world, fs::path const & framesDir, int frame, bool exportParticles) {
        std::string const f = FrameName(frame);
        auto & fluid = world.GetFluid();
        fluid.updateRenderableSurface();
        FluidSurfaceMesh const & mesh = fluid.GetRenderableSurface();
        WritePlyMesh(framesDir / ("fluid_" + f + ".ply"), mesh.positions, mesh.normals, mesh.indices);
        if (exportParticles)
            WritePlyPoints(framesDir / ("particles_" + f + ".ply"), fluid.m_particlePos);

        auto const & rigid = world.GetRigidBodies();
        int exportId = 0;
        for (int i = 0; i < static_cast<int>(rigid.Bodies.size()); ++i) {
            if (rigid.IsInternalTankBoundary(i)) continue;
            RigidBody const & body = rigid.Bodies[i];
            std::vector<glm::vec3> positions;
            std::vector<std::uint32_t> indices;
            BuildRigidBodyMesh(body, positions, indices);
            auto normals = ComputeVertexNormals(positions, indices);
            std::ostringstream name;
            name << "rigid_" << f << "_" << std::setw(3) << std::setfill('0') << exportId << "_" << Sanitize(body.name) << ".ply";
            WritePlyMesh(framesDir / name.str(), positions, normals, indices);
            ++exportId;
        }
    }
}

int main(int argc, char ** argv) {
    try {
        Options opt = ParseArgs(argc, argv);
        fs::path framesDir = opt.out / "frames";
        fs::remove_all(opt.out);
        fs::create_directories(framesDir);

        SimulationWorld world;
        world.SetRigidPreset(ParsePreset(std::to_string(opt.preset)));
        world.Setup(opt.res);
        world.SetSurfaceModelingEnabled(opt.enableSurfaceModel);
        world.GetCoupler().enableBoatBuoyancy = opt.enableBoatBuoyancy;
        world.GetCoupler().enableVariationalProjection = opt.enableVariational;
        world.GetFluid().m_renderSurfaceResolutionScale = opt.surfaceScale;
        world.GetFluid().m_renderSurfaceIsoValue = opt.surfaceIso;
        world.GetFluid().m_renderSurfaceBlurIters = static_cast<int>(std::round(opt.surfaceBlur));
        world.GetFluid().m_renderSurfaceKernelRadius = opt.surfaceKernelScale * world.GetFluid().m_h;
        world.GetFluid().m_renderSurfaceUseParticleSdf = (opt.surfaceMethod != "density");
        world.GetFluid().m_renderSurfaceSdfParticleRadius = opt.surfaceSdfRadiusScale * world.GetFluid().m_particleRadius;
        world.GetFluid().m_renderSurfaceSdfSmoothIters = opt.surfaceSdfSmooth;
        world.GetFluid().m_renderSurfaceUpdateInterval = 1;
        world.GetFluid().EnsureRenderableSurfaceFields();

        WriteMetadata(opt.out, opt);

        float const frameDt = opt.dt > 0.0f ? opt.dt : 1.0f / float(opt.fps);
        float const stepDt = frameDt / float(std::max(1, opt.stepSubsteps));

        std::cout << "[OfflineExport] out=" << opt.out.string()
                  << " frames=" << opt.frames
                  << " fps=" << opt.fps
                  << " res=" << opt.res
                  << " frameDt=" << frameDt
                  << " stepSubsteps=" << opt.stepSubsteps << "\n";

        for (int frame = 0; frame < opt.frames; ++frame) {
            if (frame == 0 || frame + 1 == opt.frames || frame % std::max(1, opt.frames / 20) == 0) {
                std::cout << "[OfflineExport] frame " << (frame + 1) << " / " << opt.frames << std::endl;
            }
            ExportFrame(world, framesDir, frame, opt.exportParticles);
            for (int s = 0; s < opt.stepSubsteps; ++s)
                world.Step(stepDt);
        }

        std::cout << "[OfflineExport] done. Render with tools/blender_render_sequence.py\n";
        return 0;
    } catch (std::exception const & e) {
        std::cerr << "[OfflineExport] ERROR: " << e.what() << "\n";
        return 1;
    }
}
