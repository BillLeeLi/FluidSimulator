#include "Simulation/FluidSimulator.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace VCX::MainScene {

    int FluidSimulator::index2GridOffset(glm::ivec3 idx) const {
        return idx.x * (m_iCellY * m_iCellZ) + idx.y * m_iCellZ + idx.z;
    }

    int FluidSimulator::clampSlot(int v, int hi) {
        return std::max(0, std::min(v, hi));
    }

    int FluidSimulator::clampCoord(int v, int lo, int hi) {
        return std::max(lo, std::min(v, hi));
    }

    glm::ivec3 FluidSimulator::worldToSlot(glm::vec3 const & p) const {
        glm::vec3 g = (p + glm::vec3(0.5f)) * m_fInvSpacing;
        return glm::ivec3(
            clampSlot(static_cast<int>(std::floor(g.x)), m_iCellX - 1),
            clampSlot(static_cast<int>(std::floor(g.y)), m_iCellY - 1),
            clampSlot(static_cast<int>(std::floor(g.z)), m_iCellZ - 1));
    }

    float FluidSimulator::weight(glm::vec3 gridPos, glm::vec3 partPos) const {
        glm::vec3 d = (partPos - gridPos) * m_fInvSpacing;
        float     w = 1.0f;
        for (int i = 0; i < 3; i++) {
            float ad = std::abs(d[i]);
            if (ad < 1.0f)
                w *= (1.0f - ad);
            else
                return 0.0f;
        }
        return w;
    }

    void FluidSimulator::buildHash() {
        std::fill(m_hashtableindex.begin(), m_hashtableindex.end(), 0);
        // 1. 统计每个格点中的粒子数
        for (int p = 0; p < m_iNumSpheres; ++p) {
            glm::ivec3 slot   = worldToSlot(m_particlePos[p]);
            m_particleSlot[p] = slot;
            int const id      = index2GridOffset(slot);
            ++m_hashtableindex[id + 1];
        }
        // 2. 前缀和 → 每个格点在哈希表中的起止范围
        for (int i = 1; i <= m_iNumCells; ++i)
            m_hashtableindex[i] += m_hashtableindex[i - 1];
        // 3. 散射粒子到哈希表
        std::vector<int> writeCursor = m_hashtableindex;
        for (int p = 0; p < m_iNumSpheres; ++p) {
            int const id                   = index2GridOffset(m_particleSlot[p]);
            m_hashtable[writeCursor[id]++] = p;
        }
    }

    bool FluidSimulator::isValidVelocity(int i, int j, int k, int dir) const {
        if (i < 0 || i >= m_iCellX || j < 0 || j >= m_iCellY || k < 0 || k >= m_iCellZ)
            return false;
        switch (dir) {
        case 0: // x-face: 有效范围 i∈[1, CX-2], j∈[0, CY-2], k∈[0, CZ-2]
            if (i == 0 || i >= m_iCellX - 1 || j >= m_iCellY - 1 || k >= m_iCellZ - 1) return false;
            break;
        case 1: // y-face: 有效范围 i∈[0, CX-2], j∈[1, CY-2], k∈[0, CZ-2]
            if (j == 0 || j >= m_iCellY - 1 || i >= m_iCellX - 1 || k >= m_iCellZ - 1) return false;
            break;
        case 2: // z-face: 有效范围 i∈[0, CX-2], j∈[0, CY-2], k∈[1, CZ-2]
            if (k == 0 || k >= m_iCellZ - 1 || i >= m_iCellX - 1 || j >= m_iCellY - 1) return false;
            break;
        default: return false;
        }
        return m_s[index2GridOffset(glm::ivec3(i, j, k))] > 0.5f;
    }

    void FluidSimulator::integrateParticles(float timeStep) {
        for (int i = 0; i < m_iNumSpheres; i++) {
            m_particleVel[i] += gravity * timeStep;
            m_particlePos[i] += m_particleVel[i] * timeStep;
        }
    }

    void FluidSimulator::handleParticleCollisions(glm::vec3 obstaclePos, float obstacleRadius, glm::vec3 obstacleVel) {
        float const minBound = -0.5f + m_h + m_particleRadius;
        float const maxBound = 0.5f - m_h - m_particleRadius;

        for (int i = 0; i < m_iNumSpheres; i++) {
            // —— 障碍球碰撞 ——
            glm::vec3 diff = m_particlePos[i] - obstaclePos;
            float     dist = glm::length(diff);
            float     minD = obstacleRadius + m_particleRadius;
            if (dist < minD && dist > 1e-8f) {
                glm::vec3 const n = diff / dist;
                // 将粒子推到障碍球表面外侧
                m_particlePos[i] = obstaclePos + n * minD;
                // 反射相对于障碍物的速度
                float vn = glm::dot(m_particleVel[i] - obstacleVel, n);
                if (vn < 0.0f)
                    m_particleVel[i] -= vn * n;
            }

            // —— 水槽六面边界 (Solid at i=0, i>=CX-2, etc.) ——
            // x方向
            if (m_particlePos[i].x < minBound) {
                m_particlePos[i].x = minBound;
                m_particleVel[i].x = 0.0f;
            } else if (m_particlePos[i].x > maxBound) {
                m_particlePos[i].x = maxBound;
                m_particleVel[i].x = 0.0f;
            }
            // y方向
            if (m_particlePos[i].y < minBound) {
                m_particlePos[i].y = minBound;
                m_particleVel[i].y = 0.0f;
            } else if (m_particlePos[i].y > maxBound) {
                m_particlePos[i].y = maxBound;
                m_particleVel[i].y = 0.0f;
            }
            // z方向
            if (m_particlePos[i].z < minBound) {
                m_particlePos[i].z = minBound;
                m_particleVel[i].z = 0.0f;
            } else if (m_particlePos[i].z > maxBound) {
                m_particlePos[i].z = maxBound;
                m_particleVel[i].z = 0.0f;
            }
        }
    }

    void FluidSimulator::pushParticlesApart(int numIters) {
        for (int iter = 0; iter < numIters; iter++) {
            buildHash(); // 每轮迭代重建哈希表以反映最新位置

            float const minDist = 2.0f * m_particleRadius;

            for (int i = 0; i < m_iNumSpheres; i++) {
                glm::ivec3 const slot = m_particleSlot[i];

                // 仅在当前粒子所在格点及其26个相邻格点中搜索
                for (int di = -1; di <= 1; di++) {
                    int ni = clampSlot(slot.x + di, m_iCellX - 1);
                    for (int dj = -1; dj <= 1; dj++) {
                        int nj = clampSlot(slot.y + dj, m_iCellY - 1);
                        for (int dk = -1; dk <= 1; dk++) {
                            int nk         = clampSlot(slot.z + dk, m_iCellZ - 1);
                            int neighborId = index2GridOffset(glm::ivec3(ni, nj, nk));

                            for (int idx = m_hashtableindex[neighborId];
                                 idx < m_hashtableindex[neighborId + 1];
                                 idx++) {
                                int j = m_hashtable[idx];
                                if (i >= j) continue; // 每对只处理一次

                                glm::vec3 dir = m_particlePos[i] - m_particlePos[j];
                                float     len = glm::length(dir);
                                if (len < minDist && len > 1e-8f) {
                                    glm::vec3 corr = 0.5f * (minDist - len) * dir / len;
                                    m_particlePos[i] += corr;
                                    m_particlePos[j] -= corr;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void FluidSimulator::transferVelocities(bool toGrid, float flipRatio) {
        if (toGrid) {
            // ============ P2G: 粒子 → 网格 ============

            // 保存当前网格速度用于FLIP增量
            std::fill(m_vel.begin(), m_vel.end(), glm::vec3(0.0f));
            for (int d = 0; d < 3; ++d)
                std::fill(m_near_num[d].begin(), m_near_num[d].end(), 0.0f);

            buildHash();

            // 辅助lambda: 在采样点samlePos处, 搜索邻域粒子并累积dir方向速度
            auto accumulateDir = [&](int dir, glm::vec3 const & samplePos, glm::ivec3 const & sampleSlot) {
                // 邻域搜索范围: ±1个格点
                int i0 = clampSlot(sampleSlot.x - 1, m_iCellX - 1);
                int j0 = clampSlot(sampleSlot.y - 1, m_iCellY - 1);
                int k0 = clampSlot(sampleSlot.z - 1, m_iCellZ - 1);
                int i1 = clampSlot(sampleSlot.x + 1, m_iCellX - 1);
                int j1 = clampSlot(sampleSlot.y + 1, m_iCellY - 1);
                int k1 = clampSlot(sampleSlot.z + 1, m_iCellZ - 1);

                int const sampleId = index2GridOffset(sampleSlot);

                for (int ii = i0; ii <= i1; ++ii) {
                    for (int jj = j0; jj <= j1; ++jj) {
                        for (int kk = k0; kk <= k1; ++kk) {
                            int const neighborId = index2GridOffset(glm::ivec3(ii, jj, kk));
                            for (int ptr = m_hashtableindex[neighborId];
                                 ptr < m_hashtableindex[neighborId + 1];
                                 ++ptr) {
                                int   p = m_hashtable[ptr];
                                float w = weight(samplePos, m_particlePos[p]);
                                if (w <= 0.0f) continue;
                                m_vel[sampleId][dir] += w * m_particleVel[p][dir];
                                m_near_num[dir][sampleId] += w;
                            }
                        }
                    }
                }
            };

            // —— 遍历所有x方向速度面 u(i, j+1/2, k+1/2) ——
            for (int i = 0; i < m_iCellX; ++i) {
                for (int j = 0; j < m_iCellY - 1; ++j) {
                    for (int k = 0; k < m_iCellZ - 1; ++k) {
                        if (! isValidVelocity(i, j, k, 0)) continue;
                        // x-速度面的物理位置
                        glm::vec3 samplePos = glm::vec3(i, j + 0.5f, k + 0.5f) * m_h + glm::vec3(-0.5f);
                        accumulateDir(0, samplePos, glm::ivec3(i, j, k));
                    }
                }
            }

            // —— 遍历所有y方向速度面 v(i+1/2, j, k+1/2) ——
            for (int i = 0; i < m_iCellX - 1; ++i) {
                for (int j = 0; j < m_iCellY; ++j) {
                    for (int k = 0; k < m_iCellZ - 1; ++k) {
                        if (! isValidVelocity(i, j, k, 1)) continue;
                        glm::vec3 samplePos = glm::vec3(i + 0.5f, j, k + 0.5f) * m_h + glm::vec3(-0.5f);
                        accumulateDir(1, samplePos, glm::ivec3(i, j, k));
                    }
                }
            }

            // —— 遍历所有z方向速度面 w(i+1/2, j+1/2, k) ——
            for (int i = 0; i < m_iCellX - 1; ++i) {
                for (int j = 0; j < m_iCellY - 1; ++j) {
                    for (int k = 0; k < m_iCellZ; ++k) {
                        if (! isValidVelocity(i, j, k, 2)) continue;
                        glm::vec3 samplePos = glm::vec3(i + 0.5f, j + 0.5f, k) * m_h + glm::vec3(-0.5f);
                        accumulateDir(2, samplePos, glm::ivec3(i, j, k));
                    }
                }
            }

            // 归一化: 每个速度面除以其累积权重
            for (int id = 0; id < m_iNumCells; ++id) {
                if (m_near_num[0][id] > 1e-6f) m_vel[id].x /= m_near_num[0][id];
                if (m_near_num[1][id] > 1e-6f) m_vel[id].y /= m_near_num[1][id];
                if (m_near_num[2][id] > 1e-6f) m_vel[id].z /= m_near_num[2][id];
            }

            m_pre_vel = m_vel;
        } else {
            // ============ G2P: 网格 → 粒子 ============

            // 辅助lambda: 在samplePos处采样dir方向的速度 (使用三线性插值)
            // 调用者负责将samplePos偏移到对应的交错速度面位置
            // useFlipDelta=true → 采样速度变化量 (m_vel-m_pre_vel) for FLIP
            // useFlipDelta=false → 采样当前速度 for PIC
            auto sampleVelocity = [&](glm::vec3 samplePos, int dir, bool useFlipDelta, float fallback) -> float {
                // 直接转换到网格坐标 (调用者已预处理偏移)
                glm::vec3 gridPos = (samplePos + glm::vec3(0.5f)) * m_fInvSpacing;

                // 钳制到有效插值范围
                gridPos.x = std::max(0.0f, std::min(gridPos.x, float(m_iCellX - 1) - 1e-4f));
                gridPos.y = std::max(0.0f, std::min(gridPos.y, float(m_iCellY - 1) - 1e-4f));
                gridPos.z = std::max(0.0f, std::min(gridPos.z, float(m_iCellZ - 1) - 1e-4f));

                int   i0 = static_cast<int>(std::floor(gridPos.x));
                int   j0 = static_cast<int>(std::floor(gridPos.y));
                int   k0 = static_cast<int>(std::floor(gridPos.z));
                float fx = gridPos.x - i0;
                float fy = gridPos.y - j0;
                float fz = gridPos.z - k0;

                float accum = 0.0f;
                float wsum  = 0.0f;
                for (int di = 0; di <= 1; ++di) {
                    float wx = di ? fx : (1.0f - fx);
                    for (int dj = 0; dj <= 1; ++dj) {
                        float wy = dj ? fy : (1.0f - fy);
                        for (int dk = 0; dk <= 1; ++dk) {
                            float      wz = dk ? fz : (1.0f - fz);
                            glm::ivec3 idx(i0 + di, j0 + dj, k0 + dk);
                            if (! isValidVelocity(idx.x, idx.y, idx.z, dir)) continue;
                            int   id  = index2GridOffset(idx);
                            float w   = wx * wy * wz;
                            float val = useFlipDelta ? (m_vel[id][dir] - m_pre_vel[id][dir])
                                                     : m_vel[id][dir];
                            accum += w * val;
                            wsum += w;
                        }
                    }
                }
                return (wsum > 1e-6f) ? (accum / wsum) : fallback;
            };

            for (int i = 0; i < m_iNumSpheres; i++) {
                glm::vec3 const oldVel = m_particleVel[i];

                // 将粒子位置偏移到交错面上的对应位置:
                // x-速度面 u 位于 (i, j+1/2, k+1/2) → 偏移 (0, -0.5h, -0.5h)
                // y-速度面 v 位于 (i+1/2, j, k+1/2) → 偏移 (-0.5h, 0, -0.5h)
                // z-速度面 w 位于 (i+1/2, j+1/2, k) → 偏移 (-0.5h, -0.5h, 0)
                glm::vec3 const px = m_particlePos[i] + glm::vec3(0.0f, -0.5f, -0.5f) * m_h;
                glm::vec3 const py = m_particlePos[i] + glm::vec3(-0.5f, 0.0f, -0.5f) * m_h;
                glm::vec3 const pz = m_particlePos[i] + glm::vec3(-0.5f, -0.5f, 0.0f) * m_h;

                // PIC: 直接从网格插值得到新速度 (耗散但稳定)
                glm::vec3 pic_vel;
                pic_vel.x = sampleVelocity(px, 0, false, oldVel.x);
                pic_vel.y = sampleVelocity(py, 1, false, oldVel.y);
                pic_vel.z = sampleVelocity(pz, 2, false, oldVel.z);

                // FLIP: 旧速度 + 网格速度变化量 (非耗散但可能有噪声)
                glm::vec3 flip_vel = oldVel;
                flip_vel.x += sampleVelocity(px, 0, true, 0.0f);
                flip_vel.y += sampleVelocity(py, 1, true, 0.0f);
                flip_vel.z += sampleVelocity(pz, 2, true, 0.0f);

                // 混合: flipRatio控制FLIP比例, (1-flipRatio)控制PIC比例
                m_particleVel[i] = flipRatio * flip_vel + (1.0f - flipRatio) * pic_vel;
            }
        }
    }

    void FluidSimulator::updateParticleDensity() {
        std::fill(m_particleDensity.begin(), m_particleDensity.end(), 0.0f);

        for (int i = 0; i < m_iCellX; ++i) {
            for (int j = 0; j < m_iCellY; ++j) {
                for (int k = 0; k < m_iCellZ; ++k) {
                    int const       id         = index2GridOffset(glm::ivec3(i, j, k));
                    glm::vec3 const gridPos    = glm::vec3(i, j, k) * m_h + glm::vec3(-0.5f);
                    glm::ivec3      centerSlot = worldToSlot(gridPos);

                    // 在3×3×3邻域格点中搜索粒子
                    int i0 = clampSlot(centerSlot.x - 1, m_iCellX - 1);
                    int j0 = clampSlot(centerSlot.y - 1, m_iCellY - 1);
                    int k0 = clampSlot(centerSlot.z - 1, m_iCellZ - 1);
                    int i1 = clampSlot(centerSlot.x + 1, m_iCellX - 1);
                    int j1 = clampSlot(centerSlot.y + 1, m_iCellY - 1);
                    int k1 = clampSlot(centerSlot.z + 1, m_iCellZ - 1);

                    float density = 0.0f;
                    for (int ii = i0; ii <= i1; ++ii) {
                        for (int jj = j0; jj <= j1; ++jj) {
                            for (int kk = k0; kk <= k1; ++kk) {
                                int neighborId = index2GridOffset(glm::ivec3(ii, jj, kk));
                                for (int ptr = m_hashtableindex[neighborId];
                                     ptr < m_hashtableindex[neighborId + 1];
                                     ++ptr) {
                                    int p = m_hashtable[ptr];
                                    density += weight(gridPos, m_particlePos[p]);
                                }
                            }
                        }
                    }
                    m_particleDensity[id] = density;
                }
            }
        }

        // 同步更新单元格类型 (供 solveIncompressibility 使用)
        for (int cell = 0; cell < m_iNumCells; ++cell) {
            if (m_s[cell] <= 0.0f) {
                m_type[cell] = SOLID_CELL;
            } else {
                int const cnt = m_hashtableindex[cell + 1] - m_hashtableindex[cell];
                m_type[cell]  = (cnt > 0) ? FLUID_CELL : EMPTY_CELL;
            }
        }
    }

    void FluidSimulator::solveIncompressibility(int numIters, float dt, float overRelaxation, bool compensateDrift) {
        // 清零压力
        std::fill(m_p.begin(), m_p.end(), 0.0f);

        for (int iter = 0; iter < numIters; iter++) {
            for (int i = 0; i < m_iCellX; i++) {
                for (int j = 0; j < m_iCellY; j++) {
                    for (int k = 0; k < m_iCellZ; k++) {
                        int const id = index2GridOffset(glm::ivec3(i, j, k));
                        if (m_type[id] != FLUID_CELL) continue;

                        // 1. 计算该格点的速度散度 (右-左 + 上-下 + 前-后)
                        float divergence = 0.0f;
                        // 流入面 (左/下/后) 贡献为负
                        if (isValidVelocity(i, j, k, 0))
                            divergence -= m_vel[id].x;
                        if (isValidVelocity(i, j, k, 1))
                            divergence -= m_vel[id].y;
                        if (isValidVelocity(i, j, k, 2))
                            divergence -= m_vel[id].z;
                        // 流出面 (右/上/前) 贡献为正
                        if (isValidVelocity(i + 1, j, k, 0))
                            divergence += m_vel[index2GridOffset(glm::ivec3(i + 1, j, k))].x;
                        if (isValidVelocity(i, j + 1, k, 1))
                            divergence += m_vel[index2GridOffset(glm::ivec3(i, j + 1, k))].y;
                        if (isValidVelocity(i, j, k + 1, 2))
                            divergence += m_vel[index2GridOffset(glm::ivec3(i, j, k + 1))].z;

                        // 2. 过松弛 (加速收敛)
                        divergence *= overRelaxation;

                        // 3. 密度漂移补偿 (补偿因插值误差导致的体积变化)
                        if (compensateDrift) {
                            float densityError = m_particleDensity[id] - m_particleRestDensity;
                            divergence += 0.2f * densityError;
                        }

                        // 4. 计算6个邻居的流体权重和
                        //    固体邻居贡献 m_s=0, 自然不计入
                        float s = 0.0f;
                        s += m_s[index2GridOffset(glm::ivec3(i + 1, j, k))];
                        s += m_s[index2GridOffset(glm::ivec3(i - 1, j, k))];
                        s += m_s[index2GridOffset(glm::ivec3(i, j - 1, k))];
                        s += m_s[index2GridOffset(glm::ivec3(i, j + 1, k))];
                        s += m_s[index2GridOffset(glm::ivec3(i, j, k + 1))];
                        s += m_s[index2GridOffset(glm::ivec3(i, j, k - 1))];

                        if (s < 1e-6f) continue;

                        // 5. 将散度修正按邻居权重分配到各速度面
                        //    流体邻居权重高 → 承担更多修正; 固体邻居 → 不修正
                        if (isValidVelocity(i, j, k, 0))
                            m_vel[id].x += divergence * m_s[index2GridOffset(glm::ivec3(i - 1, j, k))] / s;
                        if (isValidVelocity(i, j, k, 1))
                            m_vel[id].y += divergence * m_s[index2GridOffset(glm::ivec3(i, j - 1, k))] / s;
                        if (isValidVelocity(i, j, k, 2))
                            m_vel[id].z += divergence * m_s[index2GridOffset(glm::ivec3(i, j, k - 1))] / s;

                        if (isValidVelocity(i + 1, j, k, 0))
                            m_vel[index2GridOffset(glm::ivec3(i + 1, j, k))].x -= divergence * m_s[index2GridOffset(glm::ivec3(i + 1, j, k))] / s;
                        if (isValidVelocity(i, j + 1, k, 1))
                            m_vel[index2GridOffset(glm::ivec3(i, j + 1, k))].y -= divergence * m_s[index2GridOffset(glm::ivec3(i, j + 1, k))] / s;
                        if (isValidVelocity(i, j, k + 1, 2))
                            m_vel[index2GridOffset(glm::ivec3(i, j, k + 1))].z -= divergence * m_s[index2GridOffset(glm::ivec3(i, j, k + 1))] / s;

                        // 6. 累积压力 (用于可视化和分析)
                        m_p[id] += divergence * m_h / (s * dt);
                    }
                }
            }
        }
    }

    void FluidSimulator::updateParticleColors() {
        if (m_iNumSpheres <= 0) return;

        // 颜色渐变: 低速/低压(蓝) → 中速/中压(青) → 高速/高压(红)
        auto const ramp = [](float t) {
            t = glm::clamp(t, 0.0f, 1.0f);
            glm::vec3 const cLow(0.10f, 0.25f, 0.95f); // 蓝色 (低)
            glm::vec3 const cMid(0.10f, 0.90f, 0.85f); // 青色 (中)
            glm::vec3 const cHi(1.00f, 0.30f, 0.05f);  // 红色 (高)
            if (t < 0.5f)
                return glm::mix(cLow, cMid, t * 2.0f);
            return glm::mix(cMid, cHi, (t - 0.5f) * 2.0f);
        };

        // 使用三线性插值从网格压力场中采样每个粒子所在位置的压力
        // 用于视觉反馈 (检查压力求解是否正常工作)
        float maxPressure = 1e-6f;
        for (int id = 0; id < m_iNumCells; ++id)
            maxPressure = std::max(maxPressure, std::abs(m_p[id]));

        for (int i = 0; i < m_iNumSpheres; ++i) {
            // 仅从有效流体单元采样压力，避免把 solid/empty 的 0 压力混入边界粒子着色。
            glm::vec3 g  = (m_particlePos[i] + glm::vec3(0.5f)) * m_fInvSpacing;
            int       ix = clampCoord(static_cast<int>(std::floor(g.x)), 0, m_iCellX - 2);
            int       iy = clampCoord(static_cast<int>(std::floor(g.y)), 0, m_iCellY - 2);
            int       iz = clampCoord(static_cast<int>(std::floor(g.z)), 0, m_iCellZ - 2);
            float     fx = g.x - ix;
            float     fy = g.y - iy;
            float     fz = g.z - iz;

            int ix0 = ix, ix1 = std::min(ix + 1, m_iCellX - 1);
            int iy0 = iy, iy1 = std::min(iy + 1, m_iCellY - 1);
            int iz0 = iz, iz1 = std::min(iz + 1, m_iCellZ - 1);

            auto accumulatePressure = [&](int x, int y, int z, float w, float & sum, float & wsum) {
                int const id = index2GridOffset(glm::ivec3(x, y, z));
                if (m_type[id] != FLUID_CELL || w <= 0.0f) return;
                sum += w * m_p[id];
                wsum += w;
            };

            float pressureSum    = 0.0f;
            float pressureWeight = 0.0f;

            for (int dx = 0; dx <= 1; ++dx) {
                float wx = dx ? fx : (1.0f - fx);
                int   sx = dx ? ix1 : ix0;
                for (int dy = 0; dy <= 1; ++dy) {
                    float wy = dy ? fy : (1.0f - fy);
                    int   sy = dy ? iy1 : iy0;
                    for (int dz = 0; dz <= 1; ++dz) {
                        float wz = dz ? fz : (1.0f - fz);
                        int   sz = dz ? iz1 : iz0;
                        accumulatePressure(sx, sy, sz, wx * wy * wz, pressureSum, pressureWeight);
                    }
                }
            }

            float p = 0.0f;
            if (pressureWeight > 1e-6f) {
                p = pressureSum / pressureWeight;
            } else {
                int const centerId = index2GridOffset(worldToSlot(m_particlePos[i]));
                if (m_type[centerId] == FLUID_CELL) {
                    p = m_p[centerId];
                } else {
                    float nearestDist2 = std::numeric_limits<float>::max();
                    for (int dx = 0; dx <= 1; ++dx) {
                        int sx = dx ? ix1 : ix0;
                        for (int dy = 0; dy <= 1; ++dy) {
                            int sy = dy ? iy1 : iy0;
                            for (int dz = 0; dz <= 1; ++dz) {
                                int const sz = dz ? iz1 : iz0;
                                int const id = index2GridOffset(glm::ivec3(sx, sy, sz));
                                if (m_type[id] != FLUID_CELL) continue;

                                glm::vec3 const cellPos = glm::vec3(sx, sy, sz) * m_h + glm::vec3(-0.5f);
                                float const     dist2   = glm::dot(cellPos - m_particlePos[i], cellPos - m_particlePos[i]);
                                if (dist2 < nearestDist2) {
                                    nearestDist2 = dist2;
                                    p            = m_p[id];
                                }
                            }
                        }
                    }
                }
            }

            float t            = std::abs(p) / maxPressure;
            m_particleColor[i] = ramp(t);
        }
    }

    void FluidSimulator::SimulateTimestep(float const dt) {
        int   numSubSteps       = 1;
        int   numParticleIters  = 5;
        int   numPressureIters  = 30;
        bool  separateParticles = true;
        float overRelaxation    = 1.9f;
        bool  compensateDrift   = false;

        float     flipRatio = m_fRatio;
        glm::vec3 obstaclePos(0.0f);
        glm::vec3 obstacleVel(0.0f);

        float sdt = dt / numSubSteps;

        for (int step = 0; step < numSubSteps; step++) {
            integrateParticles(sdt);
            handleParticleCollisions(obstaclePos, 0.0f, obstacleVel);
            if (separateParticles)
                pushParticlesApart(numParticleIters);
            handleParticleCollisions(obstaclePos, 0.0f, obstacleVel);
            transferVelocities(true, flipRatio);
            updateParticleDensity();
            solveIncompressibility(numPressureIters, sdt, overRelaxation, compensateDrift);
            transferVelocities(false, flipRatio);
        }
        updateParticleColors();
    }

    void FluidSimulator::setupScene(int res) {
        glm::vec3 tank(1.0f);
        glm::vec3 relWater = { 0.6f, 0.8f, 0.6f };

        float _h      = tank.y / res;
        float point_r = 0.3f * _h;
        float dx      = 2.0f * point_r;
        float dy      = sqrt(3.0f) / 2.0f * dx;
        float dz      = dx;

        int numX = floor((relWater.x * tank.x - 2.0f * _h - 2.0f * point_r) / dx);
        int numY = floor((relWater.y * tank.y - 2.0f * _h - 2.0f * point_r) / dy);
        int numZ = floor((relWater.z * tank.z - 2.0f * _h - 2.0f * point_r) / dz);

        // 更新网格几何参数
        m_iNumSpheres         = numX * numY * numZ;
        m_iCellX              = res + 1;
        m_iCellY              = res + 1;
        m_iCellZ              = res + 1;
        m_h                   = 1.0f / float(res);
        m_fInvSpacing         = float(res);
        m_iNumCells           = m_iCellX * m_iCellY * m_iCellZ;
        m_particleRadius      = point_r;
        m_particleRestDensity = 1.0f;

        // 分配粒子数组
        m_particlePos.clear();
        m_particlePos.resize(m_iNumSpheres, glm::vec3(0.0f));
        m_particleVel.clear();
        m_particleVel.resize(m_iNumSpheres, glm::vec3(0.0f));
        m_particleColor.clear();
        m_particleColor.resize(m_iNumSpheres, glm::vec3(1.0f));
        m_hashtable.clear();
        m_hashtable.resize(m_iNumSpheres, 0);
        m_hashtableindex.clear();
        m_hashtableindex.resize(m_iNumCells + 1, 0);
        m_particleSlot.clear();
        m_particleSlot.resize(m_iNumSpheres, glm::ivec3(0));

        // 分配网格数组
        m_vel.clear();
        m_vel.resize(m_iNumCells, glm::vec3(0.0f));
        m_pre_vel.clear();
        m_pre_vel.resize(m_iNumCells, glm::vec3(0.0f));
        for (int i = 0; i < 3; ++i) {
            m_near_num[i].clear();
            m_near_num[i].resize(m_iNumCells, 0.0f);
        }

        m_p.clear();
        m_p.resize(m_iNumCells, 0.0);
        m_s.clear();
        m_s.resize(m_iNumCells, 0.0);
        m_type.clear();
        m_type.resize(m_iNumCells, 0);
        m_particleDensity.clear();
        m_particleDensity.resize(m_iNumCells, 0.0f);

        // 生成HCP排列的粒子
        int p = 0;
        for (int i = 0; i < numX; i++) {
            for (int j = 0; j < numY; j++) {
                for (int k = 0; k < numZ; k++) {
                    m_particlePos[p++] = glm::vec3(
                                             m_h + point_r + dx * i + (j % 2 == 0 ? 0.0f : point_r),
                                             m_h + point_r + dy * j,
                                             m_h + point_r + dz * k + (j % 2 == 0 ? 0.0f : point_r))
                        + glm::vec3(-0.5f);
                }
            }
        }

        // 设置网格固体标记: 水槽壁厚为1个格点
        int n = m_iCellY * m_iCellZ;
        int m = m_iCellZ;
        for (int i = 0; i < m_iCellX; i++) {
            for (int j = 0; j < m_iCellY; j++) {
                for (int k = 0; k < m_iCellZ; k++) {
                    float s = 1.0f; // 默认非固体
                    if (i == 0 || i == m_iCellX - 1
                        || j == 0 || j == m_iCellY - 1
                        || k == 0 || k == m_iCellZ - 1)
                        s = 0.0f; // 边界固体
                    m_s[i * n + j * m + k] = s;
                }
            }
        }

        // 构建初始哈希表并更新单元格类型
        buildHash();
        for (int cell = 0; cell < m_iNumCells; ++cell) {
            if (m_s[cell] <= 0.0f) {
                m_type[cell] = SOLID_CELL;
            } else {
                int const cnt = m_hashtableindex[cell + 1] - m_hashtableindex[cell];
                m_type[cell]  = (cnt > 0) ? FLUID_CELL : EMPTY_CELL;
            }
        }
    }

} // namespace VCX::MainScene
