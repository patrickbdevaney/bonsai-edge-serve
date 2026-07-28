// Host harness for the Vulkan low-bit GEMV shaders.
//
// Validates gemv_q2/gemv_q1 against the same CPU reference the CUDA
// ladder uses, then times them. Deliberately minimal: no staging
// buffers, because Thor is a unified-memory part and every heap that
// matters is HOST_VISIBLE|DEVICE_LOCAL, so the shader reads the same
// pages the host wrote. On a discrete GPU this harness would need
// staging copies and the numbers would not be comparable.
//
// Build:
//   glslc -O gemv_q2.comp -o gemv_q2.spv          (portable path)
//   glslc -O -DUSE_INT_DOT gemv_q2.comp -o gemv_q2_dot.spv
//   g++ -O2 -o gemv_vk_bench gemv_vk_bench.cpp -lvulkan
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <vulkan/vulkan.h>
#include "../common/bonsai_gemv.h"

#define VK(x) do { VkResult r_=(x); if(r_!=VK_SUCCESS){ \
    fprintf(stderr,"%s:%d vk error %d\n",__FILE__,__LINE__,(int)r_); exit(1);} } while(0)

struct Ctx {
    VkInstance inst{};
    VkPhysicalDevice phys{};
    VkDevice dev{};
    VkQueue q{};
    uint32_t qfam{};
    VkCommandPool pool{};
    VkPhysicalDeviceMemoryProperties memp{};
};

struct Buf { VkBuffer b{}; VkDeviceMemory m{}; void *p{}; size_t bytes{}; };

static uint32_t pick_mem(Ctx &c, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c.memp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) &&
            (c.memp.memoryTypes[i].propertyFlags & want) == want) return i;
    fprintf(stderr, "no memory type with 0x%x\n", want); exit(1);
}

static Buf make_buf(Ctx &c, size_t bytes) {
    Buf b; b.bytes = bytes;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK(vkCreateBuffer(c.dev, &bi, nullptr, &b.b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c.dev, b.b, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = pick_mem(c, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK(vkAllocateMemory(c.dev, &ai, nullptr, &b.m));
    VK(vkBindBufferMemory(c.dev, b.b, b.m, 0));
    VK(vkMapMemory(c.dev, b.m, 0, VK_WHOLE_SIZE, 0, &b.p));
    return b;
}

static std::vector<uint32_t> read_spv(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> v(n / 4);
    if (fread(v.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char **argv) {
    const int K = (argc > 1) ? atoi(argv[1]) : 8192;
    const int N = (argc > 2) ? atoi(argv[2]) : 131072;
    const int iters = (argc > 3) ? atoi(argv[3]) : 30;
    const int nblk = K / BONSAI_QK;

    // ---------------- context ----------------
    Ctx c;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;
    VK(vkCreateInstance(&ii, nullptr, &c.inst));
    uint32_t nd = 0; vkEnumeratePhysicalDevices(c.inst, &nd, nullptr);
    std::vector<VkPhysicalDevice> devs(nd);
    vkEnumeratePhysicalDevices(c.inst, &nd, devs.data());
    if (!nd) { fprintf(stderr, "no vulkan device\n"); return 1; }
    c.phys = devs[0];
    VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(c.phys, &pr);
    vkGetPhysicalDeviceMemoryProperties(c.phys, &c.memp);

    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nq, qf.data());
    c.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.qfam = i; break; }
    if (c.qfam == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); return 1; }

    float prio = 1.f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = c.qfam; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    VK(vkCreateDevice(c.phys, &di, nullptr, &c.dev));
    vkGetDeviceQueue(c.dev, c.qfam, 0, &c.q);
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = c.qfam;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK(vkCreateCommandPool(c.dev, &pi, nullptr, &c.pool));

    printf("device %s  Vulkan %u.%u.%u\n", pr.deviceName,
           VK_VERSION_MAJOR(pr.apiVersion), VK_VERSION_MINOR(pr.apiVersion),
           VK_VERSION_PATCH(pr.apiVersion));
    printf("shape  N=%d K=%d   Q2_0 weights %.1f MiB   Q1_0 %.1f MiB\n\n",
           N, K, (double)N*K/4/1048576.0, (double)N*K/8/1048576.0);

    // ---------------- data (same construction as the CUDA ladder) -------
    srand(1234);
    const size_t q2_bytes = (size_t)N * K / 4, q1_bytes = (size_t)N * K / 8;
    std::vector<uint8_t> hq2(q2_bytes), hq1(q1_bytes), hq2r(q2_bytes), hq1r(q1_bytes);
    for (size_t i = 0; i < q2_bytes; ++i) hq2[i] = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < q1_bytes; ++i) hq1[i] = (uint8_t)(rand() & 0xFF);
    for (int r = 0; r < N; ++r)
        for (int b = 0; b < nblk; ++b) {
            bonsai_q2_repack_block(&hq2[(size_t)r*(K/4) + b*BONSAI_Q2_BYTES],
                                   &hq2r[(size_t)r*(K/4) + b*BONSAI_Q2_BYTES]);
            bonsai_q1_repack_block(&hq1[(size_t)r*(K/8) + b*BONSAI_Q1_BYTES],
                                   &hq1r[(size_t)r*(K/8) + b*BONSAI_Q1_BYTES]);
        }
    // Shader takes fp32 scales.
    std::vector<float> hs((size_t)N * nblk);
    for (auto &v : hs) v = 0.005f + 0.01f * (float)rand() / RAND_MAX;
    std::vector<float> hx(K);
    for (auto &v : hx) v = 2.f * (float)rand() / RAND_MAX - 1.f;

    std::vector<int8_t> ha(K); float da;
    { float amax = 0; for (float v : hx) amax = fmaxf(amax, fabsf(v));
      da = amax / 127.f; const float id = 1.f / da;
      for (int i = 0; i < K; ++i) { int q = (int)lrintf(hx[i]*id);
          ha[i] = (int8_t)(q>127?127:(q<-127?-127:q)); } }
    std::vector<int32_t> ha4(K/4), asum16(K/16), asum32(K/32);
    for (int i = 0; i < K/4; ++i) { int32_t p = 0;
        for (int k = 0; k < 4; ++k) p |= ((int32_t)(uint8_t)ha[i*4+k]) << (8*k);
        ha4[i] = p; }
    for (int i = 0; i < K/16; ++i) { int s = 0; for (int k=0;k<16;++k) s += ha[i*16+k]; asum16[i]=s; }
    for (int i = 0; i < K/32; ++i) { int s = 0; for (int k=0;k<32;++k) s += ha[i*32+k]; asum32[i]=s; }

    const int NREF = 64;
    std::vector<float> ref2(NREF), ref1(NREF);
    for (int r = 0; r < NREF; ++r) {
        double a2 = 0, a1 = 0;
        for (int b = 0; b < nblk; ++b) {
            int8_t c2[BONSAI_QK], c1[BONSAI_QK];
            bonsai_q2_ref_decode(&hq2[(size_t)r*(K/4)+b*BONSAI_Q2_BYTES], c2);
            bonsai_q1_ref_decode(&hq1[(size_t)r*(K/8)+b*BONSAI_Q1_BYTES], c1);
            const float s = hs[(size_t)r*nblk+b];
            for (int j = 0; j < BONSAI_QK; ++j) {
                a2 += (double)c2[j]*s*(double)ha[b*BONSAI_QK+j]*da;
                a1 += (double)c1[j]*s*(double)ha[b*BONSAI_QK+j]*da;
            }
        }
        ref2[r] = (float)a2; ref1[r] = (float)a1;
    }

    // ---------------- buffers ----------------
    Buf bW2 = make_buf(c, q2_bytes), bW1 = make_buf(c, q1_bytes);
    // GGUF-order (unrepacked) weights, for the extraction path that
    // ggml-vulkan would have to use -- it receives weights in GGUF order
    // and has no opportunity to repack them.
    Buf bW2g = make_buf(c, q2_bytes);
    Buf bS  = make_buf(c, hs.size()*4);
    Buf bA  = make_buf(c, ha4.size()*4);
    Buf bS16= make_buf(c, asum16.size()*4), bS32 = make_buf(c, asum32.size()*4);
    Buf bO  = make_buf(c, (size_t)N*4);
    memcpy(bW2.p, hq2r.data(), q2_bytes);
    memcpy(bW2g.p, hq2.data(), q2_bytes);
    memcpy(bW1.p, hq1r.data(), q1_bytes);
    memcpy(bS.p,  hs.data(),   hs.size()*4);
    memcpy(bA.p,  ha4.data(),  ha4.size()*4);
    memcpy(bS16.p,asum16.data(),asum16.size()*4);
    memcpy(bS32.p,asum32.data(),asum32.size()*4);

    // ---------------- descriptors / pipeline ----------------
    VkDescriptorSetLayoutBinding lb[5]{};
    for (int i = 0; i < 5; ++i) {
        lb[i].binding = i; lb[i].descriptorCount = 1;
        lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 5; dl.pBindings = lb;
    VkDescriptorSetLayout dsl; VK(vkCreateDescriptorSetLayout(c.dev, &dl, nullptr, &dsl));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5*8};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 8; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VkDescriptorPool dpool; VK(vkCreateDescriptorPool(c.dev, &dpi, nullptr, &dpool));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1; pl.pSetLayouts = &dsl;
    pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pcr;
    VkPipelineLayout plo; VK(vkCreatePipelineLayout(c.dev, &pl, nullptr, &plo));

    auto make_set = [&](Buf &w, Buf &sums) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = dpool; ai.descriptorSetCount = 1; ai.pSetLayouts = &dsl;
        VkDescriptorSet set; VK(vkAllocateDescriptorSets(c.dev, &ai, &set));
        VkBuffer bufs[5] = {w.b, bS.b, bA.b, sums.b, bO.b};
        VkDescriptorBufferInfo bi[5]; VkWriteDescriptorSet wr[5]{};
        for (int i = 0; i < 5; ++i) {
            bi[i] = {bufs[i], 0, VK_WHOLE_SIZE};
            wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[i].dstSet = set; wr[i].dstBinding = i; wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(c.dev, 5, wr, 0, nullptr);
        return set;
    };

    auto make_pipe = [&](const char *spv) {
        auto code = read_spv(spv);
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        si.codeSize = code.size()*4; si.pCode = code.data();
        VkShaderModule sm; VK(vkCreateShaderModule(c.dev, &si, nullptr, &sm));
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = sm; ci.stage.pName = "main";
        ci.layout = plo;
        VkPipeline p; VK(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p));
        return p;
    };

    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = c.pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
    VkCommandBuffer cb; VK(vkAllocateCommandBuffers(c.dev, &cbi, &cb));

    struct Push { int N, K; float da; } push{N, K, da};

    // TEAMS*ROWS rows per workgroup, matching the shader's layout.
    const int ROWS_PER_WG = 8 * 4;
    const uint32_t groups = (uint32_t)((N + ROWS_PER_WG - 1) / ROWS_PER_WG);

    auto run = [&](VkPipeline p, VkDescriptorSet set, int n) {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK(vkBeginCommandBuffer(cb, &bi));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plo, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cb, plo, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &push);
        for (int i = 0; i < n; ++i) {
            vkCmdDispatch(cb, groups, 1, 1);
            if (i + 1 < n) {
                VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            }
        }
        VK(vkEndCommandBuffer(cb));
        VkSubmitInfo su{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        su.commandBufferCount = 1; su.pCommandBuffers = &cb;
        VK(vkQueueSubmit(c.q, 1, &su, VK_NULL_HANDLE));
        VK(vkQueueWaitIdle(c.q));
    };

    auto check = [&](const char *name, const std::vector<float> &ref) {
        const float *o = (const float *)bO.p;
        double worst = 0;
        for (int r = 0; r < NREF; ++r) {
            double den = fabs(ref[r]) > 1e-6 ? fabs(ref[r]) : 1e-6;
            worst = fmax(worst, fabs(o[r] - ref[r]) / den);
        }
        bool ok = worst <= 2e-2;
        printf("  %-26s max rel err %.3e  %s\n", name, worst, ok ? "OK" : "*** MISMATCH ***");
        return ok;
    };

    // which weight buffer each case binds: 0 = repacked, 1 = GGUF order, 2 = q1
    struct Case { const char *name; const char *spv; int buf; bool q2; };
    std::vector<Case> cases = {
        {"q2 portable",        "gemv_q2.spv",      0, true},
        {"q2 int-dot",         "gemv_q2_dot.spv",  0, true},
        {"q2 int-dot GGUF-ord","gemv_q2_gguf.spv", 1, true},
        {"q2 GGUF-ord WIDE4",   "gemv_q2_wide4.spv", 1, true},
        {"q1 portable",        "gemv_q1.spv",      2, false},
        {"q1 int-dot",         "gemv_q1_dot.spv",  2, false},
    };

    VkDescriptorSet set2  = make_set(bW2,  bS16);
    VkDescriptorSet set2g = make_set(bW2g, bS16);
    VkDescriptorSet set1  = make_set(bW1,  bS32);
    auto setfor = [&](int b) { return b == 0 ? set2 : (b == 1 ? set2g : set1); };

    printf("=== validation (first %d rows vs CPU reference) ===\n", NREF);
    std::vector<Case> live;
    std::vector<VkPipeline> pipes;
    for (auto &cs : cases) {
        FILE *f = fopen(cs.spv, "rb");
        if (!f) { printf("  %-26s (not built, skipped)\n", cs.name); continue; }
        fclose(f);
        VkPipeline p = make_pipe(cs.spv);
        memset(bO.p, 0, (size_t)N*4);
        run(p, setfor(cs.buf), 1);
        if (check(cs.name, cs.q2 ? ref2 : ref1)) { live.push_back(cs); pipes.push_back(p); }
    }
    if (live.empty()) { printf("\nnothing validated -- no timings\n"); return 1; }

    printf("\n=== throughput (%d iters) ===\n", iters);
    for (size_t i = 0; i < live.size(); ++i) {
        run(pipes[i], setfor(live[i].buf), 2);              // warm
        auto t0 = std::chrono::high_resolution_clock::now();
        run(pipes[i], setfor(live[i].buf), iters);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1-t0).count() / iters;
        double bytes = live[i].q2 ? (double)q2_bytes : (double)q1_bytes;
        printf("  %-26s %7.3f ms   %7.1f GB/s\n", live[i].name, ms, bytes/1e9/(ms/1e3));
    }
    printf("\nGB/s counts weight bytes only, matching the CUDA ladder.\n");
    return 0;
}
