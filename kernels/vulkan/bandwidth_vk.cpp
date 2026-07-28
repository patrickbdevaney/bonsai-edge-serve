// Pure streaming-read bandwidth probe for Vulkan on Thor -- the control the
// GEMV numbers were missing.
//
// Every Vulkan GEMV here lands at 100-115 GB/s, and ggml's own MMVQ at
// 93-122, while the CUDA GEMV reaches 241 GB/s on the identical shape and
// the measured achievable read is 244.7 GB/s. A GEMV cannot tell us whether
// that is a kernel deficiency or a Vulkan ceiling, because it also does
// arithmetic. This does nothing but read.
//
// It also varies the MEMORY TYPE, which is the other candidate: Thor is
// unified, so HOST_VISIBLE|HOST_COHERENT and DEVICE_LOCAL are the same
// physical DRAM, but they can differ in cacheability, and a write-combined
// mapping is slow to read.
//
// build:
//   glslc -O bandwidth.comp -o bandwidth.spv
//   glslc -O -DVEC4 bandwidth.comp -o bandwidth_vec4.spv
//   g++ -O2 -std=c++17 bandwidth_vk.cpp -o bandwidth_vk -lvulkan

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <vulkan/vulkan.h>

#define VK(x) do { VkResult r_=(x); if(r_!=VK_SUCCESS){ \
    fprintf(stderr,"%s:%d vk error %d\n",__FILE__,__LINE__,(int)r_); exit(1);} } while(0)

struct Ctx {
    VkInstance inst{}; VkPhysicalDevice phys{}; VkDevice dev{};
    VkQueue q{}; uint32_t qfam{}; VkCommandPool pool{};
    VkPhysicalDeviceMemoryProperties memp{};
};
struct Buf { VkBuffer b{}; VkDeviceMemory m{}; size_t bytes{}; };

static int pick_mem(Ctx &c, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c.memp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) &&
            (c.memp.memoryTypes[i].propertyFlags & want) == want) return (int) i;
    return -1;
}

static bool make_buf(Ctx &c, size_t bytes, VkMemoryPropertyFlags want, Buf &out) {
    out.bytes = bytes;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK(vkCreateBuffer(c.dev, &bi, nullptr, &out.b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c.dev, out.b, &mr);
    const int idx = pick_mem(c, mr.memoryTypeBits, want);
    if (idx < 0) { vkDestroyBuffer(c.dev, out.b, nullptr); out.b = VK_NULL_HANDLE; return false; }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size; ai.memoryTypeIndex = (uint32_t) idx;
    VK(vkAllocateMemory(c.dev, &ai, nullptr, &out.m));
    VK(vkBindBufferMemory(c.dev, out.b, out.m, 0));
    return true;
}

static std::vector<uint32_t> read_spv(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> v(n / 4);
    if (fread(v.data(), 1, n, f) != (size_t) n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char **argv) {
    const size_t MiB   = (argc > 1) ? (size_t) atoi(argv[1]) : 256;
    const int    iters = (argc > 2) ? atoi(argv[2]) : 20;
    const size_t bytes = MiB << 20;

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

    printf("device %s   probe %zu MiB, %d iters\n\n", pr.deviceName, MiB, iters);
    printf("memory types:\n");
    for (uint32_t i = 0; i < c.memp.memoryTypeCount; ++i) {
        const auto f = c.memp.memoryTypes[i].propertyFlags;
        printf("  [%u] heap %u %s%s%s%s\n", i, c.memp.memoryTypes[i].heapIndex,
               (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)  ? "DEVICE_LOCAL "  : "",
               (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)  ? "HOST_VISIBLE "  : "",
               (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? "HOST_COHERENT " : "",
               (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)   ? "HOST_CACHED "   : "");
    }
    printf("\n");

    // ---- descriptors / pipeline layout (2 storage buffers + push constants)
    VkDescriptorSetLayoutBinding lb[2]{};
    for (int i = 0; i < 2; ++i) {
        lb[i].binding = i; lb[i].descriptorCount = 1;
        lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 2; dl.pBindings = lb;
    VkDescriptorSetLayout dsl; VK(vkCreateDescriptorSetLayout(c.dev, &dl, nullptr, &dsl));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * 8};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 8; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VkDescriptorPool dpool; VK(vkCreateDescriptorPool(c.dev, &dpi, nullptr, &dpool));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1; pl.pSetLayouts = &dsl;
    pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pcr;
    VkPipelineLayout plo; VK(vkCreatePipelineLayout(c.dev, &pl, nullptr, &plo));

    auto make_pipe = [&](const char *spv) {
        auto code = read_spv(spv);
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        si.codeSize = code.size() * 4; si.pCode = code.data();
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
    cbi.commandPool = c.pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cb; VK(vkAllocateCommandBuffers(c.dev, &cbi, &cb));

    struct MemCase { const char *name; VkMemoryPropertyFlags flags; };
    const MemCase mems[] = {
        {"HOST_VISIBLE|COHERENT", VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT},
        {"DEVICE_LOCAL",          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT},
        {"DEVICE_LOCAL|HOST_VIS", VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT},
    };
    struct ShCase { const char *name; const char *spv; int elem_bytes; };
    const ShCase shaders[] = {
        {"scalar uint  (32-bit)", "bandwidth.spv",      4},
        {"uvec4       (128-bit)", "bandwidth_vec4.spv", 16},
        // elem_bytes 4: n_elem counts UINTS, and the shader's guard
        // (j = 4i, j+3 < n_elem) makes one in four iterations read four
        // words -- so the whole buffer is read exactly once, and the byte
        // accounting below is honest. Setting this to 16 credits the run
        // with 4x the bytes it actually touched and reports ~1 TB/s.
        {"4x consecutive uint   ", "bandwidth_scalar4.spv", 4},
    };

    printf("%-24s %-24s %10s %10s\n", "memory type", "access", "ms", "GB/s");
    // BONSAI_BW_REVERSE=1 reverses case order. Cases 1 and 3 resolve to the
    // SAME memory type, so any systematic difference between them is an
    // ordering artifact, not a property of the memory. Reversing is how you
    // tell those apart.
    std::vector<MemCase> order(std::begin(mems), std::end(mems));
    if (getenv("BONSAI_BW_REVERSE")) {
        std::reverse(order.begin(), order.end());
    }
    for (const auto &mc : order) {
        Buf in{}, out{};
        if (!make_buf(c, bytes, mc.flags, in)) {
            printf("%-24s %-24s %10s %10s\n", mc.name, "(no such memory type)", "-", "-");
            continue;
        }
        make_buf(c, 1 << 20, mc.flags, out);

        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = dpool; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
        VkDescriptorSet set; VK(vkAllocateDescriptorSets(c.dev, &dai, &set));
        VkBuffer bufs[2] = {in.b, out.b};
        VkDescriptorBufferInfo bi[2]; VkWriteDescriptorSet wr[2]{};
        for (int i = 0; i < 2; ++i) {
            bi[i] = {bufs[i], 0, VK_WHOLE_SIZE};
            wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[i].dstSet = set; wr[i].dstBinding = i; wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(c.dev, 2, wr, 0, nullptr);

        for (const auto &sc : shaders) {
            VkPipeline pipe = make_pipe(sc.spv);
            const uint32_t n_elem = (uint32_t)(bytes / sc.elem_bytes);
            // 20 SMs; give the machine plenty of blocks to hide latency.
            const uint32_t groups = 1024;
            struct { uint32_t n_elem, n_iter; } push{n_elem, 1};

            auto run = [&](int n) {
                VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                VK(vkBeginCommandBuffer(cb, &bbi));
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, plo, 0, 1,
                                        &set, 0, nullptr);
                vkCmdPushConstants(cb, plo, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &push);
                for (int i = 0; i < n; ++i) {
                    vkCmdDispatch(cb, groups, 1, 1);
                    if (i + 1 < n) {
                        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                             1, &mb, 0, nullptr, 0, nullptr);
                    }
                }
                VK(vkEndCommandBuffer(cb));
                VkSubmitInfo su{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                su.commandBufferCount = 1; su.pCommandBuffers = &cb;
                VK(vkQueueSubmit(c.q, 1, &su, VK_NULL_HANDLE));
                VK(vkQueueWaitIdle(c.q));
            };

            // Warm the BUFFER, not just the pipeline. A freshly allocated
            // buffer that has never been touched pays first-touch costs on
            // its first real use, and that landed entirely on whichever
            // case ran first -- making the first row of the table read
            // ~25% low and inventing a "memory type matters" effect that
            // does not exist. Both warm passes are outside the timer.
            run(4);
            const auto t0 = std::chrono::steady_clock::now();
            run(iters);
            const double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            const double ms   = dt * 1e3 / iters;
            const double gbs  = (double) bytes * iters / dt / 1e9;
            printf("%-24s %-24s %10.3f %10.1f\n", mc.name, sc.name, ms, gbs);
            vkDestroyPipeline(c.dev, pipe, nullptr);
        }
        vkDestroyBuffer(c.dev, in.b, nullptr);  vkFreeMemory(c.dev, in.m, nullptr);
        vkDestroyBuffer(c.dev, out.b, nullptr); vkFreeMemory(c.dev, out.m, nullptr);
    }

    printf("\nfor comparison on this device:\n");
    printf("  CUDA float4 streaming read   244.7 GB/s   (bench/bandwidth)\n");
    printf("  CUDA q2_0 GEMV v5 @21MiB     241.0 GB/s   (results/gemv-cuda-shapes.txt)\n");
    printf("  ggml Vulkan MMVQ q2_0        93-122 GB/s  (perf logger, real decode)\n");
    printf("  our Vulkan GEMV q2_0         101-115 GB/s (results/gemv-vulkan.txt)\n");
    return 0;
}
