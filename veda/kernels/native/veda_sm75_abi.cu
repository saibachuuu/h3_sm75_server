// C ABI wrapper around the unchanged Apache-2.0 VEDA/Sage CUDA kernel.
#include "sage/veda_sparse.cu"
#include <string>

#ifdef _WIN32
#define VEDA_EXPORT extern "C" __declspec(dllexport)
#else
#define VEDA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct VedaArgs {
    uint64_t q, k, v, out, qs, ks, routes, valid, policy, stream;
    int32_t device, heads, slots;
    int64_t q_stride[3], k_stride[3], v_stride[3], out_stride[3];
};

static thread_local std::string last_error;
VEDA_EXPORT int star7_veda_abi_version() { return 1; }
VEDA_EXPORT int star7_veda_args_size() { return sizeof(VedaArgs); }
VEDA_EXPORT const char* star7_veda_last_error() { return last_error.c_str(); }
VEDA_EXPORT int star7_veda_cuda_runtime_version() { return CUDART_VERSION; }

VEDA_EXPORT int star7_veda_attention(const VedaArgs* a) {
    try {
        last_error.clear();
        VEDA_CHECK(a && a->slots > 0 && a->slots % 128 == 0 && a->heads > 0,
                   "Invalid VEDA shape; slots must be a positive multiple of 128");
        VEDA_CHECK(a->q && a->k && a->v && a->out && a->qs && a->ks &&
                   a->routes && a->valid && a->policy, "Null VEDA tensor pointer");
        cudaError_t error = cudaSetDevice(a->device);
        VEDA_CHECK(error == cudaSuccess, "CUDA device: ", cudaGetErrorString(error));
        using S = AttentionStorage<128, false>;
        const int blocks = div_ceil(a->slots, kBlockTokens);
        auto kernel = sparse_attention_kernel<128, half, false, false, false, false,
                                               1, 1, true, false>;
        configure_dynamic_shared_memory(kernel, S::kAttentionSharedBytes, "VEDA SM75");
        kernel<<<dim3(blocks, a->heads, 1), dim3(WARP_SIZE, kWarps),
                 S::kAttentionSharedBytes, reinterpret_cast<cudaStream_t>(a->stream)>>>(
            reinterpret_cast<const int8_t*>(a->q), reinterpret_cast<const int8_t*>(a->k),
            reinterpret_cast<const half*>(a->v), nullptr, nullptr,
            reinterpret_cast<half*>(a->out), reinterpret_cast<const float*>(a->qs),
            reinterpret_cast<const float*>(a->ks), nullptr, nullptr, nullptr, nullptr,
            reinterpret_cast<const uint8_t*>(a->policy), nullptr,
            reinterpret_cast<const uint32_t*>(a->routes), nullptr, nullptr, nullptr, nullptr,
            reinterpret_cast<const int32_t*>(a->valid), a->slots, a->slots,
            a->heads, a->heads, blocks, blocks, 0,
            a->q_stride[0], a->q_stride[1], a->q_stride[2],
            a->k_stride[0], a->k_stride[1], a->k_stride[2],
            a->v_stride[0], a->v_stride[1], a->v_stride[2], 0, 0,
            a->out_stride[0], a->out_stride[1], a->out_stride[2],
            0.0f, 0.08838834764831845f, 0);
        check_launch("VEDA SM75");
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}
