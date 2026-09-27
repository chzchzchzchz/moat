/* Project Antigravity — MetalTransformerEngine: Full C++ Metal Decode Loop
 *
 * Implements TinyLlama-1.1B inference entirely on Metal GPU:
 *   - Safetensors weight loading → Metal shared buffers  
 *   - 22-layer transformer forward pass (RMSNorm → GQA → SwiGLU MLP)
 *   - Autoregressive decode with KV caching
 *   - N-channel parallel Best-of-N generation
 *   - Real TTFT/TPOT measurement
 *
 * Target: Apple Silicon (M1-M4, A17 Pro, A18 Pro)
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cmath>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <random>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <map>
#include "transformer_engine.h"
#include "superblock_pack.h"
#include "shader_sources.h"
#include "engine_limits.h"
#include "sampling.h"
#include "safetensors_header.h"

// BFloat16 → Float16 conversion helper
static inline uint16_t bf16_to_fp16(uint16_t bf16) {
    // BFloat16: 1 sign + 8 exp + 7 mantissa
    // Float16:  1 sign + 5 exp + 10 mantissa
    uint32_t sign = (bf16 >> 15) & 1;
    int32_t  exp  = ((bf16 >> 7) & 0xFF) - 127;  // unbias BF16 exponent
    uint32_t mant = bf16 & 0x7F;                  // 7-bit mantissa
    
    // Handle special cases
    if (exp == 128) {
        // Inf or NaN → FP16 Inf/NaN
        return (uint16_t)((sign << 15) | (0x1F << 10) | (mant >> 4));
    }
    if (exp < -24) {
        // Underflow to zero
        return (uint16_t)(sign << 15);
    }
    
    // Rebias for FP16 (bias=15)
    int32_t fp16_exp = exp + 15;
    // Extend mantissa from 7-bit to 10-bit
    uint32_t fp16_mant = mant << 3;
    
    if (fp16_exp <= 0) {
        // Subnormal in FP16
        fp16_mant = (0x400 | fp16_mant) >> (1 - fp16_exp);
        fp16_exp = 0;
    } else if (fp16_exp >= 0x1F) {
        // Overflow to Inf
        fp16_exp = 0x1F;
        fp16_mant = 0;
    }
    
    return (uint16_t)((sign << 15) | (fp16_exp << 10) | (fp16_mant & 0x3FF));
}


// ============================================================================
// Shader library loading
// ============================================================================

namespace {

// Shader files are looked up relative to a handful of plausible roots so the
// engine works from a repo checkout, from an app bundle, and from whatever
// directory a harness happens to run in. ANTIGRAVITY_SHADER_DIR wins when set.
NSArray<NSString*>* shaderSearchRoots() {
    static NSArray<NSString*>* roots = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSMutableArray<NSString*>* r = [NSMutableArray array];
        const char* env = getenv("ANTIGRAVITY_SHADER_DIR");
        if (env && *env) [r addObject:[NSString stringWithUTF8String:env]];
        NSString* bundled = [[NSBundle mainBundle] resourcePath];
        if (bundled) [r addObject:bundled];
        [r addObject:[[NSFileManager defaultManager] currentDirectoryPath]];
        [r addObject:@"."];
        [r addObject:@"antigravity-engine"];
        [r addObject:@".."];
        [r addObject:@"../antigravity-engine"];
        roots = [r copy];
    });
    return roots;
}

NSString* resolveShaderPath(NSString* rel) {
    NSFileManager* fm = [NSFileManager defaultManager];
    if ([rel isAbsolutePath] && [fm fileExistsAtPath:rel]) return rel;
    for (NSString* root in shaderSearchRoots()) {
        NSString* candidate = [root stringByAppendingPathComponent:rel];
        if ([fm fileExistsAtPath:candidate]) return candidate;
    }
    return nil;
}

bool libraryExportsAll(id<MTLLibrary> lib, NSArray<NSString*>* names) {
    if (!lib) return false;
    for (NSString* name in names) {
        if (![lib newFunctionWithName:name]) return false;
    }
    return true;
}

}  // namespace

id<MTLLibrary> MetalTransformerEngine::loadShaderLibrary(
    NSString* metallibRelPath,
    NSString* metalSourceRelPath,
    NSArray<NSString*>* requiredFunctions
) {
    NSString* libPath = resolveShaderPath(metallibRelPath);
    NSString* srcPath = resolveShaderPath(metalSourceRelPath);
    NSError* err = nil;

    id<MTLLibrary> prebuilt = nil;
    if (libPath) {
        prebuilt = [device_ newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&err];
        if (prebuilt && libraryExportsAll(prebuilt, requiredFunctions)) {
            return prebuilt;
        }
        if (prebuilt) {
            // The .metallib is checked in; a kernel added to the .metal source since
            // it was built is simply absent. Taking it anyway would drop that kernel
            // with no error at all, so prefer recompiling the source.
            std::cerr << "[MetalTransformerEngine] " << libPath.UTF8String
                      << " does not export every required kernel (stale build); "
                      << "compiling from source instead" << std::endl;
        }
    }

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];

    if (srcPath) {
        NSString* source = [NSString stringWithContentsOfFile:srcPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&err];
        if (source) {
            id<MTLLibrary> compiled = [device_ newLibraryWithSource:source options:opts error:&err];
            if (compiled) return compiled;
            std::cerr << "[MetalTransformerEngine] failed to compile " << srcPath.UTF8String
                      << ": " << (err ? err.localizedDescription.UTF8String : "unknown error")
                      << std::endl;
        }
    }

    // The floor: the source is compiled into the binary. Neither the .metal files
    // nor the .metallib files are packaged inside AntigravityEngine.xcframework, and
    // every path above is resolved relative to the process working directory — so on
    // a device nothing above this point can succeed, and without it every pipeline
    // outside the one inline fallback kernel would simply be null.
    const std::string stem = [[metalSourceRelPath.lastPathComponent
                               stringByDeletingPathExtension] UTF8String];
    if (const char* embedded = antigravity::shaders::find(stem.c_str())) {
        id<MTLLibrary> compiled =
            [device_ newLibraryWithSource:[NSString stringWithUTF8String:embedded]
                                  options:opts
                                    error:&err];
        if (compiled) return compiled;
        std::cerr << "[MetalTransformerEngine] failed to compile embedded " << stem
                  << ".metal: " << (err ? err.localizedDescription.UTF8String : "unknown error")
                  << std::endl;
    }

    // A stale library is still better than none for the kernels it does export.
    return prebuilt;
}

// ============================================================================
// Constructor
// ============================================================================

MetalTransformerEngine::MetalTransformerEngine(const TransformerConfig& config)
    : config_(config), allocatedBytes_(0) {
    weightsLoaded_ = false;
    
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        std::cerr << "[MetalTransformerEngine] Metal not supported" << std::endl;
        return;
    }
    queue_ = [device_ newCommandQueue];
    
    // ---- Load Shader Libraries ----
    // The engine ships prebuilt .metallib files, but they are checked in and go
    // stale the moment a .metal source gains a kernel. loadShaderLibrary() accepts
    // a .metallib only when it still exports everything we need, and otherwise
    // compiles the source, so adding a kernel never silently does nothing.
    gemmLib_ = loadShaderLibrary(@"src/shaders/batched_gemm.metallib",
                                 @"src/shaders/batched_gemm.metal",
                                 @[@"batched_gemm_simdgroup",
                                   @"gemv_int4_kernel",
                                   @"fused_batched_gemm_int4"]);

    opsLib_ = loadShaderLibrary(@"src/shaders/transformer_ops.metallib",
                                @"src/shaders/transformer_ops.metal",
                                @[@"gemv_kernel", @"rmsnorm_kernel", @"rope_kernel"]);

    if (!gemmLib_) {
        std::cerr << "[MetalTransformerEngine] no GEMM shader library; "
                     "set ANTIGRAVITY_SHADER_DIR to the directory holding src/shaders"
                  << std::endl;
    }
    if (!opsLib_) {
        std::cerr << "[MetalTransformerEngine] no transformer_ops shader library; "
                     "set ANTIGRAVITY_SHADER_DIR to the directory holding src/shaders"
                  << std::endl;
    }

    // ---- Create Compute Pipelines ----
    auto makePipeline = [&](id<MTLLibrary> lib, NSString* name) -> id<MTLComputePipelineState> {
        if (!lib) return nil;
        id<MTLFunction> func = [lib newFunctionWithName:name];
        if (!func) return nil;
        NSError* pipeErr = nil;
        return [device_ newComputePipelineStateWithFunction:func error:&pipeErr];
    };
    
    gemmPipeline_      = makePipeline(gemmLib_, @"batched_gemm_simdgroup");
    gemvPipeline_      = makePipeline(opsLib_, @"gemv_kernel");
    gemvInt4Pipeline_      = makePipeline(gemmLib_, @"gemv_int4_kernel");
    fusedGemmInt4Pipeline_ = makePipeline(gemmLib_, @"fused_batched_gemm_int4");
    rmsnormPipeline_   = makePipeline(opsLib_, @"rmsnorm_kernel");
    ropePipeline_      = makePipeline(opsLib_, @"rope_kernel");
    
    // Load the hybrid-architecture shaders.
    //
    // Both pipelines are created and NEITHER is dispatched: grep for deltanetPipeline_
    // and moeRouterPipeline_ and they appear only here and in the header. forwardLayer()
    // runs a dense Llama block for every layer with no branching on layer type, so
    // mixture-of-experts and linear-attention layers are not implemented.
    //
    // Kept rather than deleted, because they are the start of real work and the kernels
    // compile and have been debugged. loadWeights() now warns when a checkpoint carries
    // weights for either, since running those dense produces wrong output rather than
    // an error.
    
    id<MTLLibrary> deltaLib = loadShaderLibrary(@"src/shaders/deltanet.metallib",
                                                @"src/shaders/deltanet_forward.metal",
                                                @[@"deltanet_forward"]);
    if (deltaLib) {
        deltanetPipeline_ = makePipeline(deltaLib, @"deltanet_forward");
    }
    id<MTLLibrary> moeLib = loadShaderLibrary(@"src/shaders/moe.metallib",
                                              @"src/shaders/moe_gemm.metal",
                                              @[@"moe_router"]);
    if (moeLib) {
        moeRouterPipeline_ = makePipeline(moeLib, @"moe_router");
    }
    attnScoresPipeline_ = makePipeline(opsLib_, @"gqa_attention_scores_kernel");
    softmaxPipeline_   = makePipeline(opsLib_, @"softmax_kernel");
    attnValuePipeline_ = makePipeline(opsLib_, @"attention_value_kernel");
    siluMulPipeline_   = makePipeline(opsLib_, @"silu_elementwise_mul_kernel");
    residualPipeline_  = makePipeline(opsLib_, @"residual_add_kernel");
    embedPipeline_     = makePipeline(opsLib_, @"embedding_lookup_kernel");
    kvAppendPipeline_  = makePipeline(opsLib_, @"kv_cache_append_kernel");
    
    // Opt-in for now: INT4 changes numerics, so it is switched on explicitly by
    // the benchmark and quality harnesses rather than silently by default.
    if (const char* q = getenv("ANTIGRAVITY_INT4")) {
        quantizeOnLoad_ = (q[0] == '1' || q[0] == 't' || q[0] == 'T' || q[0] == 'y' || q[0] == 'Y');
    }
    // Both are needed: decode (M == 1) takes the GEMV, prefill and the multi-channel
    // step take the fused GEMM. Quantizing with only one available would leave the
    // other path refusing to compute rather than producing a wrong answer, but that
    // is still a dead engine, so fall back to FP16 up front instead.
    if (quantizeOnLoad_ && (!gemvInt4Pipeline_ || !fusedGemmInt4Pipeline_)) {
        std::cerr << "[MetalTransformerEngine] INT4 weights requested but "
                  << (gemvInt4Pipeline_ ? "fused_batched_gemm_int4" : "gemv_int4_kernel")
                  << " is unavailable; falling back to FP16 weights" << std::endl;
        quantizeOnLoad_ = false;
    }

    reinitBuffersAndRoPE();
    
    std::cout << "[MetalTransformerEngine] Initialized with " 
              << (allocatedBytes_ / (1024*1024)) << " MB allocated ("
              << config_.n_channels << " channels, " << config_.n_layers << " layers)" 
              << std::endl;
}

void MetalTransformerEngine::reinitBuffersAndRoPE() {
    // ---- Allocate KV Caches ----
    size_t kv_size = config_.n_kv_heads * config_.max_seq_len * config_.head_dim * sizeof(uint16_t);
    kvCaches_.clear();
    kvCaches_.resize(config_.n_layers);
    for (int l = 0; l < config_.n_layers; l++) {
        kvCaches_[l].resize(config_.n_channels);
        for (int c = 0; c < config_.n_channels; c++) {
            kvCaches_[l][c].k_cache = [device_ newBufferWithLength:kv_size options:MTLResourceStorageModeShared];
            kvCaches_[l][c].v_cache = [device_ newBufferWithLength:kv_size options:MTLResourceStorageModeShared];
            allocatedBytes_ += 2 * kv_size;
        }
    }
    
    // ---- Precompute RoPE Frequencies ----
    int half_dim = config_.head_dim / 2;
    size_t rope_size = config_.max_seq_len * half_dim * sizeof(_Float16);
    ropeFreqsCos_ = [device_ newBufferWithLength:rope_size options:MTLResourceStorageModeShared];
    ropeFreqsSin_ = [device_ newBufferWithLength:rope_size options:MTLResourceStorageModeShared];
    allocatedBytes_ += 2 * rope_size;
    
    _Float16* cos_ptr = (_Float16*)[ropeFreqsCos_ contents];
    _Float16* sin_ptr = (_Float16*)[ropeFreqsSin_ contents];
    for (int pos = 0; pos < config_.max_seq_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(config_.rope_theta, (float)(2 * i) / config_.head_dim);
            float angle = pos * freq;
            cos_ptr[pos * half_dim + i] = (_Float16)cosf(angle);
            sin_ptr[pos * half_dim + i] = (_Float16)sinf(angle);
        }
    }
    
    // ---- Allocate Scratch Buffers ----
    size_t hidden_bytes = config_.hidden_dim * sizeof(uint16_t);
    size_t inter_bytes  = config_.intermediate_dim * sizeof(uint16_t);
    size_t logits_bytes = config_.vocab_size * sizeof(uint16_t);
    size_t attn_scores_bytes = config_.n_heads * config_.max_seq_len * sizeof(uint16_t);

    size_t max_rows = std::max((size_t)8, (size_t)((config_.n_channels + 7) & ~7));
    size_t max_q_rows = max_rows * config_.q_len_max;

    scratch1_ = [device_ newBufferWithLength:max_q_rows * std::max(hidden_bytes, inter_bytes) options:MTLResourceStorageModeShared];
    scratch2_ = [device_ newBufferWithLength:max_q_rows * std::max(hidden_bytes, inter_bytes) options:MTLResourceStorageModeShared];
    scratch3_ = [device_ newBufferWithLength:max_q_rows * std::max(hidden_bytes, inter_bytes) options:MTLResourceStorageModeShared];
    scratchV_ = [device_ newBufferWithLength:max_q_rows * std::max(hidden_bytes, inter_bytes) options:MTLResourceStorageModeShared];
    scratchAttn_ = [device_ newBufferWithLength:std::max((size_t)(max_q_rows * inter_bytes), max_q_rows * attn_scores_bytes) options:MTLResourceStorageModeShared];
    scratchLogits_ = [device_ newBufferWithLength:max_q_rows * logits_bytes options:MTLResourceStorageModeShared];
    
    allocatedBytes_ += max_q_rows * (std::max(hidden_bytes, inter_bytes) * 4 + std::max(inter_bytes, attn_scores_bytes) + logits_bytes);
}

MetalTransformerEngine::~MetalTransformerEngine() {
    // ARC handles all ObjC object cleanup
}


// ============================================================================
// Safetensors Parser & Weight Loader
// ============================================================================

void MetalTransformerEngine::allocateUnifiedMemoryMap() {
    reinitBuffersAndRoPE();
    
    size_t H = config_.hidden_dim;
    size_t I = config_.intermediate_dim;
    size_t KV_DIM = config_.n_kv_heads * config_.head_dim;
    
    layerWeights_.resize(config_.n_layers);
    for (int l = 0; l < config_.n_layers; l++) {
        layerWeights_[l].input_norm = [device_ newBufferWithLength:H * 2 options:MTLResourceStorageModeShared];
        layerWeights_[l].q_proj = [device_ newBufferWithLength:H * H * 2 options:MTLResourceStorageModeShared];
        layerWeights_[l].k_proj = [device_ newBufferWithLength:H * KV_DIM * 2 options:MTLResourceStorageModeShared];
        layerWeights_[l].v_proj = [device_ newBufferWithLength:H * KV_DIM * 2 options:MTLResourceStorageModeShared];
        layerWeights_[l].o_proj = [device_ newBufferWithLength:H * H * 2 options:MTLResourceStorageModeShared];
        layerWeights_[l].post_attn_norm = [device_ newBufferWithLength:H * 2 options:MTLResourceStorageModeShared];
        layerWeights_[l].gate_proj = [device_ newBufferWithLength:H * I * 2 options:MTLResourceStorageModeShared];
        layerWeights_[l].up_proj = [device_ newBufferWithLength:H * I * 2 options:MTLResourceStorageModeShared];
        layerWeights_[l].down_proj = [device_ newBufferWithLength:I * H * 2 options:MTLResourceStorageModeShared];
    }
    
    embedWeights_ = [device_ newBufferWithLength:config_.vocab_size * H * 2 options:MTLResourceStorageModeShared];
    finalNorm_ = [device_ newBufferWithLength:H * 2 options:MTLResourceStorageModeShared];
    lmHead_ = [device_ newBufferWithLength:config_.vocab_size * H * 2 options:MTLResourceStorageModeShared];
    
    weightsLoaded_ = true;
}

bool MetalTransformerEngine::parseSafetensors(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[loadWeights] Cannot open: " << path << std::endl;
        return false;
    }

    // How big the file actually is. Nothing used to ask, which meant a tensor whose
    // offsets ran past the end of a truncated checkpoint was mmapped and read anyway.
    file.seekg(0, std::ios::end);
    const std::streamoff file_end = file.tellg();
    if (file_end < 8) {
        std::cerr << "[loadWeights] File is " << file_end
                  << " bytes, too short to be a safetensors file: " << path << std::endl;
        return false;
    }
    const uint64_t validated_file_size = (uint64_t)file_end;
    file.seekg(0, std::ios::beg);

    // Read header length (8-byte LE uint64). Check the read: a short read used to leave
    // header_len partly unwritten.
    uint64_t header_len = 0;
    if (!file.read(reinterpret_cast<char*>(&header_len), 8)) {
        std::cerr << "[loadWeights] Could not read the 8-byte header length: "
                  << path << std::endl;
        return false;
    }
    if (header_len > 100 * 1024 * 1024) {  // sanity: max 100MB header
        std::cerr << "[loadWeights] Header too large: " << header_len << std::endl;
        return false;
    }
    if (header_len == 0 || header_len > validated_file_size - 8) {
        std::cerr << "[loadWeights] Header claims " << header_len
                  << " bytes but the file holds only " << (validated_file_size - 8)
                  << " after the length field; the file is truncated or not safetensors: "
                  << path << std::endl;
        return false;
    }

    // Read JSON header, checking the read for the same reason.
    std::string header_json(header_len, '\0');
    if (!file.read(&header_json[0], (std::streamsize)header_len)) {
        std::cerr << "[loadWeights] Could not read the " << header_len
                  << "-byte JSON header: " << path << std::endl;
        return false;
    }

    const uint64_t data_start = 8 + header_len;

    // Parse and validate the header. This lives in src/safetensors_header.h so it can be
    // tested without a Metal device — see tests/test_safetensors_header.cpp. The parser
    // that used to be inline here had four defects, none of which announced itself:
    //
    //   - It skipped "__metadata__" by finding the first '}' after the key. With a nested
    //     object anywhere but last, or a '}' inside a metadata string, parsing resumed in
    //     the middle of metadata: a reproduction of that code on a header whose metadata
    //     holds {"extra":{...},"format":"pt"} returns a single tensor named "format",
    //     carrying the real tensor's data_offsets, with the real tensor absent. An absent
    //     tensor means loadTensor never runs for it and its buffer keeps whatever it held.
    //   - TensorInfo::offset_start and offset_end had no initialiser and the struct was
    //     default-constructed, so a tensor whose data_offsets failed to parse carried
    //     indeterminate offsets.
    //   - The tensor was inserted unconditionally, so that malformed entry was kept.
    //   - std::stoll and std::stoull throw, and this is reached through the extern "C"
    //     AntigravityEngineLoadModel, so a malformed header threw across a C boundary.
    //
    // And nothing compared any offset to the file size.
    const antigravity::HeaderParseResult parsed =
        antigravity::parseSafetensorsHeader(header_json, data_start, validated_file_size);
    if (!parsed.ok) {
        std::cerr << "[loadWeights] Refusing to load " << path << ": " << parsed.error
                  << std::endl;
        return false;
    }
    using TensorInfo = antigravity::TensorEntry;
    const std::map<std::string, TensorInfo>& tensors = parsed.tensors;

    std::cout << "[loadWeights] Parsed " << tensors.size() << " tensors from Safetensors" << std::endl;
    
    // Auto-detect architecture parameters from parsed tensor metadata
    int max_layer = 0;
    for (const auto& kv : tensors) {
        if (kv.first.find("layers.") != std::string::npos) {
            size_t p1 = kv.first.find("layers.");
            size_t p2 = kv.first.find('.', p1 + 7);
            if (p2 != std::string::npos) {
                // parseInt64 rather than std::stoi: a name like "model.layers.foo.x"
                // makes std::stoi throw, and this is reached through the extern "C"
                // AntigravityEngineLoadModel.
                int64_t l_idx = 0;
                if (antigravity::parseInt64(kv.first.substr(p1 + 7, p2 - p1 - 7), l_idx)
                    && l_idx >= 0 && l_idx < 100000 && l_idx + 1 > max_layer) {
                    max_layer = (int)(l_idx + 1);
                }
            }
        }
    }
    if (max_layer > 0) config_.n_layers = max_layer;

    auto it_emb = tensors.find("model.language_model.embed_tokens.weight");
    if (it_emb == tensors.end()) it_emb = tensors.find("embed_tokens.weight");
    if (it_emb != tensors.end() && it_emb->second.shape.size() == 2) {
        config_.vocab_size = (int32_t)it_emb->second.shape[0];
        config_.hidden_dim = (int32_t)it_emb->second.shape[1];
    }

    auto it_gate = tensors.find("model.layers.0.mlp.gate_proj.weight");
    if (it_gate == tensors.end()) it_gate = tensors.find("layers.0.mlp.gate_proj.weight");
    if (it_gate != tensors.end() && it_gate->second.shape.size() == 2) {
        config_.intermediate_dim = (int32_t)it_gate->second.shape[0];
    }

    auto it_k = tensors.find("model.layers.0.self_attn.k_proj.weight");
    if (it_k == tensors.end()) it_k = tensors.find("layers.0.self_attn.k_proj.weight");
    if (it_k != tensors.end() && it_k->second.shape.size() == 2) {
        int32_t k_out = (int32_t)it_k->second.shape[0];
        if (config_.vocab_size > 32000) {
            config_.n_heads = 12;
            config_.head_dim = 128;
            config_.n_kv_heads = k_out / config_.head_dim;
            config_.norm_eps = 1e-6f;
        }
    }

    // forwardLayer() is a dense Llama-style block for every layer: RMSNorm, grouped
    // query attention, SwiGLU, residual. There is no branching on layer type. The
    // engine creates deltanetPipeline_ and moeRouterPipeline_ in its constructor, which
    // makes it look as though mixture-of-experts and linear-attention layers are
    // supported, but neither pipeline is dispatched anywhere — verified by grep: they
    // are assigned once and referenced nowhere else.
    //
    // So a checkpoint whose layers are not all dense would be run as if they were, and
    // would produce plausible-looking garbage rather than failing. Say so at load time
    // rather than at benchmark time.
    {
        bool has_experts = false, has_linear_attn = false, has_router = false;
        for (const auto& kv : tensors) {
            const std::string& name = kv.first;
            if (name.find("mlp.experts.") != std::string::npos
                || name.find("shared_expert") != std::string::npos) has_experts = true;
            if (name.find("linear_attn") != std::string::npos
                || name.find("conv1d") != std::string::npos
                || name.find("A_log") != std::string::npos
                || name.find("dt_bias") != std::string::npos) has_linear_attn = true;
            if (name.find("mlp.gate.weight") != std::string::npos) has_router = true;
        }
        if (has_experts || has_linear_attn || has_router) {
            std::cerr << "[loadWeights] WARNING: this checkpoint contains tensors for an "
                         "architecture this engine does not implement —";
            if (has_experts || has_router) std::cerr << " mixture-of-experts";
            if (has_linear_attn) std::cerr << " linear-attention/state-space layers";
            std::cerr << ". forwardLayer() runs a dense block for every layer, so those "
                         "weights will be ignored or misinterpreted and the output will "
                         "be wrong without failing. deltanet_forward and moe_router are "
                         "compiled but never dispatched." << std::endl;
        }
    }

    std::cout << "[loadWeights] Auto-configured architecture: "
              << config_.n_layers << " layers, hidden_dim=" << config_.hidden_dim
              << ", inter_dim=" << config_.intermediate_dim
              << ", n_heads=" << config_.n_heads
              << ", n_kv_heads=" << config_.n_kv_heads
              << ", head_dim=" << config_.head_dim << std::endl;
    
    // Re-initialize scratch buffers, KV cache arrays, and RoPE frequency tables for auto-detected architecture
    reinitBuffersAndRoPE();
    
    // Memory-map the raw data section
    file.close(); // Close fstream as we will use raw fd for mmap
    
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "[loadWeights] Cannot open for mmap: " << path << std::endl;
        return false;
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "[loadWeights] Cannot stat file: " << path << std::endl;
        close(fd);
        return false;
    }
    size_t file_size = sb.st_size;

    // parseSafetensorsHeader validated every tensor's offsets against the size this file
    // had when the header was read. The mapping below is sized from fstat instead, so if
    // the two disagree the file changed underneath us and those offsets no longer describe
    // what is about to be mapped. Cheap to check, and the alternative is reading past the
    // end of the mapping with offsets that were "already validated".
    if ((uint64_t)file_size != validated_file_size) {
        std::cerr << "[loadWeights] File size changed while loading, from "
                  << validated_file_size << " to " << file_size
                  << " bytes; refusing to use offsets validated against the old size: "
                  << path << std::endl;
        close(fd);
        return false;
    }

    const char* mapped_data = (const char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_data == MAP_FAILED) {
        std::cerr << "[loadWeights] mmap failed for: " << path << std::endl;
        close(fd);
        return false;
    }
    
    const char* raw_data = mapped_data + data_start;
    
    // Helper: load a tensor into a Metal buffer as FP16
    // `quantizable` marks the big [K x N] projection matrices that dispatchGEMM
    // consumes; norms and the embedding table are read by kernels that expect FP16.
    auto loadTensor = [&](const std::string& name,
                          bool transpose_2d = false,
                          bool quantizable = false) -> id<MTLBuffer> {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            // Try with "model." prefix
            it = tensors.find("model." + name);
            if (it == tensors.end()) it = tensors.find("model.language_model." + name);
            if (it == tensors.end()) {
                std::cerr << "[loadWeights] Missing tensor: " << name << std::endl;
                return nil;
            }
        }
        
        const TensorInfo& info = it->second;
        std::cout << "[loadTensor] " << name << " -> key=" << it->first << " dtype=" << info.dtype 
                  << " shape=[" << (info.shape.size() > 0 ? info.shape[0] : 0) 
                  << (info.shape.size() > 1 ? "," + std::to_string(info.shape[1]) : "") << "]"
                  << " offset=" << info.offset_start << std::endl;
        size_t num_elements = 1;
        for (auto d : info.shape) num_elements *= d;
        
        size_t fp16_bytes = num_elements * sizeof(uint16_t);
        id<MTLBuffer> buf = [device_ newBufferWithLength:fp16_bytes options:MTLResourceStorageModeShared];
        if (!buf) return nil;
        
        uint16_t* dest = (uint16_t*)[buf contents];
        const uint16_t* src = (const uint16_t*)(raw_data + info.offset_start);
        
        bool is_bf16 = (info.dtype == "BF16" || info.dtype == "bf16" || info.dtype == "bfloat16");
        
        if (transpose_2d && info.shape.size() == 2) {
            size_t rows = info.shape[0]; // out_features
            size_t cols = info.shape[1]; // in_features
            for (size_t r = 0; r < rows; r++) {
                for (size_t c = 0; c < cols; c++) {
                    uint16_t val = src[r * cols + c];
                    if (is_bf16) val = bf16_to_fp16(val);
                    dest[c * rows + r] = val;
                }
            }
        } else {
            if (is_bf16) {
                // Convert BFloat16 → Float16
                for (size_t i = 0; i < num_elements; i++) {
                    dest[i] = bf16_to_fp16(src[i]);
                }
            } else {
                // Already FP16 or compatible, direct copy
                std::memcpy(dest, src, std::min(fp16_bytes, (size_t)(info.offset_end - info.offset_start)));
            }
        }
        
        allocatedBytes_ += fp16_bytes;

        if (quantizable && quantizeOnLoad_ && info.shape.size() == 2) {
            // The buffer is [K x N] row-major: safetensors stores [out, in], so a
            // transposed load leaves in_features as the row stride, which is the
            // layout both INT4 kernels index as B[k * N + col].
            uint32_t K = (uint32_t)(transpose_2d ? info.shape[1] : info.shape[0]);
            uint32_t N = (uint32_t)(transpose_2d ? info.shape[0] : info.shape[1]);
            id<MTLBuffer> packed = quantizeToSuperblocks(dest, num_elements, K, N);
            if (packed) {
                // The FP16 staging buffer is released here; only the 4-bit copy is kept.
                allocatedBytes_ -= fp16_bytes;
                allocatedBytes_ += [packed length];
                return packed;
            }
            std::cerr << "[loadTensor] " << name << " not quantizable ("
                      << num_elements << " elements is not a multiple of 256); "
                      << "keeping FP16" << std::endl;
        }

        return buf;
    };
    
    // ---- Load Embedding & Output Head ----
    embedWeights_ = loadTensor("embed_tokens.weight", false);
    finalNorm_ = loadTensor("norm.weight", false);
    lmHead_ = loadTensor("lm_head.weight", true, /*quantizable=*/true);
    if (!lmHead_ && embedWeights_) {
        std::cout << "[loadWeights] lm_head.weight tied to embed_tokens.weight (transposing)" << std::endl;
        lmHead_ = loadTensor("embed_tokens.weight", true, /*quantizable=*/true);
    }
    
    if (!embedWeights_ || !finalNorm_ || !lmHead_) {
        std::cerr << "[loadWeights] Failed to load embedding/norm/lm_head" << std::endl;
        return false;
    }
    
    // ---- Load Layer Weights ----
    layerWeights_.resize(config_.n_layers);
    for (int i = 0; i < config_.n_layers; i++) {
        std::string prefix = "layers." + std::to_string(i) + ".";
        
        layerWeights_[i].input_norm  = loadTensor(prefix + "input_layernorm.weight", false);
        layerWeights_[i].q_proj      = loadTensor(prefix + "self_attn.q_proj.weight", true, /*quantizable=*/true);
        layerWeights_[i].k_proj      = loadTensor(prefix + "self_attn.k_proj.weight", true, /*quantizable=*/true);
        layerWeights_[i].v_proj      = loadTensor(prefix + "self_attn.v_proj.weight", true, /*quantizable=*/true);
        layerWeights_[i].o_proj      = loadTensor(prefix + "self_attn.o_proj.weight", true, /*quantizable=*/true);
        layerWeights_[i].post_attn_norm = loadTensor(prefix + "post_attention_layernorm.weight", false);
        layerWeights_[i].gate_proj   = loadTensor(prefix + "mlp.gate_proj.weight", true, /*quantizable=*/true);
        layerWeights_[i].up_proj     = loadTensor(prefix + "mlp.up_proj.weight", true, /*quantizable=*/true);
        layerWeights_[i].down_proj   = loadTensor(prefix + "mlp.down_proj.weight", true, /*quantizable=*/true);
        
        // Validate all loaded
        if (!layerWeights_[i].input_norm || !layerWeights_[i].q_proj || !layerWeights_[i].k_proj ||
            !layerWeights_[i].v_proj || !layerWeights_[i].o_proj || !layerWeights_[i].post_attn_norm ||
            !layerWeights_[i].gate_proj || !layerWeights_[i].up_proj || !layerWeights_[i].down_proj) {
            std::cerr << "[loadWeights] Failed to load layer " << i << std::endl;
            return false;
        }
    }
    
    munmap((void*)mapped_data, file_size);
    close(fd);
    
    std::cout << "[loadWeights] All weights loaded: " << (allocatedBytes_ / 1024 / 1024) << " MB total" << std::endl;
    weightsLoaded_ = true;
    return true;
}

bool MetalTransformerEngine::loadWeights(const std::string& safetensors_path) {
    return parseSafetensors(safetensors_path);
}


// ============================================================================
// Metal Dispatch Helpers
// ============================================================================

id<MTLBuffer> MetalTransformerEngine::quantizeToSuperblocks(
    const uint16_t* fp16, size_t n_elements, uint32_t K, uint32_t N
) {
    const size_t bytes = antigravity::superblockBytesFor(n_elements);
    if (bytes == 0) return nil;   // not a multiple of 256 elements

    id<MTLBuffer> buf = [device_ newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];
    if (!buf) return nil;

    if (!antigravity::packSuperblocks(fp16, n_elements, (uint8_t*)[buf contents])) {
        return nil;
    }

    quantizedWeights_[(__bridge void*)buf] = QuantizedWeight{K, N};
    return buf;
}

void MetalTransformerEngine::dispatchGrid(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pso,
    MTLSize grid
) {
    if (!pso) return;
    if (grid.width == 0 || grid.height == 0 || grid.depth == 0) return;

    // Budget threads per group, clamped to what this pipeline allows. 256 is a
    // common sweet spot on Apple GPUs: several SIMD groups per threadgroup without
    // starving occupancy.
    NSUInteger budget = std::min<NSUInteger>(256, pso.maxTotalThreadsPerThreadgroup);

    // Fill from x outward, because thread_position_in_grid.x varies fastest and so
    // determines whether adjacent lanes touch adjacent memory.
    NSUInteger tx = std::min<NSUInteger>(grid.width, budget);
    NSUInteger ty = std::min<NSUInteger>(grid.height, std::max<NSUInteger>(1, budget / tx));
    NSUInteger tz = std::min<NSUInteger>(grid.depth, std::max<NSUInteger>(1, budget / (tx * ty)));

    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
}

void MetalTransformerEngine::dispatchGEMM(
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
    uint32_t M, uint32_t K, uint32_t N
) {
    // Quantized weights take the INT4 kernels: the packed bytes are not an FP16
    // matrix and must not be fed to the dense path.
    if (isQuantized(B)) {
        if (M == 1 && gemvInt4Pipeline_) {
            [enc setComputePipelineState:gemvInt4Pipeline_];
            [enc setBuffer:A offset:0 atIndex:0];
            [enc setBuffer:B offset:0 atIndex:1];
            [enc setBuffer:C offset:0 atIndex:2];
            uint32_t uK = K, uN = N;
            [enc setBytes:&uK length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&uN length:sizeof(uint32_t) atIndex:4];
            dispatchGrid(enc, gemvInt4Pipeline_, MTLSizeMake(N, 1, 1));
            return;
        }
        if (fusedGemmInt4Pipeline_) {
            [enc setComputePipelineState:fusedGemmInt4Pipeline_];
            [enc setBuffer:A offset:0 atIndex:0];
            [enc setBuffer:B offset:0 atIndex:1];
            [enc setBuffer:C offset:0 atIndex:2];
            uint32_t uN = M, uK = K, uM = N;
            [enc setBytes:&uN length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&uK length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&uM length:sizeof(uint32_t) atIndex:5];
            MTLSize tg = MTLSizeMake((N + 7) / 8, (M + 7) / 8, 1);
            [enc dispatchThreadgroups:tg threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            return;
        }
        // No INT4 pipeline: refuse rather than reinterpret packed bytes as FP16.
        std::cerr << "[dispatchGEMM] quantized weights but no INT4 pipeline available"
                  << std::endl;
        return;
    }

    if (M == 1 && gemvPipeline_) {
        [enc setComputePipelineState:gemvPipeline_];
        [enc setBuffer:A offset:0 atIndex:0];   // vector x [K]
        [enc setBuffer:B offset:0 atIndex:1];   // matrix B [K x N]
        [enc setBuffer:C offset:0 atIndex:2];   // output vector y [N]
        
        uint32_t uK = K, uN = N;
        [enc setBytes:&uK length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&uN length:sizeof(uint32_t) atIndex:4];
        
        uint32_t threadsPerTG = std::min(N, (uint32_t)256);
        MTLSize tgGroups = MTLSizeMake((N + threadsPerTG - 1) / threadsPerTG, 1, 1);
        MTLSize threadsTG = MTLSizeMake(threadsPerTG, 1, 1);
        [enc dispatchThreadgroups:tgGroups threadsPerThreadgroup:threadsTG];
        return;
    }

    if (!gemmPipeline_) return;
    
    [enc setComputePipelineState:gemmPipeline_];
    [enc setBuffer:A offset:0 atIndex:0];   // activations [M x K]
    [enc setBuffer:B offset:0 atIndex:1];   // weights [K x N]
    [enc setBuffer:C offset:0 atIndex:2];   // output [M x N]
    
    uint32_t uM = M, uK = K, uN = N;
    [enc setBytes:&uM length:sizeof(uint32_t) atIndex:3];   // N_batch
    [enc setBytes:&uK length:sizeof(uint32_t) atIndex:4];   // K_dim
    [enc setBytes:&uN length:sizeof(uint32_t) atIndex:5];   // M_dim
    
    MTLSize threadgroups = MTLSizeMake((N + 7) / 8, (M + 7) / 8, 1);
    MTLSize threadsPerTG = MTLSizeMake(32, 1, 1);
    [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerTG];
}

void MetalTransformerEngine::dispatchRMSNorm(
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> input, id<MTLBuffer> weight, id<MTLBuffer> output,
    uint32_t batch, uint32_t dim
) {
    if (!rmsnormPipeline_) return;
    
    [enc setComputePipelineState:rmsnormPipeline_];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:weight offset:0 atIndex:1];
    [enc setBuffer:output offset:0 atIndex:2];
    
    uint32_t uDim = dim;
    float eps = config_.norm_eps;
    [enc setBytes:&uDim length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&eps length:sizeof(float) atIndex:4];
    
    // One threadgroup per batch element, 256 threads per group
    uint32_t threadsPerTG = std::min(dim, (uint32_t)256);
    MTLSize threads = MTLSizeMake(threadsPerTG, 1, 1);

    // One threadgroup per batch element. (This previously carried a dead
    // MTLSizeMake(1,1,1) initialiser that made the site look like the
    // one-thread dispatches elsewhere in this file, which it never was.)
    MTLSize tg = MTLSizeMake(batch, 1, 1);
    [enc dispatchThreadgroups:tg threadsPerThreadgroup:threads];
}

void MetalTransformerEngine::dispatchRoPE(
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> q, id<MTLBuffer> k,
    uint32_t start_pos, uint32_t batch,
    uint32_t seq_len
) {
    if (!ropePipeline_) return;
    
    [enc setComputePipelineState:ropePipeline_];
    [enc setBuffer:q offset:0 atIndex:0];
    [enc setBuffer:k offset:0 atIndex:1];
    [enc setBuffer:ropeFreqsCos_ offset:0 atIndex:2];
    [enc setBuffer:ropeFreqsSin_ offset:0 atIndex:3];
    
    uint32_t n_heads = config_.n_heads;
    uint32_t n_kv_heads = config_.n_kv_heads;
    uint32_t head_dim = config_.head_dim;
    uint32_t spos = start_pos;
    
    [enc setBytes:&seq_len length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&n_heads length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&n_kv_heads length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&spos length:sizeof(uint32_t) atIndex:8];
    // max_seq, so rope_kernel can bound its read of the frequency tables itself
    // rather than trusting the caller to have bounded absolute_pos.
    uint32_t rope_max_seq = config_.max_seq_len;
    [enc setBytes:&rope_max_seq length:sizeof(uint32_t) atIndex:9];
    
    // Grid: (batch * seq_len, max(n_heads, n_kv_heads), head_dim / 2)
    uint32_t max_heads = std::max(n_heads, n_kv_heads);
    MTLSize grid = MTLSizeMake(batch * seq_len, max_heads, head_dim / 2);
    dispatchGrid(enc, ropePipeline_, grid);
}


// ============================================================================
// Forward Layer — Full Single-Token Decode Through One Transformer Layer
// ============================================================================

void MetalTransformerEngine::forwardLayer(
    id<MTLCommandBuffer> cmdBuf,
    id<MTLComputeCommandEncoder> enc,
    int layer_idx,
    id<MTLBuffer> input,    // [batch_size, hidden_dim]
    id<MTLBuffer> output,   // [batch_size, hidden_dim]
    uint32_t batch_size,
    uint32_t seq_pos,       // current position in the sequence
    uint32_t channel_offset
) {
    const auto& lw = layerWeights_[layer_idx];
    uint32_t H = config_.hidden_dim;         // 2048
    uint32_t I = config_.intermediate_dim;   // 5632
    uint32_t KV_DIM = config_.n_kv_heads * config_.head_dim;  // 256
    uint32_t M = batch_size;

    // 1. RMSNorm on input hidden state -> scratch1_
    dispatchRMSNorm(enc, input, lw.input_norm, scratch1_, M, H);

    // 2. Batched GEMM Projections (Q, K, V) across all M channels simultaneously
    dispatchGEMM(enc, scratch1_, lw.q_proj, scratch2_, M, H, H);        // Q [M, H] -> scratch2_
    dispatchGEMM(enc, scratch1_, lw.k_proj, scratch3_, M, H, KV_DIM);   // K [M, KV_DIM] -> scratch3_
    dispatchGEMM(enc, scratch1_, lw.v_proj, scratchV_, M, H, KV_DIM);   // V [M, KV_DIM] -> scratchV_

    // 3. Apply RoPE to Q and K for all M channels (seq_len = 1 in single-token decode)
    dispatchRoPE(enc, scratch2_, scratch3_, seq_pos, M, 1);

    // 4. Per-channel KV cache append and Attention calculation
    for (uint32_t c = 0; c < M; c++) {
        uint32_t ch = channel_offset + c;
        if (ch >= config_.n_channels) ch = ch % config_.n_channels;
        size_t k_offset = c * KV_DIM * sizeof(_Float16);
        size_t v_offset = c * KV_DIM * sizeof(_Float16);
        size_t q_offset = c * H * sizeof(_Float16);
        size_t attn_out_offset = c * H * sizeof(_Float16);
        size_t attn_score_offset = c * config_.n_heads * config_.max_seq_len * sizeof(_Float16);

        if (kvAppendPipeline_) {
            [enc setComputePipelineState:kvAppendPipeline_];
            // Append K
            [enc setBuffer:scratch3_ offset:k_offset atIndex:0];
            [enc setBuffer:kvCaches_[layer_idx][ch].k_cache offset:0 atIndex:1];
            uint32_t nkv = config_.n_kv_heads, maxseq = config_.max_seq_len, hdim = config_.head_dim, wpos = seq_pos;
            uint32_t ql = 1;
            [enc setBytes:&nkv length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&hdim length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&maxseq length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&wpos length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&ql length:sizeof(uint32_t) atIndex:6];
            MTLSize grid = MTLSizeMake(ql, nkv, hdim);
            dispatchGrid(enc, kvAppendPipeline_, grid);

            // Append V
            [enc setBuffer:scratchV_ offset:v_offset atIndex:0];
            [enc setBuffer:kvCaches_[layer_idx][ch].v_cache offset:0 atIndex:1];
            dispatchGrid(enc, kvAppendPipeline_, grid);
        }

        uint32_t cur_seq_len = seq_pos + 1;
        if (attnScoresPipeline_) {
            [enc setComputePipelineState:attnScoresPipeline_];
            [enc setBuffer:scratch2_ offset:q_offset atIndex:0];
            [enc setBuffer:kvCaches_[layer_idx][ch].k_cache offset:0 atIndex:1];
            [enc setBuffer:scratchAttn_ offset:attn_score_offset atIndex:2];
            uint32_t nh = config_.n_heads, nkv = config_.n_kv_heads, hd = config_.head_dim;
            uint32_t sl = cur_seq_len, ms = config_.max_seq_len, ql = 1;
            [enc setBytes:&nh length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&nkv length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&hd length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&sl length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&ms length:sizeof(uint32_t) atIndex:7];
            [enc setBytes:&ql length:sizeof(uint32_t) atIndex:8];
            MTLSize grid = MTLSizeMake(nh, ql, sl);
            dispatchGrid(enc, attnScoresPipeline_, grid);
        }

        if (softmaxPipeline_) {
            [enc setComputePipelineState:softmaxPipeline_];
            [enc setBuffer:scratchAttn_ offset:attn_score_offset atIndex:0];
            [enc setBuffer:scratchAttn_ offset:attn_score_offset atIndex:1];
            uint32_t sl = cur_seq_len;
            [enc setBytes:&sl length:sizeof(uint32_t) atIndex:2];
            MTLSize tg_count = MTLSizeMake(config_.n_heads, 1, 1);
            MTLSize tg_size = MTLSizeMake(32, 1, 1);
            [enc dispatchThreadgroups:tg_count threadsPerThreadgroup:tg_size];
        }

        if (attnValuePipeline_) {
            [enc setComputePipelineState:attnValuePipeline_];
            [enc setBuffer:scratchAttn_ offset:attn_score_offset atIndex:0];
            [enc setBuffer:kvCaches_[layer_idx][ch].v_cache offset:0 atIndex:1];
            [enc setBuffer:scratch2_ offset:attn_out_offset atIndex:2];
            uint32_t nh = config_.n_heads, nkv = config_.n_kv_heads;
            uint32_t sl = cur_seq_len, hd = config_.head_dim, ms = config_.max_seq_len, ql = 1;
            [enc setBytes:&nh length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&nkv length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&sl length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&hd length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&ms length:sizeof(uint32_t) atIndex:7];
            [enc setBytes:&ql length:sizeof(uint32_t) atIndex:8];
            MTLSize grid = MTLSizeMake(nh, ql, hd);
            dispatchGrid(enc, attnValuePipeline_, grid);
        }
    }

    // 5. O Projection (Batched GEMM M x H @ H x H)
    dispatchGEMM(enc, scratch2_, lw.o_proj, scratch1_, M, H, H);

    // 6. Residual Add 1: scratch3_ = input + scratch1_ (attn_out)
    if (residualPipeline_) {
        [enc setComputePipelineState:residualPipeline_];
        [enc setBuffer:scratch1_ offset:0 atIndex:0];
        [enc setBuffer:input offset:0 atIndex:1];
        [enc setBuffer:scratch3_ offset:0 atIndex:2];
        uint32_t total_size = M * H;
        [enc setBytes:&total_size length:sizeof(uint32_t) atIndex:3];
        MTLSize grid = MTLSizeMake(total_size, 1, 1);
        dispatchGrid(enc, residualPipeline_, grid);
    }

    // 7. MLP RMSNorm: scratch1_ = RMSNorm(scratch3_, post_attn_norm)
    dispatchRMSNorm(enc, scratch3_, lw.post_attn_norm, scratch1_, M, H);

    // 8. Gate and Up Projections (Batched GEMM M x H @ H x I)
    dispatchGEMM(enc, scratch1_, lw.gate_proj, scratch2_, M, H, I);
    id<MTLBuffer> up_buf = scratchAttn_;
    dispatchGEMM(enc, scratch1_, lw.up_proj, up_buf, M, H, I);

    // 9. SiLU(Gate) * Up -> scratch2_
    if (siluMulPipeline_) {
        [enc setComputePipelineState:siluMulPipeline_];
        [enc setBuffer:scratch2_ offset:0 atIndex:0];
        [enc setBuffer:up_buf offset:0 atIndex:1];
        [enc setBuffer:scratch2_ offset:0 atIndex:2];
        uint32_t total_inter = M * I;
        [enc setBytes:&total_inter length:sizeof(uint32_t) atIndex:3];
        MTLSize grid = MTLSizeMake(total_inter, 1, 1);
        dispatchGrid(enc, siluMulPipeline_, grid);
    }

    // 10. Down Projection (Batched GEMM M x I @ I x H)
    dispatchGEMM(enc, scratch2_, lw.down_proj, scratch1_, M, I, H);

    // 11. Final residual: output = scratch3_ + scratch1_
    if (residualPipeline_) {
        [enc setComputePipelineState:residualPipeline_];
        [enc setBuffer:scratch1_ offset:0 atIndex:0];
        [enc setBuffer:scratch3_ offset:0 atIndex:1];
        [enc setBuffer:output offset:0 atIndex:2];
        uint32_t total_size = M * H;
        [enc setBytes:&total_size length:sizeof(uint32_t) atIndex:3];
        MTLSize grid = MTLSizeMake(total_size, 1, 1);
        dispatchGrid(enc, residualPipeline_, grid);
    }
}

void MetalTransformerEngine::forwardBatched(
    id<MTLCommandBuffer> cmdBuf,
    id<MTLComputeCommandEncoder> enc,
    int layer_idx,
    id<MTLBuffer> input,    // [batch_size * q_len, hidden_dim]
    id<MTLBuffer> output,   // [batch_size * q_len, hidden_dim]
    uint32_t batch_size,
    uint32_t q_len,
    uint32_t seq_pos        // position of the start of this chunk in the KV cache
) {
    const auto& lw = layerWeights_[layer_idx];
    uint32_t H = config_.hidden_dim;
    uint32_t I = config_.intermediate_dim;
    uint32_t KV_DIM = config_.n_kv_heads * config_.head_dim;
    uint32_t M = batch_size * q_len;

    dispatchRMSNorm(enc, input, lw.input_norm, scratch1_, M, H);

    dispatchGEMM(enc, scratch1_, lw.q_proj, scratch2_, M, H, H);
    dispatchGEMM(enc, scratch1_, lw.k_proj, scratch3_, M, H, KV_DIM);
    dispatchGEMM(enc, scratch1_, lw.v_proj, scratchV_, M, H, KV_DIM);

    // RoPE takes q_len as its sequence length dimension
    dispatchRoPE(enc, scratch2_, scratch3_, seq_pos, batch_size, q_len);

    for (uint32_t c = 0; c < batch_size; c++) {
        size_t k_offset = c * q_len * KV_DIM * sizeof(_Float16);
        size_t v_offset = c * q_len * KV_DIM * sizeof(_Float16);
        size_t q_offset = c * q_len * H * sizeof(_Float16);
        size_t attn_out_offset = c * q_len * H * sizeof(_Float16);
        size_t attn_score_offset = c * q_len * config_.n_heads * config_.max_seq_len * sizeof(_Float16);

        if (kvAppendPipeline_) {
            [enc setComputePipelineState:kvAppendPipeline_];
            [enc setBuffer:scratch3_ offset:k_offset atIndex:0];
            [enc setBuffer:kvCaches_[layer_idx][c].k_cache offset:0 atIndex:1];
            uint32_t nkv = config_.n_kv_heads, maxseq = config_.max_seq_len, hdim = config_.head_dim, wpos = seq_pos;
            uint32_t ql = q_len;
            [enc setBytes:&nkv length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&hdim length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&maxseq length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&wpos length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&ql length:sizeof(uint32_t) atIndex:6];
            MTLSize grid = MTLSizeMake(ql, nkv, hdim);
            dispatchGrid(enc, kvAppendPipeline_, grid);

            [enc setBuffer:scratchV_ offset:v_offset atIndex:0];
            [enc setBuffer:kvCaches_[layer_idx][c].v_cache offset:0 atIndex:1];
            dispatchGrid(enc, kvAppendPipeline_, grid);
        }

        uint32_t cur_seq_len = seq_pos + q_len;
        if (attnScoresPipeline_) {
            [enc setComputePipelineState:attnScoresPipeline_];
            [enc setBuffer:scratch2_ offset:q_offset atIndex:0];
            [enc setBuffer:kvCaches_[layer_idx][c].k_cache offset:0 atIndex:1];
            [enc setBuffer:scratchAttn_ offset:attn_score_offset atIndex:2];
            uint32_t nh = config_.n_heads, nkv = config_.n_kv_heads, hd = config_.head_dim;
            uint32_t sl = cur_seq_len, ms = config_.max_seq_len, ql = q_len;
            [enc setBytes:&nh length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&nkv length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&hd length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&sl length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&ms length:sizeof(uint32_t) atIndex:7];
            [enc setBytes:&ql length:sizeof(uint32_t) atIndex:8];
            MTLSize grid = MTLSizeMake(1 * nh, ql, sl);
            dispatchGrid(enc, attnScoresPipeline_, grid);
        }

        if (softmaxPipeline_) {
            [enc setComputePipelineState:softmaxPipeline_];
            [enc setBuffer:scratchAttn_ offset:attn_score_offset atIndex:0];
            [enc setBuffer:scratchAttn_ offset:attn_score_offset atIndex:1];
            uint32_t sl = cur_seq_len;
            [enc setBytes:&sl length:sizeof(uint32_t) atIndex:2];
            uint32_t threadsPerTG = std::min(cur_seq_len, (uint32_t)256);
            MTLSize tg_count = MTLSizeMake(1 * config_.n_heads * q_len, 1, 1);
            MTLSize tg_size = MTLSizeMake(threadsPerTG, 1, 1);
            [enc dispatchThreadgroups:tg_count threadsPerThreadgroup:tg_size];
        }

        if (attnValuePipeline_) {
            [enc setComputePipelineState:attnValuePipeline_];
            [enc setBuffer:scratchAttn_ offset:attn_score_offset atIndex:0];
            [enc setBuffer:kvCaches_[layer_idx][c].v_cache offset:0 atIndex:1];
            [enc setBuffer:scratch2_ offset:attn_out_offset atIndex:2];
            uint32_t nh = config_.n_heads, nkv = config_.n_kv_heads;
            uint32_t sl = cur_seq_len, hd = config_.head_dim, ms = config_.max_seq_len, ql = q_len;
            [enc setBytes:&nh length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&nkv length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&sl length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&hd length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&ms length:sizeof(uint32_t) atIndex:7];
            [enc setBytes:&ql length:sizeof(uint32_t) atIndex:8];
            MTLSize grid = MTLSizeMake(1 * nh, ql, hd);
            dispatchGrid(enc, attnValuePipeline_, grid);
        }
    }

    dispatchGEMM(enc, scratch2_, lw.o_proj, scratch1_, M, H, H);

    if (residualPipeline_) {
        [enc setComputePipelineState:residualPipeline_];
        [enc setBuffer:scratch1_ offset:0 atIndex:0];
        [enc setBuffer:input offset:0 atIndex:1];
        [enc setBuffer:scratch3_ offset:0 atIndex:2];
        uint32_t total_size = M * H;
        [enc setBytes:&total_size length:sizeof(uint32_t) atIndex:3];
        MTLSize grid = MTLSizeMake(total_size, 1, 1);
        dispatchGrid(enc, residualPipeline_, grid);
    }

    dispatchRMSNorm(enc, scratch3_, lw.post_attn_norm, scratch1_, M, H);

    dispatchGEMM(enc, scratch1_, lw.gate_proj, scratch2_, M, H, I);
    id<MTLBuffer> up_buf = scratchAttn_;
    dispatchGEMM(enc, scratch1_, lw.up_proj, up_buf, M, H, I);

    if (siluMulPipeline_) {
        [enc setComputePipelineState:siluMulPipeline_];
        [enc setBuffer:scratch2_ offset:0 atIndex:0];
        [enc setBuffer:up_buf offset:0 atIndex:1];
        [enc setBuffer:scratch2_ offset:0 atIndex:2];
        uint32_t total_inter = M * I;
        [enc setBytes:&total_inter length:sizeof(uint32_t) atIndex:3];
        MTLSize grid = MTLSizeMake(total_inter, 1, 1);
        dispatchGrid(enc, siluMulPipeline_, grid);
    }

    dispatchGEMM(enc, scratch2_, lw.down_proj, scratch1_, M, I, H);

    if (residualPipeline_) {
        [enc setComputePipelineState:residualPipeline_];
        [enc setBuffer:scratch1_ offset:0 atIndex:0];
        [enc setBuffer:scratch3_ offset:0 atIndex:1];
        [enc setBuffer:output offset:0 atIndex:2];
        uint32_t total_size = M * H;
        [enc setBytes:&total_size length:sizeof(uint32_t) atIndex:3];
        MTLSize grid = MTLSizeMake(total_size, 1, 1);
        dispatchGrid(enc, residualPipeline_, grid);
    }
}


// ============================================================================
// CPU-Side Token Sampling (Top-P Nucleus Sampling)
// ============================================================================

int32_t MetalTransformerEngine::sampleToken(
    const _Float16* logits, int vocab_size,
    float temperature, float top_p, std::mt19937& rng
) {
    // The arithmetic lives in src/sampling.h, which has no Metal dependency, so the
    // NaN handling that used to pin every draw to token 0 is covered by a test that
    // runs on any machine rather than only on a Mac.
    antigravity::SamplingStats stats;
    const int32_t token = antigravity::sampleTokenFromLogits(
        logits, vocab_size, temperature, top_p, rng, stats);

    nonFiniteLogitCount_ += stats.non_finite_logits;
    if (stats.empty_distributions && emptyDistributionCount_ == 0) {
        // Once, not once per token: a failed forward pass would otherwise print this
        // for every position in every sequence.
        std::cerr << "[sampleToken] every one of " << vocab_size << " logits is NaN or "
                     "infinite; the forward pass produced no usable distribution"
                  << std::endl;
    }
    emptyDistributionCount_ += stats.empty_distributions;
    return token;
}


// ============================================================================
// Full Autoregressive Generation — The Core Decode Loop
// ============================================================================

void MetalTransformerEngine::rollbackKVCache(uint32_t step) {
    // No-op for now. KV cache state is determined purely by the `seq_pos` parameter 
    // passed to forwardBatched / forwardLayer during generation. When we rollback, 
    // we simply decrement `seq_pos` and overwrite the rejected tokens.
}

GenerationResult MetalTransformerEngine::generateSpeculative(
    ITransformerEngine* abstract_draft_engine,
    const int32_t* prompt_tokens,
    int32_t prompt_len,
    int32_t max_new_tokens,
    int32_t k_draft,
    float temperature,
    float top_p
) {
    MetalTransformerEngine* draft_engine = dynamic_cast<MetalTransformerEngine*>(abstract_draft_engine);
    if (!draft_engine) {
        std::cerr << "[generateSpeculative] Draft engine must be of type MetalTransformerEngine!" << std::endl;
        return GenerationResult();
    }

    if (!weightsLoaded_ || !draft_engine->weightsLoaded_) {
        std::cerr << "[generateSpeculative] Weights not loaded!" << std::endl;
        return GenerationResult();
    }

    // Also checked at the C API boundary, but this method is public and the Swift
    // SDK reaches it directly. k_draft + 1 rows are forwarded in one pass and
    // q_len_max is what the scratch buffers were sized for; past that, the GEMM
    // writes outside them.
    {
        antigravity::LimitError lim = antigravity::checkDraftChunk(k_draft, config_.q_len_max);
        if (lim != antigravity::LimitError::Ok) {
            std::cerr << "[generateSpeculative] " << antigravity::describe(lim)
                      << " (k_draft=" << k_draft << ", q_len_max=" << config_.q_len_max
                      << ")" << std::endl;
            return GenerationResult();
        }
        lim = antigravity::checkSequence(prompt_len, max_new_tokens, config_.max_seq_len);
        if (lim != antigravity::LimitError::Ok) {
            std::cerr << "[generateSpeculative] " << antigravity::describe(lim)
                      << " (prompt_len=" << prompt_len << ", max_new_tokens="
                      << max_new_tokens << ", max_seq_len=" << config_.max_seq_len
                      << ")" << std::endl;
            return GenerationResult();
        }
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    std::random_device rd;
    std::mt19937 rng(rd());

    GenerationResult res;
    res.channel_tokens.resize(1); // Speculative decoding prototype is 1-channel for now
    for (int i = 0; i < prompt_len; i++) {
        res.channel_tokens[0].push_back(prompt_tokens[i]);
    }
    res.channel_logprobs.resize(1, 0.0f);

    uint32_t seq_pos = 0;
    uint32_t draft_seq_pos = 0;

    // 1. Prefill both models
    int32_t current_token = prompt_tokens[0];
    for (int t = 0; t < prompt_len - 1; t++) {
        id<MTLCommandBuffer> cmdBufDraft = [draft_engine->queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encDraft = [cmdBufDraft computeCommandEncoder];
        
        [encDraft setComputePipelineState:draft_engine->embedPipeline_];
        [encDraft setBytes:&current_token length:sizeof(int32_t) atIndex:0];
        [encDraft setBuffer:draft_engine->embedWeights_ offset:0 atIndex:1];
        [encDraft setBuffer:draft_engine->scratch1_ offset:0 atIndex:2];
        uint32_t H_draft = draft_engine->config_.hidden_dim;
        [encDraft setBytes:&H_draft length:sizeof(uint32_t) atIndex:3];
        MTLSize gridD = MTLSizeMake(1, H_draft, 1);
        draft_engine->dispatchGrid(encDraft, draft_engine->embedPipeline_, gridD);

        for (int l = 0; l < draft_engine->config_.n_layers; l++) {
            draft_engine->forwardLayer(cmdBufDraft, encDraft, l, draft_engine->scratch1_, draft_engine->scratch1_, 1, draft_seq_pos);
        }
        [encDraft endEncoding];
        [cmdBufDraft commit];
        [cmdBufDraft waitUntilCompleted];
        
        id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        
        [enc setComputePipelineState:embedPipeline_];
        [enc setBytes:&current_token length:sizeof(int32_t) atIndex:0];
        [enc setBuffer:embedWeights_ offset:0 atIndex:1];
        [enc setBuffer:scratch1_ offset:0 atIndex:2];
        uint32_t H = config_.hidden_dim;
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:3];
        MTLSize grid = MTLSizeMake(1, H, 1);
        dispatchGrid(enc, embedPipeline_, grid);

        for (int l = 0; l < config_.n_layers; l++) {
            forwardLayer(cmdBuf, enc, l, scratch1_, scratch1_, 1, seq_pos);
        }
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        current_token = prompt_tokens[t + 1];
        seq_pos++;
        draft_seq_pos++;
    }

    auto first_token_time = std::chrono::high_resolution_clock::now();
    res.ttft_ms = std::chrono::duration<double, std::milli>(first_token_time - start_time).count();

    // 2. Generation Loop
    int tokens_generated = 0;
    while (tokens_generated < max_new_tokens) {
        // A. Draft Phase: Generate K tokens using Draft Engine
        std::vector<int32_t> draft_tokens;
        int32_t draft_current_token = current_token;
        
        for (int k = 0; k < k_draft; k++) {
            id<MTLCommandBuffer> cmdBufDraft = [draft_engine->queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encDraft = [cmdBufDraft computeCommandEncoder];
            
            [encDraft setComputePipelineState:draft_engine->embedPipeline_];
            [encDraft setBytes:&draft_current_token length:sizeof(int32_t) atIndex:0];
            [encDraft setBuffer:draft_engine->embedWeights_ offset:0 atIndex:1];
            [encDraft setBuffer:draft_engine->scratch1_ offset:0 atIndex:2];
            uint32_t H_draft = draft_engine->config_.hidden_dim;
            [encDraft setBytes:&H_draft length:sizeof(uint32_t) atIndex:3];
            MTLSize gridD = MTLSizeMake(1, H_draft, 1);
            draft_engine->dispatchGrid(encDraft, draft_engine->embedPipeline_, gridD);

            for (int l = 0; l < draft_engine->config_.n_layers; l++) {
                draft_engine->forwardLayer(cmdBufDraft, encDraft, l, draft_engine->scratch1_, draft_engine->scratch1_, 1, draft_seq_pos + k);
            }
            
            draft_engine->dispatchRMSNorm(encDraft, draft_engine->scratch1_, draft_engine->finalNorm_, draft_engine->scratch2_, 1, H_draft);
            draft_engine->dispatchGEMM(encDraft, draft_engine->scratch2_, draft_engine->lmHead_, draft_engine->scratchLogits_, 1, H_draft, draft_engine->config_.vocab_size);
            
            [encDraft endEncoding];
            [cmdBufDraft commit];
            [cmdBufDraft waitUntilCompleted];
            
            _Float16* d_logits = (_Float16*)[draft_engine->scratchLogits_ contents];
            // Force greedy for draft
            // Greedy (temperature 0, top_p 1) deliberately, NOT the caller's temperature
            // and top_p. The acceptance test below is an exact match against the target's
            // own greedy pick, which is only distribution-correct for greedy decoding.
            // Sampling here without the probability-ratio accept/reject step would silently
            // change the output distribution. See AntigravityEngineNativeGenerateSpeculative.
            int32_t next_t = draft_engine->sampleToken(d_logits, draft_engine->config_.vocab_size, 0.0f, 1.0f, rng);
            draft_tokens.push_back(next_t);
            draft_current_token = next_t;
        }

        // B. Verification Phase: Target Engine evaluates K+1 tokens in parallel
        // The input to Target is: [current_token, draft_tokens[0], ..., draft_tokens[k-1]]
        std::vector<int32_t> eval_tokens = {current_token};
        eval_tokens.insert(eval_tokens.end(), draft_tokens.begin(), draft_tokens.end());
        uint32_t q_len = eval_tokens.size(); // k + 1

        id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        
        [enc setComputePipelineState:embedPipeline_];
        [enc setBytes:eval_tokens.data() length:q_len * sizeof(int32_t) atIndex:0];
        [enc setBuffer:embedWeights_ offset:0 atIndex:1];
        [enc setBuffer:scratch1_ offset:0 atIndex:2];
        uint32_t H = config_.hidden_dim;
        [enc setBytes:&H length:sizeof(uint32_t) atIndex:3];
        MTLSize grid = MTLSizeMake(q_len, H, 1);
        dispatchGrid(enc, embedPipeline_, grid);

        for (int l = 0; l < config_.n_layers; l++) {
            forwardBatched(cmdBuf, enc, l, scratch1_, scratch1_, 1, q_len, seq_pos);
        }
        
        dispatchRMSNorm(enc, scratch1_, finalNorm_, scratch2_, q_len, H);
        dispatchGEMM(enc, scratch2_, lmHead_, scratchLogits_, q_len, H, config_.vocab_size);
        
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        
        _Float16* t_logits = (_Float16*)[scratchLogits_ contents];
        
        // C. Acceptance logic (Greedy)
        int accepted = 0;
        for (int i = 0; i < q_len; i++) {
            _Float16* row_logits = t_logits + i * config_.vocab_size;
            // Greedy for the same reason as the draft sampling above.
            int32_t target_tok = sampleToken(row_logits, config_.vocab_size, 0.0f, 1.0f, rng);
            
            res.channel_tokens[0].push_back(target_tok);
            tokens_generated++;
            current_token = target_tok;
            
            if (i < k_draft && target_tok == draft_tokens[i]) {
                accepted++;
            } else {
                break;
            }
        }
        
        // D. Rollback seq_pos based on rejected tokens
        seq_pos += (accepted + 1);
        draft_seq_pos = seq_pos; // sync draft seq pos
        
        if (current_token == 2) { // EOS
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    res.total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    res.tpot_ms = (res.total_ms - res.ttft_ms) / std::max(1, tokens_generated);
    res.total_tokens = tokens_generated;
    res.best_channel = 0;
    res.best_score = 0.0f;

    return res;
}

GenerationResult MetalTransformerEngine::generate(
    const int32_t* prompt_tokens,
    int32_t prompt_len,
    int32_t max_new_tokens,
    float temperature,
    float top_p
) {
    GenerationResult result;
    result.channel_tokens.resize(config_.n_channels);
    result.channel_logprobs.resize(config_.n_channels, 0.0f);
    result.ttft_ms = 0;
    result.tpot_ms = 0;
    result.total_tokens = 0;
    result.best_channel = 0;
    result.best_score = 0;
    
    if (!weightsLoaded_) {
        std::cerr << "[generate] Weights not loaded!" << std::endl;
        return result;
    }

    // The KV cache holds max_seq_len positions per layer and decode writes at
    // seq_pos = prompt_len + step. Nothing bounded that before, so a long enough
    // prompt or generation walked past the end of every layer's cache.
    {
        antigravity::LimitError lim =
            antigravity::checkSequence(prompt_len, max_new_tokens, config_.max_seq_len);
        if (lim != antigravity::LimitError::Ok) {
            std::cerr << "[generate] " << antigravity::describe(lim)
                      << " (prompt_len=" << prompt_len << ", max_new_tokens="
                      << max_new_tokens << ", max_seq_len=" << config_.max_seq_len
                      << "); room for " << antigravity::remainingCapacity(prompt_len,
                                                                          config_.max_seq_len)
                      << " more tokens" << std::endl;
            return result;
        }
    }

    const uint32_t H = config_.hidden_dim;
    const uint32_t C = config_.n_channels;
    const bool is_qwen = (config_.vocab_size > 32000);
    const int EOS_TOKEN_1 = is_qwen ? 151645 : 2;
    const int EOS_TOKEN_2 = is_qwen ? 151643 : 2;
    
    std::random_device rd;
    std::vector<std::mt19937> channel_rngs(C);
    for (uint32_t c = 0; c < C; c++) {
        channel_rngs[c].seed(rd() + c * 10007);
    }
    
    // Track active channels (not yet hit EOS)
    std::vector<bool> channel_active(C, true);
    
    // Allocate unified hidden state buffers holding all C channels contiguously
    size_t batch_hidden_bytes = C * H * sizeof(uint16_t);
    id<MTLBuffer> batch_hidden_1 = [device_ newBufferWithLength:batch_hidden_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> batch_hidden_2 = [device_ newBufferWithLength:batch_hidden_bytes options:MTLResourceStorageModeShared];
    
    auto t_start = std::chrono::high_resolution_clock::now();
    bool ttft_recorded = false;
    double sum_decode_ms = 0;
    int decode_steps = 0;
    
    // ---- Prefill: Process prompt tokens 0..prompt_len-2 into KV cache ----
    for (int t = 0; t < prompt_len - 1; t++) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            
            // Embedding lookup for prompt token t for all C channels
            if (embedPipeline_) {
                [enc setComputePipelineState:embedPipeline_];
                uint32_t tok = (uint32_t)prompt_tokens[t];
                if (tok >= (uint32_t)config_.vocab_size) tok = 0;
                for (uint32_t c = 0; c < C; c++) {
                    // setBytes: copies the 4-byte token id into the encoder. This was a fresh MTLBuffer
                    // per token per channel — thousands of object allocations inside the decode loop.
                    [enc setBytes:&tok length:sizeof(uint32_t) atIndex:0];
                    [enc setBuffer:embedWeights_ offset:0 atIndex:1];
                    [enc setBuffer:batch_hidden_1 offset:c * H * sizeof(uint16_t) atIndex:2];
                    uint32_t hdim = H;
                    [enc setBytes:&hdim length:sizeof(uint32_t) atIndex:3];
                    MTLSize grid = MTLSizeMake(1, H, 1);
                    dispatchGrid(enc, embedPipeline_, grid);
                }
            }
            
            // Forward through all layers with batch_size = C
            for (int l = 0; l < config_.n_layers; l++) {
                id<MTLBuffer> in_buf  = (l % 2 == 0) ? batch_hidden_1 : batch_hidden_2;
                id<MTLBuffer> out_buf = (l % 2 == 0) ? batch_hidden_2 : batch_hidden_1;
                forwardLayer(cmdBuf, enc, l, in_buf, out_buf, C, t);
            }
            
            [enc endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
        }
    }
    
    // ---- Decode: Autoregressive generation starting from prompt_tokens[prompt_len - 1] ----
    for (int step = 0; step < max_new_tokens; step++) {
        @autoreleasepool {
            auto t_step_start = std::chrono::high_resolution_clock::now();
            uint32_t seq_pos = prompt_len - 1 + step;
            
            id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            
            // Lookup embeddings for each active channel
            if (embedPipeline_) {
                [enc setComputePipelineState:embedPipeline_];
                for (uint32_t c = 0; c < C; c++) {
                    if (!channel_active[c]) {
                        uint16_t* h_ptr = (uint16_t*)[batch_hidden_1 contents] + c * H;
                        std::memset(h_ptr, 0, H * sizeof(uint16_t));
                        continue;
                    }
                    int32_t cur_token = (step == 0) ? prompt_tokens[prompt_len - 1] : result.channel_tokens[c].back();
                    uint32_t tok = (uint32_t)cur_token;
                    if (tok >= (uint32_t)config_.vocab_size) tok = 0;
                    // setBytes: copies the 4-byte token id into the encoder. This was a fresh MTLBuffer
                    // per token per channel — thousands of object allocations inside the decode loop.
                    [enc setBytes:&tok length:sizeof(uint32_t) atIndex:0];
                    [enc setBuffer:embedWeights_ offset:0 atIndex:1];
                    [enc setBuffer:batch_hidden_1 offset:c * H * sizeof(uint16_t) atIndex:2];
                    uint32_t hdim = H;
                    [enc setBytes:&hdim length:sizeof(uint32_t) atIndex:3];
                    MTLSize grid = MTLSizeMake(1, H, 1);
                    dispatchGrid(enc, embedPipeline_, grid);
                }
            }
            
            // Unified forward pass through all layers
            for (int l = 0; l < config_.n_layers; l++) {
                id<MTLBuffer> in_buf  = (l % 2 == 0) ? batch_hidden_1 : batch_hidden_2;
                id<MTLBuffer> out_buf = (l % 2 == 0) ? batch_hidden_2 : batch_hidden_1;
                forwardLayer(cmdBuf, enc, l, in_buf, out_buf, C, seq_pos);
            }
        
        // Final RMSNorm across all C channels [C, H]
        id<MTLBuffer> final_hidden = (config_.n_layers % 2 == 0) ? batch_hidden_1 : batch_hidden_2;
        dispatchRMSNorm(enc, final_hidden, finalNorm_, scratch1_, C, H);
        
        // Batched LM Head projection: GEMM(hidden[C, H], lm_head[H, V], logits[C, V]) with M = C
        dispatchGEMM(enc, scratch1_, lmHead_, scratchLogits_, C, H, config_.vocab_size);
        
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        
        // CPU-side sampling from logits for each active channel
        const _Float16* logits_base = (const _Float16*)[scratchLogits_ contents];
        for (uint32_t c = 0; c < C; c++) {
            if (!channel_active[c]) continue;
            
            const _Float16* logits = logits_base + c * config_.vocab_size;
            int32_t next_token = sampleToken(logits, config_.vocab_size, temperature, top_p, channel_rngs[c]);
            
            // Accumulate log-probability
            float max_logit = -1e9f;
            for (int i = 0; i < config_.vocab_size; i++) {
                float v = (float)logits[i];
                if (v > max_logit) max_logit = v;
            }
            float sum_exp = 0.0f;
            for (int i = 0; i < config_.vocab_size; i++) {
                sum_exp += expf((float)logits[i] - max_logit);
            }
            float token_logprob = (float)logits[next_token] - max_logit - logf(sum_exp);
            result.channel_logprobs[c] += token_logprob;
            
            result.channel_tokens[c].push_back(next_token);
            result.total_tokens++;
            
            if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2 || next_token == 2) {
                channel_active[c] = false;
            }
        }
        
        auto t_step_end = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
        
        if (!ttft_recorded) {
            result.ttft_ms = std::chrono::duration<double, std::milli>(t_step_end - t_start).count();
            ttft_recorded = true;
        } else {
            sum_decode_ms += step_ms;
            decode_steps++;
        }
        
            bool all_done = true;
            for (uint32_t c = 0; c < C; c++) {
                if (channel_active[c]) { all_done = false; break; }
            }
            if (all_done) break;
        }
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.tpot_ms = (decode_steps > 0) ? (sum_decode_ms / decode_steps) : 0;
    
    result.best_channel = 0;
    result.best_score = result.channel_logprobs[0];
    for (uint32_t c = 1; c < C; c++) {
        if (result.channel_logprobs[c] > result.best_score) {
            result.best_score = result.channel_logprobs[c];
            result.best_channel = c;
        }
    }
    
    std::cout << "[generate] Done: " << result.total_tokens << " tokens, "
              << "TTFT=" << result.ttft_ms << "ms, "
              << "TPOT=" << result.tpot_ms << "ms, "
              << "Total=" << result.total_ms << "ms" << std::endl;

    // A run that hit non-finite logits must not look clean. Before this, one NaN
    // anywhere in the vocabulary silently pinned sampling to token 0 for the rest of
    // the sequence, which reads as a model that has collapsed rather than as a fault.
    if (emptyDistributionCount_ > 0) {
        std::cerr << "[generate] WARNING: " << emptyDistributionCount_
                  << " sampling step(s) had no finite logit at all. Those tokens are "
                     "not model output." << std::endl;
    }
    if (nonFiniteLogitCount_ > 0) {
        std::cerr << "[generate] WARNING: discarded " << nonFiniteLogitCount_
                  << " non-finite logits during this generation. The forward pass is "
                     "producing NaN or Inf; the output above is not trustworthy."
                  << std::endl;
    }

    return result;
}

GenerationResult MetalTransformerEngine::generateMultimodal(
    const int32_t* text_tokens,
    int32_t text_len,
    const float* image_embeddings,
    int32_t n_patches,
    int32_t max_new_tokens,
    float temperature,
    float top_p
) {
    GenerationResult result;
    result.channel_tokens.resize(config_.n_channels);
    result.channel_logprobs.resize(config_.n_channels, 0.0f);
    result.ttft_ms = 0;
    result.tpot_ms = 0;
    result.total_tokens = 0;
    result.best_channel = 0;
    result.best_score = 0;

    if (!weightsLoaded_) {
        std::cerr << "[generateMultimodal] Weights not loaded!" << std::endl;
        return result;
    }

    // Prefill is the image patches followed by the text tokens, and both occupy KV
    // cache positions, so the cache has to hold them plus everything generated.
    {
        const int64_t prefill = (int64_t)(n_patches > 0 ? n_patches : 0)
                              + (int64_t)(text_len > 0 ? text_len : 0);
        antigravity::LimitError lim = antigravity::checkSequence(
            (int32_t)std::min<int64_t>(prefill, INT32_MAX), max_new_tokens,
            config_.max_seq_len);
        if (lim != antigravity::LimitError::Ok) {
            std::cerr << "[generateMultimodal] " << antigravity::describe(lim)
                      << " (patches=" << n_patches << ", text_len=" << text_len
                      << ", max_new_tokens=" << max_new_tokens
                      << ", max_seq_len=" << config_.max_seq_len << ")" << std::endl;
            return result;
        }
    }

    const uint32_t H = config_.hidden_dim;

    // EOS was hardcoded to 2, which is Llama's. On a Qwen vocabulary that token never
    // appears as a stop, so multimodal decode ran to max_new_tokens every time and
    // emitted tokens past the end of the response. Match generate()'s detection.
    const bool is_qwen = (config_.vocab_size > 32000);
    const int EOS_TOKEN_1 = is_qwen ? 151645 : 2;
    const int EOS_TOKEN_2 = is_qwen ? 151643 : 2;
    const int EOS_TOKEN = EOS_TOKEN_1;

    // Seeds were fixed constants, so every call produced identical rollouts and the
    // channels differed only by a constant offset. generate() already seeds from
    // std::random_device; do the same here.
    std::random_device rd;
    std::vector<std::mt19937> channel_rngs(config_.n_channels);
    for (int c = 0; c < config_.n_channels; c++) {
        channel_rngs[c].seed(rd() + c * 10007);
    }

    std::vector<bool> channel_active(config_.n_channels, true);

    size_t hidden_bytes = H * sizeof(uint16_t);
    std::vector<id<MTLBuffer>> hidden_bufs(config_.n_channels);
    std::vector<id<MTLBuffer>> hidden_bufs2(config_.n_channels);
    for (int c = 0; c < config_.n_channels; c++) {
        hidden_bufs[c] = [device_ newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];
        hidden_bufs2[c] = [device_ newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    bool ttft_recorded = false;
    double sum_decode_ms = 0;
    int decode_steps = 0;

    int total_prefill_len = n_patches + text_len;

    // ---- Prefill: Vision Patches first, then Text Tokens ----
    for (int t = 0; t < total_prefill_len; t++) {
        for (int c = 0; c < config_.n_channels; c++) {
            id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

            if (t < n_patches && image_embeddings != nullptr) {
                // Image patch embedding: convert FP32 to FP16 and copy directly to hidden_bufs[c]
                _Float16* dst = (_Float16*)[hidden_bufs[c] contents];
                const float* src = image_embeddings + (t * H);
                for (uint32_t i = 0; i < H; i++) {
                    dst[i] = (_Float16)src[i];
                }
            } else {
                // Text token embedding lookup
                int text_idx = t - n_patches;
                if (embedPipeline_ && text_idx >= 0 && text_idx < text_len) {
                    [enc setComputePipelineState:embedPipeline_];
                    uint32_t tok = (uint32_t)text_tokens[text_idx];
                    // Unchecked here, unlike the text path: embedding_lookup_kernel
                    // indexes embed_table[token_id * hidden_dim + ...], so an
                    // out-of-range id is an out-of-bounds GPU read.
                    if (tok >= (uint32_t)config_.vocab_size) tok = 0;
                    // setBytes: copies the 4-byte token id into the encoder. This was a fresh MTLBuffer
                    // per token per channel — thousands of object allocations inside the decode loop.
                    [enc setBytes:&tok length:sizeof(uint32_t) atIndex:0];
                    [enc setBuffer:embedWeights_ offset:0 atIndex:1];
                    [enc setBuffer:hidden_bufs[c] offset:0 atIndex:2];
                    uint32_t hdim = H;
                    [enc setBytes:&hdim length:sizeof(uint32_t) atIndex:3];
                    MTLSize grid = MTLSizeMake(1, H, 1);
                    dispatchGrid(enc, embedPipeline_, grid);
                }
            }

            // Forward through all 22 layers
            for (int l = 0; l < config_.n_layers; l++) {
                id<MTLBuffer> in_buf  = (l % 2 == 0) ? hidden_bufs[c] : hidden_bufs2[c];
                id<MTLBuffer> out_buf = (l % 2 == 0) ? hidden_bufs2[c] : hidden_bufs[c];
                forwardLayer(cmdBuf, enc, l, in_buf, out_buf, 1, t, c);
            }

            [enc endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
        }
    }

    // ---- Decode: Autoregressive generation ----
    for (int step = 0; step < max_new_tokens; step++) {
        auto t_step_start = std::chrono::high_resolution_clock::now();

        for (int c = 0; c < config_.n_channels; c++) {
            if (!channel_active[c]) continue;

            int32_t cur_token;
            if (step == 0) {
                cur_token = (text_len > 0) ? text_tokens[text_len - 1] : EOS_TOKEN;
            } else {
                cur_token = result.channel_tokens[c].back();
            }

            uint32_t seq_pos = total_prefill_len + step;

            id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

            if (embedPipeline_) {
                [enc setComputePipelineState:embedPipeline_];
                uint32_t tok = (uint32_t)cur_token;
                if (tok >= (uint32_t)config_.vocab_size) tok = 0;
                // setBytes: copies the 4-byte token id into the encoder. This was a fresh MTLBuffer
                // per token per channel — thousands of object allocations inside the decode loop.
                [enc setBytes:&tok length:sizeof(uint32_t) atIndex:0];
                [enc setBuffer:embedWeights_ offset:0 atIndex:1];
                [enc setBuffer:hidden_bufs[c] offset:0 atIndex:2];
                uint32_t hdim = H;
                [enc setBytes:&hdim length:sizeof(uint32_t) atIndex:3];
                MTLSize grid = MTLSizeMake(1, H, 1);
                dispatchGrid(enc, embedPipeline_, grid);
            }

            for (int l = 0; l < config_.n_layers; l++) {
                id<MTLBuffer> in_buf  = (l % 2 == 0) ? hidden_bufs[c] : hidden_bufs2[c];
                id<MTLBuffer> out_buf = (l % 2 == 0) ? hidden_bufs2[c] : hidden_bufs[c];
                forwardLayer(cmdBuf, enc, l, in_buf, out_buf, 1, seq_pos, c);
            }

            id<MTLBuffer> final_hidden = (config_.n_layers % 2 == 0) ? hidden_bufs[c] : hidden_bufs2[c];
            dispatchRMSNorm(enc, final_hidden, finalNorm_, scratch1_, 1, H);
            dispatchGEMM(enc, scratch1_, lmHead_, scratchLogits_, 1, H, config_.vocab_size);

            [enc endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            const _Float16* logits = (const _Float16*)[scratchLogits_ contents];
            int32_t next_token = sampleToken(logits, config_.vocab_size, temperature, top_p, channel_rngs[c]);

            float max_logit = -1e9f;
            for (int i = 0; i < config_.vocab_size; i++) {
                float v = (float)logits[i];
                if (v > max_logit) max_logit = v;
            }
            float sum_exp = 0.0f;
            for (int i = 0; i < config_.vocab_size; i++) {
                sum_exp += expf((float)logits[i] - max_logit);
            }
            float token_logprob = (float)logits[next_token] - max_logit - logf(sum_exp);
            result.channel_logprobs[c] += token_logprob;

            result.channel_tokens[c].push_back(next_token);
            result.total_tokens++;

            if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2 || next_token == 2) {
                channel_active[c] = false;
            }
        }

        auto t_step_end = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();

        if (!ttft_recorded) {
            result.ttft_ms = std::chrono::duration<double, std::milli>(t_step_end - t_start).count();
            ttft_recorded = true;
        } else {
            sum_decode_ms += step_ms;
            decode_steps++;
        }

        bool all_done = true;
        for (int c = 0; c < config_.n_channels; c++) {
            if (channel_active[c]) { all_done = false; break; }
        }
        if (all_done) break;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.tpot_ms = (decode_steps > 0) ? (sum_decode_ms / decode_steps) : 0;

    result.best_channel = 0;
    result.best_score = result.channel_logprobs[0];
    for (int c = 1; c < config_.n_channels; c++) {
        if (result.channel_logprobs[c] > result.best_score) {
            result.best_score = result.channel_logprobs[c];
            result.best_channel = c;
        }
    }

    return result;
}

MCTSResult MetalTransformerEngine::generateMCTS(
    const int32_t* prompt_tokens,
    int32_t prompt_len,
    const MCTSConfig& mcts_config
) {
    MCTSResult result;
    result.total_ms = 0;
    result.total_tokens_evaluated = 0;
    result.chunks_expanded = 0;
    result.best_score = 0;

    if (!weightsLoaded_ || prompt_len <= 0) {
        std::cerr << "[generateMCTS] Weights not loaded or invalid prompt!" << std::endl;
        return result;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // Start with prompt tokens as the active sequence prefix
    std::vector<int32_t> current_prefix(prompt_tokens, prompt_tokens + prompt_len);

    const int branches = std::min(mcts_config.branches_per_chunk, config_.n_channels);
    const int EOS_TOKEN = (config_.vocab_size > 32000) ? 151645 : 2;

    for (int chunk_idx = 0; chunk_idx < mcts_config.num_chunks; chunk_idx++) {
        // The prefix grows by a chunk each round, so the KV cache can fill part-way
        // through the search. generate() refuses a request that would not fit, and
        // its refusal still returns n_channels empty vectors — so without this the
        // loop would run every remaining chunk scoring nothing. Stop instead, and
        // ask only for what is left.
        const int32_t room = antigravity::remainingCapacity(
            (int32_t)current_prefix.size(), config_.max_seq_len);
        if (room <= 0) {
            std::cerr << "[generateMCTS] KV cache full after " << chunk_idx
                      << " chunk(s) (" << current_prefix.size() << " of "
                      << config_.max_seq_len << " positions); stopping search"
                      << std::endl;
            break;
        }
        const int32_t chunk_tokens = std::min(mcts_config.chunk_tokens, room);

        // Run parallel candidate generation across branches on Metal GPU
        GenerationResult gen = this->generate(
            current_prefix.data(),
            (int32_t)current_prefix.size(),
            chunk_tokens,
            mcts_config.temperature,
            mcts_config.top_p
        );

        result.total_tokens_evaluated += gen.total_tokens;
        result.chunks_expanded++;

        if (gen.channel_tokens.empty()) break;

        // Score each candidate branch using Process Reward heuristic:
        // 1. Length-normalized log-prob density
        // 2. Token diversity (unique token ratio)
        // 3. Step/Logic coverage
        int best_branch = 0;
        float best_branch_score = -1e9f;

        for (int b = 0; b < branches && b < (int)gen.channel_tokens.size(); b++) {
            const auto& toks = gen.channel_tokens[b];
            if (toks.empty()) continue;

            float logprob = gen.channel_logprobs[b];
            float density = logprob / std::pow((float)toks.size(), 0.6f);

            // Diversity: unique token IDs / total token count
            std::vector<int32_t> sorted_toks = toks;
            std::sort(sorted_toks.begin(), sorted_toks.end());
            int unique_cnt = (int)(std::unique(sorted_toks.begin(), sorted_toks.end()) - sorted_toks.begin());
            float diversity = (float)unique_cnt / (float)toks.size();

            float score = density + diversity * 3.0f + std::log1pf((float)toks.size()) * 0.5f;

            if (score > best_branch_score) {
                best_branch_score = score;
                best_branch = b;
            }
        }

        // Append the winning branch tokens to current_prefix
        const auto& winning_toks = gen.channel_tokens[best_branch];
        bool hit_eos = false;
        for (int32_t tok : winning_toks) {
            current_prefix.push_back(tok);
            if (tok == EOS_TOKEN) {
                hit_eos = true;
                break;
            }
        }

        result.best_score = best_branch_score;

        if (hit_eos || winning_toks.empty()) {
            break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Result best_tokens is the generated portion (excluding the initial prompt)
    if (current_prefix.size() > (size_t)prompt_len) {
        result.best_tokens.assign(current_prefix.begin() + prompt_len, current_prefix.end());
    }

    std::cout << "[generateMCTS] Finished: " << result.best_tokens.size() << " new tokens, "
              << "Chunks=" << result.chunks_expanded << ", "
              << "Evaluated=" << result.total_tokens_evaluated << " total tokens, "
              << "Score=" << result.best_score << ", "
              << "Time=" << result.total_ms << "ms" << std::endl;

    return result;
}

uint64_t MetalTransformerEngine::getAllocatedBytes() const {
    return allocatedBytes_;
}

void MetalTransformerEngine::sanitizeBuffers() {
    // Zero out all scratch and KV cache buffers
    if (scratch1_) memset([scratch1_ contents], 0, [scratch1_ length]);
    if (scratch2_) memset([scratch2_ contents], 0, [scratch2_ length]);
    if (scratch3_) memset([scratch3_ contents], 0, [scratch3_ length]);
    if (scratchLogits_) memset([scratchLogits_ contents], 0, [scratchLogits_ length]);
    if (scratchAttn_) memset([scratchAttn_ contents], 0, [scratchAttn_ length]);
    
    for (auto& layer_caches : kvCaches_) {
        for (auto& kv : layer_caches) {
            if (kv.k_cache) memset([kv.k_cache contents], 0, [kv.k_cache length]);
            if (kv.v_cache) memset([kv.v_cache contents], 0, [kv.v_cache length]);
        }
    }
}
