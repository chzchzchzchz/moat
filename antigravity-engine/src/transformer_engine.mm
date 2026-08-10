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
#include "transformer_engine.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <numeric>

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
// Constructor
// ============================================================================

MetalTransformerEngine::MetalTransformerEngine(const TransformerConfig& config)
    : weightsLoaded_(false), config_(config), allocatedBytes_(0) {
    
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        std::cerr << "[MetalTransformerEngine] Metal not supported" << std::endl;
        return;
    }
    queue_ = [device_ newCommandQueue];
    
    // ---- Load Shader Libraries ----
    NSError* err = nil;
    
    // Try compiled metallib first, fall back to runtime compilation
    NSArray<NSString*>* gemmPaths = @[
        @"src/shaders/batched_gemm.metallib",
        @"antigravity-engine/src/shaders/batched_gemm.metallib"
    ];
    for (NSString* path in gemmPaths) {
        NSURL* url = [NSURL fileURLWithPath:path];
        gemmLib_ = [device_ newLibraryWithURL:url error:&err];
        if (gemmLib_) break;
    }
    
    // Compile transformer_ops from source if metallib not available
    NSArray<NSString*>* opsPaths = @[
        @"src/shaders/transformer_ops.metallib",
        @"antigravity-engine/src/shaders/transformer_ops.metallib",
        @"src/shaders/transformer_ops.metal",
        @"antigravity-engine/src/shaders/transformer_ops.metal"
    ];
    for (NSString* path in opsPaths) {
        if ([path hasSuffix:@".metallib"]) {
            NSURL* url = [NSURL fileURLWithPath:path];
            opsLib_ = [device_ newLibraryWithURL:url error:&err];
        } else {
            // Compile from source
            NSString* source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
            if (source) {
                MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
                opsLib_ = [device_ newLibraryWithSource:source options:opts error:&err];
            }
        }
        if (opsLib_) break;
    }
    
    // Fall back: compile GEMM from inline source
    if (!gemmLib_) {
        NSString* gemmSrc = @"#include <metal_stdlib>\nusing namespace metal;\nkernel void batched_gemm_simdgroup(device const half* activations [[buffer(0)]], device const half* weights [[buffer(1)]], device half* output [[buffer(2)]], constant uint& N_batch [[buffer(3)]], constant uint& K_dim [[buffer(4)]], constant uint& M_dim [[buffer(5)]], uint2 group_id [[threadgroup_position_in_grid]]) { uint row_start = group_id.y * 8; uint col_start = group_id.x * 8; if (row_start >= N_batch || col_start >= M_dim) return; simdgroup_matrix<half, 8, 8> acc_matrix = simdgroup_matrix<half, 8, 8>(0.0h); for (uint k = 0; k < K_dim; k += 8) { simdgroup_matrix<half, 8, 8> a_tile; simdgroup_matrix<half, 8, 8> b_tile; simdgroup_load(a_tile, activations + row_start * K_dim + k, K_dim); simdgroup_load(b_tile, weights + k * M_dim + col_start, M_dim); simdgroup_multiply_accumulate(acc_matrix, a_tile, b_tile, acc_matrix); } simdgroup_store(acc_matrix, output + row_start * M_dim + col_start, M_dim); }\n";
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        gemmLib_ = [device_ newLibraryWithSource:gemmSrc options:opts error:&err];
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
    rmsnormPipeline_   = makePipeline(opsLib_, @"rmsnorm_kernel");
    ropePipeline_      = makePipeline(opsLib_, @"rope_kernel");
    attnScoresPipeline_ = makePipeline(opsLib_, @"gqa_attention_scores_kernel");
    softmaxPipeline_   = makePipeline(opsLib_, @"softmax_kernel");
    attnValuePipeline_ = makePipeline(opsLib_, @"attention_value_kernel");
    siluMulPipeline_   = makePipeline(opsLib_, @"silu_elementwise_mul_kernel");
    residualPipeline_  = makePipeline(opsLib_, @"residual_add_kernel");
    embedPipeline_     = makePipeline(opsLib_, @"embedding_lookup_kernel");
    kvAppendPipeline_  = makePipeline(opsLib_, @"kv_cache_append_kernel");
    
    // ---- Allocate KV Caches ----
    size_t kv_size = config_.n_kv_heads * config_.max_seq_len * config_.head_dim * sizeof(uint16_t);
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
    // We need several intermediate buffers for the forward pass per-channel
    size_t hidden_bytes = config_.hidden_dim * sizeof(uint16_t);
    size_t inter_bytes  = config_.intermediate_dim * sizeof(uint16_t);
    size_t q_bytes      = config_.n_heads * config_.head_dim * sizeof(uint16_t);     // = hidden_dim
    size_t kv_proj_bytes = config_.n_kv_heads * config_.head_dim * sizeof(uint16_t); // = 256*2
    size_t logits_bytes = config_.vocab_size * sizeof(uint16_t);
    size_t attn_scores_bytes = config_.n_heads * config_.max_seq_len * sizeof(uint16_t);
    size_t attn_out_bytes = config_.n_heads * config_.head_dim * sizeof(uint16_t);   // = hidden_dim

    // Per-channel scratch (we process one channel at a time during decode)
    scratch1_ = [device_ newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];  // norm output
    scratch2_ = [device_ newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];  // attn output / mlp temp
    scratch3_ = [device_ newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];  // residual store
    
    // Additional scratch for attention and MLP
    // Q projection output: [n_heads * head_dim] = [2048]
    id<MTLBuffer> scratchQ_ = [device_ newBufferWithLength:q_bytes options:MTLResourceStorageModeShared];
    // K projection output: [n_kv_heads * head_dim] = [256]
    id<MTLBuffer> scratchK_ = [device_ newBufferWithLength:kv_proj_bytes options:MTLResourceStorageModeShared];
    // V projection output
    id<MTLBuffer> scratchV_ = [device_ newBufferWithLength:kv_proj_bytes options:MTLResourceStorageModeShared];
    // Attention scores: [n_heads * max_seq_len]
    scratchAttn_ = [device_ newBufferWithLength:attn_scores_bytes options:MTLResourceStorageModeShared];
    // Logits: [vocab_size]
    scratchLogits_ = [device_ newBufferWithLength:logits_bytes options:MTLResourceStorageModeShared];
    
    // Gate/Up MLP scratch
    id<MTLBuffer> scratchGate_ = [device_ newBufferWithLength:inter_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> scratchUp_ = [device_ newBufferWithLength:inter_bytes options:MTLResourceStorageModeShared];
    
    allocatedBytes_ += hidden_bytes * 3 + q_bytes + kv_proj_bytes * 2 + attn_scores_bytes + logits_bytes + inter_bytes * 2;
    
    (void)attn_out_bytes;
    (void)scratchQ_;
    (void)scratchK_;
    (void)scratchV_;
    (void)scratchGate_;
    (void)scratchUp_;
    
    // Store extra scratch pointers as ivars via a simple approach
    // We'll access them through the existing scratch buffers or add to header later
    // For now, store gate/up/q/k/v in a member vector
    
    std::cout << "[MetalTransformerEngine] Initialized with " 
              << (allocatedBytes_ / (1024*1024)) << " MB allocated ("
              << config_.n_channels << " channels, " << config_.n_layers << " layers)" 
              << std::endl;
}

MetalTransformerEngine::~MetalTransformerEngine() {
    // ARC handles all ObjC object cleanup
}


// ============================================================================
// Safetensors Parser & Weight Loader
// ============================================================================

bool MetalTransformerEngine::parseSafetensors(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[loadWeights] Cannot open: " << path << std::endl;
        return false;
    }
    
    // Read header length (8-byte LE uint64)
    uint64_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), 8);
    if (header_len > 100 * 1024 * 1024) {  // sanity: max 100MB header
        std::cerr << "[loadWeights] Header too large: " << header_len << std::endl;
        return false;
    }
    
    // Read JSON header
    std::string header_json(header_len, '\0');
    file.read(&header_json[0], header_len);
    
    uint64_t data_start = 8 + header_len;
    
    // Simple JSON parser for safetensors format
    // Format: {"tensor_name": {"dtype": "BF16", "shape": [dim0, dim1], "data_offsets": [start, end]}, ...}
    
    struct TensorInfo {
        std::string dtype;
        std::vector<int64_t> shape;
        uint64_t offset_start;
        uint64_t offset_end;
    };
    
    // Minimal JSON parser — extract tensor name, dtype, shape, data_offsets
    std::map<std::string, TensorInfo> tensors;
    
    // Find each key-value pair
    size_t pos = 0;
    while (pos < header_json.size()) {
        // Find key string
        size_t key_start = header_json.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = header_json.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = header_json.substr(key_start + 1, key_end - key_start - 1);
        
        // Skip "__metadata__"
        if (key == "__metadata__") {
            // Skip until next top-level key
            pos = header_json.find('}', key_end);
            if (pos != std::string::npos) pos++;
            continue;
        }
        
        // Find the value object
        size_t val_start = header_json.find('{', key_end);
        if (val_start == std::string::npos) break;
        
        // Find matching closing brace
        int depth = 1;
        size_t val_end = val_start + 1;
        while (val_end < header_json.size() && depth > 0) {
            if (header_json[val_end] == '{') depth++;
            else if (header_json[val_end] == '}') depth--;
            val_end++;
        }
        
        std::string val_str = header_json.substr(val_start, val_end - val_start);
        
        TensorInfo info;
        
        // Extract dtype
        size_t dtype_pos = val_str.find("\"dtype\"");
        if (dtype_pos != std::string::npos) {
            size_t ds = val_str.find('"', dtype_pos + 7);
            if (ds != std::string::npos) {
                size_t de = val_str.find('"', ds + 1);
                if (de != std::string::npos) {
                    info.dtype = val_str.substr(ds + 1, de - ds - 1);
                }
            }
        }
        
        // Extract shape
        size_t shape_pos = val_str.find("\"shape\"");
        if (shape_pos != std::string::npos) {
            size_t arr_s = val_str.find('[', shape_pos);
            size_t arr_e = val_str.find(']', arr_s);
            if (arr_s != std::string::npos && arr_e != std::string::npos) {
                std::string shape_str = val_str.substr(arr_s + 1, arr_e - arr_s - 1);
                // Parse comma-separated integers
                std::istringstream ss(shape_str);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    token.erase(std::remove(token.begin(), token.end(), ' '), token.end());
                    if (!token.empty()) info.shape.push_back(std::stoll(token));
                }
            }
        }
        
        // Extract data_offsets
        size_t off_pos = val_str.find("\"data_offsets\"");
        if (off_pos != std::string::npos) {
            size_t arr_s = val_str.find('[', off_pos);
            size_t arr_e = val_str.find(']', arr_s);
            if (arr_s != std::string::npos && arr_e != std::string::npos) {
                std::string off_str = val_str.substr(arr_s + 1, arr_e - arr_s - 1);
                size_t comma = off_str.find(',');
                if (comma != std::string::npos) {
                    std::string s1 = off_str.substr(0, comma);
                    std::string s2 = off_str.substr(comma + 1);
                    s1.erase(std::remove(s1.begin(), s1.end(), ' '), s1.end());
                    s2.erase(std::remove(s2.begin(), s2.end(), ' '), s2.end());
                    info.offset_start = std::stoull(s1);
                    info.offset_end = std::stoull(s2);
                }
            }
        }
        
        tensors[key] = info;
        pos = val_end;
    }
    
    std::cout << "[loadWeights] Parsed " << tensors.size() << " tensors from Safetensors" << std::endl;
    
    // Memory-map the raw data section
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> file_data(file_size);
    file.read(file_data.data(), file_size);
    file.close();
    
    const char* raw_data = file_data.data() + data_start;
    
    // Helper: load a tensor into a Metal buffer as FP16
    auto loadTensor = [&](const std::string& name, bool transpose_2d = false) -> id<MTLBuffer> {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            // Try with "model." prefix
            it = tensors.find("model." + name);
            if (it == tensors.end()) {
                std::cerr << "[loadWeights] Missing tensor: " << name << std::endl;
                return nil;
            }
        }
        
        const TensorInfo& info = it->second;
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
        return buf;
    };
    
    // ---- Load Embedding & Output Head ----
    embedWeights_ = loadTensor("embed_tokens.weight", false);
    finalNorm_ = loadTensor("norm.weight", false);
    lmHead_ = loadTensor("lm_head.weight", true);
    
    if (!embedWeights_ || !finalNorm_ || !lmHead_) {
        std::cerr << "[loadWeights] Failed to load embedding/norm/lm_head" << std::endl;
        return false;
    }
    
    // ---- Load Layer Weights ----
    layerWeights_.resize(config_.n_layers);
    for (int i = 0; i < config_.n_layers; i++) {
        std::string prefix = "layers." + std::to_string(i) + ".";
        
        layerWeights_[i].input_norm  = loadTensor(prefix + "input_layernorm.weight", false);
        layerWeights_[i].q_proj      = loadTensor(prefix + "self_attn.q_proj.weight", true);
        layerWeights_[i].k_proj      = loadTensor(prefix + "self_attn.k_proj.weight", true);
        layerWeights_[i].v_proj      = loadTensor(prefix + "self_attn.v_proj.weight", true);
        layerWeights_[i].o_proj      = loadTensor(prefix + "self_attn.o_proj.weight", true);
        layerWeights_[i].post_attn_norm = loadTensor(prefix + "post_attention_layernorm.weight", false);
        layerWeights_[i].gate_proj   = loadTensor(prefix + "mlp.gate_proj.weight", true);
        layerWeights_[i].up_proj     = loadTensor(prefix + "mlp.up_proj.weight", true);
        layerWeights_[i].down_proj   = loadTensor(prefix + "mlp.down_proj.weight", true);
        
        // Validate all loaded
        if (!layerWeights_[i].input_norm || !layerWeights_[i].q_proj || !layerWeights_[i].k_proj ||
            !layerWeights_[i].v_proj || !layerWeights_[i].o_proj || !layerWeights_[i].post_attn_norm ||
            !layerWeights_[i].gate_proj || !layerWeights_[i].up_proj || !layerWeights_[i].down_proj) {
            std::cerr << "[loadWeights] Failed to load layer " << i << std::endl;
            return false;
        }
    }
    
    weightsLoaded_ = true;
    std::cout << "[loadWeights] All weights loaded: " << (allocatedBytes_ / (1024*1024)) << " MB total" << std::endl;
    return true;
}

bool MetalTransformerEngine::loadWeights(const std::string& safetensors_path) {
    return parseSafetensors(safetensors_path);
}


// ============================================================================
// Metal Dispatch Helpers
// ============================================================================

void MetalTransformerEngine::dispatchGEMM(
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
    uint32_t M, uint32_t K, uint32_t N
) {
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
    MTLSize tg = MTLSizeMake(1, 1, 1);
    MTLSize threads = MTLSizeMake(threadsPerTG, 1, 1);
    
    // batch threadgroups
    tg = MTLSizeMake(batch, 1, 1);
    [enc dispatchThreadgroups:tg threadsPerThreadgroup:threads];
}

void MetalTransformerEngine::dispatchRoPE(
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> q, id<MTLBuffer> k,
    uint32_t start_pos, uint32_t batch
) {
    if (!ropePipeline_) return;
    
    [enc setComputePipelineState:ropePipeline_];
    [enc setBuffer:q offset:0 atIndex:0];
    [enc setBuffer:k offset:0 atIndex:1];
    [enc setBuffer:ropeFreqsCos_ offset:0 atIndex:2];
    [enc setBuffer:ropeFreqsSin_ offset:0 atIndex:3];
    
    uint32_t seq_len = 1;  // decode mode: 1 token at a time
    uint32_t n_heads = config_.n_heads;
    uint32_t n_kv_heads = config_.n_kv_heads;
    uint32_t head_dim = config_.head_dim;
    uint32_t spos = start_pos;
    
    [enc setBytes:&seq_len length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&n_heads length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&n_kv_heads length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&spos length:sizeof(uint32_t) atIndex:8];
    
    // Grid: (batch * seq_len, max(n_heads, n_kv_heads), head_dim / 2)
    uint32_t max_heads = std::max(n_heads, n_kv_heads);
    MTLSize grid = MTLSizeMake(batch * seq_len, max_heads, head_dim / 2);
    MTLSize tg = MTLSizeMake(1, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
}


// ============================================================================
// Forward Layer — Full Single-Token Decode Through One Transformer Layer
// ============================================================================

void MetalTransformerEngine::forwardLayer(
    id<MTLComputeCommandEncoder> enc,
    int layer_idx,
    id<MTLBuffer> input,    // [1, hidden_dim] — current hidden state
    id<MTLBuffer> output,   // [1, hidden_dim] — output hidden state
    int channel,
    uint32_t seq_pos        // current position in the sequence (for KV cache write)
) {
    const auto& lw = layerWeights_[layer_idx];
    uint32_t H = config_.hidden_dim;         // 2048
    uint32_t I = config_.intermediate_dim;   // 5632
    uint32_t KV_DIM = config_.n_kv_heads * config_.head_dim;  // 4 * 64 = 256
    
    // scratch1_ = RMSNorm(input, input_norm)
    dispatchRMSNorm(enc, input, lw.input_norm, scratch1_, 1, H);
    
    // ---- Attention Block ----
    // Q = scratch1_ @ q_proj.T → [1, 2048] × [2048, 2048] → [1, 2048]
    // Note: Safetensors weights are stored as [out_dim, in_dim], so q_proj is [2048, 2048]
    // For GEMM: A=[1, H] × B=[H, H] → C=[1, H]
    // But our GEMM kernel does C = A × B row-major, and weights are [out, in] so B^T = [in, out]
    // We need to transpose: GEMM(scratch1_[1,H], q_proj_T[H,H], Q[1,H])
    // Since q_proj is stored as [H,H] and is square, we can use it directly with swapped dims
    dispatchGEMM(enc, scratch1_, lw.q_proj, scratch2_, 1, H, H);    // Q → scratch2_
    dispatchGEMM(enc, scratch1_, lw.k_proj, scratch3_, 1, H, KV_DIM);  // K → scratch3_
    
    // Allocate temp V buffer from one part of scratchAttn_
    // V projection: [1, H] × [H, KV_DIM] → [1, KV_DIM]
    // We need a separate buffer for V — use the end of scratchAttn_ temporarily
    id<MTLBuffer> v_temp = scratchAttn_;  // Reuse: V is small (256 elements = 512 bytes)
    dispatchGEMM(enc, scratch1_, lw.v_proj, v_temp, 1, H, KV_DIM);   // V → v_temp
    
    // Apply RoPE to Q and K
    dispatchRoPE(enc, scratch2_, scratch3_, seq_pos, 1);
    
    // Append K, V to KV cache at position seq_pos
    if (kvAppendPipeline_) {
        [enc setComputePipelineState:kvAppendPipeline_];
        
        // Append K
        [enc setBuffer:scratch3_ offset:0 atIndex:0];  // new K [1, n_kv_heads, head_dim]
        [enc setBuffer:kvCaches_[layer_idx][channel].k_cache offset:0 atIndex:1];
        uint32_t nkv = config_.n_kv_heads;
        uint32_t maxseq = config_.max_seq_len;
        uint32_t hdim = config_.head_dim;
        uint32_t wpos = seq_pos;
        [enc setBytes:&nkv length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&maxseq length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&hdim length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&wpos length:sizeof(uint32_t) atIndex:5];
        MTLSize grid = MTLSizeMake(1, nkv, hdim);
        MTLSize tg = MTLSizeMake(1, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        
        // Append V
        [enc setBuffer:v_temp offset:0 atIndex:0];
        [enc setBuffer:kvCaches_[layer_idx][channel].v_cache offset:0 atIndex:1];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }
    
    // Compute attention scores: Q[1, n_heads, head_dim] @ K_cache[n_kv_heads, seq_pos+1, head_dim]^T
    uint32_t cur_seq_len = seq_pos + 1;
    if (attnScoresPipeline_) {
        [enc setComputePipelineState:attnScoresPipeline_];
        [enc setBuffer:scratch2_ offset:0 atIndex:0];   // Q [1, n_heads * head_dim]
        [enc setBuffer:kvCaches_[layer_idx][channel].k_cache offset:0 atIndex:1];  // K cache
        [enc setBuffer:scratchAttn_ offset:0 atIndex:2];  // scores output [n_heads, seq_len]
        uint32_t nh = config_.n_heads, nkv = config_.n_kv_heads, hd = config_.head_dim;
        uint32_t sl = cur_seq_len;
        [enc setBytes:&nh length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&nkv length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&hd length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&sl length:sizeof(uint32_t) atIndex:6];
        MTLSize grid = MTLSizeMake(1, nh, sl);
        MTLSize tg = MTLSizeMake(1, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }
    
    // Softmax over attention scores
    if (softmaxPipeline_) {
        [enc setComputePipelineState:softmaxPipeline_];
        [enc setBuffer:scratchAttn_ offset:0 atIndex:0];  // scores (in-place scores)
        [enc setBuffer:scratchAttn_ offset:0 atIndex:1];  // probs (overwrite in-place)
        uint32_t sl = cur_seq_len;
        [enc setBytes:&sl length:sizeof(uint32_t) atIndex:2];
        uint32_t threadsPerTG = std::min(cur_seq_len, (uint32_t)256);
        MTLSize tg_count = MTLSizeMake(config_.n_heads, 1, 1);  // one TG per head
        MTLSize tg_size = MTLSizeMake(threadsPerTG, 1, 1);
        [enc dispatchThreadgroups:tg_count threadsPerThreadgroup:tg_size];
    }
    
    // Compute attention output: probs[n_heads, seq_len] @ V_cache[n_kv_heads, seq_len, head_dim]
    if (attnValuePipeline_) {
        [enc setComputePipelineState:attnValuePipeline_];
        [enc setBuffer:scratchAttn_ offset:0 atIndex:0];  // probs
        [enc setBuffer:kvCaches_[layer_idx][channel].v_cache offset:0 atIndex:1];
        [enc setBuffer:scratch2_ offset:0 atIndex:2];  // output [n_heads * head_dim]
        uint32_t nh = config_.n_heads, nkv = config_.n_kv_heads;
        uint32_t sl = cur_seq_len, hd = config_.head_dim;
        [enc setBytes:&nh length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&nkv length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&sl length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&hd length:sizeof(uint32_t) atIndex:6];
        MTLSize grid = MTLSizeMake(1, nh, hd);
        MTLSize tg = MTLSizeMake(1, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }
    
    // O projection: [1, 2048] → [1, 2048]
    dispatchGEMM(enc, scratch2_, lw.o_proj, scratch1_, 1, H, H);
    
    // Residual: output = input + attn_output (scratch1_)
    if (residualPipeline_) {
        [enc setComputePipelineState:residualPipeline_];
        [enc setBuffer:scratch1_ offset:0 atIndex:0];
        [enc setBuffer:input offset:0 atIndex:1];
        [enc setBuffer:scratch3_ offset:0 atIndex:2];  // scratch3_ = residual sum
        uint32_t size = H;
        [enc setBytes:&size length:sizeof(uint32_t) atIndex:3];
        MTLSize grid = MTLSizeMake(H, 1, 1);
        MTLSize tg = MTLSizeMake(1, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }
    
    // ---- MLP Block ----
    // RMSNorm: scratch1_ = RMSNorm(scratch3_, post_attn_norm)
    dispatchRMSNorm(enc, scratch3_, lw.post_attn_norm, scratch1_, 1, H);
    
    // Gate projection: [1, H] → [1, I]
    dispatchGEMM(enc, scratch1_, lw.gate_proj, scratch2_, 1, H, I);  // gate → scratch2_
    
    // Up projection: [1, H] → [1, I]
    // Need a temp buffer for up — reuse scratchAttn_ which is large enough
    id<MTLBuffer> up_buf = scratchAttn_;  // attn scratch no longer needed
    dispatchGEMM(enc, scratch1_, lw.up_proj, up_buf, 1, H, I);    // up → up_buf
    
    // SiLU(gate) * up → scratch2_
    if (siluMulPipeline_) {
        [enc setComputePipelineState:siluMulPipeline_];
        [enc setBuffer:scratch2_ offset:0 atIndex:0];  // gate
        [enc setBuffer:up_buf offset:0 atIndex:1];      // up
        [enc setBuffer:scratch2_ offset:0 atIndex:2];   // output (in-place over gate)
        uint32_t size = I;
        [enc setBytes:&size length:sizeof(uint32_t) atIndex:3];
        MTLSize grid = MTLSizeMake(I, 1, 1);
        MTLSize tg = MTLSizeMake(1, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }
    
    // Down projection: [1, I] → [1, H]
    dispatchGEMM(enc, scratch2_, lw.down_proj, scratch1_, 1, I, H);
    
    // Final residual: output = scratch3_ + scratch1_
    if (residualPipeline_) {
        [enc setComputePipelineState:residualPipeline_];
        [enc setBuffer:scratch1_ offset:0 atIndex:0];
        [enc setBuffer:scratch3_ offset:0 atIndex:1];
        [enc setBuffer:output offset:0 atIndex:2];
        uint32_t size = H;
        [enc setBytes:&size length:sizeof(uint32_t) atIndex:3];
        MTLSize grid = MTLSizeMake(H, 1, 1);
        MTLSize tg = MTLSizeMake(1, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }
}


// ============================================================================
// CPU-Side Token Sampling (Top-P Nucleus Sampling)
// ============================================================================

int32_t MetalTransformerEngine::sampleToken(
    const _Float16* logits, int vocab_size,
    float temperature, float top_p, std::mt19937& rng
) {
    std::vector<float> probs(vocab_size);
    float max_logit = -1e9f;
    
    // Temperature scaling
    float inv_temp = 1.0f / std::max(temperature, 1e-6f);
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = (float)logits[i] * inv_temp;
        if (probs[i] > max_logit) max_logit = probs[i];
    }
    
    // Stable softmax
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf(probs[i] - max_logit);
        sum_exp += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum_exp;
    }
    
    // Top-P nucleus sampling
    if (top_p < 1.0f && top_p > 0.0f) {
        // Sort by probability descending
        std::vector<int> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return probs[a] > probs[b];
        });
        
        float cumsum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[indices[i]];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // Zero out tokens below cutoff
        for (int i = cutoff; i < vocab_size; i++) {
            probs[indices[i]] = 0.0f;
        }
        
        // Re-normalize
        sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++) sum_exp += probs[i];
        for (int i = 0; i < vocab_size; i++) probs[i] /= sum_exp;
    }
    
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}


// ============================================================================
// Full Autoregressive Generation — The Core Decode Loop
// ============================================================================

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
    
    const uint32_t H = config_.hidden_dim;
    const int EOS_TOKEN = 2;
    
    std::vector<std::mt19937> channel_rngs(config_.n_channels);
    for (int c = 0; c < config_.n_channels; c++) {
        channel_rngs[c].seed(1337 + c * 10007);
    }
    
    // Track active channels (not yet hit EOS)
    std::vector<bool> channel_active(config_.n_channels, true);
    
    // Allocate per-channel hidden state buffers  
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
    
    // ---- Prefill: Process prompt tokens one at a time through all layers ----
    // (For simplicity, process each prompt token sequentially through the full model
    //  to build up the KV cache. A production implementation would process all at once.)
    
    for (int t = 0; t < prompt_len; t++) {
        for (int c = 0; c < config_.n_channels; c++) {
            id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            
            // Embedding lookup for this token
            if (embedPipeline_) {
                [enc setComputePipelineState:embedPipeline_];
                // Write token ID to a small temp buffer
                uint32_t tok = (uint32_t)prompt_tokens[t];
                id<MTLBuffer> tokBuf = [device_ newBufferWithBytes:&tok length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                [enc setBuffer:tokBuf offset:0 atIndex:0];
                [enc setBuffer:embedWeights_ offset:0 atIndex:1];
                [enc setBuffer:hidden_bufs[c] offset:0 atIndex:2];
                uint32_t hdim = H;
                [enc setBytes:&hdim length:sizeof(uint32_t) atIndex:3];
                MTLSize grid = MTLSizeMake(1, H, 1);
                MTLSize tg = MTLSizeMake(1, 1, 1);
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            }
            
            // Forward through all 22 layers
            for (int l = 0; l < config_.n_layers; l++) {
                id<MTLBuffer> in_buf  = (l % 2 == 0) ? hidden_bufs[c] : hidden_bufs2[c];
                id<MTLBuffer> out_buf = (l % 2 == 0) ? hidden_bufs2[c] : hidden_bufs[c];
                forwardLayer(enc, l, in_buf, out_buf, c, t);
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
            
            // Determine current token to process
            int32_t cur_token;
            if (step == 0) {
                cur_token = prompt_tokens[prompt_len - 1];  // Last prompt token
            } else {
                cur_token = result.channel_tokens[c].back();
            }
            
            uint32_t seq_pos = prompt_len + step;
            
            id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            
            // Embedding lookup
            if (embedPipeline_) {
                [enc setComputePipelineState:embedPipeline_];
                uint32_t tok = (uint32_t)cur_token;
                id<MTLBuffer> tokBuf = [device_ newBufferWithBytes:&tok length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                [enc setBuffer:tokBuf offset:0 atIndex:0];
                [enc setBuffer:embedWeights_ offset:0 atIndex:1];
                [enc setBuffer:hidden_bufs[c] offset:0 atIndex:2];
                uint32_t hdim = H;
                [enc setBytes:&hdim length:sizeof(uint32_t) atIndex:3];
                MTLSize grid = MTLSizeMake(1, H, 1);
                MTLSize tg = MTLSizeMake(1, 1, 1);
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            }
            
            // Forward through all layers
            for (int l = 0; l < config_.n_layers; l++) {
                id<MTLBuffer> in_buf  = (l % 2 == 0) ? hidden_bufs[c] : hidden_bufs2[c];
                id<MTLBuffer> out_buf = (l % 2 == 0) ? hidden_bufs2[c] : hidden_bufs[c];
                forwardLayer(enc, l, in_buf, out_buf, c, seq_pos);
            }
            
            // Final RMSNorm
            id<MTLBuffer> final_hidden = (config_.n_layers % 2 == 0) ? hidden_bufs[c] : hidden_bufs2[c];
            dispatchRMSNorm(enc, final_hidden, finalNorm_, scratch1_, 1, H);
            
            // LM Head projection: [1, H] × [H, V] → [1, V]
            // lm_head weight is [V, H], so we transpose: GEMM(hidden[1,H], lm_head[H,V], logits[1,V])
            dispatchGEMM(enc, scratch1_, lmHead_, scratchLogits_, 1, H, config_.vocab_size);
            
            [enc endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            
            // CPU-side sampling from logits
            const _Float16* logits = (const _Float16*)[scratchLogits_ contents];
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
            
            // Check EOS
            if (next_token == EOS_TOKEN) {
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
        
        // Check if all channels finished
        bool all_done = true;
        for (int c = 0; c < config_.n_channels; c++) {
            if (channel_active[c]) { all_done = false; break; }
        }
        if (all_done) break;
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.tpot_ms = (decode_steps > 0) ? (sum_decode_ms / decode_steps) : 0;
    
    // Determine best channel by logprob
    result.best_channel = 0;
    result.best_score = result.channel_logprobs[0];
    for (int c = 1; c < config_.n_channels; c++) {
        if (result.channel_logprobs[c] > result.best_score) {
            result.best_score = result.channel_logprobs[c];
            result.best_channel = c;
        }
    }
    
    std::cout << "[generate] Done: " << result.total_tokens << " tokens, "
              << "TTFT=" << result.ttft_ms << "ms, "
              << "TPOT=" << result.tpot_ms << "ms, "
              << "Total=" << result.total_ms << "ms" << std::endl;
    
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

    const uint32_t H = config_.hidden_dim;
    const int EOS_TOKEN = 2;

    std::vector<std::mt19937> channel_rngs(config_.n_channels);
    for (int c = 0; c < config_.n_channels; c++) {
        channel_rngs[c].seed(1337 + c * 10007);
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
                    id<MTLBuffer> tokBuf = [device_ newBufferWithBytes:&tok length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                    [enc setBuffer:tokBuf offset:0 atIndex:0];
                    [enc setBuffer:embedWeights_ offset:0 atIndex:1];
                    [enc setBuffer:hidden_bufs[c] offset:0 atIndex:2];
                    uint32_t hdim = H;
                    [enc setBytes:&hdim length:sizeof(uint32_t) atIndex:3];
                    MTLSize grid = MTLSizeMake(1, H, 1);
                    MTLSize tg = MTLSizeMake(1, 1, 1);
                    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
                }
            }

            // Forward through all 22 layers
            for (int l = 0; l < config_.n_layers; l++) {
                id<MTLBuffer> in_buf  = (l % 2 == 0) ? hidden_bufs[c] : hidden_bufs2[c];
                id<MTLBuffer> out_buf = (l % 2 == 0) ? hidden_bufs2[c] : hidden_bufs[c];
                forwardLayer(enc, l, in_buf, out_buf, c, t);
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
                id<MTLBuffer> tokBuf = [device_ newBufferWithBytes:&tok length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                [enc setBuffer:tokBuf offset:0 atIndex:0];
                [enc setBuffer:embedWeights_ offset:0 atIndex:1];
                [enc setBuffer:hidden_bufs[c] offset:0 atIndex:2];
                uint32_t hdim = H;
                [enc setBytes:&hdim length:sizeof(uint32_t) atIndex:3];
                MTLSize grid = MTLSizeMake(1, H, 1);
                MTLSize tg = MTLSizeMake(1, 1, 1);
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            }

            for (int l = 0; l < config_.n_layers; l++) {
                id<MTLBuffer> in_buf  = (l % 2 == 0) ? hidden_bufs[c] : hidden_bufs2[c];
                id<MTLBuffer> out_buf = (l % 2 == 0) ? hidden_bufs2[c] : hidden_bufs[c];
                forwardLayer(enc, l, in_buf, out_buf, c, seq_pos);
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

            if (next_token == EOS_TOKEN) {
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

    std::cout << "[generateMultimodal] Done: " << result.total_tokens << " tokens, "
              << "Patches=" << n_patches << ", "
              << "TTFT=" << result.ttft_ms << "ms, "
              << "TPOT=" << result.tpot_ms << "ms, "
              << "Total=" << result.total_ms << "ms" << std::endl;

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
