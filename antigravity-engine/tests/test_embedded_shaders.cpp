/*
 * Verifies that src/shader_sources.h carries every .metal source byte for byte.
 *
 * The engine compiles these strings when no shader file is found beside the
 * executable, which on a device is always: scripts/build_xcframework.sh packages
 * only the static library, and every path MetalTransformerEngine tries is relative
 * to the process working directory. If a .metal file is edited and the header is
 * not regenerated, the shipped binary keeps running the old kernels with nothing
 * to indicate it — so this is the check that the two cannot drift.
 *
 * No Metal, no GPU: this compares text.
 *
 * Build and run from antigravity-engine/:
 *   c++ -std=c++17 -Wall -Wextra -Isrc tests/test_embedded_shaders.cpp \
 *       -o bin/test_embedded_shaders && ./bin/test_embedded_shaders
 */

#include "shader_sources.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("%s  %s\n", ok ? "[PASS]" : "[FAIL]", what);
    if (!ok) failures++;
}

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "src/shaders";

    size_t n = 0;
    const antigravity::shaders::EmbeddedShader* list = antigravity::shaders::all(&n);
    check(n > 0, "at least one shader is embedded");

    for (size_t i = 0; i < n; i++) {
        const std::string path = dir + "/" + list[i].name + ".metal";
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            printf("[FAIL]  cannot open %s\n", path.c_str());
            failures++;
            continue;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string on_disk = ss.str();
        const std::string embedded = list[i].source;

        char label[256];
        std::snprintf(label, sizeof(label), "%s.metal matches its embedded copy (%zu bytes)",
                      list[i].name, on_disk.size());
        if (on_disk != embedded) {
            size_t at = 0;
            while (at < on_disk.size() && at < embedded.size() && on_disk[at] == embedded[at]) at++;
            printf("        first difference at byte %zu (on disk %zu bytes, embedded %zu bytes)\n",
                   at, on_disk.size(), embedded.size());
            printf("        regenerate with: python3 scripts/embed_shaders.py\n");
        }
        check(on_disk == embedded, label);
    }

    // Every library the engine asks for by stem must actually resolve.
    for (const char* stem : { "batched_gemm", "transformer_ops", "deltanet_forward", "moe_gemm" }) {
        char label[128];
        std::snprintf(label, sizeof(label), "find(\"%s\") resolves", stem);
        check(antigravity::shaders::find(stem) != nullptr, label);
    }
    check(antigravity::shaders::find("no_such_shader") == nullptr,
          "find() returns null for an unknown name");
    check(antigravity::shaders::find(nullptr) == nullptr, "find(nullptr) is null, not a crash");

    // A prefix of a real name must not match it: find() compares whole strings.
    check(antigravity::shaders::find("batched") == nullptr,
          "find() does not match on a prefix");

    printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED" : "THERE WERE FAILURES");
    return failures == 0 ? 0 : 1;
}
