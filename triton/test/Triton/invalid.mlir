// RUN: triton-opt --split-input-file %s --verify-diagnostics

tt.func @fn(%v: i32) {
  %b = tt.splat %v : i32 -> tensor<128xi32>
  // expected-error @+1 {{rank of source must be same as rank of result}}
  %c = tt.broadcast %b : tensor<128xi32> -> tensor<128x32xi32>
  tt.return
}

// -----

tt.func @fn(%v: i32) {
  %b = tt.splat %v : i32 -> tensor<2x32xi32>
  // expected-error @+1 {{Different dimensions at index 0 between source and result.  Broadcast requires the source dimension to be 1.}}
  %c = tt.broadcast %b : tensor<2x32xi32> -> tensor<128x32xi32>
  tt.return
}

// -----

tt.func public @fn(%arg0: tensor<128xf32>) {
    // expected-error @+1 {{packed_element}}
    %a = tt.elementwise_inline_asm ""
      {constraints = "=r,r", packed_element=3:i32, pure=true} %arg0 : tensor<128xf32> -> tensor<128xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<128xf32>, %arg1: tensor<64xf32>) {
    // expected-error @+1 {{same shape}}
    %a = tt.elementwise_inline_asm ""
      {constraints = "=r,r,r", packed_element=1:i32, pure=true}
      %arg0, %arg1: tensor<128xf32>, tensor<64xf32> -> tensor<128xf32>
    tt.return
}
// -----

tt.func public @reshape_different_num_elements(%arg0: tensor<32x128xf16>) {
    // expected-error @+1 {{number of src and dst elements of reshape must be the same}}
    %a = tt.reshape %arg0 {allow_reorder = false} : tensor<32x128xf16> -> tensor<64x32xf16>
    tt.return
}

// -----

// expected-note @+1 {{prior use}}
tt.func public @fn(%arg0: tensor<32xf32>, %arg1: tensor<33xf32>) {
    // expected-error @+1 {{expects different type}}
    %a = tt.join %arg0, %arg1 : tensor<32xf32> -> tensor<32x2xf32>
    tt.return
}

// -----

// expected-note @+1 {{prior use}}
tt.func public @fn(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf16>) {
    // expected-error @+1 {{expects different type}}
    %a = tt.join %arg0, %arg1 : tensor<32x32xf32> -> tensor<32x32x2xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) {
    // expected-error @+2 {{op failed to infer returned types}}
    // expected-error @+1 {{incompatible with return type}}
    %a = tt.join %arg0, %arg1 : tensor<32xf32> -> tensor<64xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) {
    // expected-error @+2 {{op failed to infer returned types}}
    // expected-error @+1 {{incompatible with return type}}
    %a = tt.join %arg0, %arg1 : tensor<32x32xf32> -> tensor<32x64xf32>
    tt.return
}

// -----

// This one is OK
tt.func public @fn(%arg0: tensor<f32>, %arg1: tensor<f32>) {
    %a = tt.join %arg0, %arg1 : tensor<f32> -> tensor<2xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: f32, %arg1: f32) {
    // expected-error @+1 {{kind of type}}
    %a = tt.join %arg0, %arg1 : f32 -> tensor<2xf32>
    tt.return
}

// -----

tt.func public @fn(%v: tensor<4x128xf64>) {
    // expected-error @+1 {{operand types and result types}}
    %a = "tt.reduce" (%v) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %add = arith.addf %arg0, %arg1 : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32}  : (tensor<4x128xf64>) -> tensor<128xf32>
    tt.return
}

// -----

tt.func public @fn(%v: tensor<4x128xf32>) {
    // expected-error @+1 {{requires the same shape}}
    %a = "tt.scan" (%v) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %add = arith.addf %arg0, %arg1 : f32
      tt.scan.return %add : f32
    }) {axis = 0 : i32, reverse = false}  : (tensor<4x128xf32>) -> tensor<128xf32>
    tt.return
}

// -----

tt.func public @fn(%v1: tensor<4x128xf32>, %v2: tensor<4x128xi64>) {
    // expected-error @+1 {{operand types and result types}}
    %a, %b = "tt.scan" (%v1, %v2) ({
    ^bb0(%arg0: f32, %arg1: i32, %arg2: f32, %arg3: i32):
      %add = arith.addf %arg0, %arg2 : f32
      tt.scan.return %add, %arg1 : f32, i32
    }) {axis = 0 : i32, reverse = false}  : (tensor<4x128xf32>, tensor<4x128xi64>) -> (tensor<4x128xi64>, tensor<4x128xf32>)
    tt.return
}

// -----

tt.func public @fn(%v1: tensor<4x128xf32>, %v2: tensor<4x128xi64>) {
    // expected-error @+1 {{operand types and result types}}
    %a, %b = "tt.reduce" (%v1, %v2) ({
    ^bb0(%arg0: f32, %arg1: i32, %arg2: f32, %arg3: i32):
      %add = arith.addf %arg0, %arg2 : f32
      tt.reduce.return %add, %arg1 : f32, i32
    }) {axis = 0 : i32}  : (tensor<4x128xf32>, tensor<4x128xi64>) -> (tensor<128xi64>, tensor<128xf32>)
    tt.return
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<32xf32, #blocked>) {
    // expected-error @+2 {{op failed to infer returned types}}
    // expected-error @+1 {{incompatible with return type}}
    %a = tt.join %arg0, %arg0 : tensor<32xf32, #blocked> -> tensor<32x2xf32>
    tt.return
}
}  // end module

// -----

// Bad order; should be [1,0]
#blocked  = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1,2], threadsPerWarp = [32,1], warpsPerCTA = [1,1], order = [0,1]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<32xf32, #blocked>) {
    // expected-error @+2 {{order}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a = tt.join %arg0, %arg0 : tensor<32xf32, #blocked> -> tensor<32x2xf32, #blocked1>
    tt.return
}
}  // end module

// -----

tt.func public @fn(%arg0: tensor<32xf32>) {
    // expected-error @+2 {{last dimension}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a, %b = tt.split %arg0 : tensor<32xf32> -> tensor<16xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<32x2xf32>) {
    // expected-error @+2 {{op inferred type}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a, %b = tt.split %arg0 : tensor<32x2xf32> -> tensor<32xf16>
    tt.return
}

// -----

tt.func public @fn(%arg0: f32) {
    // expected-error @+1 {{invalid kind of type}}
    %a, %b = tt.split %arg0 : f32 -> f16
    tt.return
}
// -----

tt.func public @fn(%arg0: tensor<2xf32>) {
    %a, %b = tt.split %arg0 : tensor<2xf32> -> tensor<f32> // OK
    tt.return
}

// -----

// Bad order; should start with 2.
#blocked  = #triton_gpu.blocked<{sizePerThread = [1,1,2], threadsPerWarp = [1,32,1], warpsPerCTA = [1,1,1], order = [1,2,0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1,1], threadsPerWarp = [1,32], warpsPerCTA = [1,1], order = [1,0]}>

module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<2x2x2xf32, #blocked>) {
    // expected-error @+2 {{last dimension}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a, %b = tt.split %arg0 : tensor<2x2x2xf32, #blocked> -> tensor<2x2xf32, #blocked1>
    tt.return
}
}  // end module

// -----

#blocked  = #triton_gpu.blocked<{sizePerThread = [1,1,2], threadsPerWarp = [1,32,1], warpsPerCTA = [1,1,1], order = [2,0,1]}>
// Bad order, should be [1,0].
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1,1], threadsPerWarp = [1,32], warpsPerCTA = [1,1], order = [1,0]}>

module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<2x2x2xf32, #blocked>) {
    // expected-error @+2 {{op inferred type}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a, %b = tt.split %arg0 : tensor<2x2x2xf32, #blocked> -> tensor<2x2xf32, #blocked1>
    tt.return
}
}  // end module

// -----

#blocked  = #triton_gpu.blocked<{sizePerThread = [1,1,2], threadsPerWarp = [1,32,1], warpsPerCTA = [1,1,1], order = [2,0,1]}>
// bad sizePerThread; should be [1,1].
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1,2], threadsPerWarp = [1,32], warpsPerCTA = [1,1], order = [0,1]}>

module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<2x2x2xf32, #blocked>) {
    // expected-error @+2 {{op inferred type}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a, %b = tt.split %arg0 : tensor<2x2x2xf32, #blocked> -> tensor<2x2xf32, #blocked1>
    tt.return
}
}  // end module

// -----

// Valid ops.
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32>) {
    %a = tt.trans %arg0 {order = array<i32: 0, 1, 2>} : tensor<16x32x64xf32> -> tensor<16x32x64xf32>
    %b = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32> -> tensor<32x16x64xf32>
    tt.return
}
}  // end module

// -----

// Valid op with blocked encoding.
#blocked  = #triton_gpu.blocked<{sizePerThread = [1,2,3,4], threadsPerWarp = [2,4,2,2], warpsPerCTA = [4,2,4,2], order = [3,2,1,0], CTAsPerCGA = [1,2,2,2], CTASplitNum = [1,2,4,8], CTAOrder = [3,2,1,0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [2,4,3,1], threadsPerWarp = [4,2,2,2], warpsPerCTA = [2,2,4,4], order = [1,2,0,3], CTAsPerCGA = [2,2,2,1], CTASplitNum = [2,8,4,1], CTAOrder = [1,2,0,3]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1,2,4], threadsPerWarp = [2,4,4], warpsPerCTA = [2,4,8], order = [0,1,2], CTAsPerCGA = [1,2,4], CTASplitNum = [1,2,4], CTAOrder = [0,1,2]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [2,1,4], threadsPerWarp = [4,2,4], warpsPerCTA = [4,2,8], order = [1,0,2], CTAsPerCGA = [2,1,4], CTASplitNum = [2,1,4], CTAOrder = [1,0,2]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 8 : i32, "triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<2x4x8x16xf32, #blocked>, %arg1: tensor<16x32x64xf32, #blocked2>) {
    %a = tt.trans %arg0 {order = array<i32: 1, 3, 2, 0>} : tensor<2x4x8x16xf32, #blocked> -> tensor<4x16x8x2xf32, #blocked1>
    %b = tt.trans %arg1 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32, #blocked2> -> tensor<32x16x64xf32, #blocked3>
    tt.return
}
}  // end module

// -----

// Valid op with shared encoding.
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [3, 2, 1, 0]}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 2, 0, 3]}>
#shared2 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 2], CTASplitNum = [2, 4], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared3 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [2, 1], CTASplitNum = [4, 2], CTAOrder = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 8 : i32, "triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: !tt.memdesc<2x4x8x16xf32, #shared>, %arg1: !tt.memdesc<16x32xf32, #shared2>) {
    %a = tt.trans %arg0 {order = array<i32: 1, 3, 2, 0>} : !tt.memdesc<2x4x8x16xf32, #shared> -> !tt.memdesc<4x16x8x2xf32, #shared1>
    %b = tt.trans %arg1 {order = array<i32: 1, 0>} : !tt.memdesc<16x32xf32, #shared2> -> !tt.memdesc<32x16xf32, #shared3>
    tt.return
}
}  // end module

// -----

// Invalid blocked encoding.
#blocked  = #triton_gpu.blocked<{sizePerThread = [1,2,4], threadsPerWarp = [2,4,4], warpsPerCTA = [2,4,8], order = [0,1,2], CTAsPerCGA = [1,2,4], CTASplitNum = [1,2,4], CTAOrder = [0,1,2]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1,2,4], threadsPerWarp = [4,2,4], warpsPerCTA = [4,2,8], order = [1,0,2], CTAsPerCGA = [2,1,4], CTASplitNum = [2,1,4], CTAOrder = [1,0,2]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 8 : i32, "triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32, #blocked>) {
    // expected-error @+1 {{type}}
    %a = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32, #blocked> -> tensor<32x16x64xf32, #blocked1>
    tt.return
}
}  // end module

// -----

// Invalid shared encoding.
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1, 2]}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 0, 1]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 8 : i32, "triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32, #shared>) {
    // expected-error @+1 {{type}}
    %a = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32, #shared> -> tensor<32x16x64xf32, #shared1>
    tt.return
}
}  // end module

// -----

module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32xf32>) {
    // expected-error @+1 {{order}}
    %a = tt.trans %arg0 {order = array<i32: 0>} : tensor<16x32xf32> -> tensor<32x16xf32>
    tt.return
}
}  // end module

// -----

module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32xf32>) {
    // expected-error @+1 {{order}}
    %a = tt.trans %arg0 {order = array<i32: 2, 1, 0>} : tensor<16x32xf32> -> tensor<32x16xf32>
    tt.return
}
}  // end module

// -----

module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32xf32>) {
    // expected-error @+1 {{order must be a permutation}}
    %a = tt.trans %arg0 {order = array<i32: 0, 0>} : tensor<16x32xf32> -> tensor<32x16xf32>
    tt.return
}
}  // end module

// -----

// Invalid tensor with shared encoding.
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1, 2]}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 0, 1]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 8 : i32, "triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32, #shared>) {
    // expected-error @+1 {{has an invalid layout: Shared layout is not allowed on tensor type.}}
    %a = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32, #shared> -> tensor<32x16x64xf32, #shared1>
    tt.return
}
}  // end module
