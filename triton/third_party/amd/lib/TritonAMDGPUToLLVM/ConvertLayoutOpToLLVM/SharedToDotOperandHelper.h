#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_SHARED_TO_DOT_OPERAND_MATRIXCORE_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_SHARED_TO_DOT_OPERAND_MATRIXCORE_H

#include "Utility.h"

namespace mlir::triton::AMD {

// Get warpId inside block of warps.
Value getWarpIdInBlock(ConversionPatternRewriter &rewriter, Location loc,
                       Value warpId, const ArrayRef<unsigned int> &wpt,
                       int elemPerInstrNonK, int tensorSizeNonK, int nonKIdx,
                       const ArrayRef<unsigned int> &order);

bool isSwizzled(gpu::SharedEncodingAttr layout);

/**
 * @brief swizzling tensor element indexes according pattern encoded in
 * SharedEncodingAttr
 *
 * @param rewriter
 * @param loc
 * @param row row of target tensor element related to the start of smemObj
 * @param col col of target tensor element related to the start of smemObj
 * @param smemObj shared memory object, contains info about tensor in LDS
 * @param attr layout attribute, contains swizzling info
 * @return swizzled row, col indexes in tensor notation
 */
std::pair<mlir::Value, mlir::Value>
swizzleIndexes(ConversionPatternRewriter &rewriter, Location loc, Value row,
               Value col, SharedMemoryObject smemObj,
               gpu::SharedEncodingAttr attr);

Value computeOffset(ConversionPatternRewriter &rewriter, Location loc,
                    Value row, Value col, SharedMemoryObject smemObj,
                    gpu::SharedEncodingAttr srcLayout);

Value computeBasePtr(ConversionPatternRewriter &rewriter, Location loc,
                     const SharedMemoryObject &smemObj);

bool isKMajor(llvm::ArrayRef<unsigned> order, int opIdx);

using computeTensorElemMappingInBlockT =
    std::function<llvm::SmallVector<llvm::SmallVector<Value>>(
        ConversionPatternRewriter &, Location, const ArrayRef<int64_t> &, Value,
        Value, int, ArrayRef<int64_t>, ArrayRef<Value>, int, unsigned,
        unsigned)>;

llvm::SmallVector<Value> computeOffsetsAType(
    ConversionPatternRewriter &rewriter, Location loc,
    computeTensorElemMappingInBlockT fn, const ArrayRef<int64_t> &elemsPerInstr,
    Value warpId, Value laneId, int warpsPerBlock, int numOfElems,
    ArrayRef<int64_t> reps, SharedMemoryObject smemObj,
    gpu::SharedEncodingAttr srcLayout, unsigned nonKDim, unsigned kDim);

llvm::SmallVector<Value> computeOffsetsBType(
    ConversionPatternRewriter &rewriter, Location loc,
    computeTensorElemMappingInBlockT fn, const ArrayRef<int64_t> &elemsPerInstr,
    Value warpId, Value laneId, int warpsPerBlock, int numOfElems,
    ArrayRef<int64_t> reps, SharedMemoryObject smemObj,
    gpu::SharedEncodingAttr srcLayout, unsigned nonKDim, unsigned kDim);

} // namespace mlir::triton::AMD

#endif
