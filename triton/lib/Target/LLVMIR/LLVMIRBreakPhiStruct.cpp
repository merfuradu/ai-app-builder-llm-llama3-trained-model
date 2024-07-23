//===----------------------------------------------------------------------===//
/// Implements a trivial pass breaking up 1 level deep structure in phi nodes.
/// This handles the common case generated by Triton and allow better
/// optimizations down the compiler pipeline.
//===----------------------------------------------------------------------===//
#include "LLVMPasses.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

static bool processPhiStruct(PHINode *phiNode) {
  StructType *STy = dyn_cast<StructType>(phiNode->getType());
  if (!STy)
    return false;
  IRBuilder<> builder(phiNode);
  unsigned numOperands = phiNode->getNumIncomingValues();
  unsigned numScalarEl = STy->getNumElements();
  Value *newStruct = UndefValue::get(STy);
  builder.SetInsertPoint(phiNode->getParent()->getFirstNonPHI());
  llvm::IRBuilderBase::InsertPoint insertInsertPt = builder.saveIP();
  for (unsigned i = 0; i < numScalarEl; i++) {
    builder.SetInsertPoint(phiNode);
    PHINode *newPhiNode =
        builder.CreatePHI(STy->getElementType(i), numOperands);
    for (unsigned j = 0; j < numOperands; ++j) {
      Value *operand = phiNode->getIncomingValue(j);
      builder.SetInsertPoint(phiNode->getIncomingBlock(j)->getTerminator());
      newPhiNode->addIncoming(builder.CreateExtractValue(operand, i),
                              phiNode->getIncomingBlock(j));
    }
    builder.restoreIP(insertInsertPt);
    newStruct = builder.CreateInsertValue(newStruct, newPhiNode, i);
    insertInsertPt = builder.saveIP();
  }
  phiNode->replaceAllUsesWith(newStruct);
  return true;
}

static bool runOnFunction(Function &F) {
  bool Changed = false;
  SmallVector<PHINode *> PhiNodes;
  for (BasicBlock &BB : F) {
    for (Instruction &inst : BB) {
      if (PHINode *phiNode = dyn_cast<PHINode>(&inst)) {
        Changed |= processPhiStruct(phiNode);
        continue;
      }
      break;
    }
  }
  return Changed;
}

PreservedAnalyses BreakStructPhiNodesPass::run(Function &F,
                                               FunctionAnalysisManager &AM) {

  bool b = runOnFunction(F);
  return b ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
