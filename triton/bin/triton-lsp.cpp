#include "./RegisterTritonDialects.h"

#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerTritonDialects(registry);

  mlir::MLIRContext context(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
