// Package hf2browser carries the files the binary needs at runtime.
//
// Everything the converter runs — the Python export pipeline and the Node CPU
// verifier — is compiled into the executable, so a downloaded binary is the
// whole product. On first run it unpacks them into a work directory (see
// internal/workspace) and drives them from there.
package hf2browser

import "embed"

// PyTools is the ONNX export + quantization pipeline. `all:` is required
// because the package files start with an underscore (`__init__.py`), which
// go:embed skips by default.
//
//go:embed all:pytools
var PyTools embed.FS

// Verify is the Node CPU verifier. Only the sources — its dependencies are
// installed with npm into the work directory on first use.
//
//go:embed verify/verify.mjs verify/package.json
var Verify embed.FS
