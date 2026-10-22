#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

def find_llama_quantize(tools_dir: Path) -> Path:
    candidates = [
        tools_dir / "llama.cpp" / "build" / "bin" / "llama-quantize",
        tools_dir / "llama.cpp" / "llama-quantize",
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    which_path = shutil.which("llama-quantize")
    if which_path:
        return Path(which_path)
    return None

def run_command(cmd: list[str], extra_lib_dir: Path = None, cwd: Path = None):
    print(f"\n[EXEC] {' '.join(cmd)}")
    env = os.environ.copy()
    if extra_lib_dir:
        current_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{extra_lib_dir}:{current_ld}" if current_ld else str(extra_lib_dir)
        
    result = subprocess.run(cmd, env=env, cwd=cwd)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Export Hugging Face merged model to GGUF and Ollama.")
    parser.add_argument(
        "--stage-dir",
        type=Path,
        required=True,
        help="Path to stage directory (e.g. experiments/metformin/stage1_domain)",
    )
    parser.add_argument(
        "--tools-dir",
        type=Path,
        default=Path("tools"),
        help="Path to tools directory containing llama.cpp (default: tools)",
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to quantize the model (default: --quantize, use --no-quantize for FP16)",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="q4_k_m",
        help="Quantization type if --quantize is active (default: q4_k_m; e.g. q5_k_m, q8_0)",
    )
    parser.add_argument(
        "--ollama-name",
        type=str,
        default=None,
        help="If provided, automatically registers model with Ollama under this name",
    )
    parser.add_argument(
        "--keep-f16",
        action="store_true",
        help="Keep the intermediate FP16 GGUF file after quantizing (default: deletes to save disk space)",
    )

    args = parser.parse_args()

    model_dir = args.stage_dir / "merged_model"
    export_dir = args.stage_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    convert_script = args.tools_dir / "llama.cpp" / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"[ERROR] Could not find {convert_script}. Ensure llama.cpp is placed under {args.tools_dir}/llama.cpp")
        sys.exit(1)

    f16_gguf_path = export_dir / "model-fp16.gguf"
    
    # 1. Convert Hugging Face Safetensors to FP16 GGUF
    print("=" * 60)
    print("Step 1: Converting Hugging Face model to GGUF (FP16)...")
    print("=" * 60)
    run_command([
        sys.executable,
        str(convert_script),
        str(model_dir),
        "--outfile", str(f16_gguf_path),
        "--outtype", "f16"
    ])

    final_gguf_path = f16_gguf_path

    # 2. Quantize (Optional)
    if args.quantize:
        print("\n" + "=" * 60)
        print(f"Step 2: Quantizing to {args.quant_type.upper()}...")
        print("=" * 60)
        
        quant_bin = find_llama_quantize(args.tools_dir)
        if not quant_bin:
            print("[ERROR] llama-quantize binary not found! Build it via cmake in tools/llama.cpp/build")
            sys.exit(1)

        quant_gguf_path = export_dir / f"model-{args.quant_type}.gguf"
        run_command([
            str(quant_bin),
            str(f16_gguf_path),
            str(quant_gguf_path),
            args.quant_type
        ], extra_lib_dir=quant_bin.parent)

        final_gguf_path = quant_gguf_path

        # Clean intermediate FP16 file to prevent disk exhaustion
        if not args.keep_f16 and f16_gguf_path.exists():
            print(f"[CLEANUP] Removing intermediate FP16 file: {f16_gguf_path}")
            f16_gguf_path.unlink()

    # 3. Generate Ollama Modelfile
    print("\n" + "=" * 60)
    print("Step 3: Creating Ollama Modelfile...")
    print("=" * 60)
    
    modelfile_path = export_dir / "Modelfile"
    modelfile_content = f"""FROM ./{final_gguf_path.name}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.2
PARAMETER stop "</s>"
PARAMETER stop "### Instruction:"
"""
    modelfile_path.write_text(modelfile_content)
    print(f"Modelfile written to: {modelfile_path}")

    # 4. Register with Ollama (Optional)
    if args.ollama_name:
        print("\n" + "=" * 60)
        print(f"Step 4: Registering model with Ollama as '{args.ollama_name}'...")
        print("=" * 60)
        # Run inside export_dir so Ollama resolves './model-q4_k_m.gguf' correctly
        run_command([
            "ollama", "create", args.ollama_name, "-f", "Modelfile"
        ], cwd=export_dir)
        print(f"\nModel '{args.ollama_name}' is ready to use via: ollama run {args.ollama_name}")

    print("\n[SUCCESS] Export process finished!")

if __name__ == "__main__":
    main()