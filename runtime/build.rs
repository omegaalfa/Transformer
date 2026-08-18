use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/bge.cu");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let object = out.join("bge_cuda.o");
    let library = out.join("libbge_cuda.a");
    let status = Command::new("nvcc")
        .args(["-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-c"])
        .arg("src/cuda/bge.cu")
        .arg("-o")
        .arg(&object)
        .status()
        .expect("nvcc must be installed for the cuda feature");
    assert!(status.success(), "nvcc failed");
    let status = Command::new("ar")
        .arg("crus")
        .arg(&library)
        .arg(&object)
        .status()
        .expect("ar must be installed");
    assert!(status.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=bge_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
