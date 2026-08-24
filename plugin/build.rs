use std::path::PathBuf;
use std::process::Command;

/// Version = timestamp of the last commit, the same scheme CueHammer itself
/// uses: never needs a manual bump and is identical across platform builds
/// of the same commit. The display form uses the commit's own timezone,
/// keeping it deterministic across machines.
fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    println!(
        "cargo:rerun-if-changed={}",
        root.join(".git/logs/HEAD").display()
    );
    let git = |args: &[&str]| -> Option<String> {
        let out = Command::new("git")
            .args(args)
            .current_dir(&root)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let display = git(&["log", "-1", "--date=format:%Y.%m.%d-%H%M", "--format=%cd"])
        .unwrap_or_else(|| "dev".into());
    println!("cargo:rustc-env=BRIDGE_VERSION_DISPLAY={display}");
}
