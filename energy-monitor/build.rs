fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    // Build scripts run as a separate process before compiling the crate.
    unsafe { std::env::set_var("PROTOC", protoc) };
    tonic_prost_build::compile_protos("proto/energy.proto")?;
    Ok(())
}
