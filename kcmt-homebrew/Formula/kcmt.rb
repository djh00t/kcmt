class Kcmt < Formula
  desc "Rust CLI for generating conventional commits"
  homepage "https://github.com/djh00t/kcmt"
  version "0.3.2"
  url "https://github.com/djh00t/kcmt/archive/refs/tags/v#{version}.tar.gz"
  sha256 "REPLACE_WITH_RELEASE_TARBALL_SHA256"
  license "MIT"

  depends_on "rust" => :build

  def install
    system(
      "cargo", "build", "--locked", "--release",
      "--manifest-path", "rust/Cargo.toml", "-p", "kcmt-cli"
    )
    bin.install "rust/target/release/kcmt"
    bin.install "rust/target/release/commit"
    bin.install "rust/target/release/kc"
  end

  test do
    assert_match "Usage:", shell_output("#{bin}/kcmt --help")
  end
end
