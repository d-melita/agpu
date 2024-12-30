{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs-unstable, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs-unstable { inherit system; config.allowUnfree = true; };
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            cudaPackages_11.cuda_nvcc
          ];
        };
      }
    );
}