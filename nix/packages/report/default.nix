{
    pkgs,
    inputs,
    system,
    ...
}:
pkgs.callPackage ./report.nix { inherit pkgs inputs system; }
