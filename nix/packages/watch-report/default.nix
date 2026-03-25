{
    pkgs,
    inputs,
    system,
    ...
}:
pkgs.callPackage ./watch-report.nix { inherit pkgs inputs system; }
