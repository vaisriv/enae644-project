{
    pkgs,
    inputs,
    system,
    ...
}:
{
    build = pkgs.callPackage ./build.nix { inherit pkgs inputs system; };
    watch = pkgs.callPackage ./watch.nix { inherit pkgs inputs system; };
}
