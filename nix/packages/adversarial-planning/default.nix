{
    pkgs,
    inputs,
    ...
}:
pkgs.callPackage ./adversarial-planning.nix { inherit pkgs inputs; }
