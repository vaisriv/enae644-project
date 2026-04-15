{
    pkgs,
    perSystem,
    ...
}:
perSystem.devshell.mkShell {
    name = "enae644 project devshell";
    motd = ''
        {141}📚 enae644 project{reset} devshell
        $(type -p menu &>/dev/null && menu)
    '';

    commands = [
        # typst helper
        {
            name = "typ";
            category = "[submission]";
            help = "compile (and watch) submission typst report";
            command = "nix run .#report.watch";
        }

        # python helper
        {
            name = "py";
            category = "[submission]";
            help = "run submission python script";
            # command = "nix run .#adversarial-planning";
            command = "uv run adversarial-planning";
        }
    ];

    packages = with pkgs; [
        # typst
        tinymist

        # python
        (python313.withPackages (
            ps: with ps; [
                # python packages here
                matplotlib
                numpy
                scipy
                cartopy
                jax
                equinox
            ]
        ))
        uv
        ty
    ];

    env = [ ];
}
