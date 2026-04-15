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
        # helpers
        ## python
        {
            name = "pyr";
            category = "[python]";
            help = "run";
            command = "uv run adversarial-planning";
        }
        {
            name = "pyl";
            category = "[python]";
            help = "lsp";
            command = "ty check -W src";
        }
        {
            name = "pys";
            category = "[python]";
            help = "compile submission";
            command = "nix run .#adversarial-planning";
        }
        ## typst
        {
            name = "tyr";
            category = "[typst]";
            help = "run";
            command = "nix run .#report.watch";
        }
        {
            name = "tyl";
            category = "[typst]";
            help = "lsp";
            command = "tinymist test --no-dashboard --ignore-system-fonts --watch reports/main.typ";
        }
        {
            name = "tys";
            category = "[typst]";
            help = "compile submission";
            command = "nix run .#report.build";
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

    env = [
        {
            name = "TYPST_FONT_PATHS";
            prefix = with pkgs; lib.makeSearchPath "share/fonts/opentype" [
                newcomputermodern
                tex-gyre.cursor
                tex-gyre.termes
            ];
        }
    ];
}
