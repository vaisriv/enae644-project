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
            name = "pyt";
            category = "[python]";
            help = "test";
            command = "uv run coverage run --omit \"/nix/store/*\" -m pytest tests -W ignore::UserWarning";
        }
        {
            name = "pyc";
            category = "[python]";
            help = "compile";
            command = "nix build .#adversarial-planning";
        }
        ## typst
        {
            name = "typ";
            category = "[typst]";
            help = "preview";
            command = "tinymist preview reports/main.typ --root=.";
        }
        {
            name = "tyl";
            category = "[typst]";
            help = "lsp";
            command = "tinymist test --no-dashboard --ignore-system-fonts --watch reports/main.typ --root=.";
        }
        {
            name = "tyc";
            category = "[typst]";
            help = "compile";
            command = "nix run .#report";
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
                pyyaml
                pytest
                coverage
                optax
            ]
        ))
        uv
        ty
    ];

    env = [
        {
            name = "TYPST_FONT_PATHS";
            prefix =
                with pkgs;
                lib.makeSearchPath "share/fonts/opentype" [
                    newcomputermodern
                    tex-gyre.cursor
                    tex-gyre.termes
                ];
        }
    ];
}
