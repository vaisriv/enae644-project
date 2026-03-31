{
    pkgs,
    inputs,
    ...
}:
inputs.treefmt-nix.lib.mkWrapper pkgs {
    projectRootFile = "flake.nix";

    # nix
    programs = {
        deadnix.enable = true;
        nixfmt = {
            enable = true;
            indent = 4;
        };
    };

    # markdown
    programs.prettier = {
        enable = true;
        settings = {
            tabWidth = 4;
        };
    };

    # typst
    programs.typstyle = {
        enable = true;
        indentWidth = 4;
    };

    # python
    programs = {
        ruff-check.enable = true;
        ruff-format.enable = true;
    };

    # yaml
    programs.yamlfmt = {
        enable = true;
        settings.formatter = {
            indent = 4;
            trim_trailing_whitespace = true;

            force_array_style = "block";
            force_quote_style = "double";
        };
    };
}
