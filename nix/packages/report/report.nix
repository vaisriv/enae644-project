{
    pkgs,
    inputs,
    system,
    ...
}:
inputs.typix.lib.${system}.buildTypstProjectLocal {
    src = ../../..;
    typstSource = "./reports/main.typ";
    typstOutput = "./reports/main.pdf";

    typstOpts = {
        root = ".";
    };

    fontPaths =
        with pkgs;
        map (p: "${p}/share/fonts/opentype") [
            newcomputermodern
            tex-gyre.cursor
            tex-gyre.termes
        ];
}
