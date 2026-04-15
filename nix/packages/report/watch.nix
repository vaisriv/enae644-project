{
    pkgs,
    inputs,
    system,
    ...
}:
inputs.typix.lib.${system}.watchTypstProject {
    typstSource = "./reports/main.typ";
    typstOutput = "./reports/main.pdf";

    fontPaths =
        with pkgs;
        map (p: "${p}/share/fonts/opentype") [
            newcomputermodern
            tex-gyre.cursor
            tex-gyre.termes
        ];
}
