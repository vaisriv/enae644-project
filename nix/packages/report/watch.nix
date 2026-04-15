{
    pkgs,
    inputs,
    system,
    ...
}:
inputs.typix.lib.${system}.watchTypstProject {
    typstSource = "./reports/main.typ";
    typstOutput = "./reports/main.pdf";

    fontPaths = [
        "${pkgs.newcomputermodern}/share/fonts/opentype"
        "${pkgs.tex-gyre.cursor}/share/fonts/opentype"
        "${pkgs.tex-gyre.termes}/share/fonts/opentype"
    ];
}
