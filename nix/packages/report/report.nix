{
    pkgs,
    inputs,
    system,
    ...
}:
inputs.typix.lib.${system}.buildTypstProjectLocal {
    src = ../../../reports;
    typstSource = "./main.typ";
    typstOutput = "./reports/main.pdf";

    fontPaths = [
        "${pkgs.newcomputermodern}/share/fonts/opentype"
        "${pkgs.tex-gyre.cursor}/share/fonts/opentype"
        "${pkgs.tex-gyre.termes}/share/fonts/opentype"
    ];
}
