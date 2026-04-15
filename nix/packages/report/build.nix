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

    fontPaths = with pkgs; map (p: "${p}/share/fonts/opentype") [
        newcomputermodern
        tex-gyre.cursor
        tex-gyre.termes
    ];
}
