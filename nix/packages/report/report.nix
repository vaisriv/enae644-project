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

    unstable_typstPackages = [
        {
            name = "bamdone-ieeeconf";
            version = "0.1.3";
            hash = "sha256-1aZsMoL/FFKyr9M3cNIPpbM/BZ7hDPYr1NgEeQOA/lE=";
        }
    ];
}
