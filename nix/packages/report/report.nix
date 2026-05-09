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

    # unstable_typstPackages = [
    #     {
    #         name = "cetz";
    #         version = "0.5.2";
    #         hash = "sha256-wttZ+L+VPlTLGKPN/exYXozRjMNdXLShhYVTQt4KV/E=";
    #     }
    # ];

    fontPaths =
        with pkgs;
        map (p: "${p}/share/fonts/opentype") [
            newcomputermodern
            tex-gyre.cursor
            tex-gyre.termes
        ];
}
