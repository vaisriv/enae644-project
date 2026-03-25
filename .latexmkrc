@default_files = ("./reports/submission/main.tex");

ensure_path('TEXINPUTS', '../../lib/ieeeconf//');
$out_dir = "../../reports/submission/";
$aux_dir = "../../.texaux/";

$aux_out_dir_report = 1;
$bibtex_use = 2;
$do_cd = 1;
$pdf_mode = 4;
$show_time = 1;
$silent = 1;
