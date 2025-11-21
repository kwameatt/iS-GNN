#!/usr/bin/env bash
set -euo pipefail

dataA="../../data/results_train/iS-GNN_grid_0p6Ar_results_ep_150_bs64_lr_5em04_k_diff_1_kNN_12_kernel_r_0_75_w_0_7em01.csv"
dataB="../../data/results_train/iS-GNN_grid_0p6Br_results_ep_150_bs64_lr_5em04_k_diff_1_kNN_12_kernel_r_0_75_w_0_7em01.csv"
dataC="../../data/results_train/iS-GNN_grid_0p6Cr_results_ep_150_bs64_lr_5em04_k_diff_1_kNN_12_kernel_r_0_75_w_0_7em01.csv"
dataD="../../data/results_train/iS-GNN_grid_0p6Dr_results_ep_150_bs64_lr_5em04_k_diff_1_kNN_12_kernel_r_0_75_w_0_7em01.csv"
dataE="../../data/results_train/iS-GNN_grid_0p6Er_results_ep_150_bs64_lr_5em04_k_diff_1_kNN_12_kernel_r_0_75_w_0_7em01.csv"
dataF="../../data/results_train/iS-GNN_grid_0p6Fr_results_ep_150_bs64_lr_5em04_k_diff_1_kNN_12_kernel_r_0_75_w_0_7em01.csv"
dataG="../../data/results_train/iS-GNN_grid_0p6Gr_results_ep_150_bs64_lr_5em04_k_diff_1_kNN_12_kernel_r_0_75_w_0_7em01.csv"
dataH="../../data/results_train/iS-GNN_grid_0p6Hr_results_ep_150_bs64_lr_5em04_k_diff_1_kNN_12_kernel_r_0_75_w_0_7em01.csv"
dataI="../../data/results_train/iS-GNN_grid_0p6Ir_results_ep_150_bs64_lr_5em04_k_diff_1_kNN_12_kernel_r_0_75_w_0_7em01.csv"

# dataA="../iS-GNN_grid_0p6Ar_results_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_12_ramps_30_b_cap_90.csv"
# dataB="../iS-GNN_grid_0p6Br_results_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_12_ramps_30_b_cap_90.csv"
# dataC="../iS-GNN_grid_0p6Cr_results_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_12_ramps_30_b_cap_90.csv"
# dataD="../iS-GNN_grid_0p6Dr_results_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_12_ramps_30_b_cap_90.csv"
# dataE="../iS-GNN_grid_0p6Er_results_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_12_ramps_30_b_cap_90.csv"
# dataF="../iS-GNN_grid_0p6Fr_results_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_12_ramps_30_b_cap_90.csv"
# dataG="../iS-GNN_grid_0p6Gr_results_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_12_ramps_30_b_cap_90.csv"
# dataH="../iS-GNN_grid_0p6Hr_results_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_12_ramps_30_b_cap_90.csv"
# dataI="../iS-GNN_grid_0p6Ir_results_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_12_ramps_30_b_cap_90.csv"
data1="../../data/results_train/iS-GNN_anchor_0p2r_results_ep100_bs64_lr5em04_z96_k1_kNN_8_ramps_30.csv"
data0="../../data/results_train/iS-GNN_grid_0p2r_results_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_12_ramps_30_b_cap_90.csv"

barsA="stress_barsA.xyz"
barsB="stress_barsB.xyz"
barsC="stress_barsC.xyz"
barsD="stress_barsD.xyz"
barsE="stress_barsE.xyz"
barsF="stress_barsF.xyz"
barsG="stress_barsG.xyz"
barsH="stress_barsH.xyz"
barsI="stress_barsI.xyz"
bars0="stress_bars0.xyz"
bars1="stress_bars1.xyz"
bars2="stress_bars2.xyz"

grid_pointsA="grid_pointsA.xyz"
grid_pointsB="grid_pointsB.xyz"
grid_pointsC="grid_pointsC.xyz"
grid_pointsD="grid_pointsD.xyz"
grid_pointsE="grid_pointsE.xyz"
grid_pointsF="grid_pointsF.xyz"
grid_pointsG="grid_pointsG.xyz"
grid_pointsH="grid_pointsH.xyz"
grid_pointsI="grid_pointsI.xyz"

image_fname="ONLY_nested_graph"
FS=$','
LEN_CONST=0.25
LEN_MIN=0.05
LEN_MAX=0.25

awk -F, '
  NR > 1 && $2 != "" && $3 != "" && $10 != "" {
    lon0 = $2
    lat0 = $3
    azi_pred = $10
    print lon0, lat0, azi_pred, 0.20
    print lon0, lat0, azi_pred+180, 0.20
  }
' "$data0" > "$bars0"

awk -F, '
  NR > 1 && $2 != "" && $3 != "" && $10 != "" {
    lon0 = $2
    lat0 = $3
    azi_true0 = $9
    print lon0, lat0, azi_true0, 0.20
    print lon0, lat0, azi_true0+180, 0.20
  }
' "$data1" > "$bars1"

awk -F, '
  NR > 1 && $2 != "" && $3 != "" && $9 != "" {
    lon0 = $2
    lat0 = $3
    azi_true0 = $9
    print lon0, lat0, azi_true0, 0.20
    print lon0, lat0, azi_true0+180, 0.20
  }
' "$data1" > "$bars2"

# Extract grid points (lon, lat only - no value needed for solid colors)
awk -F"$FS" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" {
    lon = $2; lat = $3
    print lon, lat
  }
' "$dataA" > "$grid_pointsA"

awk -F"$FS" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" {
    lon = $2; lat = $3
    print lon, lat
  }
' "$dataB" > "$grid_pointsB"

awk -F"$FS" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" {
    lon = $2; lat = $3
    print lon, lat
  }
' "$dataC" > "$grid_pointsC"

awk -F"$FS" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" {
    lon = $2; lat = $3
    print lon, lat
  }
' "$dataD" > "$grid_pointsD"

awk -F"$FS" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" {
    lon = $2; lat = $3
    print lon, lat
  }
' "$dataE" > "$grid_pointsE"

awk -F"$FS" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" {
    lon = $2; lat = $3
    print lon, lat
  }
' "$dataF" > "$grid_pointsF"

awk -F"$FS" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" {
    lon = $2; lat = $3
    print lon, lat
  }
' "$dataG" > "$grid_pointsG"

awk -F"$FS" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" {
    lon = $2; lat = $3
    print lon, lat
  }
' "$dataH" > "$grid_pointsH"

awk -F"$FS" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" {
    lon = $2; lat = $3
    print lon, lat
  }
' "$dataI" > "$grid_pointsI"

awk -F"$FS" -v LEN_MIN="$LEN_MIN" -v LEN_MAX="$LEN_MAX" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $10!="" && $12!="" {
    lon = $2; lat = $3; azi = $10; mag = $12
    if (mag < 0) mag = 0; if (mag > 1) mag = 1
    len = 0.21
    print lon, lat, azi, len, mag
    print lon, lat, azi+180.0, len, mag
  }
' "$dataA" > "$barsA"

awk -F"$FS" -v LEN_MIN="$LEN_MIN" -v LEN_MAX="$LEN_MAX" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $10!="" && $12!="" {
    lon = $2; lat = $3; azi = $10; mag = $12
    if (mag < 0) mag = 0; if (mag > 1) mag = 1
    len = 0.21
    print lon, lat, azi, len, mag
    print lon, lat, azi+180.0, len, mag
  }
' "$dataB" > "$barsB"

awk -F"$FS" -v LEN_MIN="$LEN_MIN" -v LEN_MAX="$LEN_MAX" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $10!="" && $12!="" {
    lon = $2; lat = $3; azi = $10; mag = $12
    if (mag < 0) mag = 0; if (mag > 1) mag = 1
    len = 0.21
    print lon, lat, azi, len, mag
    print lon, lat, azi+180.0, len, mag
  }
' "$dataC" > "$barsC"

awk -F"$FS" -v LEN_MIN="$LEN_MIN" -v LEN_MAX="$LEN_MAX" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $10!="" && $12!="" {
    lon = $2; lat = $3; azi = $10; mag = $12
    if (mag < 0) mag = 0; if (mag > 1) mag = 1
    len = 0.21
    print lon, lat, azi, len, mag
    print lon, lat, azi+180.0, len, mag
  }
' "$dataD" > "$barsD"

awk -F"$FS" -v LEN_MIN="$LEN_MIN" -v LEN_MAX="$LEN_MAX" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $10!="" && $12!="" {
    lon = $2; lat = $3; azi = $10; mag = $12
    if (mag < 0) mag = 0; if (mag > 1) mag = 1
    len = 0.21
    print lon, lat, azi, len, mag
    print lon, lat, azi+180.0, len, mag
  }
' "$dataE" > "$barsE"

awk -F"$FS" -v LEN_MIN="$LEN_MIN" -v LEN_MAX="$LEN_MAX" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $10!="" && $12!="" {
    lon = $2; lat = $3; azi = $10; mag = $12
    if (mag < 0) mag = 0; if (mag > 1) mag = 1
    len = 0.21
    print lon, lat, azi, len, mag
    print lon, lat, azi+180.0, len, mag
  }
' "$dataF" > "$barsF"

awk -F"$FS" -v LEN_MIN="$LEN_MIN" -v LEN_MAX="$LEN_MAX" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $10!="" && $12!="" {
    lon = $2; lat = $3; azi = $10; mag = $12
    if (mag < 0) mag = 0; if (mag > 1) mag = 1
    len = 0.21
    print lon, lat, azi, len, mag
    print lon, lat, azi+180.0, len, mag
  }
' "$dataG" > "$barsG"

awk -F"$FS" -v LEN_MIN="$LEN_MIN" -v LEN_MAX="$LEN_MAX" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $10!="" && $12!="" {
    lon = $2; lat = $3; azi = $10; mag = $12
    if (mag < 0) mag = 0; if (mag > 1) mag = 1
    len = 0.21
    print lon, lat, azi, len, mag
    print lon, lat, azi+180.0, len, mag
  }
' "$dataH" > "$barsH"

awk -F"$FS" -v LEN_MIN="$LEN_MIN" -v LEN_MAX="$LEN_MAX" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $10!="" && $12!="" {
    lon = $2; lat = $3; azi = $10; mag = $12
    if (mag < 0) mag = 0; if (mag > 1) mag = 1
    len = 0.21
    print lon, lat, azi, len, mag
    print lon, lat, azi+180.0, len, mag
  }
' "$dataI" > "$barsI"

# Compute the region with padding
read xmin xmax ymin ymax < <(
  awk '{ if(NR==1){mnx=$1;mxx=$1;mny=$2;mxy=$2;next}
         if($1<mnx)mnx=$1; if($1>mxx)mxx=$1;
         if($2<mny)mny=$2; if($2>mxy)mxy=$2 }
         END{pad=1;
         printf("%.3f %.3f %.3f %.3f\n", mnx-pad, mxx+pad, mny-pad, mxy+pad)}' "$barsA"
)
region="${xmin}/${xmax}/${ymin}/${ymax}"

gmt begin "$image_fname" png
  gmt basemap -R$region -JM8i -Bxa5f5+l"(°E)" -Bya5f5+l"(°N)" -BWsne+t" "
  gmt coast -Dl -Gwhite -Sazure1 -Wthin,gray50 -Na/thinnest,gray50

  # Plot grid points with specified solid colors
  gmt plot "$grid_pointsA" -Sc0.3c -Gblack -W0.5p,black
  # gmt plot "$barsA" -SV0.0c+e0c -W2p,black -L
  gmt plot "$grid_pointsB" -Sc0.3c -Gred -W0.5p,red
  # gmt plot "$barsB" -SV0.0c+e0c -W2p,red -L
  gmt plot "$grid_pointsC" -Sc0.3c -Gcyan -W0.5p,cyan
  # gmt plot "$barsC" -SV0.0c+e0c -W2p,cyan -L
  gmt plot "$grid_pointsD" -Sc0.3c -Gyellow -W0.5p,yellow
  # gmt plot "$barsD" -SV0.0c+e0c -W2p,yellow -L
  gmt plot "$grid_pointsE" -Sc0.3c -Gorange -W0.5p,orange
  # gmt plot "$barsE" -SV0.0c+e0c -W2p,orange -L
  gmt plot "$grid_pointsF" -Sc0.3c -Gbrown -W0.5p,brown
  # gmt plot "$barsF" -SV0.0c+e0c -W2p,brown -L
  gmt plot "$grid_pointsG" -Sc0.3c -Gviolet -W0.5p,violet
  # gmt plot "$barsG" -SV0.0c+e0c -W2p,violet -L
  gmt plot "$grid_pointsH" -Sc0.3c -Ggreen -W0.5p,
  # gmt plot "$barsH" -SV0.0c+e0c -W2p,green -L
  gmt plot "$grid_pointsI" -Sc0.3c -Gpurple -W0.5p,purple
  # gmt plot "$barsI" -SV0.0c+e0c -W2p,purple -L
  # gmt plot "$bars0" -SV0.0c+e0c -W2p,blue -L
  gmt plot "$bars1" -SV0.0c+e0c -W2p,blue -L

  # Create legend with proper syntax
  gmt set FONT_ANNOT_PRIMARY 18p,Helvetica,black
  gmt set FONT_LABEL 18p,Helvetica,black
  # gmt legend -DjBC+w2.5c+o0.3c/0.3c -F+p1p,black+gwhite@20+r0.2c << EOF
  gmt legend -DjBC+w12c+o0/0.8c -F+p1p,black+gwhite@20+r0.1c -C0.1c/0.1c << EOF
N 3
Sc0.3c c 0.3c black 0.5p,black 0.6c Grid A
S 0.3c c 0.3c red 0.5p,black 0.6c Grid B
S 0.3c c 0.3c cyan 0.5p,black 0.6c Grid C
N 3
S 0.3c c 0c yellow 0.5p,black 0.6c Grid D
S 0.3c c 0c orange 0.5p,black 0.6c Grid E
S 0.3c c 0c brown 0.5p,black 0.6c Grid F
N 3
S 0.3c c 0c violet 0.5p,black 0.6c Grid G
S 0.3c c 0c green 0.5p,black 0.6c Grid H
S 0.3c c 0c purple 0.5p,black 0.6c Grid I
N 3
S 0.35c - 0c - 2p,blue   0.95c Anchor
EOF
  gmt basemap -TdjTR+w1c+o-0.5c/-0.5c+f1

gmt end show

# Clean up temporary files
rm -f "$grid_pointsA" "$grid_pointsB" "$grid_pointsC" "$grid_pointsD" "$grid_pointsE" "$grid_pointsF" "$grid_pointsG" "$grid_pointsH" "$grid_pointsI"
rm -f "$bars1" "$barsA" "$barsB" "$barsC" "$barsD" "$barsE" "$barsF" "$barsG" "$barsH" "$barsI"