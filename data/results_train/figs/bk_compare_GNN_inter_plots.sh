#!/usr/bin/env bash
set -euo pipefail

data0="../iS-GNN_grid_0p5r_results_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_8_ramps_30_b_cap_90.csv"
data1="../iS-GNN_anchor_0p2r_results_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_8_ramps_30_b_cap_90.csv"
# data2="shmax_GNN_interpolations_grid.csv"
bars0="stress_bars0.xyz"
bars1="stress_bars1.xyz"
bars2="stress_bars2.xyz"
# bars3="stress_bars3.xyz"
# 1. Prepare the bars0 file (two lines per site: AZI_pred and AZI_pred+180)
image_fname="${data0%.csv}"
awk -F, '
  NR > 1 && $2 != "" && $3 != "" && $10 != "" {
    lon0 = $2
    lat0 = $3
    azi_pred0 = $10
    print lon0, lat0, azi_pred0, 0.20
    print lon0, lat0, azi_pred0+180, 0.20
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


# awk -F, '
#   NR > 1 && $1 != "" && $3 != "" && $2 != "" {
#     lon1 = $3
#     lat1 = $2
#     azi_true1 = $8
#     print lon1, lat1, azi_true1, 0.20
#     print lon1, lat1, azi_true1+180, 0.20
#   }
# ' "$data1" > "$bars3"

# 2. Compute the region with padding (use only valid data0)
read xmin xmax ymin ymax < <(
  awk '{ if(NR==1){mnx=$1;mxx=$1;mny=$2;mxy=$2;next}
         if($1<mnx)mnx=$1; if($1>mxx)mxx=$1;
         if($2<mny)mny=$2; if($2>mxy)mxy=$2 }
         END{pad=1;
         printf("%.3f %.3f %.3f %.3f\n", mnx-pad, mxx+pad, mny-pad, mxy+pad)}' "$bars0"
)
region="${xmin}/${xmax}/${ymin}/${ymax}"

gmt begin "$image_fname" png
  gmt basemap -R$region -JM6i -Bxa10f10 -Bya10f10
  gmt coast -Dl -Glightgray -Slightblue -Wthin -Bxa10f20 -Bya10f20

gmt plot "$bars0" -SV0.0c+e0c -W1p,darkgreen -L
gmt plot "$bars1" -SV0.0c+e0c -W1p,blue -L
# gmt plot "$bars2" -SV0.0c+e0c -W1p,red  -L
# gmt plot "$bars3" -SV0.0c+e0c -W1p,yellow -L


gmt legend -DjTR+w7c+o0.3c/0.3c -F+p1p,black+gwhite@20 << EOF
H 12p,Helvetica-Bold Legend
S 0.35c - 0c - 1p,darkgreen   0.95c GRID   
S 0.35c - 0c - 1p,blue   0.95c ANCHOR True
EOF


gmt end show