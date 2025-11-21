#!/usr/bin/env bash
set -euo pipefail

# data0="../../data/anchor_train/training_data_1p0r_grid.csv"
data0="../../data/results_train/iS-GNN_grid_0p2r_nested_graph.csv"
data1="../../data/results_train/iS-GNN_anchor_0p2r_results_ep_150_bs64_lr_5em04_k_diff_1_kNN_12_kernel_r_0_75_w_0_7em01.csv"
# data2="NEW_GRID_GNN_INTERPOLATIONS.csv"
grid_points="grid_points_regime.xyz"
grid_points1="grid_points_regime1.xyz"

# Extract grid points with stress regime types from column 8
awk -F, '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $8!="" {
    lon = $2; lat = $3; regime = $6
    # Convert your specific regime encodings to numeric values
    if (regime == "NF") regime = 1
    else if (regime == "NS") regime = 2  
    else if (regime == "SS") regime = 3
    else if (regime == "TF") regime = 4  
    else if (regime == "TS") regime = 5
    print lon, lat, regime
  }
' "$data0" > "$grid_points"

awk -F, '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $8!="" {
    lon = $2; lat = $3; regime = $6
    # Convert your specific regime encodings to numeric values
    if (regime == "NF") regime = 1
    else if (regime == "NS") regime = 2  
    else if (regime == "SS") regime = 3
    else if (regime == "TF") regime = 4  
    else if (regime == "TS") regime = 5
    print lon, lat, regime
  }
' "$data1" > "$grid_points1"

# 2. Compute the region with padding
read xmin xmax ymin ymax < <(
  awk '{ if(NR==1){mnx=$1;mxx=$1;mny=$2;mxy=$2;next}
         if($1<mnx)mnx=$1; if($1>mxx)mxx=$1;
         if($2<mny)mny=$2; if($2>mxy)mxy=$2 }
         END{pad=1;
         printf("%.3f %.3f %.3f %.3f\n", mnx-pad, mxx+pad, mny-pad, mxy+pad)}' "$grid_points"
)
region="${xmin}/${xmax}/${ymin}/${ymax}"

# Create discrete color palette for stress regime types
# Use distinct, visible colors for 3 regime types
gmt makecpt -Cred,yellow,green,blue,orange -T0.5/5.5/1 > regime_colors.cpt

gmt begin grids_stress_regime_nested_grid_map png A+m0.5c
  # Set up basemap with professional styling
  gmt basemap -R$region -JM8i -Bxa5f5+l"(°E)" -Bya5f5+l"(°N)" -BWsne+t"Kinematic Regime of Nested Grid"
  
  # Enhanced coastlines and geography
  gmt coast -Dl -Gwhite -Sazure1 -Wthinnest,gray20 -Na/thinnest,gray40
  
  # Plot grid points as small dots colored by stress regime
  gmt plot "$grid_points" -Sc0.25c -Cregime_colors.cpt -W0.3p,black


  gmt legend -DjBC+w10.5c+o-0.01c/-0.01c -F+p1.5p,black+gwhite@30+r0.2c --FONT_ANNOT_PRIMARY=18p,Helvetica << EOF
H 13p,Helvetica-Bold Regime Types at Nested Grids
G 0.1c
N 2
S 0.4c c 0.25c red 0.3p,black 1.2c NF
S 0.4c c 0.25c yellow 0.3p,black 1.2c NS
N 2
S 0.4c c 0.25c green 0.3p,black 1.2c  SS
S 0.4c c 0.25c blue 0.3p,black 1.2c TF
N 1
S 0.4c c 0.25c orange 0.3p,black 1.2c  TS
EOF

gmt basemap -TdjTR+w1c+o-0.5c/-0.5c+f1

gmt end show

# Clean up temporary files
rm -f regime_colors.cpt "$grid_points" "$grid_points1"