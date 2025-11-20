#!/usr/bin/env bash
set -euo pipefail

data0="../iS-GNN_grid_0p6Br_results_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_8_ramps_30_b_cap_90.csv"
data1="../iS-GNN_anchor_0p6Br_results_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_8_ramps_30_b_cap_90.csv"
# data2="shmax_GNN_interpolations_grid.csv"
bars0="stress_bars0.xyz"
bars1="stress_bars1.xyz"
bars2="stress_bars2.xyz"
grid_points="grid_points.xyz"
# bars3="stress_bars3.xyz"
# 1. Prepare the bars0 file (two lines per site: AZI_pred and AZI_pred+180)
image_fname="${data0%.csv}"

FS=$','         # ',' if CSV
LEN_CONST=0.22   # stick half-length (inches)
LEN_MIN=0.05   # inches: base length when MAG=0
LEN_MAX=0.22   # inches: length when MAG=1 (so span = LEN_MAX - LEN_MIN)

# Extract grid points with their values from column 12 for color mapping
awk -F"$FS" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $12!="" {
    lon = $2; lat = $3; value = $12
    # clamp values
    if (value < 0) value = 0; if (value > 1) value = 1
    print lon, lat, value
  }
' "$data0" > "$grid_points"

awk -F"$FS" -v LEN_MIN="$LEN_MIN" -v LEN_MAX="$LEN_MAX" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $10!="" && $12!="" {
    lon = $2; lat = $3; azi = $10; mag = $12
    # clamp + optional threshold to hide ultra-weak sticks
    if (mag < 0) mag = 0; if (mag > 1) mag = 1
    if (mag < 0.15) next     # <-- remove this line if you want to keep all
    len = LEN_MIN + (LEN_MAX - LEN_MIN) * mag
    print lon, lat, azi, len, mag
    print lon, lat, azi+180.0, len, mag
  }
' "$data0" > "$bars0"

# awk -F, '
#   NR > 1 && $2 != "" && $3 != "" && $10 != "" {
#     lon0 = $2
#     lat0 = $3
#     azi_pred0 = $10
#     print lon0, lat0, azi_pred0, 0.20
#     print lon0, lat0, azi_pred0+180, 0.20
#   }
# ' "$data0" > "$bars0"

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

# Create a professional color palette for interpolation confidence
gmt makecpt -Cred,orange,yellow,yellowgreen,green -T0/1/0.1 > grid_colors.cpt

gmt begin "$image_fname" png A+m0.5c
  # Set up basemap with refined grid and annotations
  gmt basemap -R$region -JM8i -Bxa2f1+l"Longitude (°E)" -Bya2f1+l"Latitude (°N)" -BWsne+t"Stress Field Analysis: GNN Interpolation vs Observed Data"
  
  # Enhanced coastlines and geography
  gmt coast -Dl -Gwhite -Sazure1 -Wthinnest,gray30 -Na/thinnest,gray50 -Lx1i/0.5i+c${ymin}+w50k+f+u
  
  # Plot grid points with enhanced styling (interpolation strength)
  gmt plot "$grid_points" -Sc0.25c -Cgrid_colors.cpt -W0.3p,white
  
  # Plot stress orientation bars with refined styling
  gmt plot "$bars0" -SV0.02c+e0.01c+a45+gblack -W1.8p,black -Gblack@50 -L
  gmt plot "$bars1" -SV0.02c+e0.01c+a45+gblue -W1.5p,blue@30 -Gblue@70 -L

  # Enhanced legend with scientific terminology (inside map, top-right)
  gmt legend -DjTR+w6.5c+o-0.5c/-0.5c -F+p1.5p,black+gwhite@10+r0.2c --FONT_ANNOT_PRIMARY=11p,Helvetica << EOF
H 13p,Helvetica-Bold GNN Stress Field Analysis
G 0.1c
S 0.4c c 0.25c - 0.3p,white 1.3c Interpolation Strength
S 0.4c v 0.6c black 1.8p,black 1.3c GNN Grid Predictions
S 0.4c v 0.6c blue@30 1.5p,blue@30 1.3c Anchor Observations
G 0.1c
L 10p,Helvetica-Oblique C Neural Network Analysis
EOF

  # Professional color scale bar beneath the legend (inside map, top-right)
  gmt colorbar -Cgrid_colors.cpt -DjTR+w4c/0.4c+o-2.5c/-3.0c+ma -Baf0.1+l"Interpolation Confidence" -G0/1 --FONT_ANNOT_PRIMARY=9p,Helvetica --FONT_LABEL=10p,Helvetica-Bold << EOF
H 13p,Helvetica-Bold GNN Stress Field Analysis
G 0.1c
S 0.4c c 0.25c - 0.3p,white 1.3c Interpolation Strength
S 0.4c v 0.6c black 1.8p,black 1.3c GNN Grid Predictions
S 0.4c v 0.6c blue@30 1.5p,blue@30 1.3c Anchor Observations  
G 0.1c
L 10p,Helvetica-Oblique C Neural Network Analysis
EOF

  # Add scale bar and north arrow
  gmt basemap -LjBL+w50k+c$((($xmin+$xmax)/2))/$((($ymin+$ymax)/2))+o0.5c/0.5c+f+u
  gmt basemap -TdjTR+w1c+o-0.5c/-0.5c+f1

gmt end show

# Clean up temporary files
rm -f grid_colors.cpt "$grid_points"