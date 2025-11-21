#!/usr/bin/env bash
set -euo pipefail

data0="../../data/grids_interp/results-3/SHiNE_GNN_compare_3.csv"
data1="../../data/results_train/iS-GNN_anchor_0p2r_results_ep_150_bs64_lr_5em04_k_diff_1_kNN_12_kernel_r_0_75_w_0_7em01.csv"
# data2="/mnt/e/GNN_paper/2025_IGNNK/data/grids_interp/SHINE.csv" # SHINE results

bars0="stress_bars0.xyz"
bars1="stress_bars1.xyz"
bars2="stress_bars2.xyz"
grid_points="grid_points.xyz"

# 1. Prepare the bars0 file (two lines per site: AZI_pred and AZI_pred+180)
image_fname="${data0%.csv}"

FS=$','         # ',' if CSV
LEN_CONST=0.25   # stick half-length (inches)
LEN_MIN=0.05   # inches: base length when MAG=0
LEN_MAX=0.25   # inches: length when MAG=1 (so span = LEN_MAX - LEN_MIN)

# Extract grid points with their values from column 12 for color mapping
awk -F"$FS" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $1!="" {
    lon = $2; lat = $3; abs_err = $6
    print lon, lat, abs_err
  }
' "$data0" > "$grid_points"

# awk -F"$FS" '
#   BEGIN{OFS=" "}
#   NR>1 && $2!="" && $3!="" && $1!="" {
#     lon = $2; lat = $3; value = $1
#     # clamp values
#     if (value < 0) value = 0; if (value > 1) value = 1
#     print lon, lat, value
#   }
# ' "$data2" > "$grid_points2"

# GNN
awk -F"$FS" -v LEN_MIN="$LEN_MIN" -v LEN_MAX="$LEN_MAX" '
  BEGIN{OFS=" "}
  NR>1 && $4!="" && $5!="" && $6!="" {
    lon = $2; lat = $3; azi = $1; abs_err = $6
    len = 0.24 
    print lon, lat, azi, len, abs_err
    print lon, lat, azi+180.0, len, abs_err
  }
' "$data0" > "$bars0"

# TRUE/ANCHOR
awk -F, '
  NR > 1 && $2 != "" && $3 != "" && $10 != "" {
    lon0 = $2
    lat0 = $3
    azi_true0 = $9
    print lon0, lat0, azi_true0, 0.20
    print lon0, lat0, azi_true0+180, 0.20
  }
' "$data1" > "$bars1"

#SHINE
awk -F"$FS" -v LEN_MIN="$LEN_MIN" -v LEN_MAX="$LEN_MAX" '
  BEGIN{OFS=" "}
  NR>1 && $4!="" && $5!="" && $6!="" {
    lon = $2; lat = $3; azi = $5; abs_err = $6
    len = 0.24 
    print lon, lat, azi, len, abs_err
    print lon, lat, azi+180.0, len, abs_err
  }
' "$data0" > "$bars2"


# 2. Compute the region with padding (use only valid data0)
read xmin xmax ymin ymax < <(
  awk '{ if(NR==1){mnx=$1;mxx=$1;mny=$2;mxy=$2;next}
         if($1<mnx)mnx=$1; if($1>mxx)mxx=$1;
         if($2<mny)mny=$2; if($2>mxy)mxy=$2 }
         END{pad=1;
         printf("%.3f %.3f %.3f %.3f\n", mnx-pad, mxx+pad, mny-pad, mxy+pad)}' "$bars0"
)
region="${xmin}/${xmax}/${ymin}/${ymax}"

# Create a color palette for the grid values (adjust range as needed)
gmt makecpt -Cgreen,yellowgreen,yellow,orange,red -T0/180/30 -Z > grid_colors.cpt

gmt begin "$image_fname" png
  gmt basemap -R$region -JM8i -Bxa5f5+l"(°E)" -Bya5f5+l"(°N)" -BWsne+t" "
  gmt coast -Dl -Gwhite -Sazure1 -Wthin,gray50 -Na/thinnest,gray50

  gmt plot "$grid_points" -Sc0.25c -Cgrid_colors.cpt -W0.5p,black

  # Plot stress orientation bars
  gmt plot "$bars0" -SV0.0c+e0c -W2.0p,black -Gblack@40 -L
  gmt plot "$bars2" -SV0.0c+e0c -W2.0p,brown -Gviolet@40 -L
  gmt plot "$bars1" -SV0.0c+e0c -W2.0p,blue@40 -L

  # Add color scale bar
    # Set larger font for colorbar annotations and label
  gmt set FONT_ANNOT_PRIMARY 18p,Helvetica,black
  gmt set FONT_LABEL 18p,Helvetica,black
  # gmt colorbar -Cgrid_colors.cpt -DjBR+w4c/0.5c+o0.5c/1c+h -Baf+l"MAE"
  gmt colorbar -Cgrid_colors.cpt -DjBC+w8c/0.6c+o0/0.6c+h -Bxaf+l"Angular Absolute Error"

  gmt legend -DjTR+w11c+o0.3c/0.3c -F+p1p,black+gwhite@20+r0.2c<< EOF
# S 0.35c c 0.15c - 0.5p,white   0.95c 
S 0.45c - 0c - 2.5p,black   0.95c iS-GNN Grid Interpolation  
S 0.45c - 0c - 2.5p,brown   0.95c SHINE Grid Interpolation
S 0.35c - 0c - 2.5p,blue   0.95c Anchor SHmax
EOF

gmt basemap -TdjTR+w1c+o-0.5c/-0.5c+f1

gmt end show

# Clean up temporary files
rm -f grid_colors.cpt "$grid_points"