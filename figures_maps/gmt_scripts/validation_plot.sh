#!/usr/bin/env bash
set -euo pipefail

data0="../../data/results_train/iS_GNN_predictions_on_val_set_ep_150_bs64_lr5em04_z96_k_diff_1_kNN_8_ramps_30_b_cap_90.csv"
bars0="stress_bars0.xyz"
bars1="stress_bars1.xyz"
bars2="stress_bars2.xyz"
grid_points="grid_points.xyz"
# bars3="stress_bars3.xyz"
# 1. Prepare the bars0 file (two lines per site: AZI_pred and AZI_pred+180)
image_fname="${data0%.csv}"

FS=$','         # ',' if CSV
LEN_CONST=0.25   # stick half-length (inches)
LEN_MIN=0.05   # inches: base length when MAG=0
LEN_MAX=0.22   # inches: length when MAG=1 (so span = LEN_MAX - LEN_MIN)

# Extract grid points with their values from column 12 for color mapping
awk -F"$FS" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $13!="" {
    lon = $2; lat = $3; value = $13
    # clamp values
    # if (value < 0) value = 0; if (value > 1) value = 1
    print lon, lat, value
  }
' "$data0" > "$grid_points"

awk -F"$FS" -v LEN_MIN="$LEN_MIN" -v LEN_MAX="$LEN_MAX" '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $9!="" {
    lon = $2; lat = $3; azi = $12; mag = $13
    len = LEN_MAX # same length for all bars #LEN_MIN + (LEN_MAX - LEN_MIN) * mag
    print lon, lat, azi, len, mag
    print lon, lat, azi+180.0, len, mag
  }
' "$data0" > "$bars0"


awk -F, '
  NR > 1 && $2 != "" && $3 != "" && $9 != "" {
    lon0 = $2
    lat0 = $3
    azi_true0 = $9
    print lon0, lat0, azi_true0, 0.20
    print lon0, lat0, azi_true0+180, 0.20
  }
' "$data0" > "$bars1"

awk -F, '
  NR > 1 && $2 != "" && $3 != "" && $9 != "" {
    lon0 = $2
    lat0 = $3
    azi_true0 = $9
    print lon0, lat0, azi_true0, 0.20
    print lon0, lat0, azi_true0+180, 0.20
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
gmt makecpt -Cgreen,yellowgreen,yellow,orange,red -T0/50/10 -Z > grid_colors.cpt
gmt begin "$image_fname" png
  gmt basemap -R$region -JM8i -Bxa5f5+l"(°E)" -Bya5f5+l"(°N)" -BWsne+t" "
  gmt coast -Dl -Gwhite -Sazure1 -Wthinnest,gray5 -Na/thinnest,gray25

  # Plot grid points colored by values in column 12
  gmt plot "$grid_points" -Sc0.35c -Cgrid_colors.cpt -W0.5p,black

  # Plot stress orientation bars
  gmt plot "$bars0" -SV0.0c+e0c -W2.5p,black -Gblack@50 -L
  gmt plot "$bars1" -SV0.0c+e0c -W2.5p,blue@20 -L

  # # Add color scale bar
  # Outside, bottom–centre, 8 cm long, 0.4 cm high, offset 0.6 cm below the frame
  gmt colorbar -Cgrid_colors.cpt -DJBC+w8c/0.4c+o0/0.6c+h -Bxaf+l"Mean Absolute Error ("@.")"

  gmt legend -DjTR+w7c+o0.3c/0.3c -F+p1p,black+gwhite@20+r0.2c<< EOF
# S 0.35c c 0.15c - 0.5p,white   0.95c 
S 0.35c - 0c - 2.5p,black   0.95c iS-GNN Prediction  
S 0.35c - 0c - 2.5p,blue   0.95c ANCHOR True
EOF

gmt basemap -TdjTR+w1c+o-0.5c/-0.5c+f1

gmt end show

# Clean up temporary files
rm -f grid_colors.cpt "$grid_points"