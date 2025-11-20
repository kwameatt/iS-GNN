#!/usr/bin/env bash
set -euo pipefail

data0="../training_data_1p0r_grid.csv"
data1="../training_data_1p0r_anchor.csv"
# data2="NEW_GRID_GNN_INTERPOLATIONS.csv"
grid_points="grid_points_plate.xyz"
grid_points1="grid_points_plate1.xyz"

# Extract grid points with stress plate types from column 8
awk -F, '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $8!="" {
    lon = $2; lat = $3; plate = $6
    # Convert your specific plate encodings to numeric values
    if (plate == "AF") plate = 1
    else if (plate == "AR") plate = 2  
    else if (plate == "AS") plate = 3
    else if (plate == "AT") plate = 4  
    else if (plate == "EU") plate = 5
    print lon, lat, plate
  }
' "$data1" > "$grid_points"

awk -F, '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $8!="" {
    lon = $2; lat = $3; plate = $6
    # Convert your specific plate encodings to numeric values
    if (plate == "AF") plate = 1
    else if (plate == "AR") plate = 2  
    else if (plate == "AS") plate = 3
    else if (plate == "AT") plate = 4  
    else if (plate == "EU") plate = 5
    print lon, lat, plate
  }
' "$data1" > "$grid_points1"

# 2. Compute the region with padding
read xmin xmax ymin ymax < <(
  awk '{ if(NR==1){mnx=$1;mxx=$1;mny=$2;mxy=$2;next}
         if($1<mnx)mnx=$1; if($1>mxx)mxx=$1;
         if($2<mny)mny=$2; if($2>mxy)mxy=$2 }
         END{pad=1;
         printf("%.3f %.3f %.3f %.3f\n", mnx-pad, mxx+pad, mny-pad, mxy+pad)}' "$grid_points1"
)
region="${xmin}/${xmax}/${ymin}/${ymax}"

# Create discrete color palette for stress plate types
# Use distinct, visible colors for 3 plate types
gmt makecpt -Cred,green,blue,purple,darkbrown -T0.5/5.5/1 > plate_colors.cpt

gmt begin anchor_stress_plate_map png A+m0.5c
  # Set up basemap with professional styling
  gmt basemap -R$region -JM8i -Bxa5f5+l"(°E)" -Bya5f5+l"(°N)" -BWsne+t"Stress Plates of Training Anchors"
  
  # Enhanced coastlines and geography
  gmt coast -Dl -Gwhite -Sazure1 -Wthinnest,gray10 -Na/thinnest,gray30
  
  # Plot grid points as small dots colored by stress plate
  gmt plot "$grid_points" -Sc0.25c -Cplate_colors.cpt -W0.3p,black

  gmt legend -DjTR+w2.6c+o0.1c/0.1c -F+p1.5p,black+gwhite@30+r0.2c --FONT_ANNOT_PRIMARY=11p,Helvetica << EOF
H 13p,Helvetica-Bold Plate ID
G 0.1c
S 0.4c c 0.25c red 0.3p,black 1.2c AF
S 0.4c c 0.25c green 0.3p,black 1.2c AR
S 0.4c c 0.25c blue 0.3p,black 1.2c  AS
S 0.4c c 0.25c purple 0.3p,black 1.2c AT
S 0.4c c 0.25c darkbrown 0.3p,black 1.2c  EU
EOF

gmt basemap -TdjTR+w1c+o-0.5c/-0.5c+f1

gmt end show

# Clean up temporary files
rm -f plate_colors.cpt "$grid_points" "$grid_points"