#!/usr/bin/env bash
set -euo pipefail

data0="../training_data_1p0r_grid.csv"
data1="../training_data_1p0r_anchor.csv"
# data2="NEW_GRID_GNN_INTERPOLATIONS.csv"
grid_points="grid_points_quality.xyz"
grid_points1="grid_points_quality1.xyz"
image_fname="${data0%.csv}"

# Extract grid points with stress quality types from column 8
awk -F, '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $7!="" {
    lon = $2; lat = $3; quality = $7
    # Convert your specific quality encodings to numeric values
    if (quality == "A") quality = 1
    else if (quality == "B") quality = 2  
    else if (quality == "C") quality = 3
    else quality = 1  # default to NF if unknown
    print lon, lat, quality
  }
' "$data0" > "$grid_points"

awk -F, '
  BEGIN{OFS=" "}
  NR>1 && $2!="" && $3!="" && $7!="" {
    lon = $2; lat = $3; quality = $7
    # Convert your specific quality encodings to numeric values
    if (quality == "A") quality = 1
    else if (quality == "B") quality = 2  
    else if (quality == "C") quality = 3
    else quality = 1  # default to NF if unknown
    print lon, lat, quality
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

# Create discrete color palette for stress quality types
# Use distinct, visible colors for 3 quality types
gmt makecpt -Cgreen,yellow,red -T0.5/3.5/1 > quality_colors.cpt

gmt begin $image_fname png A+m0.5c
  # Set up basemap with professional styling
  gmt basemap -R$region -JM8i -Bxa5f5+l"(°E)" -Bya5f5+l"(°N)" -BWsne+t"Quality Ranking of Training Grids"
  
  # Enhanced coastlines and geography
  gmt coast -Dl -Gwhite -Sazure1 -Wthinnest,gray20 -Na/thinnest,gray40
  
  # Plot grid points as small dots colored by stress quality
  gmt plot "$grid_points" -Sc0.30c -Cquality_colors.cpt -W0.3p,black

  gmt legend -DjTR+w3.5c+o-0.01c/-0.01c -F+p1.5p,black+gwhite@30+r0.2c --FONT_ANNOT_PRIMARY=11p,Helvetica << EOF
H 13p,Helvetica-Bold Quality Ranks
G 0.1c
S 0.4c c 0.30c green 0.3p,black 1.2c A
S 0.4c c 0.30c yellow 0.3p,black 1.2c B
S 0.4c c 0.30c red 0.3p,black 1.2c  C
EOF

gmt basemap -TdjTR+w1c+o-0.5c/-0.5c+f1

gmt end show

# Clean up temporary files
rm -f quality_colors.cpt "$grid_points" "$grid_points1"