# HDST

This is a public repository for all code connected to HDST (High-density spatial transcriptomics). The manuscript is currently available on bioRxiv (https://www.biorxiv.org/content/10.1101/563338v2). 

Please cite: Vickovic S et al. High-definition spatial transcriptomics  for in situ tissue profiling. Nat Methods 2019: doi: https://doi.org/10.1038/s41592-019-0548-y

# Tech workflow
![github-small](https://github.com/broadinstitute/hdst/blob/master/hdst.png)

# File Structure Overview
All processed files are available at: https://portals.broadinstitute.org/single_cell/study/SCP420

There is a total of 36 files currenlty uploaded with the study: 

![github-small](https://github.com/broadinstitute/hdst/blob/master/files.png)

We recommed using the `Bulk Download` function and to consult the `Metadata` file. 

#### `*red_ut*`files: Sorted counts tsv files with:

`bc` barcode (XxY) coordinate  
`spot_px_x` representing (x) pixel coordinate in the HE image and `X`in `bc` 
`spot_px_y` representing (y) pixel coordinate in the HE image and `Y`in `bc 
`gene` representing the gene name 
`count` representing UMI filtered expressed counts per corresponding gene 

(Note: spatial resolution is marked as `HDST`, `5x` or `segments`in all file names) 

#### `*barcodes_under_tissue_annot*`files: files conenction (x,y) coordinates to annotation regions in `HDST` with: 

`bc` barcode (XxY) coordinate  
`spot_px_x` representing (x) pixel coordinate in the HE image and `X`in `bc`  
`spot_px_y` representing (y) pixel coordinate in the HE image and `Y`in `bc  
`annotation_region` as region names to each (x,y) coordinate

#### `*HE.png` files are HE images used in the study 

#### `*HE_Probabilities_mask.tiff` files are coordinates of segmented nuclei based on corresponding HE images

# Alignment
This is [code](./alignment) for aligning HE images to (x,y) barcode coordiantes as given by ST Pipeline ([v.1.5.1](https://github.com/SpatialTranscriptomicsResearch/st_pipeline/releases/tag/1.5.1)). 

# Segmentation
This is [code](./segmentation) for segmenting HE nuclei. HE image segmentation was performed by combining Ilastik and CellProfiler. The labeled segmentation mask was used to assign the individual spots to the corresponding Cell ID. The output CSV file includes Cell IDs, X and Y position of the cells (centroid) and the corresponding spots.

# Cell typing 
This is [code](./cell_typing) for imputing cell types onto (x,y) spatial positions based on scRNA-seq data. 

# Differential expression (DE) analysis
This is [code](./Differential%20expression) for DE analysis between annotated regions.
