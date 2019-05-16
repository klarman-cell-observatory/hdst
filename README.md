# HDST

This is a public repository for all code connected to HDST (High-density spatial transcriptomics). The manuscript is currently available on bioRxiv (https://www.biorxiv.org/content/10.1101/563338v2). 

Please cite: Vickovic S et al. High-density spatial transcriptomics arrays for in situ tissue profiling. bioRxiv. 2019: doi: https://doi.org/10.1101/563338

# Tech workflow
![github-small](https://github.com/broadinstitute/hdst/blob/master/hdst.png)

# File Structure Overview
All processed files are avaialable at: https://portals.broadinstitute.org/single_cell/study/SCP446

There is a total of XX files currenlty uploaded with the study: 

![github-small](https://github.com/broadinstitute/hdst/blob/master/files.png)

We recommed using the `Bulk Download`function and to consult the `Metadata` file for file descriptions. 

#### `*red_ut*`files: Sorted counts tsv files with:

`bc` barcode (XxY) coordinate  
`spot_px_x`representing (x) pixel coordinate in the HE image and `X`in `bc`  
`spot_px_y`representing (y) pixel coordinate in the HE image and `Y`in `bc  
`gene`representing the gene name  
`count` representing UMI filtered expressed counts per corresponding gene  

(Note: spatial resolution is marked as `HDST`, `5x`, `segments`in all file names)

#### `*barcodes_under_tissue_annot*`files: files conenction (x,y) coordinates to annotation regions in `HDST` with:

`bc` barcode (XxY) coordinate  
`spot_px_x`representing (x) pixel coordinate in the HE image and `X`in `bc`  
`spot_px_y`representing (y) pixel coordinate in the HE image and `Y`in `bc  
`annotation_region` as region names to each (x,y) coordinate

#### `*HE.png` images are HE images used in the study 

#### `*HE_Probabilities_mask.tiff`are coordinates of segmented nuclei based on corresponding HE images

# Alignment
This is [code](https://github.com/broadinstitute/hdst/tree/master/alignment) for aligning HE images to (x,y) barcode coordiantes as given by ST Pipeline ([v.1.5.1](https://github.com/SpatialTranscriptomicsResearch/st_pipeline/releases/tag/1.5.1)). 

# Segmentation
This is code for segmenting HE images on HDST arrays. 

# Cell typing 
This is code for imputing cell types onto (x,y) spatial positions based on scRNA-seq data. 

# Differential expression (DE) analysis
This is code for DE analysis between annotated regions.
