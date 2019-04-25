function [] = HD_ST_Master()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Clear variable space
clear all
clc

%% Load tsv spot table
st_spot_table = readtable('CN13_D2_filtered_red_ut.csv','Delimiter','\t');

%% Load mask
st_sc_mask = imread('CN13_D2_HE_Probabilities_mask.tiff');

%% Extract all CellID's for the spot location

% Extract unique values
[unique_spots,unique_value_location,ic] = unique(st_spot_table.bc,'stable');

% Extract only the unique values
unique_x = round(st_spot_table.spot_px_x(unique_value_location));
unique_y = round(st_spot_table.spot_px_y(unique_value_location));
unique_bc = st_spot_table.bc(unique_value_location);

% Extract the overlap with the mask
for i=1:size(unique_spots,1)
    unique_bc{i,2} = st_sc_mask(unique_y(i,1),unique_x(i,1));
end

export_table = cell2table(unique_bc,'VariableNames',{'bc','cell_id'});

writetable(export_table,'CellID_Spot_Position.csv');

end

