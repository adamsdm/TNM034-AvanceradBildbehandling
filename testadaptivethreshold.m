clear;close all;
im1=imread('cameraman.tif');
bwim1=adaptivethreshold(im1,11,0.03,0);

imshow(im1);
