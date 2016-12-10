I = checkerboard();

topLeft  = [10, 10];
topRight = [67, 15];
botLeft  = [12, 73];
botRight = [71, 76];

imshow(I); hold on;

plot(topLeft(1),topLeft(2),'*');
plot(topRight(1),topRight(2),'*');
plot(botLeft(1),botLeft(2),'*');
plot(botRight(1),botRight(2),'*');

Pin = [topLeft', topRight', botLeft', botRight'];
Pout= [0 80  0 80;
       0  0 80 80];
   
V = homography_solve(Pin, Pout);

Warped = imwarp(I, )
%%

A = imread('cameraman.tif');
tform = affine2d([1.0 0.2 0.0; 
                  0.2 1.0 0.0; 
                  0.0 0.0 1.0])
              
A2 = imwarp(A, tform);

imshow(A2);

%%

img=imread('cameraman.tif');
if size(img,3)==3
   img=rgb2gray(img);
end
   

movingPoints=[20 40;200 30; 200 200; 1 236] %(x,y) coordinate

%%
fixedPoints=[1 1; 255 1;255 255;1 255];

TFORM = fitgeotrans(movingPoints,fixedPoints,'projective');
R=imref2d(size(img),[1 size(img,2)],[1 size(img,1)]);
imgTransformed=imwarp(imread('cameraman.tif'),R,TFORM,'OutputView',R);
imshow(imgTransformed,[]);
