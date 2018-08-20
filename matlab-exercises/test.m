% img = imread('saturn.png');
% noise = randn(size(img)) .* 25;
% % output = double(img) + noise;
% % recovered = output - noise;
% figure;
% subplot(121)
% imshow(img)
% title('Original Image')
% subplot(122)
% imshow(img, [200, 300])
% title('Additive Gaussian Noise')

% hsize = 15;
% sigma = 3;
% h = fspecial('gaussian', hsize, sigma);
% figure;
% subplot(131);
% surf(h);
% subplot(132);
% imagesc(h);
% colorbar;
% subplot(133);
% imshow(h,[]);
% figure;
% out = imfilter(img, h);
% imshow(out);

x = -pi:0.1:pi;
y = -pi:0.3:pi;
[xx, yy] = meshgrid(x,y);
z = (3+2*xx.^2+3*yy.^2);
surf(xx,yy,z);
colorbar;
title('x^2 + y^2');

