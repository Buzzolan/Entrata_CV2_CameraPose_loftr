function K = getInternals(imgPath)
image = imread(imgPath);

info = imfinfo(imgPath);

[H, W, C]= size(image);

SensorW=35;
Fmm=info.DigitalCamera.FocalLengthIn35mmFilm;

fp=(Fmm*W)/SensorW;

u_0=W/2;
v_0=H/2; 

K=[fp 0 u_0; 0 fp v_0; 0 0 1];
end