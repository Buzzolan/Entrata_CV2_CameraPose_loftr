function [K, R, t] = read_xmp(name)
% Umberto Castellani: Computer Vision
%
% Read intrinsic and extrinsic parameters from xmp file created by Zephyr
%
% function [K, R, t] = read_xmp(name)

%close all
%clear
%name = '.\Zephyr_Dante_Statue_Dataset\_SAM1001.xmp';


fid=fopen(name, 'r');

%Fixed structure:
string=fgetl(fid);
string=fgetl(fid);
%Third line contains intrinsic:
intrinsics=fgetl(fid);
C = textscan(intrinsics,'%s');
out=C{1};
fx=str2num(out{9}(5:end-1));
fy=str2num(out{10}(5:end-1));
cx=str2num(out{11}(5:end-1));
cy=str2num(out{12}(5:end-1));

K=[fx,    0,        cx;
    0,   fy,        cy;
    0,   0,          1];

string=fgetl(fid);
string=fgetl(fid);
%Line 6 contains rotation

r1=str2num(fgetl(fid));
r2=str2num(fgetl(fid));
r3=str2num(fgetl(fid));
R=[r1; r2; r3];

string=fgetl(fid);
string=fgetl(fid);
t(1)=str2num(fgetl(fid));
t(2)=str2num(fgetl(fid));
t(3)=str2num(fgetl(fid));
t=t';

end

