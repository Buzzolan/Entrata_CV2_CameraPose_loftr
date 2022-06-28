%env setup
clear all
close all
addpath 'functions' 'classes';

run('functions/sift/vlfeat-0.9.21-bin/vlfeat-0.9.21/toolbox/vl_setup');

%params
method = MethodName.Fiore;
modelFile = 'models/refDescriptorsCV2_159';%load descriptors subsample
load(modelFile); %variable referenceModel
xmpFile='DSC_0076.xmp';


for i = 1:1%intanto testo una sola immagine
%checkImageFile = "dante/test/1020/Sub_test_"+num2str(i)+".jpg";
checkImageFile = "Enter_CV2_model/test/159/DSC_0076.JPG";%per calcolare paramentri interni
[testK R T]=read_xmp_cv2(xmpFile);
%testK = getInternals(checkImageFile); % estimated internal params of original test image
[R, T] = pose_estimator_loftr(referenceModel, checkImageFile, method, testK);
% if i == 1
%     figure(100)
%     scatter3(referenceModel.p3D(:,1),referenceModel.p3D(:,2),referenceModel.p3D(:,3),5,'r');
%     hold on
%     plotCameraPose(referenceModel.R, referenceModel.T, '  ref');
% end
figure(200)
ptCloud = pcread('Enter_CV2_model/Mesh_1_enterCv2.ply');
pcshow(ptCloud)
set(gcf,'color','w');
set(gca,'color','w');
set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15]);
hold on
plotCameraPose(referenceModel.R, referenceModel.T, '  ref');
plotCameraPose(R, T, "  " + num2str(i));

axis equal
end