%% Creo il modello di riferimento per la pipeline di stima della posa
addpath 'Enter_CV2_model' 'functions' 'classes' 'Zephyr_Enter_CV2';
run('functions\sift\toolbox\vl_setup');

env = 'enter_cv2'; %dante or cav
disp('Loading points...');

% Load reference camera info
imageIndex = '159';
     
[K_ref, R_ref, T_ref, p2D, p3D] = enter_cv2_get_points('Enter_CV2_model/SamPointCloud.ply', ...
    "Enter_CV2_model/VisibilityRef"+imageIndex+".txt", ...
    "Zephyr_Enter_CV2/Xmp/DSC_0"+imageIndex+".xmp");
   
refImg = imread("Zephyr_Enter_CV2/Xmp/DSC_0"+imageIndex+".JPG");
    


%{
comments
 %}
nPoint = length(p3D);
fprintf('Found %i points\n',nPoint);
disp('Building descriptors...');
run('functions/sift/vlfeat-0.9.21-bin/vlfeat-0.9.21/toolbox/vl_setup');
[f, d] = vl_sift(single(rgb2gray(refImg))) ;
[sel, dist] = dsearchn(f(1:2,:)',p2D);
threshold = 4; 
valid = dist < threshold;
sel = sel(valid);

[p2D_ref, p3D_ref, f_ref, d_ref] = getRefDescriptors(p2D, p3D, f(:,sel), d(:,sel));

fprintf('Attached descriptors to %i points\n', length(p2D_ref));



referenceModel = ReferenceModel(refImg, p2D_ref, p3D_ref, K_ref, R_ref, T_ref, f_ref, d_ref);

fileName = "models/refDescriptorsCV2_"+imageIndex+".mat";
save(fileName, 'referenceModel');
fprintf('Saved model in %s\n', fileName);

correspondences_2D = p2D;
filename2D="models/point2D_"+imageIndex+".mat";
correspondences_3D =  p3D;
filename3D="models/point3D_"+imageIndex+".mat";
save(filename2D, 'correspondences_2D');
save(filename3D, 'correspondences_3D');


%% plotto immagine di riferimento con i relativi descrittori che hanno un
%punto 3D e plotto punti 3D visibili dall'immagine di riferimento
imageIndex = '159';
refImg = imread("Zephyr_Enter_CV2/Xmp/DSC_0"+imageIndex+".JPG");
load("models/refDescriptorsCV2_"+imageIndex+".mat")

figure(1)
imshow(refImg)
axis on
hold on;

for i=1:length(referenceModel.p2D)
    plot(referenceModel.p2D(i,1),referenceModel.p2D(i,2), 'r*', 'MarkerSize', 1, 'LineWidth', 2);
end
title('Punti sift trovati con corrispettivo punto 3D')

figure(2)
ptCloud = pcread('Enter_CV2_model/Mesh_1_enterCV2.ply');
pcshow(ptCloud)
set(gcf,'color','w');
set(gca,'color','w');
set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15]);
hold on
axis equal
title('Nuvola dei punti entrata cv2')



figure(4)
scatter3(referenceModel.p3D(:,1),referenceModel.p3D(:,2),referenceModel.p3D(:,3),'r','filled')
hold on

hold off
grid on
title('Punti visibili dalla camera')
xlabel('asseX')
ylabel('asseY')
zlabel('asseZ')

