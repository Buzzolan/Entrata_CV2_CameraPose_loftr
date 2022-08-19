function [R, T] = pose_estimator_loftr(checkImageFile, method, testK,x, Denv)
disp('Loading test image...');
checkImg =  imread(checkImageFile);% 2nd Img from a different point of view
load("C:\Users\tomma\Documents\Università\Tesi\Codice_LoFTR\CameraPoseEstimation_loftr\"+Denv+"\risultati_proof\Loftr_points"+x+".mat")
%checkImg = imrotate(checkImg_d,-180); 
figure(40)
imshow(checkImg)
title('punti trovati con loftr')
axis on
hold on;
% Plot cross at row 100, column 50
for i=1:length(loftr_2D_checkTest)
    plot(loftr_2D_checkTest(i,1),loftr_2D_checkTest(i,2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
end
set(gca,'Ydir','reverse')
filename="Match_point_loftr"+x+".jpg";
saveas( figure(40), filename );
% punti 3D
figure(4)
scatter3(loftr_3D_checkTest(:,1),loftr_3D_checkTest(:,2),loftr_3D_checkTest(:,3),'r','filled')
hold on
hold off
grid on
title(' punti visibili dalla camera di test')
xlabel('asseX')
ylabel('asseY')
zlabel('asseZ')
%% Loftr method
%INPUT: -devo avere struttura p2D_check nx2 con n= corrispondenze 2D
%       -struttura p3D_check nx3
%p2D_check rappresenta i punti 2D della seconda immagine che hanno una
%corrispondenza con l'immagine di riferimento
%p3D_check sono i rispettivi punti 3D di p2D_check

%% find best model for matched points loftr
% %{
inlier_threshold = floor(length(loftr_2D_checkTest)*1);
inlier_threshold
numIter = 5000;
fprintf('Applying ransac with %i iterations and a threshold of %i inliers...\n', ...
    numIter, inlier_threshold);
%dante test, ransacPose=13t
%scalinate test, ransacPose=15t, in sift minimo 25 
%fontana test, ransacPose=15t
%targa test , randacPose=15t
%dante proof, ransac=13t
%cv2 proof, randac=13t
%fontana proof, ransac=13t
%Targa proof, ransac=13t
%uso solo 15 per proof
[inliers] = ransacPose(loftr_2D_checkTest, loftr_3D_checkTest,testK,numIter,15,inlier_threshold);
numInliers = length(inliers);
fprintf('Best model has %i inliers\n', numInliers);
save("Inliers_test_loftr_"+x,'numInliers')

if  numInliers < 10
    error('Too few matching points to estimate the camera pose')
end

if numInliers < inlier_threshold
    disp('Inliers matching points less than imposed threshold: MODEL COULD BE INACCURATE!')
end

modelPoints = 1:numInliers;
p2D_best = loftr_2D_checkTest(inliers,:);%prendo indici inliers
p3D_best = loftr_3D_checkTest(inliers,:);

disp('Computing exterior parameters');
%G = compute_exterior(testK,[eye(3) zeros(3,1)], p2D_best',p3D_best', method);
G = compute_exterior(testK,[eye(3) zeros(3,1)], loftr_2D_checkTest',loftr_3D_checkTest', method);
%show on command line
save("G_test_loftr_"+x,'G')
plotOnImage(checkImg,p2D_best, p3D_best, testK, G);
title(strcat('Projection of best model points on test image with ',string(method)));
filename="Best_model_with_fiore_loftr"+x+".jpg";
saveas( figure(90), filename );
R = G(1:3,1:3);
T = G(1:3, 4);




end

