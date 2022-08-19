%env setup
clear all
close all
addpath 'functions' 'classes' 'dante' 'CV2' 'models' 'Fontana' 'Targa';

run('functions/sift/vlfeat-0.9.21-bin/vlfeat-0.9.21/toolbox/vl_setup');

%params
env='CV2'; %dante,CV2, Fontana, Targa
index='159';%159=CV2, 1020=Dante, 2=Fontana, 67=Targa
modelFile = "models/refDescriptors"+env+"_"+index+".mat";%load descriptors
load(modelFile); %variable referenceModel
method = MethodName.Fiore;

%xmpFile='DSC_0076.xmp';


for i = 1:10
    %if not(i == 5) && not(i == 6) % && not(i==7)
        checkImageFile = env+"/proof/test_"+i+".jpg";
        xmpFile=env+"/proof/test_"+i+".xmp";

        %testK = getInternals(checkImageFile); % estimated internal params of original test image
        if strcmp(env,'dante')
            [testK, Rv, Tv] = read_xmp(xmpFile);


        else

            [testK, Rv, Tv]=read_xmp_cv2(xmpFile);
        end   
        [R, T] = pose_estimator_loftr( checkImageFile, method, testK,i,env);
        % if i == 1
        %     figure(100)
        %     scatter3(referenceModel.p3D(:,1),referenceModel.p3D(:,2),referenceModel.p3D(:,3),5,'r');
        %     hold on
        %     plotCameraPose(referenceModel.R, referenceModel.T, '  ref');
        % end
        figure(200)
        ptCloud = pcread(env+"/Mesh.ply");
        pcshow(ptCloud)
        set(gcf,'color','w');
        set(gca,'color','w');
        set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15]);
        hold on
        plotCameraPose(referenceModel.R, referenceModel.T, '  ref');
        plotCameraPose(R, T, "  " + num2str(i));

        axis equal
    %end
end