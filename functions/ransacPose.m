function [bestInliers, bestOutliers] = ransacPose(p2D, p3D,K,maxIter,t,d)
    i = 0;
    minDist = Inf;
    maxD = 0;
    rng('shuffle');
    while i < maxIter
        i = i+1;
        idxs = randi(length(p2D),6,1);    
        %model
        p2D_model = p2D(idxs,:);
        p3D_model = p3D(idxs,:);
        G = compute_exterior(K,[eye(3) zeros(3,1)], p2D_model',p3D_model', MethodName.Fiore);
        P1=K*G;
        [u1,v1] = proj(P1,p3D);
        projV = [u1,v1];
        inliers = [];
        outliers = [];
        for j = 1:length(p2D)
            dist = norm(p2D(j,:)-projV(j,:));
            if (dist < minDist)
                minDist = dist;
            end
            if abs(dist) <= t
                inliers = [inliers; j];
            else
                outliers = [outliers;j];
            end
        end
        nIn = length(inliers);
        if nIn > maxD 
            bestInliers = inliers;
            bestOutliers = outliers;
            maxD = nIn;
        end
        if nIn >= d
            break;
        end       
    end
end

