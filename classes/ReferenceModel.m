classdef ReferenceModel
    properties
        image
        p2D
        p3D
        K
        R
        T
        f %sift 
        d %sift
    end
    
    methods
        function obj = ReferenceModel(image, p2D, p3D, K, R, T, f, d)
            obj.image = image;
            obj.p2D = p2D;
            obj.p3D = p3D;
            obj.K = K;
            obj.R = R;
            obj.T = T;
            obj.f = f;
            obj.d = d;
        end
    end
end

