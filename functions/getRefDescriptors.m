function [p2D_ref, p3D_ref, f_ref, d_ref] = getRefDescriptors(p2D, p3D, f, d)
    nPoints = size(f,2);
    for i = 1:nPoints
        p2DIdx = dsearchn(p2D, f(1:2, i)');
        p2D_ref(i,:) = p2D(p2DIdx,:);
        p3D_ref(i,:) = p3D(p2DIdx,:);
        f_ref(:,i) = f(:,i);
        d_ref(:,i) = d(:,i);
    end
end

