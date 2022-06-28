function plotCameraPose(R, T, label)
    pose = -R'*T;
    plot3(pose(1), pose(2), pose(3),'-o','Color','b','MarkerSize',10);
    
    ref = R;
    hold on;
    f1 = quiver3(pose(1), pose(2), pose(3),R(1,1),R(1,2),R(1,3),'Color','r','DisplayName','t');
    f2 = quiver3(pose(1), pose(2), pose(3),R(2,1),R(2,2),R(2,3),'Color','g','DisplayName','n');
    f3 = quiver3(pose(1), pose(2), pose(3),R(3,1),R(3,2),R(3,3),'Color','b','DisplayName','b');
    text(pose(1), pose(2), pose(3),label);
end

