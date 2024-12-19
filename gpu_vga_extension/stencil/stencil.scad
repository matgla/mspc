difference() {
    cube([70, 50, .3]);
    translate([-1.5, -3, 0]) minkowski() {
        linear_extrude(.2) import("st2.dxf");
        cube(.1);
    }
}
