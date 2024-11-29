import numpy as np
import pyvista
from PIL import ImageColor


import matplotlib.pyplot as plt

def make_checker_board_texture(color1='black', color2='white', width=1000, height=1000):
    c1 = np.asarray(ImageColor.getcolor(color1, 'RGB')).astype(np.uint8)
    c2 = np.asarray(ImageColor.getcolor(color2, 'RGB')).astype(np.uint8)
    hw = width // 2
    hh = width // 2
    c1_block = np.tile(c1, (hh, hw, 1))
    c2_block = np.tile(c2, (hh, hw, 1))
    tex = np.block([
        [[c1_block], [c2_block]],
        [[c2_block], [c1_block]]
    ])
    return tex

class Visualizer3D :
    def __init__(self, enable_shadow=False, anti_aliasing=True, use_floor=True, add_cube=False,
                 distance=5, elevation=20, azimuth=0,) :
        self.enable_shadow = enable_shadow
        self.anti_aliasing = anti_aliasing
        self.use_floor = use_floor
        self.add_cube = add_cube
        self.hide_env = False

        # camera
        self.distance = distance
        self.elevation = elevation
        self.azimuth = azimuth
    
    def init_camera(self):
        # self.pl.camera_position = 'yz'
        self.pl.camera.focal_point = (0, 0, 0)
        self.pl.camera.position = (self.distance, 0, 0)
        self.pl.camera.elevation = self.elevation
        self.pl.camera.azimuth = self.azimuth
        # self.pl.camera.zoom(1.0)

    def init_scene(self):
        if not self.hide_env:
            # self.pl.set_background('#DBDAD9')
            self.pl.set_background('#FCC2EB', top='#C9DFFF')    # Classic Rose -> Lavender Blue
        # shadow
        if self.enable_shadow:
            self.pl.enable_shadows()
        if self.anti_aliasing:
            self.pl.enable_anti_aliasing()
        # floor
        if self.use_floor:
            wlh = (20.0, 20.0, 0.05)
            center = np.array([0, 0, -wlh[2] * 0.5])
            self.floor_mesh = pyvista.Cube(center, *wlh)
            self.floor_mesh.t_coords *= 2 / self.floor_mesh.t_coords.max()
            tex = pyvista.numpy_to_texture(make_checker_board_texture('#81C6EB', '#D4F1F7'))
            self.pl.add_mesh(self.floor_mesh, texture=tex, ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
        else:
            self.floor_mesh = None
        # cube
        if self.add_cube:
            self.cube_mesh = pyvista.Box()
            self.cube_mesh.points *= 0.1
            self.cube_mesh.translate((0.0, 0.0, 0.1))
            self.pl.add_mesh(self.cube_mesh, color='orange', ambient=0.2, diffuse=0.8, specular=0.8, specular_power=10, smooth_shading=True)

    def show_animation(self, window_size=(800, 800), init_args=None, enable_shadow=None, frame_mode='fps', fps=30, repeat=False, show_axes=True):
        self.interactive = True
        self.frame_mode = frame_mode
        self.fps = fps
        self.repeat = repeat
        if enable_shadow is not None:
            self.enable_shadow = enable_shadow
        self.pl = pyvista.Plotter(window_size=window_size)
        self.init_camera()
        self.init_scene(init_args)
        self.update_scene()
        self.setup_key_callback()
        if show_axes:
            self.pl.show_axes()
        self.pl.show(interactive_update=True)
        if self.frame_mode == 'fps':
            self.fps_animation_loop()
        else:
            self.tframe_animation_loop()
    
if __name__ == '__main__':
    print(">> Make checker board <<")
    checker_board = make_checker_board_texture()
    plt.imshow(checker_board)
    plt.show()

    visualizer = Visualizer3D(add_cube=True, enable_shadow=True)
    visualizer.show_animation(show_axes=True)
