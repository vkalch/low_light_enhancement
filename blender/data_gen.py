bl_info = {
    # required
    'name': 'Blender Bioluminescence Imaging',
    'blender': (2, 93, 0),
    'category': 'Object',
    # optional
    'version': (1, 0, 0),
    'author': 'Avi Balsam',
    'description': 'Blender addon for practical synthetic data generation',
}

"""
File: data_gen_test_discrete.blend

Author 1: J. Kuehne
Author 2: A. Balsam
Date: Summer 2022
Project: ISSI Weizmann Institute 2022
Label: Utility

Summary:
    This file enables generation of sequences of frames to train ANN.
    It simulations the emission of photons as seen in in vivo imaging with
    bioluminescence. It features a kinetic curve used to depict the glowing
    caused by luciferin (visualisation of spline in
    luciferin_spline_visualisation.py) and an exponential decay of the glowing
    caused by the object itself. This is in this case a model of a sleeping
    mouse. The emitted light caused by the luciferin is simplified by a glowing
    object. The kinetic curve, the position and size of both mouse and light
    source are randomised, as is the decay of the glowing of the mouse itself.

Requirements: The python library scipy has to be installed. As blender uses its
    own and not the system python installation, the package has to be installed
    specifically for this installation. If this causes difficulties, this can
    be achieved by the following console commands:
        (there is somewhere in the blender installation folder the python file,
        the commands have to be executed relative to that file)
        PATH_TO_BLENDER_INSTALLATION/.../python3.x -m ensurepip --upgrade
        PATH_TO_BLENDER_INSTALLATION/.../python3.x -m pip install scipy
"""

import logging

import bpy
import random
import math
from scipy import interpolate


class ExamplePanel(bpy.types.Panel):
    bl_idname = 'VIEW3D_PT_example_panel'
    bl_label = 'Example Panel'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        self.layout.label(text='Hello there')


class BlobGeneratorPanel(bpy.types.Panel):
    bl_idname = 'VIEW3D_PT_object_renamer'
    bl_label = 'Add Blobs'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        col = self.layout.column()
        for (prop_name, _) in ADD_BLOB_PROPS:
            row = col.row()
            row.prop(context.scene, prop_name)

        col.operator('opr.object_add_blob_operator', text='Add Blob')

        for (prop_name, _) in RENDER_SCENE_PROPS:
            row = col.row()
            row.prop(context.scene, prop_name)

        col.operator('opr.object_render_scene_operator', text='Render Scene')


class BlobGeneratorOperator(bpy.types.Operator):
    bl_idname = 'opr.object_add_blob_operator'
    bl_label = 'Blob Adder'

    def execute(self, context):
        if len(DATA_GEN) == 0:
            DATA_GEN.append(DataGen(num_samples=context.scene.num_samples))
        blob = DATA_GEN[0].mouse.add_blob()

        if context.scene.randomize_blob_position:
            blob.randomize_position(DATA_GEN[0].mouse.get_x(), DATA_GEN[0].mouse.get_y(), DATA_GEN[0].mouse.get_z())
        else:
            x = context.scene.blob_position_x
            y = context.scene.blob_position_y
            z = context.scene.blob_position_z

            blob.set_position(x, y, z)

        if context.scene.randomize_blob_scale:
            blob.randomize_scale()
        else:
            x = context.scene.blob_scale_x
            y = context.scene.blob_scale_y
            z = context.scene.blob_scale_z

            blob.set_scale(x, y, z)

        if context.scene.randomize_blob_intensity:
            blob.randomize_spline()
        else:
            blob.set_spline_by_peak_intensity(highest_intensity_frame=context.scene.blob_highest_intensity_frame,
                                              peak_intensity_blob=context.scene.blob_highest_intensity)
        return {'FINISHED'}


class RenderSceneOperator(bpy.types.Operator):
    bl_idname = 'opr.object_render_scene_operator'
    bl_label = 'Scene Renderer'

    def execute(self, context):
        if len(DATA_GEN) == 0:
            DATA_GEN.append(DataGen())

        DATA_GEN[0].set_num_frames(context.scene.num_frames)
        DATA_GEN[0].set_num_samples(context.scene.num_samples)
        DATA_GEN[0].set_output_path(context.scene.output_path)
        DATA_GEN[0].render_data()

        return {'FINISHED'}


def scale_object(obj, scaler_x, scaler_y, scaler_z):
    obj.select_set(True)
    bpy.ops.transform.resize(value=(scaler_x, scaler_y, scaler_z))
    obj.select_set(False)


def duplicate_blob(blob_to_duplicate):
    mat = blob_to_duplicate.active_material
    blob_to_duplicate.select_set(True)
    bpy.ops.object.duplicate()
    new_blob = bpy.context.selected_objects[0]
    new_blob.active_material = mat.copy()
    new_blob.select_set(False)
    return new_blob


class Blob:
    def __init__(self,
                 blob=None,
                 num_frames=100,

                 original_scaling_blob=4.20,
                 min_scaling_blob=0.7,
                 max_scaling_blob=1.1,
                 max_shift_blob_x=1.5,
                 max_shift_blob_y=5.0,
                 min_intensity_blob=0.2,
                 max_intensity_blob=0.4,
                 ):
        """
        :param blob: Blender object representing a light-emitting Blob
        :param num_frames: Number of frames in each render
        :param original_scaling_blob: Original scale of Blob
        :param min_scaling_blob: Minimum final scale of Blob
        :param max_scaling_blob: Maximum final scale of Blob
        :param max_shift_blob_x: Maximum position change on the x axis (in the positive and negative directions)
        :param max_shift_blob_y: Maximum position change on the y axis (in the positive and negative directions)
        :param min_intensity_blob: Minimum peak intensity of Blob
        :param max_intensity_blob: Maximum peak intensity of Blob
        """
        self.blob = blob

        self.light_emit_mesh_blob = blob.active_material.node_tree.nodes["Emission"].inputs[1]

        self.max_shift_blob_x = max_shift_blob_x
        self.max_shift_blob_y = max_shift_blob_y
        self.original_scaling_blob = original_scaling_blob
        self.min_scaling_blob = min_scaling_blob
        self.max_scaling_blob = max_scaling_blob

        self.min_intensity_blob = min_intensity_blob
        self.max_intensity_blob = max_intensity_blob
        self.num_frames = num_frames

        self.spline = None

    def set_spline_by_frame(self, x, y):
        """
        Sets the spline based on a list of points. This function can be called manually to circumvent default points of spline.

        :param x: List of points on the x-axis
        :param y: Corresponding points on the y-axis
        :return:
        """
        self.spline = interpolate.splrep(x, y, s=0)

    def set_spline_by_peak_intensity(self, highest_intensity_frame, peak_intensity_blob):
        """
        Sets spline based on peak intensity. This function can be called manually to circumvent random generation of the spline.

        :param highest_intensity_frame: Number frame with highest intensity
        :param peak_intensity_blob: Maximum intensity of the blob

        :return: None, sets spline
        """
        x = [0,
             highest_intensity_frame / 100,
             highest_intensity_frame / 4,
             highest_intensity_frame / 2,
             highest_intensity_frame,
             (3 / 2) * highest_intensity_frame,
             (self.num_frames + highest_intensity_frame) / 2,
             (3 / 4) * self.num_frames + highest_intensity_frame / 4,
             self.num_frames - 1,
             self.num_frames]

        # y - values
        y = [0.0,
             0.0,
             peak_intensity_blob / 5,
             (4 / 5) * peak_intensity_blob,
             peak_intensity_blob,
             (7 / 10) * peak_intensity_blob,
             (3 / 10) * peak_intensity_blob,
             (1 / 10) * peak_intensity_blob,
             0.0,
             0.0]

        self.set_spline_by_frame(x, y)

    def randomize_spline(self):
        """
        Randomize the shape of the spline curve for this Blob.

        :return: This Blob object (for sequencing)
        """
        # frame with highest light intensity blob -> randomized at around 0.3 * num_frames +/- 0.05 * num_frames
        highest_intensity_frame = int(
            (self.num_frames * 0.3) + (random.uniform(-1.0, 1.0) * 0.05 * self.num_frames))

        peak_intensity_blob = random.uniform(self.min_intensity_blob, self.max_intensity_blob)
        # create spline for light emission of blob
        # x - values
        self.set_spline_by_peak_intensity(highest_intensity_frame, peak_intensity_blob)

        # For sequencing
        return self

    def get_blob(self):
        return self.blob

    def set_blob(self, blob):
        self.blob = blob

    def set_position(self, x, y, z):
        self.blob.location = [x, y, z]

    def randomize_position(self, x_mouse, y_mouse, z_mouse):
        """
        Randomize the position of this Blob within the mouse. Note: This method depends on mouse position,
        so make sure to set the position of the mouse before calling this method.

        :param x_mouse: X-coordinate of mouse position
        :param y_mouse: Y-coordinate of mouse position
        :param z_mouse: Z-coordinate of mouse position
        :return: This Blob object (for sequencing)
        """
        # placement of blob (lighting source) -> randomized -> relative to mouse placement
        x_pos = random.uniform(-1.0, 1.0) * self.max_shift_blob_x + x_mouse
        y_pos = random.uniform(-1.0, 1.0) * self.max_shift_blob_y + y_mouse
        z_pos = 0 + z_mouse
        self.set_position(x_pos, y_pos, z_pos)

        return self

    def set_scale(self, x, y, z):
        scale_object(self.blob, x, y, z)

    def randomize_scale(self):
        """
        Randomize scale of blob.

        :return: This Blob object (for sequencing)
        """
        # Set position and scale of blob
        self.blob.scale[0] = self.original_scaling_blob
        self.blob.scale[1] = self.original_scaling_blob
        self.blob.scale[2] = self.original_scaling_blob

        scaler_x_blob = random.uniform(self.min_scaling_blob, self.max_scaling_blob)
        scaler_y_blob = random.uniform(self.min_scaling_blob, self.max_scaling_blob)
        scaler_z_blob = random.uniform(self.min_scaling_blob, self.max_scaling_blob)

        self.set_scale(scaler_x_blob, scaler_y_blob, scaler_z_blob)

        return self

    def interpolate(self, frame):
        """
        Sets this blob's intensity based on previously generated spline function

        :param frame: Number frame to interpolate for
        :return: None, sets intensity of blob
        """
        self.light_emit_mesh_blob.default_value = interpolate.splev(frame, self.spline)


class Mouse:
    def __init__(self,
                 mouse,
                 original_blob,
                 light_source,
                 num_frames=100,
                 min_init_intensity_mouse=6500,
                 max_init_intensity_mouse=7500,
                 min_end_intensity_mouse=4500,
                 max_end_intensity_mouse=5500,
                 min_intensity_blob=0.2,
                 max_intensity_blob=0.4,
                 original_scaling_mouse=0.420,
                 min_scaling_mouse=0.9,
                 max_scaling_mouse=1.1,
                 max_shift_mouse_x=2,
                 max_shift_mouse_y=2,
                 ):
        """
        :param mouse: Blender object representing the mouse
        :param original_blob: Blender object representing the blob originally contained within the blender file
        :param light_source: Blender object representing light source
        :param num_frames: Number of frames to render (full kinematics will always be rendered)
        :param min_init_intensity_mouse: Minimum intensity of initial light emitted from mouse
        :param max_init_intensity_mouse: Maximum intensity of initial light emitted from mouse
        :param min_end_intensity_mouse: Minimum intensity of final light emitted from mouse (should be less than initial intensity)
        :param max_end_intensity_mouse: Maximum intensity of final light emitted from mouse (should be less than initial intensity)
        :param min_intensity_blob: Minimum intensity of blobs at highest-intensity frame
        :param max_intensity_blob: Minimum intensity of blobs at highest-intensity frame
        :param original_scaling_mouse: Original scale of mouse
        :param min_scaling_mouse: Minimum final scale of mouse
        :param max_scaling_mouse: Maximum final scale of mouse
        :param max_shift_mouse_x: Maximum shift in position on the x axis
        :param max_shift_mouse_y: Maximum shift in position on the y axis
        """
        self.num_frames = num_frames

        self.mouse = mouse
        self.light_source = light_source
        self.original_blob = original_blob
        self.blobs = list()

        self.init_intensity_mouse = random.uniform(min_init_intensity_mouse, max_init_intensity_mouse)
        self.end_intensity_mouse = random.uniform(min_end_intensity_mouse, max_end_intensity_mouse)

        self.min_intensity_blob = min_intensity_blob
        self.max_intensity_blob = max_intensity_blob

        self.original_scaling_mouse = original_scaling_mouse
        self.min_scaling_mouse = min_scaling_mouse
        self.max_scaling_mouse = max_scaling_mouse
        self.max_shift_mouse_x = max_shift_mouse_x
        self.max_shift_mouse_y = max_shift_mouse_y

        self.x_mouse = 0
        self.y_mouse = 0
        self.z_mouse = 0

    def set_blobs(self, blobs):
        """
        :param blobs: Array of Blob objects
        :return: None
        """
        for blob in blobs:
            new_blob = duplicate_blob(self.original_blob)
            blob.set_blob(new_blob)

    def add_blob(self,
                 original_scaling_blob=4.20,
                 min_scaling_blob=0.7,
                 max_scaling_blob=1.1,
                 max_shift_blob_x=1.5,
                 max_shift_blob_y=5.0,
                 min_intensity_blob=0.2,
                 max_intensity_blob=0.4,
                 blob_to_duplicate=None,
                 ):
        """
        Add a blob to this mouse.

        :param original_scaling_blob:
        :param min_scaling_blob:
        :param max_scaling_blob:
        :param max_shift_blob_x:
        :param max_shift_blob_y:
        :param min_intensity_blob:
        :param max_intensity_blob:
        :param blob_to_duplicate: If this parameter is set, the Blob passed will be duplicated instead of the original Blob
        :return:
        """
        if blob_to_duplicate is None:
            blob_to_duplicate = self.original_blob

        new_blob = duplicate_blob(blob_to_duplicate)

        blob_obj = Blob(
            new_blob,
            min_intensity_blob=min_intensity_blob,
            max_intensity_blob=max_intensity_blob,
            original_scaling_blob=original_scaling_blob,
            min_scaling_blob=min_scaling_blob,
            max_scaling_blob=max_scaling_blob,
            max_shift_blob_x=max_shift_blob_x,
            max_shift_blob_y=max_shift_blob_y,
            num_frames=self.num_frames
        )
        self.blobs.append(blob_obj)

        return blob_obj

    def get_blobs(self):
        return self.blobs

    def randomize_blobs(self):
        """
        Randomizes the position, scale and spline of all blobs connected to this mouse.

        :return: None
        """
        for blob in self.blobs:
            blob.randomize_position(self.x_mouse, self.y_mouse, self.z_mouse) \
                .randomize_scale() \
                .randomize_spline()

    def set_original_blob_visibility(self, show_blob: bool):
        """
        Sets the visibility of the original blob.

        :param show_blob: True to show the Blob, false to hide the Blob
        :return: None
        """
        self.original_blob.select_set(True)
        bpy.context.object.hide_render = not show_blob
        self.original_blob.select_set(False)

    def randomize_scale(self):
        """
        Randomizes the scale of this mouse.

        :return: None
        """
        # After initializing values, set position and scale of mouse
        self.mouse.scale[0] = self.original_scaling_mouse
        self.mouse.scale[1] = self.original_scaling_mouse
        self.mouse.scale[2] = self.original_scaling_mouse

        scaler_x_mouse = random.uniform(self.min_scaling_mouse, self.max_scaling_mouse)
        scaler_y_mouse = random.uniform(self.min_scaling_mouse, self.max_scaling_mouse)
        scaler_z_mouse = random.uniform(self.min_scaling_mouse, self.max_scaling_mouse)

        # scale
        scale_object(self.mouse, scaler_x_mouse, scaler_y_mouse, scaler_z_mouse)

        return self

    def randomize_position(self):
        """
        Randomizes the position of this mouse.

        :return: None
        """
        self.x_mouse = random.uniform(-1.0, 1.0) * self.max_shift_mouse_x
        self.y_mouse = random.uniform(-1.0, 1.0) * self.max_shift_mouse_y
        self.z_mouse = 0

        self.mouse.location = [self.x_mouse, self.y_mouse, self.z_mouse]

        return self

    def interpolate(self, frame):
        """
        Sets intensity of mouse based on exponential decay

        :param frame: Number frame to set intensity for
        :return: None, sets intensity of mouse
        """
        # illumination mouse -> exponential decay
        self.light_source.energy = (self.init_intensity_mouse - self.end_intensity_mouse) * math.exp(
            (-self.num_frames / 2000) * frame) + self.end_intensity_mouse

    def get_x(self):
        return self.x_mouse

    def get_y(self):
        return self.y_mouse

    def get_z(self):
        return self.z_mouse


class DataGen:
    def __init__(self,
                 img_size=128,
                 num_frames=100,
                 num_samples=10,
                 output_path=None,

                 min_init_intensity_mouse=6500,
                 max_init_intensity_mouse=7500,
                 min_end_intensity_mouse=4500,
                 max_end_intensity_mouse=5500,
                 original_scaling_mouse=0.420,
                 min_scaling_mouse=0.9,
                 max_scaling_mouse=1.1,
                 max_shift_mouse_x=2,
                 max_shift_mouse_y=2,

                 original_scaling_blob=4.20,
                 min_scaling_blob=0.7,
                 max_scaling_blob=1.1,
                 max_shift_blob_x=3.4,
                 max_shift_blob_y=8.9,
                 min_intensity_blob=0.2,
                 max_intensity_blob=0.4,

                 blobs=None
                 ):
        """
        :param img_size: Pixel size of images to generate
        :param num_frames: Number of frames to render in each sample
        :param num_samples: Number of samples to render
        :param min_init_intensity_mouse: Minimum intensity of initial light emitted from mouse
        :param max_init_intensity_mouse: Maximum intensity of initial light emitted from mouse
        :param min_end_intensity_mouse: Minimum intensity of final light emitted from mouse (should be less than initial intensity)
        :param max_end_intensity_mouse: Maximum intensity of final light emitted from mouse (should be less than initial intensity)
        :param original_scaling_mouse: Original scale of mouse
        :param min_scaling_mouse: Minimum final scale of mouse
        :param max_scaling_mouse: Maximum final scale of mouse
        :param max_shift_mouse_x: Maximum shift in position on the x axis
        :param max_shift_mouse_y: Maximum shift in position on the y axis
        :param original_scaling_blob: Original scale of Blob
        :param min_scaling_blob: Minimum final scale of Blob
        :param max_scaling_blob: Maximum final scale of Blob
        :param max_shift_blob_x: Maximum position change on the x axis (in the positive and negative directions)
        :param max_shift_blob_y: Maximum position change on the y axis (in the positive and negative directions)
        :param min_intensity_blob: Minimum peak intensity of Blob
        :param max_intensity_blob: Maximum peak intensity of Blob
        :param blobs: Array of Blob objects (if this is passed, Blobs will be initialized from the parameters specified in the array)
        """
        self.img_size = img_size
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.output_path = output_path

        self.mouse = Mouse(
            mouse=bpy.data.objects['mouse'],
            original_blob=bpy.data.objects['blob'],
            light_source=bpy.data.lights["Light.001"],

            min_init_intensity_mouse=min_init_intensity_mouse,
            max_init_intensity_mouse=max_init_intensity_mouse,
            min_end_intensity_mouse=min_end_intensity_mouse,
            max_end_intensity_mouse=max_end_intensity_mouse,
            original_scaling_mouse=original_scaling_mouse,
            min_scaling_mouse=min_scaling_mouse,
            max_scaling_mouse=max_scaling_mouse,
            max_shift_mouse_x=max_shift_mouse_x,
            max_shift_mouse_y=max_shift_mouse_y,
        )

        if blobs is not None and [type(b) for b in blobs].count(Blob) == len(blobs):
            self.mouse.set_blobs(blobs)

    def set_num_frames(self, num_frames):
        self.num_frames = num_frames

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples

    def set_output_path(self, output_path):
        self.output_path = output_path

    def render_data(self):
        bpy.data.scenes["Scene"].render.image_settings.file_format = 'TIFF'
        bpy.data.scenes["Scene"].render.use_overwrite = False
        bpy.data.scenes["Scene"].render.image_settings.color_mode = 'RGB'
        bpy.data.scenes["Scene"].render.resolution_x = self.img_size
        bpy.data.scenes["Scene"].render.resolution_y = self.img_size

        for sample in range(self.num_samples):
            # Randomize position and scale of mouse and blobs. Make sure to call in this order -- position of blobs
            # is relative to position of mouse.
            self.mouse.randomize_position().randomize_scale().randomize_blobs()

            for frame in range(self.num_frames):
                bpy.context.scene.render.filepath = f"{self.output_path}/{str(sample)}/frame{str(frame)}"

                for blob in self.mouse.get_blobs():
                    blob.interpolate(frame)

                self.mouse.interpolate(frame)

                # save frame to file
                bpy.ops.render.render(write_still=True, use_viewport=True)
        for blob in self.mouse.get_blobs():
            blob.blob.select_set(True)
            bpy.ops.object.delete()


CLASSES = [
    BlobGeneratorPanel,
    BlobGeneratorOperator,
    RenderSceneOperator,
]

ADD_BLOB_PROPS = [
    ('randomize_blob_intensity', bpy.props.BoolProperty(name='Randomize Blob Intensity?', default=True)),
    ('blob_highest_intensity_frame', bpy.props.IntProperty(name='Peak Intensity Frame', default=30)),
    ('blob_highest_intensity', bpy.props.FloatProperty(name='Peak Intensity Val', default=0.3)),
    ('randomize_blob_position', bpy.props.BoolProperty(name='Randomize Blob Position?', default=True)),
    ('blob_position_x', bpy.props.FloatProperty(name='Blob Pos X', default=0.0)),
    ('blob_position_y', bpy.props.FloatProperty(name='Blob Pos Y', default=0.0)),
    ('blob_position_z', bpy.props.FloatProperty(name='Blob Pos Z', default=0.0)),
    ('randomize_blob_scale', bpy.props.BoolProperty(name='Randomize Blob Scale?', default=True)),
    ('blob_scale_x', bpy.props.FloatProperty(name='Blob Scale X', default=0.0)),
    ('blob_scale_y', bpy.props.FloatProperty(name='Blob Scale Y', default=0.0)),
    ('blob_scale_z', bpy.props.FloatProperty(name='Blob Scale Z', default=0.0)),
]

RENDER_SCENE_PROPS = [
    ('num_frames', bpy.props.IntProperty(name='Num Frames', default=100)),
    ('num_samples', bpy.props.IntProperty(name='Num Samples', default=10)),
    ('output_path', bpy.props.StringProperty(name='Output Path', default="/Users/avbalsam/Desktop/blender_animations/training_data_multiblob")),
]

DATA_GEN = list()


def register():
    for (prop_name, prop_value) in ADD_BLOB_PROPS:
        setattr(bpy.types.Scene, prop_name, prop_value)

    for (prop_name, prop_value) in RENDER_SCENE_PROPS:
        setattr(bpy.types.Scene, prop_name, prop_value)

    for klass in CLASSES:
        bpy.utils.register_class(klass)


def unregister():
    for (prop_name, _) in ADD_BLOB_PROPS:
        delattr(bpy.types.Scene, prop_name)

    for (prop_name, _) in RENDER_SCENE_PROPS:
        delattr(bpy.types.Scene, prop_name)

    for klass in CLASSES:
        bpy.utils.unregister_class(klass)


if __name__ == "__main__":
    register()

