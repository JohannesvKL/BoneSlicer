

import bpy
import blf
import mathutils
from mathutils import Vector, Matrix
import time
import os
import math
import numpy as np
import bmesh
import random

import csv

from datetime import datetime


def export_measurements_to_csv(segments, original_obj=None, filepath=None, additional_data=None,
                               export_segment_meshes=True,
                               segment_export_subfolder="exported_segments",
                               segment_export_format='OBJ'):
    """
    Export segment measurements to a CSV file and optionally export segment meshes.

    Parameters:
    - segments: List of segmented mesh objects
    - original_obj: The original object before segmentation (optional)
    - filepath: Custom filepath for the CSV. If None, generates a timestamped file.
    - additional_data: Dictionary with any additional data to include in the header.
    - export_segment_meshes: Boolean, whether to export the 3D segment meshes.
    - segment_export_subfolder: String, name of the subfolder for exported meshes,
                               relative to the CSV's directory.
    - segment_export_format: String, format for exported meshes (e.g., 'OBJ', 'STL').

    Returns:
    - Path to the created CSV file, or None if CSV writing failed.
    """
    csv_directory = None
    # Determine CSV filepath and directory
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"bone_segments_{timestamp}.csv"
        blend_filepath = bpy.data.filepath
        if blend_filepath:
            csv_directory = os.path.dirname(blend_filepath)
        else:
            csv_directory = os.path.expanduser("~")
        filepath = os.path.join(csv_directory, csv_filename)
    else:
        csv_directory = os.path.dirname(filepath)

    # --- Prepare path for exporting segment meshes ---
    segments_export_path = None
    if export_segment_meshes and segments:
        segments_export_path = os.path.join(csv_directory, segment_export_subfolder)
        try:
            os.makedirs(segments_export_path, exist_ok=True)
            print(f"Segment meshes will be exported to: {segments_export_path}")
        except OSError as e:
            print(f"Error creating directory for segment meshes '{segments_export_path}': {e}")
            # Fallback: attempt to export to the main CSV directory if subfolder fails
            segments_export_path = csv_directory
            print(f"Attempting to export segment meshes to: {segments_export_path}")
            try:
                os.makedirs(segments_export_path, exist_ok=True) # Ensure main dir exists if it was custom
            except OSError as e_main:
                print(f"Error creating directory '{segments_export_path}': {e_main}")
                export_segment_meshes = False # Disable if folder creation fails completely


    # Prepare header information
    header_info = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Original Object": original_obj.name if original_obj else "N/A",
        "Total Segments": len(segments),
    }

    if original_obj:
        try:
            original_volume = calculate_object_volume(original_obj) # Ensure this function exists
            header_info["Original Volume"] = f"{original_volume:.6f}"
        except Exception as e:
            header_info["Original Volume"] = f"Error: {str(e)}"

    if additional_data and isinstance(additional_data, dict):
        header_info.update(additional_data)

    segment_data = []
    total_segment_volume = 0

    for i, seg_obj in enumerate(segments):
        if not seg_obj or seg_obj.name not in bpy.data.objects:
            print(f"Skipping invalid segment object at index {i}.")
            continue
        if seg_obj.type != 'MESH':
            print(f"Skipping non-mesh segment '{seg_obj.name}' for CSV data.")
            # Still attempt to export if flag is true, as some exporters might handle non-mesh (though unlikely for OBJ/STL)
        
        current_segment_info = {
            "Segment ID": i,
            "Segment Name": seg_obj.name,
            "Vertex Count": len(seg_obj.data.vertices) if seg_obj.type == 'MESH' else 0,
            "Face Count": len(seg_obj.data.polygons) if seg_obj.type == 'MESH' else 0
        }

        # Calculate segment volume if it's a mesh
        volume = None
        if seg_obj.type == 'MESH':
            try:
                volume = calculate_object_volume(seg_obj) # Ensure this function exists
                total_segment_volume += volume
                current_segment_info["Volume"] = f"{volume:.6f}"
            except Exception as e:
                current_segment_info["Volume"] = "Error"
                print(f"Error calculating volume for {seg_obj.name}: {e}")
        else:
            current_segment_info["Volume"] = "N/A (not a mesh)"


        dims = seg_obj.dimensions
        current_segment_info.update({
            "Length X": f"{dims.x:.6f}",
            "Length Y": f"{dims.y:.6f}",
            "Length Z": f"{dims.z:.6f}",
        })
        segment_data.append(current_segment_info)

        # --- Export individual segment mesh ---
        if export_segment_meshes and segments_export_path:
            # Ensure it's a mesh object for typical 3D exports
            if seg_obj.type == 'MESH' and len(seg_obj.data.vertices) > 0:
                original_active = bpy.context.view_layer.objects.active
                original_selected_names = [s_obj.name for s_obj in bpy.context.selected_objects]

                bpy.ops.object.select_all(action='DESELECT')
                seg_obj.select_set(True)
                bpy.context.view_layer.objects.active = seg_obj

                # Sanitize name for filename
                safe_seg_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in seg_obj.name)
                export_filename = f"{safe_seg_name}.{segment_export_format.lower()}"
                full_export_path = os.path.join(segments_export_path, export_filename)

                try:
                    if segment_export_format.upper() == 'OBJ':
                        bpy.ops.wm.obj_export(
                            filepath=full_export_path,
                            export_selected_objects=True,
                            apply_modifiers=True # Often desired for final geometry
                        )
                    elif segment_export_format.upper() == 'STL':
                        bpy.ops.export_mesh.stl(
                            filepath=full_export_path,
                            use_selection=True,
                            use_mesh_modifiers=True 
                        )
                    else:
                        print(f"Unsupported segment export format: {segment_export_format}")
                    print(f"Exported segment: {seg_obj.name} to {full_export_path}")
                except Exception as e:
                    print(f"Error exporting segment {seg_obj.name} to {full_export_path}: {e}")
                finally:
                    # Restore original selection and active object
                    bpy.ops.object.select_all(action='DESELECT')
                    for name in original_selected_names:
                        if name in bpy.data.objects:
                            bpy.data.objects[name].select_set(True)
                    if original_active and original_active.name in bpy.data.objects:
                        bpy.context.view_layer.objects.active = original_active
            elif seg_obj.type != 'MESH':
                print(f"Skipping export of '{seg_obj.name}': not a mesh object.")
            elif len(seg_obj.data.vertices) == 0:
                 print(f"Skipping export of '{seg_obj.name}': mesh has no vertices.")


    summary_data = {
        "Total Segment Volume": f"{total_segment_volume:.6f}",
    }

    if original_obj and original_obj.type == 'MESH':
        try:
            original_volume = calculate_object_volume(original_obj) # Ensure this function exists
            volume_difference = abs(original_volume - total_segment_volume)
            volume_percent_diff = (volume_difference / original_volume) * 100 if original_volume > 0 else float('inf')
            summary_data["Volume Difference"] = f"{volume_difference:.6f}"
            summary_data["Volume Percent Difference"] = f"{volume_percent_diff:.2f}%"
        except Exception as e:
            print(f"Could not calculate original object volume for summary: {e}")


    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["# Bone Segment Measurements"])
            writer.writerow(["# " + "=" * 50])
            for key, value in header_info.items():
                writer.writerow([f"# {key}:", value])
            writer.writerow(["# " + "=" * 50])
            writer.writerow([])

            if segment_data:
                fieldnames = list(segment_data[0].keys())
                dict_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                dict_writer.writeheader()
                dict_writer.writerows(segment_data)

                writer.writerow([])
                writer.writerow(["# Summary"])
                writer.writerow(["# " + "-" * 30])
                for key, value in summary_data.items():
                    writer.writerow([f"# {key}:", value])
        print(f"Measurement data successfully written to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error writing CSV file '{filepath}': {e}")
        return None

def segment_by_cutting(bone_obj, segments=10):
    """
    Segments an object by directly cutting it along planes
    """
    # Make sure object is active
    bpy.ops.object.select_all(action='DESELECT')
    bone_obj.select_set(True)
    bpy.context.view_layer.objects.active = bone_obj
    
    # Calculate bounding box
    bbox = [bone_obj.matrix_world @ mathutils.Vector(corner) for corner in bone_obj.bound_box]
    xs = [v.x for v in bbox]
    min_x, max_x = min(xs), max(xs)
    bone_length = max_x - min_x
    
    # Create a copy of the original object
    original_data = bone_obj.data.copy()
    original_name = bone_obj.name
    
    segmented_objects = []
    
    bbox = [bone_obj.matrix_world @ mathutils.Vector(corner) for corner in bone_obj.bound_box]
    xs = [v.x for v in bbox]
    ys = [v.y for v in bbox]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    extent_x = max_x - min_x
    extent_y = max_y - min_y
    
    if extent_x < extent_y:
        axis = 'y'
        min_val, bone_length = min_y, extent_y
    else:
        axis = 'x'
        min_val, bone_length = min_x, extent_x

    print(f"Axis chosen is {axis}")
    measurements = []
    # Go through each segment
    for i in range(segments):
        start = min_val + (i / segments) * bone_length
        end = min_val + ((i + 1) / segments) * bone_length
        
        # Create a new object for this segment (as before)
        segment_mesh = bpy.data.meshes.new(f"{original_name}_segment_{i}_mesh")
        segment_obj = bpy.data.objects.new(f"{original_name}_segment_{i}", segment_mesh)
        bpy.context.collection.objects.link(segment_obj)
        
        bm = bmesh.new()
        bm.from_mesh(original_data)
        
        # Define plane parameters based on chosen axis
        if axis == 'x':
            plane_co_start = (start, 0, 0)
            plane_no_start = (1, 0, 0)
            plane_co_end   = (end, 0, 0)
            plane_no_end   = (1, 0, 0)
        elif axis == 'y':
            plane_co_start = (0, start, 0)
            plane_no_start = (0, 1, 0)
            plane_co_end   = (0, end, 0)
            plane_no_end   = (0, 1, 0)
        
        # Cutting: Only cut if not the first or last segment
        if i > 0:
            bmesh.ops.bisect_plane(
                bm, 
                geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                plane_co=plane_co_start,
                plane_no=plane_no_start
            )
        if i < segments - 1:
            bmesh.ops.bisect_plane(
                bm, 
                geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                plane_co=plane_co_end,
                plane_no=plane_no_end
            )
        
        # Delete vertices outside the segment's range along the chosen axis
        verts_to_delete = []
        for v in bm.verts:
            if axis == 'x':
                if v.co.x < start or v.co.x > end:
                    verts_to_delete.append(v)
            elif axis == 'y':
                if v.co.y < start or v.co.y > end:
                    verts_to_delete.append(v)
        
        bmesh.ops.delete(bm, geom=verts_to_delete, context='VERTS')
        
        bm.to_mesh(segment_mesh)
        segment_mesh.update()
        bm.free()
        
        segmented_objects.append(segment_obj)
        
        print(f"Segment {i} dimensions: {segment_obj.dimensions}, verts: {len(segment_obj.data.vertices)}")
        if axis == 'x':
            print(f"Segment {i} length on x-axis: {segment_obj.dimensions.x}")
        elif axis == 'y':
            print(f"Segment {i} length on y-axis: {segment_obj.dimensions.y}")
            
    return segmented_objects

def align_to_xy_plane_alt(obj):
    """
    Aligns the given object to the X-Y plane using NumPy for eigenvector calculation
    """
    # Make sure the object is active and selected
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Apply any existing transformations
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    
    if obj.type == 'MESH':
        # Get vertices in world space
        mesh = obj.data
        verts = [obj.matrix_world @ v.co for v in mesh.vertices]
        
        # Calculate center of mass
        center = mathutils.Vector((0, 0, 0))
        for v in verts:
            center += v
        center /= len(verts)
        
        # Convert to numpy arrays for PCA calculation
        points = np.array([[v.x, v.y, v.z] for v in verts])
        mean = np.array([center.x, center.y, center.z])
        
        # Center the data
        points = points - mean
        
        # Calculate covariance matrix
        cov = np.cov(points, rowvar=False)
        
        # Calculate eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Convert numpy eigenvectors to mathutils vectors
        x_axis = mathutils.Vector(eigenvectors[:, 0])
        y_axis = mathutils.Vector(eigenvectors[:, 1])
        z_axis = mathutils.Vector(eigenvectors[:, 2])
        
        # Create rotation matrix
        rot_mat = mathutils.Matrix.Identity(3)
        rot_mat.col[0] = x_axis
        rot_mat.col[1] = y_axis
        rot_mat.col[2] = z_axis
        
        # Ensure it's a proper rotation matrix
        rot_mat = rot_mat.to_4x4()
        
        # Apply rotation to object
        obj.matrix_world = mathutils.Matrix.Translation(obj.location) @ rot_mat
        
        # Apply the transformation
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        
        print(f"Aligned {obj.name} to X-Y plane using NumPy")
        return True
    
    print(f"Object {obj.name} is not a mesh, cannot align")
    return False


def calculate_object_volume(obj):
    # Get evaluated mesh (with modifiers applied)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    # Create a new BMesh and load the evaluated mesh data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    # Transform the BMesh to world coordinates (optional, if you need world-space volume)
    bm.transform(obj.matrix_world)
    
    # Calculate the volume (returns a signed volume; take absolute value)
    volume = abs(bm.calc_volume(signed=True))
    
    # Free the BMesh and clear the evaluated mesh
    bm.free()
    eval_obj.to_mesh_clear()
    
    return volume

def close_mesh_simple(obj):
    """Simple approach to close a mesh using Blender's built-in tools"""
    # Make sure the object is active and selected
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Ensure we're working with vertices
    bpy.ops.mesh.select_mode(type='VERT')
    
    # Try to fill holes
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.fill_holes(sides=0)
    
    # Select non-manifold elements
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()
    
    # Try to create faces from the selection
    bpy.ops.mesh.edge_face_add()
    
    # Try one more round of non-manifold selection and filling
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()
    
    # Try grid fill as a last resort
    try:
        bpy.ops.mesh.fill_grid()
    except:
        # If grid fill fails, try a simple fill
        try:
            bpy.ops.mesh.fill()
        except:
            pass
    
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return obj

class ConfirmAlignmentAndSliceOperator(bpy.types.Operator):
    """Operator to confirm alignment then proceed with slicing."""
    bl_idname = "object.confirm_alignment_and_slice"
    bl_label = "Confirm Alignment and Slice"
    bl_options = {'REGISTER', 'UNDO_GROUPED'}

    bone_obj_name: bpy.props.StringProperty()
    _bone_obj = None
    _handler = None

    @classmethod
    def poll(cls, context):
        return True

    def invoke(self, context, event):
        if self.bone_obj_name:
            self._bone_obj = context.scene.objects.get(self.bone_obj_name)
        else:
            self._bone_obj = context.active_object

        if not self._bone_obj or self._bone_obj.type != 'MESH':
            self.report({'ERROR'}, "No valid mesh object found or specified for slicing.")
            return {'CANCELLED'}
        
        self.reverse = False # Existing initialization
        self.current_number_of_slices = 10  # Default value; can be set externally before invoking

        args = (self, context)
        ConfirmAlignmentAndSliceOperator._handler = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_callback_px, args, 'WINDOW', 'POST_PIXEL'
        )
        context.window_manager.modal_handler_add(self)
        # Updated report message to be more generic until keys for slice count are added
        self.report({'INFO'}, f"Object: '{self._bone_obj.name}'. Ready for manual alignment. Press SPACE to slice, X to flip direction, ESC to cancel.")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if not self._bone_obj or self._bone_obj.name not in bpy.data.objects:
            self.report({'WARNING'}, "Tracked bone object is no longer valid. Cancelling.")
            self._cleanup(context)
            return {'CANCELLED'}
        context.area.tag_redraw()
        if event.type == 'SPACE' and event.value == 'PRESS':
            print("Spacebar pressed, bone is oriented correctly") 
            self._cleanup(context)
            self.report({'INFO'}, "User confirmed. Proceeding with slicing...")
            try:
                self.execute_slicing_and_postprocessing(context)
                self.report({'INFO'}, "Slicing and post-processing complete.")
            except Exception as e:
                self.report({'ERROR'}, f"Error during slicing/post-processing: {e}")
                # Optionally print traceback for debugging
                import traceback
                traceback.print_exc()
                return {'CANCELLED'}
            return {'FINISHED'}
        elif event.type == 'X' and event.value == 'PRESS':
            self.reverse = not self.reverse # Toggle the reverse flag
            self.report({'INFO'}, f"Segment enumeration order reversed: {self.reverse}")
            
        elif event.type in {'ESC', 'RIGHTMOUSE'}:
            self._cleanup(context)
            self.report({'INFO'}, "Slicing cancelled by user.")
            return {'CANCELLED'}
        return {'PASS_THROUGH'}

    def _cleanup(self, context):
        if ConfirmAlignmentAndSliceOperator._handler:
            bpy.types.SpaceView3D.draw_handler_remove(ConfirmAlignmentAndSliceOperator._handler, 'WINDOW')
            ConfirmAlignmentAndSliceOperator._handler = None
        context.area.tag_redraw()

    def execute_slicing_and_postprocessing(self, context):
            # Ensure the correct object is active and selected
            bpy.context.view_layer.objects.active = self._bone_obj
            for obj in bpy.data.objects: obj.select_set(False) # Deselect all other objects
            self._bone_obj.select_set(True)

            print(f"Applying final (manually adjusted) transforms to '{self._bone_obj.name}' before slicing...")
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            print("Transforms applied.")
            # ----------------------------------------------------------

            print(f"Slicing object: {self._bone_obj.name} with {self.current_number_of_slices} slices.")
            start_time = time.perf_counter()

            segments = segment_by_cutting(self._bone_obj, segments=self.current_number_of_slices)
            # ----------------------------------------------------

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            if not segments:
                print("Slicing did not produce any segments.")
                if self._bone_obj and self._bone_obj.name in bpy.data.objects:
                    self._bone_obj.hide_set(True)
                return

            print(f"\n--- Slicing Complete ---")
            print(f"Created {len(segments)} segmented pieces in {elapsed_time:.4f} seconds.")
            total_segment_volume_final = 0
            print("\n--- Post-processing Segments ---")
            
            original_obj_volume = calculate_object_volume(self._bone_obj) # Vol of parent after manual align+apply

            for i, seg_obj in enumerate(segments):
                if not seg_obj or seg_obj.name not in bpy.data.objects:
                    print(f"Segment {i} is invalid or was removed, skipping.")
                    continue
                    
                print(f"\nProcessing Segment: {seg_obj.name}")
                bpy.context.view_layer.objects.active = seg_obj
                for obj_sel in bpy.data.objects: obj_sel.select_set(False)
                seg_obj.select_set(True)

                print(f"  Applying transforms (1st time) to {seg_obj.name}...")
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                
                vol1 = calculate_object_volume(seg_obj)
                print(f"  Object: {seg_obj.name}, Volume (after 1st apply): {vol1:.6f}")

                print(f"  Attempting to close mesh for {seg_obj.name}...")
                closed_obj = close_mesh_simple(seg_obj)
                if not closed_obj or closed_obj.name not in bpy.data.objects:
                    print(f"  Segment {seg_obj.name} seems to have been removed or invalidated. Skipping.")
                    continue

                bpy.context.view_layer.objects.active = closed_obj 
                for obj_sel in bpy.data.objects: obj_sel.select_set(False)
                closed_obj.select_set(True)
                print(f"  Applying transforms (2nd time) to {closed_obj.name} after closing...")
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                
                vol2 = calculate_object_volume(closed_obj)
                print(f"  Updated, Object: {closed_obj.name}, Volume (after close & 2nd apply): {vol2:.6f}")
                
                total_segment_volume_final += vol2

            print("\n--- Volume Summary ---")
            print(f"Original Volume of '{self._bone_obj.name}' (after manual align): {original_obj_volume:.6f}")
            print(f"Sum of Final Segment Volumes (after close & 2nd apply): {total_segment_volume_final:.6f}")

            if self._bone_obj and self._bone_obj.name in bpy.data.objects:
                self._bone_obj.hide_set(True)

                # Store segment data for CSV export
                segment_measurements = []
                for i, seg_obj in enumerate(segments):
                    if not seg_obj or seg_obj.name not in bpy.data.objects:
                        continue
                    
                    # Get final dimensions and volume after all processing
                    dims = seg_obj.dimensions
                    vol = calculate_object_volume(seg_obj)
                    
                    segment_measurements.append({
                        "Segment ID": i,
                        "Name": seg_obj.name,
                        "Volume": vol,
                        "Length X": dims.x,
                        "Length Y": dims.y,
                        "Length Z": dims.z,
                        "Vertex Count": len(seg_obj.data.vertices),
                        "Face Count": len(seg_obj.data.polygons)
                    })
                
                # Export measurements to CSV file
                additional_data = {
                    "Original File": self._bone_obj.name,
                    "Slicing Method": "segment_by_cutting",
                    "Number of Slices": self.current_number_of_slices,
                    "Segment Order Reversed": self.reverse,
                    "Processing Time (s)": elapsed_time
                }
                
                csv_filepath = export_measurements_to_csv(
                    segments,
                    original_obj=self._bone_obj,
                    additional_data=additional_data, 
                    filepath = f"/Users/johannes/Desktop/blend_folder_{self._bone_obj.name}/measurements.csv" #EDIT FILEPATH FOR WRITING HERE
                )
                
                if csv_filepath:
                    self.report({'INFO'}, f"Measurements exported to: {csv_filepath}")
                else:
                    self.report({'WARNING'}, "Failed to export measurements to CSV")

                if self._bone_obj and self._bone_obj.name in bpy.data.objects:
                    self._bone_obj.hide_set(True)
            

    def draw_callback_px(self, op_instance, context): # op_instance is 'self'
        if not op_instance._bone_obj or op_instance._bone_obj.name not in bpy.data.objects:
            return

        font_id = 0
        region = context.region
        
        current_y_offset = 20 
        line_spacing = 5 # Additional space between lines
        base_x_pos = 20

        # --- Line 1: Main Status & Manual Alignment ---
        blf.size(font_id, 18)
        message1 = f"Object: '{op_instance._bone_obj.name}' - Ready for manual alignment (G, R, S keys)."
        text_dims1 = blf.dimensions(font_id, message1)
        blf.position(font_id, base_x_pos, region.height - current_y_offset - text_dims1[1], 0)
        blf.color(font_id, 0.9, 0.9, 0.9, 1.0)
        blf.draw(font_id, message1)
        current_y_offset += text_dims1[1] + line_spacing

        # --- Line 2: Slice Count Adjustment ---
        blf.size(font_id, 16) # Slightly smaller
        # Ensure current_number_of_slices exists on op_instance (it's initialized in invoke)
        slices_info = f"Slices: {op_instance.current_number_of_slices} (Press Z to flip)"
        text_dims2 = blf.dimensions(font_id, slices_info)
        blf.position(font_id, base_x_pos, region.height - current_y_offset - text_dims2[1], 0)
        blf.color(font_id, 0.8, 1.0, 0.8, 1.0) # Greenish
        blf.draw(font_id, slices_info)
        current_y_offset += text_dims2[1] + line_spacing
        
        # --- Line 3: Confirmation/Cancellation ---
        blf.size(font_id, 18) # Back to main size
        message3 = "Press SPACE to Apply Alignment & Slice, ESC to Cancel."
        text_dims3 = blf.dimensions(font_id, message3)
        blf.position(font_id, base_x_pos, region.height - current_y_offset - text_dims3[1], 0)
        blf.color(font_id, 0.9, 0.9, 0.9, 1.0)
        blf.draw(font_id, message3)
        
classes_to_register = (
    ConfirmAlignmentAndSliceOperator,
)

_registered_classes = set() # Keep track of registered classes

def register():
    global _registered_classes
    for cls in classes_to_register:
        if cls not in _registered_classes:
            try:
                bpy.utils.register_class(cls)
                _registered_classes.add(cls)
                print(f"Registered: {cls.__name__}")
            except Exception as e:
                print(f"Error registering {cls.__name__}: {e}")
        else:
            print(f"Already registered: {cls.__name__}")


def unregister():
    global _registered_classes
    for cls in reversed(list(_registered_classes)): # Iterate a copy for removal
        try:
            bpy.utils.unregister_class(cls)
            _registered_classes.remove(cls)
            print(f"Unregistered: {cls.__name__}")
        except Exception as e:
            print(f"Error unregistering {cls.__name__}: {e}")
    _registered_classes.clear()


# --- Main Execution Flow ---
if __name__ == "__main__":
    # --- Unregister and re-register for development ---
    # This helps if you're re-running the script from Blender's text editor.
    # For a final addon, registration happens once in its __init__.py.
    print("--- Script Start ---")
    unregister() # Try to unregister any previous versions
    register()   # Register current versions of classes

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    if bpy.context.selected_objects:
        bpy.ops.object.delete()

    obj_folder = "/Users/johannes/Downloads/SendJohannes/Scans/"  # *** ADJUST PATH ***
    obj_files = ["E_Aegyptiacum_2.obj"]  # *** ADJUST FILE LIST ***

    imported_objects_this_run = []
    for obj_file in obj_files:
        file_path = os.path.join(obj_folder, obj_file)
        if os.path.exists(file_path):
            # Deselect all before import to clearly identify newly imported objects
            bpy.ops.object.select_all(action='DESELECT')
            bpy.ops.wm.obj_import(filepath=file_path)
            # bpy.ops.import_scene.obj(filepath=file_path) # Alternative import
            newly_imported = bpy.context.selected_objects[:]
            if newly_imported:
                imported_objects_this_run.extend(newly_imported)
                print(f"Imported: {obj_file} (Objects: {[o.name for o in newly_imported]})")
            else:
                print(f"Warning: Importing {obj_file} did not result in selected objects.")
        else:
            print(f"File not found: {file_path}")
    
    active_bone_obj = None
    if imported_objects_this_run:
        # Assuming the first object in the list of newly imported ones is the primary target
        active_bone_obj = imported_objects_this_run[0]
        
        # Ensure only the active_bone_obj and its potential parts (if any) are selected for processing
        bpy.ops.object.select_all(action='DESELECT')
        active_bone_obj.select_set(True)
        # If imported_objects_this_run contains multiple parts of the same bone that should be
        # processed together (e.g. after joining), adjust selection here.
        bpy.context.view_layer.objects.active = active_bone_obj
    
    if active_bone_obj and active_bone_obj.type == 'MESH':
        print(f"Processing: {active_bone_obj.name}")

        # 1. Align using PCA
        print("Aligning object using PCA...")
        align_success = align_to_xy_plane_alt(active_bone_obj)
        
        if not align_success:
            print("Alignment failed, stopping.")
        else:
            print(f"PCA Alignment successful for '{active_bone_obj.name}'.")
        

        # 2. Apply ALL transforms to the main bone object before slicing
        print(f"Applying all transforms (location, rotation, scale) to '{active_bone_obj.name}'...")
        bpy.context.view_layer.objects.active = active_bone_obj # Ensure context
        active_bone_obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        print(f"All transforms applied to '{active_bone_obj.name}'.")

        # 3. PAUSE AND WAIT FOR USER CONFIRMATION
        # Invoke the modal operator, passing the name of the bone object
        print("Invoking modal operator for manual adjustment and user confirmation...")
        bpy.ops.object.confirm_alignment_and_slice('INVOKE_DEFAULT', bone_obj_name=active_bone_obj.name)
        # The script execution effectively pauses here from the perspective of the main block.
        # The modal operator now controls the flow.
        print("Modal operator called. User can now manually align and adjust parameters.")

    elif active_bone_obj and active_bone_obj.type != 'MESH':
         print(f"Selected object '{active_bone_obj.name}' is not a MESH. Cannot process.")
    else:
        print("No suitable MESH objects were imported or selected for processing.")

    print("--- Script End (Main Block) ---")
