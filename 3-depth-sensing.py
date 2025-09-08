########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import pyzed.sl as sl
import math
import numpy as np
import sys
import math
import cv2
import os

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use NEURAL depth mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    
    # Setup output directories and video writers
    output_dir = "output"
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    cam_info = zed.get_camera_information()
    w = cam_info.camera_configuration.resolution.width
    h = cam_info.camera_configuration.resolution.height
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(os.path.join(output_dir, 'combined.avi'), fourcc, 20.0, (w*2, h))
    depth_writer = cv2.VideoWriter(os.path.join(output_dir, 'depth.avi'), fourcc, 20.0, (w, h))

    i = 0
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))

    while i < 50:
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # Get and print distance value in mm at the center of the image
            # We measure the distance camera - object using Euclidean distance
            x = round(image.get_width() / 2)
            y = round(image.get_height() / 2)
            err, point_cloud_value = point_cloud.get_value(x, y)

            if math.isfinite(point_cloud_value[2]):
                distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                    point_cloud_value[1] * point_cloud_value[1] +
                                    point_cloud_value[2] * point_cloud_value[2])
                print(f"Distance to Camera at {{{x};{y}}}: {distance}")
            else : 
                print(f"The distance can not be computed at {{{x};{y}}}")
            # Convert to OpenCV format and save frames
            rgb = image.get_data()
            # Drop alpha channel if present
            if rgb.ndim == 3 and rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
            depth_map = depth.get_data()
            # Replace NaNs with zeros for normalization
            depth_map = np.nan_to_num(depth_map, nan=0.0)
            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            combined = np.hstack((rgb, depth_colored))
            cv2.imwrite(os.path.join(frames_dir, f"frame_{i:03d}.jpg"), combined)
            cv2.imwrite(os.path.join(frames_dir, f"depth_{i:03d}.jpg"), depth_colored)
            cv2.imwrite(os.path.join(frames_dir, f"rgb_{i:03d}.jpg"), rgb)
            video_writer.write(combined)
            depth_writer.write(depth_colored)
            print(f"Saved frame {i}")
            i += 1
           

    # Release resources
    video_writer.release()
    depth_writer.release()
    print(f"Videos saved in {output_dir}")
    print(f"Frames saved in {frames_dir}")
    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()