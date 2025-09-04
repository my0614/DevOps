
import os
import cv2
import shutil
import numpy as np

from tifffile import imread
from osgeo import gdal, osr
from skimage.exposure import match_histograms

class ImageProcessor:
    def __init__(self):
        self.ref_img_path = None
        self.target_img_path = None
        self.save_dir = None
    
    def save_rgb_array_as_tif(self, output_path, rgb_array):
        height, width, channels = rgb_array.shape

        if channels != 3:
            raise ValueError("Input array must have 3 channels (RGB).")

        rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
        options = [
            'TILED=YES',
            'BLOCKXSIZE=256',
            'BLOCKYSIZE=256',
            'COMPRESS=DEFLATE',
            'BIGTIFF=YES'
        ]

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_path, width, height, 3, gdal.GDT_Byte, options)
        dataset.SetGeoTransform((0, 1, 0, 0, 0, -1)) # Default GeoTransform, adjust as needed
        
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32652)
        
        dataset.SetProjection(srs.ExportToWkt())
        dataset.GetRasterBand(3).WriteArray(rgb_array[:, :, 0])  # Red
        dataset.GetRasterBand(2).WriteArray(rgb_array[:, :, 1])  # Green
        dataset.GetRasterBand(1).WriteArray(rgb_array[:, :, 2])  # Blue
        dataset.FlushCache()
        dataset = None
        print(f"Saved: {output_path}")

    # 이미지 색상 매칭 및 저장
    def ecdf_histogram_matching(self, target_img_path, ref_img_path, save_dir):
        target_img = imread(target_img_path) 
        ref_img = imread(ref_img_path)  
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        # ECDF 기반 히스토그램 매칭
        matched = match_histograms(target_img, ref_img, channel_axis=-1)
        os.makedirs(save_dir, exist_ok=True)

        save_path = f"{target_img_path.replace('aligned', 'matched')}"
        print(f"Saving matched image to: {save_path}")
        self.save_rgb_array_as_tif(save_path, matched.astype(np.uint8)) # Save the transformed image as a GeoTIFF
        return save_path

    # L 채널만 히스토그램 매칭 후 저장
    def lab(self, target_img,ref_img, save_dir):
        ref_img = cv2.imread(ref_img)
        target_img = cv2.imread(target_img)

        target_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB)
        ref_lab    = cv2.cvtColor(ref_img, cv2.COLOR_RGB2LAB)

        target_lab[:, :, 0] = match_histograms(
            target_lab[:, :, 0], ref_lab[:, :, 0], channel_axis=None
        )
        result_img = cv2.cvtColor(target_lab, cv2.COLOR_LAB2RGB)
        os.makedirs(save_dir, exist_ok=True)

        base_name = os.path.basename(target_img)
        save_path = os.path.join(save_dir, f"{base_name}")
        cv2.imwrite(save_path, result_img)

if __name__ == "__main__":
    ref_files = ""
    target_files = ""
    save_dir = './aligned_image'

    ImageProcessor = ImageProcessor()
    ImageProcessor.color_match_and_save(ref_files, target_files, save_dir)


