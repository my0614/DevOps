import os
import cv2
import sys
import time
import torch
import shutil
import numpy as np
import os.path as osp
import matplotlib.cm as cm

from osgeo import gdal, osr
from load_model import load_model
from kornia.geometry.transform import warp_perspective

class ImageAlignment:
    # Save RGB array as a GeoTIFF file
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

    def eval_relapose(self, matcher, pair, save_figs, figures_dir=None, method=None,):
        im0 = pair['im0']
        im1 = pair['im1']
        match_res = matcher(im0, im1)
        img0_color = cv2.imread(im0)
        img1_color = cv2.imread(im1)
        img0_color = cv2.cvtColor(img0_color, cv2.COLOR_BGR2RGB)
        img1_color = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)
        mkpts0 = match_res['mkpts0']
        mkpts1 = match_res['mkpts1']
        mconf = match_res['mconf']

        if len(mconf) > 0:
            conf_min = mconf.min()
            conf_max = mconf.max()
            mconf = (mconf - conf_min) / (conf_max - conf_min + 1e-5)
        color = cm.jet(mconf)

        if len(mkpts0) >= 4:
            ret_H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
        else:
            inliers = None
            ret_H = None
            
        print(f"Number of inliers: {inliers.sum() if inliers is not None else 0}")
        
        if save_figs:
            img0_name = f"{osp.basename(pair['im0']).split('.')[0]}"
            img1_name = f"{osp.basename(pair['im1'])}"
            shutil.copyfile(pair['im1'], osp.join(figures_dir, img1_name))
            print("copy finished!")

            if ret_H is not None:
                img0_color=cv2.cvtColor(img0_color, cv2.COLOR_RGB2BGR)
                im0_tensor = torch.tensor(img0_color, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.
                ret_H = torch.tensor(ret_H, dtype=torch.float32).unsqueeze(0)
                im0_tensor = self.H_transform(im0_tensor, ret_H)
                im0 = im0_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255
                im0_uint8 = np.clip(im0, 0, 255).astype(np.uint8)
                tif_path = osp.join(figures_dir, f"{img0_name}_aligned.tif")
                self.save_rgb_array_as_tif(tif_path, im0_uint8) # Save the transformed image as a GeoTIFF
                return tif_path, osp.join(figures_dir,img1_name)

    def H_transform(self, img2_tensor, homography):
        image_shape = img2_tensor.shape[2:]
        img2_tensor = warp_perspective(img2_tensor, homography, image_shape, align_corners=True)
        return img2_tensor
            
    def align_images(self, ref_img_path, target_img_path, save_dir,method="sp_lg"):
        # method 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # * Weights 경로 수정 필요*
        if method == "loftr":
            ckpt = "minima_loftr.ckpt"
        elif method == "sp_lg":
            ckpt = "minima_lightglue.pth"

        args = {
        'ckpt': ckpt,
        'fig1' : ref_img_path,
        'fig2' : target_img_path,
        'save_dir' : save_dir,
        }

        scene_pairs = {'im0': ref_img_path, 'im1':target_img_path }
        matcher = load_model(method, args)
        t1,t2 = self.eval_relapose(matcher, scene_pairs, save_figs= True, figures_dir=save_dir, method=method)
        return t1,t2

if __name__ == '__main__':
    tt = time.time()
    ref_img_path = ""
    target_img_path = ""
    save_dir = "./ImageAligmnet"

    ImageAlignment = ImageAlignment()
    ImageAlignment.align_images(ref_img_path, target_img_path, save_dir, method="loftr")
    print(f"Elapsed time: {time.time() - tt}")
