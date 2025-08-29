import os
import sys
import shutil
import random
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd

from PIL import Image
from pathlib import Path
from affine import Affine
from rasterio import features
from shapely.geometry import box
from rasterio.windows import Window
from rasterio.features import rasterize

class GenerateDataset:
    def __init__(self):
        pass
    
    def merge_classes_to_all(self, dir_path, classes=["도로", "과수원"], target_crs="EPSG:5186"):
        """
        Args:
            dir_path (str): Dir path to shp
            classes (list): List of classes to merge
            target_crs (str): 국가공식 좌표계 / GIS 표준으로 사용
        """
        dir_list = os.listdir(dir_path)
        
        for dir in dir_list:
            gdfs = []
            # classes별 처리
            for idx, cls in enumerate(classes, start=1):
                shp_path = Path(dir_path) / dir / f"{dir}_{cls}.shp"
                if shp_path.exists():
                    gdf = gpd.read_file(shp_path)
                    if gdf.crs != target_crs:
                        gdf = gdf.to_crs(target_crs)
                    gdf["class"] = idx  
                    gdfs.append(gdf)

            if not gdfs:
                print(f"{dir} 안에 대상 클래스 shp가 없습니다.")
                continue

            merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
            out_path = Path(dir_path) / dir / f"{dir}_all.shp"
            merged.to_file(out_path, encoding="utf-8")
            print(f"{out_path} 생성 완료")
    
    def change_mask(self, earlier_shp_dir, later_shp_dir, pre_year, post_year):
        """
        Args:
            earlier_shp_dir (str): Dir path to shp of earlier image 
            later_shp_dir (str): Dir path to shp of later image
            pre_year (str): Year of the earlier image.
            post_year (int): Year of the later image.
        """
        dir_list = os.listdir(earlier_shp_dir)
        cd_shp_dir = os.path.join(earlier_shp_dir, "GT_CD")
        os.makedirs(cd_shp_dir, exist_ok=True)

        for dir in dir_list:
            earlier_shp_path = Path(earlier_shp_dir) / dir / f"{dir}_all.shp"
            post_dir = dir.replace("_{pre_year}_", "_{post_year}_")
            later_shp_path = Path(later_shp_dir) / post_dir / f"{post_dir}_all.shp"
            cd_shp_path = os.path.join(cd_shp_dir, f"{post_dir}_mask.shp")

            self.compare_shp(
                earlier_shp_path=earlier_shp_path,
                later_shp_path=later_shp_path,
                cd_shp_path=cd_shp_path
            )
            
    def compare_shp(earlier_shp_path, later_shp_path, cd_shp_path, target_crs="EPSG:5186"):
        """
        Args:
            earlier_shp_path (str): Path to shp of earlier image
            later_shp_path (str): Path to shp of later image
            cd_shp_path (str): Save shp path
            target_crs (str): 국가공식 좌표계 / GIS 표준으로 사용
        """
        gdf_earlier = gpd.read_file(earlier_shp_path).to_crs(target_crs)
        gdf_later = gpd.read_file(later_shp_path).to_crs(target_crs)
        
        common = gpd.overlay(gdf_earlier, gdf_later, how='intersection')
        added = gpd.overlay(gdf_later, gdf_earlier, how='difference')
        removed = gpd.overlay(gdf_earlier, gdf_later, how='difference')
        
        change_area = gpd.GeoDataFrame(pd.concat([added, removed], ignore_index=True), crs=target_crs)
        change_area.to_file(cd_shp_path,encoding="utf-8")
        print(f"mask file complete: {cd_shp_path}")
        
        return added, removed, change_area
        
    def tile_tif_to_png(self, tif_dir, output_dir, tile_size=512):
        """
        Args:
            tif_dir (str): Dir path to shp
            output_dir (str): Save path
            tile_size (int): Tile size
        """
        os.makedirs(output_dir, exist_ok=True)
        tiff_path = os.listdir(tif_dir)

        for tif_path in tiff_path:
            img = Image.open(Path(tif_dir, tif_path))
            width, height = img.size
            x_tiles = (width + tile_size - 1) // tile_size
            y_tiles = (height + tile_size - 1) // tile_size

            tile_count = 0
            
            for i in range(x_tiles):
                for j in range(y_tiles):
                    left = i * tile_size
                    upper = j * tile_size
                    right = min((i + 1) * tile_size, width)
                    lower = min((j + 1) * tile_size, height)
                    tile = img.crop((left, upper, right, lower))

                    # filename_x_y format
                    tile_filename = os.path.basename(tif_path)
                    tile_filename = f"{tile_filename}_{left}_{upper}.png"
                    print(f"Saving tile: {tile_filename}")
                    tile.save(os.path.join(output_dir, tile_filename), format="PNG")
                    tile_count += 1

            print(f"{tile_count} saved in {os.path.join(output_dir, tile_filename)}")

    def tile_shapefile(self, shp_path, output_dir, tile_size=512):
        """
        Args:
            shp_path (str): Dir path to shp
            output_dir (str): Save path
            tile_size (int): Tile image size
        """
        os.makedirs(output_dir, exist_ok=True)
        gdf = gpd.read_file(shp_path)
        minx, miny, maxx, maxy = gdf.total_bounds

        x_tiles = int((maxx - minx) // tile_size) + 1
        y_tiles = int((maxy - miny) // tile_size) + 1

        tile_count = 0
        for i in range(x_tiles):
            for j in range(y_tiles):
                x1 = minx + i * tile_size
                y1 = miny + j * tile_size
                x2 = x1 + tile_size
                y2 = y1 + tile_size
                tile_bbox = box(x1, y1, x2, y2)
                tile_gdf = gpd.clip(gdf, tile_bbox)

                if not tile_gdf.empty:
                    out_path = os.path.join(output_dir, f"tile_{i}_{j}.shp")
                    tile_gdf.to_file(out_path, encoding="utf-8")
                    tile_count += 1

        print(f"✅ {tile_count} tiles saved in {output_dir}")

    def tile_tif_and_shp_to_png(self, tif_dir, shp_dir, output_base, type = 'label', tile_size=512):
        """
        Args:
            tif_dir (str): Dir path to tif
            shp_dir (str): Dir path to shp
            output_base (str): Save path
            type (str): Label or binary / label -> segmentation , binary -> mask
            tile_size (int): Tile image size
        """
        output_tif_dir = os.path.join(output_base,"images")
        output_mask_dir = os.path.join(output_base,"labels")
        os.makedirs(output_tif_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        tif_list = os.listdir(tif_dir)

        for tif_path in tif_list:
            shp_path = tif_path.replace(".tif", ".shp")
            gdf = gpd.read_file(os.path.join(shp_dir, shp_path))
            tif_path = os.path.join(tif_dir,tif_path)\
            
            with rasterio.open(tif_path) as src:
                height, width = src.height, src.width
                transform = src.transform

                for i in range(0, height, tile_size):
                    for j in range(0, width, tile_size):
                        window = Window(j, i, tile_size, tile_size)
                        transform_window = src.window_transform(window)

                        img = src.read(window=window)
                        if img.shape[1] != tile_size or img.shape[2] != tile_size:
                            continue

                        bounds = src.window_bounds(window)
                        tile_bbox = box(*bounds)

                        clipped = gdf[gdf.intersects(tile_bbox)]
                        if clipped.empty:
                            continue
                        
                        img_rgb = np.moveaxis(img[:3], 0, -1)  # [H, W, C]
                        filename = os.path.basename(shp_path).split('.')[0]
                        tif_out = os.path.join(output_tif_dir, f"{filename}_{i}_{j}.png")
                        mask_out = os.path.join(output_mask_dir, f"{filename}_{i}_{j}.png")
                        Image.fromarray(img_rgb).save(tif_out)
                        
                        if type == 'label':
                            shapes = [(geom, value) for geom, value in zip(clipped.geometry, clipped['class'])]
                            mask = rasterize(
                                shapes,
                                out_shape=(tile_size, tile_size),
                                transform=transform_window,
                                fill=0,
                                dtype=np.uint8
                            )
                            
                            Image.fromarray(mask).save(mask_out)
                            print(f"label Saved: {tif_out}, {mask_out}")
                            
                        elif type == 'binary':
                            shapes = [(geom, 1) for geom in clipped.geometry]
                            mask = rasterize(
                                shapes,
                                out_shape=(tile_size, tile_size),
                                transform=transform_window,
                                fill=0,
                                dtype=np.uint8
                            )

                            Image.fromarray(mask*255).save(mask_out)
                            print(f"binary Saved: {tif_out}, {mask_out}")
                            
        print("tiling file complete")

    def rename_files(self, dir1, pre, post):
        """
        Args:
            dir1 (str): Dir path
            pre (str): Pre word
            post (str): Post word
        """
        for i, filename in enumerate(os.listdir(dir1)):
            pre_path = os.path.join(dir, filename)
            post_path = pre_path.replace(pre, post)
            os.rename(pre_path, post_path)
            print(f"Rename: {pre_path} -> {post_path}")
    
    def move_files(self, src_dir, dst_dir, format):
        """
        Args:
            src_dir (str): Dir path to src
            dst_dir (str): Dir path to dst
            format (str): Format the file to move
        """
        src_dir = Path(src_dir)
        src_list = os.listdir(src_dir)
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        for dir  in src_list:
            file = list(Path(src_dir,dir).rglob(format))[0]
            shutil.copy(str(file), os.path.join(dst_dir, os.path.basename(file)))
            print(f"copy file: {os.path.basename(file)}")

    def delete_unmatched_files(self, dir1, dir2, pre, post):
        """
        Args:
            dir1 (str): Dir1 path
            dir2 (str): Dir2 path
            pre (str): Pre word
            post (str): Post word
        """
        dir1_list = os.listdir(dir1)
        dir2_list = os.listdir(dir2)
        
        transformed_files1 = [f.replace(pre, post) for f in dir1_list]

        set1 = set(transformed_files1)
        set2 = set(dir2_list)

        only_in_dir1 = set1 - set2
        print("dir1 only file:", only_in_dir1)
        print("dir1 only file len :", len(only_in_dir1))

        only_in_dir2 = set2 - set1
        print("dir2 only file:", only_in_dir2)
        print("dir2 only file len :", len(only_in_dir2))
