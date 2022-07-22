
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import glob
import datetime
import hashlib

from sklearn.model_selection import train_test_split
from PIL import Image

import sys
sys.path.append('../')

from ML_Ops_Pipeline.pipeline_input import pipeline_data_visualizer, pipeline_dataset_interpreter, pipeline_ensembler, pipeline_model, pipeline_input, pipeline_streamlit_visualizer
from ML_Ops_Pipeline.constants import *

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



class seg_kitti(pipeline_dataset_interpreter):

	def generate_data(self, dataset):
		dataset_train, dataset_test = train_test_split(dataset, test_size=0.1, random_state=1)
		x_train, y_train = dataset_train[['image_2']], dataset_train[['semantic_rgb', 'color_map']]
		x_test, y_test = dataset_test[['image_2']], dataset_test[['semantic_rgb', 'color_map']]
		return x_train, y_train, x_test, y_test


	def load(self) -> None:

		dataset = {}
		# Mapping: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
		# TODO: Generalize mapping
		color_map = {
			'vehicle': [
				(142, 0, 0),	# Car
				(70, 0, 0),		# Truck
				(100, 60, 0),	# Bus
				(90, 0, 0),		# Caravan
				(110, 0, 0),	# Trailer
				(230, 0, 0),	# Motorcycle
				(32, 11, 119),	# Bicycle
				(0, 0, 255)		# Rider
			]
		}

		for mode in ('testing', 'training'):
			print("Loading seg_kitti from:", self.input_dir)
			image_2_folder = os.path.join(self.input_dir, mode, "image_2")
			instance_folder = os.path.join(self.input_dir, mode, "instance")
			semantic_folder = os.path.join(self.input_dir, mode, "semantic")
			semantic_rgb_folder = os.path.join(self.input_dir, mode, "semantic_rgb")

			assert os.path.exists(image_2_folder), image_2_folder
			if mode=='training':
				assert os.path.exists(instance_folder), instance_folder
				assert os.path.exists(semantic_folder), semantic_folder
				assert os.path.exists(semantic_rgb_folder), semantic_rgb_folder

			if mode=='training':
				dataset[mode] = {
					# 'image_2': [], 'instance': [], 'semantic': [], 'semantic_rgb': []
					'image_2': [], 'semantic_rgb': [], 'color_map': []
				}
			else:
				dataset[mode] = {
					'image_2': [], 'color_map': []
				}
			image_2_files_list = sorted(glob.glob(os.path.join(image_2_folder, "*.png")))
			files_list = list(map(lambda x: x.split("/")[-1].split(".png")[:-1][0], image_2_files_list))
			#print(files_list)
			print("seg_kitti: load", mode)
			for f in tqdm(files_list):
				image_2_path = os.path.join(image_2_folder, f+".png")
				instance_path = os.path.join(instance_folder, f+".png")
				semantic_path = os.path.join(semantic_folder, f+".png")
				semantic_rgb_path = os.path.join(semantic_rgb_folder, f+".png")
					
				assert os.path.exists(image_2_path), image_2_path
				if mode=="training":
					assert os.path.exists(instance_path), instance_path
					assert os.path.exists(semantic_path), semantic_path
					assert os.path.exists(semantic_rgb_path), semantic_rgb_path
				
				dataset[mode]['image_2'] += [image_2_path]
				dataset[mode]['color_map'] += [color_map]
				if mode=='training':
					dataset[mode]['semantic_rgb'] += [semantic_rgb_path]
				
			dataset[mode] = pd.DataFrame(dataset[mode])
		
		# Discard the KITTI test files, because no ground truth
		x_train, y_train, x_test, y_test = self.generate_data(dataset['training'])
		self.dataset = {
			'train': {
				'x': x_train,
				'y': y_train
			},
			'test': {
				'x': x_test,
				'y': y_test
			}
		}

class interp_airsim(pipeline_dataset_interpreter):
	NUM_CAMS = 0
	mode_name = {
        0: 'Scene', 
        1: 'DepthPlanar', 
        2: 'DepthPerspective',
        3: 'DepthVis', 
        4: 'DisparityNormalized',
        5: 'Segmentation',
        6: 'SurfaceNormals',
        7: 'Infrared'
    }
	cam_name = {
        '0': 'FrontL'
    }

	def generate_data(self, dataset):
		dataset_train, dataset_test = train_test_split(dataset, test_size=0.1, random_state=1)
		x_train, y_train = dataset_train[['image_2']], dataset_train[['semantic_rgb', 'color_map']]
		x_test, y_test = dataset_test[['image_2']], dataset_test[['semantic_rgb', 'color_map']]
		return x_train, y_train, x_test, y_test

	def load(self) -> None:
		super().load()
		airsim_txt=os.path.join(self.input_dir, 'airsim_rec.txt')
		images_folder=os.path.join(self.input_dir, 'images')
		
		assert os.path.exists(airsim_txt)
		assert os.path.exists(images_folder)

		df = pd.read_csv(airsim_txt, sep='\t')
		df.set_index('TimeStamp')

		self.NUM_CAMS = len(df["ImageFile"].iloc[0].split(";"))

		self.cam_name = {
			'0': 'FrontL',
			str(self.NUM_CAMS): 'FrontR'
		}
		for i in range(1, self.NUM_CAMS):
			self.cam_name[str(i)] = 'C' + str(i)

		x_vals = dict()
		for col in df.columns:
			if col!="ImageFile":
				x_vals[col] = []
		x_vals["ImageFile"] = []

		dataset = {
			'image_2': [], 'semantic_rgb': [], 'color_map': []
		}

		color_map = {
			'vehicle': [
				(182, 145, 184),	# Car
				(135, 150, 152),	# Car again
			]
		}

		df.sort_values('TimeStamp', ascending=False)

		for index, row in df.iterrows():
			files = row["ImageFile"].split(";")
			input_images = []
			gt_images = []
			image_2 = None
			semantic_rgb = None
			for f in files:
				cam_id, img_format = f.split("_")[2:4]

				if cam_id=='0':
					if img_format=='0':
						image_2 = os.path.join(images_folder, f)
						assert os.path.exists(image_2)
					elif img_format=='5':
						semantic_rgb = os.path.join(images_folder, f)
						assert os.path.exists(semantic_rgb)
						
			if type(image_2)!=type(None) and type(semantic_rgb)!=type(None):
				dataset['image_2'] += [image_2]
				dataset['semantic_rgb'] += [semantic_rgb]
				dataset['color_map'] += [color_map]
				
			for col in df.columns:
				if col!="ImageFile":
					x_vals[col] += [row[col]]
			x_vals["ImageFile"] += [";".join(input_images)]
		
		dataset = pd.DataFrame(dataset)
		
		x_train, y_train = dataset[['image_2']], dataset[['semantic_rgb', 'color_map']]
		self.dataset = {
			'train': {
				'x': x_train,
				'y': y_train
			},
			'test': {
				'x': x_train,
				'y': y_train
				# 'x': x_test,
				# 'y': y_test
			}
		}

class seg_cfg:
	config_path = os.path.expanduser("~/detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml")

	def get_cfg(self):
		self.cfg = get_cfg()
		#self.cfg.merge_from_file(os.path.expanduser("~/detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml"))
		self.cfg.merge_from_file(self.config_path)
		self.cfg.merge_from_list([])
		# Set score_threshold for builtin models
		self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
		self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

		self.metadata = MetadataCatalog.get(
			self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused"
		)
		self.instance_mode = ColorMode.IMAGE

		self.cpu_device = torch.device("cpu")

		self.cfg.freeze()

class seg_data_visualizer(pipeline_data_visualizer):

	def overlay_instances(
		self,
		img,
		boxes=None,
		labels=None,
		masks=None,
		keypoints=None,
		assigned_colors=None,
		alpha=0.5,
	):
		"""
		Args:
			boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
				or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
				or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
				for the N objects in a single image,
			labels (list[str]): the text to be displayed for each instance.
			masks (masks-like object): Supported types are:

				* :class:`detectron2.structures.PolygonMasks`,
				  :class:`detectron2.structures.BitMasks`.
				* list[list[ndarray]]: contains the segmentation masks for all objects in one image.
				  The first level of the list corresponds to individual instances. The second
				  level to all the polygon that compose the instance, and the third level
				  to the polygon coordinates. The third level should have the format of
				  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
				* list[ndarray]: each ndarray is a binary mask of shape (H, W).
				* list[dict]: each dict is a COCO-style RLE.
			keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
				where the N is the number of instances and K is the number of keypoints.
				The last dimension corresponds to (x, y, visibility or score).
			assigned_colors (list[matplotlib.colors]): a list of colors, where each color
				corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
				for full list of formats that the colors are accepted in.
		Returns:
			output (np.array): image object with visualizations.
		"""
		N = len(boxes)
		assert N==len(labels)
		assert N==len(masks)
		output = img
		alpha = 0.6
		beta = (1.0 - alpha)

		final_masked_color = np.zeros_like(img)
		for index, mask in enumerate(masks):
			assert mask.shape == img.shape[:2]
			mask = mask.astype(np.uint8)
			label_hash = int(hashlib.sha1(str(labels[index]).encode("utf-8")).hexdigest(), 16) % (2**32)
			color = random_color(seed=label_hash)
			masked_color = np.ones_like(img) * 255
			masked_color[:,:] = color
			masked_color = cv2.bitwise_or(masked_color, masked_color, mask=mask)
			
			final_masked_color = final_masked_color + masked_color
		output = cv2.addWeighted(output, alpha, final_masked_color, beta, 0.0)
		return output

class iou_vis(seg_data_visualizer):
	def visualize(self, x, y, results, preds, save_dir) -> None:
		for index, row in tqdm(preds.iterrows(), total=preds.shape[0]):
		# 	# TODO produce model predictions
			# img = cv2.imread(row['image_2'])
			#print(row)
			img = read_image(row['image_2'])
			semantic_rgb = read_image(y[x['image_2']==row['image_2']]['semantic_rgb'].iloc[0])

			boxes = row['boxes']
			scores = row['scores']
			classes = row['classes']
			keypoints = row['keypoints']
			masks = row['masks']

			# visualizer = Visualizer(img, self.metadata, instance_mode=self.instance_mode)
			# vis_output = visualizer.draw_instance_predictions(predictions=instances)
			
			save_path = os.path.join(save_dir, str(datetime.datetime.now()).replace(" ", "_") + ".png")
			vis_output_img = self.overlay_instances(
				img,
				masks=masks,
				boxes=boxes,
				labels=classes,
				keypoints=keypoints,
				assigned_colors=None,
				alpha=0.5,
			)

			cv2.imwrite(save_path, vis_output_img)

class video_vis(seg_data_visualizer):
	def visualize(self, x, y, results, preds, save_dir) -> None:
		writer = None
		print(save_dir)


		for index, row in tqdm(preds.iterrows(), total=preds.shape[0]):
		# 	# TODO produce model predictions
			# img = cv2.imread(row['image_2'])
			img = read_image(row['image_2'])
			semantic_rgb = read_image(y[x['image_2']==row['image_2']]['semantic_rgb'].iloc[0])

			boxes = row['boxes']
			scores = row['scores']
			classes = row['classes']
			keypoints = row['keypoints']
			masks = row['masks']

			# visualizer = Visualizer(img, self.metadata, instance_mode=self.instance_mode)
			# vis_output = visualizer.draw_instance_predictions(predictions=instances)
			
			vis_output_img = self.overlay_instances(
				img,
				masks=masks,
				boxes=boxes,
				labels=classes,
				keypoints=keypoints,
				assigned_colors=None,
				alpha=0.5,
			)
			# img.shape (270, 480, 3)
			semantic_rgb = cv2.resize(semantic_rgb, (img.shape[1], img.shape[0])) 

			# print("semantic_rgb.dtype", semantic_rgb.dtype)
			# print("img.dtype", img.dtype)
			# print("semantic_rgb.shape", semantic_rgb.shape)
			# print("img.shape", img.shape)
			
			vis_output_img = cv2.vconcat([vis_output_img, semantic_rgb])
			# cv2.imshow('vis_output_img', vis_output_img)
			# cv2.waitKey(1)

			if writer is None:
				size = vis_output_img.shape[:2]
				output_path = os.path.join(save_dir, 'output.mp4')
				if os.path.exists(output_path):
					os.remove(output_path)
				writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (size[1],size[0]))
				#writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), 5, (size[1],size[0]))
			writer.write(vis_output_img)
			# print(output_path)
			#cv2.imwrite(save_path, vis_output_img)
	

class seg_evaluator:

	def evaluate(self, x, y):
		preds = self.predict(x)
		# TODO: write a common evaluation script that is common to all models
		# Note: Optionally, you can give model specific implementations for the evaluation logic
		#		by overloading the evaluate(self, x, y) method in the model class
		
		Dice_coeff_list = []
		iou_list = []
		iou_thresh = 0.5
		yolo_metrics = {
			'tp':0, 	# iou>thresh
			'fp': 0, 	# 0<iou<thresh
			'fn':0		# iou==0	
		}

		dat_test = x.join(y)
		print("Evaluate")
		for index, row in tqdm(dat_test.iterrows(), total=dat_test.shape[0]):
			img = cv2.imread(row['image_2'])
			semantic_rgb = cv2.imread(row['semantic_rgb'])
			semantic_rgb = cv2.resize(semantic_rgb, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
			color_map = row['color_map']

			gt_mask = np.full(semantic_rgb.shape[:2], False, dtype=bool)
			for class_name in color_map:
				for color in color_map[class_name]:
					gt_mask += (semantic_rgb == color).all(-1)

			pred_mask = np.full(gt_mask.shape, False, dtype=bool)

			model_pred = preds[preds["image_2"]==row['image_2']]
			pred_mask_list = model_pred["masks"].iloc[0]
			pred_classes_list = model_pred["classes"].iloc[0]
			model_output_classes = preds['model_output_classes'].iloc[0]
			pred_classes_list_final = []
			for ind, msk in enumerate(pred_mask_list):
				pred_class = pred_classes_list[ind]
				#if pred_class in (3, 4, 6, 8, ):
				for class_name in color_map:
					if pred_class in model_output_classes[class_name]:
						pred_mask = np.logical_or(pred_mask, msk)
						pred_classes_list_final.append(msk)
			
			pred_classes_list = pred_classes_list_final

			intersection = np.logical_and(gt_mask, pred_mask)
			union = np.logical_or(gt_mask, pred_mask)
			IOU = np.sum(intersection) / np.sum(union)
			Dice_coeff = 2 * np.sum(intersection) / (np.sum(gt_mask) + np.sum(pred_mask))
			Dice_coeff_list.append(Dice_coeff)
			iou_list.append(IOU)
			if IOU==0:
				yolo_metrics['fn'] += 1
			else:
				if IOU>iou_thresh:
					yolo_metrics['tp'] += 1
				else:
					yolo_metrics['fp'] += 1

			# mask_vis = pred_mask.astype(np.uint8) * 255
			# cv2.imshow('mask', mask_vis)
			# cv2.waitKey(1)

		prec = np.float64(yolo_metrics['tp']) / float(yolo_metrics['tp'] + yolo_metrics['fp'])
		recall = np.float64(yolo_metrics['tp']) / float(yolo_metrics['tp'] + yolo_metrics['fn'])
		f1_score = np.float64(2*prec*recall)/(prec+recall)
		iou_avg = np.float64(sum(iou_list)) / len(iou_list)
		Dice_coeff_avg = sum(Dice_coeff_list) / len(Dice_coeff_list)
		results = {
			'prec': prec,
			'recall': recall,
			'f1_score': f1_score,
			'Dice_coeff_list': Dice_coeff_avg,
			'iou_avg': iou_avg,
			'confusion': yolo_metrics
		}
		return results, preds

class detectron_base_model(seg_evaluator, pipeline_model, seg_cfg):

	def load(self):
		self.get_cfg()
		self.predictor = DefaultPredictor(self.cfg)
		
	def train(self, x, y) -> np.array:
		#preds = self.predict(x)

		# TODO: Train the model		
		results, preds = self.evaluate(x,y)

		return results, preds

	def predict(self, x) -> np.array:
		# Runs prediction on list of values x of length n
		# Returns a list of values of length n

		predict_results = {
			'image_2': [],
			'boxes': [],
			'scores': [],
			'classes': [],
			'keypoints': [],
			'masks': [],
			'model_output_classes': []
		}
		for index, row in tqdm(x.iterrows(), total=x.shape[0]):
		# 	# TODO produce model predictions
			# img = cv2.imread(row['image_2'])
			img = read_image(row['image_2'])
			
			outputs = self.predictor(img)
			
			instances = outputs["instances"].to(self.cpu_device) # detectron2.structures.instances.Instances
			predictions = instances
			boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
			scores = predictions.scores if predictions.has("scores") else None
			classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
			keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

			if predictions.has("pred_masks"):
				masks = np.asarray(predictions.pred_masks)
			else:
				masks = None


			predict_results['image_2'] += [row['image_2'], ]
			predict_results['boxes'] += [boxes, ]
			predict_results['scores'] += [scores, ]
			predict_results['classes'] += [classes, ]
			predict_results['keypoints'] += [keypoints, ]
			predict_results['masks'] += [masks, ]
			predict_results['model_output_classes'] += [self.model_output_classes, ]


		predict_results = pd.DataFrame(predict_results)
		return predict_results


class mask_rcnn(detectron_base_model):
	config_path = os.path.expanduser("~/detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml")

	model_output_classes = {
		'vehicle':  (2, 3, 5, 7, )
	}

# import sys
# sys.path.append(os.path.expanduser("~/detectron2/projects/PointRend/"))
# from point_rend import add_pointrend_config

# class pointrend(detectron_base_model):
# 	#config_path = os.path.expanduser("~/detectron2/projects/PointRend/configs/InstanceSegmentation/Base-Implicit-PointRend.yaml")
# 	config_path = os.path.expanduser("~/detectron2/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml")
# 	model_output_classes = {
# 		'vehicle':  (26,51,27,52,15,67)
# 	}
# 	#config_path = os.path.expanduser("~/detectron2/projects/PointRend/configs/SemanticSegmentation/Base-PointRend-Semantic-FPN.yaml")
# 	def get_cfg(self):
# 		self.cfg = get_cfg()

# 		add_pointrend_config(self.cfg)
# 		self.cfg.merge_from_file(self.config_path)
# 		self.cfg.merge_from_list([])
# 		# Set score_threshold for builtin models
# 		self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
# 		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# 		self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

# 		self.metadata = MetadataCatalog.get(
# 			self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused"
# 		)
# 		self.instance_mode = ColorMode.IMAGE

# 		self.cpu_device = torch.device("cpu")

# 		self.cfg.freeze()

class seg_pipeline_ensembler_1(seg_evaluator, pipeline_ensembler):

	def predict(self, x: dict) -> np.array:
		model_names = list(x.keys())
		predict_results = {
			'result': [], 'another_result': []
		}
		for i in tqdm(x):
			for mod_name in model_names:
				preds = x[mod_name]
			# TODO produce ensebled results based on all the model's predictions
			predict_results["result"] += ["Some Ensembled Result"]
			predict_results["another_result"] += ["Another Ensembled Result"]
				
		predict_results = pd.DataFrame(predict_results)
		return predict_results


	def train(self, x, y):
		#preds = self.predict(x)
		# TODO: train ensemble model
		results, preds =self.evaluate(x,y)

		return results, preds

class streamlit_viz(pipeline_streamlit_visualizer):

	def visualize(self):
		self.load_data()

		self.st.markdown("# Testing Results")
		self.st.write(self.testing_results)
		#self.st.write(self.testing_predictions)

		self.st.markdown("# Training Results")
		self.st.write(self.training_results)

		self.st.markdown("# Visuals")
		training_data = False

		if training_data:
			preds = self.training_predictions
			x = self.dat['train']['x']
			y = self.dat['train']['y']
		else:
			preds = self.testing_predictions
			x = self.dat['test']['x']
			y = self.dat['test']['y']
		
		iou_list = []
		iou_thresh = 0.5
		yolo_metrics = {
			'tp':0, 	# iou>thresh
			'fp': 0, 	# 0<iou<thresh
			'fn':0		# iou==0	
		}

		iou_thresh_min, iou_thresh_max = self.st.sidebar.slider('IOU Threshold', 0, 100, [50,100])
		iou_thresh_min, iou_thresh_max = iou_thresh_min/100.0, iou_thresh_max/100.0
		
		dat_test = x.join(y)
		for index, row in dat_test.iterrows():
			img = cv2.imread(row['image_2'])
			semantic_rgb = cv2.imread(row['semantic_rgb'])
			semantic_rgb = cv2.resize(semantic_rgb, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
			color_map = row['color_map']
			

			gt_mask = np.full(semantic_rgb.shape[:2], False, dtype=bool)
			for class_name in color_map:
				for color in color_map[class_name]:
					gt_mask += (semantic_rgb == color).all(-1)

			pred_mask = np.full(gt_mask.shape, False, dtype=bool)

			model_pred = preds[preds["image_2"]==row['image_2']]
			pred_mask_list = model_pred["masks"].iloc[0]
			pred_classes_list = model_pred["classes"].iloc[0]
			model_output_classes = preds['model_output_classes'].iloc[0]
			pred_classes_list_final = []
			for ind, msk in enumerate(pred_mask_list):
				pred_class = pred_classes_list[ind]
				#if pred_class in (3, 4, 6, 8, ):
				for class_name in color_map:
					if pred_class in model_output_classes[class_name]:
						pred_mask = np.logical_or(pred_mask, msk)
						pred_classes_list_final.append(msk)
			
			intersection = np.logical_and(gt_mask, pred_mask)
			union = np.logical_or(gt_mask, pred_mask)
			IOU = 1.0 * np.sum(intersection) / np.sum(union)
			Dice_coeff = 2 * np.sum(intersection) / (np.sum(gt_mask) + np.sum(pred_mask))
			iou_list.append(IOU)
			if IOU==0:
				yolo_metrics['fn'] += 1
			else:
				if IOU>iou_thresh:
					yolo_metrics['tp'] += 1
				else:
					yolo_metrics['fp'] += 1

			if iou_thresh_min <= IOU and IOU <= iou_thresh_max:
				img = read_image(row['image_2'])
				semantic_rgb = read_image(y[x['image_2']==row['image_2']]['semantic_rgb'].iloc[0])

				gt_masked_color = np.ones_like(img)
				gt_masked_color[:,:] = (0,255,0)
				gt_masked_color = cv2.bitwise_or(gt_masked_color, gt_masked_color, mask=gt_mask.astype(np.uint8))

				final_masked_color = np.zeros_like(img)
				for index, mask in enumerate(pred_classes_list_final):
					mask = mask.astype(np.uint8)
					#label_hash = int(hashlib.sha1(str(labels[index]).encode("utf-8")).hexdigest(), 16) % (2**32)
					#color = random_color(seed=label_hash)
					masked_color = np.ones_like(img) * 255
					masked_color[:,:] = (255,0,0)
					masked_color = cv2.bitwise_or(masked_color, masked_color, mask=mask)
					final_masked_color = final_masked_color + masked_color
				
				
				final_masked_color = final_masked_color + gt_masked_color

				vis_output_img = cv2.addWeighted(img, 0.75, final_masked_color, 0.25, 0.0)

				vis_output_img = cv2.putText(vis_output_img, 'IOU='+str(round(IOU,4)), (25,50), 
					cv2.FONT_HERSHEY_SIMPLEX, 
					2, 
					(255, 0, 0), 
					2, cv2.LINE_AA
				)

				#vis_output_img = cv2.cvtColor(vis_output_img, cv2.COLOR_BGR2RGB)

				self.st.image(vis_output_img)

	def overlay_instances(
		self,
		img,
		boxes=None,
		labels=None,
		masks=None,
		keypoints=None,
		assigned_colors=None,
		color=(255,0,0),
		alpha=0.5,
	):
		N = len(boxes)
		#assert N==len(labels)
		#assert N==len(masks)
		output = img
		#alpha = 0.6
		beta = (1.0 - alpha)

		final_masked_color = np.zeros_like(img)
		for index, mask in enumerate(masks):
			assert mask.shape == img.shape[:2]
			mask = mask.astype(np.uint8)
			#label_hash = int(hashlib.sha1(str(labels[index]).encode("utf-8")).hexdigest(), 16) % (2**32)
			#color = random_color(seed=label_hash)
			masked_color = np.ones_like(img) * 255
			masked_color[:,:] = color
			masked_color = cv2.bitwise_or(masked_color, masked_color, mask=mask)
			final_masked_color = final_masked_color + masked_color
		output = cv2.addWeighted(output, alpha, final_masked_color, beta, 0.0)
		return output

seg_input = pipeline_input("seg", 
	{
		'seg_kitti': seg_kitti,
		'interp_airsim':interp_airsim
	}, {
		#'detectron_base_model': detectron_base_model,
		'mask_rcnn': mask_rcnn,
		# 'pointrend': pointrend
	}, {
		#'seg_pipeline_ensembler_1': seg_pipeline_ensembler_1
	}, {
		'iou_vis': iou_vis,
		'video_vis': video_vis
	}, p_pipeline_streamlit_visualizer=streamlit_viz)

# Write the pipeline object to exported_pipeline
exported_pipeline = seg_input

############################

def read_image(file_name, format=None):
	"""
	Read an image into the given format.
	Will apply rotation and flipping if the image has such exif information.

	Args:
		file_name (str): image file path
		format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

	Returns:
		image (np.ndarray):
			an HWC image in the given format, which is 0-255, uint8 for
			supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
	"""
	
	image = Image.open(file_name)
	image.load()
	# work around this bug: https://github.com/python-pillow/Pillow/issues/3973
	image = _apply_exif_orientation(image)
	return convert_PIL_to_numpy(image, format)

_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_SMALL_OBJECT_AREA_THRESH = 1000

def convert_PIL_to_numpy(image, format):
	"""
	Convert PIL image to numpy array of target format.

	Args:
		image (PIL.Image): a PIL image
		format (str): the format of output image

	Returns:
		(np.ndarray): also see `read_image`
	"""
	if format is not None:
		# PIL only supports RGB, so convert to RGB and flip channels over below
		conversion_format = format
		if format in ["BGR", "YUV-BT.601"]:
			conversion_format = "RGB"
		image = image.convert(conversion_format)
	image = np.asarray(image)
	# PIL squeezes out the channel dimension for "L", so make it HWC
	if format == "L":
		image = np.expand_dims(image, -1)

	# handle formats not supported by PIL
	elif format == "BGR":
		# flip channels if needed
		image = image[:, :, ::-1]
	elif format == "YUV-BT.601":
		image = image / 255.0
		image = np.dot(image, np.array(_M_RGB2YUV).T)

	return image

_EXIF_ORIENT = 274  # exif 'Orientation' tag

def _apply_exif_orientation(image):
	"""
	Applies the exif orientation correctly.

	This code exists per the bug:
	  https://github.com/python-pillow/Pillow/issues/3973
	with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
	various methods, especially `tobytes`

	Function based on:
	  https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
	  https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

	Args:
		image (PIL.Image): a PIL image

	Returns:
		(PIL.Image): the PIL image with exif orientation applied, if applicable
	"""
	if not hasattr(image, "getexif"):
		return image

	try:
		exif = image.getexif()
	except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
		exif = None

	if exif is None:
		return image

	orientation = exif.get(_EXIF_ORIENT)

	method = {
		2: Image.FLIP_LEFT_RIGHT,
		3: Image.ROTATE_180,
		4: Image.FLIP_TOP_BOTTOM,
		5: Image.TRANSPOSE,
		6: Image.ROTATE_270,
		7: Image.TRANSVERSE,
		8: Image.ROTATE_90,
	}.get(orientation)

	if method is not None:
		return image.transpose(method)
	return image

# fmt: off
# RGB:
_COLORS = np.array(
	[
		0.000, 0.447, 0.741,
		0.850, 0.325, 0.098,
		0.929, 0.694, 0.125,
		0.494, 0.184, 0.556,
		0.466, 0.674, 0.188,
		0.301, 0.745, 0.933,
		0.635, 0.078, 0.184,
		0.300, 0.300, 0.300,
		0.600, 0.600, 0.600,
		1.000, 0.000, 0.000,
		1.000, 0.500, 0.000,
		0.749, 0.749, 0.000,
		0.000, 1.000, 0.000,
		0.000, 0.000, 1.000,
		0.667, 0.000, 1.000,
		0.333, 0.333, 0.000,
		0.333, 0.667, 0.000,
		0.333, 1.000, 0.000,
		0.667, 0.333, 0.000,
		0.667, 0.667, 0.000,
		0.667, 1.000, 0.000,
		1.000, 0.333, 0.000,
		1.000, 0.667, 0.000,
		1.000, 1.000, 0.000,
		0.000, 0.333, 0.500,
		0.000, 0.667, 0.500,
		0.000, 1.000, 0.500,
		0.333, 0.000, 0.500,
		0.333, 0.333, 0.500,
		0.333, 0.667, 0.500,
		0.333, 1.000, 0.500,
		0.667, 0.000, 0.500,
		0.667, 0.333, 0.500,
		0.667, 0.667, 0.500,
		0.667, 1.000, 0.500,
		1.000, 0.000, 0.500,
		1.000, 0.333, 0.500,
		1.000, 0.667, 0.500,
		1.000, 1.000, 0.500,
		0.000, 0.333, 1.000,
		0.000, 0.667, 1.000,
		0.000, 1.000, 1.000,
		0.333, 0.000, 1.000,
		0.333, 0.333, 1.000,
		0.333, 0.667, 1.000,
		0.333, 1.000, 1.000,
		0.667, 0.000, 1.000,
		0.667, 0.333, 1.000,
		0.667, 0.667, 1.000,
		0.667, 1.000, 1.000,
		1.000, 0.000, 1.000,
		1.000, 0.333, 1.000,
		1.000, 0.667, 1.000,
		0.333, 0.000, 0.000,
		0.500, 0.000, 0.000,
		0.667, 0.000, 0.000,
		0.833, 0.000, 0.000,
		1.000, 0.000, 0.000,
		0.000, 0.167, 0.000,
		0.000, 0.333, 0.000,
		0.000, 0.500, 0.000,
		0.000, 0.667, 0.000,
		0.000, 0.833, 0.000,
		0.000, 1.000, 0.000,
		0.000, 0.000, 0.167,
		0.000, 0.000, 0.333,
		0.000, 0.000, 0.500,
		0.000, 0.000, 0.667,
		0.000, 0.000, 0.833,
		0.000, 0.000, 1.000,
		0.000, 0.000, 0.000,
		0.143, 0.143, 0.143,
		0.857, 0.857, 0.857,
		1.000, 1.000, 1.000
	]
).astype(np.float32).reshape(-1, 3)
# fmt: on


def colormap(rgb=False, maximum=255):
	"""
	Args:
		rgb (bool): whether to return RGB colors or BGR colors.
		maximum (int): either 255 or 1

	Returns:
		ndarray: a float32 array of Nx3 colors, in range [0, 255] or [0, 1]
	"""
	assert maximum in [255, 1], maximum
	c = _COLORS * maximum
	if not rgb:
		c = c[:, ::-1]
	return c



def random_color(rgb=False, maximum=255, seed=None):
	"""
	Args:
		rgb (bool): whether to return RGB colors or BGR colors.
		maximum (int): either 255 or 1

	Returns:
		ndarray: a vector of 3 numbers
	"""
	if seed:
		np.random.seed(seed)
	idx = np.random.randint(0, len(_COLORS))
	ret = _COLORS[idx] * maximum
	if not rgb:
		ret = ret[::-1]
	return ret



def random_colors(N, rgb=False, maximum=255):
	"""
	Args:
		N (int): number of unique colors needed
		rgb (bool): whether to return RGB colors or BGR colors.
		maximum (int): either 255 or 1

	Returns:
		ndarray: a list of random_color
	"""
	indices = random.sample(range(len(_COLORS)), N)
	ret = [_COLORS[i] * maximum for i in indices]
	if not rgb:
		ret = [x[::-1] for x in ret]
	return ret
