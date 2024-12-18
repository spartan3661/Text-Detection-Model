import json
import os
from PIL import Image, ImageDraw, ImageFont
import requests
from tqdm import tqdm
from zipfile import ZipFile

COCO_TEXT_ANN = "http://vision.cornell.edu/se3/wp-content/uploads/2019/05/COCO_Text.zip"
COCO_TRAIN_IMAGE = "http://images.cocodataset.org/zips/train2014.zip"
BASE_DIR = "./COCO-Text"
COCO_TEXT_PATH = "./COCO-Text/COCO_Text.json"
IMAGE_DIR = "./COCO-Text/train2014"


def download_files(file_url, output_dir, file_name):
	file_path = os.path.join(output_dir, file_name)
	if os.path.exists(file_path):
		return
	
	try:
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		response = requests.get(file_url, stream=True)
		file_size = int(response.headers['Content-Length'])
		#print(file_size)
		file_name = file_name


		with open (file_path, 'wb') as file:
			with tqdm(total=file_size,
					unit='B',
					unit_scale=True,
					desc=file_name) as pbar:
				for chunk in response.iter_content(chunk_size=8192):
					file.write(chunk)
					pbar.update(len(chunk))

		print(f"{file_name} downloaded.")
	except requests.exceptions.RequestException as e:
		print(e)

def extract_files(file, output_path):
	with ZipFile(file, 'r') as zipfile:
		file_list = zipfile.namelist()
		with tqdm(total=len(file_list), desc='Extracting files', unit='file') as pbar:
			for file_name in file_list:
				zipfile.extract(file_name, path=output_path)
				pbar.update(1)
				

def load_coco_text(coco_text_path):

	with open(coco_text_path, "r") as f:
		return json.load(f)


def filter_images_by_prefix(coco_text, prefix):

	result = []
	for img in coco_text["imgs"].values():
		if img["file_name"].startswith(prefix): # and len(get_annotations_for_image(coco_text, img["id"])) > 0  # filter out non annotated images
			result.append(img)
	return result


def get_annotations_for_image(coco_text, image_id):

	ann_ids = coco_text["imgToAnns"].get(str(image_id), [])
	return [coco_text["anns"][str(ann_id)] for ann_id in ann_ids]


def visualize_annotations(image_path, annotations):

	image = Image.open(image_path)
	draw = ImageDraw.Draw(image)

	try:
		font = ImageFont.truetype("arial.ttf", size=16)
	except IOError:
		font = ImageFont.load_default()

	for ann in annotations:
		bbox = ann["bbox"]
		text = ann.get("utf8_string", "")

		draw.rectangle(
			[(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])],
			outline="red",
			width=3,
		)

		draw.text((bbox[0], bbox[1] - 10), text, fill="blue", font=font)

	image.show()

def main():
	if not os.path.exists(BASE_DIR):
		os.makedirs(BASE_DIR)
	'''
	if not os.path.exists("./COCO-Text/COCO-Text.zip"):
		os.system(f"curl -o {os.path.join(BASE_DIR, 'COCO-Text.zip')} {COCO_TEXT_ANN}")
	'''
		
	#download_files(COCO_TRAIN_IMAGE, BASE_DIR, "train2014.zip")
	extract_files("./COCO-Text/COCO_Text.zip", BASE_DIR)
	extract_files("./COCO-Text/train2014.zip", BASE_DIR)
	coco_text = load_coco_text(COCO_TEXT_PATH)
	train_images = filter_images_by_prefix(coco_text, "COCO_train2014_")
	selected_image = train_images[300]
	
	image_id = selected_image["id"]
	image_file_name = selected_image["file_name"]
	image_path = os.path.normpath(os.path.join(IMAGE_DIR, image_file_name))

	annotations = get_annotations_for_image(coco_text, image_id)
	print(image_id)
	print(len(train_images))

	visualize_annotations(image_path, annotations)
	



if __name__ == "__main__":
	main()