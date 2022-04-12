import json

from flask import jsonify, request
from flask_restful import Resource, reqparse
from werkzeug.datastructures import FileStorage

from App.apis.common.utils import filename_transfer
from algorithm.mask_rcnn.mask_rcnn_instance_segmentation_by_coco import mask_rcnn_instance_segmentation_api

from App.common import logging_settings
import logging

parse_base = reqparse.RequestParser()
parse_base.add_argument("image", type=FileStorage, location="files", help="请输入要分割的图片")

class InstanceSegmentationService(Resource):
    def post(self):
        try:
            args = parse_base.parse_args()
            image = args.get("image")
            file_info = filename_transfer('/segmentation/', image.filename)
            filepath = file_info[0]
            image.save(filepath)

            # image = request.files['image']
            # image.save(image.filename)
            logging.info(f"====== image filepath = {filepath}")
            pred_cls = mask_rcnn_instance_segmentation_api(filepath)
            print(type(pred_cls))
            logging.info(f'pred_cls = {pred_cls}')

            data_json = json.dumps(pred_cls, ensure_ascii=False, indent=2)

            return data_json

        except Exception as e:
            return jsonify({'err_code': 400})