# -*-coding:utf-8-*-
# Time:2022/4/6 15:51
# Author:ahaqu
# Description:
# E-mail:zhouyuhua_ict@163.com
import logging

import fields as fields
from flask import request, jsonify, g
from flask_restful import Resource, reqparse, fields
from werkzeug.datastructures import FileStorage

from App.apis.api_constant import HTTP_OK
from App.apis.customer_user.utils import login_required
from algorithm.resnet.resnet_two_category_transfer_learn import resnet_two_category_transfer_learn, \
    predict_two_category_by_model

parse_base = reqparse.RequestParser()
parse_base.add_argument("train", type=str, required=True, help="请输入要检测的图片")
parse_base.add_argument("val", type=str, required=True, help="请输入要检测的图片")
parse_base.add_argument("test", type=str, required=True, help="请输入要检测的图片")

transfer_learn_param_fileds = {
    "train_loss": fields.Float,
    "val_loss": fields.Float,
    "val_acc": fields.Float,
    "test_loss": fields.Float,
    "test_acc": fields.Float
}


transfer_learn_fields = {
    "status": fields.Integer,
    "msg": fields.String,
    "data": fields.Nested(transfer_learn_param_fileds)
}

class ResnetTwoCategoryTransferLearnService(Resource):
    def post(self):
        logging.info("ResnetTwoCategoryTransferLearnService ---------- ")
        try:
            args = parse_base.parse_args()
            train = args.get('train')
            val = args.get('val')
            test = args.get('test')
            last_train_loss, last_val_loss, last_val_acc, last_test_loss, last_test_acc, g.model = \
                resnet_two_category_transfer_learn(train, val, test)
            data = {
                "status":HTTP_OK,
                "msg":"二分类迁移学习完成",
                "data":{
                    'train_loss':last_train_loss,
                    'val_loss': last_val_loss,
                    'val_acc':last_val_acc,
                    'test_loss':last_test_loss,
                    'test_acc':last_test_acc
                }
            }
        except Exception as e:
            return jsonify({'err_code': 400})

        # return jsonify({'object_name': HTTP_OK})  # jsonify 确保 response为 json格式
        return jsonify(data)  # jsonify 确保 response为 json格式

parse_base_t = reqparse.RequestParser()
parse_base_t.add_argument("image", type=FileStorage, location='files', help="请输入要检测的图片")
parse_base_t.add_argument("normal", type=str, required=True, help="请输入第一种类别名称")
parse_base_t.add_argument("off_normal", type=str, required=True, help="请输入第二种类别名称")
class ResnetTwoCategory(Resource):
    def post(self):
        args = parse_base_t.parse_args()
        image = args.get("image")
        normal = args.get("normal")
        off_normal = args.get("off_normal")
        value = predict_two_category_by_model(g.model, image, normal, off_normal)

        return jsonify({'value': value})