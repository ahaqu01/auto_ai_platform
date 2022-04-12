from flask_restful import Api

from App.apis.ai_service.image_classifier_serivce import  ImageClassifierSerivce
from App.apis.ai_service.mask_rcnn_instance_segmentation_service import InstanceSegmentationService
# from App.apis.ai_service.pytorch_yolov5_service import PytorchYolov5, PytorchYolov5Service
from App.apis.ai_service.resnet_two_category_transfer_learn_service import ResnetTwoCategoryTransferLearnService, \
    ResnetTwoCategory

ai_service_api = Api(prefix='/ai_service')

ai_service_api.add_resource(ImageClassifierSerivce, '/predict/')
# ai_service_api.add_resource(PytorchYolov5, '/yolov5/')
# ai_service_api.add_resource(PytorchYolov5Service, '/yolov5service/')
ai_service_api.add_resource(InstanceSegmentationService, '/segmentation/')
ai_service_api.add_resource(ResnetTwoCategoryTransferLearnService, '/twotransferlearn/')
ai_service_api.add_resource(ResnetTwoCategory, '/twotransferlearn/pridect/')