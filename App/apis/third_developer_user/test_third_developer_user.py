from flask_restful import Resource, reqparse, fields, marshal

from App.apis.api_constant import HTTP_OK
from App.apis.third_developer_user.utils import login_required

from App.common import logging_settings
import logging

parse_base = reqparse.RequestParser()
parse_base.add_argument("num_a", type=str, required=True, help="请输入参数a")
parse_base.add_argument("num_b", type=str, required=True, help="请输入参数b")


test_param_fields = {
    'num_a':fields.String,
    'num_b':fields.String
}

single_response_fields = {
    "status": fields.Integer,
    "msg": fields.String,
    "data": fields.String
}

class TestThridDeveloperUser(Resource):
    @login_required
    def post(self):
        logging.info('in TestThridDeveloperUser')
        args = parse_base.parse_args()
        logging.info(f'args = {args}')

        num_a = args.get("num_a")
        num_b = args.get("num_b")
        logging.debug(f'num_a = {num_a}, num_b = {num_b}')
        merge_word = num_a + num_b

        data = {
            "msg": 'ok',
            "status": HTTP_OK,
            "data": merge_word
        }

        return marshal(data, single_response_fields)