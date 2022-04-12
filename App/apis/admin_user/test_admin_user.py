from flask_restful import Resource, reqparse, fields, marshal

from App.apis.admin_user.utils import login_required
from App.apis.api_constant import HTTP_OK
from App.models.customer_user.customer_user_model import CustomerUser

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

class TestAdminUser(Resource):
    @login_required
    def post(self):
        logging.info('in TestAdminUser')
        args = parse_base.parse_args()

        num_a = args.get("num_a")
        num_b = args.get("num_b")

        merge_word = num_a + num_b

        logging.info(f'in TestAdminUser: num_a = {num_a}, num_b = {num_b}, merge_word = {merge_word}')

        data = {
            "msg":'ok',
            "status":HTTP_OK,
            "data":merge_word
        }

        return marshal(data, single_response_fields)

customer_user_fields = {
    "id": fields.Integer,
    "username": fields.String,
    "phone": fields.String,
    "password": fields.String(attribute="_password"),
    "is_verify": fields.Boolean
}

single_customer_user_fields = {
    "status": fields.Integer,
    "msg": fields.String,
    "data": fields.Nested(customer_user_fields)
}

multi_customer_user_fields = {
    "status": fields.Integer,
    "msg": fields.String,
    "data": fields.List(fields.Nested(customer_user_fields))
}

class GetCustomerUsersInfo(Resource):
    @login_required
    def get(self):
        customer_user = CustomerUser().query.all()

        data = {
            'msg':'ok',
            'status':HTTP_OK,
            'data':customer_user
        }

        return marshal(data, multi_customer_user_fields)

class GetCustomerUserInfo(Resource):
    @login_required
    def get(self, id):
        customer_user = CustomerUser().query.get(id)

        data = {
            'msg':'ok',
            'status':HTTP_OK,
            'data':customer_user
        }

        return marshal(data, single_customer_user_fields)