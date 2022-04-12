from flask_restful import Resource, reqparse, fields, marshal

from App.apis.api_constant import HTTP_OK
from App.apis.customer_user.utils import login_required

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

class TestCustomerUser(Resource):
    @login_required
    def post(self):
        args = parse_base.parse_args()

        num_a = args.get("num_a")
        num_b = args.get("num_b")

        merge_word = num_a + num_b

        data = {
            "msg": 'ok',
            "status": HTTP_OK,
            "data": merge_word
        }

        return marshal(data, single_response_fields)