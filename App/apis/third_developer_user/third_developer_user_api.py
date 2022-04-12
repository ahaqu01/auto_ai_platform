from flask_restful import Resource, reqparse, abort, fields, marshal

from App.apis.api_constant import HTTP_CREATE_OK, USER_ACTION_REGISTER, USER_ACTION_LOGIN, HTTP_OK
from App.apis.third_developer_user.model_utils import get_third_developer_user

from App.ext import cache
from App.models.third_developer_user.third_developer_user_model import ThirdDeveloperUser
from App.utils import generate_third_developer_user_token

from App.common import logging_settings
import logging

parse_base = reqparse.RequestParser()
parse_base.add_argument("password", type=str, required=True, help="请输入密码")
parse_base.add_argument("action", type=str, required=True, help="请确认请求参数")

parse_register = parse_base.copy()
parse_register.add_argument("username", type=str, required=True, help="请输入用户名")
parse_register.add_argument("phone", type=str, required=True, help="请输入手机号码")

parse_login = parse_base.copy()
parse_login.add_argument("username", type=str, help="请输入用户名")
parse_login.add_argument("phone", type=str, help="请输入手机号码")


developer_user_fields = {
    "username": fields.String,
    "phone": fields.String,
    "password": fields.String(attribute="_password"),
    "is_verify": fields.Boolean
}


single_developer_user_fields = {
    "status": fields.Integer,
    "msg": fields.String,
    "data": fields.Nested(developer_user_fields)
}


class ThirdDeveloperUsersResource(Resource):

    def post(self):

        args = parse_base.parse_args()

        password = args.get("password")
        action = args.get("action").lower()

        if action == USER_ACTION_REGISTER:
            args_register = parse_register.parse_args()
            phone = args_register.get("phone")
            username = args_register.get("username")

            developer_user = ThirdDeveloperUser()

            developer_user.username = username
            developer_user.password = password
            developer_user.phone = phone

            if not developer_user.save():
                abort(400, msg="create fail")

            data = {
                "status": HTTP_CREATE_OK,
                "msg": "用户创建成功",
                "data": developer_user
            }

            return marshal(data, single_developer_user_fields)
        elif action == USER_ACTION_LOGIN:

            args_login = parse_login.parse_args()

            username = args_login.get("username")
            phone = args_login.get("phone")

            user = get_third_developer_user(username) or get_third_developer_user(phone)

            if not user:
                abort(400, msg="用户不存在")

            if not user.check_password(password):
                abort(401, msg="密码错误")

            if user.is_delete:
                abort(401, msg="用户不存在")

            token = generate_third_developer_user_token()
            logging.debug(f'thrid_developer_user token={token}')

            cache.set(token, user.id, timeout=60*60*24*7)

            data = {
                "msg": "login success",
                "status": HTTP_OK,
                "token": token
            }

            return data

        else:
            abort(400, msg="其提供正确的参数")

