from flask_restful import Api

from App.apis.admin_user.admin_user_api import AdminUsersResource
from App.apis.admin_user.test_admin_user import TestAdminUser, GetCustomerUsersInfo, GetCustomerUserInfo

admin_user_api = Api(prefix = '/admin')

admin_user_api.add_resource(AdminUsersResource, '/adminusers/')
admin_user_api.add_resource(TestAdminUser, '/testadminusers/')
admin_user_api.add_resource(GetCustomerUsersInfo, '/getcustomerinfos/')
admin_user_api.add_resource(GetCustomerUserInfo, '/getcustomerinfos/<int:id>/')