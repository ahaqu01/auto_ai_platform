from flask_restful import Api

from App.apis.customer_user.customer_user_api import CustomerUsersResource
from App.apis.customer_user.test_customer_user import TestCustomerUser

customer_user_api = Api(prefix="/customer")

customer_user_api.add_resource(CustomerUsersResource, '/customerusers/')
customer_user_api.add_resource(TestCustomerUser, '/testcustomerusers/')