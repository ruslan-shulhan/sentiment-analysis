from azureml.core import Workspace
import urllib.request
from azureml.core.model import Model

def set_azure_workscpace(ws="", subid="", rg="", cr_group=False, loc=""):
    ws = Workspace.create(name=ws,
                   subscription_id=subid,
                   resource_group=rg,
                   create_resource_group=cr_group,
                   location=loc
                   )

    return ws

def register_model(work_space=None, model_path="", model_name=""):
    # Register model
    model = Model.register(workspace=work_space, model_path=model_path, model_name=model_name)

    return model