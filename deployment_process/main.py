# from pre_processing import PreProcessing
# from dp_sentimen_analysis_model import TweetsSentimentAnalysis
# import azure_workspace
# import source_dir.echo_score as echo_score
import os

# registering a model from a local file/folder
from azureml.core.model import Model
# import urllib.request # for downloading model from www source

# connecting to the workspace
from azureml.core import Workspace

# defining an inference configuration
from azureml.core import Environment
from azureml.core.model import InferenceConfig

# defining deployment configuration
from azureml.core.webservice import LocalWebservice

# calling into a model
import json
import requests



# gets unstructured text and converts it into clean one
# def user_input(some_text=""):
#     text_obj = PreProcessing(isTimer=True, isRemoveDigits=True)
#     text_result = text_obj.apply_all_cleaning_steps(text=some_text)
#
#     return text_result['text']


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
    loaded_model = Model.register(workspace=work_space, model_path=model_path, model_name=model_name)

    return loaded_model


if __name__ == "__main__":
    # input_text = user_input(input("enter your text: "))
    # print("User input: ", input_text)
    #
    # obj_model = TweetsSentimentAnalysis()
    #
    # if obj_model.load_model(path_to_model="model_4"):
    #     print("model is loaded successfully. \nscanning provided data. please wait...")
    # else:
    #     raise TypeError("Error: cannot load the model. please contact administrator.")
    #
    # result_prediction = obj_model.get_prediction(data_set=input_text)
    # print(result_prediction)

    # azure cloud service set up
    workspace_name = 'workspace_name'
    resource_group = 'resource_group'
    subsc_id = os.environ['subsc_id']
    cr_group = True
    location = 'location'
    model_path = 'model_path'
    model_name = 'model_name'

    # connecting to your workspace
    ws = set_azure_workscpace(ws=workspace_name,
                              rg=resource_group,
                              subid=subsc_id,
                              cr_group=cr_group,
                              loc=location)

    # registering model from a local file/folder
    model = register_model(work_space=ws,
                           model_path=model_path,
                           model_name=model_name)

    # defining an inference configuration
    env = Environment(name="project_environment")
    dummy_inference_config = InferenceConfig(
        environment=env,
        source_directory="./source_dir",
        entry_script="./echo_score.py",
    )

    # defining deployment configuration
    deployment_config = LocalWebservice.deploy_configuration(port=6789)

    # deploying machine learning model
    service = Model.deploy(
        ws,
        "myservice",
        [model],
        dummy_inference_config,
        deployment_config,
        overwrite=True,
    )
    service.wait_for_deployment(show_output=True)
    print(service.get_logs())

    # calling into a model
    uri = service.scoring_uri
    requests.get("http://localhost:6789")
    headers = {"Content-Type": "application/json"}
    data = {
        "query": "What color is the fox",
        "context": "The quick brown fox jumped over the lazy dog.",
    }
    data = json.dumps(data)
    response = requests.post(uri, data=data, headers=headers)
    print(response.json())










