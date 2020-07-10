# Deploy ONNX Models to Azure Functions

Samples for serverless deployment of ONNX models to Azure Functions. 

## Pre-requisites

Before you begin, you must have the following:

1. An Azure account with an active subscription. [Create an account for free](https://azure.microsoft.com/free).

On a Linux system or Windows (WSL or WSL2) ensure you have the following installed:

2. The [Azure Functions Core Tools](https://docs.microsoft.com/en-us/azure/azure-functions/functions-run-local#v2)
3. The [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) 
4. Python 3.7

Note: You can also use Azure Cloud Shell which comes preinstalled with Azure CLI and Functions Core Tools. 

## Develop and Test Locally

You can develop for serverless deployment on Azure functions on your Linux machine locally or a development VM (like the Azure Data Science VM) on the cloud or Azure Cloud Shell. Run the following commands to setup your Azure Functions project locally.

1. Create a Function App project directory and start directory

```
mkdir << Your projectname >>
cd << Your projectname>>
mkdir start
cd start
```

2. Initialize Function App

```
func init --worker-runtime python
func new --name classify --template "HTTP trigger"
```

3. Copy the deployment code 
```
git clone https://github.com/gopitk/functions-deploy-onnx.git ~/functions-deploy-onnx

# Copy the deployment sample to function app
cp -r ~/functions-deploy-onnx/start ..

```
The main files are **[__init__.py](https://github.com/gopitk/functions-deploy-onnx/blob/master/start/classify/__init__.py)** and **[predictonnx.py](https://github.com/gopitk/functions-deploy-onnx/blob/master/start/classify/predictonnx.py)** in ```start/classify``` dirtectory. The one in the repo works for the Bear detector example in fast.ai. It takes input from the HTTP GET request in "img" parameter which is a URL to an image which will be run through the model for prediction of the type of bear.  You can adapt the same easily for deploying other models.

4. Create and activate Python virtualenv to setup ONNX runtime along with dependencies

```
python -m venv .venv
source .venv/bin/activate

pip install --no-cache-dir -r requirements.txt  
```

5. Copy your ONNX model file (which should have a name model.onnx)  built from training the Pytorch model  and converting to ONNX into  the "start/classify" directory within your Function App project. This has been tested with the Bear detector model from fast.ai course. Here is code to generate the model.onnx file for this fast.ai (or Pytorch) model.

```
dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
onnx_path =  "./model.onnx"
torch.onnx.export(learn.model, dummy_input, onnx_path, verbose=False)
```


You also need to create a ```labels.json``` in ```start/classify``` directory. For the fast.ai 3 class Bear detector example it looks like this ```["black","grizzly","teddy"]``` matching the class label and index during training. 

6. Run the test locally

```
func start
```
In a browser on your machine you can test the local Azure Function by visiting: 

"http://localhost:7071/api/classify?img=http://3.bp.blogspot.com/-S1scRCkI3vY/UHzV2kucsPI/AAAAAAAAA-k/YQ5UzHEm9Ss/s1600/Grizzly%2BBear%2BWildlife188.jpg"


## Create Resources in Azure and Publish

1. Create Azure Function App using Azure CLI

```
# If you have not logged into Azure CLI. you must first run "az login" and follow instructions

az group create --name [[YOUR Function App name]]  --location westus2
az storage account create --name [[Your Storage Account Name]] -l westus2 --sku Standard_LRS -g [[YOUR Function App name]]
az functionapp create --name [[YOUR Function App name]] -g [[YOUR Function App name]] --consumption-plan-location westus2 --storage-account [[Your Storage Account Name]] --runtime python --runtime-version 3.7 --functions-version 3 --disable-app-insights --os-type Linux
```
2. Publish to Azure

```
# Install a local copy of ONNX runtime and dependencies to push to Azure Functions Runtime
pip install  --target="./.python_packages/lib/site-packages"  -r requirements.txt

# Publish Azure function to the 
func azure functionapp publish [[YOUR Function App name] --no-build


```

It will take a few minutes to publish and bring up the Azure functions with your ONNX model deployed and exposed as a http endpoint.  Then you can find the URL by running the following command:  ```func azure functionapp list-functions [[YOUR Function App name] --show-keys``` . Append ```&img=[[Your Image URL to run thru model]]``` to the URL on a browser to get predictions from the model running in the Azure Functions. 

## Deleting Resources
To delete all the resources (and avoiding any charges) after you are done, run the following Azure CLI command:
```
az group delete --name [[YOUR Function App name]] --yes

```
Azure Functions provides other options like auto scaling, larger instances, monitoring with Application Insights etc. 

## Notes

1. If you are using Windows WSL or WSL2 and face authentication issues while deploying Function App to Azure, one of the main reason is your clock in WSL (Linux) may be out of sync with the underlying Windows host. You can synch it by running ```sudo hwclock -s``` in WSL. 

