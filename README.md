# Building a Python ML Application on AWS EC2
Testing a python ML microservice - containerizing it with Docker on an AWS EC2 linux instance. The model is a simple legal text classification model. Using the dataset from Kaggle, it trains and predicts the case outcome of the legal case text that the user inputs. It uses Flask API to handle incoming requests and generating responses, and Prometheus and Grafana to monitor the application. Prometheus and Grafana are on separate docker containers from the flask app for best practices.

### Technologies used:
* Prometheus
* Grafana
* Docker
* EC2
* Flask
* Python
* Github Actions

### Data source
This microservice uses the legal_text_classification.csv from https://www.kaggle.com/datasets/amohankumar/legal-text-classification-dataset.

### Instructions
Assuming you are running an ubuntu instance on AWS EC2, steps for testing this application after cloning the repository into your EC2 instance:
```shell
#SSH into your ubuntu instance using your key
sudo apt update
sudo snap install docker

# This reads the docker-compose.yml file and builds the containers and images.
docker-compose up build

#If this doesn't work, run below two steps instead.
docker-compose build 
# the --network=host will ensure that the docker containers will share the host's ip address, which in our case is the ec2 public ipv4 address
sudo docker run --network=host
```
This repository uses Github Actions for CI/CD. If you go to Settings -> Secrets and variables -> Actions -> Repository secrets, you can set up your EC2 IPV4 address and the SSH key. Make sure to name your ipv4 address EC2_HOST and your key as EC2_SSH_KEY.

These are the respective ports you want to be aware of. You can connect to the end points like this -> public_ipv4_address:port

* Prometheus metrics: 8000
* Prometheus UI: 9090
* Grafana UI: 3000

If you want to test the prediction of the application, use curl in your command line. The code will look something like this:
curl -X POST PUBLIC_IPV4_DNS:5000/predict -H "Content-Type: application/json" -d '{"text": "The defendant is found not guilty."}'

This will return a prediction.
Make sure you adjust the ec2 security settings to be able to do this.

