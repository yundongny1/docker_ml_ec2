# Containerizing a Python ML Model on AWS EC2
Testing a python ML microservice - containerizing it with Docker on an AWS EC2 linux instance.

### Technologies used:
* Prometheus
* Grafana
* Docker
* EC2
* Flask
* Python

### Data source
This microservice uses the legal_text_classification.csv from https://www.kaggle.com/datasets/amohankumar/legal-text-classification-dataset.

### Instructions
Assuming you are running an ubuntu instance on AWS EC2, steps for testing this application after cloning the repository:
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

These are the respective ports you want to be aware of.

* Prometheus metrics: 8000
* Prometheus UI: 9090
* Grafana UI: 3000

You can connect by your_ec2_ipv4_address:port. Make sure you adjust the ec2 security settings to be able to do this.

