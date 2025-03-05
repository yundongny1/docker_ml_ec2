# docker_ml_ec2
Testing a python ML microservice - containerizing it with Docker on an AWS EC2 linux instance.
It uses simple Flask API to communicate between user input and the trained model.

This microservice uses the legal_text_classification.csv from https://www.kaggle.com/datasets/amohankumar/legal-text-classification-dataset.


Assuming you are running an ubuntu instance, steps for testing this application after cloning the repository:
1. sudo apt update
2. sudo snap install docker
3. docker-compose build flask_app
This reads the docker-compose.yml file and builds the containers and images.
4. sudo docker run --network=host