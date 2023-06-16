
# Self_driving_car_behavioral_cloning

This is a university project aimed at making the car in Udacity nanodegree simulator drive the left track from the simulator.
The whole specification of the assignment for this project is in the "ML D2 2023.pdf" file (3rd exercise).
To run this project you need the Udacity nanodegree simulator made in Unity. Then you need to run the model from the good_models folder. When in the root folder of the cloned project run the command 

"python drive.py -m model/model-final.h5". 

Its the same model as model-016.h5. The other models in the folder crash but were getting better and better so I was saving them in the training process.

![driving](https://github.com/Mixa26/Self_driving_car_behavioral_cloning/assets/71144280/42044155-5dcc-4e3e-b7df-b823bcdcec9d)

For this project we already had all the preprocessing steps provided to us and only had to implement the "build_model" and "train_model" in the "model.py" file. The "utils.py" has all the help functions and the "batch_generator" which is used for loading all the images and steers into code.

This is the main idea behind the training process. We drive the car around the track, record it as data where we receive images from the left, right and center of the car which we will use as features, and we also record the angle of the tires which will be our prediction target.

![training](https://github.com/Mixa26/Self_driving_car_behavioral_cloning/assets/71144280/daef5d72-4875-44d5-9fd9-e8674471ed34)

Ive played around with keras layers and came with the following architecture at the end:

![neural_network](https://github.com/Mixa26/Self_driving_car_behavioral_cloning/assets/71144280/0b214942-a617-491c-8e99-0ad82fdf215f)

I've ran the training a lot of the times and the setting that provided me with the model at the end was that
I ran 20 epochs with the batch size of 256, 100 batches per epoch and a 0.1 learning rate. The model 16 was the one that could go around the whole circuit of the left track. The setting is also provided in the "good_models/setting" folder.

This was a very rewarding project for me as it was the first time I played around with a neural network. It was very fun (bit frustrating too) to train and run the model seeing it progress, go further and further without running of the track and also it was the first time for me to implement my gained knowledge of neural networks from university lectures into code.
