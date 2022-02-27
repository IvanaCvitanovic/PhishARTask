# PhishARTask
Recognize the website by calculating deep features and comparing them to already calculated features for images from the dataset
Images from the dataset have annotations in static/data/websites.xlsx file
their fetures are calculated offline and saved (static/features)
VGG16 model with ImageNet pre-trained weights is used to calculate deep features
Flask web-interface is used to create simple web app
after the image is selected, features are calculated using the same model
features of the selected image are then compared to all other features calculated offline
After finding the most similar image from the dataset its annotion (website) is the result.




# calculate feature weights for images in the dataset
python offline.py


# run app
python app.py
