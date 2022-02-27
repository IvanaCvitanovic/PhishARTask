import numpy as np
from PIL import Image
from extract_features import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import os
import numpy
from pathlib import Path

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/features").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/data") / (feature_path.stem + ".png"))
features = np.array(features)
d = FeatureExtractor.read_table()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)  # Top 30 results
        id = ids[0]
        name = Path(img_paths[id]).name.split('.')[0]
        scores = [(dists[id], img_paths[id])]
        website = d.get(name)
        print(name)
        print()
        print(d)


        return render_template('index.html',
                               query_path=uploaded_img_path,
                               website=website)
    else:
        return render_template('index.html')

    


if __name__=="__main__":
    app.run("0.0.0.0")